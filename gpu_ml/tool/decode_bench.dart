/// Decode-speed benchmark for the qwen35 runner.
///
///   dart run tool/decode_bench.dart [--model PATH] [--tokens N]
///     [--cache-gb G] [--stack-gb G] [--prompt TEXT]
///
/// Prints per-token totals split into sync-wait (GPU readbacks) and expert
/// fetch (disk+upload on LRU miss), plus a warm tokens/sec summary
/// (excluding prefill and the first generated token).
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml_io.dart';

const defaultModel =
    r'C:\models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf';

String _arg(List<String> args, String name, String def) {
  final i = args.indexOf(name);
  return (i >= 0 && i + 1 < args.length) ? args[i + 1] : def;
}

Future<void> main(List<String> args) async {
  final model = _arg(args, '--model', defaultModel);
  final tokens = int.parse(_arg(args, '--tokens', '16'));
  final cacheGb = double.parse(_arg(args, '--cache-gb', '8'));
  final stackGb = double.parse(_arg(args, '--stack-gb', '0'));
  final gpu2 = _arg(args, '--gpu2', '');
  final gpu2Blocks = int.parse(_arg(args, '--gpu2-blocks', '20'));
  final cache2Gb = double.parse(_arg(args, '--cache2-gb', '4'));
  final stack2Gb = double.parse(_arg(args, '--stack2-gb', '0'));
  final prompt = _arg(args, '--prompt', 'The capital of France is');
  final useReference = args.contains('--reference');

  stdout.writeln('model:  $model');
  stdout.writeln('cache:  ${cacheGb.toStringAsFixed(1)} GB LRU, '
      '${stackGb.toStringAsFixed(1)} GB resident stacks');
  if (gpu2.isNotEmpty) {
    stdout.writeln("gpu2:   '$gpu2' ($gpu2Blocks blocks, "
        '${cache2Gb.toStringAsFixed(1)} GB LRU, '
        '${stack2Gb.toStringAsFixed(1)} GB stacks)');
  }

  final auto = args.contains('--auto');
  final sw = Stopwatch()..start();
  void progress(String msg) {
    if (msg == 'ready' ||
        msg.contains('stacks') ||
        msg.contains('adapter') ||
        msg.startsWith('auto:')) {
      stdout.writeln('[load ${sw.elapsed}] $msg '
          '(rss ${ProcessInfo.currentRss >> 20} MB)');
    }
  }

  final m = auto
      ? await Qwen35Model.loadAuto(model, onProgress: progress)
      : await Qwen35Model.load(
          model,
          expertCacheBytes: (cacheGb * 1024 * 1024 * 1024).round(),
          expertStackBytes: (stackGb * 1024 * 1024 * 1024).round(),
          secondaryAdapter: gpu2.isEmpty ? null : gpu2,
          secondaryBlocks: gpu2Blocks,
          secondaryCacheBytes: (cache2Gb * 1024 * 1024 * 1024).round(),
          secondaryStackBytes: (stack2Gb * 1024 * 1024 * 1024).round(),
          onProgress: progress,
        );
  stdout.writeln('[load done ${sw.elapsed}] resident MoE blocks: '
      '${m.residentMoeBlocks}/40'
      '${m.secondaryAdapterName != null ? " (gpu2: ${m.secondaryAdapterName})" : ""}'
      ' (rss ${ProcessInfo.currentRss >> 20} MB)');

  // --prompt-tokens N: pad the prompt to N tokens (repeats filler text) to
  // measure prefill throughput at realistic prompt lengths.
  final promptTokens = int.parse(_arg(args, '--prompt-tokens', '0'));
  var ids = m.tokenizer.encode(prompt);
  if (promptTokens > ids.length) {
    const filler = ' The quick brown fox jumps over the lazy dog and'
        ' continues running through the quiet forest before resting.';
    final sb = StringBuffer();
    while (m.tokenizer.encode(sb.toString()).length < promptTokens) {
      sb.write(filler);
    }
    final pad = m.tokenizer.encode(sb.toString());
    ids = [...pad.sublist(0, promptTokens - ids.length), ...ids];
  }
  stdout.writeln('prompt: ${ids.length} tokens');

  int lastSync = 0, lastFetch = 0, lastMiss = 0;
  final warmMs = <double>[];

  Future<Float32List> step(int id, int pos, String label) async {
    final t = Stopwatch()..start();
    final logits = useReference
        ? await m.forwardReference(id, pos)
        : await m.forward(id, pos);
    final ms = t.elapsedMicroseconds / 1000.0;
    final syncMs = (m.syncMicros - lastSync) / 1000.0;
    final fetchMs = (m.fetchMicros - lastFetch) / 1000.0;
    final miss = m.cacheMisses - lastMiss;
    lastSync = m.syncMicros;
    lastFetch = m.fetchMicros;
    lastMiss = m.cacheMisses;
    stdout.writeln('$label ${ms.toStringAsFixed(1)} ms '
        '(sync ${syncMs.toStringAsFixed(1)}, fetch ${fetchMs.toStringAsFixed(1)}, '
        'miss $miss)');
    return logits;
  }

  int pos = 0;
  final seqPrefill = args.contains('--seq-prefill');

  // --ab-prefill N: run prefill 2N times alternating prefillDp4a off/on,
  // zeroing the recurrent state between runs (KV is overwritten in place),
  // all in ONE process so background GPU load hits both paths equally.
  // The first run per path is discarded (pipeline compiles).
  final abPrefill = int.parse(_arg(args, '--ab-prefill', '0'));
  if (abPrefill > 0) {
    final times = <bool, List<double>>{false: [], true: []};
    for (int r = 0; r < 2 * abPrefill + 2; r++) {
      final on = r.isOdd;
      m.prefillDp4a = on;
      await m.resetRecurrentState();
      final sw = Stopwatch()..start();
      await m.prefill(ids);
      final ms = sw.elapsedMicroseconds / 1000.0;
      if (r >= 2) times[on]!.add(ms);
      stdout.writeln('prefill[$r]${on ? ' q' : ' f'} ${ms.toStringAsFixed(1)} ms');
    }
    for (final on in [false, true]) {
      final xs = times[on]!;
      if (xs.isEmpty) continue;
      final avg = xs.reduce((a, b) => a + b) / xs.length;
      stdout.writeln('abp ${on ? 'dp4a' : 'f32 '}: ${avg.toStringAsFixed(1)} ms '
          '= ${(ids.length * 1000.0 / avg).toStringAsFixed(1)} tok/s '
          '(${xs.length} runs)');
    }
    // Leave a clean default-path state for the decode that follows, and
    // re-baseline the perf counters so the normal prefill's printed
    // sync/fetch deltas exclude the A/B runs (review finding).
    m.prefillDp4a = false;
    await m.resetRecurrentState();
    lastSync = m.syncMicros;
    lastFetch = m.fetchMicros;
    lastMiss = m.cacheMisses;
  }

  final pf = Stopwatch()..start();
  Float32List logits;
  if (seqPrefill) {
    Float32List? l;
    for (int j = 0; j < ids.length; j++) {
      l = await m.forward(ids[j], j);
    }
    logits = l!;
  } else {
    logits = await m.prefill(ids);
  }
  pf.stop();
  pos = ids.length;
  {
    final syncMs = (m.syncMicros - lastSync) / 1000.0;
    final fetchMs = (m.fetchMicros - lastFetch) / 1000.0;
    lastSync = m.syncMicros;
    lastFetch = m.fetchMicros;
    lastMiss = m.cacheMisses;
    final ms = pf.elapsedMicroseconds / 1000.0;
    stdout.writeln('prefill ${ids.length} tokens in ${ms.toStringAsFixed(1)} '
        'ms = ${(ids.length * 1000.0 / ms).toStringAsFixed(1)} tok/s '
        '(sync ${syncMs.toStringAsFixed(1)}, fetch ${fetchMs.toStringAsFixed(1)}, '
        'record ${(m.recordMicros / 1000.0).toStringAsFixed(1)})');
  }

  final dumpPath = _arg(args, '--dump-logits', '');
  if (dumpPath.isNotEmpty) {
    File(dumpPath).writeAsBytesSync(logits.buffer.asUint8List(
        logits.offsetInBytes, logits.lengthInBytes));
    stdout.writeln('dumped ${logits.length} prefill logits to $dumpPath');
  }

  // --ab-dp4a N: cycle the dp4a level 0 (f32) / 1 (lm_head only) / 2 (all
  // projections) every N tokens within THIS process, so background GPU
  // load hits every path equally (separate-process A/B on this machine
  // swings 2.5-45 tok/s).  No pre-warm forwards — they would advance the
  // DeltaNet recurrent state with duplicate tokens and derail the text;
  // instead each segment's first token (which pays any pipeline compiles
  // and rebinds) is excluded from its average.
  // --ab-fuse N / --ab-embed N: same in-process alternation, cycling the
  // DeltaNet decode fusion or the decode embed gather instead of dp4a
  // levels.
  // One entry per switchable optimization; the first flag present on the
  // command line wins.  Levels cycle every N tokens.
  final abSpecs = <(String, List<String>, void Function(Qwen35Model, int))>[
    ('--ab-fuse', ['unfused  ', 'fused    '],
        (m, l) => m.deltaFusion = l == 1),
    ('--ab-embed', ['disk-emb ', 'gpu-emb  '],
        (m, l) => m.embedGatherDecode = l == 1),
    ('--ab-xs', ['uav-x    ', 'shared-x '], (m, l) => m.sharedX = l == 1),
    ('--ab-rp', ['xs-t64   ', 'xs-rpack '], (m, l) => m.xsRowPack = l == 1),
    ('--ab-f2', ['exp-5disp', 'exp-3disp'], (m, l) => m.expFuse2 = l == 1),
    ('--ab-cf', ['unchained', 'chained  '], (m, l) => m.chainFuse = l == 1),
    ('--ab-rf', ['resid-add', 'resid-fus'], (m, l) => m.residFuse = l == 1),
    ('--ab-rs', ['rec-1wg  ', 'rec-split'], (m, l) => m.recSplit = l == 1),
    ('--ab-dp4a', ['f32      ', 'dp4a-head', 'dp4a-all '],
        (m, l) => m.dp4aLevel = l),
    // Category map: one ablation per level, all inside ONE process, so the
    // per-category costs are free of the cross-process variance that made
    // the old one-process-per-ablation maps unreliable.  Needs --ignore-eos
    // (ablated arms emit garbage).  Cost of a category = its level minus
    // the control level.
    (
      '--ab-map',
      [
        'control  ',
        'no-delta ',
        'no-attn  ',
        'no-routed',
        'no-shexp ',
        'no-rec   ',
        'no-proj  ',
        'no-conv  ',
        'no-bap   ',
        'no-out   ',
        'no-norms ',
        'no-route ',
      ],
      (m, l) => m.setAblation(const [
            null,
            'delta',
            'attn',
            'routed',
            'shexp',
            'rec',
            'proj',
            'conv',
            'bap',
            'out',
            'norms',
            'route',
          ][l]),
    ),
  ];
  var abN = 0;
  var abNames = const <String>[];
  void Function(Qwen35Model, int)? abSet;
  for (final (flag, names, set) in abSpecs) {
    final n = int.parse(_arg(args, flag, '0'));
    if (n > 0) {
      abN = n;
      abNames = names;
      abSet = set;
      break;
    }
  }
  // N=1 would exclude every token (each segment's first is discarded).
  if (abN == 1) abN = 2;
  final abLevels = abNames.length;
  final segMs = {
    for (var i = 0; i < (abLevels == 0 ? 1 : abLevels); i++) i: <double>[]
  };

  // GPU-greedy decode (GPU argmax + 4-byte id readback, token chaining) is
  // the default; --logits forces the full-logits readback + CPU argmax.
  final useGreedy = !useReference && !args.contains('--logits');

  Future<int> stepGreedy(int id, int p, String label) async {
    final t = Stopwatch()..start();
    final next = await m.forwardGreedy(id, p);
    final ms = t.elapsedMicroseconds / 1000.0;
    final syncMs = (m.syncMicros - lastSync) / 1000.0;
    final fetchMs = (m.fetchMicros - lastFetch) / 1000.0;
    final miss = m.cacheMisses - lastMiss;
    lastSync = m.syncMicros;
    lastFetch = m.fetchMicros;
    lastMiss = m.cacheMisses;
    stdout.writeln('$label ${ms.toStringAsFixed(1)} ms '
        '(sync ${syncMs.toStringAsFixed(1)}, fetch ${fetchMs.toStringAsFixed(1)}, '
        'miss $miss)');
    return next;
  }

  int argmaxCpu(Float32List l) {
    var best = 0;
    var bestV = l[0];
    for (int j = 1; j < l.length; j++) {
      if (l[j] > bestV) {
        bestV = l[j];
        best = j;
      }
    }
    return best;
  }

  final out = StringBuffer();
  int best = argmaxCpu(logits);

  // --burst K: K-token GPU-resident greedy bursts (one readback per K
  // tokens).  First token runs solo to seed the argmax chain.
  // Combining --burst with an --ab-* flag switches the arm every BURST
  // instead of every token: per-token timings carry the readback's jitter
  // (+-1-2 ms), which swamped the category map, while burst ms/token is
  // stable to ~0.3 ms.
  final burstK = useGreedy ? int.parse(_arg(args, '--burst', '0')) : 0;
  // --ignore-eos: keep generating through EOS (ablation timing runs emit
  // garbage tokens that can hit EOS immediately).
  final ignoreEos = args.contains('--ignore-eos');
  final eosId = ignoreEos ? -1 : m.tokenizer.eosId;
  if (burstK > 0) {
    var made = 0;
    if (best != eosId) {
      out.write(m.tokenizer.decode([best]));
      best = await stepGreedy(best, pos++, 'decode[seed]');
      made++;
    }
    var burstIdx = 0;
    while (made < tokens && best != eosId) {
      out.write(m.tokenizer.decode([best]));
      final n = (tokens - made).clamp(1, burstK);
      // One arm per burst; each arm's FIRST burst is discarded (it pays the
      // path switch and any pipeline compile).
      final level = abN > 0 ? (burstIdx ~/ abN) % abLevels : -1;
      if (abN > 0) abSet!(m, level);
      final t = Stopwatch()..start();
      final ids =
          await m.forwardBurstGreedy(best, pos, n, stopAtEos: !ignoreEos);
      final ms = t.elapsedMicroseconds / 1000.0;
      stdout.writeln('burst[${ids.length}]${abN > 0 ? ' L$level' : ''} '
          '${ms.toStringAsFixed(1)} ms = '
          '${(ms / ids.length).toStringAsFixed(1)} ms/token');
      if (abN > 0 && burstIdx % abN != 0) {
        segMs[level]!.add(ms / ids.length);
      }
      burstIdx++;
      if (made > 1) warmMs.addAll(List.filled(ids.length, ms / ids.length));
      for (final id in ids.sublist(0, ids.length - 1)) {
        out.write(m.tokenizer.decode([id]));
      }
      pos += ids.length;
      made += ids.length;
      best = ids.last;
    }
  }

  for (int i = 0; burstK == 0 && i < tokens; i++) {
    if (best == m.tokenizer.eosId) break;
    out.write(m.tokenizer.decode([best]));
    final level = abN > 0 ? (i ~/ abN) % abLevels : -1;
    if (abN > 0) abSet!(m, level);
    final label = 'decode[$i]${abN > 0 ? ' L$level' : ''}';
    final t = Stopwatch()..start();
    if (useGreedy) {
      best = await stepGreedy(best, pos, label);
    } else {
      logits = await step(best, pos, label);
      best = argmaxCpu(logits);
    }
    final ms = t.elapsedMicroseconds / 1000.0;
    if (i >= 1) warmMs.add(ms);
    // Skip each segment's first token (compiles + path-switch rebinds).
    if (abN > 0 && i % abN != 0) segMs[level]!.add(ms);
    pos++;
  }
  if (abN > 0) {
    for (int level = 0; level < abLevels; level++) {
      final xs = segMs[level]!;
      if (xs.isEmpty) continue;
      final avg = xs.reduce((a, b) => a + b) / xs.length;
      stdout.writeln('ab ${abNames[level]}: '
          '${avg.toStringAsFixed(1)} ms/token = '
          '${(1000.0 / avg).toStringAsFixed(2)} tok/s (${xs.length} tokens)');
    }
  }

  stdout.writeln('text: "${out.toString()}"');
  stdout.writeln('rss:  ${ProcessInfo.currentRss >> 20} MB');
  if (warmMs.isNotEmpty) {
    final avg = warmMs.reduce((a, b) => a + b) / warmMs.length;
    stdout.writeln('warm: ${avg.toStringAsFixed(1)} ms/token = '
        '${(1000.0 / avg).toStringAsFixed(2)} tok/s '
        '(${warmMs.length} tokens)');
  }
  stdout.writeln(
      'cache: ${m.cacheHits} hits / ${m.cacheMisses} misses total');
  await m.destroy();
}
