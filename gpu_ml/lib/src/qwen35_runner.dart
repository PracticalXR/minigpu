/// Qwen3.5/3.6 MoE model runner (dart:io): loads a qwen35moe GGUF and
/// generates tokens.  Residency policy v1:
/// - VRAM-resident: all norms, attention/DeltaNet weights, MoE routers,
///   shared experts, lm_head (~3.3 GB for the 35B-A3B).
/// - Streamed per token: routed experts (range reads from disk -> VRAM,
///   with a byte-budgeted LRU so hot experts stay resident).
/// - Embedding rows are range-read + CPU-dequantized per token (2 KB each).
///
/// v1 is correctness-first: one dispatch-per-op, CPU-assembled KV cache,
/// sequential prefill.  Perf work (command batching, GPU cache append,
/// prefill batching) is GPU_ML_PLAN.md M5.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';

import '../gpu_ml.dart';
import 'gguf_stream.dart';
import 'qwen35_plan.dart';

/// Byte-budgeted LRU of VRAM-resident expert matrices.
///
/// Evictions are DEFERRED: the decode plan fires dispatches without awaiting
/// them, so an evicted expert's buffer may still be referenced by enqueued
/// GPU work.  Evicted tensors go to a pending list that the runner flushes
/// after each token's logits readback (which synchronizes the whole FIFO).
class _ExpertCache {
  _ExpertCache(this.budgetBytes);
  final int budgetBytes;
  final _entries = <String, QuantizedTensor>{};
  final _sizes = <String, int>{};
  final _pendingDestroy = <QuantizedTensor>[];
  int _used = 0;
  int hits = 0;
  int misses = 0;

  /// Cumulative microseconds spent loading missed experts (disk + upload).
  /// Concurrent loads overlap, so this can exceed wall time.
  int fetchMicros = 0;

  Future<QuantizedTensor> fetch(
    String key,
    Future<QuantizedTensor> Function() load,
    int sizeBytes,
  ) async {
    final existing = _entries.remove(key);
    if (existing != null) {
      _entries[key] = existing; // re-insert as most recent
      hits++;
      return existing;
    }
    misses++;
    while (_used + sizeBytes > budgetBytes && _entries.isNotEmpty) {
      final oldest = _entries.keys.first;
      _pendingDestroy.add(_entries.remove(oldest)!);
      _used -= _sizes.remove(oldest)!;
    }
    final sw = Stopwatch()..start();
    final t = await load();
    fetchMicros += sw.elapsedMicroseconds;
    _entries[key] = t;
    _sizes[key] = sizeBytes;
    _used += sizeBytes;
    return t;
  }

  /// Destroys evicted tensors.  Only call after a full GPU sync (a readback).
  void flushPending() {
    for (final t in _pendingDestroy) {
      t.destroy();
    }
    _pendingDestroy.clear();
  }

  void destroy() {
    flushPending();
    for (final t in _entries.values) {
      t.destroy();
    }
    _entries.clear();
    _sizes.clear();
    _used = 0;
  }
}

/// MoE FFN with disk-streamed experts (resident router + shared expert).
class _StreamingMoe {
  _StreamingMoe({
    required this.stream,
    required this.router,
    required this.gateInfo,
    required this.upInfo,
    required this.downInfo,
    required this.topK,
    required this.cache,
    required this.blk,
    this.gateShexp,
    this.upShexp,
    this.downShexp,
    this.sharedGate,
  });

  final GgufStream stream;
  final Tensor router; // [experts, dim] f32
  final GgufTensorInfo gateInfo, upInfo, downInfo;
  final int topK;
  final _ExpertCache cache;
  final int blk;
  final QuantizedTensor? gateShexp, upShexp, downShexp;
  final Tensor? sharedGate;

  int get experts => router.shape[0];
  int get dim => router.shape[1];

  Future<QuantizedTensor> _expert(GgufTensorInfo info, String kind, int e) {
    final rows = info.shape[1], cols = info.shape[2];
    final traits = ggmlTypeTraits[info.type]!;
    final bytesPer = rows * cols ~/ traits.blockSize * traits.typeSize;
    return cache.fetch('$blk/$kind/$e', () async {
      final packed = await stream.readTensorBytes(info,
          byteOffset: e * bytesPer, byteLength: bytesPer);
      return QuantizedTensor.create([rows, cols], info.type, packed,
          gpu: router.gpu);
    }, bytesPer);
  }

  /// Plan-path expert fetch: kind is 'g' (gate), 'u' (up), or 'd' (down).
  Future<QuantizedTensor> expertFor(String kind, int e) => switch (kind) {
        'g' => _expert(gateInfo, 'g', e),
        'u' => _expert(upInfo, 'u', e),
        'd' => _expert(downInfo, 'd', e),
        _ => throw Exception("unknown expert kind '$kind'"),
      };

  Future<Tensor> forward(Tensor xn) async {
    // Route.
    final logits = await router.matMul(xn.reshape([dim, 1]));
    final probsT = await logits.reshape([1, experts]).softmax();
    final probs = await probsT.getData() as Float32List;
    logits.destroy();
    probsT.destroy();
    final order = List<int>.generate(experts, (i) => i)
      ..sort((a, b) => probs[b].compareTo(probs[a]));
    final sel = order.take(topK).toList();
    final wSum = sel.fold(0.0, (a, i) => a + probs[i]);

    Tensor? acc;
    for (final e in sel) {
      final g = await (await _expert(gateInfo, 'g', e)).matVec(xn);
      final gAct = await g.silu();
      g.destroy();
      final u = await (await _expert(upInfo, 'u', e)).matVec(xn);
      final prod = await gAct.multiply(u);
      gAct.destroy();
      u.destroy();
      final out = await (await _expert(downInfo, 'd', e)).matVec(prod);
      prod.destroy();
      final scaled = await out.multiplyScalar(probs[e] / wSum);
      out.destroy();
      if (acc == null) {
        acc = scaled;
      } else {
        final next = await acc.add(scaled);
        acc.destroy();
        scaled.destroy();
        acc = next;
      }
    }

    if (gateShexp != null) {
      final dotT =
          await sharedGate!.reshape([1, dim]).matMul(xn.reshape([dim, 1]));
      final dot = (await dotT.getData() as Float32List)[0];
      dotT.destroy();
      final gVal = 1.0 / (1.0 + math.exp(-dot));

      final g = await gateShexp!.matVec(xn);
      final gAct = await g.silu();
      g.destroy();
      final u = await upShexp!.matVec(xn);
      final prod = await gAct.multiply(u);
      gAct.destroy();
      u.destroy();
      final sh = await downShexp!.matVec(prod);
      prod.destroy();
      final shScaled = await sh.multiplyScalar(gVal);
      sh.destroy();
      final next = await acc!.add(shScaled);
      acc.destroy();
      shScaled.destroy();
      acc = next;
    }
    return acc!;
  }
}

class _Block {
  _Block({
    required this.attnNorm,
    required this.postNorm,
    required this.moe,
    this.attn,
    this.delta,
  });
  final Tensor attnNorm;
  final Tensor postNorm;
  final _StreamingMoe moe;
  final AttentionLayer? attn; // full-attention layers
  final DeltaNetLayer? delta; // recurrent layers
  final kCache = <Float32List>[]; // per past token [kvHeads*headDim]
  final vCache = <Float32List>[];
}

class Qwen35Model {
  Qwen35Model._({
    required this.stream,
    required this.tokenizer,
    required this.blocks,
    required this.outputNorm,
    required this.lmHead,
    required this.embedInfo,
    required this.dim,
    required this.eps,
    required this.cache,
  });

  final GgufStream stream;
  final BpeTokenizer tokenizer;
  final List<_Block> blocks;
  final Tensor outputNorm;
  final QuantizedTensor lmHead;
  final GgufTensorInfo embedInfo;
  final int dim;
  final double eps;
  final _ExpertCache cache;

  /// Fired-dispatch decode plan (M5 perf path) — built at [load].  With a
  /// secondary adapter, [_plan] runs the leading blocks on the primary GPU
  /// and [_plan2] the trailing blocks + head on the secondary; the hidden
  /// state hops devices once per token (dim*4 bytes each way).
  late final Qwen35DecodePlan _plan;
  Qwen35DecodePlan? _plan2;
  Minigpu? _gpu0; // explicit primary context (loadAuto); null = default
  Minigpu? _gpu2;
  _ExpertCache? _cache2;

  /// Full-resident expert stacks (GPU-driven blocks) — owned for destroy.
  final List<QuantizedTensor> _residentStacks = [];

  /// GPU-resident embed table for on-GPU prefill row gather (null = CPU
  /// disk-read fallback path).
  QuantizedTensor? _embedGpu;

  int get vocab => lmHead.rows;
  int get cacheHits => cache.hits + (_cache2?.hits ?? 0);
  int get cacheMisses => cache.misses + (_cache2?.misses ?? 0);

  /// Perf counters (cumulative; benchmarks read deltas): microseconds spent
  /// awaiting GPU readbacks, and loading missed experts.
  int get syncMicros => _plan.syncMicros + (_plan2?.syncMicros ?? 0);
  int get fetchMicros => cache.fetchMicros + (_cache2?.fetchMicros ?? 0);
  int get recordMicros => _plan.recordMicros + (_plan2?.recordMicros ?? 0);

  /// Number of blocks whose expert stacks are fully VRAM-resident.
  int get residentMoeBlocks => _residentStacks.length ~/ 3;

  /// Sets the dot4I8Packed decode level at runtime (both devices): 0 = off,
  /// 1 = lm_head only, 2 = all q8_0 projections.  Lets a bench alternate
  /// levels within one process so background GPU load hits all paths
  /// equally.
  set dp4aLevel(int v) {
    _plan.dp4aLevel = v;
    _plan2?.dp4aLevel = v;
  }

  /// Toggles dot4I8Packed in the PREFILL grouped expert kernels.
  set prefillDp4a(bool v) {
    _plan.prefillDp4a = v;
    _plan2?.prefillDp4a = v;
  }

  /// Toggles the DeltaNet decode fusions (beta+alpha+params, rec+gnorm).
  set deltaFusion(bool v) {
    _plan.deltaFusion = v;
    _plan2?.deltaFusion = v;
  }

  /// Toggles shared-memory x staging in decode matVecs (wave 15).
  set sharedX(bool v) {
    _plan.sharedX = v;
    _plan2?.sharedX = v;
  }

  /// Toggles the shared-x row-packing retune (wave 15).
  set xsRowPack(bool v) {
    _plan.xsRowPack = v;
    _plan2?.xsRowPack = v;
  }

  /// Toggles second-stage expert fusion (gate+up merge, silumul-in-down).
  set expFuse2(bool v) {
    _plan.expFuse2 = v;
    _plan2?.expFuse2 = v;
  }

  /// Toggles chain fusion (shexp gate+up+down merge, delta qkv+z merge).
  set chainFuse(bool v) {
    _plan.chainFuse = v;
    _plan2?.chainFuse = v;
  }

  /// Toggles residual folding (out-projection and MoE combine accumulate
  /// straight into x; no zero pass, no separate add dispatches).
  set residFuse(bool v) {
    _plan.residFuse = v;
    _plan2?.residFuse = v;
  }

  /// Toggles the DeltaNet recurrence occupancy split.
  set recSplit(bool v) {
    _plan.recSplit = v;
    _plan2?.recSplit = v;
  }

  /// Timing bisect: enables exactly ONE decode ablation (null = none), so a
  /// whole category map can be taken by cycling names inside one process.
  /// Generated text is garbage while any ablation is active.
  void setAblation(String? name) {
    for (final p in [_plan, _plan2]) {
      if (p == null) continue;
      p.skipAttn = name == 'attnblk';
      p.skipMoe = name == 'moe';
      p.skipDelta = name == 'delta';
      p.skipAttnOnly = name == 'attn';
      p.skipShexp = name == 'shexp';
      p.skipExp = name == 'routed';
      p.skipProj = name == 'proj';
      p.skipBap = name == 'bap';
      p.skipConv = name == 'conv';
      p.skipRec = name == 'rec';
      p.skipOut = name == 'out';
      p.skipNorms = name == 'norms';
      p.skipRoute = name == 'route';
    }
  }

  /// Zeroes DeltaNet recurrent state on all devices — lets a bench re-run
  /// prefill from position 0 in the same process (KV needs no reset:
  /// positions are overwritten before they are attended).
  Future<void> resetRecurrentState() async {
    await _plan.zeroRecurrentState();
    if (_plan2 != null) await _plan2!.zeroRecurrentState();
  }

  /// Name of the secondary adapter, when the model is split across two GPUs.
  String? get secondaryAdapterName => _gpu2?.adapterName;

  /// Loads with AUTOMATIC device selection: enumerates hardware adapters,
  /// computes the model's per-block VRAM needs from the GGUF header, and
  /// picks single-GPU full residency when the model fits on the largest
  /// adapter (minus [reserveBytes] headroom for display/OS), otherwise a
  /// two-GPU pipeline split proportional to free VRAM.  Budgets are sized so
  /// as many expert stacks as possible go resident.
  static Future<Qwen35Model> loadAuto(
    String path, {
    int reserveBytes = 3 * 1024 * 1024 * 1024,
    int maxSeq = 4096,
    void Function(String)? onProgress,
  }) async {
    // Header-only pass to size the model.
    final s = await GgufStream.open(path);
    final md = s.metadata;
    final arch = md['general.architecture'];
    final nLayer = md['$arch.block_count'] as int;
    final stackBytes = List<int>.filled(nLayer, 0);
    final otherBytes = List<int>.filled(nLayer, 0);
    var headBytes = 0;
    for (final t in s.tensors) {
      final name = t.name;
      if (name == 'output.weight' || name == 'output_norm.weight') {
        headBytes += t.byteSize;
        continue;
      }
      final m = RegExp(r'^blk\.(\d+)\.').firstMatch(name);
      if (m == null) continue; // token_embd stays on disk/CPU
      final i = int.parse(m.group(1)!);
      if (name.contains('_exps.')) {
        stackBytes[i] += t.byteSize;
      } else {
        otherBytes[i] += t.byteSize;
      }
    }
    await s.close();
    final blockBytes = [
      for (int i = 0; i < nLayer; i++) stackBytes[i] + otherBytes[i],
    ];
    final totalBytes = blockBytes.fold(0, (a, b) => a + b) + headBytes;

    // Per-device fixed overhead: KV caches, activation scratch, logits,
    // Dawn allocator slack.
    const scratchBytes = 1 * 1024 * 1024 * 1024;
    const cacheBytes = 512 * 1024 * 1024; // streamed-block LRU floor

    final adapters = Minigpu.listAdapters()
        .where((a) => a.totalVramBytes > 6 * 1024 * 1024 * 1024)
        .toList()
      ..sort((a, b) => b.freeVramBytes.compareTo(a.freeVramBytes));
    if (adapters.isEmpty) {
      onProgress?.call('auto: no adapter info — default single-GPU load');
      return load(path, maxSeq: maxSeq, onProgress: onProgress);
    }
    for (final a in adapters) {
      onProgress?.call('auto: $a');
    }

    final a0 = adapters.first;
    final avail0 = a0.freeVramBytes - reserveBytes - scratchBytes;
    if (totalBytes <= avail0 || adapters.length < 2) {
      onProgress?.call(
          'auto: single GPU — ${totalBytes >> 20} MB model on ${a0.name} '
          '(${avail0 >> 20} MB usable)');
      return load(
        path,
        primaryAdapter: a0.name,
        expertStackBytes: avail0 - cacheBytes,
        expertCacheBytes: cacheBytes,
        maxSeq: maxSeq,
        onProgress: onProgress,
      );
    }

    // Two-GPU split: put the LEADING blocks on the biggest adapter (as many
    // as fit), the rest + head on the second.
    final a1 = adapters[1];
    final avail1 = a1.freeVramBytes - reserveBytes - scratchBytes - headBytes;
    var split = 0;
    var acc = 0;
    while (split < nLayer && acc + blockBytes[split] <= avail0 - cacheBytes) {
      acc += blockBytes[split];
      split++;
    }
    if (split == 0 || split >= nLayer) {
      throw Exception('auto: could not find a valid split '
          '(avail0 ${avail0 >> 20} MB, avail1 ${avail1 >> 20} MB)');
    }
    onProgress?.call(
        'auto: split — blocks 0..${split - 1} on ${a0.name}, '
        '$split..${nLayer - 1} + head on ${a1.name}');
    return load(
      path,
      primaryAdapter: a0.name,
      expertStackBytes: avail0 - cacheBytes,
      expertCacheBytes: cacheBytes,
      secondaryAdapter: a1.name,
      secondaryBlocks: nLayer - split,
      secondaryStackBytes: avail1 - cacheBytes,
      secondaryCacheBytes: cacheBytes,
      maxSeq: maxSeq,
      onProgress: onProgress,
    );
  }

  static Future<Qwen35Model> load(
    String path, {
    int expertCacheBytes = 8 * 1024 * 1024 * 1024,

    /// VRAM budget for FULL-RESIDENT expert stacks.  Blocks whose stacks fit
    /// run GPU-driven MoE (no routing readback, no streaming — sync-free);
    /// the rest stream through the LRU.  Total VRAM ≈ 3.3 GB resident +
    /// expertStackBytes + expertCacheBytes (+ KV/scratch).
    int expertStackBytes = 0,

    /// Multi-GPU pipeline split: when set (adapter-name substring, e.g.
    /// '3090'), the LAST [secondaryBlocks] blocks + the lm_head run on that
    /// adapter with their own stack/LRU budgets; the hidden state hops
    /// devices once per token (dim*4 bytes each way).
    String? secondaryAdapter,
    int secondaryBlocks = 20,
    int secondaryStackBytes = 0,
    int secondaryCacheBytes = 4 * 1024 * 1024 * 1024,

    /// Pins the PRIMARY context to a specific adapter (name substring)
    /// instead of the default display-adapter auto-selection.  Used by
    /// [loadAuto] so device assignment is deterministic.
    String? primaryAdapter,
    int maxSeq = 4096,
    void Function(String)? onProgress,
  }) async {
    final s = await GgufStream.open(path);
    final md = s.metadata;
    final arch = md['general.architecture'];
    if (arch != 'qwen35moe') {
      throw Exception("Qwen35Model supports qwen35moe, got '$arch'");
    }
    final nLayer = md['$arch.block_count'] as int;
    final dim = md['$arch.embedding_length'] as int;
    final interval = (md['$arch.full_attention_interval'] as int?) ?? 4;
    final topK = (md['$arch.expert_used_count'] as int?) ?? 8;
    final eps =
        (md['$arch.attention.layer_norm_rms_epsilon'] as num?)?.toDouble() ??
            1e-6;

    onProgress?.call('parsing tokenizer');
    final tokenizer = BpeTokenizer.fromGgufMetadata(md);
    final cache = _ExpertCache(expertCacheBytes);

    // Multi-GPU: the last [secondaryBlocks] blocks live on the secondary
    // adapter with their own LRU.  gpu == null means the default context;
    // [primaryAdapter] pins the primary to an explicit adapter instead.
    Minigpu? gpu0;
    if (primaryAdapter != null) {
      gpu0 = Minigpu.forAdapter(primaryAdapter);
      await gpu0.init();
      onProgress?.call("primary adapter: '${gpu0.adapterName}'");
    }
    Minigpu? gpu2;
    _ExpertCache? cache2;
    var split = nLayer;
    if (secondaryAdapter != null) {
      gpu2 = Minigpu.forAdapter(secondaryAdapter);
      await gpu2.init();
      cache2 = _ExpertCache(secondaryCacheBytes);
      split = nLayer - secondaryBlocks;
      if (split < 1 || split >= nLayer) {
        throw Exception(
            'secondaryBlocks $secondaryBlocks out of range (1..${nLayer - 1})');
      }
      onProgress?.call("secondary adapter: '${gpu2.adapterName}' "
          '(blocks $split..${nLayer - 1} + head)');
    }

    final blocks = <_Block>[];
    for (int i = 0; i < nLayer; i++) {
      final isRecurrent = (i + 1) % interval != 0;
      final dev = i < split ? gpu0 : gpu2;
      final devCache = i < split ? cache : cache2!;
      onProgress?.call(
          'loading blk.$i (${isRecurrent ? 'deltanet' : 'attention'})');
      final moe = _StreamingMoe(
        stream: s,
        router: await s.loadF32('blk.$i.ffn_gate_inp.weight', gpu: dev),
        gateInfo: s.tensor('blk.$i.ffn_gate_exps.weight')!,
        upInfo: s.tensor('blk.$i.ffn_up_exps.weight')!,
        downInfo: s.tensor('blk.$i.ffn_down_exps.weight')!,
        topK: topK,
        cache: devCache,
        blk: i,
        gateShexp:
            await s.loadQuantized('blk.$i.ffn_gate_shexp.weight', gpu: dev),
        upShexp: await s.loadQuantized('blk.$i.ffn_up_shexp.weight', gpu: dev),
        downShexp:
            await s.loadQuantized('blk.$i.ffn_down_shexp.weight', gpu: dev),
        sharedGate:
            await s.loadF32('blk.$i.ffn_gate_inp_shexp.weight', gpu: dev),
      );
      blocks.add(_Block(
        attnNorm: await s.loadF32('blk.$i.attn_norm.weight', gpu: dev),
        postNorm:
            await s.loadF32('blk.$i.post_attention_norm.weight', gpu: dev),
        moe: moe,
        attn: isRecurrent ? null : await s.loadAttentionLayer(i, gpu: dev),
        delta: isRecurrent ? await s.loadDeltaNetLayer(i, gpu: dev) : null,
      ));
    }

    onProgress?.call('loading lm_head');
    final lastDev = split < nLayer ? gpu2 : gpu0;
    final outputNorm = await s.loadF32('output_norm.weight', gpu: lastDev);
    final lmHead = await s.loadQuantized('output.weight', gpu: lastDev);
    final model = Qwen35Model._(
      stream: s,
      tokenizer: tokenizer,
      blocks: blocks,
      outputNorm: outputNorm,
      lmHead: lmHead,
      embedInfo: s.tensor('token_embd.weight')!,
      dim: dim,
      eps: eps,
      cache: cache,
    );

    // Full-resident expert stacks for as many blocks as each device's
    // budget allows: those blocks become sync-free (GPU-driven routing).
    onProgress?.call('building decode plan');
    var stackBudget = expertStackBytes;
    var stackBudget2 = secondaryStackBytes;
    final planBlocks = <PlanBlock>[];
    for (int i = 0; i < blocks.length; i++) {
      final b = blocks[i];
      final dev = i < split ? gpu0 : gpu2;
      QuantizedTensor? sg, su, sd;
      final stackBytes = b.moe.gateInfo.byteSize +
          b.moe.upInfo.byteSize +
          b.moe.downInfo.byteSize;
      final budget = i < split ? stackBudget : stackBudget2;
      if (stackBytes <= budget) {
        onProgress?.call(
            'loading blk.$i expert stacks (${stackBytes >> 20} MB resident'
            '${dev != null ? ', secondary' : ''})');
        sg = await s.loadQuantized('blk.$i.ffn_gate_exps.weight', gpu: dev);
        su = await s.loadQuantized('blk.$i.ffn_up_exps.weight', gpu: dev);
        sd = await s.loadQuantized('blk.$i.ffn_down_exps.weight', gpu: dev);
        model._residentStacks.addAll([sg, su, sd]);
        if (i < split) {
          stackBudget -= stackBytes;
        } else {
          stackBudget2 -= stackBytes;
        }
      }
      planBlocks.add(PlanBlock(
        attnNorm: b.attnNorm,
        postNorm: b.postNorm,
        attn: b.attn,
        delta: b.delta,
        moe: PlanMoe(
          router: b.moe.router,
          gateShexp: b.moe.gateShexp,
          upShexp: b.moe.upShexp,
          downShexp: b.moe.downShexp,
          sharedGate: b.moe.sharedGate,
          fetchExpert: b.moe.expertFor,
          stackGate: sg,
          stackUp: su,
          stackDown: sd,
        ),
      ));
    }

    model._gpu0 = gpu0;
    model._gpu2 = gpu2;
    model._cache2 = cache2;
    model._plan = await Qwen35DecodePlan.build(
      gpu: planBlocks.first.attnNorm.gpu,
      blocks: planBlocks.sublist(0, split),
      outputNorm: split == nLayer ? outputNorm : null,
      lmHead: split == nLayer ? lmHead : null,
      dim: dim,
      eps: eps,
      topK: topK,
      maxSeq: maxSeq,
    );
    if (split < nLayer) {
      model._plan2 = await Qwen35DecodePlan.build(
        gpu: gpu2!,
        blocks: planBlocks.sublist(split),
        outputNorm: outputNorm,
        lmHead: lmHead,
        dim: dim,
        eps: eps,
        topK: topK,
        maxSeq: maxSeq,
      );
    }
    // Resident GPU embed table (~0.5 GB): prefill gathers embedding rows
    // on-GPU instead of one awaited ~2KB disk read per prompt token (which
    // cost 1-2s of wall time per 512-token prompt).  Rides in the VRAM
    // reserve; only worth it when the batched prefill path is active.
    if (model._plan.supportsBatchedPrefill &&
        (model.embedInfo.type == GgmlType.q8_0 ||
            model.embedInfo.type == GgmlType.f16)) {
      onProgress?.call('loading embed table to GPU');
      model._embedGpu = await s.loadQuantized('token_embd.weight', gpu: gpu0);
      // Warm the prefill pipelines with a throwaway chunk: FXC compiles
      // every pipeline on first dispatch WHILE HOLDING the GPU mutex, which
      // stalls prompt recording ~2s on the first real prefill otherwise.
      // The garbage tokens' recurrent state is zeroed afterwards.
      onProgress?.call('warming prefill pipelines');
      await model._warmupPrefill();
    }
    onProgress?.call('ready');
    return model;
  }

  Future<void> _warmupPrefill() async {
    final embedGpu = _embedGpu;
    if (embedGpu == null) return;
    const t = Qwen35DecodePlan.maxPrefillChunk;
    await _plan.prefillChunkTokens(embedGpu, List.filled(t, 0), 0);
    final plan2 = _plan2;
    if (plan2 != null) {
      final h = await _plan.readPrefillX(t);
      await plan2.prefillChunk(h, t, 0);
    }
    await (plan2 ?? _plan).readLogits();
    await _plan.zeroRecurrentState();
    await plan2?.zeroRecurrentState();
    // Warmup shouldn't count toward the first prompt's perf counters.
    _plan.syncMicros = 0;
    _plan.recordMicros = 0;
    plan2?.syncMicros = 0;
    plan2?.recordMicros = 0;
  }

  /// Range-read + CPU-dequant one embedding row (~2 KB for Q8_0).
  Future<Float32List> _embedRow(int token) async {
    final traits = ggmlTypeTraits[embedInfo.type]!;
    final bytesPerRow = dim ~/ traits.blockSize * traits.typeSize;
    final packed = await stream.readTensorBytes(embedInfo,
        byteOffset: token * bytesPerRow, byteLength: bytesPerRow);
    return dequantizeCpu(embedInfo.type, packed, dim);
  }

  Future<Tensor> _embed(int token) async =>
      Tensor.create([dim], data: await _embedRow(token));

  /// When true (default) and the embed table is GPU-resident, decode feeds
  /// TOKEN IDS to a GPU gather instead of the per-token ~2KB disk read +
  /// CPU dequant + row upload.  Runtime-switchable for fair in-process A/B
  /// (--ab-embed); -DGPU_ML_NO_DEC_EMBED opts out.
  bool embedGatherDecode =
      !const bool.fromEnvironment('GPU_ML_NO_DEC_EMBED');

  /// One decode step: returns the logits for [token] at [position].
  /// Runs the fired-dispatch plan(s); on a two-GPU split the hidden state
  /// hops devices once (dim*4 bytes each way).
  Future<Float32List> forward(int token, int position) async {
    final embedGpu = _embedGpu;
    final gather = embedGatherDecode && embedGpu != null;
    final Float32List logits;
    final plan2 = _plan2;
    if (plan2 == null) {
      logits = gather
          ? await _plan.forwardToken(embedGpu, token, position)
          : await _plan.forward(await _embedRow(token), position);
    } else {
      if (gather) {
        await _plan.loadTokenX(embedGpu, token);
      } else {
        await _plan.writeX(await _embedRow(token));
      }
      await _plan.runBlocks(position);
      final h = await _plan.readX();
      await plan2.writeX(h);
      await plan2.runBlocks(position);
      logits = await plan2.readLogits();
    }
    // The readbacks synchronized every fired dispatch on both devices —
    // evicted expert buffers can be reclaimed now.
    cache.flushPending();
    _cache2?.flushPending();
    return logits;
  }

  /// Token id returned by the last [forwardGreedy] (still GPU-resident in
  /// the argmax buffer); -1 when the chain is broken.
  int _lastGreedy = -1;

  /// One GREEDY decode step: feeds [token] at [position], returns the
  /// argmax'd next token id.  On the single-GPU embed-gather path this is
  /// GPU-resident end to end (fired dispatches + a 4-byte id readback); when
  /// [token] is the id the previous call returned, the embed gather chains
  /// straight off the GPU argmax buffer with zero CPU queue writes.
  /// Falls back to [forward] + a CPU argmax elsewhere.
  Future<int> forwardGreedy(int token, int position) async {
    final embedGpu = _embedGpu;
    final int next;
    if (_plan2 == null && embedGatherDecode && embedGpu != null) {
      next = token == _lastGreedy
          ? await _plan.forwardChainGreedy(embedGpu, position)
          : await _plan.forwardTokenGreedy(embedGpu, token, position);
      cache.flushPending();
    } else {
      final logits = await forward(token, position);
      var best = 0;
      var bestV = logits[0];
      for (int j = 1; j < logits.length; j++) {
        if (logits[j] > bestV) {
          bestV = logits[j];
          best = j;
        }
      }
      next = best;
    }
    _lastGreedy = next;
    return next;
  }

  /// K-token GPU-resident greedy burst (see the plan method).  The chain
  /// must be seeded by a preceding [forwardGreedy]; [token] must be the id
  /// it returned.  Returns the k generated ids (stop at EOS).
  Future<List<int>> forwardBurstGreedy(int token, int position, int k,
      {bool stopAtEos = true}) async {
    final embedGpu = _embedGpu;
    if (_plan2 != null ||
        !embedGatherDecode ||
        embedGpu == null ||
        token != _lastGreedy) {
      // Fallback: sequential greedy steps.
      final out = <int>[];
      var t = token, p = position;
      for (int i = 0; i < k; i++) {
        t = await forwardGreedy(t, p++);
        out.add(t);
        if (stopAtEos && t == tokenizer.eosId) break;
      }
      return out;
    }
    final ids = await _plan.forwardBurstGreedy(embedGpu, position, k);
    cache.flushPending();
    final out = <int>[];
    for (int i = 0; i < k; i++) {
      out.add(ids[i]);
      if (stopAtEos && ids[i] == tokenizer.eosId) break;
    }
    // On an early EOS the argmax buffer holds ids[k-1], not out.last —
    // break the chain so the next call re-seeds explicitly.
    _lastGreedy = out.length == k ? out.last : -1;
    return out;
  }

  /// Processes prompt [tokens] starting at [startPos] and returns the LAST
  /// token's logits.  Uses BATCHED prefill (chunks of up to
  /// [Qwen35DecodePlan.maxPrefillChunk] tokens, one fired pass per op) when
  /// every block's experts are resident; falls back to sequential
  /// per-token forwards otherwise.
  Future<Float32List> prefill(List<int> tokens, {int startPos = 0}) async {
    if (tokens.isEmpty) throw Exception('empty prompt');
    final plan2 = _plan2;
    final batched = _plan.supportsBatchedPrefill &&
        (plan2?.supportsBatchedPrefill ?? true) &&
        tokens.length >= 4;
    if (!batched) {
      Float32List? logits;
      var pos = startPos;
      for (final id in tokens) {
        logits = await forward(id, pos++);
      }
      return logits!;
    }

    Float32List? logits;
    var pos = startPos;
    var i = 0;
    while (i < tokens.length) {
      var t = tokens.length - i;
      if (t > Qwen35DecodePlan.maxPrefillChunk) {
        t = Qwen35DecodePlan.maxPrefillChunk;
      }
      if (t < 4) {
        // Tiny tail: run the remainder sequentially.
        for (; i < tokens.length; i++) {
          logits = await forward(tokens[i], pos++);
        }
        return logits!;
      }
      final embedGpu = _embedGpu;
      if (embedGpu != null) {
        await _plan.prefillChunkTokens(embedGpu, tokens.sublist(i, i + t), pos);
      } else {
        final rows = Float32List(t * dim);
        for (int j = 0; j < t; j++) {
          rows.setRange(
              j * dim, (j + 1) * dim, await _embedRow(tokens[i + j]));
        }
        await _plan.prefillChunk(rows, t, pos);
      }
      if (plan2 != null) {
        final h = await _plan.readPrefillX(t);
        await plan2.prefillChunk(h, t, pos);
      }
      i += t;
      pos += t;
      // Only the final chunk needs logits; earlier chunks just advance
      // KV/state (the chunk itself syncs nothing — flush via the readback).
      if (i >= tokens.length) {
        logits = await (plan2 ?? _plan).readLogits();
      }
    }
    cache.flushPending();
    _cache2?.flushPending();
    return logits!;
  }

  /// The v1 awaited-op decode path, kept as a numerical reference for
  /// debugging the plan (dispatch-per-op, CPU KV cache).  Single-GPU only —
  /// its tensor ops cannot mix devices.
  Future<Float32List> forwardReference(int token, int position) async {
    if (_plan2 != null) {
      throw Exception('forwardReference does not support a multi-GPU split');
    }
    var x = await _embed(token);

    for (final b in blocks) {
      final xn = await x.rmsNorm(b.attnNorm, eps: eps);

      Tensor attnOut;
      if (b.delta != null) {
        attnOut = await b.delta!.forward(xn);
      } else {
        final a = b.attn!;
        final proj = await a.project(xn, position);
        b.kCache.add(await proj.k.getData() as Float32List);
        b.vCache.add(await proj.v.getData() as Float32List);
        final seqLen = b.kCache.length;
        final kvSize = a.kvHeads * a.headDim;
        final kAllData = Float32List(seqLen * kvSize);
        final vAllData = Float32List(seqLen * kvSize);
        for (int t = 0; t < seqLen; t++) {
          kAllData.setRange(t * kvSize, (t + 1) * kvSize, b.kCache[t]);
          vAllData.setRange(t * kvSize, (t + 1) * kvSize, b.vCache[t]);
        }
        final kAll = await Tensor.create([seqLen, a.kvHeads, a.headDim],
            data: kAllData);
        final vAll = await Tensor.create([seqLen, a.kvHeads, a.headDim],
            data: vAllData);
        attnOut = await a.attend(
            q: proj.q, gate: proj.gate, kAll: kAll, vAll: vAll);
        proj.q.destroy();
        proj.k.destroy();
        proj.v.destroy();
        kAll.destroy();
        vAll.destroy();
      }
      xn.destroy();

      final h = await attnOut.add(x);
      attnOut.destroy();
      x.destroy();

      final hn = await h.rmsNorm(b.postNorm, eps: eps);
      final ffn = await b.moe.forward(hn);
      hn.destroy();
      x = await ffn.add(h);
      ffn.destroy();
      h.destroy();
    }

    final xn = await x.rmsNorm(outputNorm, eps: eps);
    x.destroy();
    final logitsT = await lmHead.matVec(xn);
    xn.destroy();
    final logits = await logitsT.getData() as Float32List;
    logitsT.destroy();
    return logits;
  }

  /// Greedy generation.  Returns generated token ids (prompt excluded).
  Future<List<int>> generate(
    String prompt, {
    int maxTokens = 16,
    void Function(int token, String text)? onToken,
  }) async {
    final promptIds = tokenizer.encode(prompt);
    if (promptIds.isEmpty) {
      throw Exception('empty prompt after tokenization');
    }
    int pos = promptIds.length;
    Float32List? logits = await prefill(promptIds);
    final out = <int>[];
    for (int i = 0; i < maxTokens; i++) {
      int best = 0;
      double bestV = logits![0];
      for (int j = 1; j < logits.length; j++) {
        if (logits[j] > bestV) {
          bestV = logits[j];
          best = j;
        }
      }
      if (best == tokenizer.eosId) break;
      out.add(best);
      onToken?.call(best, tokenizer.decode([best]));
      if (i + 1 < maxTokens) {
        logits = await forward(best, pos++);
      }
    }
    return out;
  }

  Future<void> destroy() async {
    _plan.destroy();
    _plan2?.destroy();
    _embedGpu?.destroy();
    _embedGpu = null;
    for (final t in _residentStacks) {
      t.destroy();
    }
    _residentStacks.clear();
    for (final b in blocks) {
      b.attnNorm.destroy();
      b.postNorm.destroy();
      b.moe.router.destroy();
      b.moe.gateShexp?.destroy();
      b.moe.upShexp?.destroy();
      b.moe.downShexp?.destroy();
      b.moe.sharedGate?.destroy();
      b.attn?.destroy();
      b.delta?.destroy();
    }
    outputNorm.destroy();
    lmHead.destroy();
    cache.destroy();
    _cache2?.destroy();
    await _gpu2?.destroy();
    await _gpu0?.destroy();
    await stream.close();
  }
}
