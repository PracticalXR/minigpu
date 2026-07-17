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

import '../gpu_ml.dart';
import 'gguf_stream.dart';

/// Byte-budgeted LRU of VRAM-resident expert matrices.
class _ExpertCache {
  _ExpertCache(this.budgetBytes);
  final int budgetBytes;
  final _entries = <String, QuantizedTensor>{};
  final _sizes = <String, int>{};
  int _used = 0;
  int hits = 0;
  int misses = 0;

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
      _entries.remove(oldest)!.destroy();
      _used -= _sizes.remove(oldest)!;
    }
    final t = await load();
    _entries[key] = t;
    _sizes[key] = sizeBytes;
    _used += sizeBytes;
    return t;
  }

  void destroy() {
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

  int get vocab => lmHead.rows;
  int get cacheHits => cache.hits;
  int get cacheMisses => cache.misses;

  static Future<Qwen35Model> load(
    String path, {
    int expertCacheBytes = 8 * 1024 * 1024 * 1024,
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

    final blocks = <_Block>[];
    for (int i = 0; i < nLayer; i++) {
      final isRecurrent = (i + 1) % interval != 0;
      onProgress?.call(
          'loading blk.$i (${isRecurrent ? 'deltanet' : 'attention'})');
      final moe = _StreamingMoe(
        stream: s,
        router: await s.loadF32('blk.$i.ffn_gate_inp.weight'),
        gateInfo: s.tensor('blk.$i.ffn_gate_exps.weight')!,
        upInfo: s.tensor('blk.$i.ffn_up_exps.weight')!,
        downInfo: s.tensor('blk.$i.ffn_down_exps.weight')!,
        topK: topK,
        cache: cache,
        blk: i,
        gateShexp: await s.loadQuantized('blk.$i.ffn_gate_shexp.weight'),
        upShexp: await s.loadQuantized('blk.$i.ffn_up_shexp.weight'),
        downShexp: await s.loadQuantized('blk.$i.ffn_down_shexp.weight'),
        sharedGate: await s.loadF32('blk.$i.ffn_gate_inp_shexp.weight'),
      );
      blocks.add(_Block(
        attnNorm: await s.loadF32('blk.$i.attn_norm.weight'),
        postNorm: await s.loadF32('blk.$i.post_attention_norm.weight'),
        moe: moe,
        attn: isRecurrent ? null : await s.loadAttentionLayer(i),
        delta: isRecurrent ? await s.loadDeltaNetLayer(i) : null,
      ));
    }

    onProgress?.call('loading lm_head');
    final model = Qwen35Model._(
      stream: s,
      tokenizer: tokenizer,
      blocks: blocks,
      outputNorm: await s.loadF32('output_norm.weight'),
      lmHead: await s.loadQuantized('output.weight'),
      embedInfo: s.tensor('token_embd.weight')!,
      dim: dim,
      eps: eps,
      cache: cache,
    );
    onProgress?.call('ready');
    return model;
  }

  /// Range-read + CPU-dequant one embedding row (~2 KB for Q8_0).
  Future<Tensor> _embed(int token) async {
    final traits = ggmlTypeTraits[embedInfo.type]!;
    final bytesPerRow = dim ~/ traits.blockSize * traits.typeSize;
    final packed = await stream.readTensorBytes(embedInfo,
        byteOffset: token * bytesPerRow, byteLength: bytesPerRow);
    final row = dequantizeCpu(embedInfo.type, packed, dim);
    return Tensor.create([dim], data: row);
  }

  /// One decode step: returns the logits for [token] at [position].
  Future<Float32List> forward(int token, int position) async {
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
    int pos = 0;
    Float32List? logits;
    for (final id in promptIds) {
      logits = await forward(id, pos++);
    }
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
    await stream.close();
  }
}
