import 'dart:math' as math;

import 'package:gpu_tensor/gpu_tensor.dart';

import 'gpu_nn.dart';
import 'gpu_quant.dart';

/// qwen35moe full-attention layer, single-token decode.  Exact semantics:
/// docs/QWEN35MOE_SEMANTICS.md (transcribed from llama.cpp build_layer_attn):
/// interleaved per-head (q, gate) projection, per-head QK RMS-norm BEFORE
/// rope, NEOX partial rope (text-mode IMRoPE), causal GQA attention with
/// contiguous head groups (q-head h -> kv-head h / (heads/kvHeads)), sigmoid
/// output gating before the output projection.
///
/// KV-cache handling is the CALLER's (runner's) job: [project] produces this
/// token's roped k and v to append; [attend] consumes the full cache.
class AttentionLayer {
  AttentionLayer({
    required this.wq,
    required this.wk,
    required this.wv,
    required this.wo,
    required this.qNorm,
    required this.kNorm,
    required this.heads,
    required this.kvHeads,
    required this.headDim,
    required this.ropeDims,
    this.ropeThetaBase = 10000.0,
    this.eps = 1e-6,
  });

  final QuantizedTensor wq; // [heads*headDim*2, dim] (interleaved q|gate per head)
  final QuantizedTensor wk; // [kvHeads*headDim, dim]
  final QuantizedTensor wv; // [kvHeads*headDim, dim]
  final QuantizedTensor wo; // [dim, heads*headDim]
  final Tensor qNorm; // f32 [headDim]
  final Tensor kNorm; // f32 [headDim]

  final int heads;
  final int kvHeads;
  final int headDim;
  final int ropeDims;
  final double ropeThetaBase;
  final double eps;

  int get dim => wq.cols;
  int get qDim => heads * headDim;
  int get kvDim => kvHeads * headDim;

  /// Projects the normed input for one token at [position]:
  /// q [heads, headDim] (normed + roped), gate [heads*headDim],
  /// k [kvHeads, headDim] (normed + roped), v [kvHeads*headDim].
  Future<({Tensor q, Tensor gate, Tensor k, Tensor v})> project(
    Tensor xn,
    int position,
  ) async {
    // Interleaved q|gate: view as [heads, 2, headDim]; q = [:,0,:], gate = [:,1,:].
    final qFull = await wq.matVec(xn); // [heads * 2 * headDim]
    final qgView = qFull.reshape([heads, 2, headDim]);
    final qRaw = await qgView.slice(
      startIndices: [0, 0, 0],
      endIndices: [heads, 1, headDim],
    );
    final gate3 = await qgView.slice(
      startIndices: [0, 1, 0],
      endIndices: [heads, 2, headDim],
    );
    qFull.destroy();

    final qNormed =
        await qRaw.reshape([heads, headDim]).rmsNorm(qNorm, eps: eps);
    qRaw.destroy();
    final q = await qNormed.ropeNeox(
      headDim: headDim,
      heads: heads,
      positionOffset: position,
      ropeDims: ropeDims,
      thetaBase: ropeThetaBase,
    );
    qNormed.destroy();

    final kRaw = await wk.matVec(xn); // [kvDim]
    final kNormed =
        await kRaw.reshape([kvHeads, headDim]).rmsNorm(kNorm, eps: eps);
    kRaw.destroy();
    final k = await kNormed.ropeNeox(
      headDim: headDim,
      heads: kvHeads,
      positionOffset: position,
      ropeDims: ropeDims,
      thetaBase: ropeThetaBase,
    );
    kNormed.destroy();

    final v = await wv.matVec(xn); // [kvDim]

    return (q: q, gate: gate3.reshape([qDim]), k: k, v: v);
  }

  /// Causal GQA attention over the FULL cache (which already includes this
  /// token's k/v as the last row):
  /// [kAll]/[vAll] are [seqLen, kvHeads, headDim].
  /// Returns the layer output [dim].
  Future<Tensor> attend({
    required Tensor q, // [heads, headDim]
    required Tensor gate, // [qDim]
    required Tensor kAll,
    required Tensor vAll,
  }) async {
    final seqLen = kAll.size ~/ (kvHeads * headDim);
    final group = heads ~/ kvHeads;

    // Contiguous GQA groups: q [kvHeads, group, headDim].
    final q3 = q.reshape([kvHeads, group, headDim]);
    final kT = await kAll
        .reshape([seqLen, kvHeads, headDim])
        .transpose(axes: [1, 2, 0]); // [kvHeads, headDim, seq]
    final scores = await q3.matMul(kT); // [kvHeads, group, seq]
    kT.destroy();
    final scaled = await scores.multiplyScalar(1.0 / math.sqrt(headDim));
    scores.destroy();
    final probs = await scaled.softmax();
    scaled.destroy();

    final v3 = await vAll
        .reshape([seqLen, kvHeads, headDim])
        .transpose(axes: [1, 0, 2]); // [kvHeads, seq, headDim]
    final attn = await probs.matMul(v3); // [kvHeads, group, headDim]
    probs.destroy();
    v3.destroy();

    // Output gating then projection.
    final gateSig = await gate.sigmoid();
    final gated = await attn.reshape([qDim]).multiply(gateSig);
    attn.destroy();
    gateSig.destroy();
    final out = await wo.matVec(gated);
    gated.destroy();
    return out;
  }

  void destroy() {
    wq.destroy();
    wk.destroy();
    wv.destroy();
    wo.destroy();
    qNorm.destroy();
    kNorm.destroy();
  }
}
