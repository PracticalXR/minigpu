import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

import 'gpu_nn.dart';
import 'gpu_quant.dart';

/// Gated DeltaNet linear-attention layer (qwen35moe recurrent layers),
/// single-token DECODE path.  Exact formulas: docs/QWEN35MOE_SEMANTICS.md
/// (transcribed from llama.cpp build_layer_attn_linear +
/// build_delta_net_autoregressive).
///
/// Persistent per-layer state lives on GPU:
/// - conv state: previous (kernel-1) raw qkv_mixed vectors [k-1, convDim]
/// - recurrent state S per v-head, stored S_st[h][j][i] = S_math[i][j]
///   (j = v-dim row, i = k-dim contiguous), [heads * headDim * headDim]
class DeltaNetLayer {
  DeltaNetLayer({
    required this.wqkv,
    required this.wGate,
    required this.wBeta,
    required this.wAlpha,
    required this.convWeight,
    required this.ssmA,
    required this.dtBias,
    required this.ssmNorm,
    required this.wOut,
    required this.kHeads,
    required this.vHeads,
    required this.headDim,
    required this.convKernel,
    required this.convState,
    required this.ssmState,
    this.eps = 1e-6,
  });

  final QuantizedTensor wqkv; // [convDim, dim]
  final QuantizedTensor wGate; // z: [vHeads*headDim, dim]
  final QuantizedTensor wBeta; // [vHeads, dim]
  final QuantizedTensor wAlpha; // [vHeads, dim]
  final Tensor convWeight; // f32 [convDim, kernel] (ne [kernel, convDim])
  final Float32List ssmA; // [vHeads], stored negative (-exp(A_log))
  final Float32List dtBias; // [vHeads]
  final Tensor ssmNorm; // f32 [headDim]
  final QuantizedTensor wOut; // [dim, vHeads*headDim]

  final int kHeads;
  final int vHeads;
  final int headDim;
  final int convKernel;
  final double eps;

  /// [convKernel-1, convDim] rolling raw-input history (zeros initially).
  final Tensor convState;

  /// [vHeads * headDim * headDim] recurrent state (zeros initially).
  final Tensor ssmState;

  Minigpu get gpu => wqkv.gpu;
  int get dim => wqkv.cols;
  int get keyDim => kHeads * headDim;
  int get valueDim => vHeads * headDim;
  int get convDim => 2 * keyDim + valueDim;

  Buffer? _params;

  static double _softplus(double x) =>
      x > 20.0 ? x : math.log(1.0 + math.exp(x));

  /// Causal depthwise conv over [convState ; current] + SiLU, then rolls the
  /// state.  One thread per channel owns both compute and its state update.
  Future<Tensor> _convStep(Tensor qkvMixed) async {
    final out = await Tensor.create([convDim], gpu: gpu);
    final histLen = convKernel - 1;
    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> w: array<f32>;      // [convDim, kernel]
@group(0) @binding(1) var<storage, read_write> hist: array<f32>;   // [kernel-1, convDim]
@group(0) @binding(2) var<storage, read_write> cur: array<f32>;    // [convDim]
@group(0) @binding(3) var<storage, read_write> outv: array<f32>;   // [convDim]

const C: u32 = ${convDim}u;
const K: u32 = ${convKernel}u;
const HIST: u32 = ${histLen}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let c: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (c >= C) { return; }
  var acc: f32 = 0.0;
  for (var t: u32 = 0u; t < HIST; t = t + 1u) {
    acc = acc + w[c * K + t] * hist[t * C + c];
  }
  let x: f32 = cur[c];
  acc = acc + w[c * K + (K - 1u)] * x;
  outv[c] = acc / (1.0 + exp(-acc));
  // Roll history: shift left, append current RAW input.
  for (var t: u32 = 0u; t + 1u < HIST; t = t + 1u) {
    hist[t * C + c] = hist[(t + 1u) * C + c];
  }
  hist[(HIST - 1u) * C + c] = x;
}
''';
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('w', convWeight.buffer);
    shader.setBuffer('hist', convState.buffer);
    shader.setBuffer('cur', qkvMixed.buffer);
    shader.setBuffer('outv', out.buffer);
    await shader.dispatchLinear(convDim);
    return out;
  }

  /// The delta-rule recurrence for one token.  One workgroup per v-head;
  /// thread j owns state row j (v-dim).  q is pre-scaled by 1/sqrt(headDim)
  /// inside the kernel; k-head mapping is TILE (h % kHeads), matching
  /// ggml_repeat.
  Future<Tensor> _recurrence(Tensor q, Tensor k, Tensor v) async {
    if (headDim > 256) {
      throw Exception("recurrence kernel supports headDim <= 256");
    }
    final out = await Tensor.create([vHeads, headDim], gpu: gpu);
    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> q: array<f32>;      // [kHeads, D]
@group(0) @binding(1) var<storage, read_write> k: array<f32>;      // [kHeads, D]
@group(0) @binding(2) var<storage, read_write> v: array<f32>;      // [vHeads, D]
@group(0) @binding(3) var<storage, read_write> state: array<f32>;  // [vHeads, D, D] (S_st[h][j][i])
@group(0) @binding(4) var<storage, read_write> outv: array<f32>;   // [vHeads, D]
@group(0) @binding(5) var<storage, read_write> params: array<f32>; // [g[vHeads], beta[vHeads]]

const D: u32 = ${headDim}u;
const KHEADS: u32 = ${kHeads}u;
const VHEADS: u32 = ${vHeads}u;
const SCALE: f32 = ${1.0 / math.sqrt(headDim)};

var<workgroup> ks: array<f32, ${headDim}>;
var<workgroup> qs: array<f32, ${headDim}>;

@compute @workgroup_size(${headDim})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let h: u32 = wid.x;
  let j: u32 = lid.x;
  let kk: u32 = h % KHEADS; // ggml_repeat tile mapping

  ks[j] = k[kk * D + j];
  qs[j] = q[kk * D + j] * SCALE;
  workgroupBarrier();

  let decay: f32 = exp(params[h]);
  let beta: f32 = params[VHEADS + h];
  let rowBase: u32 = (h * D + j) * D;

  // Decay + S^T k in one pass over this thread's row.
  var sk: f32 = 0.0;
  for (var i: u32 = 0u; i < D; i = i + 1u) {
    let s: f32 = state[rowBase + i] * decay;
    state[rowBase + i] = s;
    sk = sk + s * ks[i];
  }

  let d: f32 = (v[h * D + j] - sk) * beta;

  // Outer-product update + S^T q readout in one pass.
  var o: f32 = 0.0;
  for (var i: u32 = 0u; i < D; i = i + 1u) {
    let s: f32 = state[rowBase + i] + ks[i] * d;
    state[rowBase + i] = s;
    o = o + s * qs[i];
  }
  outv[h * D + j] = o;
}
''';
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('q', q.buffer);
    shader.setBuffer('k', k.buffer);
    shader.setBuffer('v', v.buffer);
    shader.setBuffer('state', ssmState.buffer);
    shader.setBuffer('outv', out.buffer);
    shader.setBuffer('params', _params!);
    await shader.dispatch(vHeads, 1, 1);
    return out;
  }

  /// Single-token decode step.  [xn] is the attn_norm'ed input [dim].
  /// Returns [dim]; conv + recurrent state advance in place.
  Future<Tensor> forward(Tensor xn) async {
    // Projections.
    final qkv = await wqkv.matVec(xn); // [convDim]
    final z = await wGate.matVec(xn); // [valueDim]
    final betaT = await wBeta.matVec(xn); // [vHeads]
    final alphaT = await wAlpha.matVec(xn); // [vHeads]

    // Tiny per-head scalars -> CPU: g = a * softplus(alpha + dtBias),
    // beta = sigmoid(betaRaw); shipped to the kernel via the params buffer.
    final alpha = await alphaT.getData() as Float32List;
    final betaRaw = await betaT.getData() as Float32List;
    alphaT.destroy();
    betaT.destroy();
    final params = Float32List(vHeads * 2);
    for (int h = 0; h < vHeads; h++) {
      params[h] = ssmA[h] * _softplus(alpha[h] + dtBias[h]);
      params[vHeads + h] = 1.0 / (1.0 + math.exp(-betaRaw[h]));
    }
    _params ??= gpu.createBuffer(vHeads * 2 * 4, BufferDataType.float32);
    await _params!.write(params, vHeads * 2);

    // Conv + split + per-head L2 norm.
    final conv = await _convStep(qkv);
    qkv.destroy();
    final qRaw = await conv.sliceLinear(start: 0, end: keyDim);
    final kRaw = await conv.sliceLinear(start: keyDim, end: 2 * keyDim);
    final vT = await conv.sliceLinear(start: 2 * keyDim, end: convDim);
    conv.destroy();
    final qn = await qRaw.reshape([kHeads, headDim]).l2NormRows(eps: eps);
    final kn = await kRaw.reshape([kHeads, headDim]).l2NormRows(eps: eps);
    qRaw.destroy();
    kRaw.destroy();

    // Recurrence.
    final core = await _recurrence(qn, kn, vT);
    qn.destroy();
    kn.destroy();
    vT.destroy();

    // Gated norm: rmsnorm(core) * silu(z), per head.
    final normed = await core.rmsNorm(ssmNorm, eps: eps);
    core.destroy();
    final zAct = await z.reshape([vHeads, headDim]).silu();
    z.destroy();
    final gated = await normed.multiply(zAct);
    normed.destroy();
    zAct.destroy();

    // Output projection.
    final out = await wOut.matVec(gated.reshape([valueDim]));
    gated.destroy();
    return out;
  }

  void destroy() {
    wqkv.destroy();
    wGate.destroy();
    wBeta.destroy();
    wAlpha.destroy();
    convWeight.destroy();
    ssmNorm.destroy();
    wOut.destroy();
    convState.destroy();
    ssmState.destroy();
    _params?.destroy();
    _params = null;
  }
}
