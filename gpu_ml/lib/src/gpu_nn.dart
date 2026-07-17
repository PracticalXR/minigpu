import 'dart:typed_data';

import 'package:gpu_tensor/gpu_tensor.dart';

/// Neural-network kernels for transformer inference (the llama-family set):
/// RMSNorm, rotary position embeddings (RoPE), SiLU, and GELU.
///
/// These kernels are float32; quantized WEIGHTS live in QuantizedTensor
/// (gpu_quant.dart) — activations flowing through these ops are f32.
extension TensorNN<T extends TypedData> on Tensor<T> {
  /// RMS normalization along the last dimension:
  ///   y[r, j] = x[r, j] / sqrt(mean(x[r, :]^2) + eps) * weight[j]
  ///
  /// [weight] must have shape [lastDim].  One 256-thread workgroup per row
  /// with a shared-memory tree reduction (same pattern as softmax).
  Future<Tensor<T>> rmsNorm(Tensor weight, {double eps = 1e-5}) async {
    final d = shape.last;
    final rows = size ~/ d;
    if (weight.size != d) {
      throw Exception(
        "rmsNorm weight has ${weight.size} elements, last dim is $d",
      );
    }
    final result = await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final epsLiteral = eps.toString().contains('.') || eps.toString().contains('e')
        ? eps.toString()
        : '$eps.0';
    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const D: u32 = ${d}u;
const ROWS: u32 = ${rows}u;
const WG: u32 = 256u;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let row: u32 = wid.x + wid.y * nwg.x;
  // No early return: barriers must be reached uniformly.
  let inRange: bool = row < ROWS;
  let base: u32 = select(0u, row * D, inRange);

  var acc: f32 = 0.0;
  if (inRange) {
    for (var j: u32 = lid.x; j < D; j = j + WG) {
      let v: f32 = input[base + j];
      acc = acc + v * v;
    }
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s];
    }
    workgroupBarrier();
  }
  let inv: f32 = inverseSqrt(scratch[0] / f32(D) + $epsLiteral);

  if (inRange) {
    for (var j: u32 = lid.x; j < D; j = j + WG) {
      output[base + j] = input[base + j] * inv * weight[j];
    }
  }
}
''';
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('weight', weight.buffer);
    shader.setBuffer('output', result.buffer);
    final int wgX = rows <= 65535 ? rows : 65535;
    final int wgY = (rows + wgX - 1) ~/ wgX;
    await shader.dispatch(wgX, wgY, 1);
    return result;
  }

  /// Rotary position embeddings, llama "norm" convention: within each head,
  /// consecutive element pairs (2i, 2i+1) are rotated by
  /// angle = position * thetaBase^(-2i/headDim).
  ///
  /// The tensor is interpreted as [tokens, heads, headDim] flattened
  /// (row-major); token t gets position `positionOffset + t`.  For a
  /// single-token decode step pass the tensor as [heads, headDim] (or [dim])
  /// with [positionOffset] = the token's position.
  Future<Tensor<T>> rope({
    required int headDim,
    required int heads,
    required int positionOffset,
    double thetaBase = 10000.0,
  }) async {
    if (headDim.isOdd) {
      throw Exception("rope requires even headDim, got $headDim");
    }
    if (size % (headDim * heads) != 0) {
      throw Exception(
        "Tensor size $size is not a multiple of heads*headDim (${heads * headDim})",
      );
    }
    final result = await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final totalPairs = size ~/ 2;
    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const HEAD_DIM: u32 = ${headDim}u;
const HALF: u32 = ${headDim ~/ 2}u;
const HEADS: u32 = ${heads}u;
const POS_OFFSET: u32 = ${positionOffset}u;
const TOTAL_PAIRS: u32 = ${totalPairs}u;
const THETA_BASE: f32 = $thetaBase;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let p: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (p >= TOTAL_PAIRS) { return; }
  let headRow: u32 = p / HALF;
  let i: u32 = p % HALF;
  let token: u32 = headRow / HEADS;
  let pos: f32 = f32(POS_OFFSET + token);
  let freq: f32 = pow(THETA_BASE, -f32(2u * i) / f32(HEAD_DIM));
  let angle: f32 = pos * freq;
  let c: f32 = cos(angle);
  let s: f32 = sin(angle);
  let i0: u32 = headRow * HEAD_DIM + 2u * i;
  let x0: f32 = input[i0];
  let x1: f32 = input[i0 + 1u];
  output[i0] = x0 * c - x1 * s;
  output[i0 + 1u] = x0 * s + x1 * c;
}
''';
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    await shader.dispatchLinear(totalPairs);
    return result;
  }

  /// NEOX-style rotary embeddings with PARTIAL rotation: within each
  /// [headDim]-sized head, pair index i in 0..ropeDims/2 rotates elements
  /// (x[i], x[i + ropeDims/2]) by angle position * thetaBase^(-2i/ropeDims);
  /// dims >= ropeDims pass through.  This is qwen35moe's IMRoPE reduced to
  /// text (all position channels equal — see docs/QWEN35MOE_SEMANTICS.md).
  ///
  /// Tensor is [tokens, heads, headDim] flattened; token t gets position
  /// positionOffset + t.
  Future<Tensor<T>> ropeNeox({
    required int headDim,
    required int heads,
    required int positionOffset,
    int? ropeDims,
    double thetaBase = 10000.0,
  }) async {
    final rot = ropeDims ?? headDim;
    if (rot.isOdd || rot > headDim) {
      throw Exception("ropeNeox: ropeDims must be even and <= headDim");
    }
    if (size % (headDim * heads) != 0) {
      throw Exception(
        "Tensor size $size is not a multiple of heads*headDim (${heads * headDim})",
      );
    }
    final result = await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final headRows = size ~/ headDim;
    // Each headRow needs ropeDims/2 rotation threads + (headDim - ropeDims)
    // pass-through copy threads.
    final perRow = rot ~/ 2 + (headDim - rot);
    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const HEAD_DIM: u32 = ${headDim}u;
const ROT: u32 = ${rot}u;
const HALF_ROT: u32 = ${rot ~/ 2}u;
const HEADS: u32 = ${heads}u;
const POS_OFFSET: u32 = ${positionOffset}u;
const HEAD_ROWS: u32 = ${headRows}u;
const PER_ROW: u32 = ${perRow}u;
const THETA_BASE: f32 = $thetaBase;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (t >= HEAD_ROWS * PER_ROW) { return; }
  let headRow: u32 = t / PER_ROW;
  let r: u32 = t % PER_ROW;
  let rowBase: u32 = headRow * HEAD_DIM;
  if (r < HALF_ROT) {
    let token: u32 = headRow / HEADS;
    let pos: f32 = f32(POS_OFFSET + token);
    let angle: f32 = pos * pow(THETA_BASE, -f32(2u * r) / f32(ROT));
    let c: f32 = cos(angle);
    let s: f32 = sin(angle);
    let x0: f32 = input[rowBase + r];
    let x1: f32 = input[rowBase + r + HALF_ROT];
    output[rowBase + r] = x0 * c - x1 * s;
    output[rowBase + r + HALF_ROT] = x0 * s + x1 * c;
  } else {
    let i: u32 = ROT + (r - HALF_ROT);
    output[rowBase + i] = input[rowBase + i];
  }
}
''';
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    await shader.dispatchLinear(headRows * perRow);
    return result;
  }

  /// Per-row L2 normalization: y = x / max(||x||_2, eps) along the last
  /// dimension (ggml_l2_norm semantics: eps FLOORS the norm).  Used by
  /// DeltaNet on q/k heads.
  Future<Tensor<T>> l2NormRows({double eps = 1e-6}) async {
    final d = shape.last;
    final rows = size ~/ d;
    final result = await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final epsLiteral =
        eps.toString().contains('.') || eps.toString().contains('e')
            ? eps.toString()
            : '$eps.0';
    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const D: u32 = ${d}u;
const ROWS: u32 = ${rows}u;
const WG: u32 = 256u;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let row: u32 = wid.x + wid.y * nwg.x;
  let inRange: bool = row < ROWS;
  let base: u32 = select(0u, row * D, inRange);

  var acc: f32 = 0.0;
  if (inRange) {
    for (var j: u32 = lid.x; j < D; j = j + WG) {
      let v: f32 = input[base + j];
      acc = acc + v * v;
    }
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s];
    }
    workgroupBarrier();
  }
  let inv: f32 = 1.0 / max(sqrt(scratch[0]), $epsLiteral);

  if (inRange) {
    for (var j: u32 = lid.x; j < D; j = j + WG) {
      output[base + j] = input[base + j] * inv;
    }
  }
}
''';
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    final int wgX = rows <= 65535 ? rows : 65535;
    final int wgY = (rows + wgX - 1) ~/ wgX;
    await shader.dispatch(wgX, wgY, 1);
    return result;
  }

  /// SiLU (swish): x * sigmoid(x).  The llama FFN gate activation.
  Future<Tensor<T>> silu() => _elementwiseNN('v / (1.0 + exp(-v))');

  /// GELU, tanh approximation (the transformer-standard variant):
  /// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))).
  Future<Tensor<T>> gelu() => _elementwiseNN(
        '0.5 * v * (1.0 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)))',
      );

  Future<Tensor<T>> _elementwiseNN(String expr) async {
    final result = await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= ${size}u) { return; }
  let v: f32 = A[i];
  B[i] = $expr;
}
''';
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }
}
