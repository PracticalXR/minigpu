import 'dart:async' show unawaited;
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:gpu_tensor/src/gpu_helpers.dart';
import 'package:minigpu/minigpu.dart';
import '../gpu_tensor.dart';

extension GpuFft on Tensor {
  /// Upgrades a real tensor to a complex one by interleaving a zero for the
  /// imaginary part using the GPU.
  /// For 1D tensors the output is flat ([N*2]), for others a new dimension is appended.
  Future<Tensor> upgradeRealToComplex() async {
    int total = shape.reduce((a, b) => a * b);

    // 1D: flat [N*2].  Higher ranks: append a complex dimension so the
    // result is directly consumable by fft2d ([rows, cols, 2]) / fft3d
    // ([D, R, C, 2]) — the old always-flat shape made those paths throw.
    List<int> newShape = shape.length == 1 ? [total * 2] : [...shape, 2];

    Tensor out = await Tensor.create(newShape, gpu: gpu);

    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const N: u32 = ${total}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= N) { return; }
  output[i * 2u] = input[i];
  output[i * 2u + 1u] = 0.0;
}
''';
    final shader = gpu.cachedShader(shaderTemplate);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', out.buffer);

    int workgroups = (total + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    return out;
  }

  /// Same as [upgradeRealToComplex] but writes into the caller-supplied
  /// [output] tensor instead of allocating a new one.  [output] must already
  /// have shape [N*2] where N = this.shape.reduce(*).
  /// No GPU allocations are made; the caller owns [output].
  Future<void> upgradeRealToComplexInto(Tensor output) async {
    final int total = shape.reduce((a, b) => a * b);
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const N: u32 = ${total}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= N) { return; }
  output[i * 2u] = input[i];
  output[i * 2u + 1u] = 0.0;
}
''';
    final shader = gpu.cachedShader(shaderTemplate);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', output.buffer);
    final int workgroups = (total + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
  }

  /// A unified FFT that upgrades a real tensor if necessary and then calls the
  /// appropriate FFT method based on the tensor's dimensions.
  Future<Tensor> fft({bool isRealInput = false}) async {
    // For 1D: if the tensor is real (length not even) upgrade to a flat complex tensor.
    if (shape.length == 1) {
      if (isRealInput) {
        //print("Upgrading real 1D tensor to complex for FFT operation.");
        // Input is real samples, upgrade to complex
        Tensor upgraded = await upgradeRealToComplex();
        // fft1d() creates its own internal scratch tensors; upgraded is
        // the input-promotion buffer and is no longer referenced after the
        // butterfly stages complete.  Destroy it explicitly so Dawn can
        // reclaim the GPU memory without waiting for Dart's GC finalizers.
        final result = await upgraded.fft1d();
        upgraded.destroy();
        return result;
      } else {
        // Input is already complex pairs (existing behavior)
        // Only upgrade if odd length (original logic)
        if (shape[0] % 2 != 0) {
          Tensor upgraded = await upgradeRealToComplex();
          final result = await upgraded.fft1d();
          upgraded.destroy();
          return result;
        }
        return fft1d();
      }
    }
    if (isRealInput) {
      // Real ND input: fft2d/fft3d upgrade real tensors internally.
      if (shape.length == 2) return fft2d();
      if (shape.length == 3) return fft3d();
      throw Exception("FFT not implemented for real rank-${shape.length}");
    }
    int dims = shape.length - 1;
    if (dims == 1) {
      return fft1d();
    } else if (dims == 2) {
      return fft2d();
    } else if (dims == 3) {
      return fft3d();
    } else {
      throw Exception("FFT not implemented for dimensions $dims");
    }
  }

  /// Bit-reverse reorders complex data along one axis of an interleaved
  /// complex tensor, writing into [output] (same shape).  The tensor is
  /// treated as [outer, len, inner] complex points; the `len` axis index is
  /// bit-reversed.  Required before each axis's DIT butterfly stages —
  /// fft2d/fft3d used to skip this entirely, producing wrong values for any
  /// input that isn't invariant under the bit-reversal permutation.
  Future<void> _bitReverseAxis({
    required Tensor input,
    required Tensor output,
    required int outer,
    required int len,
    required int inner,
  }) async {
    final int total = outer * len * inner;
    final int numBits = (math.log(len) / math.ln2).round();
    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const TOTAL: u32 = ${total}u;
const LEN_INNER: u32 = ${len * inner}u;
const INNER: u32 = ${inner}u;
const BITS: u32 = ${numBits}u;

fn bitReverse(x: u32, bits: u32) -> u32 {
  var result: u32 = 0u;
  var temp: u32 = x;
  for (var i: u32 = 0u; i < bits; i = i + 1u) {
    result = (result << 1u) | (temp & 1u);
    temp = temp >> 1u;
  }
  return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= TOTAL) { return; }
  let o: u32 = i / LEN_INNER;
  let rem: u32 = i % LEN_INNER;
  let k: u32 = rem / INNER;
  let r: u32 = rem % INNER;
  let src: u32 = o * LEN_INNER + bitReverse(k, BITS) * INNER + r;
  output[i * 2u] = input[src * 2u];
  output[i * 2u + 1u] = input[src * 2u + 1u];
}
''';
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', input.buffer);
    shader.setBuffer('output', output.buffer);
    await shader.dispatchLinear(total);
  }

  /// Computes a 1D FFT on a tensor representing complex numbers in interleaved format.
  /// Expects a flat tensor ([N*2]) with an even number of elements.
  /// Bit-reverse reordering for FFT output
  Future<Tensor> bitReverseReorder(Tensor input) async {
    final n = input.shape[0] ~/ 2; // number of complex points
    if (n <= 1) return input;

    final output = await Tensor.create(input.shape, gpu: gpu);
    final numBits = (math.log(n) / math.ln2).round();

    // This should now work with correct input size
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

fn bitReverse(x: u32, bits: u32) -> u32 {
  var result: u32 = 0u;
  var temp: u32 = x;
  for (var i: u32 = 0u; i < bits; i = i + 1u) {
    result = (result << 1u) | (temp & 1u);
    temp = temp >> 1u;
  }
  return result;
}

const N: u32 = ${n}u;
const BITS: u32 = ${numBits}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= N) { return; }
  
  let j: u32 = bitReverse(i, BITS);
  
  let srcIdx: u32 = j * 2u;
  let dstIdx: u32 = i * 2u;
  
  output[dstIdx] = input[srcIdx];
  output[dstIdx + 1u] = input[srcIdx + 1u];
}
''';

    final shader = gpu.cachedShader(shaderTemplate);
    shader.setBuffer('input', input.buffer);
    shader.setBuffer('output', output.buffer);

    int workgroups = (n + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    return output;
  }

  /// Same as [bitReverseReorder] but writes into the caller-supplied
  /// [outputTensor] instead of allocating a new one.  [outputTensor] must
  /// already have the same shape as [input].
  /// No GPU allocations are made; the caller owns [outputTensor].
  Future<void> bitReverseReorderInto(Tensor input, Tensor outputTensor) async {
    final n = input.shape[0] ~/ 2;
    if (n <= 1) return;

    final numBits = (math.log(n) / math.ln2).round();
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

fn bitReverse(x: u32, bits: u32) -> u32 {
  var result: u32 = 0u;
  var temp: u32 = x;
  for (var i: u32 = 0u; i < bits; i = i + 1u) {
    result = (result << 1u) | (temp & 1u);
    temp = temp >> 1u;
  }
  return result;
}

const N: u32 = ${n}u;
const BITS: u32 = ${numBits}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= N) { return; }
  let j: u32 = bitReverse(i, BITS);
  let srcIdx: u32 = j * 2u;
  let dstIdx: u32 = i * 2u;
  output[dstIdx] = input[srcIdx];
  output[dstIdx + 1u] = input[srcIdx + 1u];
}
''';
    final shader = gpu.cachedShader(shaderTemplate);
    shader.setBuffer('input', input.buffer);
    shader.setBuffer('output', outputTensor.buffer);
    final int workgroups = (n + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
  }

  /// Runs a 1D FFT using caller-supplied, pre-allocated workspace tensors and
  /// buffers.  NOTHING is allocated or destroyed — the caller owns all objects.
  ///
  /// [this] must contain the bit-reversed complex input (shape [N*2]).
  /// [pong] is the scratch tensor (same shape as [this]).
  /// [twiddleBuffer] must hold N precomputed twiddle factors (N*2 floats).
  /// [paramBuffer] is a 16-byte uniform-like buffer for per-stage params.
  ///
  /// Returns either [this] or [pong] depending on FFT stage count parity —
  /// the caller must not assume which one holds the result.
  Future<Tensor> fft1dPreallocated({
    required Tensor pong,
    required Buffer twiddleBuffer,
    required Buffer paramBuffer,
  }) async {
    if (shape.length != 1)
      throw Exception("fft1dPreallocated requires 1D tensor.");
    final int n = shape[0] ~/ 2;
    final int stages = (math.log(n) / math.ln2).toInt();

    Tensor ping = this;
    Tensor pongRef = pong;

    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> params: array<u32>; // [stage, n, m, half]
@group(0) @binding(3) var<storage, read_write> twiddles: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n: u32 = params[1];
  let m: u32 = params[2];
  let half: u32 = params[3];
  let numOperations: u32 = n >> 1u;
  let t: u32 = gid.x;
  if (t >= numOperations) { return; }
  let group: u32 = t / half;
  let pos: u32 = t % half;
  let i0: u32 = group * m + pos;
  let i1: u32 = i0 + half;
  if (i1 >= n) { return; }
  let idx0: u32 = i0 * 2u;
  let idx1: u32 = i1 * 2u;
  let a: vec2<f32> = vec2<f32>(input[idx0], input[idx0 + 1u]);
  let b: vec2<f32> = vec2<f32>(input[idx1], input[idx1 + 1u]);
  let twiddle_stride: u32 = n / m;
  let twiddle_idx: u32 = pos * twiddle_stride * 2u;
  let w: vec2<f32> = vec2<f32>(twiddles[twiddle_idx], twiddles[twiddle_idx + 1u]);
  let b_w: vec2<f32> = vec2<f32>(b.x * w.x - b.y * w.y, b.x * w.y + b.y * w.x);
  let result1: vec2<f32> = a + b_w;
  let result2: vec2<f32> = a - b_w;
  output[idx0] = result1.x;
  output[idx0 + 1u] = result1.y;
  output[idx1] = result2.x;
  output[idx1 + 1u] = result2.y;
}
''';
    // Reuse cached shader on this tensor (the caller's persistent complex tensor).
    final shader = gpu.cachedShader(shaderTemplate);

    for (int s = 0; s < stages; s++) {
      final int m = 1 << (s + 1);
      final int half = m >> 1;
      final params = Uint32List(4);
      params[0] = s;
      params[1] = n;
      params[2] = m;
      params[3] = half;
      await paramBuffer.write(
        params,
        params.length,
        dataType: BufferDataType.uint32,
      );

      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pongRef.buffer);
      shader.setBuffer('params', paramBuffer);
      shader.setBuffer('twiddles', twiddleBuffer);

      final int numOperations = n >> 1;
      final int workgroups = (numOperations + 255) ~/ 256;
      await shader.dispatch(workgroups, 1, 1);

      final Tensor tmp = ping;
      ping = pongRef;
      pongRef = tmp;
    }
    return ping;
  }

  Future<Tensor> fft1d() async {
    if (shape.length != 1) {
      throw Exception("FFT supports 1D tensors only.");
    }
    int totalFloats = shape[0];
    if (totalFloats % 2 != 0) {
      throw Exception("FFT tensor length must be even (for complex numbers).");
    }
    int n = totalFloats ~/ 2;
    if ((n & (n - 1)) != 0) {
      throw Exception("FFT size ($n) must be a power of 2.");
    }
    if (n == 1) return this;

    // Precompute all twiddle factors
    final twiddleFactors = Float32List(n * 2);
    for (int i = 0; i < n; i++) {
      double angle = -2.0 * math.pi * i / n;
      twiddleFactors[i * 2] = math.cos(angle);
      twiddleFactors[i * 2 + 1] = math.sin(angle);
    }

    final Buffer twiddleBuffer = gpu.createBuffer(
      n * 2 * 4,
      BufferDataType.float32,
    );
    await twiddleBuffer.write(twiddleFactors, twiddleFactors.length);

    // Add bit-reversal back
    Tensor bitReversed = await bitReverseReorder(this);
    int stages = (math.log(n) / math.ln2).toInt();

    Tensor ping = bitReversed;
    Tensor pong = await Tensor.create(shape, gpu: gpu);
    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> params: array<u32>; // [stage, n, m, half]
@group(0) @binding(3) var<storage, read_write> twiddles: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n: u32 = params[1];
  let m: u32 = params[2];
  let half: u32 = params[3];
  let numOperations: u32 = n >> 1u;
  
  let t: u32 = gid.x;
  if (t >= numOperations) { return; }
  
  let group: u32 = t / half;
  let pos: u32 = t % half;
  let i0: u32 = group * m + pos;
  let i1: u32 = i0 + half;
  
  if (i1 >= n) { return; }
  
  let idx0: u32 = i0 * 2u;
  let idx1: u32 = i1 * 2u;
  
  // Load complex numbers as vec2 for better performance
  let a: vec2<f32> = vec2<f32>(input[idx0], input[idx0 + 1u]);
  let b: vec2<f32> = vec2<f32>(input[idx1], input[idx1 + 1u]);
  
  // Use precomputed twiddle factors
  let twiddle_stride: u32 = n / m;
  let twiddle_idx: u32 = pos * twiddle_stride * 2u;
  let w: vec2<f32> = vec2<f32>(twiddles[twiddle_idx], twiddles[twiddle_idx + 1u]);
  
  // Complex multiplication: b * w
  let b_w: vec2<f32> = vec2<f32>(
    b.x * w.x - b.y * w.y,
    b.x * w.y + b.y * w.x
  );
  
  // Butterfly operation
  let result1: vec2<f32> = a + b_w;
  let result2: vec2<f32> = a - b_w;
  
  // Store results
  output[idx0] = result1.x;
  output[idx0 + 1u] = result1.y;
  output[idx1] = result2.x;
  output[idx1 + 1u] = result2.y;
}
''';
    final shader = gpu.cachedShader(shaderTemplate);

    final Buffer paramBuffer = gpu.createBuffer(16, BufferDataType.uint32);

    try {
      for (int s = 0; s < stages; s++) {
        int m = 1 << (s + 1);
        int half = m >> 1;

        final params = Uint32List(4);
        params[0] = s; // stage
        params[1] = n; // n
        params[2] = m; // m
        params[3] = half; // half
        await paramBuffer.write(
          params,
          params.length,
          dataType: BufferDataType.uint32,
        );

        shader.setBuffer('input', ping.buffer);
        shader.setBuffer('output', pong.buffer);
        shader.setBuffer('params', paramBuffer);
        shader.setBuffer('twiddles', twiddleBuffer);

        int numOperations = n >> 1;
        int workgroups = (numOperations + 255) ~/ 256;
        await shader.dispatch(workgroups, 1, 1);

        Tensor temp = ping;
        ping = pong;
        pong = temp;
      }
    } finally {
      paramBuffer.destroy();
      twiddleBuffer.destroy();
      // pong is always the stale scratch buffer after the final swap.
      // ping is the valid result that will be returned to the caller.
      // Do NOT destroy bitReversed separately: it is always either ping
      // (returned, freed by caller) or pong (freed here), depending on
      // whether stages is even or odd. Calling bitReversed.destroy() here
      // would cause a double-free (if pong==bitReversed) or destroy the
      // return value (if ping==bitReversed).
      pong.destroy();
    }

    return ping;
  }

  /// Computes a 2D FFT. If a real tensor is supplied, it is upgraded.
  ///
  /// The input tensor is never mutated; the result is a fresh tensor owned
  /// by the caller.
  Future<Tensor> fft2d() async {
    // Upgrade real tensor case.
    if (shape.length == 2) {
      int rows = shape[0], cols = shape[1];
      bool isPow2(int x) => (x & (x - 1)) == 0;
      if (!isPow2(rows) || !isPow2(cols)) {
        throw Exception("Both rows and cols must be powers of 2.");
      }
      // Use GPU to upgrade instead of CPU iteration.
      final Tensor upgraded = await upgradeRealToComplex();
      final result = await upgraded.fft2d();
      upgraded.destroy();
      return result;
    }
    if (shape.length != 3 || shape[2] != 2) {
      throw Exception("fft2d requires a tensor of shape [rows, cols, 2].");
    }
    int rows = shape[0], cols = shape[1];
    bool isPow2(int x) => (x & (x - 1)) == 0;
    if (!isPow2(rows) || !isPow2(cols)) {
      throw Exception("Both rows and cols must be powers of 2.");
    }

    // Two scratch tensors ping-pong through every pass; `this` is only ever
    // READ (by the first bit-reverse).  The old code ping-ponged through
    // `this` (mutating the caller's input on odd swap counts) and leaked the
    // first scratch when allocating a second one for the column pass.
    Tensor ping = await Tensor.create(shape, gpu: gpu);
    Tensor pong = await Tensor.create(shape, gpu: gpu);

    // FFT on rows (transform along the column index).
    // DIT butterflies require bit-reversed input order along the axis.
    await _bitReverseAxis(
      input: this,
      output: ping,
      outer: rows,
      len: cols,
      inner: 1,
    );
    int stagesRow = (math.log(cols) / math.ln2).toInt();
    for (int s = 0; s < stagesRow; s++) {
      int m = 1 << (s + 1);
      int half = m >> 1;
      int numOperations = rows * (cols >> 1);
      final shaderTemplate =
          '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let t: u32 = gid.x;
  if (t >= ${numOperations}u) { return; }
  let row: u32 = t / ${(cols >> 1)}u;
  let t_row: u32 = t % ${(cols >> 1)}u;
  let half: u32 = ${half}u;
  let m: u32 = ${m}u;
  let group: u32 = t_row / half;
  let pos: u32 = t_row % half;
  let i0: u32 = group * m + pos;
  let i1: u32 = i0 + half;
  let base: u32 = row * (${cols}u * 2u);
  let idx0: u32 = base + i0 * 2u;
  let idx1: u32 = base + i1 * 2u;
  let a: vec2<f32> = vec2<f32>(input[idx0], input[idx0+1u]);
  let b: vec2<f32> = vec2<f32>(input[idx1], input[idx1+1u]);
  let angle: f32 = -6.28318530718 * f32(pos) / f32(m);
  let w: vec2<f32> = vec2<f32>(cos(angle), sin(angle));
  let b_twiddled: vec2<f32> = vec2<f32>(
    b.x * w.x - b.y * w.y,
    b.x * w.y + b.y * w.x
  );
  let temp1: vec2<f32> = a + b_twiddled;
  let temp2: vec2<f32> = a - b_twiddled;
  output[idx0] = temp1.x;
  output[idx0+1u] = temp1.y;
  output[idx1] = temp2.x;
  output[idx1+1u] = temp2.y;
}
''';
      final shader = gpu.cachedShader(shaderTemplate);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      await shader.dispatchLinear(numOperations);
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }
    // FFT on columns (transform along the row index) — bit-reverse first.
    await _bitReverseAxis(
      input: ping,
      output: pong,
      outer: 1,
      len: rows,
      inner: cols,
    );
    {
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }
    int stagesCol = (math.log(rows) / math.ln2).toInt();
    for (int s = 0; s < stagesCol; s++) {
      int m = 1 << (s + 1);
      int half = m >> 1;
      int numOperations = cols * (rows >> 1);
      final shaderTemplate =
          '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let t: u32 = gid.x;
  if (t >= ${numOperations}u) { return; }
  let col: u32 = t / ${(rows >> 1)}u;
  let t_col: u32 = t % ${(rows >> 1)}u;
  let half: u32 = ${half}u;
  let m: u32 = ${m}u;
  let group: u32 = t_col / half;
  let pos: u32 = t_col % half;
  let i0: u32 = group * m + pos;
  let i1: u32 = i0 + half;
  let stride: u32 = ${cols}u * 2u;
  let base0: u32 = i0 * stride + col * 2u;
  let base1: u32 = i1 * stride + col * 2u;
  let a: vec2<f32> = vec2<f32>(input[base0], input[base0+1u]);
  let b: vec2<f32> = vec2<f32>(input[base1], input[base1+1u]);
  let angle: f32 = -6.28318530718 * f32(pos) / f32(m);
  let w: vec2<f32> = vec2<f32>(cos(angle), sin(angle));
  let b_twiddled: vec2<f32> = vec2<f32>(
    b.x * w.x - b.y * w.y,
    b.x * w.y + b.y * w.x
  );
  let temp1: vec2<f32> = a + b_twiddled;
  let temp2: vec2<f32> = a - b_twiddled;
  output[base0] = temp1.x;
  output[base0+1u] = temp1.y;
  output[base1] = temp2.x;
  output[base1+1u] = temp2.y;
}
''';
      final shader = gpu.cachedShader(shaderTemplate);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      await shader.dispatchLinear(numOperations);
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }
    pong.destroy();
    return ping;
  }

  /// Computes a 3D FFT on a tensor representing complex numbers in interleaved format.
  /// If a real tensor with shape [D, R, C] is supplied, it is upgraded.
  ///
  /// The input tensor is never mutated; the result is a fresh tensor owned
  /// by the caller.
  Future<Tensor> fft3d() async {
    // Upgrade case: if a real tensor is supplied.
    if (shape.length == 3) {
      int D = shape[0], R = shape[1], C = shape[2];
      bool isPow2(int x) => (x & (x - 1)) == 0;
      if (!isPow2(D) || !isPow2(R) || !isPow2(C)) {
        throw Exception("D, R, and C must be powers of 2.");
      }
      // Upgrade on GPU.
      final Tensor upgraded = await upgradeRealToComplex();
      final result = await upgraded.fft3d();
      upgraded.destroy();
      return result;
    }

    // If already complex, expect shape [D, R, C, 2].
    if (shape.length != 4 || shape[3] != 2) {
      throw Exception(
        "fft3d requires a tensor of shape [D,R,C,2] or a real tensor of shape [D,R,C].",
      );
    }
    int D = shape[0], R = shape[1], C = shape[2];
    bool isPow2(int x) => (x & (x - 1)) == 0;
    if (!isPow2(D) || !isPow2(R) || !isPow2(C)) {
      throw Exception("D, R, and C must be powers of 2.");
    }

    // Same non-mutating ping/pong discipline as fft2d, with a bit-reversal
    // reorder before each axis's DIT butterfly stages.
    Tensor ping = await Tensor.create(shape, gpu: gpu);
    Tensor pong = await Tensor.create(shape, gpu: gpu);
    await _bitReverseAxis(
      input: this,
      output: ping,
      outer: 1,
      len: D,
      inner: R * C,
    );

    // FFT along depth dimension.
    int stagesD = (math.log(D) / math.ln2).toInt();
    for (int s = 0; s < stagesD; s++) {
      int m = 1 << (s + 1);
      int half = m >> 1;
      int numOperations = R * C * (D >> 1);
      final shaderTemplate =
          '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const R: u32 = ${R}u;
const C: u32 = ${C}u;
const D: u32 = ${D}u;
const m: u32 = ${m}u;
const half: u32 = ${half}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t: u32 = gid.x;
  if (t >= ${numOperations}u) { return; }
  // Decode row and col.
  let rc: u32 = t / (D >> 1u);
  let pos: u32 = t % (D >> 1u);
  let row: u32 = rc / C;
  let col: u32 = rc % C;
  let group: u32 = pos / half;
  let offset: u32 = pos % half;
  let d0: u32 = group * m + offset;
  let d1: u32 = d0 + half;
  let strideD: u32 = R * C * 2u;
  let strideR: u32 = C * 2u;
  let base: u32 = row * strideR + col * 2u;
  let idx0: u32 = d0 * strideD + base;
  let idx1: u32 = d1 * strideD + base;
  let a: vec2<f32> = vec2<f32>(input[idx0], input[idx0+1u]);
  let b: vec2<f32> = vec2<f32>(input[idx1], input[idx1+1u]);
  let angle: f32 = -6.28318530718 * f32(offset) / f32(m);
  let w: vec2<f32> = vec2<f32>(cos(angle), sin(angle));
  let b_twiddled: vec2<f32> = vec2<f32>(
    b.x * w.x - b.y * w.y,
    b.x * w.y + b.y * w.x
  );
  let temp1: vec2<f32> = a + b_twiddled;
  let temp2: vec2<f32> = a - b_twiddled;
  output[idx0] = temp1.x;
  output[idx0+1u] = temp1.y;
  output[idx1] = temp2.x;
  output[idx1+1u] = temp2.y;
}
''';
      final shader = gpu.cachedShader(shaderTemplate);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      await shader.dispatchLinear(numOperations);
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }

    // FFT along row dimension — bit-reverse the row axis first.
    await _bitReverseAxis(
      input: ping,
      output: pong,
      outer: D,
      len: R,
      inner: C,
    );
    {
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }
    int stagesR = (math.log(R) / math.ln2).toInt();
    for (int s = 0; s < stagesR; s++) {
      int m = 1 << (s + 1);
      int half = m >> 1;
      int numOperations = D * C * (R >> 1);
      final shaderTemplate =
          '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const D: u32 = ${D}u;
const R: u32 = ${R}u;
const C: u32 = ${C}u;
const m: u32 = ${m}u;
const half: u32 = ${half}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t: u32 = gid.x;
  if (t >= ${numOperations}u) { return; }
  // Decode depth and col.
  let dc: u32 = t / (R >> 1u);
  let pos: u32 = t % (R >> 1u);
  let depth: u32 = dc / C;
  let col: u32 = dc % C;
  let group: u32 = pos / half;
  let offset: u32 = pos % half;
  let r0: u32 = group * m + offset;
  let r1: u32 = r0 + half;
  let strideD: u32 = R * C * 2u;
  let strideR: u32 = C * 2u;
  let base: u32 = depth * strideD + col * 2u;
  let idx0: u32 = base + r0 * strideR;
  let idx1: u32 = base + r1 * strideR;
  let a: vec2<f32> = vec2<f32>(input[idx0], input[idx0+1u]);
  let b: vec2<f32> = vec2<f32>(input[idx1], input[idx1+1u]);
  let angle: f32 = -6.28318530718 * f32(offset) / f32(m);
  let w: vec2<f32> = vec2<f32>(cos(angle), sin(angle));
  let b_twiddled: vec2<f32> = vec2<f32>(
    b.x * w.x - b.y * w.y,
    b.x * w.y + b.y * w.x
  );
  let temp1: vec2<f32> = a + b_twiddled;
  let temp2: vec2<f32> = a - b_twiddled;
  output[idx0] = temp1.x;
  output[idx0+1u] = temp1.y;
  output[idx1] = temp2.x;
  output[idx1+1u] = temp2.y;
}
''';
      final shader = gpu.cachedShader(shaderTemplate);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      await shader.dispatchLinear(numOperations);
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }

    // FFT along column dimension — bit-reverse the column axis first.
    await _bitReverseAxis(
      input: ping,
      output: pong,
      outer: D * R,
      len: C,
      inner: 1,
    );
    {
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }
    int stagesC = (math.log(C) / math.ln2).toInt();
    for (int s = 0; s < stagesC; s++) {
      int m = 1 << (s + 1);
      int half = m >> 1;
      int numOperations = D * R * (C >> 1);
      final shaderTemplate =
          '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const D: u32 = ${D}u;
const R: u32 = ${R}u;
const C: u32 = ${C}u;
const m: u32 = ${m}u;
const half: u32 = ${half}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let t: u32 = gid.x;
  if (t >= ${numOperations}u) { return; }
  // Decode depth and row.
  let dr: u32 = t / (C >> 1u);
  let pos: u32 = t % (C >> 1u);
  let depth: u32 = dr / R;
  let row: u32 = dr % R;
  let group: u32 = pos / half;
  let offset: u32 = pos % half;
  let c0: u32 = group * m + offset;
  let c1: u32 = c0 + half;
  let strideD: u32 = R * C * 2u;
  let strideR: u32 = C * 2u;
  let base: u32 = depth * strideD + row * strideR;
  let idx0: u32 = base + c0 * 2u;
  let idx1: u32 = base + c1 * 2u;
  let a: vec2<f32> = vec2<f32>(input[idx0], input[idx0+1u]);
  let b: vec2<f32> = vec2<f32>(input[idx1], input[idx1+1u]);
  let angle: f32 = -6.28318530718 * f32(offset) / f32(m);
  let w: vec2<f32> = vec2<f32>(cos(angle), sin(angle));
  let b_twiddled: vec2<f32> = vec2<f32>(
    b.x * w.x - b.y * w.y,
    b.x * w.y + b.y * w.x
  );
  let temp1: vec2<f32> = a + b_twiddled;
  let temp2: vec2<f32> = a - b_twiddled;
  output[idx0] = temp1.x;
  output[idx0+1u] = temp1.y;
  output[idx1] = temp2.x;
  output[idx1+1u] = temp2.y;
}
''';
      final shader = gpu.cachedShader(shaderTemplate);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      await shader.dispatchLinear(numOperations);
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }
    pong.destroy();
    return ping;
  }
}

/// A reusable execution plan for repeated same-size 1D real-input FFTs
/// (the audio-frame hot path).
///
/// [GpuFft.fft1d] pays a heavy per-call tax that a per-frame caller cannot
/// afford: twiddle factors recomputed on the CPU (2N trig calls), four GPU
/// buffers created and destroyed, and — worst — every butterfly stage
/// individually awaited with a param-buffer write in between, serialising
/// ~2·log2(N)+4 Dart↔GPU round trips per frame.
///
/// The plan compiles everything once per (gpu, n):
///  • per-stage butterfly shaders with n/m/half baked in as WGSL constants
///    (no param buffer, no per-stage writes),
///  • real→complex promotion and bit-reversal shaders,
///  • the twiddle buffer and all workspace tensors.
///
/// [executeReal] then enqueues the whole chain — promotion, bit-reversal and
/// all log2(N) butterfly stages — back-to-back on minigpu's FIFO WebGPU
/// thread and awaits only the FINAL dispatch: one Dart↔GPU round trip per
/// FFT instead of ~34 at N = 32768.
///
/// The shaders are PLAN-OWNED, deliberately NOT acquired through the global
/// [CachedShaderAcquire] cache: a fire-and-forget dispatch reads its buffer
/// bindings when the GPU task RUNS, so a shader object shared with any other
/// caller could be rebound while this plan's tasks are still queued.  A
/// private object dispatched at most once per [executeReal] cannot race.
///
/// Not reentrant: callers must not start a second [executeReal] before the
/// previous one completes (the final await guarantees every queued task has
/// executed, so sequential callers are always safe).
class Fft1dPlan {
  Fft1dPlan._(
    this.gpu,
    this.n,
    this._stages,
    this._upgradeShader,
    this._bitrevShader,
    this._butterflyShaders,
    this._twiddleBuffer,
    this._complex,
    this._bitRev,
    this._pong,
  );

  final Minigpu gpu;

  /// FFT size: number of real input samples = number of complex points.
  final int n;

  final int _stages;
  final ComputeShader _upgradeShader;
  final ComputeShader _bitrevShader;
  final List<ComputeShader> _butterflyShaders;
  final Buffer _twiddleBuffer;
  final Tensor _complex; // real→complex promotion target [n*2]
  final Tensor _bitRev; // bit-reversed input / butterfly ping [n*2]
  final Tensor _pong; // butterfly pong [n*2]
  bool _destroyed = false;

  /// The tensor [executeReal] returns.  Fixed for a given plan (the butterfly
  /// ping/pong parity depends only on the stage count), so downstream code
  /// can bind it once.
  Tensor get output => _stages.isEven ? _bitRev : _pong;

  /// Builds a plan for [n]-point real-input FFTs ([n] must be a power of two
  /// ≥ 2).  [scale] is baked into the final butterfly stage — pass `1 / n`
  /// to fold the conventional 1/N normalisation into the FFT itself and save
  /// a separate normalisation pass.
  static Future<Fft1dPlan> create(
    Minigpu gpu,
    int n, {
    double scale = 1.0,
  }) async {
    if (n < 2 || (n & (n - 1)) != 0) {
      throw ArgumentError.value(n, 'n', 'must be a power of 2 and >= 2');
    }
    final int stages = (math.log(n) / math.ln2).round();

    final complex = await Tensor.create([n * 2], gpu: gpu);
    final bitRev = await Tensor.create([n * 2], gpu: gpu);
    final pong = await Tensor.create([n * 2], gpu: gpu);

    final twiddleFactors = Float32List(n * 2);
    for (int i = 0; i < n; i++) {
      final double angle = -2.0 * math.pi * i / n;
      twiddleFactors[i * 2] = math.cos(angle);
      twiddleFactors[i * 2 + 1] = math.sin(angle);
    }
    final twiddleBuffer = gpu.createBuffer(n * 2 * 4, BufferDataType.float32);
    await twiddleBuffer.write(twiddleFactors, twiddleFactors.length);

    final upgradeShader = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const N: u32 = ${n}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= N) { return; }
  output[i * 2u] = input[i];
  output[i * 2u + 1u] = 0.0;
}
''');

    final int numBits = stages;
    final bitrevShader = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const N: u32 = ${n}u;
const BITS: u32 = ${numBits}u;

fn bitReverse(x: u32, bits: u32) -> u32 {
  var result: u32 = 0u;
  var temp: u32 = x;
  for (var i: u32 = 0u; i < bits; i = i + 1u) {
    result = (result << 1u) | (temp & 1u);
    temp = temp >> 1u;
  }
  return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= N) { return; }
  let j: u32 = bitReverse(i, BITS);
  output[i * 2u] = input[j * 2u];
  output[i * 2u + 1u] = input[j * 2u + 1u];
}
''');

    final butterflies = <ComputeShader>[];
    for (int s = 0; s < stages; s++) {
      final int m = 1 << (s + 1);
      final int half = m >> 1;
      final bool last = s == stages - 1;
      // Scale only on the last stage so intermediate stages stay exact.
      final String store = last && scale != 1.0
          ? '''
  output[idx0] = result1.x * SCALE;
  output[idx0 + 1u] = result1.y * SCALE;
  output[idx1] = result2.x * SCALE;
  output[idx1 + 1u] = result2.y * SCALE;'''
          : '''
  output[idx0] = result1.x;
  output[idx0 + 1u] = result1.y;
  output[idx1] = result2.x;
  output[idx1 + 1u] = result2.y;''';
      butterflies.add(
        gpu.createComputeShader()
          ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> twiddles: array<f32>;

const N: u32 = ${n}u;
const M: u32 = ${m}u;
const HALF: u32 = ${half}u;
${last && scale != 1.0 ? 'const SCALE: f32 = $scale;' : ''}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let numOperations: u32 = N >> 1u;
  let t: u32 = gid.x;
  if (t >= numOperations) { return; }
  let group: u32 = t / HALF;
  let pos: u32 = t % HALF;
  let i0: u32 = group * M + pos;
  let i1: u32 = i0 + HALF;
  if (i1 >= N) { return; }
  let idx0: u32 = i0 * 2u;
  let idx1: u32 = i1 * 2u;
  let a: vec2<f32> = vec2<f32>(input[idx0], input[idx0 + 1u]);
  let b: vec2<f32> = vec2<f32>(input[idx1], input[idx1 + 1u]);
  let twiddle_idx: u32 = pos * (N / M) * 2u;
  let w: vec2<f32> = vec2<f32>(twiddles[twiddle_idx], twiddles[twiddle_idx + 1u]);
  let b_w: vec2<f32> = vec2<f32>(b.x * w.x - b.y * w.y, b.x * w.y + b.y * w.x);
  let result1: vec2<f32> = a + b_w;
  let result2: vec2<f32> = a - b_w;
$store
}
'''),
      );
    }

    return Fft1dPlan._(
      gpu,
      n,
      stages,
      upgradeShader,
      bitrevShader,
      butterflies,
      twiddleBuffer,
      complex,
      bitRev,
      pong,
    );
  }

  /// Runs the full real-input FFT chain on [realInput] (shape `[n]`, real
  /// samples) and returns the interleaved complex result (shape `[n*2]`,
  /// identical to [output]).
  ///
  /// The returned tensor is plan-owned and valid until the next
  /// [executeReal] call or [destroy].
  Future<Tensor> executeReal(Tensor realInput) async {
    if (_destroyed) throw StateError('Fft1dPlan used after destroy()');
    if (realInput.size != n) {
      throw ArgumentError(
        'realInput has ${realInput.size} samples, plan expects $n',
      );
    }

    // All dispatches below are enqueued to minigpu's single WebGPU thread in
    // call order; only the final one is awaited.  Bindings are re-set every
    // frame (cheap pointer-compare no-ops when unchanged) and each shader is
    // private to this plan and dispatched exactly once per call, so nothing
    // can rebind a shader between its enqueue and its execution.
    final int wgN = (n + 255) ~/ 256;

    _upgradeShader.setBuffer('input', realInput.buffer);
    _upgradeShader.setBuffer('output', _complex.buffer);
    unawaited(_upgradeShader.dispatch(wgN, 1, 1));

    _bitrevShader.setBuffer('input', _complex.buffer);
    _bitrevShader.setBuffer('output', _bitRev.buffer);
    unawaited(_bitrevShader.dispatch(wgN, 1, 1));

    Tensor ping = _bitRev;
    Tensor pong = _pong;
    final int wgOps = ((n >> 1) + 255) ~/ 256;
    for (int s = 0; s < _stages; s++) {
      final shader = _butterflyShaders[s];
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      shader.setBuffer('twiddles', _twiddleBuffer);
      final done = shader.dispatch(wgOps, 1, 1);
      if (s == _stages - 1) {
        await done;
      } else {
        unawaited(done);
      }
      final Tensor tmp = ping;
      ping = pong;
      pong = tmp;
    }
    return ping;
  }

  /// Releases all plan-owned GPU resources.  Safe to call once; the plan is
  /// unusable afterwards.  Must not be called while an [executeReal] is in
  /// flight.
  void destroy() {
    if (_destroyed) return;
    _destroyed = true;
    _upgradeShader.destroy();
    _bitrevShader.destroy();
    for (final s in _butterflyShaders) {
      s.destroy();
    }
    _twiddleBuffer.destroy();
    _complex.destroy();
    _bitRev.destroy();
    _pong.destroy();
  }
}
