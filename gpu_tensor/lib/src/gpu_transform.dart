import 'dart:math' as math;
import 'package:gpu_tensor/src/gpu_helpers.dart';
import 'package:minigpu/minigpu.dart';
import '../gpu_tensor.dart';

extension GpuFft on Tensor {
  /// Upgrades a real tensor to a complex one by interleaving a zero for the
  /// imaginary part using the GPU.
  /// For 1D tensors the output is flat ([N*2]), for others a new dimension is appended.
  Future<Tensor> upgradeRealToComplex() async {
    int total = shape.reduce((a, b) => a * b);
    // For 1D tensors, produce a flat complex tensor.
    List<int> newShape = (shape.length == 1)
        ? [total * 2]
        : (List.from(shape)..add(2));
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
    final ComputeShader shader = gpu.createComputeShader();
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'total': total,
    });
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', out.buffer);
    int workgroups = (total + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    buffer.destroy();
    shader.destroy();
    return out;
  }

  /// A unified FFT that upgrades a real tensor if necessary and then calls the
  /// appropriate FFT method based on the tensor's dimensions.
  Future<Tensor> fft({bool isRealInput = false}) async {
    // For 1D: if the tensor is real (length not even) upgrade to a flat complex tensor.
    if (shape.length == 1) {
      if (isRealInput) {
        print("Upgrading real 1D tensor to complex for FFT operation.");
        // Input is real samples, upgrade to complex
        Tensor upgraded = await upgradeRealToComplex();
        return upgraded.fft1d();
      } else {
        // Input is already complex pairs (existing behavior)
        // Only upgrade if odd length (original logic)
        if (shape[0] % 2 != 0) {
          Tensor upgraded = await upgradeRealToComplex();
          return upgraded.fft1d();
        }
        return fft1d();
      }
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

  /// Computes a 1D FFT on a tensor representing complex numbers in interleaved format.
  /// Expects a flat tensor ([N*2]) with an even number of elements.
  /// Bit-reverse reordering for FFT output
  Future<Tensor> bitReverseReorder(Tensor input) async {
    final n = input.shape[0] ~/ 2; // number of complex points
    if (n <= 1) return input;

    final output = await Tensor.create(input.shape, gpu: gpu);

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= ${n}u) { return; }
  
  let numBits: u32 = ${(math.log(n) / math.ln2).round()}u;
  let j: u32 = bitReverse(i, numBits);
  
  let srcIdx: u32 = j * 2u;
  let dstIdx: u32 = i * 2u;
  
  output[dstIdx] = input[srcIdx];
  output[dstIdx + 1u] = input[srcIdx + 1u];
}
''';

    final ComputeShader shader = gpu.createComputeShader();
    final shaderCode = prepareShader(shaderTemplate, dataType, {'n': n});
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input', input.buffer);
    shader.setBuffer('output', output.buffer);

    int workgroups = (n + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return output;
  }

  Future<Tensor> fft1d() async {
    if (shape.length != 1) {
      throw Exception("FFT supports 1D tensors only.");
    }
    int totalFloats = shape[0];
    if (totalFloats % 2 != 0) {
      throw Exception("FFT tensor length must be even (for complex numbers).");
    }
    int n = totalFloats ~/ 2; // number of complex points
    if ((n & (n - 1)) != 0) {
      print("actually n = $n");
      throw Exception("FFT size ($n) must be a power of 2.");
    }
    if (n == 1) return this; // FFT of a single point

    int stages = (math.log(n) / math.ln2).toInt();

    // FIXED: Apply bit-reversal BEFORE FFT stages (input reordering)
    Tensor bitReversedInput = await bitReverseReorder(this);

    Tensor ping = bitReversedInput;
    Tensor pong = await Tensor.create(shape, gpu: gpu);

    for (int s = 0; s < stages; s++) {
      int m = 1 << (s + 1);
      int half = m >> 1;
      int numOperations = n >> 1; // REVERTED: This was correct

      final shaderTemplate =
          '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let t: u32 = gid.x;
  if (t >= ${numOperations}u) { return; }
  let half: u32 = ${half}u;
  let m: u32 = ${m}u;
  let group: u32 = t / half;
  let pos: u32 = t % half;
  let i0: u32 = group * m + pos;
  let i1: u32 = i0 + half;
  let idx0: u32 = i0 * 2u;
  let idx1: u32 = i1 * 2u;
  let a: vec2<f32> = vec2<f32>(input[idx0], input[idx0 + 1u]);
  let b: vec2<f32> = vec2<f32>(input[idx1], input[idx1 + 1u]);
  let angle: f32 = -6.28318530718 * f32(pos) / f32(m);
  let w: vec2<f32> = vec2<f32>(cos(angle), sin(angle));
  let b_twiddled: vec2<f32> = vec2<f32>(
    b.x * w.x - b.y * w.y,
    b.x * w.y + b.y * w.x
  );
  let temp1: vec2<f32> = a + b_twiddled;
  let temp2: vec2<f32> = a - b_twiddled;
  output[idx0] = temp1.x;
  output[idx0 + 1u] = temp1.y;
  output[idx1] = temp2.x;
  output[idx1 + 1u] = temp2.y;
}
''';
      final ComputeShader shader = gpu.createComputeShader();
      final shaderCode = prepareShader(shaderTemplate, dataType, {
        'half': half,
        'm': m,
        'numOperations': numOperations,
      });
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      int workgroups = (numOperations + 255) ~/ 256;
      await shader.dispatch(workgroups, 1, 1);
      shader.destroy();

      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }

    return ping;
  }

  /// Computes a 2D FFT. If a real tensor is supplied, it is upgraded.
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
      return upgraded.fft2d();
    }
    if (shape.length != 3 || shape[2] != 2) {
      throw Exception("fft2d requires a tensor of shape [rows, cols, 2].");
    }
    int rows = shape[0], cols = shape[1];
    bool isPow2(int x) => (x & (x - 1)) == 0;
    if (!isPow2(rows) || !isPow2(cols)) {
      throw Exception("Both rows and cols must be powers of 2.");
    }

    // FFT on rows.
    int stagesRow = (math.log(cols) / math.ln2).toInt();
    Tensor ping = this;
    Tensor pong = await Tensor.create(shape, gpu: gpu);
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
      final ComputeShader shader = gpu.createComputeShader();
      final shaderCode = prepareShader(shaderTemplate, dataType, {
        'half': half,
        'm': m,
        'numOperations': numOperations,
      });
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      int workgroups = (numOperations + 255) ~/ 256;
      await shader.dispatch(workgroups, 1, 1);
      shader.destroy();
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }
    // FFT on columns.
    int stagesCol = (math.log(rows) / math.ln2).toInt();
    pong = await Tensor.create(shape);
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
      final ComputeShader shader = gpu.createComputeShader();
      final shaderCode = prepareShader(shaderTemplate, dataType, {
        'half': half,
        'm': m,
        'numOperations': numOperations,
      });
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      int workgroups = (numOperations + 255) ~/ 256;
      await shader.dispatch(workgroups, 1, 1);
      shader.destroy();
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }
    return ping;
  }

  /// Computes a 3D FFT on a tensor representing complex numbers in interleaved format.
  /// If a real tensor with shape [D, R, C] is supplied, it is upgraded.
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
      return upgraded.fft3d();
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

    Tensor ping = this;
    Tensor pong = await Tensor.create(shape, gpu: gpu);

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
      final ComputeShader shader = gpu.createComputeShader();
      final shaderCode = prepareShader(shaderTemplate, dataType, {
        'D': D,
        'R': R,
        'C': C,
        'm': m,
        'half': half,
        'numOperations': numOperations,
      });
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      int workgroups = (numOperations + 255) ~/ 256;
      await shader.dispatch(workgroups, 1, 1);
      shader.destroy();
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }

    // FFT along row dimension.
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
      final ComputeShader shader = gpu.createComputeShader();
      final shaderCode = prepareShader(shaderTemplate, dataType, {
        'D': D,
        'R': R,
        'C': C,
        'm': m,
        'half': half,
        'numOperations': numOperations,
      });

      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      int workgroups = (numOperations + 255) ~/ 256;
      await shader.dispatch(workgroups, 1, 1);
      shader.destroy();
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }

    // FFT along column dimension.
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
      final ComputeShader shader = gpu.createComputeShader();
      final shaderCode = prepareShader(shaderTemplate, dataType, {
        'D': D,
        'R': R,
        'C': C,
        'm': m,
        'half': half,
        'numOperations': numOperations,
      });
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', ping.buffer);
      shader.setBuffer('output', pong.buffer);
      int workgroups = (numOperations + 255) ~/ 256;
      await shader.dispatch(workgroups, 1, 1);
      shader.destroy();
      Tensor temp = ping;
      ping = pong;
      pong = temp;
    }
    return ping;
  }
}
