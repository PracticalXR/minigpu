import 'dart:typed_data';

import 'package:gpu_tensor/src/gpu_helpers.dart';
import 'package:minigpu/minigpu.dart';
import '../gpu_tensor.dart';

extension GpuActivation<T extends TypedData> on Tensor<T> {
  /// Applies the ReLU activation function elementwise.
  Future<Tensor<T>> relu() async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final wgslType = getWGSLType(result.dataType);
    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> input: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> output: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    let v: $wgslType = input[i];
    output[i] = select(v, ${wgslType == 'f32' ? '0.0' : '0'}, v < ${wgslType == 'f32' ? '0.0' : '0'});
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'wgslType': wgslType,
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Applies the Sigmoid activation function elementwise.
  Future<Tensor<T>> sigmoid() async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final wgslType = getWGSLType(result.dataType);
    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> input: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> output: array<$wgslType>;

fn sigmoid(x: $wgslType) -> $wgslType {
  return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    output[i] = sigmoid(input[i]);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'wgslType': wgslType,
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Computes the sine of each element.
  Future<Tensor<T>> sin() async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final wgslType = getWGSLType(result.dataType);
    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> A: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> B: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if(i < ${size}u) {
    B[i] = sin(A[i]);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'wgslType': wgslType,
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Computes the cosine of each element.
  Future<Tensor<T>> cos() async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final wgslType = getWGSLType(result.dataType);
    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> A: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> B: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if(i < ${size}u) {
    B[i] = cos(A[i]);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'wgslType': wgslType,
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Applies the Tanh activation function elementwise.
  Future<Tensor<T>> tanh() async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final wgslType = getWGSLType(result.dataType);
    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> input: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> output: array<$wgslType>;

fn tanh_func(x: $wgslType) -> $wgslType {
  let expPos = exp(x);
  let expNeg = exp(-x);
  return (expPos - expNeg) / (expPos + expNeg);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    output[i] = tanh_func(input[i]);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'wgslType': wgslType,
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Applies the Softmax activation function along the last dimension.
  /// For higher dimensional tensors, softmax is applied to each slice
  /// of size (last dimension) independently.
  Future<Tensor<T>> softmax({int axis = -1}) async {
    // For now, we only support softmax along the last dimension.
    int d = shape.last; // inner dimension size
    int total = size;
    if (axis != -1 && axis != shape.length - 1) {
      throw Exception("Only softmax along the last dimension is supported.");
    }
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );

    // Large rows: one WORKGROUP per row with shared-memory tree reductions.
    // The per-element kernel below re-scans the whole row from every thread
    // (O(d^2) work per row) — unusable at attention/logit sizes.
    if (d >= 64) {
      return _softmaxWorkgroupPerRow(result, d, total ~/ d);
    }
    final wgslType = getWGSLType(result.dataType);
    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> input: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> output: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let global: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (global < ${total}u) {
    let d: u32 = ${d}u;
    let batchIndex: u32 = global / d;
    let offset: u32 = batchIndex * d;
    
    // Compute maximum value within this softmax group.
    var max_val: $wgslType = input[offset];
    for (var j: u32 = 1u; j < d; j = j + 1u) {
      max_val = max(max_val, input[offset + j]);
    }
    
    let shifted: $wgslType = input[global] - max_val;
    let exp_val: $wgslType = exp(shifted);
    var sum_exp: $wgslType = ${wgslType == 'f32' ? '0.0' : '0'};
    for (var j: u32 = 0u; j < d; j = j + 1u) {
      sum_exp = sum_exp + exp(input[offset + j] - max_val);
    }
    output[global] = exp_val / sum_exp;
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'wgslType': wgslType,
      'total': total.toString(),
      'd': d.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    await shader.dispatchLinear(total);
    return result;
  }

  /// Softmax with one 256-thread workgroup per row: strided local max/sum,
  /// shared-memory tree reduction, then a strided normalize pass.  O(d) work
  /// per row vs the small-row kernel's O(d^2).
  Future<Tensor<T>> _softmaxWorkgroupPerRow(
    Tensor<T> result,
    int d,
    int rows,
  ) async {
    final wgslType = getWGSLType(result.dataType);
    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> input: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> output: array<$wgslType>;

const D: u32 = ${d}u;
const ROWS: u32 = ${rows}u;
const WG: u32 = 256u;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  // One workgroup per row; rows fold over (x, y) to clear the 65535 limit.
  let row: u32 = wid.x + wid.y * nwg.x;
  // No early return: barriers below must be reached uniformly.
  let inRange: bool = row < ROWS;
  let base: u32 = select(0u, row * D, inRange);

  // Pass 1: row max.
  var m: f32 = -3.4e38;
  if (inRange) {
    for (var j: u32 = lid.x; j < D; j = j + WG) {
      m = max(m, input[base + j]);
    }
  }
  scratch[lid.x] = m;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + s]);
    }
    workgroupBarrier();
  }
  let rowMax: f32 = scratch[0];
  workgroupBarrier();

  // Pass 2: sum of exp(x - max).
  var sum: f32 = 0.0;
  if (inRange) {
    for (var j: u32 = lid.x; j < D; j = j + WG) {
      sum = sum + exp(input[base + j] - rowMax);
    }
  }
  scratch[lid.x] = sum;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s];
    }
    workgroupBarrier();
  }
  let rowSum: f32 = scratch[0];

  // Pass 3: normalize.
  if (inRange) {
    for (var j: u32 = lid.x; j < D; j = j + WG) {
      output[base + j] = exp(input[base + j] - rowMax) / rowSum;
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
}
