import 'dart:typed_data';

import 'package:gpu_tensor/src/gpu_helpers.dart';
import 'package:minigpu/minigpu.dart';
import '../gpu_tensor.dart';

extension GpuActivation<T extends TypedData> on Tensor<T> {
  /// Applies the ReLU activation function elementwise.
  Future<Tensor<T>> relu() async {
    Tensor<T> result = await Tensor.create<T>(shape);
    final wgslType = getWGSLType(result.dataType);
    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> input: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> output: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    let v: $wgslType = input[i];
    output[i] = select(v, ${wgslType == 'f32' ? '0.0' : '0'}, v < ${wgslType == 'f32' ? '0.0' : '0'});
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'wgslType': wgslType,
      'size': size.toString(),
    });
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Applies the Sigmoid activation function elementwise.
  Future<Tensor<T>> sigmoid() async {
    Tensor<T> result = await Tensor.create<T>(shape);
    final wgslType = getWGSLType(result.dataType);
    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> input: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> output: array<$wgslType>;

fn sigmoid(x: $wgslType) -> $wgslType {
  return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    output[i] = sigmoid(input[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'wgslType': wgslType,
      'size': size.toString(),
    });
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Computes the sine of each element.
  Future<Tensor<T>> sin() async {
    Tensor<T> result = await Tensor.create<T>(shape);
    final wgslType = getWGSLType(result.dataType);
    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> A: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> B: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if(i < ${size}u) {
    B[i] = sin(A[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'wgslType': wgslType,
      'size': size.toString(),
    });
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int wk = (size + 255) ~/ 256;
    await shader.dispatch(wk, 1, 1);
    shader.destroy();
    return result;
  }

  /// Computes the cosine of each element.
  Future<Tensor<T>> cos() async {
    Tensor<T> result = await Tensor.create<T>(shape);
    final wgslType = getWGSLType(result.dataType);
    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> A: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> B: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if(i < ${size}u) {
    B[i] = cos(A[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'wgslType': wgslType,
      'size': size.toString(),
    });
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int wk = (size + 255) ~/ 256;
    await shader.dispatch(wk, 1, 1);
    shader.destroy();
    return result;
  }

  /// Applies the Tanh activation function elementwise.
  Future<Tensor<T>> tanh() async {
    Tensor<T> result = await Tensor.create<T>(shape);
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
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    output[i] = tanh_func(input[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'wgslType': wgslType,
      'size': size.toString(),
    });
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
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
    Tensor<T> result = await Tensor.create<T>(shape);
    final wgslType = getWGSLType(result.dataType);
    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> input: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> output: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let global: u32 = gid.x;
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
    final ComputeShader shader = gpu.createComputeShader();
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'wgslType': wgslType,
      'total': total.toString(),
      'd': d.toString(),
    });
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    int workgroups = (total + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }
}
