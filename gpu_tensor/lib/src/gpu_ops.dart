import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';
import 'gpu_tensor_base.dart';

extension TensorOperator<T extends TypedData> on Tensor<T> {
  /// Elementwise addition. Assumes both tensors have the same shape.
  Future<Tensor<T>> add(Tensor<T> other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise addition");
    }
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    C[i] = A[i] + B[i];
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Elementwise subtraction.
  Future<Tensor<T>> subtract(Tensor<T> other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise subtraction");
    }
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    C[i] = A[i] - B[i];
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Operator overloads for more natural syntax.
  Future<Tensor<T>> operator +(dynamic other) async {
    if (other is num) {
      return addScalar(other.toDouble());
    } else if (other is Tensor<T>) {
      return add(other);
    } else {
      throw Exception("Unsupported operand type for +");
    }
  }

  Future<Tensor<T>> operator -(dynamic other) async {
    if (other is num) {
      return subtractScalar(other.toDouble());
    } else if (other is Tensor<T>) {
      return subtract(other);
    } else {
      throw Exception("Unsupported operand type for -");
    }
  }

  Future<Tensor<T>> operator *(dynamic other) async {
    if (other is num) {
      return multiplyScalar(other.toDouble());
    } else if (other is Tensor<T>) {
      return multiply(other);
    } else {
      throw Exception("Unsupported operand type for *");
    }
  }

  Future<Tensor<T>> operator /(dynamic other) async {
    if (other is num) {
      return divideScalar(other.toDouble());
    } else if (other is Tensor<T>) {
      return divide(other);
    } else {
      throw Exception("Unsupported operand type for /");
    }
  }

  Future<Tensor<T>> operator %(dynamic other) async {
    if (other is num) {
      return modScalar(other.toDouble());
    } else if (other is Tensor<T>) {
      return mod(other);
    } else {
      throw Exception("Unsupported operand type for %");
    }
  }

  /// Elementwise multiplication (Hadamard product).
  Future<Tensor<T>> multiply(Tensor<T> other) async {
    if (other.size != size) {
      throw Exception(
          "Tensor sizes do not match for elementwise multiplication");
    }
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    C[i] = A[i] * B[i];
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Adds a scalar value to every element in the tensor.
  Future<Tensor<T>> addScalar(double scalar) async {
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = A[i] + $scalar;
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Subtracts a scalar value from every element in the tensor.
  Future<Tensor<T>> subtractScalar(double scalar) async {
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = A[i] - $scalar;
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Multiplies every element in the tensor by a scalar value.
  Future<Tensor<T>> multiplyScalar(double scalar) async {
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = A[i] * $scalar;
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Elementwise division (A / B).
  Future<Tensor<T>> divide(Tensor<T> other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise division");
    }
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    C[i] = A[i] / B[i];
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Divides every element in the tensor by a scalar.
  Future<Tensor<T>> divideScalar(double scalar) async {
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = A[i] / $scalar;
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Raises every element in the tensor to the power of [exponent].
  Future<Tensor<T>> powScalar(double exponent) async {
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = pow(A[i], $exponent);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Computes the natural logarithm (ln) of each element.
  Future<Tensor<T>> log() async {
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = log(A[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Computes the exponential (e^x) of each element.
  Future<Tensor<T>> exp() async {
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = exp(A[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Computes the square root of each element.
  Future<Tensor<T>> sqrt() async {
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = sqrt(A[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Computes the modulus (remainder) of each element by [divisor].
  Future<Tensor<T>> modScalar(double divisor) async {
    String divisorLiteral = divisor.toStringAsFixed(1);
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = A[i] % $divisorLiteral;
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Elementwise modulus for two tensors.
  Future<Tensor<T>> mod(Tensor<T> other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise modulus");
    }
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    C[i] = A[i] % B[i];
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Performs elementwise "greater than" comparison.
  Future<Tensor<T>> greaterThan(Tensor<T> other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise comparison");
    }
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    C[i] = select(0.0, 1.0, A[i] > B[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Performs elementwise "less than" comparison.
  Future<Tensor<T>> lessThan(Tensor<T> other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise comparison");
    }
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    C[i] = select(0.0, 1.0, A[i] < B[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Performs elementwise equality comparison.
  Future<Tensor<T>> equalTo(Tensor<T> other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise comparison");
    }
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    C[i] = select(0.0, 1.0, A[i] == B[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Performs elementwise "not equal" comparison.
  Future<Tensor<T>> notEqualTo(Tensor<T> other) async {
    Tensor<T> eq = await equalTo(other);
    Tensor<T> ones =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    ones = await ones.addScalar(1.0);
    Tensor<T> result = await ones.subtract(eq);
    eq.destroy();
    ones.destroy();
    return result;
  }

  /// Greater than or equal (A >= B) is equivalent to NOT(A < B).
  Future<Tensor<T>> greaterThanOrEqual(Tensor<T> other) async {
    Tensor<T> lt = await lessThan(other);
    Tensor<T> ones =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    ones = await ones.addScalar(1.0);
    Tensor<T> result = await ones.subtract(lt);
    lt.destroy();
    ones.destroy();
    return result;
  }

  /// Less than or equal (A <= B) is equivalent to NOT(A > B).
  Future<Tensor<T>> lessThanOrEqual(Tensor<T> other) async {
    Tensor<T> gt = await greaterThan(other);
    Tensor<T> ones =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    ones = await ones.addScalar(1.0);
    Tensor<T> result = await ones.subtract(gt);
    gt.destroy();
    ones.destroy();
    return result;
  }

  /// Computes the absolute value of each element.
  Future<Tensor<T>> abs() async {
    Tensor<T> result =
        await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    B[i] = abs(A[i]);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Reduces the tensor by summing values along the last dimension.
  Future<Tensor<T>> sum({int axis = -1}) async {
    int n = shape.length;
    if (axis < 0) axis += n;
    if (axis < 0 || axis >= n) {
      throw Exception("Axis out of range.");
    }
    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= shape[i];
    int d = shape[axis];
    int inner = 1;
    for (int i = axis + 1; i < n; i++) inner *= shape[i];
    int totalOut = outer * inner;
    List<int> outShape = List.from(shape)..removeAt(axis);
    Tensor<T> result =
        await Tensor.create<T>(outShape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

const d: u32 = ${d}u;
const inner: u32 = ${inner}u;
const totalOut: u32 = ${totalOut}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx < totalOut) {
    let outer: u32 = idx / inner;
    let r: u32 = idx % inner;
    let base: u32 = outer * (d * inner) + r;
    var sum: f32 = 0.0;
    for (var a: u32 = 0u; a < d; a = a + 1u) {
      sum = sum + A[base + a * inner];
    }
    B[idx] = sum;
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (totalOut + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Computes the mean along the given dimension.
  Future<Tensor<T>> mean({int axis = -1}) async {
    Tensor<T> sums = await sum(axis: axis);
    int d = shape[(axis < 0 ? axis + shape.length : axis)];
    Tensor<T> result = await sums.multiplyScalar(1.0 / d);
    sums.destroy();
    return result;
  }

  /// Reduces the tensor by taking the maximum value along the last dimension.
  Future<Tensor<T>> maxReduction({int axis = -1}) async {
    int n = shape.length;
    if (axis < 0) axis += n;
    if (axis < 0 || axis >= n) {
      throw Exception("Axis out of range.");
    }
    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= shape[i];
    int d = shape[axis];
    int inner = 1;
    for (int i = axis + 1; i < n; i++) inner *= shape[i];
    int totalOut = outer * inner;
    List<int> outShape = List.from(shape)..removeAt(axis);
    Tensor<T> result =
        await Tensor.create<T>(outShape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

const d: u32 = ${d}u;
const inner: u32 = ${inner}u;
const totalOut: u32 = ${totalOut}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx < totalOut) {
    let outer: u32 = idx / inner;
    let r: u32 = idx % inner;
    let base: u32 = outer * (d * inner) + r;
    var max_val: f32 = A[base];
    for (var j: u32 = 1u; j < d; j = j + 1u) {
      max_val = max(max_val, A[base + j * inner]);
    }
    B[idx] = max_val;
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (totalOut + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Reduces the tensor by taking the minimum value along the last dimension.
  Future<Tensor<T>> minReduction({int axis = -1}) async {
    int n = shape.length;
    if (axis < 0) axis += n;
    if (axis < 0 || axis >= n) {
      throw Exception("Axis out of range.");
    }
    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= shape[i];
    int d = shape[axis];
    int inner = 1;
    for (int i = axis + 1; i < n; i++) inner *= shape[i];
    int totalOut = outer * inner;
    List<int> outShape = List.from(shape)..removeAt(axis);
    Tensor<T> result =
        await Tensor.create<T>(outShape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

const d: u32 = ${d}u;
const inner: u32 = ${inner}u;
const totalOut: u32 = ${totalOut}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx < totalOut) {
    let outer: u32 = idx / inner;
    let r: u32 = idx % inner;
    let base: u32 = outer * (d * inner) + r;
    var min_val: f32 = A[base];
    for (var j: u32 = 1u; j < d; j = j + 1u) {
      min_val = min(min_val, A[base + j * inner]);
    }
    B[idx] = min_val;
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (totalOut + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }

  /// Argmax: Finds the index of the maximum element along the last dimension.
  Future<Tensor<T>> argmax({int axis = -1}) async {
    int n = shape.length;
    if (axis < 0) axis += n;
    if (axis < 0 || axis >= n) {
      throw Exception("Axis out of range.");
    }
    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= shape[i];
    int d = shape[axis];
    int inner = 1;
    for (int i = axis + 1; i < n; i++) inner *= shape[i];
    int totalOut = outer * inner;
    List<int> outShape = List.from(shape)..removeAt(axis);
    Tensor<T> result =
        await Tensor.create<T>(outShape, gpu: gpu, dataType: dataType);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

const d: u32 = ${d}u;
const inner: u32 = ${inner}u;
const totalOut: u32 = ${totalOut}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx < totalOut) {
    let outer: u32 = idx / inner;
    let r: u32 = idx % inner;
    let base: u32 = outer * (d * inner) + r;
    var max_val: f32 = A[base];
    var max_index: u32 = 0u;
    for (var j: u32 = 1u; j < d; j = j + 1u) {
      let val = A[base + j * inner];
      if (val > max_val) {
         max_val = val;
         max_index = j;
      }
    }
    B[idx] = f32(max_index);
  }
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    int workgroups = (totalOut + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    return result;
  }
}
