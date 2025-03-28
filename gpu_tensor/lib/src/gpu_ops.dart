import 'package:minigpu/minigpu.dart';

import 'gpu_tensor_base.dart';

extension TensorOperator on Tensor {
  /// Elementwise addition. Returns a new tensor with the result.
  /// (Assumes both tensors have the same shape.)
  Future<Tensor> add(Tensor other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise addition");
    }
    Tensor result = await Tensor.create(shape);
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
  Future<Tensor> subtract(Tensor other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise subtraction");
    }
    Tensor result = await Tensor.create(shape);
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
  Future<Tensor> operator +(dynamic other) async {
    if (other is num) {
      return addScalar(other.toDouble());
    } else if (other is Tensor) {
      return add(other);
    } else {
      throw Exception("Unsupported operand type for +");
    }
  }

  Future<Tensor> operator -(dynamic other) async {
    if (other is num) {
      return subtractScalar(other.toDouble());
    } else if (other is Tensor) {
      return subtract(other);
    } else {
      throw Exception("Unsupported operand type for -");
    }
  }

  Future<Tensor> operator *(dynamic other) async {
    if (other is num) {
      return multiplyScalar(other.toDouble());
    } else if (other is Tensor) {
      return multiply(other);
    } else {
      throw Exception("Unsupported operand type for *");
    }
  }

  Future<Tensor> operator %(dynamic other) async {
    if (other is num) {
      return modScalar(other.toDouble());
    } else if (other is Tensor) {
      return mod(other);
    } else {
      throw Exception("Unsupported operand type for %");
    }
  }

  Future<Tensor> operator /(dynamic other) async {
    if (other is num) {
      return divideScalar(other.toDouble());
    } else if (other is Tensor) {
      return divide(other);
    } else {
      throw Exception("Unsupported operand type for /");
    }
  }

  /// Elementwise multiplication (Hadamard product).
  Future<Tensor> multiply(Tensor other) async {
    if (other.size != size) {
      throw Exception(
          "Tensor sizes do not match for elementwise multiplication");
    }
    Tensor result = await Tensor.create(shape);
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
  Future<Tensor> addScalar(double scalar) async {
    Tensor result = await Tensor.create(shape);
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
  Future<Tensor> subtractScalar(double scalar) async {
    Tensor result = await Tensor.create(shape);
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
  Future<Tensor> multiplyScalar(double scalar) async {
    Tensor result = await Tensor.create(shape);
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
  Future<Tensor> divide(Tensor other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise division");
    }
    Tensor result = await Tensor.create(shape);
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
  Future<Tensor> divideScalar(double scalar) async {
    Tensor result = await Tensor.create(shape);
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
  Future<Tensor> powScalar(double exponent) async {
    Tensor result = await Tensor.create(shape);
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
  Future<Tensor> log() async {
    Tensor result = await Tensor.create(shape);
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
  Future<Tensor> exp() async {
    Tensor result = await Tensor.create(shape);
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
  Future<Tensor> sqrt() async {
    Tensor result = await Tensor.create(shape);
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
  Future<Tensor> modScalar(double divisor) async {
    // Ensure the divisor is expressed as a float literal (e.g. "3.0")
    String divisorLiteral = divisor.toStringAsFixed(1);
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
  Future<Tensor> mod(Tensor other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise modulus");
    }
    Tensor result = await Tensor.create(shape);
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

  /// Performs an elementwise "greater than" comparison.
  /// Returns a tensor where each element is 1.0 if A[i] > other[i], otherwise 0.0.
  Future<Tensor> greaterThan(Tensor other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise comparison");
    }
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    if (A[i] > B[i]) {
      C[i] = 1.0;
    } else {
      C[i] = 0.0;
    }
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

  /// Performs an elementwise "less than" comparison.
  /// Returns a tensor where each element is 1.0 if A[i] < other[i], otherwise 0.0.
  Future<Tensor> lessThan(Tensor other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise comparison");
    }
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    if (A[i] < B[i]) {
      C[i] = 1.0;
    } else {
      C[i] = 0.0;
    }
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

  /// Performs an elementwise equality comparison.
  /// Returns a tensor where each element is 1.0 if A[i] equals other[i], otherwise 0.0.
  Future<Tensor> equalTo(Tensor other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise comparison");
    }
    Tensor result = await Tensor.create(shape);
    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < ${size}u) {
    if (A[i] == B[i]) {
      C[i] = 1.0;
    } else {
      C[i] = 0.0;
    }
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

  /// Performs an elementwise "not equal" comparison.
  /// Returns a tensor where each element is 1.0 if A[i] != other[i], otherwise 0.0.
  Future<Tensor> notEqualTo(Tensor other) async {
    // Implement by reusing equalTo and then subtracting from a tensor of ones.
    Tensor eq = await equalTo(other);
    Tensor ones = await Tensor.create(shape);
    // Fill 'ones' with 1.0.
    ones = await ones.addScalar(1.0);
    // ones - eq gives 1.0 for not equal cases.
    Tensor result = await ones.subtract(eq);
    eq.destroy();
    ones.destroy();
    return result;
  }

  /// Performs an elementwise "greater than or equal to" comparison.
  /// Returns a tensor with 1.0 if A[i] >= other[i], otherwise 0.0.
  Future<Tensor> greaterThanOrEqual(Tensor other) async {
    // A >= B is equivalent to NOT (A < B)
    Tensor lt = await lessThan(other);
    Tensor ones = await Tensor.create(shape);
    ones = await ones.addScalar(1.0);
    Tensor result = await ones.subtract(lt);
    lt.destroy();
    ones.destroy();
    return result;
  }

  /// Performs an elementwise "less than or equal to" comparison.
  /// Returns a tensor with 1.0 if A[i] <= other[i], otherwise 0.0.
  Future<Tensor> lessThanOrEqual(Tensor other) async {
    // A <= B is equivalent to NOT (A > B)
    Tensor gt = await greaterThan(other);
    Tensor ones = await Tensor.create(shape);
    ones = await ones.addScalar(1.0);
    Tensor result = await ones.subtract(gt);
    gt.destroy();
    ones.destroy();
    return result;
  }

  /// Computes the absolute value of each element.
  Future<Tensor> abs() async {
    Tensor result = await Tensor.create(shape);
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
    int wk = (size + 255) ~/ 256;
    await shader.dispatch(wk, 1, 1);
    shader.destroy();
    return result;
  }

  /// Reduces the tensor by summing values along the last dimension.
  /// For a tensor of shape [..., d], returns a tensor of shape [...].
  Future<Tensor> sum({int axis = -1}) async {
    int n = shape.length;
    // Normalize negative axis.
    if (axis < 0) {
      axis += n;
    }
    if (axis < 0 || axis >= n) {
      throw Exception("Axis out of range.");
    }

    // Compute product of dimensions before the axis (outer) and after (inner).
    int outer = 1;
    for (int i = 0; i < axis; i++) {
      outer *= shape[i];
    }
    int d = shape[axis]; // Dimension to reduce.
    int inner = 1;
    for (int i = axis + 1; i < n; i++) {
      inner *= shape[i];
    }
    int totalOut = outer * inner;

    // Build output shape by removing the reduced axis.
    List<int> outShape = List.from(shape)..removeAt(axis);
    Tensor result = await Tensor.create(outShape);

    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

const d: u32 = ${d}u;
const inner: u32 = ${inner}u;
const totalOut: u32 = ${totalOut}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx < totalOut) {
    let outer: u32 = idx / inner;
    let r: u32 = idx % inner;
    // Calculate the base index for the reduction segment.
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

  /// Reduces the tensor by computing the mean along the given dimension.
  Future<Tensor> mean({int axis = -1}) async {
    int n = shape.length;
    // Normalize negative axis.
    if (axis < 0) {
      axis += n;
    }
    if (axis < 0 || axis >= n) {
      throw Exception("Axis out of range.");
    }
    // Compute sum along the specified axis.
    Tensor sums = await sum(axis: axis);
    // Use the original dimension for division.
    int d = shape[axis];
    Tensor result = await sums.multiplyScalar(1.0 / d);
    sums.destroy();
    return result;
  }

  /// Reduces the tensor by taking the maximum value along the last dimension.
  Future<Tensor> maxReduction({int axis = -1}) async {
    int n = shape.length;
    // Normalize negative axis.
    if (axis < 0) {
      axis += n;
    }
    if (axis < 0 || axis >= n) {
      throw Exception("Axis out of range.");
    }

    // Compute product of dimensions before the axis (outer) and after (inner).
    int outer = 1;
    for (int i = 0; i < axis; i++) {
      outer *= shape[i];
    }
    int d = shape[axis]; // Dimension to reduce.
    int inner = 1;
    for (int i = axis + 1; i < n; i++) {
      inner *= shape[i];
    }
    int totalOut = outer * inner;

    // Build output shape by removing the reduced axis.
    List<int> outShape = List.from(shape)..removeAt(axis);
    Tensor result = await Tensor.create(outShape);

    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

const d: u32 = ${d}u;
const inner: u32 = ${inner}u;
const totalOut: u32 = ${totalOut}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx < totalOut) {
    let outer: u32 = idx / inner;
    let r: u32 = idx % inner;
    // Calculate the base index for this reduction slice.
    let base: u32 = outer * (d * inner) + r;
    var max_val: f32 = A[base];
    // Iterate over the reduced dimension.
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
  Future<Tensor> minReduction({int axis = -1}) async {
    int n = shape.length;
    // Normalize negative axis.
    if (axis < 0) {
      axis += n;
    }
    if (axis < 0 || axis >= n) {
      throw Exception("Axis out of range.");
    }

    // Compute product of dimensions before the axis (outer) and after (inner).
    int outer = 1;
    for (int i = 0; i < axis; i++) {
      outer *= shape[i];
    }
    int d = shape[axis]; // Dimension to reduce.
    int inner = 1;
    for (int i = axis + 1; i < n; i++) {
      inner *= shape[i];
    }
    int totalOut = outer * inner;

    // Build output shape by removing the reduced axis.
    List<int> outShape = List.from(shape)..removeAt(axis);
    Tensor result = await Tensor.create(outShape);

    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

const d: u32 = ${d}u;
const inner: u32 = ${inner}u;
const totalOut: u32 = ${totalOut}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx < totalOut) {
    let outer: u32 = idx / inner;
    let r: u32 = idx % inner;
    // Calculate base index for the reduction slice.
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
  /// The resulting tensor has shape equal to the input shape minus the last dimension
  /// and always stores indices as f32 values.
  Future<Tensor> argmax({int axis = -1}) async {
    int n = shape.length;
    // Normalize negative axis.
    if (axis < 0) {
      axis += n;
    }
    if (axis < 0 || axis >= n) {
      throw Exception("Axis out of range.");
    }

    // Compute product of dimensions before the axis (outer) and after (inner).
    int outer = 1;
    for (int i = 0; i < axis; i++) {
      outer *= shape[i];
    }
    int d = shape[axis]; // Dimension to reduce.
    int inner = 1;
    for (int i = axis + 1; i < n; i++) {
      inner *= shape[i];
    }
    int totalOut = outer * inner;

    // Build output shape by removing the reduced axis.
    List<int> outShape = List.from(shape)..removeAt(axis);
    Tensor result = await Tensor.create(outShape);

    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

const d: u32 = ${d}u;
const inner: u32 = ${inner}u;
const totalOut: u32 = ${totalOut}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx < totalOut) {
    let outer: u32 = idx / inner;
    let r: u32 = idx % inner;
    // Calculate the base index for this reduction slice.
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
