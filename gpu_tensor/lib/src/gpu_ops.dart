import 'dart:typed_data';

import 'package:gpu_tensor/src/gpu_helpers.dart';
import 'gpu_tensor_base.dart';

extension TensorOperator<T extends TypedData> on Tensor<T> {
  bool _shapeEquals(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  /// NumPy broadcast shape of [a] and [b], or null if incompatible.
  List<int>? _broadcastShape(List<int> a, List<int> b) {
    final rank = a.length > b.length ? a.length : b.length;
    final out = List<int>.filled(rank, 1);
    for (int i = 0; i < rank; i++) {
      final ad = i < rank - a.length ? 1 : a[i - (rank - a.length)];
      final bd = i < rank - b.length ? 1 : b[i - (rank - b.length)];
      if (ad != bd && ad != 1 && bd != 1) return null;
      out[i] = ad > bd ? ad : bd;
    }
    return out;
  }

  /// Decides how a binary elementwise op should run against [other]:
  /// - null: same-shape (or legacy same-size) flat fast path.
  /// - non-null: broadcast kernel with the returned output shape.
  /// Throws when the shapes are neither broadcastable nor size-equal.
  List<int>? _broadcastPlan(Tensor<T> other, String opName) {
    if (_shapeEquals(shape, other.shape)) return null;
    final out = _broadcastShape(shape, other.shape);
    if (out != null) return out;
    // Legacy behavior: equal SIZE with un-broadcastable shapes ran flat
    // elementwise (e.g. [8] + [2, 4]); preserved for compatibility.
    if (other.size == size) return null;
    throw Exception(
      "Tensor shapes $shape and ${other.shape} are not broadcast-compatible for $opName",
    );
  }

  /// Stride-0 broadcast gather: each input's stride is 0 along dimensions of
  /// extent 1, so a single index computation serves both broadcast and
  /// non-broadcast dims with no divergence.
  Future<Tensor<T>> _broadcastBinary(
    Tensor<T> other,
    String op,
    List<int> outShape,
  ) async {
    final rank = outShape.length;
    List<int> padShape(List<int> s) => [
      ...List.filled(rank - s.length, 1),
      ...s,
    ];
    final aShape = padShape(shape);
    final bShape = padShape(other.shape);

    List<int> inputStrides(List<int> s) {
      final st = List<int>.filled(rank, 0);
      int acc = 1;
      for (int i = rank - 1; i >= 0; i--) {
        st[i] = s[i] == 1 ? 0 : acc;
        acc *= s[i];
      }
      return st;
    }

    final aStrides = inputStrides(aShape);
    final bStrides = inputStrides(bShape);
    final outStrides = List<int>.filled(rank, 1);
    for (int i = rank - 2; i >= 0; i--) {
      outStrides[i] = outStrides[i + 1] * outShape[i + 1];
    }
    final outSize = outShape.reduce((a, b) => a * b);

    final result = await Tensor.create<T>(
      outShape,
      gpu: gpu,
      dataType: dataType,
    );

    String u32Array(List<int> arr) =>
        'array<u32, $rank>(${arr.map((x) => '${x}u').join(', ')})';
    final shaderTemplate =
        '''
const RANK: u32 = ${rank}u;
const outStrides: array<u32, $rank> = ${u32Array(outStrides)};
const aStrides: array<u32, $rank> = ${u32Array(aStrides)};
const bStrides: array<u32, $rank> = ${u32Array(bStrides)};

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= ${outSize}u) { return; }
  var rem: u32 = i;
  var aIdx: u32 = 0u;
  var bIdx: u32 = 0u;
  for (var d: u32 = 0u; d < RANK; d = d + 1u) {
    let coord: u32 = rem / outStrides[d];
    rem = rem % outStrides[d];
    aIdx = aIdx + coord * aStrides[d];
    bIdx = bIdx + coord * bStrides[d];
  }
  C[i] = A[aIdx] $op B[bIdx];
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {});
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    await shader.dispatchLinear(outSize);
    return result;
  }

  /// Elementwise addition with NumPy-style broadcasting.
  Future<Tensor<T>> add(Tensor<T> other) async {
    final broadcast = _broadcastPlan(other, 'elementwise addition');
    if (broadcast != null) {
      return _broadcastBinary(other, '+', broadcast);
    }
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    C[i] = A[i] + B[i];
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Elementwise subtraction with NumPy-style broadcasting.
  Future<Tensor<T>> subtract(Tensor<T> other) async {
    final broadcast = _broadcastPlan(other, 'elementwise subtraction');
    if (broadcast != null) {
      return _broadcastBinary(other, '-', broadcast);
    }
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    C[i] = A[i] - B[i];
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    await shader.dispatchLinear(size);
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

  /// Element-wise maximum of two tensors
  Future<Tensor<T>> max(Tensor<T> other) async {
    if (!shapesCompatible(shape, other.shape)) {
      throw Exception(
        'Tensors must have compatible shapes for element-wise max',
      );
    }

    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    C[i] = max(A[i], B[i]);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Element-wise minimum of two tensors
  Future<Tensor<T>> min(Tensor<T> other) async {
    if (!shapesCompatible(shape, other.shape)) {
      throw Exception(
        'Tensors must have compatible shapes for element-wise min',
      );
    }

    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    C[i] = min(A[i], B[i]);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Average multiple tensors in a single GPU operation
  Future<Tensor<T>> avg(List<Tensor<T>> tensors) async {
    if (tensors.isEmpty) {
      throw Exception('Cannot average empty tensor list');
    }

    if (tensors.length == 1) {
      return tensors.first;
    }

    // Validate all tensors have compatible shapes
    for (final tensor in tensors) {
      if (!shapesCompatible(shape, tensor.shape)) {
        throw Exception(
          'All tensors must have compatible shapes for averaging',
        );
      }
    }

    return _weightedSum(
      tensors,
      List<double>.filled(tensors.length, 1.0 / tensors.length),
    );
  }

  /// Weighted average of multiple tensors
  Future<Tensor<T>> avgWeighted(
    List<Tensor<T>> tensors,
    List<double> weights,
  ) async {
    if (tensors.isEmpty || weights.isEmpty) {
      throw Exception('Cannot average empty tensor or weights list');
    }

    if (tensors.length != weights.length) {
      throw Exception('Number of tensors must match number of weights');
    }

    if (tensors.length == 1) {
      return tensors.first;
    }

    // Normalize weights
    final weightSum = weights.reduce((a, b) => a + b);
    final normalizedWeights = weights.map((w) => w / weightSum).toList();

    // Validate all tensors have compatible shapes
    for (final tensor in tensors) {
      if (!shapesCompatible(shape, tensor.shape)) {
        throw Exception(
          'All tensors must have compatible shapes for weighted averaging',
        );
      }
    }

    return _weightedSum(tensors, normalizedWeights);
  }

  /// output = sum_i tensors[i] * weights[i], chunked so no single dispatch
  /// binds more than 8 storage buffers (WebGPU's default
  /// maxStorageBuffersPerShaderStage) — one unchunked shader with N+1
  /// bindings fails pipeline creation for N > ~7.
  Future<Tensor<T>> _weightedSum(
    List<Tensor<T>> tensors,
    List<double> weights,
  ) async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );

    const int maxInputsPerPass = 7; // + 1 output binding = 8 total

    for (int start = 0; start < tensors.length; start += maxInputsPerPass) {
      final end = (start + maxInputsPerPass < tensors.length)
          ? start + maxInputsPerPass
          : tensors.length;
      final chunkLen = end - start;
      final accumulate = start > 0;

      final inputBindings = StringBuffer();
      final weightedSum = StringBuffer();
      for (int i = 0; i < chunkLen; i++) {
        inputBindings.writeln(
          '@group(0) @binding($i) var<storage, read_write> input_$i: array<f32>;',
        );
        final weight = weights[start + i].toString();
        if (i > 0) weightedSum.write(' + ');
        weightedSum.write('input_$i[i] * $weight');
      }
      final outputBinding =
          '@group(0) @binding($chunkLen) var<storage, read_write> output: array<f32>;';
      final outputExpr = accumulate
          ? 'output[i] = output[i] + ($weightedSum);'
          : 'output[i] = $weightedSum;';

      final shaderTemplate =
          '''
$inputBindings
$outputBinding
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    $outputExpr
  }
}
''';

      final shaderCode = prepareShader(shaderTemplate, dataType, {
        'size': size.toString(),
      });
      final shader = gpu.cachedShader(shaderCode);
      for (int i = 0; i < chunkLen; i++) {
        shader.setBuffer('input_$i', tensors[start + i].buffer);
      }
      shader.setBuffer('output', result.buffer);
      await shader.dispatchLinear(size);
    }

    return result;
  }

  /// Elementwise multiplication (Hadamard product) with NumPy-style
  /// broadcasting.
  Future<Tensor<T>> multiply(Tensor<T> other) async {
    final broadcast = _broadcastPlan(other, 'elementwise multiplication');
    if (broadcast != null) {
      return _broadcastBinary(other, '*', broadcast);
    }
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    C[i] = A[i] * B[i];
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Adds a scalar value to every element in the tensor.
  Future<Tensor<T>> addScalar(double scalar) async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    B[i] = A[i] + ${scalar}f;
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Subtracts a scalar value from every element in the tensor.
  Future<Tensor<T>> subtractScalar(double scalar) async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    B[i] = A[i] - $scalar;
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Multiplies every element in the tensor by a scalar value.
  Future<Tensor<T>> multiplyScalar(double scalar) async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    B[i] = A[i] * $scalar;
  }
}
''';

    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Elementwise division (A / B) with NumPy-style broadcasting.
  Future<Tensor<T>> divide(Tensor<T> other) async {
    final broadcast = _broadcastPlan(other, 'elementwise division');
    if (broadcast != null) {
      return _broadcastBinary(other, '/', broadcast);
    }
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    C[i] = A[i] / B[i];
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Divides every element in the tensor by a scalar.
  Future<Tensor<T>> divideScalar(double scalar) async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    B[i] = A[i] / $scalar;
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Raises every element in the tensor to the power of [exponent].
  Future<Tensor<T>> powScalar(double exponent) async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    B[i] = pow(A[i], $exponent);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Computes the natural logarithm (ln) of each element.
  Future<Tensor<T>> log() async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    B[i] = log(A[i]);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Computes the exponential (e^x) of each element.
  Future<Tensor<T>> exp() async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    B[i] = exp(A[i]);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Computes the square root of each element.
  Future<Tensor<T>> sqrt() async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    B[i] = sqrt(A[i]);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Computes the modulus (remainder) of each element by [divisor].
  Future<Tensor<T>> modScalar(double divisor) async {
    // Exact literal — toStringAsFixed(1) truncated the divisor (0.25 -> 0.3).
    String divisorLiteral = divisor.toString().contains('.')
        ? divisor.toString()
        : '$divisor.0';
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    B[i] = A[i] % $divisorLiteral;
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Elementwise modulus for two tensors.
  Future<Tensor<T>> mod(Tensor<T> other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise modulus");
    }
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    C[i] = A[i] % B[i];
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Performs elementwise "greater than" comparison.
  Future<Tensor<T>> greaterThan(Tensor<T> other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise comparison");
    }
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    C[i] = select(0.0, 1.0, A[i] > B[i]);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Performs elementwise "less than" comparison.
  Future<Tensor<T>> lessThan(Tensor<T> other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise comparison");
    }
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    C[i] = select(0.0, 1.0, A[i] < B[i]);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Performs elementwise equality comparison.
  Future<Tensor<T>> equalTo(Tensor<T> other) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise comparison");
    }
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    C[i] = select(0.0, 1.0, A[i] == B[i]);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Shared kernel for the fused elementwise comparisons below.  Previously
  /// notEqualTo/greaterThanOrEqual/lessThanOrEqual were composed from THREE
  /// dispatches (compare, ones, subtract) and leaked the intermediate zeros
  /// tensor.  One dispatch, no intermediates.
  Future<Tensor<T>> _compare(Tensor<T> other, String wgslCondition) async {
    if (other.size != size) {
      throw Exception("Tensor sizes do not match for elementwise comparison");
    }
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    C[i] = select(0.0, 1.0, $wgslCondition);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', other.buffer);
    shader.setBuffer('C', result.buffer);
    await shader.dispatchLinear(size);
    return result;
  }

  /// Performs elementwise "not equal" comparison.
  Future<Tensor<T>> notEqualTo(Tensor<T> other) => _compare(other, 'A[i] != B[i]');

  /// Greater than or equal (A >= B).
  Future<Tensor<T>> greaterThanOrEqual(Tensor<T> other) =>
      _compare(other, 'A[i] >= B[i]');

  /// Less than or equal (A <= B).
  Future<Tensor<T>> lessThanOrEqual(Tensor<T> other) =>
      _compare(other, 'A[i] <= B[i]');

  /// Computes the absolute value of each element.
  Future<Tensor<T>> abs() async {
    Tensor<T> result = await Tensor.create<T>(
      shape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < ${size}u) {
    B[i] = abs(A[i]);
  }
}
''';
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);

    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(size);
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
    Tensor<T> result = await Tensor.create<T>(
      outShape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

const d: u32 = ${d}u;
const inner: u32 = ${inner}u;
const totalOut: u32 = ${totalOut}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let idx: u32 = gid.x + gid.y * (nwg.x * 256u);
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
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'd': d.toString(),
      'inner': inner.toString(),
      'totalOut': totalOut.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);

    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(totalOut);
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
    Tensor<T> result = await Tensor.create<T>(
      outShape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

const d: u32 = ${d}u;
const inner: u32 = ${inner}u;
const totalOut: u32 = ${totalOut}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let idx: u32 = gid.x + gid.y * (nwg.x * 256u);
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
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'd': d.toString(),
      'inner': inner.toString(),
      'totalOut': totalOut.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(totalOut);
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
    Tensor<T> result = await Tensor.create<T>(
      outShape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

const d: u32 = ${d}u;
const inner: u32 = ${inner}u;
const totalOut: u32 = ${totalOut}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let idx: u32 = gid.x + gid.y * (nwg.x * 256u);
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
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'd': d.toString(),
      'inner': inner.toString(),
      'totalOut': totalOut.toString(),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(totalOut);
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
    Tensor<T> result = await Tensor.create<T>(
      outShape,
      gpu: gpu,
      dataType: dataType,
    );
    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;

const d: u32 = ${d}u;
const inner: u32 = ${inner}u;
const totalOut: u32 = ${totalOut}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let idx: u32 = gid.x + gid.y * (nwg.x * 256u);
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
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'shape': shape,
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('A', buffer);
    shader.setBuffer('B', result.buffer);
    await shader.dispatchLinear(totalOut);
    return result;
  }
}
