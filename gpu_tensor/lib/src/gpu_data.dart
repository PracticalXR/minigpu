import 'dart:typed_data';
import 'dart:math' as math;
import 'package:gpu_tensor/src/gpu_helpers.dart';
import 'package:minigpu/minigpu.dart';

import 'gpu_tensor_base.dart';

extension TensorData<T extends TypedData> on Tensor<T> {
  /// Creates a new tensor by slicing the flattened tensor data.
  /// [start] is the starting flat index and [end] is the ending flat index (exclusive).
  /// GPU-side copy — no readback.
  Future<Tensor<T>> sliceLinear({required int start, required int end}) async {
    if (start < 0 || end > size || start >= end) {
      throw Exception(
        "Invalid slice indices: start=$start, end=$end, size=$size.",
      );
    }
    int newSize = end - start;
    final result = await Tensor.create<T>(
      [newSize],
      gpu: gpu,
      dataType: dataType,
    );
    final wgslType = getWGSLType(dataType);
    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> input: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> output: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= ${newSize}u) { return; }
  output[i] = input[i + ${start}u];
}
''';
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    await shader.dispatchLinear(newSize);
    return result;
  }

  /// Slices the tensor based on multi-dimensional indices.
  Future<Tensor<T>> slice({
    required List<int> startIndices,
    required List<int> endIndices,
  }) async {
    if (startIndices.length != shape.length ||
        endIndices.length != shape.length) {
      throw Exception(
        "startIndices and endIndices must match tensor rank (${shape.length}).",
      );
    }

    // Calculate strides (row-major order).
    List<int> strides = List.filled(shape.length, 1);
    for (int i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    for (int i = 0; i < shape.length; i++) {
      if (startIndices[i] < 0 ||
          endIndices[i] > shape[i] ||
          startIndices[i] >= endIndices[i]) {
        throw Exception(
          "Invalid slice indices for dimension $i: start=${startIndices[i]}, end=${endIndices[i]}, shape=${shape[i]}.",
        );
      }
    }

    // Compute new shape and total elements.
    List<int> newShape = [];
    int numElems = 1;
    for (int i = 0; i < shape.length; i++) {
      int dimSize = endIndices[i] - startIndices[i];
      newShape.add(dimSize);
      numElems *= dimSize;
    }

    // GPU-side strided gather — previously this read the ENTIRE tensor back
    // to the CPU and re-uploaded the slice.  Note the old flat-sublist logic
    // was also wrong for inner-dimension slices (non-contiguous regions);
    // the gather indexes per-dimension so any hyper-rectangle is correct.
    final rank = shape.length;
    final result = await Tensor.create<T>(
      newShape,
      gpu: gpu,
      dataType: dataType,
    );

    // Output strides (row-major over newShape).
    List<int> outStrides = List.filled(rank, 1);
    for (int i = rank - 2; i >= 0; i--) {
      outStrides[i] = outStrides[i + 1] * newShape[i + 1];
    }

    String u32Array(List<int> arr) =>
        'array<u32, $rank>(${arr.map((x) => '${x}u').join(', ')})';
    final wgslType = getWGSLType(dataType);
    final shaderCode =
        '''
const rank: u32 = ${rank}u;
const outStrides: array<u32, $rank> = ${u32Array(outStrides)};
const inStrides: array<u32, $rank> = ${u32Array(strides)};
const starts: array<u32, $rank> = ${u32Array(startIndices)};

@group(0) @binding(0) var<storage, read_write> input: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> output: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= ${numElems}u) { return; }
  var rem: u32 = i;
  var inIndex: u32 = 0u;
  for (var d: u32 = 0u; d < rank; d = d + 1u) {
    let coord: u32 = rem / outStrides[d];
    rem = rem % outStrides[d];
    inIndex = inIndex + (coord + starts[d]) * inStrides[d];
  }
  output[i] = input[inIndex];
}
''';
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', buffer);
    shader.setBuffer('output', result.buffer);
    await shader.dispatchLinear(numElems);
    return result;
  }

  /// Returns the value of the tensor element at the given [indices].
  Future<double> getElement(List<int> indices) async {
    if (indices.length != shape.length) {
      throw Exception(
        "Indices length (${indices.length}) does not match tensor rank (${shape.length}).",
      );
    }

    // Calculate strides (row-major order).
    List<int> strides = List.filled(shape.length, 1);
    for (int i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    int flatIndex = 0;
    for (int i = 0; i < shape.length; i++) {
      if (indices[i] < 0 || indices[i] >= shape[i]) {
        throw Exception(
          "Index out of bounds for dimension $i: ${indices[i]} not in [0, ${shape[i] - 1}].",
        );
      }
      flatIndex += indices[i] * strides[i];
    }

    // Single-element readback — previously this read the ENTIRE tensor.
    final TypedData out = switch (dataType) {
      BufferDataType.int8 => Int8List(1),
      BufferDataType.uint8 => Uint8List(1),
      BufferDataType.int16 => Int16List(1),
      BufferDataType.uint16 || BufferDataType.float16 => Uint16List(1),
      BufferDataType.int32 => Int32List(1),
      BufferDataType.uint32 => Uint32List(1),
      BufferDataType.int64 => Int64List(1),
      BufferDataType.uint64 => Uint64List(1),
      BufferDataType.float32 => Float32List(1),
      BufferDataType.float64 => Float64List(1),
    };
    await buffer.read(out, 1, readOffset: flatIndex, dataType: dataType);
    return getTypedDataElement(out, 0).toDouble();
  }

  /// Sets the value of the tensor element at the given [indices] to [value].
  Future<void> setElement(List<int> indices, double value) async {
    // Calculate strides (row-major order).
    List<int> strides = List.filled(shape.length, 1);
    for (int i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    int flatIndex = 0;
    for (int i = 0; i < shape.length; i++) {
      if (indices[i] < 0 || indices[i] >= shape[i]) {
        throw Exception(
          "Index out of bounds for dimension $i: ${indices[i]} not in [0, ${shape[i] - 1}].",
        );
      }
      // Ensure the result of multiplication is treated as int
      flatIndex += (indices[i] * strides[i]);
    }

    // Get the correct WGSL type string
    final wgslType = getWGSLType(dataType);
    // Format the value correctly for WGSL (e.g., using f32(), i32(), etc.)
    // This assumes getWGSLType returns types like 'f32', 'i32', 'u32'
    final wgslValueString;
    // Basic example, might need refinement based on getWGSLType results and desired precision
    if (wgslType.startsWith('f')) {
      // Ensure it has a decimal or use constructor
      wgslValueString = value.toString().contains('.')
          ? value.toString()
          : '$value.0';
      // Alternative: wgslValueString = '$wgslType($value)'; // e.g., f32(99.0)
    } else if (wgslType.startsWith('i') || wgslType.startsWith('u')) {
      wgslValueString = '${value.toInt()}'; // Convert double to int for i32/u32
      // Alternative: wgslValueString = '$wgslType(${value.toInt()})'; // e.g., i32(99)
    } else {
      // Fallback or error for unsupported types
      wgslValueString = value.toString();
    }

    final shaderTemplate =
        '''
@group(0) @binding(0) var<storage, read_write> A: array<$wgslType>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  A[${flatIndex}u] = $wgslValueString; // Use the formatted value string
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    try {
      final shaderCode = prepareShader(shaderTemplate, dataType, {
        'wgslType': wgslType,
        'flatIndex': flatIndex.toString(),
        'wgslValueString': wgslValueString,
      });
      // Add try-finally for shader destruction
      shader.loadKernelString(shaderCode);
      shader.setBuffer('A', buffer); // Use buffer directly
      await shader.dispatch(1, 1, 1);
    } finally {
      shader.destroy();
    }
  }

  /// Reshapes the tensor into a new shape without changing the underlying data.
  ///
  /// Returns a non-owning view (see [Tensor.view]).  The previous
  /// `Tensor.fromBuffer` implementation attached a SECOND finalizer to the
  /// shared buffer, destroying it under whichever tensor outlived the other.
  Tensor<T> reshape(List<int> newShape) {
    int newSize = newShape.reduce((a, b) => a * b);
    if (newSize != size) {
      throw Exception(
        "New shape $newShape does not match total number of elements $size",
      );
    }
    return Tensor<T>.view(this, newShape);
  }

  /// Simple resize that creates a new tensor with the output shape
  /// Copies existing data and fills missing elements with zeros
  Future<Tensor<T>> resize(List<int> targetShape) async {
    final targetSize = targetShape.reduce((a, b) => a * b);
    final currentSize = size;

    // Create new tensor — WebGPU zero-initializes it, no fill needed.
    final result = await Tensor.create<T>(
      targetShape,
      gpu: gpu,
      dataType: dataType,
    );

    // If current tensor is larger, we'll only copy what fits
    // If output is larger, extra elements stay zero
    final copySize = math.min(currentSize, targetSize);

    if (copySize > 0) {
      await copyElements(this, result, copySize);
    }

    return result;
  }

  /// Fill the entire tensor with a single value
  Future<void> fill(double value) async {
    final wgslType = getWGSLType(dataType);
    final wgslValue = _formatWGSLValue(value, dataType.name);

    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> data: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = global_id.x + global_id.y * (nwg.x * 256u);
  if (i >= ${size}u) { return; }
  
  data[i] = $wgslValue;
}
''';

    final shader = gpu.createComputeShader();
    try {
      shader.loadKernelString(shaderCode);
      shader.setBuffer('data', buffer);

      await shader.dispatchLinear(size);
    } finally {
      shader.destroy();
    }
  }

  /// Helper to format values for WGSL based on data type
  String _formatWGSLValue(double value, String dataType) {
    switch (dataType) {
      case 'float32':
        return '${value}f';
      case 'int32':
        return '${value.toInt()}';
      case 'uint32':
        return '${value.toInt()}u';
      default:
        return '${value}f'; // Default to float
    }
  }

  /// Create a copy of this tensor with the same data and shape
  Future<Tensor<T>> copy() async {
    // Create new tensor with same shape
    final result = await Tensor.create<T>(shape, gpu: gpu, dataType: dataType);

    await _copyAllData(this, result);

    return result;
  }

  /// Fast GPU-based full tensor copy
  Future<void> _copyAllData(Tensor<T> input, Tensor<T> output) async {
    if (input.size != output.size) {
      throw Exception('Tensors must have same size for copying');
    }

    final wgslType = getWGSLType(dataType);

    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> input: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> output: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = global_id.x + global_id.y * (nwg.x * 256u);
  if (i >= ${input.size}u) { return; }
  
  output[i] = input[i];
}
''';

    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', input.buffer);
    shader.setBuffer('output', output.buffer);

    await shader.dispatchLinear(input.size);
  }

  /// Fast GPU-based element copy
  Future<void> copyElements(
    Tensor<T> input,
    Tensor<T> output,
    int count,
  ) async {
    final wgslType = getWGSLType(dataType);

    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> input: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> output: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = global_id.x + global_id.y * (nwg.x * 256u);
  if (i >= ${count}u) { return; }
  
  output[i] = input[i];
}
''';

    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', input.buffer);
    shader.setBuffer('output', output.buffer);

    await shader.dispatchLinear(count);
  }

  /// Simplified reshape - just changes the shape view without changing data
  /// Only works if total elements match exactly
  Tensor<T> reshapeView(List<int> newShape) {
    final newSize = newShape.reduce((a, b) => a * b);
    if (newSize != size) {
      throw Exception(
        'Cannot reshape: current size $size != new size $newSize. '
        'Use resize() if you want to change the tensor size.',
      );
    }
    return Tensor<T>.view(this, newShape);
  }

  /// Pad tensor to output shape by adding zeros
  Future<Tensor<T>> padTo(List<int> targetShape) async {
    // Check if output shape is valid (each dimension >= current)
    if (targetShape.length != shape.length) {
      throw Exception('Target shape rank must match current tensor rank');
    }

    for (int i = 0; i < shape.length; i++) {
      if (targetShape[i] < shape[i]) {
        throw Exception(
          'Target dimension $i ($targetShape[i]) cannot be smaller than current ($shape[i])',
        );
      }
    }

    // If shapes are identical, just copy
    if (_shapesEqual(targetShape, shape)) {
      return copy();
    }

    return await padTensorGPU(targetShape);
  }

  /// GPU-based padding implementation
  Future<Tensor<T>> padTensorGPU(List<int> targetShape) async {
    // Fresh tensor is zero-initialized by WebGPU; only the data copy needed.
    final result = await Tensor.create<T>(
      targetShape,
      gpu: gpu,
      dataType: dataType,
    );

    // Copy existing data to the "top-left" corner
    await copyWithPadding(this, result);

    return result;
  }

  Future<void> copyWithPadding(Tensor<T> input, Tensor<T> output) async {
    final rank = input.shape.length;
    final wgslType = getWGSLType(dataType);

    // Generate stride calculations for both tensors
    String generateStrides(String name, List<int> shape) {
      final strides = <int>[];
      var stride = 1;
      for (int i = shape.length - 1; i >= 0; i--) {
        strides.insert(0, stride);
        stride *= shape[i];
      }
      return 'const ${name}_strides : array<u32, $rank> = array<u32, $rank>(${strides.map((s) => '${s}u').join(', ')});';
    }

    final sourceStrides = generateStrides('input', input.shape);
    final targetStrides = generateStrides('output', output.shape);
    final sourceShapeArray =
        'array<u32, $rank>(${input.shape.map((s) => '${s}u').join(', ')})';

    final shaderCode =
        '''
$sourceStrides
$targetStrides
const source_shape : array<u32, $rank> = $sourceShapeArray;
const rank : u32 = ${rank}u;

@group(0) @binding(0) var<storage, read_write> input: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> output: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let source_index: u32 = global_id.x + global_id.y * (nwg.x * 256u);
  if (source_index >= ${input.size}u) { return; }
  
  // Convert flat index to multi-dimensional indices
  var indices: array<u32, rank>;
  var remainder: u32 = source_index;
  
  for (var i: u32 = 0u; i < rank; i = i + 1u) {
    indices[i] = remainder / source_strides[i];
    remainder = remainder % source_strides[i];
  }
  
  // Calculate output flat index using output strides
  var target_index: u32 = 0u;
  for (var i: u32 = 0u; i < rank; i = i + 1u) {
    target_index = target_index + indices[i] * target_strides[i];
  }
  
  output[target_index] = input[source_index];
}
''';

    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input', input.buffer);
    shader.setBuffer('output', output.buffer);

    await shader.dispatchLinear(input.size);
  }

  bool _shapesEqual(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  /// Transposes the tensor according to the given [axes] permutation.
  Future<Tensor<T>> transpose({List<int>? axes}) async {
    final int rank = shape.length;
    // Default: reverse dimensions.
    axes ??= List<int>.generate(rank, (i) => rank - i - 1);
    if (axes.length != rank) {
      throw Exception(
        "Axes length (${axes.length}) must equal tensor rank ($rank).",
      );
    }
    final sortedAxes = List<int>.from(axes)..sort();
    for (int i = 0; i < rank; i++) {
      if (sortedAxes[i] != i) {
        throw Exception("Invalid axes permutation: $axes.");
      }
    }

    // Compute the new shape.
    List<int> newShape = axes.map((i) => shape[i]).toList();
    final int outSize = newShape.fold(1, (prod, e) => prod * e);

    // Compute input strides (row-major order).
    List<int> inputStrides = List.filled(rank, 1);
    for (int i = rank - 2; i >= 0; i--) {
      inputStrides[i] = inputStrides[i + 1] * shape[i + 1];
    }

    // Compute output factors.
    List<int> outFactors = List.filled(rank, 1);
    for (int i = 0; i < rank; i++) {
      int prod = 1;
      for (int j = i + 1; j < rank; j++) {
        prod *= newShape[j];
      }
      outFactors[i] = prod;
    }

    // Compute inverse permutation.
    List<int> invPermutation = List.filled(rank, 0);
    for (int j = 0; j < rank; j++) {
      invPermutation[axes[j]] = j;
    }

    String formatArray(List<int> arr) => arr.map((x) => "${x}u").join(", ");

    final String shaderCode =
        '''
const outSize : u32 = ${outSize}u;
const rank : u32 = ${rank}u;
const outFactors : array<u32, $rank> = array<u32, $rank>(${formatArray(outFactors)});
const inputStrides : array<u32, $rank> = array<u32, $rank>(${formatArray(inputStrides)});
const invPermutation : array<u32, $rank> = array<u32, $rank>(${formatArray(invPermutation)});

@group(0) @binding(0) var<storage, read_write> input: array<${getWGSLType(dataType)}>;
@group(0) @binding(1) var<storage, read_write> output: array<${getWGSLType(dataType)}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = global_id.x + global_id.y * (nwg.x * 256u);
  if(i >= outSize) { return; }
  var remainder: u32 = i;
  var outIndices: array<u32, rank>;
  for(var j: u32 = 0u; j < rank; j = j + 1u) {
    outIndices[j] = remainder / outFactors[j];
    remainder = remainder % outFactors[j];
  }
  var inIndex: u32 = 0u;
  for(var d: u32 = 0u; d < rank; d = d + 1u) {
    let pos = outIndices[invPermutation[d]];
    inIndex = inIndex + pos * inputStrides[d];
  }
  output[i] = input[inIndex];
}
''';

    Tensor<T> result = await Tensor.create<T>(
      newShape,
      gpu: gpu,
      dataType: dataType,
    );
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer("input", buffer);
    shader.setBuffer("output", result.buffer);
    await shader.dispatchLinear(outSize);

    return result;
  }
}
