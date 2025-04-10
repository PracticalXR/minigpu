import 'dart:typed_data';
import 'package:gpu_tensor/src/gpu_helpers.dart';
import 'package:minigpu/minigpu.dart';

import 'gpu_tensor_base.dart';

extension TensorData<T extends TypedData> on Tensor<T> {
  /// Creates a new tensor by slicing the flattened tensor data.
  /// [start] is the starting flat index and [end] is the ending flat index (exclusive).
  Future<Tensor<T>> sliceLinear({required int start, required int end}) async {
    if (start < 0 || end > size || start >= end) {
      throw Exception(
          "Invalid slice indices: start=$start, end=$end, size=$size.");
    }
    int newSize = end - start;
    // Read the tensor data then slice it using getTypedDataSublist.
    T fullData = await getData();
    final T slicedData = getTypedDataSublist<T>(fullData, start, end);
    // Create a new tensor with 1D shape.
    return await Tensor.create<T>(
      [newSize],
      data: slicedData,
      dataType: dataType,
    );
  }

  /// Slices the tensor based on multi-dimensional indices.
  Future<Tensor<T>> slice({
    required List<int> startIndices,
    required List<int> endIndices,
  }) async {
    if (startIndices.length != shape.length ||
        endIndices.length != shape.length) {
      throw Exception(
          "startIndices and endIndices must match tensor rank (${shape.length}).");
    }

    // Calculate strides (row-major order).
    List<int> strides = List.filled(shape.length, 1);
    for (int i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Compute flat offset.
    int flatOffset = 0;
    for (int i = 0; i < shape.length; i++) {
      if (startIndices[i] < 0 ||
          endIndices[i] > shape[i] ||
          startIndices[i] >= endIndices[i]) {
        throw Exception(
            "Invalid slice indices for dimension $i: start=${startIndices[i]}, end=${endIndices[i]}, shape=${shape[i]}.");
      }
      flatOffset += startIndices[i] * strides[i];
    }

    // Compute new shape and total elements.
    List<int> newShape = [];
    int numElems = 1;
    for (int i = 0; i < shape.length; i++) {
      int dimSize = endIndices[i] - startIndices[i];
      newShape.add(dimSize);
      numElems *= dimSize;
    }

    // Read the entire tensor data then extract the slice.
    T fullData = await getData();
    final T slicedData =
        getTypedDataSublist(fullData, flatOffset, flatOffset + numElems);
    return await Tensor.create<T>(
      newShape,
      data: slicedData,
      dataType: dataType,
    );
  }

  /// Returns the value of the tensor element at the given [indices].
  Future<double> getElement(List<int> indices) async {
    if (indices.length != shape.length) {
      throw Exception(
          "Indices length (${indices.length}) does not match tensor rank (${shape.length}).");
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
            "Index out of bounds for dimension $i: ${indices[i]} not in [0, ${shape[i] - 1}].");
      }
      flatIndex += indices[i] * strides[i];
    }

    T data = await getData();
    // Use our helper to retrieve the element.
    return getTypedDataElement(data, flatIndex).toDouble();
  }

  /// Sets the value of the tensor element at the given [indices] to [value].
  Future<void> setElement(List<int> indices, double value) async {
    if (indices.length != shape.length) {
      throw Exception(
          "Indices length (${indices.length}) does not match tensor rank (${shape.length}).");
    }
    List<int> strides = List.filled(shape.length, 1);
    for (int i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    int flatIndex = 0;
    for (int i = 0; i < shape.length; i++) {
      if (indices[i] < 0 || indices[i] >= shape[i]) {
        throw Exception(
            "Index out of bounds for dimension $i: ${indices[i]} not in [0, ${shape[i] - 1}].");
      }
      flatIndex += indices[i] * strides[i];
    }

    final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
  
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  A[${flatIndex}u] = $value;
}
''';
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer('A', buffer);
    await shader.dispatch(1, 1, 1);
    shader.destroy();
  }

  /// Reshapes the tensor into a new shape without changing the underlying data.
  Tensor<T> reshape(List<int> newShape) {
    int newSize = newShape.reduce((a, b) => a * b);
    if (newSize != size) {
      throw Exception(
          "New shape $newShape does not match total number of elements $size");
    }
    // Use the generic fromBuffer constructor.
    return Tensor.fromBuffer(buffer, newShape, gpu: gpu, dataType: dataType);
  }

  /// Transposes the tensor according to the given [axes] permutation.
  Future<Tensor<T>> transpose({List<int>? axes}) async {
    final int rank = shape.length;
    // Default: reverse dimensions.
    axes ??= List<int>.generate(rank, (i) => rank - i - 1);
    if (axes.length != rank) {
      throw Exception(
          "Axes length (${axes.length}) must equal tensor rank ($rank).");
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

    final String shaderCode = '''
const outSize : u32 = ${outSize}u;
const rank : u32 = ${rank}u;
const outFactors : array<u32, $rank> = array<u32, $rank>(${formatArray(outFactors)});
const inputStrides : array<u32, $rank> = array<u32, $rank>(${formatArray(inputStrides)});
const invPermutation : array<u32, $rank> = array<u32, $rank>(${formatArray(invPermutation)});

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let i: u32 = global_id.x;
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
    final ComputeShader shader = gpu.createComputeShader();
    shader.loadKernelString(shaderCode);
    shader.setBuffer("input", buffer);
    shader.setBuffer("output", result.buffer);
    int wgCount = (outSize + 255) ~/ 256;
    await shader.dispatch(wgCount, 1, 1);
    shader.destroy();

    return result;
  }
}
