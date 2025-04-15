import 'dart:typed_data';
import 'package:minigpu/minigpu.dart';

int _elementSize(BufferDataType type) {
  switch (type) {
    case BufferDataType.int8:
    case BufferDataType.uint8:
      return 1;
    case BufferDataType.int16:
    case BufferDataType.uint16:
      return 2;
    case BufferDataType.int32:
    case BufferDataType.uint32:
    case BufferDataType.float:
      return 4;
    case BufferDataType.int64:
    case BufferDataType.uint64:
    case BufferDataType.double:
      return 8;
  }
}

/// A helper that creates (or reuses) a default GPU context.
class DefaultMinigpu {
  static final instance = Minigpu();
}

/// A generic tensor that works with a specific [TypedData] type.
class Tensor<T extends TypedData> {
  /// The shape in terms of dimensions.
  final List<int> shape;

  /// Total number of elements.
  final int size;
  int get rank => shape.length;
  final Minigpu gpu;
  late Buffer buffer;

  /// The underlying data type.
  final BufferDataType dataType;

  // Private constructor.
  Tensor._(this.shape,
      {required this.gpu, T? data, this.dataType = BufferDataType.float})
      : size = shape.reduce((a, b) => a * b) {
    final int byteSize = size * _elementSize(dataType);
    buffer = gpu.createBuffer(byteSize);
    if (data != null) {
      if (data.lengthInBytes ~/ _elementSize(dataType) != size) {
        throw Exception(
            "Provided data length (${data.lengthInBytes ~/ _elementSize(dataType)}) does not match tensor size ($size)");
      }
      buffer.setData(data, size, dataType: dataType);
    } else {
      // Initialize with zeros.
      if (T == TypedData) {
        buffer.setData(Float32List(size), size, dataType: dataType);
      } else if (T == Int8List) {
        buffer.setData(Int8List(size), size, dataType: dataType);
      } else if (T == Int16List) {
        buffer.setData(Int16List(size), size, dataType: dataType);
      } else if (T == Int32List) {
        buffer.setData(Int32List(size), size, dataType: dataType);
      } else if (T == Int64List) {
        buffer.setData(Int64List(size), size, dataType: dataType);
      } else if (T == Uint8List) {
        buffer.setData(Uint8List(size), size, dataType: dataType);
      } else if (T == Uint16List) {
        buffer.setData(Uint16List(size), size, dataType: dataType);
      } else if (T == Uint32List) {
        buffer.setData(Uint32List(size), size, dataType: dataType);
      } else if (T == Float32List) {
        buffer.setData(Float32List(size), size, dataType: dataType);
      } else if (T == Float64List) {
        buffer.setData(Float64List(size), size, dataType: dataType);
      } else {
        throw Exception("Unsupported TypedData type: ${T.toString()}");
      }
    }
  }

  /// Asynchronous factory to create a tensor.
  static Future<Tensor<T>> create<T extends TypedData>(
    List<int> shape, {
    Minigpu? gpu,
    T? data,
    BufferDataType dataType = BufferDataType.float,
  }) async {
    gpu = gpu ?? DefaultMinigpu.instance;
    if (!gpu.isInitialized) {
      await gpu.init();
    }
    return Tensor._(shape, gpu: gpu, data: data, dataType: dataType);
  }

  /// Returns the data from the GPU buffer as type T.
  Future<T> getData() async {
    late T result;
    if (T == TypedData) {
      result = Float32List(size) as T;
    } else if (T == Int8List) {
      result = Int8List(size) as T;
    } else if (T == Int16List) {
      result = Int16List(size) as T;
    } else if (T == Int32List) {
      result = Int32List(size) as T;
    } else if (T == Int64List) {
      result = Int64List(size) as T;
    } else if (T == Uint8List) {
      result = Uint8List(size) as T;
    } else if (T == Uint16List) {
      result = Uint16List(size) as T;
    } else if (T == Uint32List) {
      result = Uint32List(size) as T;
    } else if (T == Float32List) {
      result = Float32List(size) as T;
    } else if (T == Float64List) {
      result = Float64List(size) as T;
    } else {
      throw Exception("Unsupported TypedData type: ${T.toString()}");
    }
    await buffer.read(result, size, dataType: dataType);
    return result;
  }

  /// Writes [data] of type T to the GPU buffer.
  void setData(T data) {
    buffer.setData(data, size, dataType: dataType);
  }

  /// Destroys the tensor's GPU buffer.
  void destroy() {
    buffer.destroy();
  }

  /// Creates a tensor from an existing buffer.
  Tensor.fromBuffer(this.buffer, this.shape,
      {Minigpu? gpu, this.dataType = BufferDataType.float})
      : gpu = gpu ?? DefaultMinigpu.instance,
        size = shape.reduce((a, b) => a * b);
}
