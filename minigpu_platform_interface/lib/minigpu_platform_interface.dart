import 'dart:typed_data';

import 'platform_stub/minigpu_platform_stub.dart'
    if (dart.library.ffi) 'package:minigpu_ffi/minigpu_ffi.dart'
    if (dart.library.js) 'package:minigpu_web/minigpu_web.dart';

/// Enum representing supported buffer data types.
enum BufferDataType {
  float16,
  float32, // 0 (Renamed from float)
  float64, // 1 (Renamed from double)
  int8, // 2
  int16, // 3
  int32, // 4
  int64, // 5
  uint8, // 6
  uint16, // 7
  uint32, // 8
  uint64, // 9
}

String getWGSLType(BufferDataType type) {
  switch (type) {
    case BufferDataType.int8:
    case BufferDataType.int16:
    case BufferDataType.int32:
    case BufferDataType.int64: // Packed as i32
      return 'i32';
    case BufferDataType.uint8:
    case BufferDataType.uint16:
    case BufferDataType.uint32:
    case BufferDataType.uint64: // Packed as u32
      return 'u32';
    case BufferDataType.float16:
    case BufferDataType.float32:
      return 'f32';
    case BufferDataType.float64:
      return 'u32';
  }
}

int getBufferSizeForType(BufferDataType type, int count) {
  switch (type) {
    case BufferDataType.float16:
      return count * (Float32List.bytesPerElement / 2).toInt();
    case BufferDataType.float32:
      return count * Float32List.bytesPerElement;
    case BufferDataType.float64:
      // Even though doubles are packed internally, the API “appears” to use 8 bytes per element.
      return count * Float64List.bytesPerElement;
    case BufferDataType.int32:
      return count * Int32List.bytesPerElement;
    case BufferDataType.int64:
      return count * Int64List.bytesPerElement;
    case BufferDataType.int8:
      return count * Int8List.bytesPerElement;
    case BufferDataType.uint8:
      return count * Uint8List.bytesPerElement;
    case BufferDataType.int16:
      return count * Int16List.bytesPerElement;
    case BufferDataType.uint16:
      return count * Uint16List.bytesPerElement;
    case BufferDataType.uint32:
      return count * Uint32List.bytesPerElement;
    case BufferDataType.uint64:
      return count * Uint64List.bytesPerElement;
  }
}

abstract class MinigpuPlatform {
  MinigpuPlatform();

  static MinigpuPlatform? _instance;

  /// Returns the current instance; creates if not yet initialized.
  static MinigpuPlatform get instance {
    _instance ??= registeredInstance();
    return _instance!;
  }

  MinigpuPlatform registerInstance() =>
      throw UnimplementedError('No platform implementation available.');

  Future<void> initializeContext();
  Future<void> destroyContext();
  PlatformComputeShader createComputeShader();
  PlatformBuffer createBuffer(int bufferSize, BufferDataType dataType);
}

abstract class PlatformComputeShader {
  void loadKernelString(String kernelString);
  bool hasKernel();
  void setBuffer(int tag, PlatformBuffer buffer);
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ);
  void destroy();
}

abstract class PlatformBuffer {
  Future<void> read(
    TypedData outputData,
    int readElements, {
    int elementOffset = 0,
    int readBytes = 0,
    int byteOffset = 0,
    BufferDataType dataType = BufferDataType.float32,
  });
  void setData(
    TypedData inputData,
    int size, {
    BufferDataType dataType = BufferDataType.float32,
  });
  void destroy();
}

final class MinigpuPlatformOutOfMemoryException implements Exception {
  @override
  String toString() => 'Out of memory';
}
