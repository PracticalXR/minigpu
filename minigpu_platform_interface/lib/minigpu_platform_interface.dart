import 'dart:typed_data';

import 'platform_stub/minigpu_platform_stub.dart'
    if (dart.library.ffi) 'package:minigpu_ffi/minigpu_ffi.dart'
    if (dart.library.js) 'package:minigpu_web/minigpu_web.dart';

/// Enum representing supported buffer data types.
enum BufferDataType {
  int8,
  int16,
  int32,
  int64,
  uint8,
  uint16,
  uint32,
  float,
  double,
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
  void destroyContext();
  PlatformComputeShader createComputeShader();
  PlatformBuffer createBuffer(int bufferSize);
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
    BufferDataType dataType = BufferDataType.float,
  });
  void setData(
    TypedData inputData,
    int size, {
    BufferDataType dataType = BufferDataType.float,
  });
  void destroy();
}

final class MinigpuPlatformOutOfMemoryException implements Exception {
  @override
  String toString() => 'Out of memory';
}
