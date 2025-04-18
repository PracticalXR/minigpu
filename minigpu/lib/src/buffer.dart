import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

/// A buffer.
final class Buffer {
  Buffer(PlatformBuffer buffer) : platformBuffer = buffer;

  final PlatformBuffer platformBuffer;

  /// Reads data from the buffer synchronously.
  Future<void> read(
    TypedData outputData,
    int size, {
    int readOffset = 0,
    BufferDataType dataType = BufferDataType.float32,
  }) async =>
      platformBuffer.read(
        outputData,
        size,
        elementOffset: readOffset,
        dataType: dataType,
      );

  /// Writes data to the buffer.
  void setData(
    TypedData inputData,
    int size, {
    BufferDataType dataType = BufferDataType.float32,
  }) =>
      platformBuffer.setData(
        inputData,
        size,
        dataType: dataType,
      );

  /// Destroys the buffer.
  void destroy() => platformBuffer.destroy();
}

class MinigpuAlreadyInitError extends Error {
  MinigpuAlreadyInitError([this.message]);

  final String? message;

  @override
  String toString() => message == null
      ? 'Minigpu already initialized'
      : 'Minigpu already initialized: $message';
}
