import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

/// A buffer.
final class Buffer {
  Buffer(PlatformBuffer buffer) : _platformBuffer = buffer;

  // Store the platform-specific buffer implementation. Marked as potentially nullable
  // if we consider the buffer invalid after destruction.
  PlatformBuffer? _platformBuffer;
  bool _isValid = true; // Flag to track if destroy has been called

  /// Returns true if the buffer has not been destroyed.
  bool get isValid => _isValid && _platformBuffer != null;

  /// Reads data from the buffer asynchronously.
  /// Throws an error if the buffer has been destroyed.
  Future<void> read(
    TypedData outputData,
    int size, {
    // Note: This 'size' likely means element count based on previous context
    int readOffset = 0,
    BufferDataType dataType = BufferDataType.float32,
  }) async {
    if (!isValid) {
      throw StateError('Cannot read from a destroyed buffer.');
    }
    // Use the null assertion operator (!) as isValid check ensures it's not null
    return _platformBuffer!.read(
      outputData,
      size, // Pass element count
      elementOffset: readOffset,
      dataType: dataType,
    );
  }

  /// Writes data to the buffer.
  /// Throws an error if the buffer has been destroyed.
  void setData(
    TypedData inputData,
    int size, {
    // Note: This 'size' likely means element count based on previous context
    BufferDataType dataType = BufferDataType.float32,
  }) {
    if (!isValid) {
      throw StateError('Cannot write to a destroyed buffer.');
    }
    // Use the null assertion operator (!)
    _platformBuffer!.setData(
      inputData,
      size, // Pass element count
      dataType: dataType,
    );
  }

  /// Destroys the buffer and releases associated resources.
  /// Can be called multiple times safely.
  void destroy() {
    if (_isValid && _platformBuffer != null) {
      _platformBuffer!.destroy();
      _platformBuffer = null; // Release the reference
      _isValid = false; // Mark as destroyed
    }
    // If already destroyed (_isValid is false), do nothing.
  }

  // Internal access for ComputeShader (consider if this is the best pattern)
  // Or adjust ComputeShader to accept Buffer directly.
  PlatformBuffer? get platformBuffer => _platformBuffer;
}

class MinigpuAlreadyInitError extends Error {
  MinigpuAlreadyInitError([this.message]);

  final String? message;

  @override
  String toString() => message == null
      ? 'Minigpu already initialized'
      : 'Minigpu already initialized: $message';
}

class MinigpuNotInitializedError extends Error {
  MinigpuNotInitializedError([this.message]);

  final String? message;

  @override
  String toString() => message == null
      ? 'Minigpu not initialized'
      : 'Minigpu not initialized: $message';
}
