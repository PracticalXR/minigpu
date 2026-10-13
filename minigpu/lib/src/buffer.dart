import 'dart:typed_data';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

import 'resource_retirement.dart';

/// A buffer.
final class Buffer {
  /// [owner] is an optional callback invoked exactly once when [destroy] is
  /// called (or the finalizer runs).  [Minigpu] passes [_onBufferDestroyed]
  /// here so it can maintain [liveBufferCount] without a circular import.
  Buffer(
    PlatformBuffer buffer, [
    void Function()? owner,
    this.descriptor,
    GpuResourceRetirement? retirement,
  ]) : _platformBuffer = buffer,
       _retirement =
           retirement ?? GpuResourceRetirement(buffer.destroy, owner) {
    _finalizer.attach(this, _retirement, detach: this);
  }

  static final Finalizer<GpuResourceRetirement> _finalizer = Finalizer((token) {
    token.retire();
  });

  PlatformBuffer? _platformBuffer;
  final GpuResourceRetirement _retirement;
  final BufferDescriptor? descriptor;
  bool _isValid = true;

  /// Returns true if the buffer has not been destroyed.
  bool get isValid =>
      _isValid && _platformBuffer != null && !_retirement.isRetired;
  GpuResourceRetirement get retirement => _retirement;

  /// Web only: the Emscripten integer handle of the underlying `WGPUBuffer`
  /// (0 natively, or when destroyed) — the zero-readback web display path.
  /// A view layer resolves the JS `GPUBuffer` from it and blits into a canvas
  /// on the same device queue; see `PlatformBuffer.webBufferHandle`.
  int get webBufferHandle =>
      isValid ? (_platformBuffer?.webBufferHandle ?? 0) : 0;

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
    if (descriptor != null && descriptor!.usage & BufferUsage.copySrc == 0) {
      throw UnsupportedError('Buffer was not created with copySrc usage');
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
  Future<void> write(
    TypedData inputData,
    int size, {
    // Note: This 'size' likely means element count based on previous context
    BufferDataType dataType = BufferDataType.float32,
  }) async {
    if (!isValid) {
      throw StateError('Cannot write to a destroyed buffer.');
    }
    if (descriptor != null && descriptor!.usage & BufferUsage.copyDst == 0) {
      throw UnsupportedError('Buffer was not created with copyDst usage');
    }
    // Use the null assertion operator (!)
    await _platformBuffer!.write(
      inputData,
      size, // Pass element count
      dataType: dataType,
    );
  }

  /// Raw byte upload at [dstByteOffset] (both must be 4-byte multiples),
  /// streamed in bounded chunks — use for LARGE uploads (model weights) so
  /// host scratch/driver staging never holds the whole payload.
  Future<void> writeRawBytes(Uint8List bytes, {int dstByteOffset = 0}) async {
    if (!isValid) {
      throw StateError('Cannot write to a destroyed buffer.');
    }
    if (descriptor != null && descriptor!.usage & BufferUsage.copyDst == 0) {
      throw UnsupportedError('Buffer was not created with copyDst usage');
    }
    await _platformBuffer!.writeRawBytes(bytes, dstByteOffset: dstByteOffset);
  }

  /// Destroys the buffer and releases associated resources.
  /// Can be called multiple times safely.
  void destroy() {
    final buffer = _platformBuffer;
    if (_isValid && buffer != null) {
      _finalizer.detach(this);
      _platformBuffer = null;
      _isValid = false;
      _retirement.retire();
    }
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
