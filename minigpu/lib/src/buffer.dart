import 'dart:typed_data';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

/// A buffer.
final class Buffer {
  /// [owner] is an optional callback invoked exactly once when [destroy] is
  /// called (or the finalizer runs).  [Minigpu] passes [_onBufferDestroyed]
  /// here so it can maintain [liveBufferCount] without a circular import.
  Buffer(PlatformBuffer buffer, [void Function()? owner])
    : _platformBuffer = buffer,
      _owner = owner {
    _finalizer.attach(this, _DestroyArgs(buffer, owner), detach: this);
  }

  static final Finalizer<_DestroyArgs> _finalizer = Finalizer((args) {
    args.platformBuffer.destroy();
    args.owner?.call();
  });

  PlatformBuffer? _platformBuffer;
  final void Function()? _owner;
  bool _isValid = true;

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
  Future<void> write(
    TypedData inputData,
    int size, {
    // Note: This 'size' likely means element count based on previous context
    BufferDataType dataType = BufferDataType.float32,
  }) async {
    if (!isValid) {
      throw StateError('Cannot write to a destroyed buffer.');
    }
    // Use the null assertion operator (!)
    await _platformBuffer!.write(
      inputData,
      size, // Pass element count
      dataType: dataType,
    );
  }

  /// Destroys the buffer and releases associated resources.
  /// Can be called multiple times safely.
  void destroy() {
    if (_isValid && _platformBuffer != null) {
      _finalizer.detach(this);
      _platformBuffer!.destroy();
      _owner?.call();
      _platformBuffer = null;
      _isValid = false;
    }
  }

  // Internal access for ComputeShader (consider if this is the best pattern)
  // Or adjust ComputeShader to accept Buffer directly.
  PlatformBuffer? get platformBuffer => _platformBuffer;
}

/// Internal holder used by [Buffer]'s [Finalizer] so the finalizer can invoke
/// both the platform destroy and the owner callback without capturing `this`.
class _DestroyArgs {
  const _DestroyArgs(this.platformBuffer, this.owner);
  final PlatformBuffer platformBuffer;
  final void Function()? owner;
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
