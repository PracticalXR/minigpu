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
  ///
  /// [dataType] defaults to the element type of [outputData]; an explicit one
  /// must match it (see [write]).
  Future<void> read(
    TypedData outputData,
    int size, {
    // Note: This 'size' likely means element count based on previous context
    int readOffset = 0,
    BufferDataType? dataType,
  }) async {
    if (!isValid) {
      throw StateError('Cannot read from a destroyed buffer.');
    }
    if (descriptor != null && descriptor!.usage & BufferUsage.copySrc == 0) {
      throw UnsupportedError('Buffer was not created with copySrc usage');
    }
    final type = _resolveDataType(outputData, dataType, 'outputData');
    // Use the null assertion operator (!) as isValid check ensures it's not null
    return _platformBuffer!.read(
      _plainView(outputData),
      size, // Pass element count
      elementOffset: readOffset,
      dataType: type,
    );
  }

  /// Writes data to the buffer.
  /// Throws an error if the buffer has been destroyed.
  ///
  /// [dataType] defaults to the element type of [inputData]. An explicit one
  /// must match it ([ArgumentError] otherwise): [ByteData] is raw bytes and
  /// needs one, and a [Uint16List] may carry [BufferDataType.float16] bits.
  Future<void> write(
    TypedData inputData,
    int size, {
    // Note: This 'size' likely means element count based on previous context
    BufferDataType? dataType,
  }) async {
    if (!isValid) {
      throw StateError('Cannot write to a destroyed buffer.');
    }
    if (descriptor != null && descriptor!.usage & BufferUsage.copyDst == 0) {
      throw UnsupportedError('Buffer was not created with copyDst usage');
    }
    final type = _resolveDataType(inputData, dataType, 'inputData');
    // Use the null assertion operator (!)
    await _platformBuffer!.write(
      _plainView(inputData),
      size, // Pass element count
      dataType: type,
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

/// [dataType] when given, else the element type of [data]. Both backends take
/// the bytes of [data] as [dataType] elements, also when the list holds
/// another type, which silently reinterprets its bits: that is rejected here.
/// [ByteData] is raw bytes and needs an explicit type; a [Uint16List] may
/// carry float16 bits.
BufferDataType _resolveDataType(
  TypedData data,
  BufferDataType? dataType,
  String name,
) {
  final elementType = _elementTypeOf(data);
  if (dataType == null) {
    if (elementType != null) return elementType;
    throw ArgumentError(
      '$name is ${data.runtimeType}, which has no element type: pass dataType',
    );
  }
  if (data is ByteData ||
      dataType == elementType ||
      (dataType == BufferDataType.float16 && data is Uint16List)) {
    return dataType;
  }
  throw ArgumentError.value(
    dataType,
    'dataType',
    'does not match the ${elementType?.name ?? data.runtimeType} elements of '
        '$name',
  );
}

BufferDataType? _elementTypeOf(TypedData data) {
  if (data is Float32List || data is Float32x4List) {
    return BufferDataType.float32;
  }
  if (data is Uint32List) return BufferDataType.uint32;
  if (data is Int32List || data is Int32x4List) return BufferDataType.int32;
  if (data is Uint8List || data is Uint8ClampedList) {
    return BufferDataType.uint8;
  }
  if (data is Int8List) return BufferDataType.int8;
  if (data is Uint16List) return BufferDataType.uint16;
  if (data is Int16List) return BufferDataType.int16;
  if (data is Float64List || data is Float64x2List) {
    return BufferDataType.float64;
  }
  if (data is Int64List) return BufferDataType.int64;
  if (data is Uint64List) return BufferDataType.uint64;
  return null; // ByteData
}

/// [data] as the plain list the backends cast to for its element type, over
/// the same bytes, so a read still lands in the caller's list.
TypedData _plainView(TypedData data) {
  if (data is Uint8ClampedList) {
    return data.buffer.asUint8List(data.offsetInBytes, data.length);
  }
  if (data is Float32x4List) {
    return data.buffer.asFloat32List(data.offsetInBytes, data.length * 4);
  }
  if (data is Int32x4List) {
    return data.buffer.asInt32List(data.offsetInBytes, data.length * 4);
  }
  if (data is Float64x2List) {
    return data.buffer.asFloat64List(data.offsetInBytes, data.length * 2);
  }
  return data;
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
