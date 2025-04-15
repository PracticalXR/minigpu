// ignore_for_file: omit_local_variable_types

import 'dart:async';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:minigpu_ffi/minigpu_ffi_bindings.dart' as ffi;
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

typedef ReadAsyncCallbackFunc = Void Function(Pointer<Void>);
typedef ReadAsyncCallback = Pointer<NativeFunction<ReadAsyncCallbackFunc>>;

MinigpuPlatform registeredInstance() => MinigpuFfi();

// Minigpu FFI
class MinigpuFfi extends MinigpuPlatform {
  MinigpuFfi();

  @override
  Future<void> initializeContext() async {
    final completer = Completer<void>();

    void nativeCallback() {
      completer.complete();
    }

    final nativeCallable =
        NativeCallable<Void Function()>.listener(nativeCallback);

    ffi.mgpuInitializeContextAsync(nativeCallable.nativeFunction);

    await completer.future;
    nativeCallable.close();
  }

  @override
  void destroyContext() {
    ffi.mgpuDestroyContext();
  }

  @override
  PlatformComputeShader createComputeShader() {
    final self = ffi.mgpuCreateComputeShader();
    if (self == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return FfiComputeShader(self);
  }

  @override
  PlatformBuffer createBuffer(int bufferSize) {
    final self = ffi.mgpuCreateBuffer(bufferSize);
    if (self == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return FfiBuffer(self);
  }
}

// Compute shader FFI
final class FfiComputeShader implements PlatformComputeShader {
  FfiComputeShader(Pointer<ffi.MGPUComputeShader> self) : _self = self;

  final Pointer<ffi.MGPUComputeShader> _self;

  @override
  void loadKernelString(String kernelString) {
    final kernelStringPtr = kernelString.toNativeUtf8();
    try {
      ffi.mgpuLoadKernel(_self, kernelStringPtr.cast());
    } finally {
      malloc.free(kernelStringPtr);
    }
  }

  @override
  bool hasKernel() {
    return ffi.mgpuHasKernel(_self) != 0;
  }

  @override
  void setBuffer(int tag, PlatformBuffer buffer) {
    try {
      ffi.mgpuSetBuffer(_self, tag, (buffer as FfiBuffer)._self);
    } finally {}
  }

  @override
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ) async {
    try {
      final completer = Completer<void>();

      void nativeCallback() {
        completer.complete();
      }

      final nativeCallable =
          NativeCallable<Void Function()>.listener(nativeCallback);
      ffi.mgpuDispatchAsync(
          _self, groupsX, groupsY, groupsZ, nativeCallable.nativeFunction);
      await completer.future;
      nativeCallable.close();
    } finally {}
  }

  @override
  void destroy() {
    ffi.mgpuDestroyComputeShader(_self);
  }
}

// Buffer FFI
final class FfiBuffer implements PlatformBuffer {
  FfiBuffer(Pointer<ffi.MGPUBuffer> self) : _self = self;

  final Pointer<ffi.MGPUBuffer> _self;

  @override
  Future<void> read(
    TypedData outputData,
    int readElements, {
    int elementOffset = 0,
    int readBytes = 0,
    int byteOffset = 0,
    BufferDataType dataType = BufferDataType.float,
  }) async {
    // Determine element size based on data type.
    final int elementSize;
    switch (dataType) {
      case BufferDataType.int8:
        elementSize = sizeOf<Int8>();
        break;
      case BufferDataType.int16:
        elementSize = sizeOf<Int16>();
        break;
      case BufferDataType.int32:
        elementSize = sizeOf<Int32>();
        break;
      case BufferDataType.int64:
        elementSize = sizeOf<Int64>();
        break;
      case BufferDataType.uint8:
        elementSize = sizeOf<Uint8>();
        break;
      case BufferDataType.uint16:
        elementSize = sizeOf<Uint16>();
        break;
      case BufferDataType.uint32:
        elementSize = sizeOf<Uint32>();
        break;
      case BufferDataType.uint64:
        elementSize = sizeOf<Uint64>();
        break;
      case BufferDataType.float:
        elementSize = sizeOf<Float>();
        break;
      case BufferDataType.double:
        elementSize = sizeOf<Double>();
        break;
    }

    final int totalElements = outputData.elementSizeInBytes ~/ elementSize;
    final int sizeToRead = readElements != 0
        ? readElements
        : (readBytes != 0
            ? readBytes ~/ elementSize
            : totalElements - elementOffset);
    final int effectiveByteOffset =
        readElements != 0 ? elementOffset * elementSize : byteOffset;
    final int byteSize = sizeToRead * elementSize;

    final completer = Completer<void>();

    // Native callback; called when the async operation completes.
    void nativeCallback() {
      completer.complete();
    }

    final nativeCallable =
        NativeCallable<Void Function()>.listener(nativeCallback);

    // Switch to call the proper native function.
    switch (dataType) {
      case BufferDataType.int8:
        {
          final Pointer<Int8> nativePtr = malloc.allocate<Int8>(byteSize);

          ffi.mgpuReadBufferAsyncInt8(
            _self,
            nativePtr,
            byteSize,
            effectiveByteOffset,
            nativeCallable.nativeFunction,
          );
          await completer.future;
          final List<int> data = nativePtr.asTypedList(sizeToRead);
          if (outputData is Int8List) {
            outputData.setAll(0, data);
          } else if (outputData is ByteData) {
            outputData.buffer.asInt8List().setAll(0, data);
          }
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.int16:
        {
          final Pointer<Int16> nativePtr = malloc.allocate<Int16>(byteSize);
          ffi.mgpuReadBufferAsyncInt16(
            _self,
            nativePtr,
            byteSize,
            effectiveByteOffset,
            nativeCallable.nativeFunction,
          );
          await completer.future;
          final List<int> data = nativePtr.asTypedList(sizeToRead);
          if (outputData is Int16List) {
            outputData.setAll(0, data);
          } else if (outputData is ByteData) {
            outputData.buffer.asInt16List().setAll(0, data);
          }
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.int32:
        {
          final Pointer<Int32> nativePtr = malloc.allocate<Int32>(byteSize);
          ffi.mgpuReadBufferAsyncInt32(
            _self,
            nativePtr,
            byteSize,
            effectiveByteOffset,
            nativeCallable.nativeFunction,
          );
          await completer.future;
          final List<int> data = nativePtr.asTypedList(sizeToRead);
          if (outputData is Int32List) {
            outputData.setAll(0, data);
          } else if (outputData is ByteData) {
            outputData.buffer.asInt32List().setAll(0, data);
          }
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.int64:
        {
          final Pointer<Int64> nativePtr = malloc.allocate<Int64>(byteSize);
          ffi.mgpuReadBufferAsyncInt64(
            _self,
            nativePtr,
            byteSize,
            effectiveByteOffset,
            nativeCallable.nativeFunction,
          );
          await completer.future;
          final List<int> data = nativePtr.asTypedList(sizeToRead);
          if (outputData is Int64List) {
            outputData.setAll(0, data);
          }
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.uint8:
        {
          final Pointer<Uint8> nativePtr = malloc.allocate<Uint8>(byteSize);
          ffi.mgpuReadBufferAsyncUint8(
            _self,
            nativePtr,
            byteSize,
            effectiveByteOffset,
            nativeCallable.nativeFunction,
          );
          await completer.future;
          final List<int> data = nativePtr.asTypedList(sizeToRead);
          if (outputData is Uint8List) {
            outputData.setAll(0, data);
          } else if (outputData is ByteData) {
            outputData.buffer.asUint8List().setAll(0, data);
          }
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.uint16:
        {
          final Pointer<Uint16> nativePtr = malloc.allocate<Uint16>(byteSize);
          ffi.mgpuReadBufferAsyncUint16(
            _self,
            nativePtr,
            byteSize,
            effectiveByteOffset,
            nativeCallable.nativeFunction,
          );
          await completer.future;
          final List<int> data = nativePtr.asTypedList(sizeToRead);
          if (outputData is Uint16List) {
            outputData.setAll(0, data);
          } else if (outputData is ByteData) {
            outputData.buffer.asUint16List().setAll(0, data);
          }
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.uint32:
        {
          final Pointer<Uint32> nativePtr = malloc.allocate<Uint32>(byteSize);
          ffi.mgpuReadBufferAsyncUint32(
            _self,
            nativePtr,
            byteSize,
            effectiveByteOffset,
            nativeCallable.nativeFunction,
          );
          await completer.future;
          final List<int> data = nativePtr.asTypedList(sizeToRead);
          if (outputData is Uint32List) {
            outputData.setAll(0, data);
          } else if (outputData is ByteData) {
            outputData.buffer.asUint32List().setAll(0, data);
          }
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.uint64:
        {
          final Pointer<Uint64> nativePtr = malloc.allocate<Uint64>(byteSize);
          ffi.mgpuReadBufferAsyncUint64(
            _self,
            nativePtr,
            byteSize,
            effectiveByteOffset,
            nativeCallable.nativeFunction,
          );
          await completer.future;
          final List<int> data = nativePtr.asTypedList(sizeToRead);
          if (outputData is Uint64List) {
            outputData.setAll(0, data);
          }
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.float:
        {
          final Pointer<Float> nativePtr = malloc.allocate<Float>(byteSize);
          ffi.mgpuReadBufferAsyncFloat(
            _self,
            nativePtr,
            byteSize,
            effectiveByteOffset,
            nativeCallable.nativeFunction,
          );
          await completer.future;
          final List<double> data = nativePtr.asTypedList(sizeToRead);
          if (outputData is Float32List) {
            outputData.setAll(0, data);
          } else if (outputData is ByteData) {
            outputData.buffer.asFloat32List().setAll(0, data);
          }
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.double:
        {
          final Pointer<Double> nativePtr = malloc.allocate<Double>(byteSize);
          ffi.mgpuReadBufferAsyncDouble(
            _self,
            nativePtr,
            byteSize,
            effectiveByteOffset,
            nativeCallable.nativeFunction,
          );
          await completer.future;
          final List<double> data = nativePtr.asTypedList(sizeToRead);
          if (outputData is Float64List) {
            outputData.setAll(0, data);
          }
          malloc.free(nativePtr);
        }
        break;
    }
    nativeCallable.close();
    return;
  }

  @override
  void setData(
    TypedData inputData,
    int size, {
    BufferDataType dataType = BufferDataType.float,
  }) {
    // Determine element size based on data type.
    final int elementSize;
    switch (dataType) {
      case BufferDataType.int8:
        elementSize = sizeOf<Int8>();
        break;
      case BufferDataType.int16:
        elementSize = sizeOf<Int16>();
        break;
      case BufferDataType.int32:
        elementSize = sizeOf<Int32>();
        break;
      case BufferDataType.int64:
        elementSize = sizeOf<Int64>();
        break;
      case BufferDataType.uint8:
        elementSize = sizeOf<Uint8>();
        break;
      case BufferDataType.uint16:
        elementSize = sizeOf<Uint16>();
        break;
      case BufferDataType.uint32:
        elementSize = sizeOf<Uint32>();
        break;
      case BufferDataType.uint64:
        elementSize = sizeOf<Uint64>();
        break;
      case BufferDataType.float:
        elementSize = sizeOf<Float>();
        break;
      case BufferDataType.double:
        elementSize = sizeOf<Double>();
        break;
    }

    final int byteSize = size * elementSize;

    switch (dataType) {
      case BufferDataType.int8:
        {
          final Pointer<Int8> nativePtr = malloc.allocate<Int8>(byteSize);
          final List<int> data = (inputData as Int8List).toList();
          nativePtr.asTypedList(byteSize).setAll(0, data);
          ffi.mgpuSetBufferDataInt8(_self, nativePtr, byteSize);
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.int16:
        {
          final Pointer<Int16> nativePtr = malloc.allocate<Int16>(byteSize);
          final List<int> data = (inputData as Int16List).toList();
          nativePtr.asTypedList(size).setAll(0, data);
          ffi.mgpuSetBufferDataInt16(_self, nativePtr, byteSize);
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.int32:
        {
          final Pointer<Int32> nativePtr = malloc.allocate<Int32>(byteSize);
          final List<int> data = (inputData as Int32List).toList();
          nativePtr.asTypedList(size).setAll(0, data);
          ffi.mgpuSetBufferDataInt32(_self, nativePtr, byteSize);
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.int64:
        {
          final Pointer<Int64> nativePtr = malloc.allocate<Int64>(byteSize);
          final List<int> data = (inputData as Int64List).toList();
          nativePtr.asTypedList(size).setAll(0, data);
          ffi.mgpuSetBufferDataInt64(_self, nativePtr, byteSize);
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.uint8:
        {
          final Pointer<Uint8> nativePtr = malloc.allocate<Uint8>(byteSize);
          final List<int> data = (inputData as Uint8List).toList();
          nativePtr.asTypedList(byteSize).setAll(0, data);
          ffi.mgpuSetBufferDataUint8(_self, nativePtr, byteSize);
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.uint16:
        {
          final Pointer<Uint16> nativePtr = malloc.allocate<Uint16>(byteSize);
          final List<int> data = (inputData as Uint16List).toList();
          nativePtr.asTypedList(size).setAll(0, data);
          ffi.mgpuSetBufferDataUint16(_self, nativePtr, byteSize);
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.uint32:
        {
          final Pointer<Uint32> nativePtr = malloc.allocate<Uint32>(byteSize);
          final List<int> data = (inputData as Uint32List).toList();
          nativePtr.asTypedList(size).setAll(0, data);
          ffi.mgpuSetBufferDataUint32(_self, nativePtr, byteSize);
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.uint64:
        {
          final Pointer<Uint64> nativePtr = malloc.allocate<Uint64>(byteSize);
          final List<int> data = (inputData as Uint64List).toList();
          nativePtr.asTypedList(size).setAll(0, data);
          ffi.mgpuSetBufferDataUint64(_self, nativePtr, byteSize);
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.float:
        {
          final Pointer<Float> nativePtr = malloc.allocate<Float>(byteSize);
          final List<double> data = (inputData as Float32List).toList();
          nativePtr.asTypedList(size).setAll(0, data);
          ffi.mgpuSetBufferDataFloat(_self, nativePtr, byteSize);
          malloc.free(nativePtr);
        }
        break;
      case BufferDataType.double:
        {
          final Pointer<Double> nativePtr = malloc.allocate<Double>(byteSize);
          final List<double> data = (inputData as Float64List).toList();
          nativePtr.asTypedList(size).setAll(0, data);
          ffi.mgpuSetBufferDataDouble(_self, nativePtr, byteSize);
          malloc.free(nativePtr);
        }
        break;
    }
  }

  @override
  void destroy() {
    ffi.mgpuDestroyBuffer(_self);
  }
}
