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
  Future<void> destroyContext() async {
    ffi.mgpuDestroyContext();
  }

  @override
  PlatformComputeShader createComputeShader() {
    final self = ffi.mgpuCreateComputeShader();
    if (self == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return FfiComputeShader(self);
  }

  @override
  PlatformBuffer createBuffer(
    int bufferSize,
    BufferDataType dataType,
  ) {
    final self = ffi.mgpuCreateBuffer(bufferSize, dataType.index);
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
    BufferDataType dataType = BufferDataType.float32,
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
      case BufferDataType.float16:
        elementSize = (sizeOf<Float>() / 2).toInt();
        break; // Approx
      case BufferDataType.float32:
        elementSize = sizeOf<Float>();
        break;
      case BufferDataType.float64:
        elementSize = sizeOf<Double>();
        break;
    }
    if (elementSize == 0) {
      throw ArgumentError('Unsupported BufferDataType for read: $dataType');
    }

    // Calculate the number of elements available in the output buffer
    final int totalElementsInOutput = outputData.lengthInBytes ~/ elementSize;

    // Determine the number of elements to actually read
    final int elementsToRead =
        (readElements > 0) ? readElements : totalElementsInOutput;

    // --- Input Validation ---
    if (elementOffset < 0) {
      throw RangeError.value(
          elementOffset, 'elementOffset', 'Cannot be negative');
    }
    if (elementsToRead < 0) {
      throw RangeError.value(
          elementsToRead, 'readElements', 'Cannot be negative');
    }
    // Check if requested range is valid within the output buffer
    if (elementsToRead > totalElementsInOutput) {
      throw RangeError(
          'Read range (offset: $elementOffset, count: $elementsToRead) exceeds output buffer capacity ($totalElementsInOutput elements)');
    }
    // --- End Input Validation ---

    if (elementsToRead == 0) {
      // Nothing to read
      return;
    }

    final completer = Completer<void>();
    void nativeCallback() {
      completer.complete();
    }

    final nativeCallable =
        NativeCallable<Void Function()>.listener(nativeCallback);

    // Allocate temporary native memory based on the number of elements to read
    final int bytesToAllocate = elementsToRead * elementSize;
    final Pointer<NativeType> nativePtr =
        malloc.allocate<NativeType>(bytesToAllocate);

    try {
      // Switch to call the proper native function, passing ELEMENT counts/offsets
      switch (dataType) {
        case BufferDataType.int8:
          {
            ffi.mgpuReadBufferAsyncInt8(
              _self,
              nativePtr.cast<Int8>(),
              elementsToRead,
              elementOffset,
              nativeCallable.nativeFunction,
            );
            await completer.future;
            final List<int> data =
                nativePtr.cast<Int8>().asTypedList(elementsToRead);
            // Copy data into the correct portion of the outputData
            if (outputData is Int8List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setInt8(startByte + i * elementSize, data[i]);
              }
            } else {/* Handle other potential TypedData types if needed */}
          }
          break;
        case BufferDataType.int16:
          {
            ffi.mgpuReadBufferAsyncInt16(
              _self,
              nativePtr.cast<Int16>(),
              elementsToRead, // Pass ELEMENT count
              elementOffset, // Pass ELEMENT offset
              nativeCallable.nativeFunction,
            );
            await completer.future;
            final List<int> data =
                nativePtr.cast<Int16>().asTypedList(elementsToRead);
            if (outputData is Int16List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setInt16(
                    startByte + i * elementSize, data[i], Endian.host);
              }
            }
          }
          break;
        case BufferDataType.int32:
          {
            ffi.mgpuReadBufferAsyncInt32(
              _self,
              nativePtr.cast<Int32>(),
              elementsToRead, // Pass ELEMENT count
              elementOffset, // Pass ELEMENT offset
              nativeCallable.nativeFunction,
            );
            await completer.future;
            final List<int> data =
                nativePtr.cast<Int32>().asTypedList(elementsToRead);
            if (outputData is Int32List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setInt32(
                    startByte + i * elementSize, data[i], Endian.host);
              }
            }
          }
          break;
        case BufferDataType.int64:
          {
            ffi.mgpuReadBufferAsyncInt64(
              _self,
              nativePtr.cast<Int64>(),
              elementsToRead, // Pass ELEMENT count
              elementOffset, // Pass ELEMENT offset
              nativeCallable.nativeFunction,
            );
            await completer.future;
            final List<int> data =
                nativePtr.cast<Int64>().asTypedList(elementsToRead);
            if (outputData is Int64List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setInt64(
                    startByte + i * elementSize, data[i], Endian.host);
              }
            }
          }
          break;
        case BufferDataType.uint8:
          {
            ffi.mgpuReadBufferAsyncUint8(
              _self,
              nativePtr.cast<Uint8>(),
              elementsToRead, // ELEMENT count
              elementOffset, // ELEMENT offset
              nativeCallable.nativeFunction,
            );
            await completer.future;
            final List<int> data =
                nativePtr.cast<Uint8>().asTypedList(elementsToRead);
            if (outputData is Uint8List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setUint8(startByte + i * elementSize, data[i]);
              }
            }
          }
          break;
        case BufferDataType.uint16:
          {
            ffi.mgpuReadBufferAsyncUint16(
              _self,
              nativePtr.cast<Uint16>(),
              elementsToRead, // Pass ELEMENT count
              elementOffset, // Pass ELEMENT offset
              nativeCallable.nativeFunction,
            );
            await completer.future;
            final List<int> data =
                nativePtr.cast<Uint16>().asTypedList(elementsToRead);
            if (outputData is Uint16List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setUint16(
                    startByte + i * elementSize, data[i], Endian.host);
              }
            }
          }
          break;
        case BufferDataType.uint32:
          {
            ffi.mgpuReadBufferAsyncUint32(
              _self,
              nativePtr.cast<Uint32>(),
              elementsToRead, // Pass ELEMENT count
              elementOffset, // Pass ELEMENT offset
              nativeCallable.nativeFunction,
            );
            await completer.future;
            final List<int> data =
                nativePtr.cast<Uint32>().asTypedList(elementsToRead);
            if (outputData is Uint32List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setUint32(
                    startByte + i * elementSize, data[i], Endian.host);
              }
            }
          }
          break;
        case BufferDataType.uint64:
          {
            ffi.mgpuReadBufferAsyncUint64(
              _self,
              nativePtr.cast<Uint64>(),
              elementsToRead, // ELEMENT count
              elementOffset, // ELEMENT offset
              nativeCallable.nativeFunction,
            );
            await completer.future;
            final List<int> data =
                nativePtr.cast<Uint64>().asTypedList(elementsToRead);
            if (outputData is Uint64List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setUint64(
                    startByte + i * elementSize, data[i], Endian.host);
              }
            }
          }
          break;
        case BufferDataType.float16:
          throw UnimplementedError(
              'BufferDataType.float16 read is not implemented yet.');
        case BufferDataType.float32:
          {
            ffi.mgpuReadBufferAsyncFloat(
              _self,
              nativePtr.cast<Float>(),
              elementsToRead, // ELEMENT count
              elementOffset, // ELEMENT offset
              nativeCallable.nativeFunction,
            );
            await completer.future;
            final List<double> data =
                nativePtr.cast<Float>().asTypedList(elementsToRead);
            if (outputData is Float32List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setFloat32(
                    startByte + i * elementSize, data[i], Endian.host);
              }
            }
          }
          break;
        case BufferDataType.float64:
          {
            ffi.mgpuReadBufferAsyncDouble(
              _self,
              nativePtr.cast<Double>(),
              elementsToRead, // Pass ELEMENT count
              elementOffset, // Pass ELEMENT offset
              nativeCallable.nativeFunction,
            );
            await completer.future;
            final List<double> data =
                nativePtr.cast<Double>().asTypedList(elementsToRead);
            if (outputData is Float64List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setFloat64(
                    startByte + i * elementSize, data[i], Endian.host);
              }
            }
          }
          break;
      }
    } finally {
      malloc.free(nativePtr);
      nativeCallable.close();
    }
  }

  @override
  void setData(
    TypedData inputData,
    int elementCount, {
    // Renamed parameter for clarity
    BufferDataType dataType = BufferDataType.float32,
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
      case BufferDataType.float16:
        elementSize = (sizeOf<Float>() / 2).toInt();
        break; // Approx
      case BufferDataType.float32:
        elementSize = sizeOf<Float>();
        break;
      case BufferDataType.float64:
        elementSize = sizeOf<Double>();
        break;
    }
    if (elementSize == 0) {
      throw ArgumentError('Unsupported BufferDataType for setData: $dataType');
    }

    // --- Input Validation ---
    final int totalElementsInInput = inputData.lengthInBytes ~/ elementSize;
    if (elementCount < 0) {
      throw RangeError.value(
          elementCount, 'elementCount', 'Cannot be negative');
    }
    if (elementCount > totalElementsInInput) {
      throw RangeError(
          'elementCount ($elementCount) exceeds input data capacity ($totalElementsInInput elements)');
    }
    // --- End Input Validation ---

    if (elementCount == 0) {
      // Allow setting zero elements (C++ should handle this)
    }

    final int byteSize = elementCount * elementSize;
    final Pointer<NativeType> nativePtr = (inputData is Int8List ||
            inputData is Uint8List)
        ? malloc.allocate<NativeType>((byteSize + 3) & ~3) // Align to 4 bytes
        : malloc.allocate<NativeType>(byteSize);

    try {
      // Copy data from inputData (up to elementCount) to nativePtr
      // Use efficient view/copy methods where possible
      if (inputData is Int8List && dataType == BufferDataType.int8) {
        nativePtr
            .cast<Int8>()
            .asTypedList(byteSize)
            .setRange(0, byteSize, inputData);
      } else if (inputData is Int16List && dataType == BufferDataType.int16) {
        nativePtr
            .cast<Int16>()
            .asTypedList(elementCount)
            .setRange(0, elementCount, inputData);
      } else if (inputData is Int32List && dataType == BufferDataType.int32) {
        nativePtr
            .cast<Int32>()
            .asTypedList(elementCount)
            .setRange(0, elementCount, inputData);
      } else if (inputData is Int64List && dataType == BufferDataType.int64) {
        nativePtr
            .cast<Int64>()
            .asTypedList(elementCount)
            .setRange(0, elementCount, inputData);
      } else if (inputData is Uint8List && dataType == BufferDataType.uint8) {
        nativePtr
            .cast<Uint8>()
            .asTypedList(byteSize)
            .setRange(0, byteSize, inputData);
      } else if (inputData is Uint16List && dataType == BufferDataType.uint16) {
        nativePtr
            .cast<Uint16>()
            .asTypedList(elementCount)
            .setRange(0, elementCount, inputData);
      } else if (inputData is Uint32List && dataType == BufferDataType.uint32) {
        nativePtr
            .cast<Uint32>()
            .asTypedList(elementCount)
            .setRange(0, elementCount, inputData);
      } else if (inputData is Uint64List && dataType == BufferDataType.uint64) {
        nativePtr
            .cast<Uint64>()
            .asTypedList(elementCount)
            .setRange(0, elementCount, inputData);
      } else if (inputData is Float32List &&
          dataType == BufferDataType.float32) {
        nativePtr
            .cast<Float>()
            .asTypedList(elementCount)
            .setRange(0, elementCount, inputData);
      } else if (inputData is Float64List &&
          dataType == BufferDataType.float64) {
        nativePtr
            .cast<Double>()
            .asTypedList(elementCount)
            .setRange(0, elementCount, inputData);
      } else {
        // Fallback using ByteData view (less efficient but handles generic TypedData)
        final inputBytes =
            ByteData.view(inputData.buffer, inputData.offsetInBytes, byteSize);
        final nativeBytes = nativePtr.cast<Uint8>().asTypedList(byteSize);
        for (int i = 0; i < byteSize; i++) {
          nativeBytes[i] = inputBytes.getUint8(i);
        }
      }

      // Switch to call the proper native function
      switch (dataType) {
        case BufferDataType.int8:
          ffi.mgpuSetBufferDataInt8(_self, nativePtr.cast<Int8>(), byteSize);
          break;
        case BufferDataType.int16:
          ffi.mgpuSetBufferDataInt16(_self, nativePtr.cast<Int16>(), byteSize);
          break;
        case BufferDataType.int32:
          ffi.mgpuSetBufferDataInt32(_self, nativePtr.cast<Int32>(), byteSize);
          break;
        case BufferDataType.int64:
          ffi.mgpuSetBufferDataInt64(_self, nativePtr.cast<Int64>(), byteSize);
          break;
        case BufferDataType.uint8:
          ffi.mgpuSetBufferDataUint8(_self, nativePtr.cast<Uint8>(), byteSize);
          break;
        case BufferDataType.uint16:
          ffi.mgpuSetBufferDataUint16(
              _self, nativePtr.cast<Uint16>(), byteSize);
          break;
        case BufferDataType.uint32:
          ffi.mgpuSetBufferDataUint32(
              _self, nativePtr.cast<Uint32>(), byteSize);
          break;
        case BufferDataType.uint64:
          ffi.mgpuSetBufferDataUint64(
              _self, nativePtr.cast<Uint64>(), byteSize);
          break;
        case BufferDataType.float16:
          throw UnimplementedError(
              'BufferDataType.float16 setData is not implemented yet.');
        case BufferDataType.float32:
          ffi.mgpuSetBufferDataFloat(_self, nativePtr.cast<Float>(), byteSize);
          break;
        case BufferDataType.float64:
          ffi.mgpuSetBufferDataDouble(
              _self, nativePtr.cast<Double>(), byteSize);
          break;
      }
    } finally {
      malloc.free(nativePtr);
    }
  }

  @override
  void destroy() {
    ffi.mgpuDestroyBuffer(_self);
  }
}
