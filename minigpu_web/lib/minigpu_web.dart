import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:minigpu_web/bindings/minigpu_bindings.dart' as wasm;

MinigpuPlatform registeredInstance() => MinigpuWeb._();

class MinigpuWeb extends MinigpuPlatform {
  MinigpuWeb._();

  static void registerWith(dynamic _) => MinigpuWeb._();

  @override
  Future<void> initializeContext() async {
    await wasm.mgpuInitializeContext();
  }

  @override
  Future<void> destroyContext() async {
    await wasm.mgpuDestroyContext();
  }

  @override
  PlatformComputeShader createComputeShader() {
    final shader = wasm.mgpuCreateComputeShader();
    return WebComputeShader(shader);
  }

  @override
  PlatformBuffer createBuffer(int bufferSize, BufferDataType dataType) {
    final buff = wasm.mgpuCreateBuffer(bufferSize, dataType.index);
    return WebBuffer(buff);
  }
}

class WebComputeShader implements PlatformComputeShader {
  final wasm.MGPUComputeShader _shader;

  WebComputeShader(this._shader);

  @override
  void loadKernelString(String kernelString) {
    wasm.mgpuLoadKernel(_shader, kernelString);
  }

  @override
  bool hasKernel() {
    return wasm.mgpuHasKernel(_shader);
  }

  @override
  void setBuffer(int tag, PlatformBuffer buffer) {
    // Updated: Pass the shader pointer as first argument
    wasm.mgpuSetBuffer(_shader, tag, (buffer as WebBuffer)._buffer);
  }

  @override
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ) async {
    await wasm.mgpuDispatch(_shader, groupsX, groupsY, groupsZ);
  }

  @override
  void destroy() {
    wasm.mgpuDestroyComputeShader(_shader);
  }
}

class WebBuffer implements PlatformBuffer {
  final wasm.MGPUBuffer _buffer;

  WebBuffer(this._buffer);

  @override
  Future<void> read(
    TypedData outputData,
    int readElements, {
    int elementOffset = 0,
    int readBytes = 0,
    int byteOffset = 0,
    BufferDataType dataType = BufferDataType.float32,
  }) async {
    switch (dataType) {
      case BufferDataType.int8:
        await wasm.mgpuReadBufferAsyncInt8(
          _buffer,
          outputData as Int8List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.int16:
        await wasm.mgpuReadBufferAsyncInt16(
          _buffer,
          outputData as Int16List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.int32:
        await wasm.mgpuReadBufferAsyncInt32(
          _buffer,
          outputData as Int32List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.int64:
        await wasm.mgpuReadBufferAsyncInt64(
          _buffer,
          outputData is ByteData
              ? outputData
              : (outputData.buffer.asByteData()),
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.uint8:
        await wasm.mgpuReadBufferAsyncUint8(
          _buffer,
          outputData as Uint8List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.uint16:
        await wasm.mgpuReadBufferAsyncUint16(
          _buffer,
          outputData as Uint16List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.uint32:
        await wasm.mgpuReadBufferAsyncUint32(
          _buffer,
          outputData as Uint32List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.uint64:
        await wasm.mgpuReadBufferAsyncUint64(
          _buffer,
          outputData as Uint64List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.float16:
        throw UnimplementedError(
          'float16 is not supported in WebAssembly.',
        );
      case BufferDataType.float32:
        await wasm.mgpuReadBufferAsyncFloat(
          _buffer,
          outputData as Float32List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.float64:
        await wasm.mgpuReadBufferAsyncDouble(
          _buffer,
          outputData as Float64List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
    }
  }

  @override
  void setData(
    TypedData inputData,
    int size, {
    BufferDataType dataType = BufferDataType.float32,
  }) {
    if (inputData.elementSizeInBytes != dataType.bytesPerElement) {
      return;
    }

    switch (dataType) {
      case BufferDataType.int8:
        if (inputData is! Int8List) {
          break;
        }
        wasm.mgpuSetBufferDataInt8(_buffer, inputData, size);
        break;
      case BufferDataType.int16:
        if (inputData is! Int16List) {
          break;
        }
        wasm.mgpuSetBufferDataInt16(_buffer, inputData as Int16List, size);
        break;
      case BufferDataType.int32:
        if (inputData is! Int32List) {
          break;
        }
        wasm.mgpuSetBufferDataInt32(_buffer, inputData as Int32List, size);
        break;
      case BufferDataType.int64:
        if (inputData is! Int64List && inputData is! ByteData) {
          break;
        }
        wasm.mgpuSetBufferDataInt64(_buffer, inputData as Int64List, size);
        break;
      case BufferDataType.uint8:
        if (inputData is! Uint8List) {
          break;
        }
        wasm.mgpuSetBufferDataUint8(_buffer, inputData as Uint8List, size);
        break;
      case BufferDataType.uint16:
        if (inputData is! Uint16List) {
          break;
        }
        wasm.mgpuSetBufferDataUint16(_buffer, inputData as Uint16List, size);
        break;
      case BufferDataType.uint32:
        if (inputData is! Uint32List) {
          break;
        }
        wasm.mgpuSetBufferDataUint32(_buffer, inputData as Uint32List, size);
        break;
      case BufferDataType.uint64:
        if (inputData is! Uint64List && inputData is! ByteData) {
          break;
        }
        wasm.mgpuSetBufferDataUint64(_buffer, inputData as Uint64List, size);
        break;
      case BufferDataType.float16:
        throw UnimplementedError(
          'float16 is not supported in WebAssembly.',
        );
      case BufferDataType.float32:
        if (inputData is! Float32List) {
          break;
        }
        wasm.mgpuSetBufferDataFloat(_buffer, inputData as Float32List, size);
        break;
      case BufferDataType.float64:
        if (inputData is! Float64List) {
          break;
        }
        wasm.mgpuSetBufferDataDouble(_buffer, inputData as Float64List, size);
        break;
    }
  }

  @override
  void destroy() {
    wasm.mgpuDestroyBuffer(_buffer);
  }
}
