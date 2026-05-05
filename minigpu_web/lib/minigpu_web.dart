import 'dart:async';
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:minigpu_web/bindings/minigpu_bindings.dart' as wasm;

MinigpuPlatform registeredInstance() => MinigpuWeb._();

class MinigpuWeb extends MinigpuPlatform {
  MinigpuWeb._();

  static void registerWith(dynamic _) => MinigpuWeb._();

  /// Creates an instance for use in tests. Equivalent to the private
  /// constructor but accessible from test code.
  factory MinigpuWeb.createForTest() => MinigpuWeb._();

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

  @override
  bool isExternalContentTypeSupported(ExternalContentType type) =>
      type == ExternalContentType.webVideoFrame;

  @override
  bool isExternalPixelFormatSupported(ExternalPixelFormat format) =>
      // GPUExternalTexture is opaque — the browser handles all pixel formats
      format != ExternalPixelFormat.unknown;

  @override
  PlatformVideoTexture? importVideoFrame(ExternalVideoBuffer buf) {
    if (buf.contentType != ExternalContentType.webVideoFrame) return null;
    // Web callers use importVideoFrameWeb() directly to pass a JSAny VideoFrame.
    // This path is a no-op to satisfy the platform interface contract.
    return null;
  }

  /// Web-specific: import a VideoFrame (WebCodecs) as a GPUExternalTexture.
  /// [videoFrame] must be a [JSAny] pointing to a VideoFrame JS object.
  WebVideoTexture? importVideoFrameWeb(
    JSAny videoFrame,
    ExternalPixelFormat pixelFormat,
    int width,
    int height,
  ) {
    final tex = wasm.mgpuImportExternalTexture(videoFrame);
    if (tex == null) return null;
    return WebVideoTexture(
      externalTexture: tex,
      pixelFormat: pixelFormat,
      width: width,
      height: height,
    );
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

  /// Store a GPUExternalTexture (from a VideoFrame import) at [slot].
  /// This is a Web-only method used by [WebVideoTexture.setOnShader].
  final _externalTextures = <int, JSObject>{};

  void setExternalTexture(int slot, JSObject texture) {
    _externalTextures[slot] = texture;
  }

  /// Returns all stored external textures (keyed by slot) for use in bind groups.
  Map<int, JSObject> get externalTextures =>
      Map.unmodifiable(_externalTextures);
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
        await wasm.mgpuReadAsyncInt8(
          _buffer,
          outputData as Int8List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.int16:
        await wasm.mgpuReadAsyncInt16(
          _buffer,
          outputData as Int16List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.int32:
        await wasm.mgpuReadAsyncInt32(
          _buffer,
          outputData as Int32List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.int64:
        await wasm.mgpuReadAsyncInt64(
          _buffer,
          outputData is ByteData
              ? outputData
              : (outputData.buffer.asByteData()),
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.uint8:
        await wasm.mgpuReadAsyncUint8(
          _buffer,
          outputData as Uint8List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.uint16:
        await wasm.mgpuReadAsyncUint16(
          _buffer,
          outputData as Uint16List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.uint32:
        await wasm.mgpuReadAsyncUint32(
          _buffer,
          outputData as Uint32List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.uint64:
        await wasm.mgpuReadAsyncUint64(
          _buffer,
          outputData as Uint64List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.float16:
        throw UnimplementedError('float16 is not supported in WebAssembly.');
      case BufferDataType.float32:
        await wasm.mgpuReadAsyncFloat(
          _buffer,
          outputData as Float32List,
          readElements: readElements,
          elementOffset: elementOffset,
        );

        break;
      case BufferDataType.float64:
        await wasm.mgpuReadAsyncDouble(
          _buffer,
          outputData as Float64List,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
    }
  }

  @override
  Future<void> write(
    TypedData inputData,
    int size, {
    BufferDataType dataType = BufferDataType.float32,
  }) async {
    if (inputData.elementSizeInBytes != dataType.bytesPerElement) {
      return;
    }

    switch (dataType) {
      case BufferDataType.int8:
        if (inputData is! Int8List) {
          break;
        }
        wasm.mgpuWriteInt8(_buffer, inputData, size);
        break;
      case BufferDataType.int16:
        if (inputData is! Int16List) {
          break;
        }
        wasm.mgpuWriteInt16(_buffer, inputData as Int16List, size);
        break;
      case BufferDataType.int32:
        if (inputData is! Int32List) {
          break;
        }
        wasm.mgpuWriteInt32(_buffer, inputData as Int32List, size);
        break;
      case BufferDataType.int64:
        if (inputData is! Int64List && inputData is! ByteData) {
          break;
        }
        wasm.mgpuWriteInt64(_buffer, inputData as Int64List, size);
        break;
      case BufferDataType.uint8:
        if (inputData is! Uint8List) {
          break;
        }
        wasm.mgpuWriteUint8(_buffer, inputData as Uint8List, size);
        break;
      case BufferDataType.uint16:
        if (inputData is! Uint16List) {
          break;
        }
        wasm.mgpuWriteUint16(_buffer, inputData as Uint16List, size);
        break;
      case BufferDataType.uint32:
        if (inputData is! Uint32List) {
          break;
        }
        wasm.mgpuWriteUint32(_buffer, inputData as Uint32List, size);
        break;
      case BufferDataType.uint64:
        if (inputData is! Uint64List && inputData is! ByteData) {
          break;
        }
        wasm.mgpuWriteUint64(_buffer, inputData as Uint64List, size);
        break;
      case BufferDataType.float16:
        throw UnimplementedError('float16 is not supported in WebAssembly.');
      case BufferDataType.float32:
        if (inputData is! Float32List) {
          break;
        }
        await wasm.mgpuWriteFloat(_buffer, inputData as Float32List, size);
        break;
      case BufferDataType.float64:
        if (inputData is! Float64List) {
          break;
        }
        wasm.mgpuWriteDouble(_buffer, inputData as Float64List, size);
        break;
    }
  }

  @override
  void destroy() {
    wasm.mgpuDestroyBuffer(_buffer);
  }
}

// ---------------------------------------------------------------------------
// Web VideoFrame import � wraps GPUExternalTexture via JS importExternalTexture
// ---------------------------------------------------------------------------
class WebVideoTexture implements PlatformVideoTexture {
  WebVideoTexture({
    required JSObject externalTexture,
    required this.pixelFormat,
    required this.width,
    required this.height,
  }) : _externalTexture = externalTexture;

  final JSObject _externalTexture;

  /// The underlying GPUExternalTexture (valid for current task only per WebGPU spec).
  JSObject get externalTexture => _externalTexture;

  @override
  final ExternalPixelFormat pixelFormat;
  @override
  final int width;
  @override
  final int height;

  @override
  int get numPlanes => 1; // GPUExternalTexture is always single-plane on Web

  @override
  void setOnShader(PlatformComputeShader shader, int slot, int planeIndex) {
    // Web: pass the external texture to the WebComputeShader via a custom method.
    // The shader implementation must call device.setBindGroup with the external texture.
    // For now we store the texture on the shader via the JS interop tag mechanism.
    if (shader is WebComputeShader) {
      shader.setExternalTexture(slot, _externalTexture);
    } else {
      throw UnsupportedError('setOnShader requires WebComputeShader on Web');
    }
  }

  @override
  PlatformBuffer toRGBA() => throw UnsupportedError(
    'toRGBA() is not supported for Web VideoFrames. '
    'Read back via WebCodecs VideoFrame.copyTo() instead.',
  );

  /// D3D11 shared-output path is Windows-only; always returns false on Web.
  @override
  bool bgraToRgbaSharedOutput(PlatformSharedOutputTexture dst) => false;

  @override
  void destroy() {
    // GPUExternalTexture lifetime is managed by the browser; no explicit destroy.
  }
}
