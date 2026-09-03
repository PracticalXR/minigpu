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

  /// Staged log level, applied the moment the wasm module is up. The host
  /// calls `Minigpu.setLogCallback(..., level: N)` at main() — long before the
  /// module loads on web — so the level must survive until it can land.
  int? _pendingLogLevel;

  @override
  void setLogCallback(
    void Function(int level, String message)? callback, {
    int level = 1,
  }) {
    // Only the LEVEL is honored on web: routing native log lines through a
    // Dart callback needs a wasm function-pointer table slot
    // (Emscripten addFunction), which this build does not reserve. Messages
    // at/above [level] go to the browser console via stderr, as before.
    _pendingLogLevel = level;
    if (wasm.isMinigpuModuleLoaded) wasm.mgpuSetLogLevel(level);
  }

  @override
  Future<void> initializeContext() async {
    // The level must land BETWEEN module load and context init, or the
    // adapter/device bring-up narrates at the wasm default (INFO).
    await wasm.ensureMinigpuModuleLoaded();
    final lvl = _pendingLogLevel;
    if (lvl != null) wasm.mgpuSetLogLevel(lvl);
    await wasm.mgpuInitializeContext();
    // 🔴 VERDICT PROBE for the asyncify early-resolution failure: the await
    // above can resolve while the wasm-side init is still suspended in its
    // adapter/device poll loop (observed: "Requesting WebGPU adapter..." is
    // the last init log, yet callers proceed). The device handle is a SYNC
    // export, so reading 0 here is proof the resolution was a lie — and the
    // page's every later async read will "resolve" with zeros the same way.
    final dev = wasm.mgpuGetWGPUDeviceHandle();
    if (dev == 0) {
      // ignore: avoid_print
      print('[minigpu_web] 🔴 initializeContext RESOLVED but the WGPUDevice '
          'handle is 0 — the wasm init never actually completed (asyncify '
          'early-resolution); all GPU work on this page will silently '
          'produce zeros');
    }
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

  /// 🔴 **THE IMPORT IS ONLY HALF THE ANSWER — THE CONSUMER HAS TO EXIST TOO.**
  /// This answered `true` for `webVideoFrame` on the strength of
  /// [importVideoFrameWeb] alone, while nothing could actually READ the
  /// resulting texture: the dispatch runs inside the wasm module, which builds
  /// its bind group from the WGSL declaration, and every landing kernel there
  /// declares `texture_2d<f32>` where a `GPUExternalTexture` requires
  /// `texture_external`. The pipeline failed validation, every dispatch after
  /// it was invalid, and — because WebGPU validation errors are UNCAPTURED and
  /// ASYNCHRONOUS — nothing threw. The staging call returned `true` while the
  /// GPU wrote nothing: a black screen under a completely healthy log.
  ///
  /// Both halves now exist. [WebVideoTexture.landPackedRgba8] runs the pass
  /// from Dart on minigpu's own device with a `texture_external` kernel, and
  /// writes into the same packed-RGBA8 buffer the native path targets, so the
  /// answer here is true again — and it is now a claim about the whole route,
  /// not just the import.
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
    // 🔴 THE GENERIC PATH WORKS ON WEB NOW. It used to `return null` here with
    // a note saying web callers should reach for `importVideoFrameWeb()`
    // directly — which meant every cross-platform caller had to grow a
    // conditional import to get a GPU texture on web, so none of them did, and
    // the web capture path stayed on a full-frame canvas readback while this
    // backend could import a VideoFrame the whole time.
    //
    // `ExternalVideoBuffer.externalHandle` is what closed that gap: a JS object
    // has no integer address to put in `planes[0].dataPtr`, so it rides as an
    // opaque `Object?` and is cast back here.
    final handle = buf.externalHandle;
    if (handle == null) return null;
    return importVideoFrameWeb(
      handle as JSAny,
      buf.pixelFormat,
      buf.width,
      buf.height,
    );
  }

  @override
  PlatformSharedOutputTexture? createSharedOutputTexture(
    int width,
    int height,
  ) {
    // On web we expose the GPU buffer handle directly to the view layer via
    // webGpuTextureJs.  There is no separate WGPUTexture allocation here —
    // the view plugin reads the buffer handle and blits via copyBufferToTexture
    // on the canvas device.  We use a lightweight wrapper that carries only
    // width/height and a reference to nothing until copyFromBufferF32 is called.
    return WebPlatformSharedOutputTexture(width, height);
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
  void setBufferFire(int tag, PlatformBuffer buffer) {
    // Single-threaded wasm executes GPU tasks in call order, so the plain
    // bind already has FIFO semantics.
    setBuffer(tag, buffer);
  }

  @override
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ) async {
    await wasm.mgpuDispatch(_shader, groupsX, groupsY, groupsZ);
  }

  /// One-shot marker: a fired dispatch that REJECTED. Static — one report
  /// per page is enough to name the failure.
  static bool _saidFireFailed = false;

  @override
  void dispatchFire(int groupsX, int groupsY, int groupsZ) {
    // queue.submit is synchronous in JS WebGPU; the returned promise only
    // covers call plumbing, so dropping it preserves submission order.
    //
    // 🔴 "Dropping it" must still HANDLE rejection: an uncaught C++ exception
    // in the wasm surfaces as a bare number (the exception pointer) on this
    // promise, and unhandled it kills the surrounding Dart zone with no
    // frames pointing anywhere — that is exactly the shape of an entire test
    // failing with just "67157776" and zone plumbing for a stack.
    wasm.mgpuDispatch(_shader, groupsX, groupsY, groupsZ).catchError((Object e) {
      if (!_saidFireFailed) {
        _saidFireFailed = true;
        // ignore: avoid_print
        print('[minigpu_web] fired dispatch($groupsX,$groupsY,$groupsZ) '
            'REJECTED: $e — a bare number is a wasm C++ exception pointer; '
            'raise the log level (Minigpu.setLogCallback level 0/1) to see '
            'the native narration up to the throw');
      }
    });
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
  int get webBufferHandle => wasm.mgpuGetWGPUBufferHandle(_buffer);

  @override
  Future<void> writeRawBytes(Uint8List bytes, {int dstByteOffset = 0}) {
    if (dstByteOffset != 0) {
      throw UnsupportedError('offset writeRawBytes not supported on web');
    }
    if (bytes.length % 4 != 0) {
      throw ArgumentError('writeRawBytes needs a 4-byte-aligned length');
    }
    // A VIEW over the caller's bytes, not a fresh list. This used to allocate a
    // `Uint32List` the size of the payload and memcpy into it on EVERY call —
    // on the streaming path that is a whole-frame allocation plus a whole-frame
    // copy per frame (33 MB at 4K), which is both the copy itself and a
    // per-frame garbage source large enough to show up as a frame-time spike.
    // A 4-byte-aligned view aliases the same store and costs nothing.
    //
    // The fallback is for the case a view cannot describe: a list whose
    // offsetInBytes is not 4-aligned (only possible for a caller-made view over
    // an odd offset). Then, and only then, a copy is unavoidable.
    final Uint32List words;
    if (bytes.offsetInBytes % 4 == 0) {
      words = bytes.buffer
          .asUint32List(bytes.offsetInBytes, bytes.lengthInBytes ~/ 4);
    } else {
      words = Uint32List(bytes.length ~/ 4);
      words.buffer.asUint8List().setRange(0, bytes.length, bytes);
    }
    return write(words, words.length, dataType: BufferDataType.uint32);
  }

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

  /// 🔴 **THIS DOES NOT BIND ANYTHING — USE [landPackedRgba8].** It stores the
  /// texture in a map nothing reads, because the wasm-side pipeline that would
  /// consume it declares `texture_2d<f32>` and cannot take an external texture
  /// at all. Kept only so the platform interface stays satisfied; a caller that
  /// reaches it has taken a route that silently produces no pixels, so it says
  /// so once rather than looking like plumbing.
  @override
  void setOnShader(PlatformComputeShader shader, int slot, int planeIndex) {
    if (shader is! WebComputeShader) {
      throw UnsupportedError('setOnShader requires WebComputeShader on Web');
    }
    if (!_saidStub) {
      _saidStub = true;
      // ignore: avoid_print
      print('[minigpu] 🔴 WebVideoTexture.setOnShader binds NOTHING on web — '
          'a GPUExternalTexture needs a texture_external kernel and a bind '
          'group built JS-side. Use landPackedRgba8() instead; this pass will '
          'produce no pixels.');
    }
    shader.setExternalTexture(slot, _externalTexture);
  }

  static bool _saidStub = false;

  /// 🔴 Always true: a `GPUExternalTexture` has no other consumer here.
  @override
  bool get requiresExternalLanding => true;

  /// Land this external texture into [dst] as packed RGBA8.
  ///
  /// The web answer to "bind a video texture to a compute pass": the pass runs
  /// from Dart, on minigpu's own device, with a kernel declared
  /// `texture_external` — the only form that can read one.
  @override
  bool landPackedRgba8(
    PlatformBuffer dst, {
    required int width,
    required int height,
    int downscale = 1,
  }) {
    if (dst is! WebBuffer) return false;
    return wasm.mgpuLandExternalTexture(
      externalTexture: _externalTexture,
      dstBufferHandle: dst.webBufferHandle,
      width: width,
      height: height,
      downscale: downscale,
    );
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
  Future<bool> bgraToRgbaSharedOutputAsync(
    PlatformSharedOutputTexture dst,
  ) async => false;

  @override
  void destroy() {
    // GPUExternalTexture lifetime is managed by the browser; no explicit destroy.
  }
}

// ---------------------------------------------------------------------------
// Web shared-output texture — GPU buffer → JS GPUBuffer handle → canvas blit
// ---------------------------------------------------------------------------

/// Web implementation of [PlatformSharedOutputTexture].
///
/// On web there is no D3D12/D3D11 texture sharing.  Instead the Dart view
/// layer reads [webGpuTextureJs] to obtain the JS GPUBuffer object for the
/// last buffer written via [copyFromBufferF32], then blits it to the canvas
/// using `device.queue.copyBufferToTexture` (handled by `minigpu_view_web`).
class WebPlatformSharedOutputTexture implements PlatformSharedOutputTexture {
  WebPlatformSharedOutputTexture(this._width, this._height);

  final int _width;
  final int _height;

  /// The last buffer that was passed to [copyFromBufferF32].
  /// Its WGPUBuffer handle is exposed as [webGpuTextureJs].
  wasm.MGPUBuffer? _lastBuffer;

  @override
  int get width => _width;

  @override
  int get height => _height;

  @override
  int get d3d11Handle => 0;

  @override
  int get d3d11TexturePtr => 0;

  @override
  bool copyFromBuffer(PlatformBuffer src) => false;

  @override
  Future<bool> copyFromBufferAsync(PlatformBuffer src) async => false;

  @override
  bool copyFromBufferF32(PlatformBuffer src) {
    if (src is! WebBuffer) return false;
    // Store the buffer reference so that webGpuTextureJs can return the
    // correct JS object when asPreviewSource() is called immediately after.
    _lastBuffer = src._buffer;
    return true;
  }

  /// Returns the Emscripten integer handle for the WGPUBuffer last written
  /// via [copyFromBufferF32].  The view plugin resolves the JS GPUBuffer via
  /// `WebGPU.getJsObject(handle)` locally — integers are codec-safe.
  /// Returns 0 if no buffer has been written yet.
  @override
  Object? get webGpuTextureJs {
    final buf = _lastBuffer;
    if (buf == null) return null;
    final handle = wasm.mgpuGetWGPUBufferHandle(buf);
    return handle == 0 ? null : handle; // int, not JSObject — codec-safe
  }

  @override
  int debugReadFirstPixel() => 0;

  @override
  int debugReadFirstPixelDawn() => 0;

  @override
  void destroy() {
    _lastBuffer = null;
  }
}
