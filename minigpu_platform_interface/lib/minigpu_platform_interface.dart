import 'dart:typed_data';

import 'platform_stub/minigpu_platform_stub.dart'
    if (dart.library.ffi) 'package:minigpu_ffi/minigpu_ffi.dart'
    if (dart.library.js) 'package:minigpu_web/minigpu_web.dart';

/// Enum representing supported buffer data types.
enum BufferDataType {
  float16,
  float32, // 0
  float64, // 1
  int8, // 2
  int16, // 3
  int32, // 4
  int64, // 5
  uint8, // 6
  uint16, // 7
  uint32, // 8
  uint64, // 9
}

extension BufferDataTypeExtension on BufferDataType {
  /// Returns the size in bytes for this data type
  int get bytesPerElement {
    switch (this) {
      case BufferDataType.int8:
      case BufferDataType.uint8:
        return 1;
      case BufferDataType.float16:
      case BufferDataType.int16:
      case BufferDataType.uint16:
        return 2;
      case BufferDataType.float32:
      case BufferDataType.int32:
      case BufferDataType.uint32:
        return 4;
      case BufferDataType.float64:
      case BufferDataType.int64:
      case BufferDataType.uint64:
        return 8;
    }
  }

  /// Returns whether this type is signed
  bool get isSigned {
    switch (this) {
      case BufferDataType.int8:
      case BufferDataType.int16:
      case BufferDataType.int32:
      case BufferDataType.int64:
      case BufferDataType.float16:
      case BufferDataType.float32:
      case BufferDataType.float64:
        return true;
      case BufferDataType.uint8:
      case BufferDataType.uint16:
      case BufferDataType.uint32:
      case BufferDataType.uint64:
        return false;
    }
  }

  /// Returns whether this type is a floating point type
  bool get isFloatingPoint {
    switch (this) {
      case BufferDataType.float16:
      case BufferDataType.float32:
      case BufferDataType.float64:
        return true;
      default:
        return false;
    }
  }
}

String getWGSLType(BufferDataType type) {
  switch (type) {
    case BufferDataType.int8:
    case BufferDataType.int16:
    case BufferDataType.int32:
    case BufferDataType.int64: // Packed as i32
      return 'i32';
    case BufferDataType.uint8:
    case BufferDataType.uint16:
    case BufferDataType.uint32:
    case BufferDataType.uint64: // Packed as u32
      return 'u32';
    case BufferDataType.float16:
    case BufferDataType.float32:
      return 'f32';
    case BufferDataType.float64:
      return 'u32';
  }
}

int getBufferSizeForType(BufferDataType type, int count) {
  switch (type) {
    case BufferDataType.float16:
      return count * (Float32List.bytesPerElement / 2).toInt();
    case BufferDataType.float32:
      return count * Float32List.bytesPerElement;
    case BufferDataType.float64:
      return count * Float64List.bytesPerElement;
    case BufferDataType.int32:
      return count * Int32List.bytesPerElement;
    case BufferDataType.int64:
      return count * Int64List.bytesPerElement;
    case BufferDataType.int8:
      return count * Int8List.bytesPerElement;
    case BufferDataType.uint8:
      return count * Uint8List.bytesPerElement;
    case BufferDataType.int16:
      return count * Int16List.bytesPerElement;
    case BufferDataType.uint16:
      return count * Uint16List.bytesPerElement;
    case BufferDataType.uint32:
      return count * Uint32List.bytesPerElement;
    case BufferDataType.uint64:
      return count * Uint64List.bytesPerElement;
  }
}

/// A hardware GPU adapter as reported by the OS (DXGI on Windows): its
/// device name, total dedicated VRAM, and current dedicated-VRAM usage
/// (-1 when the usage query is unavailable).
class GpuAdapterInfo {
  const GpuAdapterInfo({
    required this.name,
    required this.totalVramBytes,
    required this.usedVramBytes,
  });

  final String name;
  final int totalVramBytes;
  final int usedVramBytes;

  /// Free dedicated VRAM, or total when usage is unknown.
  int get freeVramBytes =>
      usedVramBytes >= 0 ? totalVramBytes - usedVramBytes : totalVramBytes;

  @override
  String toString() =>
      "GpuAdapterInfo('$name', ${totalVramBytes >> 20} MB total, "
      '${usedVramBytes >= 0 ? '${usedVramBytes >> 20} MB used' : 'usage n/a'})';
}

abstract class MinigpuPlatform {
  MinigpuPlatform();

  static MinigpuPlatform? _instance;

  /// Returns the current instance; creates if not yet initialized.
  static MinigpuPlatform get instance {
    _instance ??= registeredInstance();
    return _instance!;
  }

  MinigpuPlatform registerInstance() =>
      throw UnimplementedError('No platform implementation available.');

  Future<void> initializeContext();
  Future<void> destroyContext();
  PlatformComputeShader createComputeShader();
  PlatformBuffer createBuffer(int bufferSize, BufferDataType dataType);

  /// Install a log callback for the native minigpu/Dawn layer.
  ///
  /// [callback] receives `(int level, String message)` where level matches
  /// `mgpu::LogLevel`: 0=DEBUG 1=INFO 2=WARN 3=ERROR.
  /// Pass `null` to revert to the default (native stderr) output.
  ///
  /// [level] sets the minimum verbosity; messages below this level are
  /// suppressed before the callback is invoked. -1 silences everything.
  void setLogCallback(
    void Function(int level, String message)? callback, {
    int level = 1,
  }) {}

  /// Returns dedicated VRAM usage in bytes for the primary GPU.
  /// Returns -1 on platforms where the query is unavailable.
  int queryVramBytes() => -1;

  /// Enumerates hardware adapters with their dedicated-VRAM totals and
  /// current usage.  Empty on platforms without adapter enumeration (web).
  List<GpuAdapterInfo> listAdapters() => const [];

  // ---------------------------------------------------------------------------
  // Video texture interop (optional — returns null if unsupported)
  // ---------------------------------------------------------------------------

  /// Returns true if the given content type can be imported on this platform.
  bool isExternalContentTypeSupported(ExternalContentType type) => false;

  /// Returns true if the given pixel format can be imported on this platform.
  bool isExternalPixelFormatSupported(ExternalPixelFormat format) => false;

  /// Import a video frame into GPU texture memory.
  /// Returns null if unsupported or on failure.
  PlatformVideoTexture? importVideoFrame(ExternalVideoBuffer buf) => null;

  /// Create a cross-API shared output texture (Windows: D3D12<->D3D11
  /// zero-copy via NT shared handle). Returns null on unsupported platforms.
  PlatformSharedOutputTexture? createSharedOutputTexture(
    int width,
    int height,
  ) => null;

  /// Create a D3D11 device on the same DXGI adapter as Dawn's D3D12 device.
  /// Returns the ID3D11Device* address (non-zero) or 0 on failure.
  /// Windows-only; returns 0 on all other platforms.
  int createD3D11DeviceOnDawnAdapter() => 0;

  /// Pre-init hint (Windows): bind Dawn to the adapter driving the PRIMARY
  /// display so screen capture (Desktop Duplication / WGC), GPU processing
  /// and any D3D11 encoder created on Dawn's adapter share one GPU —
  /// same-adapter zero-copy import — even on multi-output hybrid systems
  /// where the discrete GPU also drives a monitor. The `MGPU_ADAPTER_NAME`
  /// env var still overrides.
  ///
  /// Must be called BEFORE the context is initialized (before any other
  /// minigpu use). Returns `true` when the hint was stored before init;
  /// `false` when the context was already initialized (the hint is kept for
  /// a future re-init but the live context is unchanged) or on platforms
  /// without adapter selection.
  bool preferDisplayAdapter([bool enable = true]) => false;

  /// Name of the adapter Dawn actually selected, or `null` when the context
  /// is not initialized / the platform does not expose it.
  String? get selectedAdapterName => null;

  /// Yield-spin budget (ms) the LOADED native binary implements before the
  /// event drain degrades to coarse sleeping, or `null` where it can't be
  /// asked (web, or a binary predating the export).
  ///
  /// Purpose is runtime provenance: a build WITHOUT the drain fix burns a full
  /// Windows timer quantum (~15.6 ms) on every GPU wait, which shows up as a
  /// present cost that is identical at 720p and 4K. A caller on a latency path
  /// can assert this is > 0 instead of trusting that it linked a fresh
  /// artifact.
  int? get drainSpinBudgetMs => null;

  /// Creates an INDEPENDENT platform context bound to the adapter whose name
  /// contains [adapterFilter] (case-insensitive substring, e.g. "3090") —
  /// its own device, queue, and task FIFO.  Buffers and shaders created from
  /// the returned platform live on that adapter and must not be mixed with
  /// another context's resources in a single dispatch.  Returns null on
  /// platforms without multi-adapter support (web).
  MinigpuPlatform? createSecondaryPlatform(String adapterFilter) => null;
}

// ---------------------------------------------------------------------------
// External video buffer types (platform-neutral Dart layer)
// ---------------------------------------------------------------------------

enum ExternalContentType {
  cpu,
  d3d11SharedHandle,
  metalIOSurface,
  dmabuf,
  aHardwareBuffer,
  webVideoFrame,
}

enum ExternalPixelFormat {
  unknown,
  rgba32,
  bgra32,
  nv12,
  gray8,
  rgba64Half,
  yuv420pAsNV12Planes,
  yuv420pAsRGBPlanes,
}

class ExternalPlane {
  final int dataPtr; // Native pointer / fd as integer
  final int width;
  final int height;
  final int strideBytes;
  final int offsetBytes;
  final int subresourceIndex;
  final int dmabufFd;
  final int drmFormatModifier;

  const ExternalPlane({
    required this.dataPtr,
    required this.width,
    required this.height,
    required this.strideBytes,
    this.offsetBytes = 0,
    this.subresourceIndex = 0,
    this.dmabufFd = -1,
    this.drmFormatModifier = 0,
  });
}

class ExternalFence {
  final int syncFd; // -1 if none
  final int d3d11FencePtr; // 0 if none
  final int metalSharedEventPtr; // 0 if none
  final int metalFenceValue;

  const ExternalFence({
    this.syncFd = -1,
    this.d3d11FencePtr = 0,
    this.metalSharedEventPtr = 0,
    this.metalFenceValue = 0,
  });
}

class ExternalVideoBuffer {
  final ExternalContentType contentType;
  final ExternalPixelFormat pixelFormat;
  final int width;
  final int height;
  final List<ExternalPlane> planes;
  final ExternalFence fence;
  final int timestampUs;

  const ExternalVideoBuffer({
    required this.contentType,
    required this.pixelFormat,
    required this.width,
    required this.height,
    required this.planes,
    this.fence = const ExternalFence(),
    this.timestampUs = 0,
  });
}

/// Opaque handle to an imported video texture on the GPU.
abstract class PlatformVideoTexture {
  int get numPlanes;
  int get width;
  int get height;
  ExternalPixelFormat get pixelFormat;

  /// Bind a plane to a compute shader binding slot.
  void setOnShader(PlatformComputeShader shader, int slot, int planeIndex);

  /// Convert to RGBA8 via an internal compute pass.
  /// Returns a [PlatformBuffer] (RGBA8, row-major). Caller owns it.
  PlatformBuffer toRGBA();

  /// Convert a BGRA source video texture into the given cross-API shared
  /// output RGBA texture (zero-copy on Windows via D3D12/D3D11 shared NT
  /// handle). Returns true on success. Returns false if unsupported or on
  /// validation/dispatch failure.
  bool bgraToRgbaSharedOutput(PlatformSharedOutputTexture dst) => false;

  /// Async variant of [bgraToRgbaSharedOutput] that completes when the GPU
  /// work (including the cross-device present sync) has finished, without
  /// blocking the calling isolate on a busy-poll. The default implementation
  /// runs the synchronous [bgraToRgbaSharedOutput]; backends with a worker
  /// thread should override it to actually run off-thread.
  Future<bool> bgraToRgbaSharedOutputAsync(PlatformSharedOutputTexture dst) async =>
      bgraToRgbaSharedOutput(dst);

  void destroy();
}

/// Opaque handle to a cross-API shared output texture. On Windows this is a
/// D3D12 resource exported as an NT shared handle that an external API
/// (e.g. FFmpeg's D3D11 device) can open with `OpenSharedResource1`.
abstract class PlatformSharedOutputTexture {
  int get width;
  int get height;

  /// Native NT HANDLE (as integer) suitable for D3D11
  /// `OpenSharedResource1`. Caller MUST NOT call `CloseHandle` on it; it is
  /// owned by this object and will be closed in [destroy].
  int get d3d11Handle;

  /// Pointer (as integer) to the underlying `ID3D11Texture2D*` that backs
  /// this shared output. The texture lives on the same `ID3D11Device` as
  /// the one returned by [PlatformMinigpu.createD3D11DeviceOnDawnAdapter],
  /// so an FFmpeg encoder configured with that device may use this texture
  /// directly without `OpenSharedResource1`. The pointer's lifetime is
  /// owned by this object — do NOT `Release` it.
  int get d3d11TexturePtr => 0;

  /// Copy the contents of [src] (an RGBA8 GPU storage buffer, e.g. the output
  /// of a GpuEffect dispatch) into this shared texture on the GPU.
  /// Returns true on success, false if unsupported or on failure.
  bool copyFromBuffer(PlatformBuffer src) => false;

  /// Async variant of [copyFromBuffer] that completes when the GPU copy
  /// (including the cross-device present sync) has finished, without blocking
  /// the calling isolate on a busy-poll. The default implementation runs the
  /// synchronous [copyFromBuffer]; backends with a worker thread should override
  /// it to actually run off-thread.
  Future<bool> copyFromBufferAsync(PlatformBuffer src) async =>
      copyFromBuffer(src);

  /// Variant of [copyFromBuffer] for buffers that hold 4 f32 components per
  /// pixel (R,G,B,A in [0,1]) instead of packed RGBA8 u32.  Used by
  /// visualizers (e.g. the spectrogram) that produce float colors directly.
  bool copyFromBufferF32(PlatformBuffer src) => false;

  /// Debug-only: synchronously read the first pixel (BGRA8 packed u32) of the
  /// underlying D3D11 texture using the cached Dawn-adapter D3D11 device.
  /// Used to verify that Dawn writes are visible to the D3D11 consumer.
  /// Returns 0 if unsupported, or 0xDEAD000N codes on failure.
  int debugReadFirstPixel() => 0;

  /// Debug-only: read the first pixel via Dawn's CopyTextureToBuffer + map.
  int debugReadFirstPixelDawn() => 0;

  /// On web, returns the JS GPUTexture object from the Emscripten WebGPU
  /// object table. Null on all native platforms.
  Object? get webGpuTextureJs => null;

  void destroy();
}

abstract class PlatformComputeShader {
  void loadKernelString(String kernelString);
  bool hasKernel();
  void setBuffer(int tag, PlatformBuffer buffer);

  /// Ordered bind: the binding update joins the same FIFO as dispatches and
  /// buffer reads/writes, so bind→fire→rebind→fire sequences on one shader
  /// are race-free even with [dispatchFire].  Backends whose task execution
  /// is inherently in-order (single-threaded web) may alias [setBuffer].
  void setBufferFire(int tag, PlatformBuffer buffer) => setBuffer(tag, buffer);
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ);

  /// Fire-and-forget dispatch: enqueues the compute pass and returns without
  /// waiting for queue submission.  Ordering is still guaranteed — dispatches,
  /// writes and reads execute in call order — so a subsequent awaited
  /// [PlatformBuffer.read] acts as the synchronization point for any number
  /// of fired dispatches.
  ///
  /// SAFETY: the shader's bindings are snapshotted when the dispatch actually
  /// runs, not when this returns. Never call [setBuffer] on a shader with a
  /// fired-but-unsynchronized dispatch outstanding; give each call site its
  /// own shader instance with stable bindings instead.
  void dispatchFire(int groupsX, int groupsY, int groupsZ) {
    dispatch(groupsX, groupsY, groupsZ);
  }

  void destroy();
}

abstract class PlatformBuffer {
  Future<void> read(
    TypedData outputData,
    int readElements, {
    int elementOffset = 0,
    int readBytes = 0,
    int byteOffset = 0,
    BufferDataType dataType = BufferDataType.float32,
  });
  Future<void> write(
    TypedData inputData,
    int size, {
    BufferDataType dataType = BufferDataType.float32,
  });

  /// Raw byte upload of [bytes] at [dstByteOffset], intended for LARGE
  /// transfers: implementations should stream in bounded chunks so neither
  /// host scratch nor driver staging ever holds the full payload (a whole
  /// multi-GB write can otherwise spike/pin host RAM).  [dstByteOffset] and
  /// the byte length must be multiples of 4.  Default falls back to [write]
  /// when the offset is 0.
  Future<void> writeRawBytes(Uint8List bytes, {int dstByteOffset = 0}) {
    if (dstByteOffset != 0) {
      throw UnsupportedError('offset writeRawBytes not supported here');
    }
    final words = Uint32List(bytes.length ~/ 4);
    words.buffer.asUint8List().setRange(0, bytes.length, bytes);
    return write(words, words.length, dataType: BufferDataType.uint32);
  }

  void destroy();
}

final class MinigpuPlatformOutOfMemoryException implements Exception {
  @override
  String toString() => 'Out of memory';
}
