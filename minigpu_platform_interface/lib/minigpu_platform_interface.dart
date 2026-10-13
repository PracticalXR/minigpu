import 'dart:typed_data';
import 'src/buffer_descriptor.dart';
import 'src/texture_descriptor.dart';
import 'src/render_descriptor.dart';

export 'src/buffer_descriptor.dart';
export 'src/texture_descriptor.dart';
export 'src/render_descriptor.dart';

import 'platform_stub/minigpu_platform_stub.dart'
    if (dart.library.ffi) 'package:minigpu_ffi/minigpu_ffi.dart'
    // 🔴 `dart.library.js_interop`, NOT `dart.library.js`.
    //
    // `dart:js` does not exist under dart2wasm — only `dart:js_interop` does —
    // so `if (dart.library.js)` is FALSE there and this falls all the way
    // through to the stub. The symptom is not a compile error: the app builds,
    // starts, and throws `UnsupportedError: No platform implementation
    // available.` out of a wasm frame with no Dart stack, which reads as a
    // broken wasm build rather than as an unselected implementation.
    //
    // `js_interop` is provided by dart2js AND dart2wasm and by neither the VM
    // nor AOT, so it selects exactly the same web implementation the old
    // condition did, and one more target besides.
    if (dart.library.js_interop) 'package:minigpu_web/minigpu_web.dart';

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

/// Exact local hardware/backend identity; names and model numbers are not
/// sufficient to distinguish identical adapters. Not a room/public identifier.
final class DeviceAdapterIdentity {
  const DeviceAdapterIdentity({
    required this.backend,
    required this.kind,
    required this.low,
    required this.high,
  });
  final int backend, kind, low, high;

  @override
  bool operator ==(Object other) => other is DeviceAdapterIdentity &&
      backend == other.backend && kind == other.kind &&
      low == other.low && high == other.high;
  @override
  int get hashCode => Object.hash(backend, kind, low, high);
}

/// Host-local replacement requirements, never room authorization or a device ID
/// that may be serialized. Unsupported platforms expose no target.
sealed class DeviceRecoveryTarget {
  const DeviceRecoveryTarget();
}

final class SameAdapterRecoveryTarget extends DeviceRecoveryTarget {
  const SameAdapterRecoveryTarget(this.adapter);
  final DeviceAdapterIdentity adapter;
  @override
  bool operator ==(Object other) =>
      other is SameAdapterRecoveryTarget && adapter == other.adapter;
  @override
  int get hashCode => adapter.hashCode;
}

/// A fresh browser-selected device must preserve these capabilities. Identity
/// is this exact backend-owned object, not adapter name/model or physical GPU.
final class BrowserRecoveryTarget extends DeviceRecoveryTarget {
  BrowserRecoveryTarget({required Iterable<String> features,
      required Map<String, int> limits})
      : features = Set.unmodifiable(features), limits = Map.unmodifiable(limits);
  final Set<String> features;
  final Map<String, int> limits;

  bool accepts(Set<String> offeredFeatures, Map<String, int> offeredLimits) {
    if (!offeredFeatures.containsAll(features)) return false;
    for (final entry in limits.entries) {
      final offered = offeredLimits[entry.key];
      if (offered == null) return false;
      final alignment = entry.key == 'minUniformBufferOffsetAlignment' ||
          entry.key == 'minStorageBufferOffsetAlignment';
      if (alignment ? offered > entry.value : offered < entry.value) return false;
    }
    return true;
  }
}

/// A failed replacement whose candidate is conclusively gone. Only a trusted
/// backend may issue this receipt, after final context retirement (or proof
/// that no candidate was allocated). An initialization error, a timeout or a
/// returned attachment alone is not evidence of retirement.
///
/// This certifies only this replacement attempt, never the original device's
/// consumers. The root must separately join all original retirement receipts.
class DeviceRecoveryDiscarded extends StateError {
  DeviceRecoveryDiscarded({required this.target, required String message})
      : super(message);
  final DeviceRecoveryTarget target;
}

/// Compatible native receipt; the physical-adapter contract is unchanged.
final class DeviceReplacementDiscarded extends DeviceRecoveryDiscarded {
  DeviceReplacementDiscarded({
    required this.expectedAdapter,
    required super.message,
  }) : super(target: SameAdapterRecoveryTarget(expectedAdapter));

  final DeviceAdapterIdentity expectedAdapter;
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

  /// Captured while attached; may remain available after observed loss.
  DeviceAdapterIdentity? get creationAdapterIdentity => null;

  /// Browser implementations return their exact local capability target;
  /// native implementations default to the existing physical-adapter contract.
  DeviceRecoveryTarget? get creationRecoveryTarget {
    final adapter = creationAdapterIdentity;
    return adapter == null ? null : SameAdapterRecoveryTarget(adapter);
  }

  /// Root-owned successor creation after final context and reader retirement.
  /// No browser fallback is inferred for a native physical-adapter request.
  Future<void> initializeContextForReplacement(DeviceRecoveryTarget expected) =>
      switch (expected) {
        SameAdapterRecoveryTarget(:final adapter) => initializeContextForRecovery(adapter),
        BrowserRecoveryTarget() => Future.error(
            UnsupportedError('Browser device replacement is unsupported')),
      };

  /// Called by the root only after final old-context retirement. A new context
  /// must match the exact physical adapter/backend before it becomes usable.
  /// Owned secondary contexts allocate a fresh native handle. Unsupported
  /// backends refuse; this must never choose a fallback adapter silently.
  /// A failure permits explicit host retry only when accompanied by a
  /// [DeviceReplacementDiscarded] receipt for [expected]. All other failures
  /// retain uncertain cleanup and prohibit replacement.
  Future<void> initializeContextForRecovery(DeviceAdapterIdentity expected) =>
      Future.error(UnsupportedError('Same-adapter recovery is unsupported'));

  /// Trusted resource-retirement operation for a previously observed lost
  /// creation device. Success means that device's GPU access has ended,
  /// possibly aborted; it never certifies rendering or external-reader cleanup.
  /// The backend must retain the exact generation while ready and refuse any
  /// replacement/healthy device. Unsupported backends fail closed.
  Future<void> retireLostDevice({required int deviceGeneration}) =>
      Future.error(UnsupportedError('Lost-device retirement is unsupported'));

  /// Register a non-owning receipt before detaching this creation device.
  /// It completes only after final context teardown, including other module/native
  /// attachments. A worker must not await it before releasing its own owners.
  Future<void> observeContextRetirement({required int deviceGeneration}) =>
      Future.error(UnsupportedError('Final context retirement is unsupported'));

  /// Limits of this platform context's current, live GPUDevice. Null when the
  /// device is absent, lost, or the backend cannot prove its limits.
  DeviceBufferLimits? get currentDeviceBufferLimits => null;

  /// Exact current-device texture limits and bounded upload support, or null
  /// on older backends. This does not initialize or replace a device.
  DeviceTextureLimits? get currentDeviceTextureLimits => null;

  /// Called once for each observed device loss. Implementations must discard
  /// stale callbacks from devices replaced by a later initialization.
  void setDeviceLostCallback(void Function()? callback) {}
  PlatformComputeShader createComputeShader();
  PlatformBuffer createBuffer(int bufferSize, BufferDataType dataType);
  PlatformTexture createTexture(TextureDescriptor descriptor) =>
      throw UnsupportedError('Owned textures are unsupported by this backend');
  PlatformSampler createSampler(SamplerDescriptor descriptor) =>
      throw UnsupportedError('Owned samplers are unsupported by this backend');
  PlatformRenderPipeline createRenderPipeline(
    RenderPipelineDescriptor descriptor,
  ) => throw UnsupportedError(
    'Raster pipelines are unsupported by this backend',
  );

  PlatformBuffer createBufferFromDescriptor(BufferDescriptor descriptor) {
    descriptor.validate();
    if (descriptor.usage != BufferUsage.legacy) {
      throw UnsupportedError(
        'Explicit buffer usage is unsupported by this backend',
      );
    }
    return createBuffer(descriptor.sizeBytes, BufferDataType.uint32);
  }

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

  /// BLOCKS until every GPU task already queued has run.
  ///
  /// The native side runs dispatches, readbacks and blits on its own worker
  /// thread, so `destroy()` on a buffer or texture can free a resource that a
  /// still-queued task is about to touch. Draining first makes the teardown
  /// ordered rather than hopeful.
  ///
  /// It is SYNCHRONOUS on purpose: the place that needs it most is a hook that
  /// cannot await — Flutter's [State.reassemble] during hot reload, where the
  /// app has to stop and release GPU resources before the framework rebuilds
  /// on top of them. Cost is bounded by the work already queued, so drain
  /// after you have stopped producing, never per frame.
  ///
  /// No-op on platforms with no worker thread (web) and on a binary predating
  /// the export.
  void drainWorkQueue() {}

  /// Waits until work submitted before this call has completed on this
  /// platform context's GPU queue.
  ///
  /// Implementations must flush any recorded work before registering the
  /// completion point. Platforms that cannot observe GPU queue completion
  /// must reject this request rather than treating submission as completion.
  Future<void> waitForSubmittedWork() async {
    throw UnsupportedError(
      'Submitted GPU work completion is not supported on this platform',
    );
  }

  /// Creates an INDEPENDENT platform context bound to the adapter whose name
  /// contains [adapterFilter] (case-insensitive substring, e.g. "3090") —
  /// its own device, queue, and task FIFO.  Buffers and shaders created from
  /// the returned platform live on that adapter and must not be mixed with
  /// another context's resources in a single dispatch.  Returns null on
  /// platforms without multi-adapter support (web).
  MinigpuPlatform? createSecondaryPlatform(String adapterFilter) => null;

  /// Configures the persistent shader cache. Returns `true` when the change
  /// was stored before any context was initialized (so it affects the next
  /// device), `false` when a context is already live or the platform has no
  /// shader cache. See [Minigpu.configureShaderCache] for the full contract.
  bool configureShaderCache({
    bool? enabled,
    String? directory,
    int? maxBytes,
    String? extraKey,
  }) => false;

  /// Cache counters for this process, or `null` where there is no cache.
  ShaderCacheStats? get shaderCacheStats => null;

  /// Directory the cache is using, or `null` when it is not using one.
  String? get shaderCacheDirectory => null;

  /// Deletes every cached entry; returns how many files were removed.
  int clearShaderCache() => 0;
}

/// Buffer limits reported by the current device, in bytes.
///
/// This value has no lifetime of its own. Callers must bind it to their
/// Minigpu device epoch and discard it when the device is lost or retired.
final class DeviceBufferLimits {
  const DeviceBufferLimits({
    required this.deviceGeneration,
    required this.maxBufferSize,
    required this.maxStorageBufferBindingSize,
  });

  /// Backend generation of the device these limits came from.
  final int deviceGeneration;
  final int maxBufferSize;
  final int maxStorageBufferBindingSize;
}

final class DeviceTextureLimits {
  const DeviceTextureLimits({
    required this.deviceGeneration,
    required this.maxTextureDimension2D,
    required this.maxTextureArrayLayers,
    required this.supportsRegionWrite,
  });

  final int deviceGeneration, maxTextureDimension2D, maxTextureArrayLayers;
  final bool supportsRegionWrite;
}

/// Snapshot of persistent shader cache counters.
///
/// Absolute milliseconds on a loaded machine are noisy — treat [hits] /
/// [misses] and the RATIO of cold to warm [pipelineCreateMs] as the signal,
/// not any single timing.
final class ShaderCacheStats {
  const ShaderCacheStats({
    required this.hits,
    required this.misses,
    required this.stores,
    required this.storeFailures,
    required this.evictions,
    required this.bytesOnDisk,
    required this.entryCount,
    required this.loadMs,
    required this.storeMs,
    required this.pipelineCreateMs,
    required this.enabled,
    required this.usingDefaultProvider,
  });

  /// Lookups served from the cache.
  final int hits;

  /// Lookups that had to compile. Includes entries rejected as corrupt or as
  /// belonging to a different key — a rejected entry is always a miss, never a
  /// wrong pipeline.
  final int misses;

  /// Compiled blobs written to storage.
  final int stores;

  /// Writes that failed (unwritable directory, disk full, lost rename race).
  /// Costs a recompile next launch and nothing else.
  final int storeFailures;

  /// Entries deleted to stay under the size cap.
  final int evictions;

  final int bytesOnDisk;
  final int entryCount;

  /// Time spent inside the cache itself.
  final int loadMs;
  final int storeMs;

  /// Total time in compute pipeline creation — the cost the cache exists to
  /// remove. Compare a cold launch against a warm one.
  final int pipelineCreateMs;

  final bool enabled;

  /// False when a custom native provider is installed.
  final bool usingDefaultProvider;

  @override
  String toString() =>
      'ShaderCacheStats(hits: $hits, misses: $misses, stores: $stores, '
      'storeFailures: $storeFailures, evictions: $evictions, '
      'entries: $entryCount, bytes: $bytesOnDisk, '
      'pipelineCreateMs: $pipelineCreateMs, enabled: $enabled)';
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

/// D3D11-only provenance stored in [ExternalPlane.drmFormatModifier].
///
/// The producer must have completed all GPU writes to a unique snapshot and
/// must never reuse or modify its allocation after handoff. Import retains the
/// resource even after the producer closes its handle. Other values keep the
/// conservative copy path. Linux DRM modifiers retain their existing meaning.
const int kD3D11ImmutableReadyTag = 0x57474331; // "WGC1"

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

  /// An opaque platform object that cannot be expressed as an integer address.
  ///
  /// Every other content type names its resource with a number — an NT handle,
  /// a pointer, an fd — which [ExternalPlane.dataPtr] carries. The web has no
  /// such number: a WebCodecs `VideoFrame` is a JS object, and the only thing
  /// that can be handed to `importExternalTexture` is the object itself. So
  /// [ExternalContentType.webVideoFrame] puts it HERE and leaves [planes]
  /// empty.
  ///
  /// Typed as [Object] rather than `JSAny` on purpose: this file is imported by
  /// native builds that must not see `dart:js_interop`. The web backend is the
  /// only reader and casts it back.
  final Object? externalHandle;

  const ExternalVideoBuffer({
    required this.contentType,
    required this.pixelFormat,
    required this.width,
    required this.height,
    required this.planes,
    this.fence = const ExternalFence(),
    this.timestampUs = 0,
    this.externalHandle,
  });

  /// Whether the producer explicitly permits immutable D3D11 snapshot import.
  /// The native backend additionally validates texture format and capabilities.
  bool get isD3D11ImmutableReady =>
      contentType == ExternalContentType.d3d11SharedHandle &&
      pixelFormat == ExternalPixelFormat.bgra32 &&
      planes.length == 1 &&
      planes.first.drmFormatModifier == kD3D11ImmutableReadyTag;
}

/// Opaque handle to an imported video texture on the GPU.
abstract class PlatformVideoTexture {
  /// True when [landPackedRgba8] is the ONLY way to consume this texture.
  ///
  /// 🔴 THE CALLER MUST NOT FALL BACK when this is true and the land fails.
  /// A web `GPUExternalTexture` cannot be bound to the ordinary
  /// texture-to-shader route at all — that route's pipelines declare
  /// `texture_2d<f32>` — so "try the normal way instead" produces a validation
  /// error per frame and, because WebGPU validation is UNCAPTURED and
  /// ASYNCHRONOUS, still reports success. That is how a capture path accepted a
  /// configuration that could never draw a pixel.
  bool get requiresExternalLanding => false;

  /// Land this texture into [dst] as packed RGBA8 (`array<u32>`, one texel per
  /// element, row-major) — the destination layout every downstream kernel
  /// already reads.
  ///
  /// 🔴 EXISTS FOR THE WEB, where the imported resource is a
  /// `GPUExternalTexture` and the ordinary route cannot carry it: an external
  /// texture must be declared `texture_external` in WGSL and bound where the
  /// texture actually lives, which on web is the JS side rather than the wasm
  /// module that owns every other pipeline. Backends whose normal
  /// bind-a-texture-to-a-shader path already works return false here and are
  /// driven the usual way.
  ///
  /// [downscale] > 1 box-means that many source texels per landed pixel, in the
  /// same dispatch — the simulcast-rung input path.
  bool landPackedRgba8(
    PlatformBuffer dst, {
    required int width,
    required int height,
    int downscale = 1,
  }) => false;

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
  Future<bool> bgraToRgbaSharedOutputAsync(
    PlatformSharedOutputTexture dst,
  ) async => bgraToRgbaSharedOutput(dst);

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

  void setBufferRange(
    int slot,
    PlatformBuffer buffer,
    BufferBindingDescriptor descriptor,
  ) {
    throw UnsupportedError(
      'Explicit buffer ranges are unsupported by this backend',
    );
  }

  void setTexture(
    int slot,
    PlatformTextureView view,
    TextureBindingKind kind,
  ) => throw UnsupportedError('Owned textures are unsupported by this backend');
  /// Trusted native input cleanup, after actual GPU reads. Retires cached
  /// input/bind-group references while preserving the shader's layout.
  void releaseTextureBindingAfterGpuRead(int slot) => throw UnsupportedError(
    'Completed texture binding release is unsupported by this backend',
  );
  void setSampler(int slot, PlatformSampler sampler, SamplerBindingKind kind) =>
      throw UnsupportedError('Owned samplers are unsupported by this backend');

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

abstract class PlatformTexture {
  PlatformTextureView createView(TextureViewDescriptor descriptor);
  Future<void> write(Uint8List bytes, {int mipLevel = 0, int arrayLayer = 0});
  bool get supportsRegionWrite => false;

  /// Tightly packed rectangle; source bytes may be released after completion.
  /// This is not a submitted GPU work fence.
  Future<void> writeRegion(
    Uint8List bytes, {
    required int x,
    required int y,
    required int width,
    required int height,
    int mipLevel = 0,
    int arrayLayer = 0,
  }) => throw UnsupportedError('Texture region uploads are unsupported');
  void destroy();
}

abstract class PlatformTextureView {
  void destroy();
}

abstract class PlatformSampler {
  void destroy();
}

abstract class PlatformRenderPipeline {
  int get status;
  String get diagnostic;

  /// Native/backend counters for retained render bind groups, ordered as
  /// created, reused, retired and live. Null means the loaded backend predates
  /// this diagnostic; it does not imply that bindings are retained.
  List<int>? get bindingCacheStats => null;

  Future<void> submitRender({
    required PlatformTextureView color,
    PlatformTextureView? depth,
    required List<double> clear,
    required int loadFlags,
    required List<PlatformBuffer> buffers,
    required List<PlatformTextureView> views,
    required List<PlatformSampler> samplers,
    required List<int> commands,
  });
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

  /// Web only: the Emscripten integer handle of the underlying `WGPUBuffer`,
  /// or 0 where the concept does not exist (native, or WASM not up).
  ///
  /// This is the zero-readback web display path: a view layer (minigpu_view's
  /// canvas plugin) resolves the JS `GPUBuffer` from the handle via
  /// `WebGPU.getJsObject(handle)` and blits it into a canvas on the SAME
  /// device queue. Integers are method-channel-codec-safe, which is why the
  /// handle and not the JS object crosses layers.
  int get webBufferHandle => 0;

  void destroy();
}

final class MinigpuPlatformOutOfMemoryException implements Exception {
  @override
  String toString() => 'Out of memory';
}
