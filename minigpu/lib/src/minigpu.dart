import 'package:minigpu/src/buffer.dart';
import 'package:minigpu/src/compute_shader.dart';
import 'package:minigpu/src/shared_output_texture.dart';
import 'package:minigpu/src/video_texture.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

/// Controls the initialization and destruction of the minigpu context.
final class Minigpu {
  Minigpu() {
    _finalizer.attach(this, _platform);
  }

  static final _finalizer = Finalizer<MinigpuPlatform>(
    (platform) => platform.destroyContext(),
  );
  static final _shaderFinalizer = Finalizer<PlatformComputeShader>(
    (shader) => shader.destroy(),
  );
  static final _bufferFinalizer = Finalizer<Buffer>(
    (buffer) => buffer.destroy(),
  );

  final _platform = MinigpuPlatform.instance;
  bool isInitialized = false;

  // Internal shader cache: code hash -> shader
  // Live allocation counters — incremented on createBuffer/createComputeShader
  // and decremented in Buffer.destroy() / ComputeShader.destroy().
  // Inspect via [liveBufferCount] and [liveShaderCount] during debugging or
  // in tests to catch per-frame allocation leaks without needing DXGI/VRAM
  // measurement (which only detects leaks after the D3D12 pool grows past its
  // high-water mark).
  int _liveBufferCount = 0;
  int _liveShaderCount = 0;

  // Tracks all live shaders so they can be bulk-destroyed between test groups.
  // Dawn/D3D12 silently fails dispatch when too many compute pipelines are
  // active simultaneously. Destroying old shaders between tests (not during
  // active GPU work) is safe and frees D3D12 resources.
  final List<ComputeShader> _liveShaders = [];

  /// Number of GPU buffers currently alive (created but not yet destroyed).
  ///
  /// Use this in tests or debug builds to assert that buffer counts are
  /// stable across frames:
  ///
  /// ```dart
  /// final before = gpu.liveBufferCount;
  /// await stage.process(inputs);   // run one frame
  /// expect(gpu.liveBufferCount, equals(before)); // must not grow
  /// ```
  int get liveBufferCount => _liveBufferCount;

  /// Number of [ComputeShader]s currently alive (created but not yet destroyed).
  int get liveShaderCount => _liveShaderCount;

  /// Initializes the minigpu context.
  Future<void> init() async {
    if (isInitialized) throw MinigpuAlreadyInitError();

    await _platform.initializeContext();
    isInitialized = true;
  }

  /// Destroys the minigpu context.
  Future<void> destroy() async {
    if (!isInitialized) throw MinigpuNotInitializedError();

    _copyShader?.destroy();
    _copyShader = null;

    await _platform.destroyContext();
    isInitialized = false;
  }

  /// Install a log callback for the native minigpu/Dawn layer.
  ///
  /// [callback] receives `(int level, String message)` where level matches
  /// `mgpu::LogLevel`: 0=DEBUG 1=INFO 2=WARN 3=ERROR.
  /// Pass `null` to revert to the default (native stderr) output.
  ///
  /// [level] controls minimum verbosity (-1=none 0=debug 1=info 2=warn 3=error).
  ///
  /// This is a static method — there is one native log channel regardless of
  /// how many [Minigpu] instances exist.
  static void setLogCallback(
    void Function(int level, String message)? callback, {
    int level = 1,
  }) {
    MinigpuPlatform.instance.setLogCallback(callback, level: level);
  }

  /// Pre-init hint (Windows): bind Dawn to the adapter driving the PRIMARY
  /// display so screen capture (Desktop Duplication / WGC), GPU processing
  /// and any D3D11 encoder created on Dawn's adapter share one GPU —
  /// same-adapter zero-copy import — even on multi-output hybrid systems
  /// where the discrete GPU also drives a monitor (e.g. AMD/Intel iGPU
  /// showing the desktop while an NVIDIA dGPU drives a secondary output).
  /// The `MGPU_ADAPTER_NAME` env var still overrides.
  ///
  /// Must be called BEFORE any minigpu context is initialized (there is one
  /// process-global native context, hence a static). Returns `true` when the
  /// hint was stored before init; `false` when the context was already
  /// initialized (kept for a future re-init, live context unchanged) or on
  /// platforms without adapter selection (web).
  static bool preferDisplayAdapter([bool enable = true]) =>
      MinigpuPlatform.instance.preferDisplayAdapter(enable);

  /// Name of the adapter Dawn actually selected, or `null` when the context
  /// is not initialized / the platform does not expose it.
  static String? get selectedAdapterName =>
      MinigpuPlatform.instance.selectedAdapterName;

  /// Synchronous variant of [destroy] intended for use in Flutter hot-restart
  /// teardown hooks where `await` is not available (e.g. inside
  /// [WidgetsBindingObserver.reassemble]).
  ///
  /// The underlying C call (`mgpuDestroyContext`) is synchronous; the async
  /// wrapper on [destroy] exists only for API consistency.  This method is
  /// a no-op if the context is not initialized.
  void destroySync() {
    if (!isInitialized) return;
    _platform.destroyContext();
    isInitialized = false;
  }

  /// Creates a compute shader.
  ComputeShader createComputeShader() {
    final platformShader = _platform.createComputeShader();
    final shader = CachedComputeShader(platformShader, this);
    _liveShaders.add(shader);
    _liveShaderCount++;
    return shader;
  }

  /// Called by [ComputeShader.destroy] to decrement [liveShaderCount].
  // Accessed from compute_shader.dart (same package — library-private is fine).
  // ignore: use_setters_to_change_properties
  void onShaderDestroyed(ComputeShader shader) {
    if (_liveShaderCount > 0) _liveShaderCount--;
    _liveShaders.remove(shader);
    if (identical(shader, _copyShader)) _copyShader = null;
  }

  /// Destroys all currently tracked live shaders and clears the tracking list.
  ///
  /// Call this between test groups (in `setUpAll`) to free D3D12 pipeline
  /// resources accumulated by previous tests. Safe to call when no GPU work
  /// is pending.
  void destroyAllTrackedShaders() {
    final shaders = List<ComputeShader>.from(_liveShaders);
    _liveShaders.clear();
    for (final shader in shaders) {
      try {
        shader.destroy();
      } catch (_) {}
    }
  }

  /// Creates a buffer.
  Buffer createBuffer(int bufferSize, BufferDataType dataType) {
    if (!isInitialized) throw MinigpuNotInitializedError();

    final platformBuffer = _platform.createBuffer(bufferSize, dataType);
    final buff = Buffer(platformBuffer, _onBufferDestroyed);
    _liveBufferCount++;
    return buff;
  }

  /// Called by [Buffer.destroy] to decrement [liveBufferCount].
  // ignore: use_setters_to_change_properties
  void _onBufferDestroyed() {
    if (_liveBufferCount > 0) _liveBufferCount--;
  }

  /// Returns dedicated VRAM usage in bytes for the primary GPU (Windows D3D12
  /// via DXGI QueryVideoMemoryInfo).  Returns -1 on unsupported platforms.
  int queryVramBytes() => _platform.queryVramBytes();

  /// Returns true if the given content type can be imported on this platform.
  bool isExternalContentTypeSupported(ExternalContentType type) =>
      _platform.isExternalContentTypeSupported(type);

  /// Returns true if the given pixel format can be imported on this platform.
  bool isExternalPixelFormatSupported(ExternalPixelFormat format) =>
      _platform.isExternalPixelFormatSupported(format);

  /// Import an external video frame as a GPU texture.
  /// Returns null if unsupported or on failure.
  VideoTexture? importVideoFrame(ExternalVideoBuffer buf) {
    if (!isInitialized) throw MinigpuNotInitializedError();
    final platformTex = _platform.importVideoFrame(buf);
    if (platformTex == null) return null;
    return VideoTexture(platformTex, this);
  }

  /// Create a cross-API shared output RGBA8 texture suitable for zero-copy
  /// hand-off to an external API (currently: Windows D3D12<->D3D11 via NT
  /// shared handle). Returns null if unsupported on this platform/backend.
  SharedOutputTexture? createSharedOutputTexture(int width, int height) {
    if (!isInitialized) throw MinigpuNotInitializedError();
    final platformTex = _platform.createSharedOutputTexture(width, height);
    if (platformTex == null) return null;
    return SharedOutputTexture(platformTex, this);
  }

  /// Returns an `ID3D11Device*` (as an integer address) created on the same
  /// DXGI adapter as Dawn's internal D3D12 device.  Non-zero only on Windows
  /// with Dawn's D3D12 backend.  Ownership is transferred to the caller —
  /// FFmpeg will call `Release()` when the AVHWDeviceContext is freed.
  int createD3D11DeviceOnDawnAdapter() {
    if (!isInitialized) throw MinigpuNotInitializedError();
    return _platform.createD3D11DeviceOnDawnAdapter();
  }

  // -------------------------------------------------------------------------
  // Buffer copy
  // -------------------------------------------------------------------------

  /// WGSL kernel: copies [elementCount] u32 words from src to dst.
  static const _kCopyWgsl = r'''
@group(0) @binding(0) var<storage, read_write> src : array<u32>;
@group(0) @binding(1) var<storage, read_write> dst : array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let n = arrayLength(&src);
  if (gid.x >= n) { return; }
  dst[gid.x] = src[gid.x];
}
''';

  /// Lazily-created shader for [copyBuffer].
  ComputeShader? _copyShader;

  /// GPU-side buffer copy — copies [elementCount] elements from [src] to [dst]
  /// entirely on the GPU using a WGSL compute shader (no CPU round-trip).
  ///
  /// Both buffers must have been created on this [Minigpu] instance and must
  /// be large enough for [elementCount] elements at their respective data types.
  ///
  /// [elementCount] must be provided — use [getBufferSizeForType] divided by
  /// the element byte-size if you only know the byte size.
  ///
  /// The shader is created once and reused across calls.
  Future<void> copyBuffer(
    Buffer src,
    Buffer dst, {
    required int elementCount,
  }) async {
    if (!isInitialized) throw MinigpuNotInitializedError();

    _copyShader ??= createComputeShader()..loadKernelString(_kCopyWgsl);

    _copyShader!
      ..setBufferAtSlot(0, src)
      ..setBufferAtSlot(1, dst);

    final int groups = (elementCount + 63) ~/ 64;
    await _copyShader!.dispatch(groups, 1, 1);
  }
}
