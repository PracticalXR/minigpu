import 'dart:async';
import 'dart:typed_data';

import 'package:minigpu/src/buffer.dart';
import 'package:minigpu/src/compute_shader.dart';
import 'package:minigpu/src/gpu_texture.dart';
import 'package:minigpu/src/render_pipeline.dart';
import 'package:minigpu/src/shared_output_texture.dart';
import 'package:minigpu/src/video_texture.dart';
import 'package:minigpu/src/resource_retirement.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

part 'device_coordinator.dart';
part 'device_recovery.dart';

/// Monotonic resource lifecycle counters for one [Minigpu] wrapper.
///
/// Created and retired totals never reset when the device is reopened, so a
/// create-and-retire cycle remains visible even when every live count returns
/// to its earlier value. The counters describe wrapper allocations, not driver
/// heap bytes or memory pressure.
final class GpuResourceAllocationSnapshot {
  const GpuResourceAllocationSnapshot({
    required this.buffersCreated,
    required this.buffersRetired,
    required this.buffersLive,
    required this.texturesCreated,
    required this.texturesRetired,
    required this.texturesLive,
    required this.textureViewsCreated,
    required this.textureViewsRetired,
    required this.textureViewsLive,
    required this.samplersCreated,
    required this.samplersRetired,
    required this.samplersLive,
    required this.computeShadersCreated,
    required this.computeShadersRetired,
    required this.computeShadersLive,
    required this.renderPipelinesCreated,
    required this.renderPipelinesRetired,
    required this.renderPipelinesLive,
  });

  final int buffersCreated, buffersRetired, buffersLive;
  final int texturesCreated, texturesRetired, texturesLive;
  final int textureViewsCreated, textureViewsRetired, textureViewsLive;
  final int samplersCreated, samplersRetired, samplersLive;
  final int computeShadersCreated, computeShadersRetired, computeShadersLive;
  final int renderPipelinesCreated, renderPipelinesRetired, renderPipelinesLive;

  int get totalCreated =>
      buffersCreated +
      texturesCreated +
      textureViewsCreated +
      samplersCreated +
      computeShadersCreated +
      renderPipelinesCreated;

  int get totalRetired =>
      buffersRetired +
      texturesRetired +
      textureViewsRetired +
      samplersRetired +
      computeShadersRetired +
      renderPipelinesRetired;

  Map<String, Object> toJson() => {
    'buffers': _resourceJson(buffersCreated, buffersRetired, buffersLive),
    'textures': _resourceJson(texturesCreated, texturesRetired, texturesLive),
    'texture_views': _resourceJson(
      textureViewsCreated,
      textureViewsRetired,
      textureViewsLive,
    ),
    'samplers': _resourceJson(samplersCreated, samplersRetired, samplersLive),
    'compute_shaders': _resourceJson(
      computeShadersCreated,
      computeShadersRetired,
      computeShadersLive,
    ),
    'render_pipelines': _resourceJson(
      renderPipelinesCreated,
      renderPipelinesRetired,
      renderPipelinesLive,
    ),
    'total_created': totalCreated,
    'total_retired': totalRetired,
  };

  static Map<String, int> _resourceJson(int created, int retired, int live) => {
    'created': created,
    'retired': retired,
    'live': live,
  };
}

/// Buffer limits captured from one current GPU device generation.
final class GpuDeviceBufferLimitsSnapshot {
  const GpuDeviceBufferLimitsSnapshot({
    required this.deviceEpoch,
    required this.maxBufferSize,
    required this.maxStorageBufferBindingSize,
  });

  final int deviceEpoch;
  final int maxBufferSize;
  final int maxStorageBufferBindingSize;
}

/// Texture limits captured from this wrapper's current device epoch.
final class GpuDeviceTextureLimitsSnapshot {
  const GpuDeviceTextureLimitsSnapshot({
    required this.deviceEpoch,
    required this.maxTextureDimension2D,
    required this.maxTextureArrayLayers,
    required this.supportsRegionWrite,
  });

  final int deviceEpoch, maxTextureDimension2D, maxTextureArrayLayers;
  final bool supportsRegionWrite;
}

/// Resource cleanup evidence. Aborted device work cannot be used as a frame.
enum GpuWorkRetirement { completed, deviceAborted }

/// Controls the initialization and destruction of the minigpu context.
///
/// There is ONE process-global native context, so this wrapper is a
/// per-isolate SINGLETON: every `Minigpu()` returns the same instance.
///
/// Historically each construction was a fresh object that attached a
/// Finalizer calling `destroyContext()` — so any temporary wrapper (e.g.
/// `Minigpu().isInitialized` in a test setUp) would, whenever the GC ran,
/// DESTROY THE DEVICE out from under every live buffer and shader in the
/// process.  The context would auto-reinitialize, but resources created
/// before the loss were invalid on the new device, and dispatches using
/// them were silently dropped (zero outputs, no Dart-visible error) — the
/// source of "identical inputs produce different outputs" flakes.  The
/// context is now destroyed ONLY by explicit [destroy] / [destroySync].
final class Minigpu {
  factory Minigpu() => _instance;
  Minigpu._() : _platform = MinigpuPlatform.instance;

  /// Creates an INDEPENDENT context on the adapter whose name contains
  /// [adapterFilter] (case-insensitive substring, e.g. `'3090'`) — its own
  /// device, queue, and task FIFO.  NOT the singleton: call [init] before
  /// use and [destroy] when done.  Buffers/shaders from different [Minigpu]
  /// instances must never be mixed in one dispatch.  Throws
  /// [UnsupportedError] on platforms without multi-adapter support (web).
  factory Minigpu.forAdapter(String adapterFilter) {
    final platform = MinigpuPlatform.instance.createSecondaryPlatform(
      adapterFilter,
    );
    if (platform == null) {
      throw UnsupportedError(
        'Multi-adapter contexts are not supported on this platform',
      );
    }
    return Minigpu._withPlatform(platform);
  }

  /// Creates an independent wrapper around an embedder-supplied platform.
  ///
  /// Constructing it does not create a device. Its [init] and [destroy] use
  /// the same attachment and resource-lifetime contract as other wrappers.
  factory Minigpu.withPlatform(MinigpuPlatform platform) =>
      Minigpu._withPlatform(platform);

  Minigpu._withPlatform(this._platform);

  static final Minigpu _instance = Minigpu._();

  final MinigpuPlatform _platform;
  GpuDeviceCoordinator? _coordinator;

  /// Trusted lifetime ownership for this wrapper, including external readers.
  GpuDeviceCoordinator get deviceCoordinator =>
      _coordinator ??= GpuDeviceCoordinator._(this);
  bool _isInitialized = false;

  /// True from a completed [init] until [destroy] has released the context.
  bool get isInitialized => _isInitialized;

  /// Kept for source compatibility only: [init] and [destroy] own this state,
  /// and writing it detaches the wrapper from the real context.
  @Deprecated('Lifecycle state is owned by init() and destroy()')
  set isInitialized(bool value) {
    _isInitialized = value;
  }

  int _deviceEpoch = 0;

  /// Increments after each successful device initialization. Shader warm
  /// residency must be scoped to this epoch, not only to source text.
  int get deviceEpoch => _deviceEpoch;
  GpuDeviceBufferLimitsSnapshot? _currentDeviceBufferLimits;
  int? _expectedPlatformDeviceGeneration;
  bool _deviceInvalidated = false;
  DeviceAdapterIdentity? _creationAdapterIdentity;
  DeviceRecoveryTarget? _creationRecoveryTarget;
  _GpuDeviceRecovery? _recovery;
  bool _recoveryDetach = false;
  final _recoveryListeners = <Object, void Function(GpuRecoverySnapshot)>{};
  void Function(int deviceEpoch)? _deviceLostCallback;
  final _deviceLostListeners = <Object, void Function(int)>{};

  /// The exact limits of this wrapper's current device, or null when the
  /// device is unavailable, lost, retiring, or cannot be queried safely.
  /// A previously returned snapshot must be discarded after its epoch ends.
  GpuDeviceBufferLimitsSnapshot? get currentDeviceBufferLimits {
    if (!isInitialized || _closing || _deviceInvalidated) return null;
    final limits = _platform.currentDeviceBufferLimits;
    if (limits == null ||
        limits.deviceGeneration != _expectedPlatformDeviceGeneration) {
      if (_expectedPlatformDeviceGeneration != null) {
        _onPlatformDeviceLost(_deviceEpoch);
      }
      return null;
    }
    final old = _currentDeviceBufferLimits;
    if (old != null && old.deviceEpoch == _deviceEpoch &&
        old.maxBufferSize == limits.maxBufferSize &&
        old.maxStorageBufferBindingSize == limits.maxStorageBufferBindingSize) {
      return old;
    }
    return _currentDeviceBufferLimits = GpuDeviceBufferLimitsSnapshot(
      deviceEpoch: _deviceEpoch,
      maxBufferSize: limits.maxBufferSize,
      maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    );
  }

  /// True only when this backend can verify the current device and limits.
  bool get isDeviceCurrent => currentDeviceBufferLimits != null;

  GpuDeviceTextureLimitsSnapshot? get currentDeviceTextureLimits {
    if (!isDeviceCurrent) return null;
    final limits = _platform.currentDeviceTextureLimits;
    if (limits == null ||
        limits.deviceGeneration != _expectedPlatformDeviceGeneration ||
        limits.maxTextureDimension2D < 1 ||
        limits.maxTextureArrayLayers < 1) return null;
    return GpuDeviceTextureLimitsSnapshot(
      deviceEpoch: _deviceEpoch,
      maxTextureDimension2D: limits.maxTextureDimension2D,
      maxTextureArrayLayers: limits.maxTextureArrayLayers,
      supportsRegionWrite: limits.supportsRegionWrite,
    );
  }

  /// Installs a callback for observed loss of this wrapper's current device.
  /// Native loss is detected by a subsequent limits/status query; web listens
  /// to the owned GPUDevice.lost promise. Replacing the callback is supported.
  void setDeviceLostCallback(void Function(int deviceEpoch)? callback) {
    _deviceLostCallback = callback;
  }

  /// Observes device loss without replacing another consumer's callback.
  ///
  /// Notification is synchronous and once per lost epoch, so publications can
  /// be fenced before another frame is submitted. Each subscription is owned
  /// independently; the returned removal is idempotent. Listeners added during
  /// a notification start with the next loss. This does not replay a past loss:
  /// callers must still check [isDeviceCurrent] before using device resources.
  /// Listeners survive explicit teardown/reinitialization until removed.
  void Function() onDeviceLost(void Function(int deviceEpoch) listener) {
    final token = Object();
    _deviceLostListeners[token] = listener;
    return () => _deviceLostListeners.remove(token);
  }

  void _onPlatformDeviceLost(int epoch) {
    if (_closing || !isInitialized || _deviceInvalidated ||
        epoch != _deviceEpoch) return;
    _deviceInvalidated = true;
    _currentDeviceBufferLimits = null;
    _coordinator?.drain().ignore();
    // One faulty observer must not prevent other consumers from fencing their
    // handles. Report errors only after every still-subscribed observer ran.
    final failures = <(Object, StackTrace)>[];
    void notify(void Function(int) callback) {
      try {
        callback(epoch);
      } catch (error, stack) {
        failures.add((error, stack));
      }
    }
    final listeners = _deviceLostListeners.entries.toList(growable: false);
    final legacy = _deviceLostCallback;
    if (legacy != null) notify(legacy);
    for (final entry in listeners) {
      if (_deviceLostListeners.containsKey(entry.key)) notify(entry.value);
    }
    for (final (error, stack) in failures) {
      scheduleMicrotask(() => Error.throwWithStackTrace(error, stack));
    }
  }
  Future<void>? _initializing;
  Future<void>? _retiring;

  /// The [destroy] future whose teardown reached the context release. Its
  /// failure (a child that would not destroy, a failed in-flight init or
  /// release) then belongs to the [destroy] callers alone: a backend still
  /// holding a context it could not release refuses the next init itself. A
  /// teardown stopped before the release (owners quarantined) blocks [init].
  Future<void>? _teardownReachedRelease;

  /// Whether that release succeeded: device recovery may only wait for the
  /// final context receipt of a context that was actually returned.
  bool _teardownReleased = false;
  Future<void>? _lostWorkRetirement;
  Future<void>? _contextRetirement;
  bool _closing = false;

  // Internal shader cache: code hash -> shader
  // Live allocation counters — incremented on createBuffer/createComputeShader
  // and decremented in Buffer.destroy() / ComputeShader.destroy().
  // Inspect via [liveBufferCount] and [liveShaderCount] during debugging or
  // in tests to catch per-frame allocation leaks without needing DXGI/VRAM
  // measurement (which only detects leaks after the D3D12 pool grows past its
  // high-water mark).
  int _bufferCreateCount = 0, _bufferRetireCount = 0;
  int _textureCreateCount = 0, _textureRetireCount = 0;
  int _textureViewCreateCount = 0, _textureViewRetireCount = 0;
  int _samplerCreateCount = 0, _samplerRetireCount = 0;
  int _shaderCreateCount = 0, _shaderRetireCount = 0;
  int _renderPipelineCreateCount = 0, _renderPipelineRetireCount = 0;

  final Map<WeakReference<Buffer>, GpuResourceRetirement> _liveBuffers = {};
  final Map<WeakReference<GpuTexture>, GpuResourceRetirement> _liveTextures =
      {};
  final Map<WeakReference<GpuTextureView>, GpuResourceRetirement>
  _liveTextureViews = {};
  final Map<WeakReference<GpuSampler>, GpuResourceRetirement> _liveSamplers =
      {};

  // Tracks all live shaders so they can be bulk-destroyed between test groups.
  // Dawn/D3D12 silently fails dispatch when too many compute pipelines are
  // active simultaneously. Destroying old shaders between tests (not during
  // active GPU work) is safe and frees D3D12 resources.
  // Held weakly, like buffers: a dropped shader or pipeline is collected and
  // its finalizer retires it (the token holds only the platform object).
  final Map<WeakReference<ComputeShader>, GpuResourceRetirement>
  _shaderRegistry = {};
  final Map<WeakReference<GpuRenderPipeline>, GpuResourceRetirement>
  _renderPipelineRegistry = {};

  // Imported frames and shared output textures are not counted, but they
  // retire with the context, so none outlives the device it was made on.
  final Map<WeakReference<VideoTexture>, GpuResourceRetirement>
  _liveVideoTextures = {};
  final Map<WeakReference<SharedOutputTexture>, GpuResourceRetirement>
  _liveSharedOutputTextures = {};

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
  int get liveBufferCount {
    return _liveBuffers.length;
  }

  /// Whether this device context created a still-live buffer. This proves
  /// device provenance only; callers must separately check access authority.
  bool ownsBuffer(Buffer buffer) =>
      isInitialized &&
      buffer.isValid &&
      _bufferInfo[buffer]?.epoch == _deviceEpoch;

  /// Number of [ComputeShader]s currently alive (created but not yet destroyed).
  int get liveShaderCount => _shaderRegistry.length;

  /// Resource lifecycle counters for residency diagnostics and focused tests.
  GpuResourceAllocationSnapshot get resourceAllocationSnapshot =>
      GpuResourceAllocationSnapshot(
        buffersCreated: _bufferCreateCount,
        buffersRetired: _bufferRetireCount,
        buffersLive: _liveBuffers.length,
        texturesCreated: _textureCreateCount,
        texturesRetired: _textureRetireCount,
        texturesLive: _liveTextures.length,
        textureViewsCreated: _textureViewCreateCount,
        textureViewsRetired: _textureViewRetireCount,
        textureViewsLive: _liveTextureViews.length,
        samplersCreated: _samplerCreateCount,
        samplersRetired: _samplerRetireCount,
        samplersLive: _liveSamplers.length,
        computeShadersCreated: _shaderCreateCount,
        computeShadersRetired: _shaderRetireCount,
        computeShadersLive: _shaderRegistry.length,
        renderPipelinesCreated: _renderPipelineCreateCount,
        renderPipelinesRetired: _renderPipelineRetireCount,
        renderPipelinesLive: _renderPipelineRegistry.length,
      );

  /// Initializes the minigpu context.
  ///
  /// Idempotent and concurrency-safe: with a singleton wrapper, several
  /// libraries commonly race `if (!gpu.isInitialized) await gpu.init()` at
  /// startup (tests + Tensor.create's DefaultMinigpu).  A second native
  /// init used to tear the live context down; now late callers simply await
  /// the in-flight initialization.
  Future<void> init() async {
    final recovery = _recovery;
    if (recovery != null && (recovery.pending || recovery.failed)) {
      await recovery.done.future;
      return;
    }
    final phase = _coordinator?.snapshot.phase;
    if (phase == GpuDeviceDrainPhase.draining ||
        phase == GpuDeviceDrainPhase.quarantined) {
      throw StateError('GPU device owners have not retired');
    }
    final retiring = _retiring;
    if (retiring != null) {
      try {
        await retiring;
      } catch (_) {
        // A teardown that reached the release leaves nothing here to fence:
        // its error went to the destroy() callers, and a backend still holding
        // the context refuses the init itself. A quarantined one blocks.
        if (!identical(_teardownReachedRelease, retiring)) rethrow;
      }
      if (identical(_retiring, retiring)) _retiring = null;
    }
    if (isInitialized) return Future.value();
    final initializing = _initializing;
    if (initializing != null) return initializing;
    final completer = Completer<void>();
    _initializing = completer.future;
    _completeInitialization(completer).ignore();
    return completer.future;
  }

  Future<void> _completeInitialization(Completer<void> completer,
      {DeviceRecoveryTarget? recoveryTarget}) async {
    try {
      final nextEpoch = _deviceEpoch + 1;
      _platform.setDeviceLostCallback(() => _onPlatformDeviceLost(nextEpoch));
      if (recoveryTarget == null) {
        await _platform.initializeContext();
      } else {
        await _platform.initializeContextForReplacement(recoveryTarget);
      }
      if (!_closing) {
        _isInitialized = true;
        _deviceEpoch++;
        _lostWorkRetirement = null;
        _contextRetirement = null;
        _coordinator?._initialized(_deviceEpoch);
        _currentDeviceBufferLimits = null;
        _deviceInvalidated = false;
        _expectedPlatformDeviceGeneration =
            _platform.currentDeviceBufferLimits?.deviceGeneration;
        _creationAdapterIdentity = _platform.creationAdapterIdentity;
        _creationRecoveryTarget = _platform.creationRecoveryTarget;
      }
      completer.complete();
    } catch (error, stackTrace) {
      _platform.setDeviceLostCallback(null);
      completer.completeError(error, stackTrace);
    } finally {
      if (identical(_initializing, completer.future)) {
        _initializing = null;
      }
    }
  }

  /// One small buffer per context that lives as long as the context does.
  /// [ComputeShader] binds it for an instant when a slot's previous buffer
  /// has been destroyed, so the native layer — which compares raw handles —
  /// sees a change even when the new buffer was handed the destroyed one's
  /// handle. Created on first need, often mid-frame, so it is kept out of
  /// [liveBufferCount], the allocation counters and [ownsBuffer]; [destroy]
  /// releases it.
  Buffer? _rebindSentinel;
  Buffer get rebindSentinel {
    final existing = _rebindSentinel;
    if (existing != null) return existing;
    _checkCanCreate();
    return _rebindSentinel = Buffer(
      _platform.createBuffer(16, BufferDataType.uint32),
    );
  }

  /// Destroys the minigpu context and all owned buffer and shader wrappers.
  Future<void> destroy() {
    final recovery = _recovery;
    if (!_recoveryDetach && recovery != null &&
        (recovery.pending || (recovery.failed && !isInitialized) ||
         (recovery.snapshot.phase == GpuRecoveryPhase.cancelled &&
          !isInitialized && _initializing == null))) {
      return recovery.cancelAndDrain();
    }
    final retiring = _retiring;
    if (retiring != null) return retiring;
    if (!isInitialized && _initializing == null) {
      return Future<void>.error(MinigpuNotInitializedError());
    }

    final completer = Completer<void>();
    _retiring = completer.future;
    _retire(completer);
    return completer.future;
  }

  Future<void> _retire(Completer<void> completer) async {
    // External presenters and imported source borrows are not in _liveBuffers.
    // Keep the old context intact when their actual retirement is unconfirmed.
    try {
      final coordinator = deviceCoordinator;
      final hasOwners = coordinator._owners.isNotEmpty;
      final draining = coordinator.drain();
      if (hasOwners) await draining;
    } catch (error, stack) {
      completer.completeError(error, stack);
      return;
    }
    _closing = true;
    _currentDeviceBufferLimits = null;
    _expectedPlatformDeviceGeneration = null;
    _deviceInvalidated = true;
    _platform.setDeviceLostCallback(null);
    Object? failure;
    StackTrace? failureStack;
    void recordFailure(Object error, StackTrace stackTrace) {
      failure ??= error;
      failureStack ??= stackTrace;
    }

    final initializing = _initializing;
    if (initializing != null) {
      try {
        await initializing;
      } catch (error, stackTrace) {
        recordFailure(error, stackTrace);
      }
    }

    _copyShader = null; // registered like any other shader, retired below
    _retireResources(
      _shaderRegistry,
      (shader) => shader.destroy(),
      recordFailure,
    );
    _retireResources(
      _renderPipelineRegistry,
      (pipeline) => pipeline.destroy(),
      recordFailure,
    );

    _retireResources(
      _liveTextureViews,
      (view) => view.destroy(),
      recordFailure,
    );
    _retireResources(
      _liveTextures,
      (texture) => texture.destroy(),
      recordFailure,
    );
    _retireResources(
      _liveSamplers,
      (sampler) => sampler.destroy(),
      recordFailure,
    );
    _retireResources(
      _liveVideoTextures,
      (texture) => texture.destroy(),
      recordFailure,
    );
    _retireResources(
      _liveSharedOutputTextures,
      (texture) => texture.destroy(),
      recordFailure,
    );

    final sentinel = _rebindSentinel;
    _rebindSentinel = null;
    _retireResources(_liveBuffers, (buffer) => buffer.destroy(), recordFailure);
    if (sentinel != null && !sentinel.retirement.isRetired) {
      try {
        sentinel.destroy();
      } catch (error, stackTrace) {
        recordFailure(error, stackTrace);
      }
    }
    // A copy still in flight sees its params invalid and does not return it.
    for (final params in _copyParams) {
      try {
        params.destroy();
      } catch (error, stackTrace) {
        recordFailure(error, stackTrace);
      }
    }
    _copyParams.clear();
    _copyParamsFree.clear();

    var released = false;
    try {
      await _platform.destroyContext();
      released = true;
    } catch (error, stackTrace) {
      recordFailure(error, stackTrace);
    } finally {
      _isInitialized = false;
      _closing = false;
      _teardownReachedRelease = completer.future;
      _teardownReleased = released;
    }

    if (failure != null) {
      completer.completeError(failure!, failureStack);
    } else {
      completer.complete();
    }
  }

  /// Install a log callback for the native minigpu/Dawn layer.
  ///
  /// [callback] receives `(int level, String message)` where level matches
  /// `mgpu::LogLevel`: 0=DEBUG 1=INFO 2=WARN 3=ERROR.
  /// Pass `null` to revert to the default (native stderr) output.
  ///
  /// [level] controls minimum verbosity (-1=none 0=debug 1=info 2=warn 3=error).
  /// It defaults to warn, the native default, so installing or clearing a
  /// callback does not turn info lines on.
  ///
  /// This is a static method — there is one native log channel regardless of
  /// how many [Minigpu] instances exist.
  static void setLogCallback(
    void Function(int level, String message)? callback, {
    int level = 2,
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

  /// Configures the persistent shader cache.
  ///
  /// Turning WGSL into a backend shader is expensive — on Windows/D3D11 it
  /// runs through FXC, whose optimiser cost grows superlinearly with kernel
  /// size, so a large compute kernel can spend tens of seconds in
  /// [ComputeShader.loadKernelString]'s first dispatch. minigpu caches the
  /// compiled result on disk BY DEFAULT, making that a once-per-machine cost
  /// rather than a once-per-process one. You do not need to call this to get
  /// the benefit; it exists to change where and whether caching happens.
  ///
  /// - [enabled] master switch (default true). False makes every lookup a miss
  ///   and drops every store.
  /// - [directory] where the default provider stores blobs. Pass `''` to
  ///   restore the per-platform default (Windows `%LOCALAPPDATA%`, macOS/iOS
  ///   `Library/Caches`, Linux `$XDG_CACHE_HOME`). **Android has no default** —
  ///   there is no way to discover an app-private cache directory from C++, so
  ///   caching stays off there until a host passes one in.
  /// - [maxBytes] size cap (default 256 MiB); eviction is LRU by last-write
  ///   time. 0 disables eviction.
  /// - [extraKey] extra text folded into every cache key. Changing it makes
  ///   all existing entries unreachable — an app-level cache-bust control.
  ///
  /// Must be called BEFORE a context is initialized (one process-global native
  /// context, hence a static). Returns `true` when every requested change
  /// landed pre-init; `false` when a context is already live (the values are
  /// still stored and apply to the next init) or the platform has no cache.
  ///
  /// Nothing here can break startup: an unwritable directory, a full disk, a
  /// corrupt entry or a race with another process all degrade to "compile it"
  /// and carry on. Web is a no-op — there is no Dawn and no FXC to cache.
  static bool configureShaderCache({
    bool? enabled,
    String? directory,
    int? maxBytes,
    String? extraKey,
  }) => MinigpuPlatform.instance.configureShaderCache(
    enabled: enabled,
    directory: directory,
    maxBytes: maxBytes,
    extraKey: extraKey,
  );

  /// Shader cache counters for this process, or `null` on platforms without a
  /// cache.
  ///
  /// The honest signal is [ShaderCacheStats.hits] / [ShaderCacheStats.misses]
  /// and the RATIO of cold to warm [ShaderCacheStats.pipelineCreateMs].
  /// Absolute milliseconds vary widely with machine load, so assert on
  /// counters and report timings.
  static ShaderCacheStats? get shaderCacheStats =>
      MinigpuPlatform.instance.shaderCacheStats;

  /// Directory the shader cache is using, or `null` when it is not using one
  /// (disabled, no platform default, or resolution failed).
  static String? get shaderCacheDirectory =>
      MinigpuPlatform.instance.shaderCacheDirectory;

  /// Deletes every cached shader blob. Returns how many files were removed.
  ///
  /// Cached blobs are keyed by Dawn version, adapter and driver, so a stale
  /// entry is already unreachable rather than dangerous — this is for
  /// reclaiming disk, or for forcing a clean measurement of cold compile time.
  static int clearShaderCache() => MinigpuPlatform.instance.clearShaderCache();

  /// Yield-spin budget (ms) the LOADED native binary implements before the
  /// event drain degrades to coarse sleeping — `null` when it can't be asked
  /// (web, or a binary predating the export), which for a native build means
  /// the drain fix is NOT in it.
  ///
  /// Latency-sensitive callers (screen recording, live present) should assert
  /// this is > 0 at startup. Without the fix every GPU wait rounds up to the
  /// Windows timer quantum (~15.6 ms): the shared-texture present measured p50
  /// 15.69 ms at BOTH 720p and 4K, and a cost that doesn't move with 9× the
  /// pixels is a clock tick, not work. Loading a stale native artifact is a
  /// mistake this repo has made more than once, and it is silent — this is how
  /// you catch it in-process rather than in a profile.
  static int? get drainSpinBudgetMs =>
      MinigpuPlatform.instance.drainSpinBudgetMs;

  /// BLOCKS until every GPU task already queued has run.
  ///
  /// Dispatches, readbacks and shared-texture blits execute on a native worker
  /// thread, so destroying a buffer or texture can free a resource a queued
  /// task is about to touch. Draining first makes teardown ordered instead of
  /// hopeful.
  ///
  /// SYNCHRONOUS on purpose — the caller that needs it most is one that cannot
  /// await. In Flutter, [State.reassemble] runs on hot reload and is the only
  /// hook you get before the framework rebuilds the tree on top of your GPU
  /// resources; it is synchronous, so `await stop()` is not available there.
  /// The shape that works:
  ///
  /// ```dart
  /// @override
  /// void reassemble() {
  ///   super.reassemble();
  ///   stopProducingFrames();   // nothing new may be queued
  ///   Minigpu.drainWorkQueue(); // everything queued has now finished
  ///   releaseGpuResources();    // safe: nothing is still pointing at them
  /// }
  /// ```
  ///
  /// Drain AFTER you have stopped producing — draining while a loop is still
  /// submitting just waits for a queue that keeps refilling. Never per frame.
  /// No-op on web and on a native binary predating the export.
  static void drainWorkQueue() => MinigpuPlatform.instance.drainWorkQueue();

  /// Waits for work submitted before this call to complete on this instance's
  /// device queue.
  ///
  /// Unlike [drainWorkQueue], this is a GPU completion point: it flushes
  /// pending fire-and-forget batches and completes only after the queue has
  /// retired them. This is the boundary for releasing resources used by an
  /// already-submitted frame.
  Future<void> waitForSubmittedWork() async {
    if (!isInitialized) throw MinigpuNotInitializedError();
    final epoch = _deviceEpoch;
    bool current() => isInitialized && !_closing && !_deviceInvalidated &&
        (_expectedPlatformDeviceGeneration == null || isDeviceCurrent);
    // Older backends without limit discovery still provide their own queue
    // fence. When discovery is supported, also detect silent native replacement.
    if (!current()) throw StateError('GPU completion device is unavailable');
    await _platform.waitForSubmittedWork();
    if (_deviceEpoch != epoch || !current()) {
      throw StateError('GPU completion belongs to a retired device');
    }
  }

  /// Waits until this epoch's submitted GPU accesses have ended. Unlike
  /// [waitForSubmittedWork], a positive native device-abortion receipt may
  /// retire resources after observed loss. Use only for GPU resource cleanup,
  /// never frame readiness, presentation, external readers or worker shutdown.
  /// Failed/unsupported retirement remains failed; it does not reset owners.
  Future<GpuWorkRetirement> retireSubmittedWork({required int deviceEpoch}) async {
    void checkEpoch() {
      if (!isInitialized || _deviceEpoch != deviceEpoch) {
        throw StateError('GPU resource retirement belongs to another epoch');
      }
    }
    checkEpoch();
    try {
      await waitForSubmittedWork();
      return GpuWorkRetirement.completed;
    } catch (_) {
      checkEpoch();
      // A plain queue/validation failure on a healthy device is not authority
      // to destroy it. The current-device query also delivers observed loss.
      if (!_deviceInvalidated) isDeviceCurrent;
      final generation = _expectedPlatformDeviceGeneration;
      if (!_deviceInvalidated || generation == null) rethrow;
      final retired = _lostWorkRetirement ??= Future<void>.sync(
        () => _platform.retireLostDevice(deviceGeneration: generation),
      );
      await retired;
      checkEpoch();
      return GpuWorkRetirement.deviceAborted;
    }
  }

  /// Observe final backend context teardown without retaining another owner.
  /// Register while attached, then retain this future across [destroy]. It
  /// joins all module/native attachments to this exact generation, including workers
  /// and exports. It neither stops them nor authorizes replacement/admission.
  ///
  /// Only the root recovery host should await this after local owners detach;
  /// waiting inside a worker Stop that the root is joining would deadlock.
  /// Failure remains failure. Backends without a final receipt refuse it.
  Future<void> observeContextRetirement({required int deviceEpoch}) {
    if (deviceEpoch != _deviceEpoch) {
      return Future.error(StateError('Context retirement belongs to another epoch'));
    }
    final existing = _contextRetirement;
    if (existing != null) return existing;
    final generation = _expectedPlatformDeviceGeneration;
    if (!isInitialized || _closing || generation == null) {
      return Future.error(StateError('Context retirement must be observed before detach'));
    }
    final result = _contextRetirement = Future<void>.sync(
      () => _platform.observeContextRetirement(deviceGeneration: generation),
    );
    // Observation may fail before the host has finished joining local owners.
    // Awaiting the retained future still reports that failure.
    result.ignore();
    return result;
  }

  /// Local physical adapter/backend identity captured at initialization.
  /// Unsupported platforms return null; names are never identity evidence.
  DeviceAdapterIdentity? get adapterIdentity => _creationAdapterIdentity;

  /// Exact host-local recovery scope. Browser targets promise compatible
  /// capabilities on a browser-selected device, not the same physical adapter.
  DeviceRecoveryTarget? get recoveryTarget => _creationRecoveryTarget;

  GpuRecoverySnapshot get recoveryStatus => _recovery?.snapshot ??
      GpuRecoverySnapshot(GpuRecoveryPhase.idle, deviceEpoch: _deviceEpoch);

  /// Observes host recovery without granting authority to reset a device.
  /// Listeners run after every consumer has been fenced on loss. They must
  /// recheck their own mount, release, audience and source grants on recovery.
  void Function() onRecoveryChanged(void Function(GpuRecoverySnapshot) listener) {
    final token = Object();
    _recoveryListeners[token] = listener;
    return () => _recoveryListeners.remove(token);
  }

  /// Trusted root operation, not a per-fixture retry. Joins every registered
  /// consumer and final context before replacing the original recovery target:
  /// the same native physical adapter, or a capability-compatible browser device.
  /// Never restarts capture or changes simulation authority.
  /// Repeated calls for this lost epoch share the same outcome; failures remain
  /// retained. Unsupported platforms and healthy/stale devices refuse.
  Future<int> recoverLostDevice({required int deviceEpoch}) {
    final previous = _recovery;
    if (previous != null && previous.epoch == deviceEpoch) return previous.done.future;
    if (previous != null && (previous.pending || previous.failed)) {
      return Future.error(StateError('Previous GPU recovery has not succeeded'));
    }
    if (deviceEpoch != _deviceEpoch || !isInitialized || isDeviceCurrent) {
      return Future.error(StateError('Recovery requires the current lost device'));
    }
    final operation = _recovery = _GpuDeviceRecovery(
      this, deviceEpoch, _creationRecoveryTarget,
    );
    operation.start();
    return operation.done.future;
  }

  /// Explicit root retry of a conclusively discarded replacement. Ordinary
  /// [init] and [recoverLostDevice] retain failures. The epoch and failed
  /// attempt fence stale UI actions; concurrent requests for the same attempt
  /// share one result. No reader, worker or context retirement is skipped.
  Future<int> retryLostDevice({
    required int deviceEpoch,
    required int failedAttempt,
  }) {
    final previous = _recovery;
    if (previous == null || previous.epoch != deviceEpoch) {
      return Future.error(StateError('GPU retry refers to a stale device'));
    }
    // Joining an already accepted retry is safe even after its completion.
    if (previous.attempt > 1 && previous.attempt == failedAttempt + 1) {
      return previous.done.future;
    }
    if (previous.pending || previous.attempt != failedAttempt ||
        !previous.snapshot.canRetry || previous.cancelled ||
        deviceEpoch != _deviceEpoch || isInitialized || _initializing != null) {
      return Future.error(StateError('GPU replacement retirement is unconfirmed'));
    }
    final operation = _recovery = _GpuDeviceRecovery.retry(previous);
    operation.start();
    return operation.done.future;
  }

  /// Name of the adapter THIS instance selected — meaningful for
  /// [Minigpu.forAdapter] contexts, where the process-global
  /// [selectedAdapterName] refers to the default context.
  String? get adapterName => _platform.selectedAdapterName;

  /// Borrow this instance's backend for trusted host-native integration. The
  /// host must not destroy the backend or bypass this instance's resource
  /// lifetime. Native consumers retain their own exact context/device lease
  /// inside the backend's synchronous start callback before moving work to a
  /// thread. This is not an interface for shader/guest code or downloaded apps.
  T withPlatformForHostInterop<T>(T Function(MinigpuPlatform) borrow) {
    if (!isInitialized || _closing) throw MinigpuNotInitializedError();
    return borrow(_platform);
  }

  /// Enumerates hardware adapters with total/used dedicated VRAM (empty on
  /// platforms without adapter enumeration).  Static: this queries the OS,
  /// not any particular context.
  static List<GpuAdapterInfo> listAdapters() =>
      MinigpuPlatform.instance.listAdapters();

  /// Synchronous variant of [destroy] for teardown paths that cannot `await`.
  ///
  /// Begins the same teardown as [destroy] without waiting for completion.
  ///
  /// A following [init] waits for the pending teardown. Await [destroy] when
  /// the caller needs its completion or cleanup error.
  ///
  /// Not a hot-reload hook: [State.reassemble] runs on hot RELOAD, where the
  /// app's GPU objects survive and nothing re-inits the context, so calling
  /// this there leaves every later dispatch failing until a restart. A hot
  /// RESTART never runs `reassemble`.
  void destroySync() {
    if (!isInitialized && _initializing == null && _retiring == null &&
        _recovery?.pending != true && _recovery?.snapshot.canRetry != true) return;
    destroy().ignore();
  }

  /// Creates a compute shader.
  ComputeShader createComputeShader() {
    _checkCanCreate();
    final platformShader = _platform.createComputeShader();
    late final WeakReference<ComputeShader> registration;
    final retirement = GpuResourceRetirement(platformShader.destroy, () {
      if (_shaderRegistry.remove(registration) != null) _shaderRetireCount++;
    });
    final shader = CachedComputeShader(platformShader, this, retirement);
    registration = WeakReference(shader);
    _shaderRegistry[registration] = retirement;
    _shaderCreateCount++;
    return shader;
  }

  GpuTexture createTexture(TextureDescriptor descriptor) {
    _checkCanCreate();
    descriptor.validate();
    return _registerTexture(_platform.createTexture(descriptor), descriptor);
  }

  GpuSampler createSampler(SamplerDescriptor descriptor) {
    _checkCanCreate();
    descriptor.validate();
    return _registerSampler(_platform.createSampler(descriptor), descriptor);
  }

  Future<GpuRenderPipeline> createRenderPipeline(
    RenderPipelineDescriptor descriptor,
  ) async {
    _checkCanCreate();
    descriptor.validate();
    final platformPipeline = _platform.createRenderPipeline(descriptor);
    late final WeakReference<GpuRenderPipeline> registration;
    final retirement = GpuResourceRetirement(platformPipeline.destroy, () {
      if (_renderPipelineRegistry.remove(registration) != null) {
        _renderPipelineRetireCount++;
      }
    });
    final pipeline = GpuRenderPipeline(
      platformPipeline,
      descriptor,
      null,
      retirement,
    );
    registration = WeakReference(pipeline);
    _renderPipelineRegistry[registration] = retirement;
    _renderPipelineCreateCount++;
    try {
      await pipeline.prepare();
      if (!pipeline.isValid) throw StateError('Render pipeline was destroyed');
      return pipeline;
    } catch (_) {
      pipeline.destroy();
      rethrow;
    }
  }

  /// Called by [ComputeShader.destroy]. [liveShaderCount] already dropped when
  /// the shader's retirement ran; this forgets the cached copy shader.
  // Accessed from compute_shader.dart (same package — library-private is fine).
  // ignore: use_setters_to_change_properties
  void onShaderDestroyed(ComputeShader shader) {
    if (identical(shader, _copyShader)) _copyShader = null;
  }

  /// Destroys all currently tracked live shaders and clears the tracking list.
  ///
  /// Call this between test groups (in `setUpAll`) to free D3D12 pipeline
  /// resources accumulated by previous tests. Safe to call when no GPU work
  /// is pending.
  void destroyAllTrackedShaders() {
    _retireResources(_shaderRegistry, (shader) => shader.destroy(), (_, _) {});
  }

  /// Creates a buffer.
  Buffer createBuffer(int bufferSize, BufferDataType dataType) {
    _checkCanCreate();

    final platformBuffer = _platform.createBuffer(bufferSize, dataType);
    return _registerBuffer(
      platformBuffer,
      null,
      bufferSize,
      dataType.bytesPerElement,
    );
  }

  Buffer createBufferFromDescriptor(BufferDescriptor descriptor) {
    _checkCanCreate();
    descriptor.validate();
    return _registerBuffer(
      _platform.createBufferFromDescriptor(descriptor),
      descriptor,
      descriptor.sizeBytes,
    );
  }

  /// [sizeBytes] is null when the creator cannot know it; [elementBytes] is
  /// the element size [copyBuffer] counts in (4 for untyped buffers).
  Buffer _registerBuffer(
    PlatformBuffer platformBuffer, [
    BufferDescriptor? descriptor,
    int? sizeBytes,
    int elementBytes = 4,
  ]) {
    late final WeakReference<Buffer> registration;
    final retirement = GpuResourceRetirement(platformBuffer.destroy, () {
      if (_liveBuffers.remove(registration) != null) _bufferRetireCount++;
    });
    final buffer = Buffer(platformBuffer, null, descriptor, retirement);
    registration = WeakReference(buffer);
    _liveBuffers[registration] = retirement;
    _bufferInfo[buffer] = _BufferInfo(_deviceEpoch, sizeBytes, elementBytes);
    _bufferCreateCount++;
    return buffer;
  }

  /// O(1) provenance for [ownsBuffer] and sizes for [copyBuffer], like
  /// [_textureOwnership] for textures.
  final _bufferInfo = Expando<_BufferInfo>();

  GpuTexture _registerTexture(
    PlatformTexture platformTexture,
    TextureDescriptor descriptor,
  ) {
    late final WeakReference<GpuTexture> registration;
    final retirement = GpuResourceRetirement(platformTexture.destroy, () {
      if (_liveTextures.remove(registration) != null) _textureRetireCount++;
    });
    final texture = GpuTexture(
      platformTexture,
      descriptor,
      null,
      _registerTextureView,
      retirement,
    );
    registration = WeakReference(texture);
    _liveTextures[registration] = retirement;
    _textureOwnership[texture] = _deviceEpoch;
    _textureCreateCount++;
    return texture;
  }

  final _textureViewOwnership = Expando<bool>();
  final _textureOwnership = Expando<int>();

  /// Live texture created by this device epoch. This is not an audience grant.
  bool ownsTexture(GpuTexture texture) => isDeviceCurrent && texture.isValid &&
      _textureOwnership[texture] == _deviceEpoch;

  /// Whether a live texture view was created by this device context. This is
  /// process-local validation, not permission to consume another owner's data.
  bool ownsTextureView(GpuTextureView view) =>
      isInitialized && view.isValid && _textureViewOwnership[view] == true;

  GpuTextureView _registerTextureView(
    PlatformTextureView platformView,
    GpuTexture texture,
    TextureViewDescriptor descriptor,
  ) {
    late final WeakReference<GpuTextureView> registration;
    final retirement = GpuResourceRetirement(platformView.destroy, () {
      if (_liveTextureViews.remove(registration) != null) {
        _textureViewRetireCount++;
      }
    });
    final view = GpuTextureView(
      platformView,
      texture,
      descriptor,
      null,
      retirement,
    );
    registration = WeakReference(view);
    _liveTextureViews[registration] = retirement;
    _textureViewCreateCount++;
    _textureViewOwnership[view] = true;
    return view;
  }

  GpuSampler _registerSampler(
    PlatformSampler platformSampler,
    SamplerDescriptor descriptor,
  ) {
    late final WeakReference<GpuSampler> registration;
    final retirement = GpuResourceRetirement(platformSampler.destroy, () {
      if (_liveSamplers.remove(registration) != null) _samplerRetireCount++;
    });
    final sampler = GpuSampler(platformSampler, descriptor, null, retirement);
    registration = WeakReference(sampler);
    _liveSamplers[registration] = retirement;
    _samplerCreateCount++;
    return sampler;
  }

  void _retireResources<T extends Object>(
    Map<WeakReference<T>, GpuResourceRetirement> registrations,
    void Function(T resource) destroy,
    void Function(Object error, StackTrace stackTrace) recordFailure,
  ) {
    for (final entry
        in List<MapEntry<WeakReference<T>, GpuResourceRetirement>>.from(
          registrations.entries,
        )) {
      try {
        final resource = entry.key.target;
        if (resource == null) {
          entry.value.retire();
        } else {
          destroy(resource);
        }
      } catch (error, stackTrace) {
        recordFailure(error, stackTrace);
      }
    }
  }

  void _checkCanCreate() {
    if (!isInitialized || _closing) throw MinigpuNotInitializedError();
    if (_deviceInvalidated || _coordinator?._accepting == false) {
      throw StateError('GPU device is not accepting new resources');
    }
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
    _checkCanCreate();
    final platformTex = _platform.importVideoFrame(buf);
    if (platformTex == null) return null;
    late final WeakReference<VideoTexture> registration;
    final retirement = GpuResourceRetirement(platformTex.destroy, () {
      _liveVideoTextures.remove(registration);
    });
    // toRGBA() buffers register like createBuffer's, so they are counted,
    // owned and retired with this context too. An async conversion can finish
    // after this device retired, and then its buffer belongs to no context.
    final epoch = _deviceEpoch;
    Buffer adopt(PlatformBuffer platformBuffer) {
      if (_deviceEpoch != epoch || !isInitialized || _closing ||
          _deviceInvalidated) {
        platformBuffer.destroy();
        throw StateError('toRGBA() finished on a retired GPU device');
      }
      return _registerBuffer(platformBuffer);
    }
    final texture = VideoTexture(platformTex, this, retirement, adopt);
    registration = WeakReference(texture);
    _liveVideoTextures[registration] = retirement;
    return texture;
  }

  /// Create a cross-API shared output RGBA8 texture suitable for zero-copy
  /// hand-off to an external API (currently: Windows D3D12<->D3D11 via NT
  /// shared handle). Returns null if unsupported on this platform/backend.
  SharedOutputTexture? createSharedOutputTexture(int width, int height) {
    _checkCanCreate();
    final platformTex = _platform.createSharedOutputTexture(width, height);
    if (platformTex == null) return null;
    late final WeakReference<SharedOutputTexture> registration;
    final retirement = GpuResourceRetirement(platformTex.destroy, () {
      _liveSharedOutputTextures.remove(registration);
    });
    final texture = SharedOutputTexture(platformTex, this, retirement);
    registration = WeakReference(texture);
    _liveSharedOutputTextures[registration] = retirement;
    return texture;
  }

  /// Returns an `ID3D11Device*` (as an integer address) created on the same
  /// DXGI adapter as Dawn's internal D3D12 device.  Non-zero only on Windows
  /// with Dawn's D3D12 backend.  Ownership is transferred to the caller —
  /// FFmpeg will call `Release()` when the AVHWDeviceContext is freed.
  int createD3D11DeviceOnDawnAdapter() {
    _checkCanCreate();
    return _platform.createD3D11DeviceOnDawnAdapter();
  }

  // -------------------------------------------------------------------------
  // Buffer copy
  // -------------------------------------------------------------------------

  /// WGSL kernel: copies `params[0]` whole u32 words, then the low
  /// `params[1]` bytes of the next word, from src to dst. `params[2]` is the
  /// threads per dispatch row, so a copy past 65535 groups folds into y.
  static const _kCopyWgsl = r'''
@group(0) @binding(0) var<storage, read_write> src : array<u32>;
@group(0) @binding(1) var<storage, read_write> dst : array<u32>;
@group(0) @binding(2) var<storage, read_write> params : array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.y * params[2] + gid.x;
  if (i >= arrayLength(&src) || i >= arrayLength(&dst)) { return; }
  if (i < params[0]) {
    dst[i] = src[i];
  } else if (i == params[0] && params[1] != 0u) {
    let mask = (1u << (params[1] * 8u)) - 1u;
    dst[i] = (dst[i] & ~mask) | (src[i] & mask);
  }
}
''';

  /// Lazily-created shader for [copyBuffer].
  ComputeShader? _copyShader;

  /// [copyBuffer]'s params buffers. A dispatch reads its params when it runs,
  /// after the call returned, and a buffer write is not ordered behind queued
  /// dispatches, so each copy in flight owns one until its dispatch completed.
  /// Kept out of [liveBufferCount] like [rebindSentinel]; [destroy] frees them.
  final _copyParams = <Buffer>[];
  final _copyParamsFree = <Buffer>[];

  /// GPU-side buffer copy — copies the first [elementCount] elements of [src]
  /// to the start of [dst] entirely on the GPU using a WGSL compute shader
  /// (no CPU round-trip).
  ///
  /// Both buffers must have been created on this [Minigpu] instance with
  /// storage usage. Elements are counted in [src]'s data type (4 bytes for a
  /// buffer made from a descriptor) and the copy is byte-exact: nothing of
  /// [dst] past those bytes is written, also when they end inside a 4-byte
  /// word. Throws [RangeError] when [elementCount] is negative or exceeds the
  /// size either buffer was created with. Copying a buffer onto itself is a
  /// no-op.
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
    if (src.descriptor != null &&
        src.descriptor!.usage & BufferUsage.storage == 0) {
      throw UnsupportedError(
        'Source buffer was not created with storage usage',
      );
    }
    if (dst.descriptor != null &&
        dst.descriptor!.usage & BufferUsage.storage == 0) {
      throw UnsupportedError(
        'Destination buffer was not created with storage usage',
      );
    }
    if (!src.isValid || !dst.isValid) {
      throw StateError('Cannot copy from or to a destroyed buffer');
    }
    if (elementCount < 0) {
      throw RangeError.value(elementCount, 'elementCount', 'Must be >= 0');
    }
    final byteCount = elementCount * (_bufferInfo[src]?.elementBytes ?? 4);
    void checkFits(Buffer buffer, String name) {
      final sizeBytes = _bufferInfo[buffer]?.sizeBytes;
      if (sizeBytes != null && byteCount > sizeBytes) {
        throw RangeError(
          'elementCount $elementCount is $byteCount bytes, more than the '
          '$sizeBytes bytes of $name',
        );
      }
    }

    checkFits(src, 'src');
    checkFits(dst, 'dst');
    // Both bindings are writable, and WebGPU rejects aliasing them.
    if (byteCount == 0 || identical(src, dst)) return;

    final words = byteCount ~/ 4;
    final tail = byteCount % 4;
    final groups = (words + (tail == 0 ? 0 : 1) + 63) ~/ 64;
    const maxGroups = ComputeShader.maxWorkgroupsPerDim;
    final groupsX = groups < maxGroups ? groups : maxGroups;
    final groupsY = (groups + groupsX - 1) ~/ groupsX;

    final shader = _copyShader ??= createComputeShader()
      ..loadKernelString(_kCopyWgsl);
    final Buffer params;
    if (_copyParamsFree.isNotEmpty) {
      params = _copyParamsFree.removeLast();
    } else {
      _checkCanCreate();
      params = Buffer(_platform.createBuffer(16, BufferDataType.uint32));
      _copyParams.add(params);
    }
    try {
      // Issued together with no await between them, so a copy that is not
      // awaited cannot have its bindings replaced before it dispatches.
      shader
        ..setBufferAtSlot(0, src)
        ..setBufferAtSlot(1, dst)
        ..setBufferAtSlot(2, params);
      final written = params.write(
        Uint32List.fromList([words, tail, groupsX * 64, 0]),
        4,
      );
      final dispatched = shader.dispatch(groupsX, groupsY, 1);
      await Future.wait([written, dispatched]);
    } finally {
      if (params.isValid) _copyParamsFree.add(params);
    }
  }
}

/// What a [Minigpu] recorded when it created a buffer.
final class _BufferInfo {
  const _BufferInfo(this.epoch, this.sizeBytes, this.elementBytes);

  final int epoch;
  final int? sizeBytes;
  final int elementBytes;
}
