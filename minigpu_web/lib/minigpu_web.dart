import 'dart:async';
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:minigpu_web/bindings/minigpu_bindings.dart' as wasm;

MinigpuPlatform registeredInstance() => MinigpuWeb._();

void _checkTextureStatus(int status, String operation) {
  switch (status) {
    case 0:
      return;
    case 1:
      throw ArgumentError('$operation was rejected');
    case 2:
      throw StateError('GPU device is unavailable');
    default:
      throw StateError('$operation failed');
  }
}

class MinigpuWeb extends MinigpuPlatform {
  MinigpuWeb._();

  JSObject? _ownedDevice;
  Future<void>? _ownedDeviceLost, _lostDeviceRetirement;
  wasm.WebContextRetirement? _contextObservation;
  Future<void>? _contextRetirement, _detaching;
  BrowserRecoveryTarget? _recoveryTarget;
  bool _contextRetired = false, _starting = false, _attached = false;
  int _deviceGeneration = 0;
  bool _deviceLost = true;
  bool _observedDeviceLoss = false;
  void Function()? _deviceLostCallback;

  @override
  void setDeviceLostCallback(void Function()? callback) =>
      _deviceLostCallback = callback;

  @override
  DeviceRecoveryTarget? get creationRecoveryTarget => _recoveryTarget;

  void _markDeviceLost(int generation, {bool observed = false}) {
    if (generation != _deviceGeneration) return;
    if (observed) _observedDeviceLoss = true;
    if (_deviceLost) return;
    _deviceLost = true;
    // Retain this exact creation device for resource retirement. Its identity
    // is not permission to use it: all readiness queries remain fenced below.
    _deviceLostCallback?.call();
  }

  @override
  DeviceBufferLimits? get currentDeviceBufferLimits {
    final device = _ownedDevice;
    if (_deviceLost || device == null) return null;
    // Dart/Wasm may wrap the same JS device in different Dart objects.
    // Compare the underlying JS identity without weakening device ownership.
    if (_contextObservation?.ownsCurrentModule == false ||
        !device.strictEquals(wasm.mgpuGetOwnDeviceObject()).toDart) {
      _markDeviceLost(_deviceGeneration);
      return null;
    }
    final values = wasm.mgpuReadOwnDeviceBufferLimits(device);
    if (values == null) {
      _markDeviceLost(_deviceGeneration);
      return null;
    }
    return DeviceBufferLimits(
      deviceGeneration: _deviceGeneration,
      maxBufferSize: values.$1,
      maxStorageBufferBindingSize: values.$2,
    );
  }

  static void registerWith(dynamic _) => MinigpuWeb._();

  @override
  DeviceTextureLimits? get currentDeviceTextureLimits {
    if (currentDeviceBufferLimits == null) return null;
    final values = wasm.mgpuReadOwnDeviceTextureLimits(_ownedDevice!);
    if (values == null) return null;
    return DeviceTextureLimits(
      deviceGeneration: _deviceGeneration,
      maxTextureDimension2D: values.$1,
      maxTextureArrayLayers: values.$2,
      supportsRegionWrite: wasm.hasTextureRegionUploadV1,
    );
  }

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
    int level = 2,
  }) {
    // Only the LEVEL is honored on web: routing native log lines through a
    // Dart callback needs a wasm function-pointer table slot
    // (Emscripten addFunction), which this build does not reserve. Messages
    // at/above [level] go to the browser console via stderr, as before.
    _pendingLogLevel = level;
    if (wasm.isMinigpuModuleLoaded) wasm.mgpuSetLogLevel(level);
  }

  @override
  Future<void> initializeContext() => _initializeContext();

  @override
  Future<void> initializeContextForReplacement(DeviceRecoveryTarget expected) {
    if (expected is! BrowserRecoveryTarget ||
        !identical(expected, _recoveryTarget) || !_contextRetired ||
        _contextObservation?.isRetired != true ||
        _contextObservation?.ownsCurrentModule != true) {
      return Future.error(StateError(
        'Browser replacement requires its exact retired module and target',
      ));
    }
    return _initializeContext(expected: expected);
  }

  Future<void> _initializeContext({BrowserRecoveryTarget? expected}) async {
    if (_starting || _attached || _ownedDevice != null) {
      throw StateError('The previous WebGPU context must be detached first');
    }
    if (_contextObservation != null &&
        (!_contextRetired || !_contextObservation!.ownsCurrentModule)) {
      throw StateError('The previous WebGPU module has not finally retired');
    }
    _starting = true;
    try {
      await _attachContext(expected);
    } catch (error, stack) {
      // Retry is authorized only by actual candidate teardown. A failed init
      // without a creation receipt remains quarantined, never presumed empty.
      if (_attached && _contextObservation != null) {
        final retired = _observeFinalContext();
        await destroyContext();
        await retired;
        if (expected != null) {
          throw DeviceRecoveryDiscarded(target: expected,
              message: 'Browser replacement was discarded: $error');
        }
      }
      Error.throwWithStackTrace(error, stack);
    } finally {
      _starting = false;
    }
  }

  Future<void> _attachContext(BrowserRecoveryTarget? expected) async {
    final previousModule = _contextObservation;
    _deviceGeneration++;
    _ownedDevice = null;
    _ownedDeviceLost = null;
    _lostDeviceRetirement = null;
    _contextObservation = null;
    _contextRetirement = null;
    _contextRetired = false;
    _detaching = null;
    _deviceLost = true;
    _observedDeviceLoss = false;
    // The level must land BETWEEN module load and context init, or the
    // adapter/device bring-up narrates at the wasm default (INFO).
    await wasm.ensureMinigpuModuleLoaded();
    if (previousModule?.ownsCurrentModule == false) {
      throw StateError('The browser module changed before replacement');
    }
    final lvl = _pendingLogLevel;
    if (lvl != null) wasm.mgpuSetLogLevel(lvl);
    if (previousModule == null) {
      await wasm.mgpuInitializeContext();
    } else {
      await previousModule.attachReplacement();
    }
    _attached = true;
    if (previousModule?.ownsCurrentModule == false) {
      throw StateError('The browser module changed during replacement');
    }
    // 🔴 VERDICT PROBE for the asyncify early-resolution failure: the await
    // above can resolve while the wasm-side init is still suspended in its
    // adapter/device poll loop (observed: "Requesting WebGPU adapter..." is
    // the last init log, yet callers proceed). The device handle is a SYNC
    // export, so reading 0 here is proof the resolution was a lie — and the
    // page's every later async read will "resolve" with zeros the same way.
    int dev;
    try {
      dev = wasm.mgpuGetWGPUDeviceHandle();
    } catch (_) {
      dev = 0;
    }
    if (dev == 0) {
      // ignore: avoid_print
      print(
        '[minigpu_web] 🔴 initializeContext RESOLVED but the WGPUDevice '
        'handle is 0 — the wasm init never actually completed (asyncify '
        'early-resolution); all GPU work on this page will silently '
        'produce zeros',
      );
    }
    final ownDevice = wasm.mgpuGetOwnDeviceObject();
    if (ownDevice != null && wasm.mgpuReadOwnDeviceBufferLimits(ownDevice) != null) {
      final lost = wasm.mgpuOwnDeviceLost(ownDevice);
      if (lost != null) {
        final generation = _deviceGeneration;
        _ownedDevice = ownDevice;
        _ownedDeviceLost = lost;
        _deviceLost = false;
        lost.then((_) => _markDeviceLost(generation, observed: true),
            onError: (Object _) => _markDeviceLost(generation));
        _contextObservation = wasm.mgpuObserveOwnContextRetirement(dev);
        // An older backend may render, but cannot promise recovery safety.
        if (_contextObservation != null) {
          final requirements = wasm.mgpuOwnDeviceRequirements(ownDevice);
          if (expected != null &&
              !expected.accepts(requirements.$1, requirements.$2)) {
            throw StateError('Replacement WebGPU capabilities are insufficient');
          }
          _recoveryTarget = expected ?? BrowserRecoveryTarget(
              features: requirements.$1, limits: requirements.$2);
        } else {
          _recoveryTarget = null;
          if (expected != null) {
            throw StateError('Replacement module lacks final retirement');
          }
        }
        // Deliver already-queued loss before the root publishes a new epoch.
        await Future<void>.delayed(Duration.zero);
        if (_deviceLost) throw StateError('WebGPU was lost during initialization');
        return;
      }
    }
    throw StateError('WebGPU initialization did not produce a usable device');
  }

  @override
  Future<void> observeContextRetirement({required int deviceGeneration}) {
    if (deviceGeneration != _deviceGeneration || !_attached ||
        _ownedDevice == null || _contextObservation == null) {
      return Future.error(StateError(
        'Final browser retirement must be observed on the creation context',
      ));
    }
    return _observeFinalContext();
  }

  Future<void> _observeFinalContext() {
    final existing = _contextRetirement;
    if (existing != null) return existing;
    final observation = _contextObservation!;
    final device = _ownedDevice!, lost = _ownedDeviceLost!;
    final retired = _contextRetirement = Future<void>.sync(() async {
      await observation.completed;
      // A returned attachment alone must never destroy another owner's live
      // device. Only the last module detach permits this final GPU boundary.
      wasm.mgpuDestroyOwnDevice(device);
      await lost;
      if (!identical(observation, _contextObservation)) {
        throw StateError('Browser retirement belongs to an obsolete module');
      }
      _contextRetired = true;
    });
    retired.ignore();
    return retired;
  }

  /// A browser loss receipt certifies ended GPU access, not successful frames,
  /// host-reader cleanup, module teardown or physical-adapter identity.
  @override
  Future<void> retireLostDevice({required int deviceGeneration}) {
    final device = _ownedDevice, lost = _ownedDeviceLost;
    if (deviceGeneration != _deviceGeneration || !_observedDeviceLoss ||
        !_deviceLost ||
        device == null || lost == null) {
      return Future.error(StateError(
        'WebGPU retirement requires the exact observed lost creation device',
      ));
    }
    return _lostDeviceRetirement ??= Future<void>.sync(() async {
      // Never resolve a device from the mutable module/global handle here.
      // destroy() is idempotent and prevents further device operations; await
      // this device's loss promise as the GPU-only abort boundary. A rejected
      // promise remains failure, and does not authorize a replacement.
      wasm.mgpuDestroyOwnDevice(device);
      await lost;
      if (deviceGeneration != _deviceGeneration ||
          !identical(device, _ownedDevice) || !_deviceLost) {
        throw StateError('WebGPU retirement belongs to a detached context');
      }
    });
  }

  @override
  Future<void> destroyContext() => _detaching ??= _detachContext();

  Future<void> _detachContext() async {
    // Keep the retained creation device stable through admitted GPU retirement.
    // Failure remains quarantined; detaching cannot convert it into success.
    await _lostDeviceRetirement;
    if (!_attached) throw StateError('No browser context is attached');
    final observation = _contextObservation;
    if (observation != null) _observeFinalContext();
    // Fence late callbacks before returning our attachment. The final receipt
    // remains available independently and does not hold another attachment.
    _deviceGeneration++;
    _deviceLost = true;
    _observedDeviceLoss = false;
    if (observation != null) {
      await observation.detach();
    } else {
      await wasm.mgpuDestroyContext();
    }
    _attached = false;
    _ownedDevice = null;
    _ownedDeviceLost = null;
    _lostDeviceRetirement = null;
    // A last detach has already settled the module. Join its receipt now so
    // ordinary close-then-init does not race the observer's next timer turn.
    // A secondary attachment returns promptly while other owners remain.
    if (observation?.hasSettled == true) await _contextRetirement;
  }

  @override
  Future<void> waitForSubmittedWork() {
    wasm.mgpuFlushBatch();
    return wasm.mgpuWaitForSubmittedWork();
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
  PlatformBuffer createBufferFromDescriptor(BufferDescriptor descriptor) {
    descriptor.validate();
    if (wasm.hasBufferDescriptorV1) {
      final explicit = wasm.mgpuCreateBufferWithUsage(
        descriptor.sizeBytes,
        descriptor.usage,
      );
      if (explicit == null) throw MinigpuPlatformOutOfMemoryException();
      return WebBuffer(explicit);
    }
    if (descriptor.usage == BufferUsage.legacy) {
      return createBuffer(descriptor.sizeBytes, BufferDataType.uint32);
    }
    throw UnsupportedError(
      'Explicit buffer usage is unsupported by this binary',
    );
  }

  @override
  PlatformTexture createTexture(TextureDescriptor descriptor) {
    if (!wasm.hasTextureResourceV1) {
      throw UnsupportedError('Owned textures are unsupported by this binary');
    }
    final texture = wasm.mgpuCreateTexture2D(
      descriptor.width,
      descriptor.height,
      descriptor.format.wireCode,
      descriptor.usage,
      descriptor.arrayLayerCount,
      descriptor.mipLevelCount,
    );
    if (texture == null) throw StateError('Texture creation failed');
    return WebTexture(texture);
  }

  @override
  PlatformSampler createSampler(SamplerDescriptor descriptor) {
    if (!wasm.hasTextureResourceV1) {
      throw UnsupportedError('Owned samplers are unsupported by this binary');
    }
    final sampler = wasm.mgpuCreateSampler(
      descriptor.addressModeU.wireCode,
      descriptor.addressModeV.wireCode,
      descriptor.addressModeW.wireCode,
      descriptor.magFilter.wireCode,
      descriptor.minFilter.wireCode,
      descriptor.mipmapFilter.wireCode,
      descriptor.lodMinClamp,
      descriptor.lodMaxClamp,
      descriptor.compare?.wireCode ?? 0,
    );
    if (sampler == null) throw StateError('Sampler creation failed');
    return WebSampler(sampler);
  }

  @override
  PlatformRenderPipeline createRenderPipeline(
    RenderPipelineDescriptor descriptor,
  ) {
    if (!wasm.hasRasterResourceV1) {
      throw UnsupportedError('Raster pipelines are unsupported by this binary');
    }
    final pipeline = wasm.mgpuCreateRenderPipeline(
      descriptor.code,
      descriptor.vertexEntry,
      descriptor.fragmentEntry,
      descriptor.encode(),
    );
    if (pipeline == null) {
      throw StateError('Render pipeline creation failed');
    }
    return WebRenderPipeline(pipeline);
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
  @override
  void releaseTextureBindingAfterGpuRead(int slot) => throw UnsupportedError(
    'Native input loan cleanup is unavailable on the browser backend',
  );
  final wasm.MGPUComputeShader _shader;
  bool _destroyed = false;

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
  void setBufferRange(
    int slot,
    PlatformBuffer buffer,
    BufferBindingDescriptor descriptor,
  ) {
    if (!wasm.hasBufferBindingV1) {
      throw UnsupportedError(
        'Explicit buffer ranges are unsupported by this binary',
      );
    }
    final result = wasm.mgpuSetBufferRange(
      _shader,
      slot,
      (buffer as WebBuffer)._buffer,
      descriptor.kind.wireCode,
      descriptor.offsetBytes,
      descriptor.sizeBytes,
    );
    switch (result) {
      case 0:
        return;
      case 1:
        throw ArgumentError('Invalid buffer range binding');
      case 2:
        throw StateError('GPU device is unavailable');
      default:
        throw StateError('Buffer range binding failed');
    }
  }

  @override
  void setTexture(int slot, PlatformTextureView view, TextureBindingKind kind) {
    if (_destroyed) {
      throw StateError('Compute shader is destroyed');
    }
    if (!wasm.hasTextureResourceV1) {
      throw UnsupportedError('Owned textures are unsupported by this binary');
    }
    if (view is! WebTextureView) {
      throw ArgumentError.value(view, 'view', 'Expected a web texture view');
    }
    if (view._destroyed) {
      throw StateError('Texture view is destroyed');
    }
    final result = wasm.mgpuSetOwnedTexture(
      _shader,
      slot,
      (view as WebTextureView)._view,
      kind.wireCode,
    );
    _checkTextureStatus(result, 'Texture binding');
  }

  @override
  void setSampler(int slot, PlatformSampler sampler, SamplerBindingKind kind) {
    if (_destroyed) {
      throw StateError('Compute shader is destroyed');
    }
    if (!wasm.hasTextureResourceV1) {
      throw UnsupportedError('Owned samplers are unsupported by this binary');
    }
    if (sampler is! WebSampler) {
      throw ArgumentError.value(sampler, 'sampler', 'Expected a web sampler');
    }
    if (sampler._destroyed) {
      throw StateError('Sampler is destroyed');
    }
    final result = wasm.mgpuSetOwnedSampler(
      _shader,
      slot,
      (sampler as WebSampler)._sampler,
      kind.wireCode,
    );
    _checkTextureStatus(result, 'Sampler binding');
  }

  @override
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ) async {
    await wasm.mgpuDispatch(_shader, groupsX, groupsY, groupsZ);
  }

  @override
  Future<void> buildPipeline() async {
    // A module from before the export warms up the old way, one workgroup.
    if (!wasm.hasComputePipelineBuild) return dispatch(1, 1, 1);
    if (!wasm.mgpuComputeShaderBuildPipeline(_shader)) {
      throw StateError(
        'Compute pipeline was not built: no kernel or bindings, the shader is '
        'closed, or the pipeline could not be created',
      );
    }
  }

  /// One-shot marker: a fired dispatch that REJECTED. Static — one report
  /// per page is enough to name the failure.
  static bool _saidFireFailed = false;

  @override
  void dispatchFire(int groupsX, int groupsY, int groupsZ) {
    // The dispatch is recorded into minigpu's open batch synchronously, inside
    // this call (a direct export call that never suspends); the returned
    // future only carries a failure, so dropping it keeps program order.
    //
    // 🔴 "Dropping it" must still HANDLE rejection: an uncaught C++ exception
    // in the wasm surfaces as a bare number (the exception pointer) on this
    // promise, and unhandled it kills the surrounding Dart zone with no
    // frames pointing anywhere — that is exactly the shape of an entire test
    // failing with just "67157776" and zone plumbing for a stack.
    wasm.mgpuDispatch(_shader, groupsX, groupsY, groupsZ).catchError((
      Object e,
    ) {
      if (!_saidFireFailed) {
        _saidFireFailed = true;
        // ignore: avoid_print
        print(
          '[minigpu_web] fired dispatch($groupsX,$groupsY,$groupsZ) '
          'REJECTED: $e — a bare number is a wasm C++ exception pointer; '
          'raise the log level (Minigpu.setLogCallback level 0/1) to see '
          'the native narration up to the throw',
        );
      }
    });
  }

  @override
  void destroy() {
    if (_destroyed) {
      return;
    }
    _destroyed = true;
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

class WebTexture implements PlatformTexture {
  WebTexture(this._texture);

  final wasm.MGPUTexture _texture;
  bool _destroyed = false;

  @override
  bool get supportsRegionWrite => !_destroyed && wasm.hasTextureRegionUploadV1;

  @override
  PlatformTextureView createView(TextureViewDescriptor descriptor) {
    _checkLive();
    final view = wasm.mgpuCreateTextureView(
      _texture,
      descriptor.dimension.wireCode,
      descriptor.baseMipLevel,
      descriptor.mipLevelCount,
      descriptor.baseArrayLayer,
      descriptor.arrayLayerCount,
    );
    if (view == null) {
      throw StateError('Texture view creation failed');
    }
    return WebTextureView(view);
  }

  @override
  Future<void> write(
    Uint8List bytes, {
    int mipLevel = 0,
    int arrayLayer = 0,
  }) async {
    _checkLive();
    _checkTextureStatus(
      wasm.mgpuWriteTexture2D(_texture, bytes, mipLevel, arrayLayer),
      'Texture upload',
    );
  }

  @override
  Future<void> writeRegion(
    Uint8List bytes, {
    required int x,
    required int y,
    required int width,
    required int height,
    int mipLevel = 0,
    int arrayLayer = 0,
  }) async {
    _checkLive();
    _checkTextureStatus(
      wasm.mgpuWriteTextureRegion2D(
        _texture,
        bytes,
        x,
        y,
        width,
        height,
        mipLevel,
        arrayLayer,
      ),
      'Texture region upload',
    );
  }

  @override
  void destroy() {
    if (_destroyed) {
      return;
    }
    _destroyed = true;
    wasm.mgpuDestroyTexture(_texture);
  }

  void _checkLive() {
    if (_destroyed) {
      throw StateError('Texture is destroyed');
    }
  }
}

class WebTextureView implements PlatformTextureView {
  WebTextureView(this._view);

  final wasm.MGPUTextureView _view;
  bool _destroyed = false;

  @override
  void destroy() {
    if (_destroyed) {
      return;
    }
    _destroyed = true;
    wasm.mgpuDestroyTextureView(_view);
  }
}

class WebSampler implements PlatformSampler {
  WebSampler(this._sampler);

  final wasm.MGPUSampler _sampler;
  bool _destroyed = false;

  @override
  void destroy() {
    if (_destroyed) {
      return;
    }
    _destroyed = true;
    wasm.mgpuDestroySampler(_sampler);
  }
}

class WebRenderPipeline implements PlatformRenderPipeline {
  WebRenderPipeline(this._pipeline);

  final wasm.MGPURenderPipeline _pipeline;
  bool _destroyed = false;
  Completer<void>? _settled;

  /// Pipelines awaiting their settle, by handle; one module hook serves all.
  static final Map<int, Completer<void>> _settleWaiters = {};

  // The wasm passes the handle as a signed int: compare unsigned.
  int get _handleKey => _pipeline.toDartInt & 0xFFFFFFFF;

  static void _onSettled(int handle) {
    final waiter = _settleWaiters.remove(handle & 0xFFFFFFFF);
    if (waiter != null && !waiter.isCompleted) waiter.complete();
  }

  @override
  int get status => _destroyed ? 3 : wasm.mgpuRenderPipelineStatus(_pipeline);

  @override
  Future<void>? get settled {
    final known = _settled;
    if (known != null) return known.future;
    if (!wasm.hasRenderPipelineSettledSignal) return null;
    final settled = _settled = Completer<void>();
    if (status != 0) {
      settled.complete();
    } else {
      // Set on every registration: cheap, and survives a replaced hook.
      wasm.setRenderPipelineSettledHook(_onSettled);
      _settleWaiters[_handleKey] = settled;
    }
    return settled.future;
  }

  @override
  String get diagnostic => _destroyed
      ? 'Render pipeline is destroyed'
      : wasm.mgpuRenderPipelineDiagnostic(_pipeline);

  @override
  List<int>? get bindingCacheStats {
    if (_destroyed) return null;
    if (!wasm.hasRenderPipelineBindingStats) return null;
    return List<int>.unmodifiable([
      for (var selector = 0; selector < 4; selector++)
        wasm.mgpuRenderPipelineBindingStat(_pipeline, selector),
    ]);
  }

  @override
  Future<void> submitRender({
    required PlatformTextureView color,
    PlatformTextureView? depth,
    required List<double> clear,
    required int loadFlags,
    required List<PlatformBuffer> buffers,
    required List<PlatformTextureView> views,
    required List<PlatformSampler> samplers,
    required List<int> commands,
  }) async {
    if (_destroyed) throw StateError('Render pipeline is destroyed');
    if (color is! WebTextureView ||
        (depth != null && depth is! WebTextureView) ||
        buffers.any((buffer) => buffer is! WebBuffer) ||
        views.any((view) => view is! WebTextureView) ||
        samplers.any((sampler) => sampler is! WebSampler)) {
      throw ArgumentError('Render resources belong to another backend');
    }
    final webColor = color;
    final webDepth = depth as WebTextureView?;
    final webBuffers = buffers.cast<WebBuffer>();
    final webViews = views.cast<WebTextureView>();
    final webSamplers = samplers.cast<WebSampler>();
    if (webColor._destroyed || (webDepth?._destroyed ?? false)) {
      throw StateError('Render attachment view is destroyed');
    }
    if (webBuffers.any((buffer) => buffer._destroyed)) {
      throw StateError('Render buffer is destroyed');
    }
    if (webViews.any((view) => view._destroyed)) {
      throw StateError('Render texture view is destroyed');
    }
    if (webSamplers.any((sampler) => sampler._destroyed)) {
      throw StateError('Render sampler is destroyed');
    }
    final result = wasm.mgpuSubmitRender(
      _pipeline,
      webColor._view,
      webDepth?._view,
      clear,
      loadFlags,
      [for (final buffer in webBuffers) buffer._buffer],
      [for (final view in webViews) view._view],
      [for (final sampler in webSamplers) sampler._sampler],
      commands,
    );
    _checkTextureStatus(result, 'Render submission');
  }

  @override
  void destroy() {
    if (_destroyed) return;
    wasm.mgpuDestroyRenderPipeline(_pipeline);
    _destroyed = true;
    // The close normally fires the hook; never leave a waiter behind, nor
    // drop one a later pipeline at the same handle registered.
    final settled = _settled;
    if (settled == null) return;
    if (identical(_settleWaiters[_handleKey], settled)) {
      _settleWaiters.remove(_handleKey);
    }
    if (!settled.isCompleted) settled.complete();
  }
}

class WebBuffer implements PlatformBuffer {
  final wasm.MGPUBuffer _buffer;
  bool _destroyed = false;

  WebBuffer(this._buffer);

  // 0 once destroyed: the C object may already be freed and its handle reused.
  @override
  int get webBufferHandle =>
      _destroyed ? 0 : wasm.mgpuGetWGPUBufferHandle(_buffer);

  @override
  Future<void> writeRawBytes(Uint8List bytes, {int dstByteOffset = 0}) {
    if (_destroyed) throw StateError('Buffer is destroyed');
    if (bytes.length % 4 != 0 || dstByteOffset % 4 != 0) {
      throw ArgumentError('writeRawBytes needs 4-byte-aligned length/offset');
    }
    if (dstByteOffset < 0) {
      throw RangeError.value(dstByteOffset, 'dstByteOffset', 'Cannot be negative');
    }
    // Every offset, 0 included, goes straight from the caller's bytes to
    // GPUQueue.writeBuffer: no wasm-heap scratch the size of the payload, no
    // C call, and any byte alignment of the caller's list.
    wasm.mgpuWriteRawBytesAt(_buffer, bytes, dstByteOffset);
    return Future.value();
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
    // The C object is freed, or queued to be, once destroy() ran.
    if (_destroyed) throw StateError('Buffer is destroyed');
    // Every reader takes any TypedData and fills it from its start, as the
    // native backend does: a ByteData or another list type gets the bytes.
    switch (dataType) {
      case BufferDataType.int8:
        await wasm.mgpuReadAsyncInt8(
          _buffer,
          outputData,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.int16:
        await wasm.mgpuReadAsyncInt16(
          _buffer,
          outputData,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.int32:
        await wasm.mgpuReadAsyncInt32(
          _buffer,
          outputData,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.int64:
        await wasm.mgpuReadAsyncInt64(
          _buffer,
          outputData,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.uint8:
        await wasm.mgpuReadAsyncUint8(
          _buffer,
          outputData,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.uint16:
        await wasm.mgpuReadAsyncUint16(
          _buffer,
          outputData,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.uint32:
        await wasm.mgpuReadAsyncUint32(
          _buffer,
          outputData,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.uint64:
        await wasm.mgpuReadAsyncUint64(
          _buffer,
          outputData,
          readElements: readElements,
          elementOffset: elementOffset,
        );
        break;
      case BufferDataType.float16:
        throw UnimplementedError('float16 is not supported in WebAssembly.');
      case BufferDataType.float32:
        await wasm.mgpuReadAsyncFloat(
          _buffer,
          outputData,
          readElements: readElements,
          elementOffset: elementOffset,
        );

        break;
      case BufferDataType.float64:
        await wasm.mgpuReadAsyncDouble(
          _buffer,
          outputData,
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
    if (_destroyed) throw StateError('Buffer is destroyed');
    // Any TypedData uploads its bytes as [dataType] elements, as natively: a
    // ByteData is raw bytes for every type, and a list of another element type
    // is reinterpreted rather than dropped.
    switch (dataType) {
      case BufferDataType.int8:
        wasm.mgpuWriteInt8(_buffer, inputData, size);
        break;
      case BufferDataType.int16:
        wasm.mgpuWriteInt16(_buffer, inputData, size);
        break;
      case BufferDataType.int32:
        wasm.mgpuWriteInt32(_buffer, inputData, size);
        break;
      case BufferDataType.int64:
        wasm.mgpuWriteInt64(_buffer, inputData, size);
        break;
      case BufferDataType.uint8:
        wasm.mgpuWriteUint8(_buffer, inputData, size);
        break;
      case BufferDataType.uint16:
        wasm.mgpuWriteUint16(_buffer, inputData, size);
        break;
      case BufferDataType.uint32:
        wasm.mgpuWriteUint32(_buffer, inputData, size);
        break;
      case BufferDataType.uint64:
        wasm.mgpuWriteUint64(_buffer, inputData, size);
        break;
      case BufferDataType.float16:
        throw UnimplementedError('float16 is not supported in WebAssembly.');
      case BufferDataType.float32:
        await wasm.mgpuWriteFloat(_buffer, inputData, size);
        break;
      case BufferDataType.float64:
        wasm.mgpuWriteDouble(_buffer, inputData, size);
        break;
    }
  }

  @override
  void destroy() {
    if (_destroyed) return;
    _destroyed = true;
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
      print(
        '[minigpu] 🔴 WebVideoTexture.setOnShader binds NOTHING on web — '
        'a GPUExternalTexture needs a texture_external kernel and a bind '
        'group built JS-side. Use landPackedRgba8() instead; this pass will '
        'produce no pixels.',
      );
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

  @override
  Future<PlatformBuffer> toRGBAAsync() async => toRGBA();

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
  /// Its WGPUBuffer handle is exposed as [webGpuTextureJs]. Kept as the
  /// [WebBuffer], not its C pointer, so a destroyed one answers no handle.
  WebBuffer? _lastBuffer;

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
    _lastBuffer = src;
    // The consumer blits from the JS GPUBuffer with its own queue.submit, which
    // would run ahead of the dispatches that filled it while they wait in
    // minigpu's open batch. An empty batch submits nothing.
    wasm.mgpuFlushBatch();
    return true;
  }

  /// [copyFromBufferF32] records and flushes the copy at once; there is no GPU
  /// wait to move off the caller.
  @override
  Future<bool> copyFromBufferF32Async(PlatformBuffer src) async =>
      copyFromBufferF32(src);

  /// Returns the Emscripten integer handle for the WGPUBuffer last written
  /// via [copyFromBufferF32].  The view plugin resolves the JS GPUBuffer via
  /// `WebGPU.getJsObject(handle)` locally — integers are codec-safe.
  /// Returns 0 if no buffer has been written yet.
  @override
  Object? get webGpuTextureJs {
    final buf = _lastBuffer;
    if (buf == null) return null;
    final handle = buf.webBufferHandle;
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
