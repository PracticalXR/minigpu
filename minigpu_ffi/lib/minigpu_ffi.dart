// ignore_for_file: omit_local_variable_types

import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:minigpu_ffi/minigpu_ffi_bindings.dart' as ffi;
import 'package:minigpu_ffi/minigpu_ffi_completion.dart';
import 'package:minigpu_ffi/minigpu_ffi_log_port.dart' as log_port;
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

typedef ReadAsyncCallbackFunc = Void Function(Pointer<Void>);
typedef ReadAsyncCallback = Pointer<NativeFunction<ReadAsyncCallbackFunc>>;

MinigpuPlatform registeredInstance() => MinigpuFfi();

DeviceBufferLimits? _readCurrentDeviceBufferLimits(
  int Function(Pointer<Uint64>, Pointer<Uint64>, Pointer<Uint64>) query,
) {
  final values = calloc<Uint64>(3);
  try {
    final ok = query(values, values + 1, values + 2);
    if (ok != 1 || values.value <= 0 || (values + 1).value <= 0 ||
        (values + 2).value <= 0) {
      return null;
    }
    return DeviceBufferLimits(
      deviceGeneration: (values + 2).value,
      maxBufferSize: values.value,
      maxStorageBufferBindingSize: (values + 1).value,
    );
  } on ArgumentError {
    // Older native binaries do not carry the bounded query export.
    return null;
  } finally {
    calloc.free(values);
  }
}

bool _hasBufferDescriptorV1() {
  try {
    return ffi.mgpuBufferDescriptorVersion() == 1;
  } on ArgumentError {
    return false;
  }
}

DeviceTextureLimits? _readCurrentDeviceTextureLimits(
  int Function(Pointer<Uint64>, Pointer<Uint64>, Pointer<Uint64>) query,
) {
  final values = calloc<Uint64>(3);
  try {
    if (query(values, values + 1, values + 2) != 1 ||
        values.value < 1 || (values + 1).value < 1 ||
        (values + 2).value < 1) return null;
    return DeviceTextureLimits(
      deviceGeneration: (values + 2).value,
      maxTextureDimension2D: values.value,
      maxTextureArrayLayers: (values + 1).value,
      supportsRegionWrite: ffi.mgpuTextureRegionUploadVersion() == 1,
    );
  } on ArgumentError {
    return null; // Optional export absent on older packaged backends.
  } finally {
    calloc.free(values);
  }
}

bool _hasBufferBindingV1() {
  try {
    return ffi.mgpuBufferBindingVersion() == 1;
  } on ArgumentError {
    return false;
  }
}

bool _hasTextureResourceV1() {
  try {
    return ffi.mgpuTextureResourceVersion() == 1;
  } on ArgumentError {
    return false;
  }
}

bool _hasRasterResourceV1() {
  try {
    return ffi.mgpuRasterResourceVersion() == 1;
  } on ArgumentError {
    return false;
  }
}

const int _maxRasterDescriptorWords = 4096;
const int _maxRasterCommandWords = 65536;
const int _maxRasterResourcesPerFamily = 256;
const int _maxRasterDiagnosticBytes = 512;
const int _maxUint32 = 0xffffffff;

void _checkUint32Words(List<int> words, int maximum, String name) {
  if (words.length > maximum) {
    throw ArgumentError.value(words.length, name, 'Exceeds $maximum words');
  }
  for (final word in words) {
    if (word < 0 || word > _maxUint32) {
      throw ArgumentError.value(word, name, 'Must contain only uint32 values');
    }
  }
}

void _checkRasterSubmission(
  List<double> clear,
  int loadFlags,
  bool hasDepth,
  List<PlatformBuffer> buffers,
  List<PlatformTextureView> views,
  List<PlatformSampler> samplers,
  List<int> commands,
) {
  if (clear.length != 5) {
    throw ArgumentError.value(clear, 'clear', 'Must contain RGBA and depth');
  }
  final float32 = Float32List(1);
  for (final value in clear) {
    if (!value.isFinite) {
      throw ArgumentError.value(value, 'clear', 'Must be finite');
    }
    float32[0] = value;
    if (!float32[0].isFinite) {
      throw ArgumentError.value(
        value,
        'clear',
        'Must be representable as float32',
      );
    }
  }
  if (clear[4] < 0 || clear[4] > 1) {
    throw ArgumentError.value(clear[4], 'clear', 'Depth must be in 0..1');
  }
  if (loadFlags < 0 || loadFlags > 3 || (!hasDepth && (loadFlags & 2) != 0)) {
    throw ArgumentError.value(
      loadFlags,
      'loadFlags',
      'Invalid render load flags',
    );
  }
  if (buffers.length > _maxRasterResourcesPerFamily ||
      views.length > _maxRasterResourcesPerFamily ||
      samplers.length > _maxRasterResourcesPerFamily) {
    throw ArgumentError('Render resource count exceeds 256');
  }
  _checkUint32Words(commands, _maxRasterCommandWords, 'commands');
}

Never _textureStatus(int status, String operation) {
  switch (status) {
    case 1:
      throw ArgumentError('$operation was rejected');
    case 2:
      throw StateError('GPU device is unavailable');
    default:
      throw StateError('$operation failed');
  }
}

Future<void> _waitForSubmittedWork(PortIssue viaPort) async {
  if (!GpuCompletion.usesPort) {
    throw UnsupportedError(
      'Submitted-work completion requires status-bearing native port support',
    );
  }
  final bool completed;
  try {
    completed = await GpuCompletion.run(
      viaPort,
      (_) => throw UnsupportedError(
        'Submitted-work completion requires status-bearing native port support',
      ),
    );
  } on ArgumentError {
    throw UnsupportedError(
      'This native minigpu build does not expose submitted-work completion',
    );
  }
  if (!completed) {
    throw StateError('The GPU queue was unavailable or the device was lost');
  }
}

PlatformRenderPipeline _createFfiRenderPipeline(
  RenderPipelineDescriptor descriptor,
  Pointer<ffi.MGPUContextHandle>? context,
) {
  if (!_hasRasterResourceV1()) {
    throw UnsupportedError('Raster pipelines are unsupported by this binary');
  }
  final words = descriptor.encode();
  _checkUint32Words(words, _maxRasterDescriptorWords, 'descriptor');
  return using((Arena arena) {
    final code = descriptor.code.toNativeUtf8(allocator: arena).cast<Char>();
    final vertex = descriptor.vertexEntry
        .toNativeUtf8(allocator: arena)
        .cast<Char>();
    final fragment = descriptor.fragmentEntry
        .toNativeUtf8(allocator: arena)
        .cast<Char>();
    final nativeWords = arena<Uint32>(words.length);
    nativeWords.asTypedList(words.length).setAll(0, words);
    final Pointer<ffi.MGPURenderPipeline> pipeline;
    try {
      pipeline = context == null
          ? ffi.mgpuCreateRenderPipeline(
              code,
              vertex,
              fragment,
              nativeWords,
              words.length,
            )
          : ffi.mgpuContextCreateRenderPipeline(
              context,
              code,
              vertex,
              fragment,
              nativeWords,
              words.length,
            );
    } on ArgumentError {
      throw UnsupportedError(
        'This native minigpu build does not expose raster pipeline creation',
      );
    }
    if (pipeline == nullptr) {
      throw StateError('Render pipeline creation failed');
    }
    return FfiRenderPipeline(pipeline);
  });
}

/// Retained while ready, since a lost device cannot issue new creation leases.
/// This owner proves only native GPU retirement; imported/exported readers keep
/// their own pins. Releasing it does not claim the whole context detached.
final class _FfiDeviceEpoch {
  _FfiDeviceEpoch(this._lease, this.generation)
      : adapterIdentity = _readAdapterIdentity(_lease, generation);
  Pointer<NativeContextLease> _lease;
  final int generation;
  final DeviceAdapterIdentity? adapterIdentity;
  Future<void>? _retiring;
  Future<void>? _contextRetirement;

  static DeviceAdapterIdentity? _readAdapterIdentity(
    Pointer<NativeContextLease> lease, int generation,
  ) {
    final out = calloc<NativeDeviceAdapterIdentity>();
    try {
      if (mgpuContextLeaseGetAdapterIdentity(
            lease, out, sizeOf<NativeDeviceAdapterIdentity>()) != 0 ||
          out.ref.version != 1 || out.ref.kind != 1 ||
          out.ref.deviceGeneration != generation ||
          !const {1, 2}.contains(out.ref.backend)) return null;
      return DeviceAdapterIdentity(
        backend: out.ref.backend, kind: out.ref.kind,
        low: out.ref.low, high: out.ref.high,
      );
    } on ArgumentError {
      return null; // Older binaries remain usable, without recovery support.
    } finally {
      calloc.free(out);
    }
  }

  static _FfiDeviceEpoch retain([Pointer<ffi.MGPUContextHandle>? handle]) {
    final lease = calloc<Pointer<NativeContextLease>>();
    final context = calloc<Pointer<ffi.MGPUContextHandle>>();
    final info = calloc<NativeContextLeaseInfo>();
    try {
      final result = handle == null
          ? mgpuDefaultContextRetainLease(lease, context)
          : mgpuContextRetainLease(handle, lease);
      if (result != 0 ||
          mgpuContextLeaseGetInfo(
                lease.value, info, sizeOf<NativeContextLeaseInfo>()) != 0 ||
          info.ref.current != 1) {
        throw StateError('GPU creation device cannot be retained ($result)');
      }
      final retained = _FfiDeviceEpoch(lease.value, info.ref.deviceGeneration);
      lease.value = nullptr;
      return retained;
    } finally {
      if (lease.value != nullptr) mgpuContextReleaseLease(lease.value);
      calloc.free(info);
      calloc.free(context);
      calloc.free(lease);
    }
  }

  Future<void> retireLost(int expectedGeneration) {
    if (_lease == nullptr || expectedGeneration != generation) {
      return Future.error(StateError('GPU retirement creation epoch changed'));
    }
    final existing = _retiring;
    if (existing != null) return existing;
    final info = calloc<NativeContextLeaseInfo>();
    try {
      if (mgpuContextLeaseGetInfo(
                _lease, info, sizeOf<NativeContextLeaseInfo>()) != 0 ||
          info.ref.deviceGeneration != generation ||
          info.ref.current != 0) {
        return Future.error(
          StateError('GPU retirement requires an observed lost device'));
      }
    } finally {
      calloc.free(info);
    }
    return _retiring = () async {
      final ended = await GpuCompletion.run(
        (port, token) =>
            mgpuContextLeaseRetireDeviceAsyncToPort(_lease, port, token),
        (_) => throw UnsupportedError('Native retirement requires port receipts'),
      );
      if (!ended) throw StateError('Native GPU retirement was not confirmed');
    }();
  }

  Future<void> observeContextRetirement(int expectedGeneration) {
    if (expectedGeneration != generation) {
      return Future.error(StateError('Context retirement creation epoch changed'));
    }
    final existing = _contextRetirement;
    if (existing != null) return existing;
    if (_lease == nullptr) {
      return Future.error(StateError('Context observation requires its creation lease'));
    }
    return _contextRetirement = () async {
      final completed = await GpuCompletion.run(
        (port, token) =>
            mgpuContextLeaseObserveRetirementAsyncToPort(_lease, port, token),
        (_) => throw UnsupportedError('Context retirement requires port receipts'),
      );
      if (!completed) throw StateError('Final native context retirement was not confirmed');
    }();
  }

  Future<void> verifyReplacement(
    DeviceAdapterIdentity expected, Future<void> Function() detach,
  ) async {
    if (adapterIdentity == expected) return;
    // A name filter can select a different physical device. It must never be
    // published to callers as a replacement; retire that candidate first.
    final retired = observeContextRetirement(generation);
    retired.ignore();
    await detach();
    await retired;
    throw DeviceReplacementDiscarded(
      expectedAdapter: expected,
      message: 'Replacement GPU does not match the creation adapter',
    );
  }

  void release() {
    if (_lease == nullptr) return;
    mgpuContextReleaseLease(_lease);
    _lease = nullptr;
  }
}

// Minigpu FFI
class MinigpuFfi extends MinigpuPlatform {
  _FfiDeviceEpoch? _epoch;

  @override
  DeviceAdapterIdentity? get creationAdapterIdentity => _epoch?.adapterIdentity;

  @override
  Future<void> initializeContextForRecovery(DeviceAdapterIdentity expected) async {
    if (_epoch != null) throw StateError('Old GPU attachment has not retired');
    await initializeContext();
    await _epoch!.verifyReplacement(expected, destroyContext);
  }

  @override
  Future<void> observeContextRetirement({required int deviceGeneration}) =>
      _epoch?.observeContextRetirement(deviceGeneration) ??
      Future.error(StateError('GPU creation epoch was not retained'));

  @override
  Future<void> retireLostDevice({required int deviceGeneration}) =>
      _epoch?.retireLost(deviceGeneration) ??
      Future.error(StateError('GPU creation epoch was not retained'));

  @override
  DeviceTextureLimits? get currentDeviceTextureLimits =>
      _readCurrentDeviceTextureLimits(ffi.mgpuGetCurrentDeviceTextureLimits);
  MinigpuFfi();

  void Function()? _deviceLostCallback;
  bool _hadCurrentDevice = false;

  @override
  void setDeviceLostCallback(void Function()? callback) =>
      _deviceLostCallback = callback;

  @override
  DeviceBufferLimits? get currentDeviceBufferLimits {
    final limits = _readCurrentDeviceBufferLimits(
      ffi.mgpuGetCurrentDeviceBufferLimits,
    );
    if (limits == null && _hadCurrentDevice) _deviceLostCallback?.call();
    _hadCurrentDevice = limits != null;
    return limits;
  }

  /// Start a trusted native consumer on the already-ready display device.
  /// The consumer must retain its own ready-context pin inside [start] before
  /// scheduling asynchronous work. This synchronous scope never initializes
  /// or replaces the renderer's device and grants no source or room access.
  int withLiveExternalPoolContext(
    int Function(Pointer<Void> installedQuery, Pointer<Void> context) start,
  ) {
    final lease = calloc<Pointer<NativeContextLease>>();
    final context = calloc<Pointer<ffi.MGPUContextHandle>>();
    try {
      final result = mgpuDefaultContextRetainLease(lease, context);
      if (result != 0) {
        throw StateError('Display GPU context cannot be retained ($result)');
      }
      return start(
        externalPoolQueryFunctionForHost,
        context.value.cast<Void>(),
      );
    } finally {
      if (lease.value != nullptr) mgpuContextReleaseLease(lease.value);
      calloc.free(context);
      calloc.free(lease);
    }
  }

  @override
  Future<void> initializeContext() async {
    // Completion arrives on a Dart port, not through a NativeCallable — see
    // minigpu_ffi_completion.dart for why the callback form aborts the VM.
    if (_epoch != null) {
      if (currentDeviceBufferLimits?.deviceGeneration == _epoch!.generation) return;
      throw StateError('Lost GPU context requires explicit retirement');
    }
    final initialized = await GpuCompletion.run(
      (port, token) => mgpuInitializeContextAsyncToPort(port, token),
      (callback) => ffi.mgpuInitializeContextAsync(callback),
    );
    if (!initialized) {
      throw StateError('GPU context initialization failed or device is lost');
    }
    if (currentDeviceBufferLimits == null) {
      // A positive port receipt owns an attachment even if loss races Dart's
      // delivery. Return that attachment before rejecting the initialization.
      // The legacy void callback does not prove whether one was acquired.
      if (GpuCompletion.usesPort) ffi.mgpuDestroyContext();
      throw StateError('GPU device was lost during initialization');
    }
    try {
      _epoch = _FfiDeviceEpoch.retain();
    } catch (_) {
      ffi.mgpuDestroyContext();
      rethrow;
    }
  }

  @override
  Future<void> destroyContext() async {
    final epoch = _epoch;
    if (epoch == null) return;
    _hadCurrentDevice = false;
    epoch.release();
    _epoch = null;
    ffi.mgpuDestroyContext();
  }

  @override
  PlatformComputeShader createComputeShader() {
    final self = ffi.mgpuCreateComputeShader();
    if (self == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return FfiComputeShader(self);
  }

  @override
  PlatformBuffer createBuffer(int bufferSize, BufferDataType dataType) {
    final self = ffi.mgpuCreateBuffer(bufferSize, dataType.index);
    if (self == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return FfiBuffer(self);
  }

  @override
  PlatformBuffer createBufferFromDescriptor(BufferDescriptor descriptor) {
    descriptor.validate();
    if (!_hasBufferDescriptorV1()) {
      if (descriptor.usage == BufferUsage.legacy) {
        return createBuffer(descriptor.sizeBytes, BufferDataType.uint32);
      }
      throw UnsupportedError(
        'Explicit buffer usage is unsupported by this binary',
      );
    }
    final self = ffi.mgpuCreateBufferWithUsage(
      descriptor.sizeBytes,
      descriptor.usage,
    );
    if (self == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return FfiBuffer(self);
  }

  @override
  PlatformTexture createTexture(TextureDescriptor descriptor) {
    if (!_hasTextureResourceV1()) {
      throw UnsupportedError('Owned textures are unsupported by this binary');
    }
    final texture = ffi.mgpuCreateTexture2D(
      descriptor.width,
      descriptor.height,
      descriptor.format.wireCode,
      descriptor.usage,
      descriptor.arrayLayerCount,
      descriptor.mipLevelCount,
    );
    if (texture == nullptr) throw StateError('Texture creation failed');
    return FfiTexture(texture);
  }

  @override
  PlatformSampler createSampler(SamplerDescriptor descriptor) {
    if (!_hasTextureResourceV1()) {
      throw UnsupportedError('Owned samplers are unsupported by this binary');
    }
    final sampler = ffi.mgpuCreateSampler(
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
    if (sampler == nullptr) throw StateError('Sampler creation failed');
    return FfiSampler(sampler);
  }

  @override
  PlatformRenderPipeline createRenderPipeline(
    RenderPipelineDescriptor descriptor,
  ) => _createFfiRenderPipeline(descriptor, null);

  @override
  int queryVramBytes() => ffi.mgpuQueryVramBytes();

  @override
  List<GpuAdapterInfo> listAdapters() {
    const cap = 16;
    final names = malloc.allocate<Char>(cap * 128);
    final totals = malloc.allocate<Int64>(cap);
    final used = malloc.allocate<Int64>(cap);
    try {
      final n = ffi.mgpuEnumAdapters(names, totals, used, cap);
      final count = n < cap ? n : cap;
      return [
        for (int i = 0; i < count; i++)
          GpuAdapterInfo(
            name: _decodeCString(names + i * 128),
            totalVramBytes: totals[i],
            usedVramBytes: used[i],
          ),
      ];
    } finally {
      malloc.free(names);
      malloc.free(totals);
      malloc.free(used);
    }
  }

  @override
  bool isExternalContentTypeSupported(ExternalContentType type) =>
      ffi.mgpuIsExternalContentTypeSupported(
        ffi.MGPUExternalContentType.fromValue(type.index),
      ) !=
      0;

  @override
  bool isExternalPixelFormatSupported(ExternalPixelFormat format) =>
      ffi.mgpuIsExternalPixelFormatSupported(
        ffi.MGPUExternalPixelFormat.fromValue(format.index),
      ) !=
      0;

  @override
  PlatformVideoTexture? importVideoFrame(ExternalVideoBuffer buf) {
    final nativeBuf = _allocExternalVideoBuffer(buf);
    try {
      final ptr = ffi.mgpuImportVideoFrame(nativeBuf.cast());
      if (ptr == nullptr) return null;
      return FfiVideoTexture(ptr, buf.pixelFormat, buf.width, buf.height);
    } finally {
      malloc.free(nativeBuf);
    }
  }

  @override
  PlatformSharedOutputTexture? createSharedOutputTexture(
    int width,
    int height,
  ) {
    final ptr = ffi.mgpuCreateSharedOutputTexture(width, height);
    if (ptr == nullptr) return null;
    return FfiSharedOutputTexture(ptr);
  }

  @override
  int createD3D11DeviceOnDawnAdapter() {
    return ffi.mgpuCreateD3D11DeviceOnDawnAdapter().address;
  }

  @override
  bool preferDisplayAdapter([bool enable = true]) =>
      ffi.mgpuPreferDisplayAdapter(enable ? 1 : 0) == 0;

  @override
  void drainWorkQueue() {
    try {
      mgpuDrainWorkQueue();
    } catch (_) {
      // Binary predates the export. Nothing to drain that we can reach; the
      // caller's teardown is no worse off than before this API existed.
    }
  }

  @override
  Future<void> waitForSubmittedWork() => _waitForSubmittedWork(
    (port, token) => mgpuWaitForSubmittedWorkAsyncToPort(port, token),
  );

  @override
  int? get drainSpinBudgetMs {
    try {
      return ffi.mgpuDrainSpinBudgetMs();
    } catch (_) {
      // Binary predates the export — which is itself the answer: no drain fix.
      return null;
    }
  }

  @override
  String? get selectedAdapterName {
    const cap = 256;
    final buf = malloc.allocate<Char>(cap);
    try {
      final len = ffi.mgpuGetSelectedAdapterName(buf, cap);
      if (len <= 0) return null;
      return _decodeCString(buf);
    } finally {
      malloc.free(buf);
    }
  }

  @override
  MinigpuPlatform? createSecondaryPlatform(String adapterFilter) {
    final filterPtr = adapterFilter.toNativeUtf8();
    try {
      final handle = ffi.mgpuCreateContextHandle(filterPtr.cast());
      if (handle == nullptr) return null;
      return FfiSecondaryPlatform(handle, adapterFilter: adapterFilter);
    } finally {
      malloc.free(filterPtr);
    }
  }

  // ---- Persistent shader cache ---------------------------------------------
  // Each setter reports whether it landed before a context existed; the result
  // is the AND of them, so configuring several at once is true only when every
  // one of them can still affect the next device.

  @override
  bool configureShaderCache({
    bool? enabled,
    String? directory,
    int? maxBytes,
    String? extraKey,
  }) {
    var preInit = true;
    if (enabled != null) {
      preInit &= ffi.mgpuShaderCacheSetEnabled(enabled ? 1 : 0) == 0;
    }
    if (directory != null) {
      final ptr = directory.toNativeUtf8();
      try {
        preInit &= ffi.mgpuShaderCacheSetDirectory(ptr.cast()) == 0;
      } finally {
        malloc.free(ptr);
      }
    }
    if (maxBytes != null) {
      preInit &= ffi.mgpuShaderCacheSetCapBytes(maxBytes) == 0;
    }
    if (extraKey != null) {
      final ptr = extraKey.toNativeUtf8();
      try {
        preInit &= ffi.mgpuShaderCacheSetExtraKey(ptr.cast()) == 0;
      } finally {
        malloc.free(ptr);
      }
    }
    return preInit;
  }

  @override
  ShaderCacheStats? get shaderCacheStats {
    final out = malloc<ffi.MGPUShaderCacheStats>();
    try {
      ffi.mgpuGetShaderCacheStats(out);
      final s = out.ref;
      return ShaderCacheStats(
        hits: s.hits,
        misses: s.misses,
        stores: s.stores,
        storeFailures: s.storeFailures,
        evictions: s.evictions,
        bytesOnDisk: s.bytesOnDisk,
        entryCount: s.entryCount,
        loadMs: s.loadMs,
        storeMs: s.storeMs,
        pipelineCreateMs: s.pipelineCreateMs,
        enabled: s.enabled != 0,
        usingDefaultProvider: s.usingDefaultProvider != 0,
      );
    } finally {
      malloc.free(out);
    }
  }

  @override
  String? get shaderCacheDirectory {
    const cap = 1024;
    final buf = malloc.allocate<Char>(cap);
    try {
      final len = ffi.mgpuGetShaderCacheDirectory(buf, cap);
      if (len <= 0) return null;
      return _decodeCString(buf);
    } finally {
      malloc.free(buf);
    }
  }

  @override
  int clearShaderCache() => ffi.mgpuShaderCacheClear();

  // ---- Log callback --------------------------------------------------------
  // Delivery is a Dart NATIVE PORT, not a NativeCallable.
  //
  // mgpuSetLogCallback installs a PROCESS-GLOBAL function pointer. A
  // NativeCallable there is owned by ONE isolate, and when that isolate exits
  // the VM deletes the trampoline while the C++ library keeps the pointer — the
  // next line logged by mgpu (or by a Dawn worker thread, which has no isolate
  // at all) aborts the whole process with "Callback invoked after it has been
  // deleted". A whole-suite `dart test` run reproduces this every time,
  // because each test file is its own isolate inside one VM process. A closed
  // port is merely inert.
  //
  // PROCESS-GLOBAL, LAST WRITER WINS: with several isolates registered only the
  // most recent one receives log lines — exactly the old function-pointer
  // behaviour, minus the crash.

  /// Receive port for native log lines in THIS isolate.
  ReceivePort? _logPort;

  /// Whether `mgpuInitDartApi` has succeeded in this isolate.
  bool _dartApiInitialised = false;

  /// Decodes a null-terminated C string using [Utf8Decoder] with
  /// [allowMalformed] so that log messages containing non-UTF-8 bytes
  /// (e.g. driver strings with Latin-1 characters) never throw.
  static String _decodeCString(Pointer<Char> ptr) {
    if (ptr.address == 0) return '';
    final bytes = ptr.cast<Uint8>();
    var len = 0;
    while (bytes[len] != 0) len++;
    return const Utf8Decoder(
      allowMalformed: true,
    ).convert(Uint8List.view(bytes.asTypedList(len).buffer, 0, len));
  }

  @override
  void setLogCallback(
    void Function(int level, String message)? callback, {
    int level = 1,
  }) {
    ffi.mgpuSetLogLevel(level);

    // Clear/replace the NATIVE registration before releasing any old resource:
    // the reverse order leaves a window in which the native side still points
    // at something we have already closed.
    final old = _logPort;
    _logPort = null;

    if (callback == null) {
      log_port.mgpuSetLogPort(0);
      old?.close();
      return;
    }

    if (!_dartApiInitialised) {
      if (log_port.mgpuInitDartApi(NativeApi.initializeApiDLData) != 0) {
        // Built without the Dart API, or an SDK mismatch. Leave the native
        // library on its stderr sink rather than fall back to the crash-prone
        // function-pointer path.
        old?.close();
        return;
      }
      _dartApiInitialised = true;
    }

    final port = ReceivePort('minigpu_ffi.log');
    port.listen((dynamic message) {
      // Wire format: [int32 level, Uint8List utf8Bytes]. Bytes rather than a
      // string because Dawn/driver lines may carry non-UTF-8 (Latin-1)
      // sequences; allowMalformed keeps those from throwing.
      if (message is! List || message.length != 2) return;
      callback(
        message[0] as int,
        const Utf8Decoder(
          allowMalformed: true,
        ).convert(message[1] as Uint8List),
      );
    });
    _logPort = port;
    log_port.mgpuSetLogPort(port.sendPort.nativePort);
    old?.close();
  }

  Pointer<ffi.MGPUExternalVideoBuffer> _allocExternalVideoBuffer(
    ExternalVideoBuffer buf,
  ) {
    final p = malloc<ffi.MGPUExternalVideoBuffer>();
    p.ref.content_typeAsInt = buf.contentType.index;
    p.ref.pixel_formatAsInt = buf.pixelFormat.index;
    p.ref.width = buf.width;
    p.ref.height = buf.height;
    p.ref.num_planes = buf.planes.length;
    for (int i = 0; i < buf.planes.length && i < 4; i++) {
      final sp = buf.planes[i];
      p.ref.planes[i].data_ptr = Pointer.fromAddress(sp.dataPtr);
      p.ref.planes[i].width = sp.width;
      p.ref.planes[i].height = sp.height;
      p.ref.planes[i].stride_bytes = sp.strideBytes;
      p.ref.planes[i].offset_bytes = sp.offsetBytes;
      p.ref.planes[i].subresource_index = sp.subresourceIndex;
      p.ref.planes[i].dmabuf_fd = sp.dmabufFd;
      p.ref.planes[i].drm_format_modifier = sp.drmFormatModifier;
    }
    p.ref.fence.sync_fd = buf.fence.syncFd;
    p.ref.fence.d3d11_fence = Pointer.fromAddress(buf.fence.d3d11FencePtr);
    p.ref.fence.metal_shared_event = Pointer.fromAddress(
      buf.fence.metalSharedEventPtr,
    );
    p.ref.fence.metal_fence_value = buf.fence.metalFenceValue;
    p.ref.timestamp_us = buf.timestampUs;
    return p;
  }
}

// Video texture FFI
final class FfiVideoTexture implements PlatformVideoTexture {
  /// Native backends bind a texture to a shader the ordinary way
  /// ([setOnShader]) and drive the dispatch through the same native pipeline as
  /// every other kernel, so there is nothing for this to do. It exists because
  /// the WEB backend's imported resource is a `GPUExternalTexture`, which that
  /// route cannot carry at all — see `WebVideoTexture.landPackedRgba8`.
  ///
  /// Declared here rather than inherited because this class `implements` the
  /// interface, which takes the signature and not the default body.
  @override
  bool get requiresExternalLanding => false;

  @override
  bool landPackedRgba8(
    PlatformBuffer dst, {
    required int width,
    required int height,
    int downscale = 1,
  }) => false;

  FfiVideoTexture(
    Pointer<ffi.MGPUVideoTexture> ptr,
    ExternalPixelFormat fmt,
    int w,
    int h,
  ) : _self = ptr,
      _pixelFormat = fmt,
      _width = w,
      _height = h;

  final Pointer<ffi.MGPUVideoTexture> _self;
  final ExternalPixelFormat _pixelFormat;
  final int _width;
  final int _height;

  @override
  int get numPlanes {
    switch (_pixelFormat) {
      case ExternalPixelFormat.nv12:
      case ExternalPixelFormat.yuv420pAsNV12Planes:
        return 2;
      case ExternalPixelFormat.yuv420pAsRGBPlanes:
        return 3;
      default:
        return 1;
    }
  }

  @override
  int get width => _width;
  @override
  int get height => _height;
  @override
  ExternalPixelFormat get pixelFormat => _pixelFormat;

  @override
  void setOnShader(PlatformComputeShader shader, int slot, int planeIndex) {
    ffi.mgpuSetVideoTexture(
      (shader as FfiComputeShader)._self,
      slot,
      _self,
      planeIndex,
    );
  }

  @override
  PlatformBuffer toRGBA() {
    final ptr = ffi.mgpuVideoTextureToRGBA(_self);
    if (ptr == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return FfiBuffer(ptr);
  }

  @override
  bool bgraToRgbaSharedOutput(PlatformSharedOutputTexture dst) {
    if (dst is! FfiSharedOutputTexture) return false;
    return ffi.mgpuVideoTextureBGRAToRGBASharedOutput(_self, dst._self) != 0;
  }

  @override
  Future<bool> bgraToRgbaSharedOutputAsync(
    PlatformSharedOutputTexture dst,
  ) async {
    if (dst is! FfiSharedOutputTexture) return false;
    return GpuCompletion.runInt(
      (port, token) => mgpuVideoTextureBGRAToRGBASharedOutputAsyncToPort(
        _self,
        dst._self,
        port,
        token,
      ),
      (callback) => ffi.mgpuVideoTextureBGRAToRGBASharedOutputAsync(
        _self,
        dst._self,
        callback,
      ),
    );
  }

  @override
  void destroy() => ffi.mgpuDestroyVideoTexture(_self);
}

// Cross-API shared output texture FFI wrapper.
final class FfiSharedOutputTexture implements PlatformSharedOutputTexture {
  FfiSharedOutputTexture(Pointer<ffi.MGPUSharedOutputTexture> self)
    : _self = self;

  final Pointer<ffi.MGPUSharedOutputTexture> _self;
  bool _destroyed = false;

  @override
  int get width => ffi.mgpuSharedOutputTextureGetWidth(_self);

  @override
  int get height => ffi.mgpuSharedOutputTextureGetHeight(_self);

  @override
  int get d3d11Handle =>
      ffi.mgpuSharedOutputTextureGetD3D11Handle(_self).address;

  @override
  Object? get webGpuTextureJs => null; // native-only; no JS objects on FFI

  @override
  int get d3d11TexturePtr =>
      ffi.mgpuSharedOutputTextureGetD3D11Texture(_self).address;

  @override
  bool copyFromBuffer(PlatformBuffer src) {
    return ffi.mgpuCopyBufferToSharedOutputTexture(
          (src as FfiBuffer)._self,
          _self,
        ) !=
        0;
  }

  @override
  // `async` is load-bearing for compatibility, not style: without it a bad
  // `src` would throw SYNCHRONOUSLY out of a method whose callers may only
  // have a `.catchError` on the returned future.
  Future<bool> copyFromBufferAsync(PlatformBuffer src) async {
    // THE ZERO-COPY PRESENT PATH — one of these per presented frame. It used
    // to build and close a NativeCallable every time, i.e. it re-armed the
    // "callback invoked after it has been deleted" abort 30-60 times a second.
    final self = (src as FfiBuffer)._self;
    return GpuCompletion.runInt(
      (port, token) => mgpuCopyBufferToSharedOutputTextureAsyncToPort(
        self,
        _self,
        port,
        token,
      ),
      (callback) =>
          ffi.mgpuCopyBufferToSharedOutputTextureAsync(self, _self, callback),
    );
  }

  @override
  bool copyFromBufferF32(PlatformBuffer src) {
    return ffi.mgpuCopyBufferF32ToSharedOutputTexture(
          (src as FfiBuffer)._self,
          _self,
        ) !=
        0;
  }

  @override
  int debugReadFirstPixel() =>
      ffi.mgpuSharedOutputTextureDebugReadFirstPixel(_self);

  @override
  int debugReadFirstPixelDawn() =>
      ffi.mgpuSharedOutputTextureDebugReadFirstPixelDawn(_self);

  @override
  void destroy() {
    if (_destroyed) return;
    _destroyed = true;
    ffi.mgpuDestroySharedOutputTexture(_self);
  }
}

/// An independent native context bound to a specific adapter (multi-GPU).
/// Created via [MinigpuFfi.createSecondaryPlatform]; owns its MGPU instance
/// (device, queue, WebGPU thread).  Buffers/shaders created here route every
/// later operation through that instance automatically — only creation and
/// context lifecycle need handle-aware calls.
class FfiSecondaryPlatform extends MinigpuPlatform {
  FfiSecondaryPlatform(this._handle, {String adapterFilter = ''})
      : _adapterFilter = adapterFilter;

  _FfiDeviceEpoch? _epoch;
  final String _adapterFilter;

  @override
  DeviceAdapterIdentity? get creationAdapterIdentity => _epoch?.adapterIdentity;

  @override
  Future<void> initializeContextForRecovery(DeviceAdapterIdentity expected) async {
    if (!_destroyed) throw StateError('Old GPU handle has not retired');
    final filter = _adapterFilter.toNativeUtf8();
    try {
      _handle = ffi.mgpuCreateContextHandle(filter.cast<Char>());
    } finally {
      malloc.free(filter);
    }
    if (_handle == nullptr) throw StateError('Replacement GPU context unavailable');
    _destroyed = false;
    try {
      await initializeContext();
      await _epoch!.verifyReplacement(expected, destroyContext);
    } catch (_) {
      await destroyContext();
      rethrow;
    }
  }

  @override
  Future<void> observeContextRetirement({required int deviceGeneration}) =>
      _epoch?.observeContextRetirement(deviceGeneration) ??
      Future.error(StateError('GPU creation epoch was not retained'));

  @override
  Future<void> retireLostDevice({required int deviceGeneration}) =>
      _epoch?.retireLost(deviceGeneration) ??
      Future.error(StateError('GPU creation epoch was not retained'));

  Pointer<ffi.MGPUContextHandle> _handle;
  bool _destroyed = false;
  @override
  DeviceTextureLimits? get currentDeviceTextureLimits => _destroyed ? null :
      _readCurrentDeviceTextureLimits((dimension, layers, generation) =>
        ffi.mgpuContextGetCurrentDeviceTextureLimits(
          _handle, dimension, layers, generation));
  bool _hadCurrentDevice = false;
  void Function()? _deviceLostCallback;

  @override
  void setDeviceLostCallback(void Function()? callback) =>
      _deviceLostCallback = callback;

  @override
  DeviceBufferLimits? get currentDeviceBufferLimits {
    if (_destroyed) return null;
    final limits = _readCurrentDeviceBufferLimits(
      (maxBufferSize, maxStorageBufferBindingSize, deviceGeneration) =>
          ffi.mgpuContextGetCurrentDeviceBufferLimits(
            _handle, maxBufferSize, maxStorageBufferBindingSize,
            deviceGeneration,
          ),
    );
    if (limits == null && _hadCurrentDevice) _deviceLostCallback?.call();
    _hadCurrentDevice = limits != null;
    return limits;
  }

  @override
  Future<void> initializeContext() async {
    if (_destroyed) throw StateError('GPU context has retired');
    if (_epoch != null) {
      if (currentDeviceBufferLimits?.deviceGeneration == _epoch!.generation) return;
      throw StateError('Lost GPU context requires explicit retirement');
    }
    final initialized = await GpuCompletion.run(
      (port, token) => mgpuContextInitializeAsyncToPort(_handle, port, token),
      (callback) => ffi.mgpuContextInitializeAsync(_handle, callback),
    );
    if (!initialized || currentDeviceBufferLimits == null) {
      throw StateError('GPU context initialization failed or device is lost');
    }
    _epoch = _FfiDeviceEpoch.retain(_handle);
  }

  /// Synchronously start a trusted native consumer of this context. The
  /// consumer must retain its ready-context lease inside [start] before
  /// transferring work to a thread. An int callback cannot suspend across
  /// context destruction. This does not initialize or replace a device and is
  /// available separately on [MinigpuFfi] for the ready display context.
  int withLiveExternalPoolContext(
    int Function(Pointer<Void> installedQuery, Pointer<Void> context) start,
  ) {
    if (_destroyed) throw StateError('GPU context has retired');
    return start(externalPoolQueryFunctionForHost, _handle.cast<Void>());
  }

  @override
  Future<void> destroyContext() async {
    if (_destroyed) return;
    _destroyed = true;
    _hadCurrentDevice = false;
    _epoch?.release();
    _epoch = null;
    ffi.mgpuDestroyContextHandle(_handle);
  }

  @override
  Future<void> waitForSubmittedWork() => _waitForSubmittedWork(
    (port, token) =>
        mgpuContextWaitForSubmittedWorkAsyncToPort(_handle, port, token),
  );

  @override
  PlatformComputeShader createComputeShader() {
    final self = ffi.mgpuContextCreateComputeShader(_handle);
    if (self == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return FfiComputeShader(self);
  }

  @override
  PlatformBuffer createBuffer(int bufferSize, BufferDataType dataType) {
    final self = ffi.mgpuContextCreateBuffer(
      _handle,
      bufferSize,
      dataType.index,
    );
    if (self == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return FfiBuffer(self);
  }

  @override
  PlatformBuffer createBufferFromDescriptor(BufferDescriptor descriptor) {
    descriptor.validate();
    if (!_hasBufferDescriptorV1()) {
      if (descriptor.usage == BufferUsage.legacy) {
        return createBuffer(descriptor.sizeBytes, BufferDataType.uint32);
      }
      throw UnsupportedError(
        'Explicit buffer usage is unsupported by this binary',
      );
    }
    final self = ffi.mgpuContextCreateBufferWithUsage(
      _handle,
      descriptor.sizeBytes,
      descriptor.usage,
    );
    if (self == nullptr) throw MinigpuPlatformOutOfMemoryException();
    return FfiBuffer(self);
  }

  @override
  PlatformTexture createTexture(TextureDescriptor descriptor) {
    if (!_hasTextureResourceV1()) {
      throw UnsupportedError('Owned textures are unsupported by this binary');
    }
    final texture = ffi.mgpuContextCreateTexture2D(
      _handle,
      descriptor.width,
      descriptor.height,
      descriptor.format.wireCode,
      descriptor.usage,
      descriptor.arrayLayerCount,
      descriptor.mipLevelCount,
    );
    if (texture == nullptr) throw StateError('Texture creation failed');
    return FfiTexture(texture);
  }

  @override
  PlatformSampler createSampler(SamplerDescriptor descriptor) {
    if (!_hasTextureResourceV1()) {
      throw UnsupportedError('Owned samplers are unsupported by this binary');
    }
    final sampler = ffi.mgpuContextCreateSampler(
      _handle,
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
    if (sampler == nullptr) throw StateError('Sampler creation failed');
    return FfiSampler(sampler);
  }

  @override
  PlatformRenderPipeline createRenderPipeline(
    RenderPipelineDescriptor descriptor,
  ) {
    if (_destroyed) throw StateError('GPU context is destroyed');
    return _createFfiRenderPipeline(descriptor, _handle);
  }

  @override
  String? get selectedAdapterName {
    const cap = 256;
    final buf = malloc.allocate<Char>(cap);
    try {
      final len = ffi.mgpuContextGetAdapterName(_handle, buf, cap);
      if (len <= 0) return null;
      return MinigpuFfi._decodeCString(buf);
    } finally {
      malloc.free(buf);
    }
  }
}

// Compute shader FFI
final class FfiComputeShader implements PlatformComputeShader {
  FfiComputeShader(Pointer<ffi.MGPUComputeShader> self) : _self = self;

  final Pointer<ffi.MGPUComputeShader> _self;
  bool _destroyed = false;

  /// Synchronously bind a trusted host's retained GPU input. Never serialize
  /// this pointer or retain it after this scope. The native binder must check
  /// the shader's actual device/context and its current input authority.
  int withNativeShaderForHostInterop(int Function(Pointer<Void>) bind) {
    if (_destroyed) throw StateError('Compute shader has retired');
    return bind(_self.cast<Void>());
  }

  /// Trusted host cleanup after actual GPU input reads. Removes cached input
  /// and bind-group references on the same GPU FIFO while keeping the pipeline.
  @override
  void releaseTextureBindingAfterGpuRead(int slot) {
    if (_destroyed) throw StateError('Compute shader has retired');
    if (mgpuReleaseTextureBinding(_self, slot) != 0) {
      throw StateError('GPU input binding could not retire');
    }
  }

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
  void setBufferFire(int tag, PlatformBuffer buffer) {
    ffi.mgpuSetBufferFire(_self, tag, (buffer as FfiBuffer)._self);
  }

  @override
  void setBufferRange(
    int slot,
    PlatformBuffer buffer,
    BufferBindingDescriptor descriptor,
  ) {
    if (!_hasBufferBindingV1()) {
      throw UnsupportedError(
        'Explicit buffer ranges are unsupported by this binary',
      );
    }
    final result = ffi.mgpuSetBufferRange(
      _self,
      slot,
      (buffer as FfiBuffer)._self,
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
    if (!_hasTextureResourceV1()) {
      throw UnsupportedError('Owned textures are unsupported by this binary');
    }
    if (view is! FfiTextureView) {
      throw ArgumentError.value(view, 'view', 'Expected an FFI texture view');
    }
    if (view._destroyed) {
      throw StateError('Texture view is destroyed');
    }
    final status = ffi.mgpuSetOwnedTexture(
      _self,
      slot,
      view._self,
      kind.wireCode,
    );
    if (status != 0) {
      _textureStatus(status, 'Texture binding');
    }
  }

  @override
  void setSampler(int slot, PlatformSampler sampler, SamplerBindingKind kind) {
    if (_destroyed) {
      throw StateError('Compute shader is destroyed');
    }
    if (!_hasTextureResourceV1()) {
      throw UnsupportedError('Owned samplers are unsupported by this binary');
    }
    if (sampler is! FfiSampler) {
      throw ArgumentError.value(sampler, 'sampler', 'Expected an FFI sampler');
    }
    if (sampler._destroyed) {
      throw StateError('Sampler is destroyed');
    }
    final status = ffi.mgpuSetOwnedSampler(
      _self,
      slot,
      sampler._self,
      kind.wireCode,
    );
    if (status != 0) {
      _textureStatus(status, 'Sampler binding');
    }
  }

  @override
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ) async {
    // Awaited dispatches are the densest source of completions in a codec
    // frame (5-10 per frame), so this is where the per-call NativeCallable hurt
    // most: the C layer cannot cancel an enqueued dispatch, and a trampoline
    // closed while one is still queued aborts the process.
    await GpuCompletion.run(
      (port, token) => mgpuDispatchAsyncToPort(
        _self,
        groupsX,
        groupsY,
        groupsZ,
        port,
        token,
      ),
      (callback) =>
          ffi.mgpuDispatchAsync(_self, groupsX, groupsY, groupsZ, callback),
    );
  }

  @override
  void dispatchFire(int groupsX, int groupsY, int groupsZ) {
    // mgpuDispatch enqueues the pass on the WebGPU thread and returns
    // immediately — no completer/NativeCallable round trip per dispatch.
    ffi.mgpuDispatch(_self, groupsX, groupsY, groupsZ);
  }

  @override
  void destroy() {
    if (_destroyed) {
      return;
    }
    _destroyed = true;
    ffi.mgpuDestroyComputeShader(_self);
  }
}

// Owned texture FFI
final class FfiTexture implements PlatformTexture {
  FfiTexture(this._self);

  final Pointer<ffi.MGPUTexture> _self;
  bool _destroyed = false;

  @override
  bool get supportsRegionWrite {
    if (_destroyed) return false;
    try {
      return ffi.mgpuTextureRegionUploadVersion() == 1;
    } on ArgumentError {
      return false; // The installed binary may predate this additive export.
    }
  }

  @override
  PlatformTextureView createView(TextureViewDescriptor descriptor) {
    _checkLive();
    final view = ffi.mgpuCreateTextureView(
      _self,
      descriptor.dimension.wireCode,
      descriptor.baseMipLevel,
      descriptor.mipLevelCount,
      descriptor.baseArrayLayer,
      descriptor.arrayLayerCount,
    );
    if (view == nullptr) {
      throw StateError('Texture view creation failed');
    }
    return FfiTextureView(view);
  }

  @override
  Future<void> write(
    Uint8List bytes, {
    int mipLevel = 0,
    int arrayLayer = 0,
  }) async {
    _checkLive();
    final copy = malloc<Uint8>(bytes.lengthInBytes);
    try {
      copy.asTypedList(bytes.lengthInBytes).setAll(0, bytes);
      final status = ffi.mgpuWriteTexture2D(
        _self,
        copy,
        bytes.lengthInBytes,
        mipLevel,
        arrayLayer,
      );
      if (status != 0) {
        _textureStatus(status, 'Texture upload');
      }
    } finally {
      malloc.free(copy);
    }
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
    // Refuse unsafe narrowing even if called without the public wrapper.
    if (x < 0 ||
        x >= 32768 ||
        y < 0 ||
        y >= 32768 ||
        width <= 0 ||
        width > 32768 ||
        height <= 0 ||
        height > 32768 ||
        mipLevel < 0 ||
        mipLevel > 15 ||
        arrayLayer < 0 ||
        arrayLayer >= 32768 ||
        bytes.isEmpty ||
        bytes.lengthInBytes > 0x7fffffff) {
      throw ArgumentError('Invalid texture region upload');
    }
    if (!supportsRegionWrite) {
      throw UnsupportedError('Region uploads are unsupported by this binary');
    }
    final copy = malloc<Uint8>(bytes.lengthInBytes);
    try {
      copy.asTypedList(bytes.lengthInBytes).setAll(0, bytes);
      final int status;
      try {
        status = ffi.mgpuWriteTextureRegion2D(
          _self,
          copy,
          bytes.lengthInBytes,
          x,
          y,
          width,
          height,
          mipLevel,
          arrayLayer,
        );
      } on ArgumentError {
        throw UnsupportedError('Region uploads are unsupported by this binary');
      }
      if (status != 0) _textureStatus(status, 'Texture region upload');
    } finally {
      malloc.free(copy);
    }
  }

  @override
  void destroy() {
    if (_destroyed) {
      return;
    }
    _destroyed = true;
    ffi.mgpuDestroyTexture(_self);
  }

  void _checkLive() {
    if (_destroyed) {
      throw StateError('Texture is destroyed');
    }
  }
}

final class FfiTextureView implements PlatformTextureView {
  FfiTextureView(this._self);

  final Pointer<ffi.MGPUTextureView> _self;
  bool _destroyed = false;

  @override
  void destroy() {
    if (_destroyed) {
      return;
    }
    _destroyed = true;
    ffi.mgpuDestroyTextureView(_self);
  }
}

final class FfiSampler implements PlatformSampler {
  FfiSampler(this._self);

  final Pointer<ffi.MGPUSampler> _self;
  bool _destroyed = false;

  @override
  void destroy() {
    if (_destroyed) {
      return;
    }
    _destroyed = true;
    ffi.mgpuDestroySampler(_self);
  }
}

final class FfiRenderPipeline implements PlatformRenderPipeline {
  FfiRenderPipeline(this._self);

  final Pointer<ffi.MGPURenderPipeline> _self;
  bool _destroyed = false;

  @override
  int get status {
    if (_destroyed) return 3;
    try {
      return ffi.mgpuRenderPipelineStatus(_self);
    } on ArgumentError {
      throw UnsupportedError(
        'This native minigpu build does not expose raster pipeline status',
      );
    }
  }

  @override
  String get diagnostic {
    if (_destroyed) return 'Render pipeline is destroyed';
    return using((Arena arena) {
      final bytes = arena<Uint8>(_maxRasterDiagnosticBytes + 1);
      final int length;
      try {
        length = ffi.mgpuRenderPipelineDiagnostic(
          _self,
          bytes.cast<Char>(),
          _maxRasterDiagnosticBytes + 1,
        );
      } on ArgumentError {
        throw UnsupportedError(
          'This native minigpu build does not expose raster diagnostics',
        );
      }
      if (length < 0 || length > _maxRasterDiagnosticBytes) {
        return 'Render pipeline diagnostic was invalid';
      }
      return utf8.decode(bytes.asTypedList(length), allowMalformed: true);
    });
  }

  @override
  List<int>? get bindingCacheStats {
    if (_destroyed) return null;
    try {
      return List<int>.unmodifiable([
        for (var selector = 0; selector < 4; selector++)
          ffi.mgpuRenderPipelineBindingStat(_self, selector).toInt(),
      ]);
    } on ArgumentError {
      // A stale packaged native library is still usable, but cannot prove
      // retained bindings. Callers surface the missing receipt explicitly.
      return null;
    }
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
    if (color is! FfiTextureView ||
        (depth != null && depth is! FfiTextureView) ||
        buffers.any((buffer) => buffer is! FfiBuffer) ||
        views.any((view) => view is! FfiTextureView) ||
        samplers.any((sampler) => sampler is! FfiSampler)) {
      throw ArgumentError('Render resources belong to another backend');
    }
    final ffiColor = color;
    final ffiDepth = depth as FfiTextureView?;
    final ffiBuffers = buffers.cast<FfiBuffer>();
    final ffiViews = views.cast<FfiTextureView>();
    final ffiSamplers = samplers.cast<FfiSampler>();
    if (ffiColor._destroyed || (ffiDepth?._destroyed ?? false)) {
      throw StateError('Render attachment view is destroyed');
    }
    if (ffiBuffers.any((buffer) => buffer._destroyed)) {
      throw StateError('Render buffer is destroyed');
    }
    if (ffiViews.any((view) => view._destroyed)) {
      throw StateError('Render texture view is destroyed');
    }
    if (ffiSamplers.any((sampler) => sampler._destroyed)) {
      throw StateError('Render sampler is destroyed');
    }
    _checkRasterSubmission(
      clear,
      loadFlags,
      ffiDepth != null,
      buffers,
      views,
      samplers,
      commands,
    );
    final result = using((Arena arena) {
      final clearPtr = arena<Float>(5);
      final commandPtr = commands.isEmpty
          ? nullptr.cast<Uint32>()
          : arena<Uint32>(commands.length);
      final bufferPtr = ffiBuffers.isEmpty
          ? nullptr.cast<Pointer<ffi.MGPUBuffer>>()
          : arena<Pointer<ffi.MGPUBuffer>>(ffiBuffers.length);
      final viewPtr = ffiViews.isEmpty
          ? nullptr.cast<Pointer<ffi.MGPUTextureView>>()
          : arena<Pointer<ffi.MGPUTextureView>>(ffiViews.length);
      final samplerPtr = ffiSamplers.isEmpty
          ? nullptr.cast<Pointer<ffi.MGPUSampler>>()
          : arena<Pointer<ffi.MGPUSampler>>(ffiSamplers.length);
      clearPtr.asTypedList(5).setAll(0, clear);
      if (commands.isNotEmpty) {
        commandPtr.asTypedList(commands.length).setAll(0, commands);
      }
      for (var i = 0; i < ffiBuffers.length; i++) {
        bufferPtr[i] = ffiBuffers[i]._self;
      }
      for (var i = 0; i < ffiViews.length; i++) {
        viewPtr[i] = ffiViews[i]._self;
      }
      for (var i = 0; i < ffiSamplers.length; i++) {
        samplerPtr[i] = ffiSamplers[i]._self;
      }
      try {
        return ffi.mgpuSubmitRender(
          _self,
          ffiColor._self,
          ffiDepth?._self ?? nullptr,
          clearPtr,
          loadFlags,
          bufferPtr,
          ffiBuffers.length,
          viewPtr,
          ffiViews.length,
          samplerPtr,
          ffiSamplers.length,
          commandPtr,
          commands.length,
        );
      } on ArgumentError {
        throw UnsupportedError(
          'This native minigpu build does not expose raster submission',
        );
      }
    });
    if (result != 0) _textureStatus(result, 'Render submission');
  }

  @override
  void destroy() {
    if (_destroyed) return;
    try {
      ffi.mgpuDestroyRenderPipeline(_self);
      _destroyed = true;
    } on ArgumentError {
      throw UnsupportedError(
        'This native minigpu build does not expose raster pipeline destruction',
      );
    }
  }
}

// Buffer FFI
final class FfiBuffer implements PlatformBuffer {
  FfiBuffer(Pointer<ffi.MGPUBuffer> self) : _self = self;

  final Pointer<ffi.MGPUBuffer> _self;

  /// Web-only concept (Emscripten WGPUBuffer handle); nothing here.
  @override
  int get webBufferHandle => 0;

  // --- Pooled scratch + readback callback ---------------------------------
  // The read/write hot path used to `malloc` a scratch buffer and allocate a
  // `NativeCallable.listener` on *every* call. For a 30/60 fps feed that is
  // thousands of allocations per second. We instead keep one growable scratch
  // (per direction), freed in [destroy], and take the completion callback from
  // the process-wide never-closed pool in minigpu_ffi_completion.dart.
  //
  // Dart's event loop is single-threaded, so re-entrancy can only happen
  // across an `await`. The `_readInFlight` / `_writeInFlight` guards detect
  // that rare case (e.g. two concurrent reads on the same buffer) and fall
  // back to a temporary local allocation so the pooled SCRATCH is never
  // aliased. The callback no longer needs that guard — every operation gets
  // its own slot.
  Pointer<NativeType>? _readScratch;
  int _readScratchBytes = 0;
  bool _readInFlight = false;

  Pointer<NativeType>? _writeScratch;
  int _writeScratchBytes = 0;
  bool _writeInFlight = false;

  /// Pool scratch only for SMALL transfers.  The pool exists for per-frame
  /// streaming buffers (thousands of small writes/sec); pooling large
  /// transfers instead PINS a native allocation the size of the transfer for
  /// the buffer's whole lifetime — with thousands of live weight buffers
  /// (e.g. 40 GB of model experts) that leaked tens of GB of host RAM.
  /// Large transfers malloc/free per call.
  static const int _maxPooledScratchBytes = 1 << 20;

  // NOTE: the raw-upload path deliberately has NO scratch of its own.
  //
  // It used to keep an isolate-wide 32 MiB staging buffer and memcpy each
  // payload into it before the FFI call, because a Dart list has no address an
  // ordinary native call can take. `TypedData.address` in a leaf call removes
  // that constraint, so [writeRawBytes] now passes the caller's own store: no
  // scratch to grow, no pinned host allocation, and one whole frame-sized
  // memcpy per frame less on the streaming path.

  bool _destroyed = false;

  Pointer<NativeType> _ensureReadScratch(int bytes) {
    if (_readScratch == null || _readScratchBytes < bytes) {
      if (_readScratch != null) malloc.free(_readScratch!);
      _readScratch = malloc.allocate<NativeType>(bytes);
      _readScratchBytes = bytes;
    }
    return _readScratch!;
  }

  /// Maps a [BufferDataType] onto the C `MGPUElementType` code.
  ///
  /// NOT `dataType.index`: the Dart and C++ `BufferDataType` enums are ordered
  /// differently, so an index means different things on the two sides of the
  /// boundary. This mapping is explicit for that reason.
  static int _elementTypeCode(BufferDataType dataType) {
    switch (dataType) {
      case BufferDataType.int8:
        return MgpuElementType.i8;
      case BufferDataType.uint8:
        return MgpuElementType.u8;
      case BufferDataType.int16:
        return MgpuElementType.i16;
      case BufferDataType.uint16:
        return MgpuElementType.u16;
      case BufferDataType.int32:
        return MgpuElementType.i32;
      case BufferDataType.uint32:
        return MgpuElementType.u32;
      case BufferDataType.int64:
        return MgpuElementType.i64;
      case BufferDataType.uint64:
        return MgpuElementType.u64;
      case BufferDataType.float32:
        return MgpuElementType.f32;
      case BufferDataType.float64:
        return MgpuElementType.f64;
      case BufferDataType.float16:
        throw UnimplementedError(
          'BufferDataType.float16 read is not implemented yet.',
        );
    }
  }

  /// Fallback issue path: one typed native call per element type, used only
  /// when completions cannot be delivered by port.
  void _issueTypedRead(
    BufferDataType dataType,
    Pointer<NativeType> nativePtr,
    int elementsToRead,
    int elementOffset,
    Pointer<NativeFunction<Void Function()>> callback,
  ) {
    switch (dataType) {
      case BufferDataType.int8:
        ffi.mgpuReadAsyncInt8(
          _self,
          nativePtr.cast<Int8>(),
          elementsToRead,
          elementOffset,
          callback,
        );
      case BufferDataType.uint8:
        ffi.mgpuReadAsyncUint8(
          _self,
          nativePtr.cast<Uint8>(),
          elementsToRead,
          elementOffset,
          callback,
        );
      case BufferDataType.int16:
        ffi.mgpuReadAsyncInt16(
          _self,
          nativePtr.cast<Int16>(),
          elementsToRead,
          elementOffset,
          callback,
        );
      case BufferDataType.uint16:
        ffi.mgpuReadAsyncUint16(
          _self,
          nativePtr.cast<Uint16>(),
          elementsToRead,
          elementOffset,
          callback,
        );
      case BufferDataType.int32:
        ffi.mgpuReadAsyncInt32(
          _self,
          nativePtr.cast<Int32>(),
          elementsToRead,
          elementOffset,
          callback,
        );
      case BufferDataType.uint32:
        ffi.mgpuReadAsyncUint32(
          _self,
          nativePtr.cast<Uint32>(),
          elementsToRead,
          elementOffset,
          callback,
        );
      case BufferDataType.int64:
        ffi.mgpuReadAsyncInt64(
          _self,
          nativePtr.cast<Int64>(),
          elementsToRead,
          elementOffset,
          callback,
        );
      case BufferDataType.uint64:
        ffi.mgpuReadAsyncUint64(
          _self,
          nativePtr.cast<Uint64>(),
          elementsToRead,
          elementOffset,
          callback,
        );
      case BufferDataType.float32:
        ffi.mgpuReadAsyncFloat(
          _self,
          nativePtr.cast<Float>(),
          elementsToRead,
          elementOffset,
          callback,
        );
      case BufferDataType.float64:
        ffi.mgpuReadAsyncDouble(
          _self,
          nativePtr.cast<Double>(),
          elementsToRead,
          elementOffset,
          callback,
        );
      case BufferDataType.float16:
        throw UnimplementedError(
          'BufferDataType.float16 read is not implemented yet.',
        );
    }
  }

  Pointer<NativeType> _ensureWriteScratch(int bytes) {
    if (_writeScratch == null || _writeScratchBytes < bytes) {
      if (_writeScratch != null) malloc.free(_writeScratch!);
      _writeScratch = malloc.allocate<NativeType>(bytes);
      _writeScratchBytes = bytes;
    }
    return _writeScratch!;
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
    final int elementsToRead = (readElements > 0)
        ? readElements
        : totalElementsInOutput;

    // --- Input Validation ---
    if (elementOffset < 0) {
      throw RangeError.value(
        elementOffset,
        'elementOffset',
        'Cannot be negative',
      );
    }
    if (elementsToRead < 0) {
      throw RangeError.value(
        elementsToRead,
        'readElements',
        'Cannot be negative',
      );
    }
    // Check if requested range is valid within the output buffer
    if (elementsToRead > totalElementsInOutput) {
      throw RangeError(
        'Read range (offset: $elementOffset, count: $elementsToRead) exceeds output buffer capacity ($totalElementsInOutput elements)',
      );
    }
    // --- End Input Validation ---

    if (elementsToRead == 0) {
      // Nothing to read
      return;
    }

    // Allocate temporary native memory based on the number of elements to read
    final int bytesToAllocate = elementsToRead * elementSize;

    // Reject what the native side has no reader for BEFORE issuing anything —
    // the read is started once, above the copy-out switch.
    if (dataType == BufferDataType.float16) {
      throw UnimplementedError(
        'BufferDataType.float16 read is not implemented yet.',
      );
    }

    // Use the pooled scratch on the common (non-reentrant) path; fall back to
    // a private local allocation if a read is already in flight or the
    // transfer is too large to pin (see _maxPooledScratchBytes).
    final bool usePool =
        !_readInFlight &&
        !_destroyed &&
        bytesToAllocate <= _maxPooledScratchBytes;
    final Pointer<NativeType> nativePtr;
    final bool ownsLocal;
    if (usePool) {
      ownsLocal = false;
      _readInFlight = true;
      nativePtr = _ensureReadScratch(bytesToAllocate);
    } else {
      ownsLocal = true;
      nativePtr = malloc.allocate<NativeType>(bytesToAllocate);
    }

    try {
      // ONE issue + ONE await for every element type. The port entry point
      // takes the element type as an argument, so the ten typed native calls
      // collapse into one; the fallback keeps the typed form. What remains in
      // the switch below is only the copy-out into [outputData].
      await GpuCompletion.run(
        (port, token) => mgpuReadAsyncToPort(
          _self,
          nativePtr.cast<Void>(),
          elementsToRead,
          elementOffset,
          _elementTypeCode(dataType),
          port,
          token,
        ),
        (callback) => _issueTypedRead(
          dataType,
          nativePtr,
          elementsToRead,
          elementOffset,
          callback,
        ),
      );

      // Copy the landed bytes into the caller's typed buffer.
      switch (dataType) {
        case BufferDataType.int8:
          {
            final List<int> data = nativePtr.cast<Int8>().asTypedList(
              elementsToRead,
            );
            // Copy data into the correct portion of the outputData
            if (outputData is Int8List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setInt8(startByte + i * elementSize, data[i]);
              }
            } else {
              /* Handle other potential TypedData types if needed */
            }
          }
          break;
        case BufferDataType.int16:
          {
            final List<int> data = nativePtr.cast<Int16>().asTypedList(
              elementsToRead,
            );
            if (outputData is Int16List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setInt16(
                  startByte + i * elementSize,
                  data[i],
                  Endian.host,
                );
              }
            }
          }
          break;
        case BufferDataType.int32:
          {
            final List<int> data = nativePtr.cast<Int32>().asTypedList(
              elementsToRead,
            );
            if (outputData is Int32List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setInt32(
                  startByte + i * elementSize,
                  data[i],
                  Endian.host,
                );
              }
            }
          }
          break;
        case BufferDataType.int64:
          {
            final List<int> data = nativePtr.cast<Int64>().asTypedList(
              elementsToRead,
            );
            if (outputData is Int64List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setInt64(
                  startByte + i * elementSize,
                  data[i],
                  Endian.host,
                );
              }
            }
          }
          break;
        case BufferDataType.uint8:
          {
            final List<int> data = nativePtr.cast<Uint8>().asTypedList(
              elementsToRead,
            );
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
            final List<int> data = nativePtr.cast<Uint16>().asTypedList(
              elementsToRead,
            );
            if (outputData is Uint16List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setUint16(
                  startByte + i * elementSize,
                  data[i],
                  Endian.host,
                );
              }
            }
          }
          break;
        case BufferDataType.uint32:
          {
            final List<int> data = nativePtr.cast<Uint32>().asTypedList(
              elementsToRead,
            );
            if (outputData is Uint32List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setUint32(
                  startByte + i * elementSize,
                  data[i],
                  Endian.host,
                );
              }
            }
          }
          break;
        case BufferDataType.uint64:
          {
            final List<int> data = nativePtr.cast<Uint64>().asTypedList(
              elementsToRead,
            );
            if (outputData is Uint64List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setUint64(
                  startByte + i * elementSize,
                  data[i],
                  Endian.host,
                );
              }
            }
          }
          break;
        case BufferDataType.float16:
          break; // rejected above, before the read was issued
        case BufferDataType.float32:
          {
            final List<double> data = nativePtr.cast<Float>().asTypedList(
              elementsToRead,
            );
            if (outputData is Float32List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setFloat32(
                  startByte + i * elementSize,
                  data[i],
                  Endian.host,
                );
              }
            }
          }
          break;
        case BufferDataType.float64:
          {
            final List<double> data = nativePtr.cast<Double>().asTypedList(
              elementsToRead,
            );
            if (outputData is Float64List) {
              outputData.setRange(0, elementsToRead, data);
            } else if (outputData is ByteData) {
              final int startByte = elementOffset * elementSize;
              for (int i = 0; i < elementsToRead; ++i) {
                outputData.setFloat64(
                  startByte + i * elementSize,
                  data[i],
                  Endian.host,
                );
              }
            }
          }
          break;
      }
    } finally {
      if (ownsLocal) {
        malloc.free(nativePtr);
      } else {
        _readInFlight = false;
      }
    }
  }

  @override
  Future<void> write(
    TypedData inputData,
    int elementCount, {
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
      throw ArgumentError('Unsupported BufferDataType for write: $dataType');
    }

    // --- Input Validation ---
    final int totalElementsInInput = inputData.lengthInBytes ~/ elementSize;
    if (elementCount < 0) {
      return; // Allow setting zero elements
    }
    if (elementCount > totalElementsInInput) {
      throw RangeError(
        'elementCount ($elementCount) exceeds input data capacity ($totalElementsInInput elements)',
      );
    }
    // --- End Input Validation ---

    if (elementCount == 0) {
      // Allow setting zero elements (C++ should handle this)
    }

    final int byteSize = elementCount * elementSize;
    // Int8/Uint8 writes are padded up to a 4-byte boundary by the native side.
    final int scratchBytes = (inputData is Int8List || inputData is Uint8List)
        ? ((byteSize + 3) & ~3)
        : byteSize;

    final bool usePool =
        !_writeInFlight &&
        !_destroyed &&
        scratchBytes <= _maxPooledScratchBytes;
    final Pointer<NativeType> nativePtr;
    final bool ownsLocal;
    if (usePool) {
      ownsLocal = false;
      _writeInFlight = true;
      nativePtr = _ensureWriteScratch(scratchBytes);
    } else {
      ownsLocal = true;
      nativePtr = malloc.allocate<NativeType>(scratchBytes);
    }

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
        final inputBytes = ByteData.view(
          inputData.buffer,
          inputData.offsetInBytes,
          byteSize,
        );
        final nativeBytes = nativePtr.cast<Uint8>().asTypedList(byteSize);
        for (int i = 0; i < byteSize; i++) {
          nativeBytes[i] = inputBytes.getUint8(i);
        }
      }

      // Switch to call the proper native function
      switch (dataType) {
        case BufferDataType.int8:
          ffi.mgpuWriteInt8(_self, nativePtr.cast<Int8>(), byteSize);
          break;
        case BufferDataType.int16:
          ffi.mgpuWriteInt16(_self, nativePtr.cast<Int16>(), byteSize);
          break;
        case BufferDataType.int32:
          ffi.mgpuWriteInt32(_self, nativePtr.cast<Int32>(), byteSize);
          break;
        case BufferDataType.int64:
          ffi.mgpuWriteInt64(_self, nativePtr.cast<Int64>(), byteSize);
          break;
        case BufferDataType.uint8:
          ffi.mgpuWriteUint8(_self, nativePtr.cast<Uint8>(), byteSize);
          break;
        case BufferDataType.uint16:
          ffi.mgpuWriteUint16(_self, nativePtr.cast<Uint16>(), byteSize);
          break;
        case BufferDataType.uint32:
          ffi.mgpuWriteUint32(_self, nativePtr.cast<Uint32>(), byteSize);
          break;
        case BufferDataType.uint64:
          ffi.mgpuWriteUint64(_self, nativePtr.cast<Uint64>(), byteSize);
          break;
        case BufferDataType.float16:
          throw UnimplementedError(
            'BufferDataType.float16 write is not implemented yet.',
          );
        case BufferDataType.float32:
          ffi.mgpuWriteFloat(_self, nativePtr.cast<Float>(), byteSize);
          break;
        case BufferDataType.float64:
          ffi.mgpuWriteDouble(_self, nativePtr.cast<Double>(), byteSize);
          break;
      }
    } finally {
      if (ownsLocal) {
        malloc.free(nativePtr);
      } else {
        _writeInFlight = false;
      }
    }
  }

  /// Chunked raw upload: hands [bytes]'s own backing store to
  /// `mgpuWriteBufferAt` a chunk (32 MiB) at a time, so a multi-GB weight
  /// upload never makes Dawn's staging ring hold more than one chunk.
  ///
  /// NO HOST COPY. `TypedData.address` in a leaf call gives C the list's real
  /// address, so the bytes go straight from the caller's list into
  /// `wgpuQueueWriteBuffer`. This path used to memcpy the whole payload into a
  /// native scratch first, purely to obtain a stable pointer — at 4K that was a
  /// 33 MB host copy per frame, about half of the whole upload stage, on top of
  /// the copy `wgpuQueueWriteBuffer` makes into its own staging ring.
  ///
  /// See [ffi.mgpuWriteBufferAtLeaf] for why a leaf call is safe here.
  @override
  Future<void> writeRawBytes(Uint8List bytes, {int dstByteOffset = 0}) async {
    if (bytes.length % 4 != 0 || dstByteOffset % 4 != 0) {
      throw ArgumentError('writeRawBytes needs 4-byte-aligned length/offset');
    }
    if (bytes.isEmpty) return;
    const chunkBytes = 32 << 20;
    // Fast path: one chunk, so the list itself is the argument and no view
    // object is built at all. This is the per-frame streaming shape (a 4K RGBA
    // frame is 33.2 MB, under the 33.55 MB chunk).
    if (bytes.length <= chunkBytes) {
      ffi.mgpuWriteBufferAtLeaf(
        _self,
        bytes.address,
        bytes.length,
        dstByteOffset,
      );
      return;
    }
    var off = 0;
    while (off < bytes.length) {
      final n = (bytes.length - off) < chunkBytes
          ? (bytes.length - off)
          : chunkBytes;
      // A view, not a copy: `sublistView` aliases the same store and `.address`
      // resolves through its offsetInBytes.
      final view = Uint8List.sublistView(bytes, off, off + n);
      ffi.mgpuWriteBufferAtLeaf(_self, view.address, n, dstByteOffset + off);
      off += n;
    }
  }

  @override
  void destroy() {
    if (_destroyed) return;
    _destroyed = true;
    if (_readScratch != null) {
      malloc.free(_readScratch!);
      _readScratch = null;
    }
    if (_writeScratch != null) {
      malloc.free(_writeScratch!);
      _writeScratch = null;
    }
    // NO CALLBACK TEARDOWN HERE. destroy() used to close this buffer's pooled
    // read listener, which is a process abort if a read is still in flight —
    // the C layer cannot be told to forget a pointer it has already been
    // handed. Completion slots are pooled and never closed; see
    // minigpu_ffi_completion.dart.
    ffi.mgpuDestroyBuffer(_self);
  }
}
