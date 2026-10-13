// Shared public-Dart raster lifecycle checks: the bounded retained bind-group
// cache, pipeline closure during preparation, the bounded preparation wait and
// cross-context refusal. Reused by the native and browser smokes beside it.
import 'dart:async';
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart'
    show PlatformRenderPipeline;

/// `kMaxBindGroupsPerPipeline` in `minigpu_ffi/src/src/minigpu_render.cpp`.
const rasterBindGroupCacheBound = 64;

/// The caller-visible bound of `GpuRenderPipeline.prepare`.
const rasterPreparationBound = Duration(seconds: 30);

/// Withholds the backend's render-pipeline completions so that a preparation
/// stays pending for as long as a check needs. Only a runner that can stand
/// between the backend and the device's asynchronous pipeline creation can
/// provide one. What is withheld is the delivery; the device still compiles.
abstract interface class RasterPipelineHold {
  /// Withhold the next [count] render-pipeline completions.
  void arm(int count);

  /// Completions currently withheld.
  int get held;

  /// Withheld completions that have since been handed to the backend.
  int get delivered;

  /// Hands every withheld completion to the backend; returns how many.
  int release();
}

const _width = 16, _height = 8; // One pixel per binding set.
const _completed = Object();

void _check(bool condition, String message) {
  if (!condition) throw StateError(message);
}

/// The error a future completed with, or [_completed].
Future<Object> _settled(Future<Object?> operation) =>
    operation.then<Object>((_) => _completed, onError: (Object error) => error);

Future<void> _refuses(Future<void> Function() action, String message) async {
  try {
    await action();
  } on ArgumentError {
    return;
  } on StateError {
    return;
  }
  throw StateError(message);
}

bool _same(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

bool _near(List<int> actual, List<int> expected) {
  for (var i = 0; i < 4; i++) {
    if ((actual[i] - expected[i]).abs() > 1) return false;
  }
  return true;
}

/// One uniform colour over the whole target, without vertex buffers. Each
/// [variant] is distinct shader text with the same output, so that a pipeline
/// cannot be served from another variant's compiled result.
RenderPipelineDescriptor _fillDescriptor(int variant) =>
    RenderPipelineDescriptor(
      code:
          '''
struct Params { color: vec4<f32> }
@group(0) @binding(0) var<uniform> params: Params;
@vertex fn vsMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  var corners = array<vec2<f32>, 3>(
    vec2<f32>(-1., -1.), vec2<f32>(3., -1.), vec2<f32>(-1., 3.));
  return vec4<f32>(corners[index], 0., 1.);
}
@fragment fn fsMain() -> @location(0) vec4<f32> {
  return params.color + vec4<f32>(0., 0., 0., $variant.e-9);
}
''',
      colorFormat: TextureFormat.rgba8Unorm,
      bindings: [
        RenderBindingLayout(
          slot: 0,
          kind: RenderBindingKind.uniformBuffer,
          visibility: 2,
          minBindingSize: 16,
        ),
      ],
    );

Map<String, Object> _device(Minigpu gpu) => {
  'adapter': gpu.adapterName ?? '',
  'backend': gpu.adapterIdentity?.backend ?? -1,
  'device_epoch': gpu.deviceEpoch,
};

GpuTexture _target(Minigpu gpu, int width, int height) => gpu.createTexture(
  TextureDescriptor(
    width: width,
    height: height,
    format: TextureFormat.rgba8Unorm,
    usage: TextureUsage.renderAttachment | TextureUsage.textureBinding,
  ),
);

Future<Buffer> _uniform(Minigpu gpu, Float32List values) async {
  final buffer = gpu.createBufferFromDescriptor(
    BufferDescriptor(
      sizeBytes: values.lengthInBytes,
      usage: BufferUsage.uniform | BufferUsage.copyDst,
    ),
  );
  await buffer.write(values, values.length);
  return buffer;
}

Float32List _colour(List<int> rgba) =>
    Float32List.fromList([for (final channel in rgba) channel / 255]);

/// Reads a small rgba8 colour target back through the existing compute path.
final class _PixelReader {
  _PixelReader(Minigpu gpu, GpuTextureView view, this.width, this.height)
    : _output = gpu.createBufferFromDescriptor(
        BufferDescriptor(
          sizeBytes: width * height * 16,
          usage: BufferUsage.storage | BufferUsage.copySrc,
        ),
      ),
      _shader = gpu.createComputeShader() {
    _shader.loadKernelString('''
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
let pixel = vec4<u32>(round(textureLoad(src, vec2<i32>(id.xy), 0) * 255.));
let index = (id.y * ${width}u + id.x) * 4u;
dst[index] = pixel.r; dst[index + 1u] = pixel.g;
dst[index + 2u] = pixel.b; dst[index + 3u] = pixel.a;
}
''');
    _shader.setTexture(0, view, TextureBindingKind.sampledUnfilterableFloat);
    _shader.setBufferRange(
      1,
      _output,
      BufferBindingDescriptor(
        kind: BufferBindingKind.storage,
        sizeBytes: width * height * 16,
      ),
    );
  }

  final int width, height;
  final Buffer _output;
  final ComputeShader _shader;

  /// Row-major pixels, four channel values each.
  Future<Uint32List> read() async {
    await _shader.dispatch(width, height, 1);
    final pixels = Uint32List(width * height * 4);
    await _output.read(pixels, pixels.length, dataType: BufferDataType.uint32);
    return pixels;
  }

  void dispose() {
    _shader.destroy();
    _output.destroy();
  }
}

/// Prepares [descriptor] on the ordinary route, draws one colour and reads it
/// back: the device still serves new pipelines after the case before it.
Future<void> _renderProbe(
  Minigpu gpu,
  String label,
  RenderPipelineDescriptor descriptor,
) async {
  const expected = [51, 102, 153, 255];
  final color = _target(gpu, 4, 4);
  final view = color.createView();
  final reader = _PixelReader(gpu, view, 4, 4);
  final params = await _uniform(gpu, _colour(expected));
  final pipeline = await gpu.createRenderPipeline(descriptor);
  try {
    final pass = pipeline.createPass(color: view);
    pass.setBuffer(0, params);
    pass.draw(3);
    await pass.submit();
    await gpu.waitForSubmittedWork();
    final pixels = await reader.read();
    final actual = pixels.sublist(5 * 4, 5 * 4 + 4); // Pixel 1,1.
    _check(_near(actual, expected), '$label: drew $actual, not $expected');
  } finally {
    pipeline.destroy();
    params.destroy();
    reader.dispose();
    view.destroy();
    color.destroy();
  }
  // The retirements above finish before the caller allocates anything else.
  await gpu.waitForSubmittedWork();
}

/// A distinct opaque colour for each binding set id below 128.
List<int> _setRgba(int id) => [id + 1, 255 - id, (id * 7) % 256, 255];

final class _BindingSet {
  _BindingSet(this.id, this.buffer, this.offset, this.key);

  /// Also the pixel this set paints and the seed of its colour.
  final int id;
  final Buffer buffer;
  final int offset;

  /// Identity of the bound buffer range, as the cache keys it.
  final int key;

  List<int> get rgba => _setRgba(id);
}

/// The cache contract restated: exact keys, at most
/// [rasterBindGroupCacheBound] entries, least recently used leaves first.
final class _CacheModel {
  final _order = <int>[]; // Least recently used first.
  int created = 0, reused = 0, retired = 0;

  int get live => _order.length;

  void use(int key) {
    if (_order.remove(key)) {
      reused++;
    } else {
      created++;
      if (_order.length == rasterBindGroupCacheBound) {
        _order.removeAt(0);
        retired++;
      }
    }
    _order.add(key);
  }
}

/// Fills one pipeline's retained bind-group cache to its bound and past it.
///
/// Every binding set paints its own pixel, so each stage proves both the
/// counters and that every draw used the bind group of its own buffer range,
/// whether that group was new, reused, or rebuilt after eviction. Returns the
/// counters recorded after each stage.
Future<Map<String, Object>> runRasterBindingCacheScenario() async {
  const bound = rasterBindGroupCacheBound;
  final gpu = Minigpu();
  final stages = <Map<String, Object>>[];
  try {
    await gpu.init();
    final device = _device(gpu);
    final color = _target(gpu, _width, _height);
    final colorView = color.createView();
    final reader = _PixelReader(gpu, colorView, _width, _height);
    final pipeline = await gpu.createRenderPipeline(_fillDescriptor(0));

    var nextKey = 0;
    Future<_BindingSet> ownSet(int id) async {
      final set = _BindingSet(
        id,
        gpu.createBufferFromDescriptor(
          BufferDescriptor(
            sizeBytes: 16,
            usage: BufferUsage.uniform | BufferUsage.copyDst,
          ),
        ),
        0,
        nextKey++,
      );
      await set.buffer.write(_colour(set.rgba), 4);
      return set;
    }

    // Sets 0..119 own a 16-byte buffer each. Sets 120..127 are eight ranges of
    // one buffer, 256 bytes apart: their keys differ only by offset.
    final sets = <_BindingSet>[
      for (var id = 0; id < 120; id++) await ownSet(id),
    ];
    final ranged = Float32List(8 * 64);
    for (var i = 0; i < 8; i++) {
      ranged.setAll(i * 64, _colour(_setRgba(120 + i)));
    }
    final shared = await _uniform(gpu, ranged);
    for (var i = 0; i < 8; i++) {
      sets.add(_BindingSet(120 + i, shared, i * 256, nextKey++));
    }

    final model = _CacheModel();
    Future<void> drawPass(
      String stage,
      List<_BindingSet> drawn,
      List<int> expected,
    ) async {
      final pass = pipeline.createPass(color: colorView);
      for (final set in drawn) {
        pass.setScissorRect(set.id % _width, set.id ~/ _width, 1, 1);
        pass.setBuffer(0, set.buffer, offset: set.offset, sizeBytes: 16);
        pass.draw(3);
        model.use(set.key);
      }
      await pass.submit();
      await gpu.waitForSubmittedWork();
      final cache = pipeline.bindingCacheSnapshot;
      _check(cache != null, 'Render binding cache diagnostics unavailable');
      final actual = [cache!.created, cache.reused, cache.retired, cache.live];
      stages.add({
        'stage': stage,
        'draws': drawn.length,
        'created': actual[0],
        'reused': actual[1],
        'retired': actual[2],
        'live': actual[3],
      });
      _check(
        _same(actual, expected),
        '$stage: created/reused/retired/live $actual, expected $expected',
      );
      _check(
        _same(actual, [model.created, model.reused, model.retired, model.live]),
        '$stage: counters $actual differ from the least-recently-used model',
      );
      final pixels = await reader.read();
      final painted = {for (final set in drawn) set.id: set.rgba};
      for (var id = 0; id < _width * _height; id++) {
        final pixel = pixels.sublist(id * 4, id * 4 + 4);
        final wanted = painted[id] ?? const [0, 0, 0, 0];
        _check(
          _near(pixel, wanted),
          '$stage: binding set $id drew $pixel, not $wanted',
        );
      }
    }

    // created, reused, retired, live after each stage.
    await drawPass('fill to the bound', sets.sublist(0, bound), [64, 0, 0, 64]);
    await drawPass('reuse at the bound', sets.sublist(0, bound), [
      64,
      64,
      0,
      64,
    ]);
    await drawPass('touch the oldest entry', [sets[0]], [64, 65, 0, 64]);
    // The least recently USED entry (set 1) leaves, not the oldest (set 0).
    await drawPass('one past the bound', [sets[bound]], [65, 65, 1, 64]);
    await drawPass('recently used entry survived', [sets[0]], [65, 66, 1, 64]);
    await drawPass('evicted entry rebuilt', [sets[1]], [66, 66, 2, 64]);
    // 128 sets in one pass. Every draw after the second evicts an entry, and
    // the last 64 evict groups this same pass set earlier and still encodes.
    await drawPass('twice the bound in one pass', sets, [192, 68, 128, 64]);

    // Buffers retired while their bind groups are still cached, then replaced:
    // a replacement must never be served a group built for a retired buffer.
    // Every retirement is allowed to finish before a replacement is allocated,
    // so that this stage exercises the cache and nothing else.
    for (var id = bound; id < 120; id++) {
      sets[id].buffer.destroy();
    }
    await gpu.waitForSubmittedWork();
    for (var id = bound; id < 120; id++) {
      sets[id] = await ownSet(id);
    }
    await drawPass('replaced buffers', sets.sublist(bound, 120), [
      248,
      68,
      184,
      64,
    ]);
    await drawPass('replacements and ranges reused', sets.sublist(bound), [
      248,
      132,
      184,
      64,
    ]);

    pipeline.destroy();
    _check(
      pipeline.bindingCacheSnapshot == null,
      'A destroyed pipeline still reports cache counters',
    );
    reader.dispose();
    await gpu.waitForSubmittedWork();
    return {'device': device, 'bound': bound, 'stages': stages};
  } finally {
    if (gpu.isInitialized) await gpu.destroy();
  }
}

/// Closes render pipelines while they are still preparing: the wrapper alone
/// first, then the public creation route through context retirement.
///
/// A backend with its own compile thread may finish a pipeline before it can
/// be closed, so several are started at once and each one's backend status is
/// read at the moment of closure. [requireEveryPending] is for a backend that
/// cannot complete anything before control returns to the event loop.
Future<Map<String, Object>> runRasterPendingCloseScenario({
  bool requireEveryPending = false,
}) async {
  const count = 8;
  final gpu = Minigpu();
  try {
    await gpu.init();
    final device = _device(gpu);
    final color = _target(gpu, 4, 4);
    final view = color.createView();

    // Wrapper closure.
    final platforms = <PlatformRenderPipeline>[];
    final wrappers = <GpuRenderPipeline>[];
    final preparing = <Future<Object>>[];
    for (var i = 0; i < count; i++) {
      final descriptor = _fillDescriptor(1 + i);
      final platform = gpu.withPlatformForHostInterop(
        (backend) => backend.createRenderPipeline(descriptor),
      );
      final wrapper = GpuRenderPipeline(platform, descriptor);
      platforms.add(platform);
      wrappers.add(wrapper);
      preparing.add(_settled(wrapper.prepare()));
    }
    final statusAtClose = [for (final platform in platforms) platform.status];
    for (final wrapper in wrappers) {
      wrapper.destroy();
    }
    final pending = statusAtClose.where((status) => status == 0).length;
    _check(
      requireEveryPending ? pending == count : pending > 0,
      'Only $pending of $count pipelines were still preparing when closed: '
      '$statusAtClose',
    );
    final outcomes = await Future.wait(preparing).timeout(
      const Duration(seconds: 5),
      onTimeout: () =>
          throw StateError('A closed preparation neither ended nor rejected'),
    );
    for (var i = 0; i < count; i++) {
      _check(
        statusAtClose[i] != 0 || outcomes[i] is StateError,
        'Closing preparing pipeline $i did not reject it: ${outcomes[i]}',
      );
      _check(
        !wrappers[i].isValid && platforms[i].status == 3,
        'Closed pipeline $i is still usable',
      );
    }
    // The backend finishes compiling what was closed. Nothing may come back.
    await Future<void>.delayed(const Duration(milliseconds: 250));
    await gpu.waitForSubmittedWork();
    for (var i = 0; i < count; i++) {
      _check(
        !wrappers[i].isValid && platforms[i].status == 3,
        'A late completion revived closed pipeline $i',
      );
      await _refuses(
        () async => wrappers[i].createPass(color: view),
        'Closed pipeline $i still records passes',
      );
    }
    await _renderProbe(gpu, 'after wrapper closure', _fillDescriptor(1));

    // Context retirement on the public creation route.
    final before = gpu.resourceAllocationSnapshot;
    final creations = [
      for (var i = 0; i < count; i++)
        _settled(gpu.createRenderPipeline(_fillDescriptor(101 + i))),
    ];
    final registered = gpu.resourceAllocationSnapshot;
    _check(
      registered.renderPipelinesCreated ==
              before.renderPipelinesCreated + count &&
          registered.renderPipelinesLive == before.renderPipelinesLive + count,
      'Preparing pipelines were not registered before preparation ended',
    );
    final closing = gpu.destroy();
    final closed = gpu.resourceAllocationSnapshot;
    _check(
      closed.renderPipelinesRetired == before.renderPipelinesRetired + count &&
          closed.renderPipelinesLive == before.renderPipelinesLive,
      'Context retirement left a preparing pipeline registered',
    );
    final rejected = await Future.wait(creations).timeout(
      const Duration(seconds: 5),
      onTimeout: () =>
          throw StateError('A creation closed by its context never ended'),
    );
    for (var i = 0; i < count; i++) {
      _check(
        rejected[i] is StateError,
        'Context retirement did not reject creation $i: ${rejected[i]}',
      );
    }
    await closing;
    _check(!gpu.isInitialized, 'The context stayed open');
    await gpu.init();
    await _renderProbe(gpu, 'after context retirement', _fillDescriptor(101));
    final reopened = gpu.resourceAllocationSnapshot;
    _check(
      reopened.renderPipelinesLive == 0 &&
          reopened.renderPipelinesCreated == reopened.renderPipelinesRetired,
      'A pipeline closed during preparation is registered again',
    );
    return {
      'device': device,
      'pipelines': count,
      'wrapper_status_at_close': statusAtClose,
      'wrapper_pending_at_close': pending,
      'wrapper_rejections': outcomes.whereType<StateError>().length,
      'context_rejections': rejected.whereType<StateError>().length,
      'pipelines_created': reopened.renderPipelinesCreated,
      'pipelines_retired': reopened.renderPipelinesRetired,
      'pipelines_live': reopened.renderPipelinesLive,
    };
  } finally {
    if (gpu.isInitialized) await gpu.destroy();
  }
}

/// The bounded preparation wait. Two preparations are withheld past
/// [rasterPreparationBound], one on the public creation route and one on a
/// bare wrapper. Each must reject with a timeout and destroy its pipeline, and
/// neither may come back when [hold] delivers the completions afterwards.
Future<Map<String, Object>> runRasterPreparationTimeoutScenario(
  RasterPipelineHold hold,
) async {
  final gpu = Minigpu();
  Timer? ticker;
  try {
    await gpu.init();
    final device = _device(gpu);
    final color = _target(gpu, 4, 4);
    final view = color.createView();
    final before = gpu.resourceAllocationSnapshot;
    final routed = _fillDescriptor(301), bare = _fillDescriptor(302);

    hold.arm(2);
    final watch = Stopwatch()..start();
    final viaContext = _settled(gpu.createRenderPipeline(routed));
    final platform = gpu.withPlatformForHostInterop(
      (backend) => backend.createRenderPipeline(bare),
    );
    final wrapper = GpuRenderPipeline(platform, bare);
    final viaWrapper = _settled(wrapper.prepare());
    // Preparation must leave the event loop free for the whole wait.
    var turns = 0;
    ticker = Timer.periodic(const Duration(milliseconds: 100), (_) => turns++);

    final sampledAt = <int>[];
    Future<void> stillPending(Duration at) async {
      final wait = at - watch.elapsed;
      if (wait > Duration.zero) await Future<void>.delayed(wait);
      sampledAt.add(watch.elapsedMilliseconds);
      _check(
        hold.held == 2 && hold.delivered == 0,
        'The runner withheld ${hold.held} of 2 completions',
      );
      _check(
        platform.status == 0 &&
            wrapper.isValid &&
            gpu.resourceAllocationSnapshot.renderPipelinesLive ==
                before.renderPipelinesLive + 1,
        'A withheld preparation was not pending after ${watch.elapsed}',
      );
    }

    await stillPending(const Duration(seconds: 1));
    await stillPending(const Duration(seconds: 15));
    await stillPending(const Duration(seconds: 29));
    final outcomes = await Future.wait([viaContext, viaWrapper]).timeout(
      const Duration(seconds: 10),
      onTimeout: () => throw StateError(
        'A withheld preparation was still waiting after ${watch.elapsed}',
      ),
    );
    final elapsed = watch.elapsed;
    ticker.cancel();
    _check(
      outcomes.every((outcome) => outcome is TimeoutException),
      'A withheld preparation did not time out: $outcomes',
    );
    _check(
      elapsed >= rasterPreparationBound &&
          elapsed < rasterPreparationBound + const Duration(seconds: 3),
      'Preparation gave up after $elapsed, not at $rasterPreparationBound',
    );
    _check(turns >= 200, 'The event loop turned $turns times in $elapsed');
    _check(
      !wrapper.isValid && platform.status == 3,
      'The timed-out pipeline was not destroyed',
    );
    final timedOut = gpu.resourceAllocationSnapshot;
    _check(
      timedOut.renderPipelinesCreated == before.renderPipelinesCreated + 1 &&
          timedOut.renderPipelinesRetired ==
              before.renderPipelinesRetired + 1 &&
          timedOut.renderPipelinesLive == before.renderPipelinesLive,
      'The timed-out creation was not retired',
    );
    _check(
      hold.held == 2 && hold.delivered == 0,
      'A completion reached the backend before the wait ended',
    );

    // The completions arrive late.
    final released = hold.release();
    _check(released == 2, 'Released $released of 2 withheld completions');
    await Future<void>.delayed(const Duration(milliseconds: 500));
    _check(
      hold.delivered == 2 && hold.held == 0,
      '${hold.delivered} of 2 late completions reached the backend',
    );
    await gpu.waitForSubmittedWork();
    _check(
      !wrapper.isValid && platform.status == 3,
      'A late completion revived the timed-out pipeline',
    );
    await _refuses(
      () async => wrapper.createPass(color: view),
      'The timed-out pipeline records passes after its late completion',
    );
    final afterLate = gpu.resourceAllocationSnapshot;
    _check(
      afterLate.renderPipelinesCreated == timedOut.renderPipelinesCreated &&
          afterLate.renderPipelinesRetired == timedOut.renderPipelinesRetired &&
          afterLate.renderPipelinesLive == timedOut.renderPipelinesLive,
      'A late completion changed the pipeline counters',
    );
    _check(gpu.isDeviceCurrent, 'The device did not survive the timeout');
    // Nothing is withheld now: the same two sources prepare and draw.
    await _renderProbe(gpu, 'after the routed timeout', routed);
    await _renderProbe(gpu, 'after the bare timeout', bare);
    return {
      'device': device,
      'bound_ms': rasterPreparationBound.inMilliseconds,
      'timed_out_after_ms': elapsed.inMilliseconds,
      'pending_sampled_at_ms': sampledAt,
      'event_loop_turns': turns,
      'withheld': 2,
      'delivered_late': hold.delivered,
      'timeouts': outcomes.whereType<TimeoutException>().length,
    };
  } finally {
    ticker?.cancel();
    hold.release();
    if (gpu.isInitialized) await gpu.destroy();
  }
}

/// A pipeline must refuse an attachment and a buffer created by another
/// context, at the public wrapper. Returns false on a platform that has no
/// independent contexts, where the case does not exist.
Future<bool> runRasterForeignOwnerScenario() async {
  const expected = [204, 153, 51, 255];
  final gpu = Minigpu();
  Minigpu? other;
  try {
    await gpu.init();
    final adapter = gpu.adapterName;
    if (adapter == null || adapter.isEmpty) return false;
    try {
      // The same adapter as the first context, by its own reported name.
      other = Minigpu.forAdapter(adapter);
    } on UnsupportedError {
      return false;
    }
    await other.init();
    final pipeline = await gpu.createRenderPipeline(_fillDescriptor(401));
    final ownColor = _target(gpu, 4, 4), foreignColor = _target(other, 4, 4);
    final ownView = ownColor.createView();
    final foreignView = foreignColor.createView();
    final ownParams = await _uniform(gpu, _colour(expected));
    final foreignParams = await _uniform(other, _colour(expected));

    final toForeign = pipeline.createPass(color: foreignView);
    toForeign.setBuffer(0, ownParams);
    toForeign.draw(3);
    await _refuses(toForeign.submit, 'A foreign attachment was accepted');

    final withForeign = pipeline.createPass(color: ownView);
    withForeign.setBuffer(0, foreignParams);
    withForeign.draw(3);
    await _refuses(withForeign.submit, 'A foreign buffer was accepted');

    // The same pass with this context's own resources is accepted and draws.
    final own = pipeline.createPass(color: ownView);
    own.setBuffer(0, ownParams);
    own.draw(3);
    await own.submit();
    await gpu.waitForSubmittedWork();
    final reader = _PixelReader(gpu, ownView, 4, 4);
    final pixels = await reader.read();
    reader.dispose();
    final actual = pixels.sublist(5 * 4, 5 * 4 + 4);
    _check(_near(actual, expected), 'Own pass drew $actual, not $expected');
    return true;
  } finally {
    if (other != null && other.isInitialized) await other.destroy();
    if (gpu.isInitialized) await gpu.destroy();
  }
}
