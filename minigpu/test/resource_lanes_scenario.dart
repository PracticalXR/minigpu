// Shared public-Dart acceptance scenarios for the resource lanes: explicit byte
// allocations, typed buffer ranges, owned textures with views and samplers,
// and the owning context's retirement of its children. Reused by the native
// and browser smokes beside it.
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';
// A closed context refuses with this error, which the library does not export.
import 'package:minigpu/src/buffer.dart' show MinigpuNotInitializedError;

void _check(bool condition, String message) {
  if (!condition) throw StateError(message);
}

bool _isRefusal(Object error) =>
    error is ArgumentError ||
    error is StateError ||
    error is UnsupportedError ||
    error is MinigpuNotInitializedError;

/// [operation] must throw a refusal before it returns.
void _refuses(void Function() operation, String message) {
  try {
    operation();
  } catch (error) {
    if (_isRefusal(error)) return;
    rethrow;
  }
  throw StateError(message);
}

/// [operation] must throw a refusal, at once or when awaited.
Future<void> _refusesLater(
  Future<void> Function() operation,
  String message,
) async {
  try {
    await operation();
  } catch (error) {
    if (_isRefusal(error)) return;
    rethrow;
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

Uint8List _words(List<int> values) =>
    Uint32List.fromList(values).buffer.asUint8List();

/// What the run used. `backend` is the native adapter identity's backend code
/// where the platform reports one (1 for D3D11, 2 for D3D12), otherwise -1.
Map<String, Object> _device(Minigpu gpu) => {
  'adapter': gpu.adapterName ?? '',
  'backend': gpu.adapterIdentity?.backend ?? -1,
  'adapter_identity': gpu.adapterIdentity != null,
  'device_epoch': gpu.deviceEpoch,
};

const _readOnlyKernel = '''
struct Params { addend: u32, multiplier: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputValues: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<u32>;
@compute @workgroup_size(4) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < 4u) {
    outputValues[id.x] = inputValues[id.x] * params.multiplier + params.addend;
  }
}
''';

const _legacyKernel = '''
struct Params { addend: u32, multiplier: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> inputValues: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<u32>;
@compute @workgroup_size(4) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < 4u) {
    outputValues[id.x] = inputValues[id.x] * params.multiplier + params.addend;
  }
}
''';

/// Explicit byte allocations and typed buffer ranges on the real device.
///
/// [replacementRounds] above zero adds a stage that destroys a buffer and
/// allocates its replacement at once, with no wait between the two, and checks
/// what each replacement holds. It is off by default: a native library that
/// allocates buffers on the caller's thread crashes the process in that order
/// on the Dawn D3D12 backend.
Future<Map<String, Object>> runBufferLaneScenario({
  int replacementRounds = 0,
}) async {
  final gpu = Minigpu();
  try {
    await gpu.init();
    final device = _device(gpu);

    // Malformed descriptors refuse before any backend call.
    for (final bad in const [
      [0, BufferUsage.storage],
      [6, BufferUsage.storage],
      [-4, BufferUsage.storage],
      [4, 0],
      [4, 0x100],
    ]) {
      _refuses(
        () => BufferDescriptor(sizeBytes: bad[0], usage: bad[1]),
        'Malformed buffer descriptor $bad was accepted',
      );
    }

    Buffer allocate(int bytes, int usage) {
      final buffer = gpu.createBufferFromDescriptor(
        BufferDescriptor(sizeBytes: bytes, usage: usage),
      );
      _check(
        buffer.isValid &&
            buffer.descriptor?.sizeBytes == bytes &&
            buffer.descriptor?.usage == usage &&
            gpu.ownsBuffer(buffer),
        'Allocation of $bytes bytes with usage $usage lost its descriptor',
      );
      return buffer;
    }

    final created = gpu.resourceAllocationSnapshot.buffersCreated;
    final params = allocate(768, BufferUsage.uniform | BufferUsage.copyDst);
    final input = allocate(768, BufferUsage.storage | BufferUsage.copyDst);
    final output = allocate(16, BufferUsage.storage | BufferUsage.copySrc);
    final vertex = allocate(256, BufferUsage.vertex | BufferUsage.copyDst);
    final index = allocate(256, BufferUsage.index | BufferUsage.copyDst);
    _check(
      gpu.resourceAllocationSnapshot.buffersCreated == created + 5,
      'Five allocations were not counted',
    );

    // Transfer methods honour the declared usage.
    await _refusesLater(
      () => params.read(Uint32List(4), 4, dataType: BufferDataType.uint32),
      'A buffer without copySrc was read',
    );
    await _refusesLater(
      () => output.write(Uint32List(4), 4, dataType: BufferDataType.uint32),
      'A buffer without copyDst was written',
    );

    // The legacy allocation route is unchanged and round-trips its contents.
    final legacy = gpu.createBuffer(16, BufferDataType.uint32);
    _check(legacy.descriptor == null, 'A legacy buffer gained a descriptor');
    final legacyValues = Uint32List.fromList([2, 3, 5, 7]);
    await legacy.write(legacyValues, 4, dataType: BufferDataType.uint32);
    final legacyRead = Uint32List(4);
    await legacy.read(legacyRead, 4, dataType: BufferDataType.uint32);
    _check(_same(legacyRead, legacyValues), 'Legacy round trip: $legacyRead');

    // Typed ranges: a uniform and a read-only storage range at byte offsets
    // inside larger buffers, and a storage output.
    const firstParams = [3, 2], secondParams = [100, 1];
    const atZero = [31, 37, 41, 43];
    const at256 = [5, 7, 11, 13];
    const at512 = [17, 19, 23, 29];
    await params.writeRawBytes(
      _words([...firstParams, 0, 0]),
      dstByteOffset: 256,
    );
    await params.writeRawBytes(
      _words([...secondParams, 0, 0]),
      dstByteOffset: 512,
    );
    await input.writeRawBytes(_words(atZero));
    await input.writeRawBytes(_words(at256), dstByteOffset: 256);
    await input.writeRawBytes(_words(at512), dstByteOffset: 512);

    final shader = gpu.createComputeShader()..loadKernelString(_readOnlyKernel);
    BufferBindingDescriptor range(BufferBindingKind kind, int offset) =>
        BufferBindingDescriptor(kind: kind, sizeBytes: 16, offsetBytes: offset);
    shader.setBufferRange(0, params, range(BufferBindingKind.uniform, 256));
    shader.setBufferRange(
      1,
      input,
      range(BufferBindingKind.readOnlyStorage, 256),
    );
    shader.setBufferRange(2, output, range(BufferBindingKind.storage, 0));

    final results = <String, List<int>>{};
    Future<void> expectOutput(
      String stage,
      List<int> values,
      List<int> addendMultiplier,
    ) async {
      await shader.dispatch(1, 1, 1);
      final actual = Uint32List(4);
      await output.read(actual, 4, dataType: BufferDataType.uint32);
      final expected = [
        for (final value in values)
          value * addendMultiplier[1] + addendMultiplier[0],
      ];
      results[stage] = actual.toList();
      _check(_same(actual, expected), '$stage: read $actual, not $expected');
    }

    await expectOutput('ranges at offset 256', at256, firstParams);
    // The same handles at other offsets must rebuild the binding.
    shader.setBufferRange(
      1,
      input,
      range(BufferBindingKind.readOnlyStorage, 512),
    );
    await expectOutput('storage range moved to 512', at512, firstParams);
    shader.setBufferRange(0, params, range(BufferBindingKind.uniform, 512));
    await expectOutput('uniform range moved to 512', at512, secondParams);

    // Refusals. The first only the device can know (its offset alignment).
    _refuses(
      () =>
          shader.setBufferRange(0, params, range(BufferBindingKind.uniform, 4)),
      'A uniform range at an unaligned offset was accepted',
    );
    _refuses(
      () =>
          shader.setBufferRange(0, output, range(BufferBindingKind.uniform, 0)),
      'A storage buffer was accepted as a uniform range',
    );
    _refuses(
      () => shader.setBufferRange(
        1,
        input,
        range(BufferBindingKind.readOnlyStorage, 768),
      ),
      'A range past the end of its buffer was accepted',
    );
    _refuses(
      () => shader.setBufferRange(
        32,
        input,
        range(BufferBindingKind.readOnlyStorage, 0),
      ),
      'Binding slot 32 was accepted',
    );
    // A refused request leaves the accepted bindings in place.
    await expectOutput('after refused requests', at512, secondParams);

    // A plain bind on the same slot restores the whole-buffer storage binding.
    shader.setBufferAtSlot(1, input);
    shader.loadKernelString(_legacyKernel);
    await expectOutput('legacy binding restored', atZero, secondParams);

    var replaced = 0;
    if (replacementRounds > 0) {
      const live = 64;
      const usage =
          BufferUsage.uniform | BufferUsage.copyDst | BufferUsage.copySrc;
      final ring = [for (var i = 0; i < live; i++) allocate(16, usage)];
      final written = List<int>.filled(live, -1);
      for (var round = 0; round < replacementRounds; round++) {
        final slot = round % live;
        ring[slot].destroy();
        ring[slot] = allocate(16, usage); // No wait: the release is queued.
        written[slot] = 0x0b000000 + round;
        await ring[slot].write(
          Uint32List.fromList([written[slot], 1, 2, 3]),
          4,
          dataType: BufferDataType.uint32,
        );
        replaced++;
      }
      final words = Uint32List(4);
      for (var slot = 0; slot < live; slot++) {
        if (written[slot] < 0) continue;
        await ring[slot].read(words, 4, dataType: BufferDataType.uint32);
        _check(
          _same(words, [written[slot], 1, 2, 3]),
          'Replacement in slot $slot holds $words',
        );
      }
      for (final buffer in ring) {
        buffer.destroy();
      }
    }

    shader.destroy();
    for (final buffer in [params, input, output, vertex, index, legacy]) {
      buffer.destroy();
    }
    _check(
      !params.isValid && !gpu.ownsBuffer(params),
      'A destroyed buffer is still owned',
    );
    await _refusesLater(
      () => input.writeRawBytes(_words(atZero)),
      'A destroyed buffer was written',
    );
    await gpu.waitForSubmittedWork();
    return {
      'device': device,
      'outputs': results,
      'back_to_back_replacements': replaced,
    };
  } finally {
    if (gpu.isInitialized) await gpu.destroy();
  }
}

const _invertKernel = '''
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var nearest: sampler;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let color = textureSampleLevel(src, nearest, (vec2<f32>(id.xy) + .5) / 2., 0.);
  textureStore(dst, vec2<i32>(id.xy), vec4<f32>(1. - color.rgb, color.a));
}
''';

const _readTextureKernel = '''
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let color = vec4<u32>(round(textureLoad(src, vec2<i32>(id.xy), 0) * 255.));
  let index = (id.y * textureDimensions(src).x + id.x) * 4u;
  dst[index] = color.r; dst[index + 1u] = color.g;
  dst[index + 2u] = color.b; dst[index + 3u] = color.a;
}
''';

/// Owned textures, views and samplers on the real device: whole-mip and region
/// uploads, nearest sampling into a storage texture, readback through compute,
/// refusals, and the retirement of every owned resource with its context.
Future<Map<String, Object>> runTextureLaneScenario() async {
  final gpu = Minigpu();
  try {
    await gpu.init();
    final device = _device(gpu);
    final limits = gpu.currentDeviceTextureLimits;
    _check(
      limits != null &&
          limits.deviceEpoch == gpu.deviceEpoch &&
          limits.maxTextureDimension2D >= 4 &&
          limits.maxTextureArrayLayers >= 2,
      'Current-device texture limits are unavailable',
    );

    final source = gpu.createTexture(
      TextureDescriptor(
        width: 2,
        height: 2,
        format: TextureFormat.rgba8Unorm,
        usage: TextureUsage.textureBinding | TextureUsage.copyDst,
      ),
    );
    final view = source.createView();
    final destination = gpu.createTexture(
      TextureDescriptor(
        width: 2,
        height: 2,
        format: TextureFormat.rgba8Unorm,
        usage: TextureUsage.textureBinding | TextureUsage.storageBinding,
      ),
    );
    final outputView = destination.createView();
    final sampler = gpu.createSampler(SamplerDescriptor());
    _check(
      gpu.ownsTexture(source) && view.isValid && sampler.isValid,
      'An owned texture, view or sampler is not live after creation',
    );

    final sampled = gpu.createComputeShader()..loadKernelString(_invertKernel);
    sampled.setTexture(0, view, TextureBindingKind.sampledFloat);
    sampled.setSampler(1, sampler, SamplerBindingKind.nonFiltering);
    sampled.setTexture(2, outputView, TextureBindingKind.writeOnlyStorage);
    await source.write(
      Uint8List.fromList([
        255, 0, 0, 255, 0, 255, 0, 255, //
        0, 0, 255, 255, 255, 255, 255, 255,
      ]),
    );
    await sampled.dispatch(2, 2, 1);

    final output = gpu.createBufferFromDescriptor(
      BufferDescriptor(
        sizeBytes: 64,
        usage: BufferUsage.storage | BufferUsage.copySrc,
      ),
    );
    final read = gpu.createComputeShader()
      ..loadKernelString(_readTextureKernel);
    read.setTexture(0, outputView, TextureBindingKind.sampledUnfilterableFloat);
    read.setBufferRange(
      1,
      output,
      BufferBindingDescriptor(kind: BufferBindingKind.storage, sizeBytes: 64),
    );
    final pixels = Uint32List(16);
    Future<List<int>> readPixels(int count, int groupsX, int groupsY) async {
      await read.dispatch(groupsX, groupsY, 1);
      await output.read(pixels, count * 4, dataType: BufferDataType.uint32);
      return pixels.sublist(0, count * 4);
    }

    final inverted = await readPixels(4, 2, 2);
    _check(
      _same(inverted, const [
        0, 255, 255, 255, 255, 0, 255, 255, //
        255, 255, 0, 255, 0, 0, 0, 255,
      ]),
      'Sampled and stored pixels: $inverted',
    );

    // The last mip of a thin texture is one texel.
    final thin = gpu.createTexture(
      TextureDescriptor(
        width: 1,
        height: 4,
        mipLevelCount: 3,
        format: TextureFormat.rgba8Unorm,
        usage: TextureUsage.textureBinding | TextureUsage.copyDst,
      ),
    );
    final thinView = thin.createView(TextureViewDescriptor(baseMipLevel: 2));
    await thin.write(Uint8List.fromList([42, 99, 150, 255]), mipLevel: 2);
    read.setTexture(0, thinView, TextureBindingKind.sampledUnfilterableFloat);
    final thinTexel = await readPixels(1, 1, 1);
    _check(
      _same(thinTexel, const [42, 99, 150, 255]),
      'Thin mip texel: $thinTexel',
    );

    // Region uploads into one mip of one layer of a layered texture.
    final region = gpu.createTexture(
      TextureDescriptor(
        width: 4,
        height: 4,
        mipLevelCount: 3,
        arrayLayerCount: 2,
        format: TextureFormat.rgba8Unorm,
        usage: TextureUsage.textureBinding | TextureUsage.copyDst,
      ),
    );
    _check(region.supportsRegionWrite, 'The backend lacks region uploads');
    final regionView = region.createView(
      TextureViewDescriptor(baseMipLevel: 1, baseArrayLayer: 1),
    );
    await region.write(Uint8List(16), mipLevel: 1, arrayLayer: 1);
    // A sliced view of a larger host buffer: only its eight bytes are sent.
    final backing = Uint8List.fromList([
      99,
      255,
      0,
      0,
      255,
      0,
      255,
      0,
      255,
      99,
    ]);
    final column = Uint8List.sublistView(backing, 1, 9);
    await region.writeRegion(
      column,
      x: 1,
      y: 0,
      width: 1,
      height: 2,
      mipLevel: 1,
      arrayLayer: 1,
    );
    // The source may be reused as soon as the upload was accepted.
    column.fillRange(0, column.length, 0);
    await region.writeRegion(
      Uint8List.fromList([0, 0, 255, 255]),
      x: 0,
      y: 1,
      width: 1,
      height: 1,
      mipLevel: 1,
      arrayLayer: 1,
    );
    read.setTexture(0, regionView, TextureBindingKind.sampledUnfilterableFloat);
    final regionPixels = await readPixels(4, 2, 2);
    _check(
      _same(regionPixels, const [
        0, 0, 0, 0, 255, 0, 0, 255, //
        0, 0, 255, 255, 0, 255, 0, 255,
      ]),
      'Region upload changed other texels or lost its bytes: $regionPixels',
    );
    final otherLayer = region.createView(
      TextureViewDescriptor(baseMipLevel: 1),
    );
    read.setTexture(0, otherLayer, TextureBindingKind.sampledUnfilterableFloat);
    final untouched = await readPixels(4, 2, 2);
    _check(
      untouched.every((value) => value == 0),
      'A region upload reached another layer: $untouched',
    );
    final lastMip = region.createView(
      TextureViewDescriptor(baseMipLevel: 2, baseArrayLayer: 1),
    );
    await region.writeRegion(
      Uint8List.fromList([42, 99, 150, 255]),
      x: 0,
      y: 0,
      width: 1,
      height: 1,
      mipLevel: 2,
      arrayLayer: 1,
    );
    read.setTexture(0, lastMip, TextureBindingKind.sampledUnfilterableFloat);
    final lastTexel = await readPixels(1, 1, 1);
    _check(
      _same(lastTexel, const [42, 99, 150, 255]),
      'Region upload into the last mip: $lastTexel',
    );

    // Refusals.
    for (final bounds in const [
      [-1, 0, 1, 1],
      [2, 0, 1, 1],
      [1, 0, 2, 1],
      [0, 0, 0, 1],
      [0x100000000, 0, 1, 1],
    ]) {
      await _refusesLater(
        () => region.writeRegion(
          Uint8List(4),
          x: bounds[0],
          y: bounds[1],
          width: bounds[2],
          height: bounds[3],
          mipLevel: 1,
          arrayLayer: 1,
        ),
        'Region bounds $bounds were accepted',
      );
    }
    await _refusesLater(
      () => region.writeRegion(Uint8List(3), x: 0, y: 0, width: 1, height: 1),
      'A region upload of the wrong length was accepted',
    );
    region.destroy();
    _check(!region.supportsRegionWrite, 'A destroyed texture offers uploads');
    await _refusesLater(
      () => region.writeRegion(Uint8List(4), x: 0, y: 0, width: 1, height: 1),
      'A destroyed texture accepted a region upload',
    );
    _refuses(
      () => source.createView(TextureViewDescriptor(baseMipLevel: 1)),
      'A view past the last mip was accepted',
    );
    _refuses(
      () => sampled.setTexture(2, view, TextureBindingKind.writeOnlyStorage),
      'A texture without storage usage was bound as storage',
    );
    _refuses(
      () => sampled.setSampler(1, sampler, SamplerBindingKind.comparison),
      'A sampler without comparison was bound as a comparison sampler',
    );
    source.destroy();
    _check(!view.isValid, 'A view outlived its texture');
    _refuses(
      () => sampled.dispatchFire(1, 1, 1),
      'A shader bound to a destroyed texture was dispatched',
    );
    // Replacing the slot removes the destroyed resource from the shader.
    sampled.setTexture(0, thinView, TextureBindingKind.sampledFloat);
    await sampled.dispatch(2, 2, 1);
    await gpu.waitForSubmittedWork();

    final counted = gpu.resourceAllocationSnapshot;
    await gpu.destroy();
    _check(
      !destination.isValid &&
          !outputView.isValid &&
          !sampler.isValid &&
          !thin.isValid &&
          !output.isValid,
      'Closing the context left an owned resource live',
    );
    final closed = gpu.resourceAllocationSnapshot;
    _check(
      closed.texturesLive == 0 &&
          closed.textureViewsLive == 0 &&
          closed.samplersLive == 0 &&
          closed.buffersLive == 0 &&
          closed.computeShadersLive == 0 &&
          closed.texturesCreated == counted.texturesCreated &&
          closed.texturesRetired == closed.texturesCreated &&
          closed.textureViewsRetired == closed.textureViewsCreated &&
          closed.samplersRetired == closed.samplersCreated,
      'Closing the context did not retire every texture, view and sampler: '
      '${closed.toJson()}',
    );
    _refuses(() => read.platformShader, 'A closed shader exposes its handle');
    return {
      'device': device,
      'max_texture_dimension_2d': limits!.maxTextureDimension2D,
      'max_texture_array_layers': limits.maxTextureArrayLayers,
      'region_upload': limits.supportsRegionWrite,
      'textures_created': closed.texturesCreated,
      'texture_views_created': closed.textureViewsCreated,
      'samplers_created': closed.samplersCreated,
    };
  } finally {
    if (gpu.isInitialized) await gpu.destroy();
  }
}

/// The owning context on the real device: closing it retires every child
/// exactly once, the children stay closed when the context is opened again,
/// and the reopened device is a new epoch that serves new resources.
Future<Map<String, Object>> runOwnedLifetimeScenario() async {
  final gpu = Minigpu();
  try {
    await gpu.init();
    final device = _device(gpu);
    final epoch = gpu.deviceEpoch;
    final before = gpu.resourceAllocationSnapshot;

    final buffer = gpu.createBufferFromDescriptor(
      BufferDescriptor(
        sizeBytes: 16,
        usage: BufferUsage.storage | BufferUsage.copyDst | BufferUsage.copySrc,
      ),
    );
    final texture = gpu.createTexture(
      TextureDescriptor(
        width: 2,
        height: 2,
        format: TextureFormat.rgba8Unorm,
        usage: TextureUsage.textureBinding | TextureUsage.copyDst,
      ),
    );
    final view = texture.createView();
    final sampler = gpu.createSampler(SamplerDescriptor());
    final shader = gpu.createComputeShader()
      ..loadKernelString(_readTextureKernel);
    shader.setTexture(0, view, TextureBindingKind.sampledUnfilterableFloat);
    final held = gpu.resourceAllocationSnapshot;
    _check(
      held.buffersLive == before.buffersLive + 1 &&
          held.texturesLive == before.texturesLive + 1 &&
          held.textureViewsLive == before.textureViewsLive + 1 &&
          held.samplersLive == before.samplersLive + 1 &&
          held.computeShadersLive == before.computeShadersLive + 1,
      'Live counts after one of each: ${held.toJson()}',
    );

    await gpu.destroy();
    _check(!gpu.isInitialized, 'The context stayed open');
    _check(
      !buffer.isValid && !texture.isValid && !view.isValid && !sampler.isValid,
      'Closing the context left a child live',
    );
    final closed = gpu.resourceAllocationSnapshot;
    _check(
      closed.buffersLive == 0 &&
          closed.texturesLive == 0 &&
          closed.textureViewsLive == 0 &&
          closed.samplersLive == 0 &&
          closed.computeShadersLive == 0 &&
          closed.buffersRetired == before.buffersRetired + 1 &&
          closed.texturesRetired == before.texturesRetired + 1 &&
          closed.textureViewsRetired == before.textureViewsRetired + 1 &&
          closed.samplersRetired == before.samplersRetired + 1 &&
          closed.computeShadersRetired == before.computeShadersRetired + 1,
      'Each child was not retired exactly once: ${closed.toJson()}',
    );
    // Destroying a child again changes nothing.
    buffer.destroy();
    texture.destroy();
    view.destroy();
    sampler.destroy();
    shader.destroy();
    final again = gpu.resourceAllocationSnapshot;
    _check(
      again.totalRetired == closed.totalRetired,
      'A second destroy retired a child twice',
    );
    _refuses(
      () => gpu.createBufferFromDescriptor(
        BufferDescriptor(sizeBytes: 16, usage: BufferUsage.storage),
      ),
      'A closed context allocated a buffer',
    );

    await gpu.init();
    _check(
      gpu.deviceEpoch == epoch + 1 && gpu.isDeviceCurrent,
      'Reopening did not start a new device epoch',
    );
    _check(
      !buffer.isValid && !texture.isValid && !view.isValid && !sampler.isValid,
      'Reopening the context revived a child of the closed one',
    );
    await _refusesLater(
      () => buffer.write(Uint32List(4), 4, dataType: BufferDataType.uint32),
      'A buffer of the closed context was written',
    );
    _refuses(() => texture.createView(), 'A closed texture made a view');
    _refuses(
      () => shader.loadKernelString(_readTextureKernel),
      'A closed shader loaded a kernel',
    );
    _check(!gpu.ownsTexture(texture), 'The new epoch owns an old texture');

    final fresh = gpu.createBufferFromDescriptor(
      BufferDescriptor(
        sizeBytes: 16,
        usage: BufferUsage.storage | BufferUsage.copyDst | BufferUsage.copySrc,
      ),
    );
    final values = Uint32List.fromList([19, 29, 31, 37]);
    await fresh.write(values, 4, dataType: BufferDataType.uint32);
    final readBack = Uint32List(4);
    await fresh.read(readBack, 4, dataType: BufferDataType.uint32);
    _check(_same(readBack, values), 'New epoch round trip: $readBack');
    fresh.destroy();
    await gpu.waitForSubmittedWork();
    return {
      'device': device,
      'first_epoch': epoch,
      'reopened_epoch': gpu.deviceEpoch,
      'counters_after_close': closed.toJson(),
    };
  } finally {
    if (gpu.isInitialized) await gpu.destroy();
  }
}

/// A compute shader must refuse a buffer, a texture view and a sampler created
/// by another context, and accept its own. Returns false on a platform that
/// has no independent contexts, where the case does not exist.
Future<bool> runForeignContextScenario() async {
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

    ({Buffer buffer, GpuTextureView view, GpuSampler sampler}) make(
      Minigpu owner,
    ) {
      final texture = owner.createTexture(
        TextureDescriptor(
          width: 2,
          height: 2,
          format: TextureFormat.rgba8Unorm,
          usage: TextureUsage.textureBinding | TextureUsage.copyDst,
        ),
      );
      return (
        buffer: owner.createBufferFromDescriptor(
          BufferDescriptor(
            sizeBytes: 64,
            usage: BufferUsage.storage | BufferUsage.copySrc,
          ),
        ),
        view: texture.createView(),
        sampler: owner.createSampler(SamplerDescriptor()),
      );
    }

    final own = make(gpu), foreign = make(other);
    final storage = BufferBindingDescriptor(
      kind: BufferBindingKind.storage,
      sizeBytes: 64,
    );
    final shader = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var nearest: sampler;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let color = textureSampleLevel(src, nearest, (vec2<f32>(id.xy) + .5) / 2., 0.);
  let pixel = vec4<u32>(round(color * 255.));
  let index = (id.y * 2u + id.x) * 4u;
  dst[index] = pixel.r; dst[index + 1u] = pixel.g;
  dst[index + 2u] = pixel.b; dst[index + 3u] = pixel.a;
}
''');
    _refuses(
      () => shader.setBufferRange(2, foreign.buffer, storage),
      'A buffer of another context was accepted',
    );
    _refuses(
      () => shader.setTexture(0, foreign.view, TextureBindingKind.sampledFloat),
      'A texture view of another context was accepted',
    );
    _refuses(
      () => shader.setSampler(
        1,
        foreign.sampler,
        SamplerBindingKind.nonFiltering,
      ),
      'A sampler of another context was accepted',
    );

    // The same bindings with this context's own resources compute.
    shader.setTexture(0, own.view, TextureBindingKind.sampledFloat);
    shader.setSampler(1, own.sampler, SamplerBindingKind.nonFiltering);
    shader.setBufferRange(2, own.buffer, storage);
    await own.view.texture.write(
      Uint8List.fromList([
        10, 20, 30, 255, 40, 50, 60, 255, //
        70, 80, 90, 255, 100, 110, 120, 255,
      ]),
    );
    await shader.dispatch(2, 2, 1);
    final pixels = Uint32List(16);
    await own.buffer.read(pixels, 16, dataType: BufferDataType.uint32);
    _check(
      _same(pixels, const [
        10, 20, 30, 255, 40, 50, 60, 255, //
        70, 80, 90, 255, 100, 110, 120, 255,
      ]),
      'Own bindings computed $pixels',
    );
    await gpu.waitForSubmittedWork();
    return true;
  } finally {
    if (other != null && other.isInitialized) await other.destroy();
    if (gpu.isInitialized) await gpu.destroy();
  }
}
