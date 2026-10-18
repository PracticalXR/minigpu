// Actual public Dart/Wasm API against the generated minigpu WebGPU backend.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';

void check(bool condition, String message) {
  if (!condition) throw StateError(message);
}

void expectRefusal(void Function() operation, String message) {
  try {
    operation();
  } on ArgumentError {
    return;
  } on StateError {
    return;
  }
  throw StateError(message);
}

Future<void> main() async {
  final gpu = Minigpu();
  var initialized = false;
  var result = 'passed';
  try {
    await gpu.init();
    initialized = true;
    final source = gpu.createTexture(TextureDescriptor(
      width: 2,
      height: 2,
      format: TextureFormat.rgba8Unorm,
      usage: TextureUsage.textureBinding | TextureUsage.copyDst,
    ));
    final view = source.createView();
    final destination = gpu.createTexture(TextureDescriptor(
      width: 2,
      height: 2,
      format: TextureFormat.rgba8Unorm,
      usage: TextureUsage.textureBinding | TextureUsage.storageBinding,
    ));
    final outputView = destination.createView();
    final sampler = gpu.createSampler(SamplerDescriptor());
    final sampled = gpu.createComputeShader()..loadKernelString('''
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var nearest: sampler;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let color = textureSampleLevel(src, nearest, (vec2<f32>(id.xy) + .5) / 2., 0.);
  textureStore(dst, vec2<i32>(id.xy), vec4<f32>(1. - color.rgb, color.a));
}
''');
    sampled.setTexture(0, view, TextureBindingKind.sampledFloat);
    sampled.setSampler(1, sampler, SamplerBindingKind.nonFiltering);
    sampled.setTexture(2, outputView, TextureBindingKind.writeOnlyStorage);
    await source.write(Uint8List.fromList([
      255, 0, 0, 255, 0, 255, 0, 255,
      0, 0, 255, 255, 255, 255, 255, 255,
    ]));
    await sampled.dispatch(2, 2, 1);

    final output = gpu.createBufferFromDescriptor(BufferDescriptor(
      sizeBytes: 64,
      usage: BufferUsage.storage | BufferUsage.copySrc,
    ));
    final read = gpu.createComputeShader()..loadKernelString('''
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let color = vec4<u32>(round(textureLoad(src, vec2<i32>(id.xy), 0) * 255.));
  let index = (id.y * textureDimensions(src).x + id.x) * 4u;
  dst[index] = color.r; dst[index + 1u] = color.g;
  dst[index + 2u] = color.b; dst[index + 3u] = color.a;
}
''');
    read.setTexture(0, outputView, TextureBindingKind.sampledUnfilterableFloat);
    read.setBufferRange(1, output, BufferBindingDescriptor(
      kind: BufferBindingKind.storage, sizeBytes: 64,
    ));
    await read.dispatch(2, 2, 1);
    final pixels = Uint32List(16);
    await output.read(pixels, 16, dataType: BufferDataType.uint32);
    const expected = [0, 255, 255, 255, 255, 0, 255, 255,
      255, 255, 0, 255, 0, 0, 0, 255];
    for (var i = 0; i < expected.length; i++) {
      check(pixels[i] == expected[i], 'Sample/storage pixel mismatch: $pixels');
    }

    final thin = gpu.createTexture(TextureDescriptor(
      width: 1, height: 4, mipLevelCount: 3,
      format: TextureFormat.rgba8Unorm,
      usage: TextureUsage.textureBinding | TextureUsage.copyDst,
    ));
    final thinView = thin.createView(TextureViewDescriptor(baseMipLevel: 2));
    await thin.write(Uint8List.fromList([42, 99, 150, 255]), mipLevel: 2);
    read.setTexture(0, thinView, TextureBindingKind.sampledUnfilterableFloat);
    await read.dispatch(1, 1, 1);
    await output.read(pixels, 4, dataType: BufferDataType.uint32);
    check(pixels.take(4).join(',') == '42,99,150,255', 'Thin mip extent was lost');

    final region = gpu.createTexture(TextureDescriptor(
      width: 4, height: 4, mipLevelCount: 3, arrayLayerCount: 2,
      format: TextureFormat.rgba8Unorm,
      usage: TextureUsage.textureBinding | TextureUsage.copyDst,
    ));
    check(region.supportsRegionWrite, 'Loaded module lacks region uploads');
    final regionView = region.createView(TextureViewDescriptor(
      baseMipLevel: 1, baseArrayLayer: 1,
    ));
    await region.write(Uint8List(16), mipLevel: 1, arrayLayer: 1);
    final backing = Uint8List.fromList([99, 255, 0, 0, 255, 0, 255, 0, 255, 99]);
    final column = Uint8List.sublistView(backing, 1, 9);
    await region.writeRegion(column, x: 1, y: 0, width: 1, height: 2,
        mipLevel: 1, arrayLayer: 1);
    column.fillRange(0, column.length, 0);
    await region.writeRegion(Uint8List.fromList([0, 0, 255, 255]),
        x: 0, y: 1, width: 1, height: 1, mipLevel: 1, arrayLayer: 1);
    read.setTexture(0, regionView, TextureBindingKind.sampledUnfilterableFloat);
    await read.dispatch(2, 2, 1);
    await output.read(pixels, 16, dataType: BufferDataType.uint32);
    const regionExpected = [0, 0, 0, 0, 255, 0, 0, 255,
      0, 0, 255, 255, 0, 255, 0, 255];
    check(pixels.join(',') == regionExpected.join(','),
        'Region upload changed untouched texels or lost source ownership: $pixels');
    final untouchedView = region.createView(TextureViewDescriptor(baseMipLevel: 1));
    read.setTexture(0, untouchedView, TextureBindingKind.sampledUnfilterableFloat);
    await read.dispatch(2, 2, 1);
    await output.read(pixels, 16, dataType: BufferDataType.uint32);
    check(pixels.every((value) => value == 0), 'Region changed another layer');
    final lastMip = region.createView(TextureViewDescriptor(
        baseMipLevel: 2, baseArrayLayer: 1));
    await region.writeRegion(Uint8List.fromList([42, 99, 150, 255]),
        x: 0, y: 0, width: 1, height: 1, mipLevel: 2, arrayLayer: 1);
    read.setTexture(0, lastMip, TextureBindingKind.sampledUnfilterableFloat);
    await read.dispatch(1, 1, 1);
    await output.read(pixels, 4, dataType: BufferDataType.uint32);
    check(pixels.take(4).join(',') == '42,99,150,255', 'Region thin mip failed');
    for (final bounds in [[-1, 0, 1, 1], [2, 0, 1, 1], [1, 0, 2, 1],
      [0, 0, 0, 1], [0x100000000, 0, 1, 1]]) {
      expectRefusal(() => region.writeRegion(Uint8List(4), x: bounds[0],
          y: bounds[1], width: bounds[2], height: bounds[3],
          mipLevel: 1, arrayLayer: 1), 'Unsafe region bounds accepted');
    }
    expectRefusal(() => region.writeRegion(Uint8List(3), x: 0, y: 0,
        width: 1, height: 1), 'Incorrect region byte length accepted');
    region.destroy();
    check(!region.supportsRegionWrite, 'Destroyed texture claims upload support');
    expectRefusal(() => region.writeRegion(Uint8List(4), x: 0, y: 0,
        width: 1, height: 1), 'Destroyed texture accepted a region upload');
    expectRefusal(
      () => source.createView(TextureViewDescriptor(baseMipLevel: 1)),
      'Out-of-range view accepted',
    );
    expectRefusal(
      () => sampled.setTexture(2, view, TextureBindingKind.writeOnlyStorage),
      'Wrong texture usage accepted',
    );
    source.destroy();
    expectRefusal(() => sampled.dispatchFire(1, 1, 1), 'Dead texture was dispatched');
    // Replacing the old resource must remove its stale wrapper from the slot.
    sampled.setTexture(0, thinView, TextureBindingKind.sampledFloat);
    await sampled.dispatch(2, 2, 1);
    await gpu.waitForSubmittedWork();
    await gpu.destroy();
    initialized = false;
    check(!destination.isValid && !outputView.isValid && !sampler.isValid,
        'Context teardown left an owned resource live');
    expectRefusal(() => read.platformShader, 'Closed shader still exposes its handle');
  } catch (error, stack) {
    result = 'failed: $error\n$stack';
  } finally {
    if (initialized) {
      try { await gpu.destroy(); } catch (error) { result += '; cleanup: $error'; }
    }
    globalContext.setProperty('bufferSmokeResult'.toJS, result.toJS);
  }
}
