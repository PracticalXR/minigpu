/// FfiTexture uploads hand C the list's own storage through a LEAF call
/// (`Uint8List.address`) instead of a malloc'd copy of the frame. A view with
/// a non-zero offsetInBytes is the case a wrong address gets wrong, so the
/// texels are read back through a compute pass and compared.
///
///     dart test test/texture_upload_leaf_test.dart
@TestOn('vm')
library;

import 'dart:typed_data';

import 'package:minigpu_ffi/minigpu_ffi.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:test/test.dart';

const String _readTexture = '''
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let color = vec4<u32>(round(textureLoad(src, vec2<i32>(id.xy), 0) * 255.));
  let index = (id.y * textureDimensions(src).x + id.x) * 4u;
  dst[index] = color.r; dst[index + 1u] = color.g;
  dst[index + 2u] = color.b; dst[index + 3u] = color.a;
}
''';

void main() {
  late MinigpuFfi platform;

  setUpAll(() async {
    platform = MinigpuFfi();
    await platform.initializeContext();
  });

  tearDownAll(() async {
    await platform.destroyContext();
  });

  PlatformTexture texture2x2() {
    final texture = platform.createTexture(
      TextureDescriptor(
        width: 2,
        height: 2,
        format: TextureFormat.rgba8Unorm,
        usage: TextureUsage.textureBinding | TextureUsage.copyDst,
      ),
    );
    addTearDown(texture.destroy);
    return texture;
  }

  test('uploads from offset views land the right texels', () async {
    final texture = texture2x2();
    final view = texture.createView(TextureViewDescriptor());
    addTearDown(view.destroy);
    final output = platform.createBufferFromDescriptor(
      BufferDescriptor(sizeBytes: 64, usage: BufferUsage.legacy),
    );
    addTearDown(output.destroy);
    final shader = platform.createComputeShader()
      ..loadKernelString(_readTexture);
    addTearDown(shader.destroy);
    shader.setTexture(0, view, TextureBindingKind.sampledUnfilterableFloat);
    shader.setBufferRange(
      1,
      output,
      BufferBindingDescriptor(kind: BufferBindingKind.storage, sizeBytes: 64),
    );

    Future<List<int>> texels() async {
      await shader.dispatch(2, 2, 1);
      final pixels = Uint32List(16);
      await output.read(pixels, 16, dataType: BufferDataType.uint32);
      return pixels;
    }

    const frame = [
      255, 0, 0, 255, 0, 255, 0, 255, //
      0, 0, 255, 255, 255, 255, 255, 255,
    ];
    // Junk in front: the upload must start at the view's offsetInBytes.
    final backing = Uint8List(3 + frame.length)..setRange(3, 19, frame);
    await texture.write(Uint8List.sublistView(backing, 3));
    expect(await texels(), frame);

    if (!texture.supportsRegionWrite) return;
    final region = Uint8List.fromList([9, 9, 1, 2, 3, 4]);
    await texture.writeRegion(
      Uint8List.sublistView(region, 2),
      x: 1,
      y: 1,
      width: 1,
      height: 1,
    );
    expect(await texels(), [...frame.sublist(0, 12), 1, 2, 3, 4]);
  });

  test('a wrongly sized upload is still refused by the native check', () async {
    final texture = texture2x2();
    await expectLater(texture.write(Uint8List(15)), throwsArgumentError);
    if (!texture.supportsRegionWrite) return;
    await expectLater(
      texture.writeRegion(Uint8List(3), x: 0, y: 0, width: 1, height: 1),
      throwsArgumentError,
    );
  });
}
