/// The f32 shared-output present has a port-completed twin, so the calling
/// isolate no longer blocks on the GPU wait the synchronous copy makes. The
/// texel is read back through Dawn to show the async copy really landed.
///
///     dart test test/shared_output_f32_async_test.dart
@TestOn('vm && windows')
library;

import 'dart:typed_data';

import 'package:minigpu_ffi/minigpu_ffi.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:test/test.dart';

void main() {
  late MinigpuFfi platform;

  setUpAll(() async {
    platform = MinigpuFfi();
    await platform.initializeContext();
  });

  tearDownAll(() async {
    await platform.destroyContext();
  });

  test('copyFromBufferF32Async completes with the copy landed', () async {
    const w = 8;
    const h = 8;
    final texture = platform.createSharedOutputTexture(w, h);
    if (texture == null) {
      markTestSkipped('no shared output texture on this adapter/backend');
      return;
    }
    addTearDown(texture.destroy);

    // 0.2 * 255 is 51 plus float error, nowhere near a rounding tie.
    final pixels = Float32List(w * h * 4);
    for (var i = 0; i < pixels.length; i += 4) {
      pixels.setAll(i, [1, 0.2, 0, 1]);
    }
    final buffer = platform.createBuffer(
      pixels.lengthInBytes,
      BufferDataType.float32,
    );
    addTearDown(buffer.destroy);
    await buffer.write(pixels, pixels.length);

    final ok = await (texture as FfiSharedOutputTexture)
        .copyFromBufferF32Async(buffer)
        .timeout(const Duration(seconds: 20));
    expect(ok, isTrue);
    // A bgra8unorm texel's bytes B,G,R,A read as a little-endian u32.
    expect(texture.debugReadFirstPixelDawn(), 0xFFFF3300);
  });
}
