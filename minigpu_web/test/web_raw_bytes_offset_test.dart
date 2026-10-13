/// Browser-only coverage for WebBuffer.writeRawBytes with a nonzero offset.
///
/// Run from the staged Flutter app so package Web assets, the WASM module, and
/// Chrome's real WebGPU implementation are all present:
/// flutter test -p chrome <this file> --web-browser-flag=--enable-unsafe-webgpu
@TestOn('browser')
library;

import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:minigpu_web/minigpu_web.dart';
import 'package:test/test.dart';

void main() {
  group('WebBuffer.writeRawBytes offset upload', () {
    late MinigpuWeb gpu;

    setUpAll(() async {
      gpu = MinigpuWeb.createForTest();
      await gpu.initializeContext();
    });

    tearDownAll(() async {
      await gpu.destroyContext();
    });

    test(
      'writes exactly a Uint8List subview at a nonzero byte offset',
      () async {
        final buffer = gpu.createBuffer(32, BufferDataType.uint8);
        addTearDown(buffer.destroy);

        final initial = Uint8List.fromList(
          List<int>.generate(32, (i) => 0x80 + i),
        );
        await buffer.write(
          initial,
          initial.length,
          dataType: BufferDataType.uint8,
        );

        // The payload starts four bytes into its backing store. This catches an
        // interop path that uploads the whole ArrayBuffer instead of the view.
        final source = Uint8List.fromList(<int>[
          0xee,
          0xee,
          0xee,
          0xee,
          1,
          3,
          5,
          7,
          9,
          11,
          13,
          15,
          0xdd,
          0xdd,
        ]);
        final payload = Uint8List.sublistView(source, 4, 12);
        await buffer.writeRawBytes(payload, dstByteOffset: 8);
        await gpu.waitForSubmittedWork();

        final actual = Uint8List(32);
        await buffer.read(
          actual,
          actual.length,
          dataType: BufferDataType.uint8,
        );

        final expected = Uint8List.fromList(initial);
        expected.setRange(8, 16, payload);
        expect(actual, orderedEquals(expected));
      },
    );

    test('rejects unaligned and out-of-range offset uploads', () async {
      final buffer = gpu.createBuffer(32, BufferDataType.uint8);
      addTearDown(buffer.destroy);

      expect(
        () => buffer.writeRawBytes(Uint8List(4), dstByteOffset: 2),
        throwsArgumentError,
      );
      expect(
        () => buffer.writeRawBytes(Uint8List(8), dstByteOffset: 28),
        throwsRangeError,
      );
    });
  });
}
