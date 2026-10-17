/// Browser-only coverage for the WebBuffer read/write contract: an offset read
/// lands at the front of the output, any TypedData is read and written as
/// bytes, a count the input or the GPU buffer cannot hold throws, and reads
/// that overlap (or a destroy or a write during one) stay ordered.
///
/// Opens a VISIBLE Chrome window (dart_test.yaml runs it headed). From
/// minigpu_web:
/// flutter test -p chrome test/web_read_write_contract_test.dart --web-browser-flag=--enable-unsafe-webgpu
@TestOn('browser')
library;

import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:minigpu_web/minigpu_web.dart';
import 'package:test/test.dart';

void main() {
  group('WebBuffer read/write contract', () {
    late MinigpuWeb gpu;

    setUpAll(() async {
      gpu = MinigpuWeb.createForTest();
      await gpu.initializeContext();
    });

    tearDownAll(() async {
      await gpu.destroyContext();
    });

    // Sixteen u32 words 0x100 + i.
    final words = List<int>.generate(16, (i) => 0x100 + i);

    Future<PlatformBuffer> filled(BufferDataType type) async {
      final buffer = gpu.createBuffer(64, type);
      await buffer.writeRawBytes(
        Uint32List.fromList(words).buffer.asUint8List(),
      );
      return buffer;
    }

    test('an offset read lands at the front of the output', () async {
      final buffer = await filled(BufferDataType.uint32);
      addTearDown(buffer.destroy);
      final out = Uint32List(8)..fillRange(0, 8, 0xdead);
      await buffer.read(
        out,
        4,
        elementOffset: 6,
        dataType: BufferDataType.uint32,
      );
      expect(out.sublist(0, 4), orderedEquals(words.sublist(6, 10)));
      expect(out.sublist(4), everyElement(0xdead));
    });

    test('64-bit reads copy bytes into a ByteData view', () async {
      final buffer = await filled(BufferDataType.int64);
      addTearDown(buffer.destroy);
      final backing = Uint8List(40);
      await buffer.read(
        ByteData.sublistView(backing, 8, 24),
        2,
        elementOffset: 1,
        dataType: BufferDataType.int64,
      );
      // int64 elements 1 and 2 are u32 words 2..5, landing at view byte 0.
      expect(backing.sublist(0, 8), everyElement(0));
      expect(
        Uint32List.sublistView(backing, 8, 24),
        orderedEquals(words.sublist(2, 6)),
      );
      expect(backing.sublist(24), everyElement(0));

      final u64 = ByteData(8);
      await buffer.read(
        u64,
        1,
        elementOffset: 7,
        dataType: BufferDataType.uint64,
      );
      expect(u64.getUint32(0, Endian.little), words[14]);
      expect(u64.getUint32(4, Endian.little), words[15]);
    });

    test('a list of another element type writes its bytes', () async {
      final buffer = gpu.createBuffer(16, BufferDataType.uint32);
      addTearDown(buffer.destroy);
      final bytes = Uint32List.fromList([7, 8, 9, 10]).buffer.asUint8List();
      await buffer.write(bytes, 4, dataType: BufferDataType.uint32);
      final out = Uint32List(4);
      await buffer.read(out, 4, dataType: BufferDataType.uint32);
      expect(out, orderedEquals([7, 8, 9, 10]));

      final raw = ByteData(8)
        ..setUint32(0, 11, Endian.little)
        ..setUint32(4, 12, Endian.little);
      await buffer.write(raw, 2, dataType: BufferDataType.uint32);
      await buffer.read(out, 4, dataType: BufferDataType.uint32);
      expect(out, orderedEquals([11, 12, 9, 10]));
    });

    test('writeRawBytes at offset 0 takes an unaligned view', () async {
      final buffer = gpu.createBuffer(16, BufferDataType.uint32);
      addTearDown(buffer.destroy);
      final expected = Uint32List.fromList([21, 22, 23, 24]);
      // The payload starts one byte into its backing store.
      final source = Uint8List(18)
        ..setRange(1, 17, expected.buffer.asUint8List());
      await buffer.writeRawBytes(Uint8List.sublistView(source, 1, 17));
      final out = Uint32List(4);
      await buffer.read(out, 4, dataType: BufferDataType.uint32);
      expect(out, orderedEquals(expected));
    });

    test('a count the input or the GPU buffer cannot hold throws', () async {
      final buffer = gpu.createBuffer(16, BufferDataType.float32);
      addTearDown(buffer.destroy);
      const f32 = BufferDataType.float32;
      // The list holds 2 elements; the GPU buffer holds 4.
      await expectLater(
        buffer.write(Float32List(2), 3, dataType: f32),
        throwsRangeError,
      );
      await expectLater(
        buffer.write(Float32List(8), 8, dataType: f32),
        throwsRangeError,
      );
      await expectLater(
        buffer.read(Float32List(2), 3, dataType: f32),
        throwsRangeError,
      );
      await expectLater(
        buffer.read(Float32List(4), 4, elementOffset: 1, dataType: f32),
        throwsRangeError,
      );
    });

    test('overlapping reads, and a destroy while they are pending', () async {
      final buffer = await filled(BufferDataType.uint32);
      final a = Uint32List(16), b = Uint32List(16), c = Uint32List(4);
      final reads = Future.wait([
        buffer.read(a, 16, dataType: BufferDataType.uint32),
        buffer.read(b, 16, dataType: BufferDataType.uint32),
        buffer.read(c, 4, elementOffset: 12, dataType: BufferDataType.uint32),
      ]);
      // Queued behind the reads, not freed under them.
      buffer.destroy();
      expect(buffer.webBufferHandle, 0);
      await reads;
      expect(a, orderedEquals(words));
      expect(b, orderedEquals(words));
      expect(c, orderedEquals(words.sublist(12)));
      await expectLater(
        buffer.read(a, 1, dataType: BufferDataType.uint32),
        throwsStateError,
      );
    });

    test('a write while a read is pending neither hangs nor reaches it',
        () async {
      final buffer = await filled(BufferDataType.uint32);
      addTearDown(buffer.destroy);
      final before = Uint32List(16), after = Uint32List(16);
      // An idle queue starts the read at once: its copy is submitted and it is
      // asleep in C, holding minigpu's GPU lock, when the write below runs.
      final pending = buffer.read(before, 16, dataType: BufferDataType.uint32);
      await buffer.write(
        Uint32List(16)..fillRange(0, 16, 7),
        16,
        dataType: BufferDataType.uint32,
      );
      await pending;
      expect(before, orderedEquals(words));
      await buffer.read(after, 16, dataType: BufferDataType.uint32);
      expect(after, everyElement(7));
    });
  });
}
