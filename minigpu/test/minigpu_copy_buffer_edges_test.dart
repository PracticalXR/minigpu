// Minigpu.copyBuffer edges on a real device: counts in the source's element
// type (1-, 2- and 8-byte types ending inside a word), a copy past 65535
// workgroups, nothing written past the count, self-copy, and copies in flight
// together. Arguments and params against a fake backend:
// minigpu_public_api_test.dart.

import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  late Minigpu gpu;

  setUpAll(() async {
    gpu = Minigpu();
    await gpu.init();
  });

  tearDownAll(() async {
    gpu.destroyAllTrackedShaders();
    await gpu.destroy();
  });

  test('uint8 counts bytes and keeps the rest of a partial word', () async {
    final src = gpu.createBuffer(16, BufferDataType.uint8);
    final dst = gpu.createBuffer(16, BufferDataType.uint8);
    await src.write(Uint8List.fromList(List.generate(16, (i) => i + 1)), 16);
    for (final count in [1, 2, 3, 5, 7, 12, 16]) {
      await dst.write(Uint8List(16)..fillRange(0, 16, 0xEE), 16);
      await gpu.copyBuffer(src, dst, elementCount: count);
      final out = Uint8List(16);
      await dst.read(out, 16);
      expect(out, [
        for (var i = 0; i < 16; i++) i < count ? i + 1 : 0xEE,
      ], reason: 'elementCount $count');
    }
  });

  test('int16 copies half a word', () async {
    final src = gpu.createBuffer(16, BufferDataType.int16);
    final dst = gpu.createBuffer(16, BufferDataType.int16);
    await src.write(Int16List.fromList(List.generate(8, (i) => -1 - i)), 8);
    await dst.write(Int16List(8)..fillRange(0, 8, 0x1EEE), 8);
    await gpu.copyBuffer(src, dst, elementCount: 3);
    final out = Int16List(8);
    await dst.read(out, 8);
    expect(out, [-1, -2, -3, for (var i = 0; i < 5; i++) 0x1EEE]);
  });

  test('float64 counts 8-byte elements', () async {
    final src = gpu.createBuffer(32, BufferDataType.float64);
    final dst = gpu.createBuffer(32, BufferDataType.float64);
    await src.write(Float64List.fromList([1.5, -2.25, 3e100, 4.0]), 4);
    await dst.write(Float64List(4)..fillRange(0, 4, 9.0), 4);
    await gpu.copyBuffer(src, dst, elementCount: 3);
    final out = Float64List(4);
    await dst.read(out, 4);
    expect(out, [1.5, -2.25, 3e100, 9.0]);
  });

  test('a copy past 65535 workgroups folds into y and stops at the count',
      () async {
    const rowWords = 65535 * 64; // one full dispatch row
    const words = rowWords + 128;
    const count = rowWords + 64; // 64 words into the second row, 64 short
    final src = gpu.createBuffer(words * 4, BufferDataType.uint32);
    final dst = gpu.createBuffer(words * 4, BufferDataType.uint32);
    final data = Uint32List(words);
    for (var i = 0; i < words; i++) {
      data[i] = i + 1;
    }
    await src.write(data, words);
    await dst.write(Uint32List(words)..fillRange(0, words, 0xEEEEEEEE), words);
    await gpu.copyBuffer(src, dst, elementCount: count);
    final out = Uint32List(words);
    await dst.read(out, words);
    var firstWrong = -1;
    for (var i = 0; i < words; i++) {
      if (out[i] != (i < count ? i + 1 : 0xEEEEEEEE)) {
        firstWrong = i;
        break;
      }
    }
    expect(
      firstWrong,
      -1,
      reason: firstWrong < 0
          ? null
          : 'word $firstWrong is 0x${out[firstWrong].toRadixString(16)}',
    );
  });

  test('copying a buffer onto itself leaves it unchanged', () async {
    final buffer = gpu.createBuffer(64, BufferDataType.uint32);
    final data = Uint32List.fromList(List.generate(16, (i) => i * 3));
    await buffer.write(data, 16);
    await gpu.copyBuffer(buffer, buffer, elementCount: 16);
    final out = Uint32List(16);
    await buffer.read(out, 16);
    expect(out, data);
  });

  test('copies issued without awaiting each keep their own count', () async {
    final src = gpu.createBuffer(64, BufferDataType.uint32);
    final a = gpu.createBuffer(64, BufferDataType.uint32);
    final b = gpu.createBuffer(64, BufferDataType.uint32);
    await src.write(Uint32List.fromList(List.generate(16, (i) => 100 + i)), 16);
    await a.write(Uint32List(16)..fillRange(0, 16, 7), 16);
    await b.write(Uint32List(16)..fillRange(0, 16, 7), 16);
    await Future.wait([
      gpu.copyBuffer(src, a, elementCount: 5),
      gpu.copyBuffer(src, b, elementCount: 11),
    ]);
    final outA = Uint32List(16);
    final outB = Uint32List(16);
    await a.read(outA, 16);
    await b.read(outB, 16);
    expect(outA, [for (var i = 0; i < 16; i++) i < 5 ? 100 + i : 7]);
    expect(outB, [for (var i = 0; i < 16; i++) i < 11 ? 100 + i : 7]);
  });

  test('refuses counts past either buffer and destroyed buffers', () async {
    final src = gpu.createBuffer(64, BufferDataType.float32);
    final small = gpu.createBuffer(32, BufferDataType.float32);
    await expectLater(
      gpu.copyBuffer(src, small, elementCount: 9),
      throwsRangeError,
    );
    await expectLater(
      gpu.copyBuffer(src, small, elementCount: -1),
      throwsRangeError,
    );
    small.destroy();
    await expectLater(
      gpu.copyBuffer(src, small, elementCount: 1),
      throwsStateError,
    );
  });
}
