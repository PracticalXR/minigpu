/// A typed write of whole words skips the scratch copy: the list's own store
/// goes to C through the leaf raw write. These pin that the same bytes land as
/// through the typed exports, from views, prefixes, ByteData, other element
/// types and payloads over one 32 MiB chunk, and that a partial last word
/// still takes the zero-padding path.
///
///     dart test test/typed_write_leaf_test.dart
@TestOn('vm')
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

  PlatformBuffer buffer(
    int bytes, [
    BufferDataType type = BufferDataType.float32,
  ]) {
    final b = platform.createBuffer(bytes, type);
    addTearDown(b.destroy);
    return b;
  }

  test('a view uploads its own elements, not its backing start', () async {
    final b = buffer(16);
    final backing = Float32List.fromList([9, 9, 1, 2, 3, 4, 9, 9]);
    await b.write(Float32List.sublistView(backing, 2, 6), 4);
    final out = Float32List(4);
    await b.read(out, 4);
    expect(out, [1, 2, 3, 4]);
  });

  test('elementCount below the list length uploads only that prefix', () async {
    final b = buffer(16);
    await b.write(Float32List.fromList([5, 6, 7, 8]), 4);
    await b.write(Float32List.fromList([1, 2, 3, 4]), 2);
    final out = Float32List(4);
    await b.read(out, 4);
    expect(out, [1, 2, 7, 8]);
  });

  test('ByteData and other element types upload their raw bytes', () async {
    final b = buffer(8, BufferDataType.uint32);
    final data = ByteData(8)
      ..setUint32(0, 0x04030201, Endian.host)
      ..setUint32(4, 0x08070605, Endian.host);
    await b.write(data, 2, dataType: BufferDataType.uint32);
    final out = Uint32List(2);
    await b.read(out, 2, dataType: BufferDataType.uint32);
    expect(out, [0x04030201, 0x08070605]);

    final floats = Float32List.fromList([1.5, -2]);
    await b.write(floats, 2, dataType: BufferDataType.uint32);
    await b.read(out, 2, dataType: BufferDataType.uint32);
    expect(out, Uint32List.view(floats.buffer));
  });

  test('packed and 64-bit whole-word writes read back unchanged', () async {
    final u8 = buffer(8, BufferDataType.uint8);
    final bytes = Uint8List.fromList([1, 2, 3, 4, 5, 6, 7, 8]);
    await u8.write(bytes, 8, dataType: BufferDataType.uint8);
    final outU8 = Uint8List(8);
    await u8.read(outU8, 8, dataType: BufferDataType.uint8);
    expect(outU8, bytes);

    final i16 = buffer(8, BufferDataType.int16);
    final shorts = Int16List.fromList([-1, 2, -3, 4]);
    await i16.write(shorts, 4, dataType: BufferDataType.int16);
    final outI16 = Int16List(4);
    await i16.read(outI16, 4, dataType: BufferDataType.int16);
    expect(outI16, shorts);

    final f64 = buffer(16, BufferDataType.float64);
    final doubles = Float64List.fromList([1.25, -3.5]);
    await f64.write(doubles, 2, dataType: BufferDataType.float64);
    final outF64 = Float64List(2);
    await f64.read(outF64, 2, dataType: BufferDataType.float64);
    expect(outF64, doubles);
  });

  test('a partial last word is still zero-filled, not merged', () async {
    final b = buffer(8, BufferDataType.uint8);
    final nines = Uint8List(8)..fillRange(0, 8, 9);
    await b.write(nines, 8, dataType: BufferDataType.uint8);
    final five = Uint8List.fromList([1, 2, 3, 4, 5]);
    await b.write(five, 5, dataType: BufferDataType.uint8);
    final out = Uint8List(8);
    await b.read(out, 8, dataType: BufferDataType.uint8);
    expect(out, [1, 2, 3, 4, 5, 0, 0, 0]);
  });

  test('a write over one 32 MiB chunk lands every chunk in place', () async {
    const n = 10 * 1024 * 1024; // 40 MiB of float32: a full chunk + 8 MiB
    final values = Float32List(n);
    for (var i = 0; i < n; i += 4096) {
      values[i] = i.toDouble();
    }
    values[n - 1] = -1;
    final b = buffer(n * 4);
    await b.write(values, n);
    final out = Float32List(1);
    for (final i in [0, 8 * 1024 * 1024 - 4096, 8 * 1024 * 1024, n - 4096]) {
      await b.read(out, 1, elementOffset: i);
      expect(out.first, i.toDouble(), reason: 'element $i');
    }
    await b.read(out, 1, elementOffset: n - 1);
    expect(out.first, -1);
  });
}
