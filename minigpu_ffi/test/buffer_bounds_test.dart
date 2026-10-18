/// FfiBuffer range checks, copy-out and destroy-during-read.
///
/// A typed write past the end of the GPU buffer used to throw a C++ exception
/// through the FFI boundary (a process abort); a read past the end completed
/// with stale scratch bytes; a ByteData read landed at the GPU-side offset and
/// a list of another element type was left untouched; and destroy() freed the
/// read scratch the worker was about to fill. Each test pins the Dart side.
///
/// The destroy-during-read tests also need the native side to delete a buffer
/// behind its queued reads (Buffer::destroyQueued); against a library built
/// before that they can crash, which is the bug they guard.
///
///     dart test test/buffer_bounds_test.dart
@TestOn('vm')
library;

import 'dart:typed_data';

import 'package:minigpu_ffi/minigpu_ffi.dart';
import 'package:minigpu_ffi/minigpu_ffi_completion.dart';
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

  group('range checks', () {
    test('a typed write past the end throws; a fitting one lands', () async {
      final b = buffer(16);
      await expectLater(b.write(Float32List(5), 5), throwsRangeError);
      await b.write(Float32List.fromList([1, 2, 3, 4]), 4);
      final out = Float32List(4);
      await b.read(out, 4);
      expect(out, [1, 2, 3, 4]);
    });

    test('packed types are checked against the padded allocation', () async {
      // Six uint8 elements occupy two 4-byte words natively, so eight fit.
      final b = buffer(6, BufferDataType.uint8);
      await b.write(Uint8List(8), 8, dataType: BufferDataType.uint8);
      await expectLater(
        b.write(Uint8List(9), 9, dataType: BufferDataType.uint8),
        throwsRangeError,
      );
    });

    test('a descriptor buffer is checked against its exact size', () async {
      final b = platform.createBufferFromDescriptor(
        BufferDescriptor(sizeBytes: 8, usage: BufferUsage.legacy),
      );
      addTearDown(b.destroy);
      await expectLater(b.write(Float32List(3), 3), throwsRangeError);
    });

    test('a read past the end throws RangeError', () async {
      final b = buffer(16);
      await expectLater(b.read(Float32List(5), 5), throwsRangeError);
      await expectLater(
        b.read(Float32List(4), 4, elementOffset: 1),
        throwsRangeError,
      );
      await b.read(Float32List(3), 3, elementOffset: 1);
    });

    test('writeRawBytes rejects a negative or overlong destination', () async {
      final b = buffer(16);
      final one = Float32List.fromList([1]).buffer.asUint8List();
      // -4 passes the alignment test; C would see it as a size_t near 2^64.
      await expectLater(
        b.writeRawBytes(one, dstByteOffset: -4),
        throwsRangeError,
      );
      await expectLater(
        b.writeRawBytes(Uint8List(8), dstByteOffset: 12),
        throwsRangeError,
      );
      await b.writeRawBytes(one, dstByteOffset: 12);
      final out = Float32List(1);
      await b.read(out, 1, elementOffset: 3);
      expect(out.first, 1);
    });
  });

  group('copy-out', () {
    test('ByteData receives the read at its start', () async {
      final b = buffer(16);
      await b.write(Float32List.fromList([1, 2, 3, 4]), 4);
      final out = ByteData(8);
      await b.read(out, 2, elementOffset: 2);
      expect(out.getFloat32(0, Endian.host), 3);
      expect(out.getFloat32(4, Endian.host), 4);
    });

    test('a list of another element type receives the raw bytes', () async {
      final b = buffer(8, BufferDataType.uint32);
      final words = Uint32List.fromList([0x04030201, 0x08070605]);
      await b.write(words, 2, dataType: BufferDataType.uint32);
      final out = Uint8List(8);
      await b.read(out, 2, dataType: BufferDataType.uint32);
      expect(out, words.buffer.asUint8List());
    });

    test('a view is filled from its own offset', () async {
      final b = buffer(8);
      await b.write(Float32List.fromList([1, 2]), 2);
      final backing = Float32List(6);
      await b.read(Float32List.sublistView(backing, 2, 4), 2);
      expect(backing, [0, 0, 1, 2, 0, 0]);
    });
  });

  group('destroy', () {
    test('use after destroy throws StateError', () async {
      final b = platform.createBuffer(16, BufferDataType.float32)..destroy();
      await expectLater(b.read(Float32List(4), 4), throwsStateError);
      await expectLater(b.write(Float32List(4), 4), throwsStateError);
      await expectLater(b.writeRawBytes(Uint8List(4)), throwsStateError);
    });

    test('a read pending at destroy still completes with its data', () async {
      // The read owns the pooled scratch until its completion arrives, and the
      // native delete queues behind it. Repeated so a freed-scratch race shows.
      const n = 16 * 1024;
      final values = Float32List(n);
      for (var i = 0; i < n; i++) {
        values[i] = i.toDouble();
      }
      for (var round = 0; round < 20; round++) {
        final b = platform.createBuffer(n * 4, BufferDataType.float32);
        await b.write(values, n);
        final out = Float32List(n);
        final pending = b.read(out, n);
        b.destroy();
        await pending;
        expect(out.first, 0);
        expect(out.last, n - 1);
        await expectLater(b.read(out, n), throwsStateError);
      }
      expect(GpuCompletion.pendingCount, 0);
    });
  });
}
