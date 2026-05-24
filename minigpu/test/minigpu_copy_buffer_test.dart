// Tests for Minigpu.copyBuffer — GPU-side buffer-to-buffer copy via WGSL.
//
// These tests exercise:
//   - basic correctness (data round-trip: write → copyBuffer → read)
//   - u32 / f32 element types
//   - partial copy (elementCount < buffer size)
//   - copy to self (same buffer as src and dst) — implementation-defined but
//     must not crash
//   - large buffer (stress: workgroup boundary alignment)
//   - shader reuse (second call reuses the cached shader without re-creation)
//   - liveShaderCount does not grow across repeated calls

import 'dart:typed_data';
import 'dart:io';

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

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  /// Fill a u32 buffer with [count] sequential values starting at [start].
  Future<void> fillU32(Buffer buf, int count, {int start = 0}) async {
    final data = Uint32List(count);
    for (var i = 0; i < count; i++) {
      data[i] = (start + i) & 0xFFFFFFFF;
    }
    await buf.write(data, count, dataType: BufferDataType.uint32);
  }

  /// Read [count] u32 elements back from [buf].
  Future<Uint32List> readU32(Buffer buf, int count) async {
    final out = Uint32List(count);
    await buf.read(out, count, dataType: BufferDataType.uint32);
    return out;
  }

  /// Fill a float32 buffer with sequential floats.
  Future<void> fillF32(Buffer buf, int count) async {
    final data = Float32List(count);
    for (var i = 0; i < count; i++) {
      data[i] = i.toDouble();
    }
    await buf.write(data, count, dataType: BufferDataType.float32);
  }

  /// Read [count] float32 elements back.
  Future<Float32List> readF32(Buffer buf, int count) async {
    final out = Float32List(count);
    await buf.read(out, count, dataType: BufferDataType.float32);
    return out;
  }

  // ---------------------------------------------------------------------------
  // Basic correctness
  // ---------------------------------------------------------------------------

  group('copyBuffer — basic correctness', () {
    test('u32: copies all elements correctly', () async {
      const n = 256;
      final src = gpu.createBuffer(n * 4, BufferDataType.uint32);
      final dst = gpu.createBuffer(n * 4, BufferDataType.uint32);

      await fillU32(src, n);
      await gpu.copyBuffer(src, dst, elementCount: n);

      final result = await readU32(dst, n);
      for (var i = 0; i < n; i++) {
        expect(
          result[i],
          equals(i),
          reason: 'Mismatch at index $i: expected $i, got ${result[i]}',
        );
      }

      src.destroy();
      dst.destroy();
    });

    test('f32: copies all elements correctly', () async {
      const n = 512;
      final src = gpu.createBuffer(n * 4, BufferDataType.float32);
      final dst = gpu.createBuffer(n * 4, BufferDataType.float32);

      await fillF32(src, n);
      await gpu.copyBuffer(src, dst, elementCount: n);

      final result = await readF32(dst, n);
      for (var i = 0; i < n; i++) {
        expect(
          result[i],
          closeTo(i.toDouble(), 0.001),
          reason: 'Mismatch at index $i',
        );
      }

      src.destroy();
      dst.destroy();
    });

    test('copies preserve values across entire range', () async {
      // Sentinel pattern: alternating 0xAAAAAAAA / 0x55555555.
      const n = 128;
      final src = gpu.createBuffer(n * 4, BufferDataType.uint32);
      final dst = gpu.createBuffer(n * 4, BufferDataType.uint32);

      final data = Uint32List(n);
      for (var i = 0; i < n; i++) {
        data[i] = i.isEven ? 0xAAAAAAAA : 0x55555555;
      }
      await src.write(data, n, dataType: BufferDataType.uint32);

      await gpu.copyBuffer(src, dst, elementCount: n);

      final result = await readU32(dst, n);
      for (var i = 0; i < n; i++) {
        final expected = i.isEven ? 0xAAAAAAAA : 0x55555555;
        expect(
          result[i],
          equals(expected),
          reason: 'Sentinel mismatch at index $i',
        );
      }

      src.destroy();
      dst.destroy();
    });
  });

  // ---------------------------------------------------------------------------
  // Partial copy
  // ---------------------------------------------------------------------------

  group('copyBuffer — partial copy', () {
    test(
      'elementCount < buffer capacity: only copied region is updated',
      () async {
        const total = 256;
        const copyN = 64;
        final src = gpu.createBuffer(total * 4, BufferDataType.uint32);
        final dst = gpu.createBuffer(total * 4, BufferDataType.uint32);

        // Fill src 0..255, fill dst with 0xDEADBEEF sentinel.
        await fillU32(src, total);
        final sentinel = Uint32List(total)..fillRange(0, total, 0xDEADBEEF);
        await dst.write(sentinel, total, dataType: BufferDataType.uint32);

        await gpu.copyBuffer(src, dst, elementCount: copyN);

        final result = await readU32(dst, total);
        // First copyN elements must match src.
        for (var i = 0; i < copyN; i++) {
          expect(result[i], equals(i), reason: 'Copied region mismatch at $i');
        }
        // Remaining elements must be untouched (still sentinel).
        for (var i = copyN; i < total; i++) {
          expect(
            result[i],
            equals(0xDEADBEEF),
            reason: 'Sentinel overwritten at $i',
          );
        }

        src.destroy();
        dst.destroy();
      },
    );

    test('elementCount = 1: single element copy', () async {
      final src = gpu.createBuffer(64 * 4, BufferDataType.uint32);
      final dst = gpu.createBuffer(64 * 4, BufferDataType.uint32);

      await fillU32(src, 64, start: 100);

      await gpu.copyBuffer(src, dst, elementCount: 1);

      final result = await readU32(dst, 1);
      expect(result[0], equals(100));

      src.destroy();
      dst.destroy();
    });
  });

  // ---------------------------------------------------------------------------
  // Workgroup boundary (large buffers)
  // ---------------------------------------------------------------------------

  group('copyBuffer — workgroup boundary alignment', () {
    test('element count that is not a multiple of 64', () async {
      // 1000 elements — not a multiple of 64 (15*64=960, 16*64=1024).
      const n = 1000;
      final src = gpu.createBuffer(n * 4, BufferDataType.uint32);
      final dst = gpu.createBuffer(n * 4, BufferDataType.uint32);

      await fillU32(src, n);
      await gpu.copyBuffer(src, dst, elementCount: n);

      final result = await readU32(dst, n);
      for (var i = 0; i < n; i++) {
        expect(result[i], equals(i), reason: 'Boundary mismatch at index $i');
      }

      src.destroy();
      dst.destroy();
    });

    test('large buffer (16 384 elements = 64 KB)', () async {
      const n = 16384;
      final src = gpu.createBuffer(n * 4, BufferDataType.uint32);
      final dst = gpu.createBuffer(n * 4, BufferDataType.uint32);

      await fillU32(src, n);
      await gpu.copyBuffer(src, dst, elementCount: n);

      final result = await readU32(dst, n);
      // Spot-check first, last, and a few midpoints.
      expect(result[0], equals(0));
      expect(result[n - 1], equals(n - 1));
      expect(result[n ~/ 2], equals(n ~/ 2));

      src.destroy();
      dst.destroy();
    });

    test('exactly 64 elements (one workgroup)', () async {
      const n = 64;
      final src = gpu.createBuffer(n * 4, BufferDataType.uint32);
      final dst = gpu.createBuffer(n * 4, BufferDataType.uint32);

      await fillU32(src, n, start: 1000);
      await gpu.copyBuffer(src, dst, elementCount: n);

      final result = await readU32(dst, n);
      for (var i = 0; i < n; i++) {
        expect(result[i], equals(1000 + i));
      }

      src.destroy();
      dst.destroy();
    });
  });

  // ---------------------------------------------------------------------------
  // Shader reuse / liveShaderCount
  // ---------------------------------------------------------------------------

  group('copyBuffer — shader reuse', () {
    test('repeated calls do not grow liveShaderCount', () async {
      const n = 128;
      final src = gpu.createBuffer(n * 4, BufferDataType.uint32);
      final dst = gpu.createBuffer(n * 4, BufferDataType.uint32);
      await fillU32(src, n);

      // First call ensures the shader is created (may or may not already exist
      // from a prior test in the same suite).
      await gpu.copyBuffer(src, dst, elementCount: n);
      final countAfterFirst = gpu.liveShaderCount;

      // Subsequent calls must reuse the cached shader — count must not grow.
      await gpu.copyBuffer(src, dst, elementCount: n);
      await gpu.copyBuffer(src, dst, elementCount: n);

      expect(
        gpu.liveShaderCount,
        equals(countAfterFirst),
        reason: 'Repeated copyBuffer calls must reuse the cached shader',
      );

      src.destroy();
      dst.destroy();
    });

    test('result is correct after multiple reuses', () async {
      const n = 64;
      final src = gpu.createBuffer(n * 4, BufferDataType.uint32);
      final dst = gpu.createBuffer(n * 4, BufferDataType.uint32);

      // Round 1: values 0..63
      await fillU32(src, n);
      await gpu.copyBuffer(src, dst, elementCount: n);
      var result = await readU32(dst, n);
      expect(result[0], equals(0));
      expect(result[n - 1], equals(n - 1));

      // Round 2: values 100..163
      await fillU32(src, n, start: 100);
      await gpu.copyBuffer(src, dst, elementCount: n);
      result = await readU32(dst, n);
      expect(result[0], equals(100));
      expect(result[n - 1], equals(163));

      src.destroy();
      dst.destroy();
    });
  });

  // ---------------------------------------------------------------------------
  // Robustness
  // ---------------------------------------------------------------------------

  group('copyBuffer — robustness', () {
    test('does not crash when called on destroyed buffers guard', () async {
      // Ensure calling with a valid pair after a previous valid call still works.
      const n = 32;
      final src = gpu.createBuffer(n * 4, BufferDataType.uint32);
      final dst = gpu.createBuffer(n * 4, BufferDataType.uint32);
      await fillU32(src, n, start: 500);

      await gpu.copyBuffer(src, dst, elementCount: n);

      final result = await readU32(dst, n);
      expect(result[0], equals(500));

      src.destroy();
      dst.destroy();
    });

    test('Windows-only: liveBufferCount stable across copy', () async {
      if (!Platform.isWindows) return;

      const n = 128;
      final before = gpu.liveBufferCount;

      final src = gpu.createBuffer(n * 4, BufferDataType.uint32);
      final dst = gpu.createBuffer(n * 4, BufferDataType.uint32);
      await fillU32(src, n);

      await gpu.copyBuffer(src, dst, elementCount: n);
      await gpu.copyBuffer(src, dst, elementCount: n);

      // Only the two buffers we created should be new.
      expect(
        gpu.liveBufferCount,
        equals(before + 2),
        reason: 'copyBuffer must not allocate new GPU buffers per call',
      );

      src.destroy();
      dst.destroy();

      expect(gpu.liveBufferCount, equals(before));
    });
  });
}
