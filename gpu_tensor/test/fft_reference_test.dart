import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

/// CPU reference DFTs. Interleaved complex layout: [re0, im0, re1, im1, ...].
/// Existing FFT tests only use delta-at-origin / DC inputs, which are
/// invariant under bit-reversal permutation — they cannot catch ordering
/// bugs. These tests use asymmetric inputs checked against a naive DFT.

Float32List dft1d(Float32List x) {
  final n = x.length ~/ 2;
  final out = Float32List(n * 2);
  for (int k = 0; k < n; k++) {
    double re = 0, im = 0;
    for (int t = 0; t < n; t++) {
      final angle = -2 * math.pi * k * t / n;
      final c = math.cos(angle), s = math.sin(angle);
      final xr = x[t * 2], xi = x[t * 2 + 1];
      re += xr * c - xi * s;
      im += xr * s + xi * c;
    }
    out[k * 2] = re;
    out[k * 2 + 1] = im;
  }
  return out;
}

Float32List dft2d(Float32List x, int rows, int cols) {
  final out = Float32List(rows * cols * 2);
  for (int k1 = 0; k1 < rows; k1++) {
    for (int k2 = 0; k2 < cols; k2++) {
      double re = 0, im = 0;
      for (int n1 = 0; n1 < rows; n1++) {
        for (int n2 = 0; n2 < cols; n2++) {
          final angle = -2 * math.pi * (k1 * n1 / rows + k2 * n2 / cols);
          final c = math.cos(angle), s = math.sin(angle);
          final idx = (n1 * cols + n2) * 2;
          final xr = x[idx], xi = x[idx + 1];
          re += xr * c - xi * s;
          im += xr * s + xi * c;
        }
      }
      final o = (k1 * cols + k2) * 2;
      out[o] = re;
      out[o + 1] = im;
    }
  }
  return out;
}

Float32List dft3d(Float32List x, int d, int r, int c) {
  final out = Float32List(d * r * c * 2);
  for (int k0 = 0; k0 < d; k0++) {
    for (int k1 = 0; k1 < r; k1++) {
      for (int k2 = 0; k2 < c; k2++) {
        double re = 0, im = 0;
        for (int n0 = 0; n0 < d; n0++) {
          for (int n1 = 0; n1 < r; n1++) {
            for (int n2 = 0; n2 < c; n2++) {
              final angle = -2 *
                  math.pi *
                  (k0 * n0 / d + k1 * n1 / r + k2 * n2 / c);
              final cc = math.cos(angle), ss = math.sin(angle);
              final idx = ((n0 * r + n1) * c + n2) * 2;
              final xr = x[idx], xi = x[idx + 1];
              re += xr * cc - xi * ss;
              im += xr * ss + xi * cc;
            }
          }
        }
        final o = ((k0 * r + k1) * c + k2) * 2;
        out[o] = re;
        out[o + 1] = im;
      }
    }
  }
  return out;
}

void expectClose(Float32List actual, Float32List expected, double tol) {
  expect(actual.length, equals(expected.length));
  for (int i = 0; i < expected.length; i++) {
    expect(actual[i], closeTo(expected[i], tol),
        reason: 'mismatch at index $i');
  }
}

Float32List seededComplex(int complexCount, int seed) {
  final rng = math.Random(seed);
  final data = Float32List(complexCount * 2);
  for (int i = 0; i < data.length; i++) {
    data[i] = rng.nextDouble() * 2 - 1;
  }
  return data;
}

Future<void> main() async {
  group('FFT vs CPU reference DFT', () {
    test('fft1d matches DFT on random complex input (control)', () async {
      const n = 16;
      final data = seededComplex(n, 42);
      final tensor = await Tensor.create([n * 2], data: data);
      final result = await tensor.fft1d();
      final actual = await result.getData() as Float32List;
      expectClose(actual, dft1d(data), 1e-3);
    });

    test('fft2d matches DFT on shifted delta', () async {
      // Delta at (1, 2): DFT is a complex exponential — NOT invariant under
      // bit-reversal, unlike the delta-at-origin used in existing tests.
      const rows = 4, cols = 4;
      final data = Float32List(rows * cols * 2);
      data[(1 * cols + 2) * 2] = 1.0;
      final tensor = await Tensor.create([rows, cols, 2], data: data);
      final result = await tensor.fft2d();
      final actual = await result.getData() as Float32List;
      expectClose(actual, dft2d(data, rows, cols), 1e-3);
    });

    test('fft2d matches DFT on random complex input', () async {
      const rows = 8, cols = 8;
      final data = seededComplex(rows * cols, 7);
      final tensor = await Tensor.create([rows, cols, 2], data: data);
      final result = await tensor.fft2d();
      final actual = await result.getData() as Float32List;
      expectClose(actual, dft2d(data, rows, cols), 1e-2);
    });

    test('fft3d matches DFT on random complex input', () async {
      const d = 4, r = 4, c = 4;
      final data = seededComplex(d * r * c, 13);
      final tensor = await Tensor.create([d, r, c, 2], data: data);
      final result = await tensor.fft3d();
      final actual = await result.getData() as Float32List;
      expectClose(actual, dft3d(data, d, r, c), 1e-2);
    });

    test('fft2d does not mutate its complex input tensor', () async {
      const rows = 8, cols = 8;
      final data = seededComplex(rows * cols, 21);
      final tensor = await Tensor.create([rows, cols, 2], data: data);
      await tensor.fft2d();
      final after = await tensor.getData() as Float32List;
      expectClose(after, data, 0.0);
    });

    test('fft2d accepts a real [rows, cols] tensor', () async {
      const rows = 4, cols = 4;
      final rng = math.Random(3);
      final real = Float32List(rows * cols);
      for (int i = 0; i < real.length; i++) {
        real[i] = rng.nextDouble() * 2 - 1;
      }
      final complex = Float32List(rows * cols * 2);
      for (int i = 0; i < rows * cols; i++) {
        complex[i * 2] = real[i];
      }
      final tensor = await Tensor.create([rows, cols], data: real);
      final result = await tensor.fft2d();
      final actual = await result.getData() as Float32List;
      expectClose(actual, dft2d(complex, rows, cols), 1e-3);
    });
  });

  group('fromBytes', () {
    test('float32 round-trip through toBytes/fromBytes', () async {
      final data = Float32List.fromList([1.5, -2.25, 3.75, 0.125, 9.0, -0.5]);
      final tensor = await Tensor.create([2, 3], data: data);
      final bytes = await tensor.toBytes();
      final restored = await Tensor.fromBytes<Float32List>(
        bytes,
        dataType: BufferDataType.float32,
      );
      expect(restored.shape, equals([2, 3]));
      final actual = await restored.getData();
      expect(actual, equals(data));
    });
  });
}
