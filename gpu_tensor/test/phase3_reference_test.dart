import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';

/// CPU references for the Phase 3 kernels: tiled matMul (incl. non-tile-
/// multiple dims + batched), GEMV fast path, workgroup softmax (d >= 64),
/// and broadcasting elementwise ops.

Float32List seeded(int n, int seed) {
  final rng = math.Random(seed);
  final out = Float32List(n);
  for (int i = 0; i < n; i++) {
    out[i] = rng.nextDouble() * 2 - 1;
  }
  return out;
}

Float32List cpuMatMul(Float32List a, Float32List b, int m, int n, int p) {
  final c = Float32List(m * p);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      double sum = 0;
      for (int k = 0; k < n; k++) {
        sum += a[i * n + k] * b[k * p + j];
      }
      c[i * p + j] = sum;
    }
  }
  return c;
}

void expectClose(Float32List actual, Float32List expected, double tol) {
  expect(actual.length, equals(expected.length));
  for (int i = 0; i < expected.length; i++) {
    expect(actual[i], closeTo(expected[i], tol),
        reason: 'mismatch at index $i');
  }
}

Future<void> main() async {
  group('tiled matMul vs CPU', () {
    test('non-tile-multiple dims (33x47 @ 47x29)', () async {
      const m = 33, n = 47, p = 29;
      final aData = seeded(m * n, 1);
      final bData = seeded(n * p, 2);
      final a = await Tensor.create([m, n], data: aData);
      final b = await Tensor.create([n, p], data: bData);
      final c = await a.matMul(b);
      expect(c.shape, equals([m, p]));
      final actual = await c.getData() as Float32List;
      expectClose(actual, cpuMatMul(aData, bData, m, n, p), 1e-3);
    });

    test('exact tile multiple (32x32 @ 32x32)', () async {
      const m = 32, n = 32, p = 32;
      final aData = seeded(m * n, 3);
      final bData = seeded(n * p, 4);
      final a = await Tensor.create([m, n], data: aData);
      final b = await Tensor.create([n, p], data: bData);
      final c = await a.matMul(b);
      final actual = await c.getData() as Float32List;
      expectClose(actual, cpuMatMul(aData, bData, m, n, p), 1e-3);
    });

    test('GEMV fast path (1x50 @ 50x37)', () async {
      const m = 1, n = 50, p = 37;
      final aData = seeded(n, 5);
      final bData = seeded(n * p, 6);
      final a = await Tensor.create([m, n], data: aData);
      final b = await Tensor.create([n, p], data: bData);
      final c = await a.matMul(b);
      expect(c.shape, equals([1, p]));
      final actual = await c.getData() as Float32List;
      expectClose(actual, cpuMatMul(aData, bData, m, n, p), 1e-3);
    });

    test('batched (3 batches of 17x21 @ 21x13)', () async {
      const batch = 3, m = 17, n = 21, p = 13;
      final aData = seeded(batch * m * n, 7);
      final bData = seeded(batch * n * p, 8);
      final a = await Tensor.create([batch, m, n], data: aData);
      final b = await Tensor.create([batch, n, p], data: bData);
      final c = await a.matMul(b);
      expect(c.shape, equals([batch, m, p]));
      final actual = await c.getData() as Float32List;
      final expected = Float32List(batch * m * p);
      for (int bi = 0; bi < batch; bi++) {
        final cb = cpuMatMul(
          Float32List.sublistView(aData, bi * m * n, (bi + 1) * m * n),
          Float32List.sublistView(bData, bi * n * p, (bi + 1) * n * p),
          m,
          n,
          p,
        );
        expected.setRange(bi * m * p, (bi + 1) * m * p, cb);
      }
      expectClose(actual, expected, 1e-3);
    });
  });

  group('softmax', () {
    Float32List cpuSoftmax(Float32List x, int rows, int d) {
      final out = Float32List(rows * d);
      for (int r = 0; r < rows; r++) {
        double mx = -double.maxFinite;
        for (int j = 0; j < d; j++) {
          mx = math.max(mx, x[r * d + j]);
        }
        double sum = 0;
        for (int j = 0; j < d; j++) {
          sum += math.exp(x[r * d + j] - mx);
        }
        for (int j = 0; j < d; j++) {
          out[r * d + j] = math.exp(x[r * d + j] - mx) / sum;
        }
      }
      return out;
    }

    test('large row (workgroup kernel, d=300 not multiple of 256)', () async {
      const rows = 5, d = 300;
      final data = seeded(rows * d, 9);
      final t = await Tensor.create([rows, d], data: data);
      final s = await t.softmax();
      final actual = await s.getData() as Float32List;
      expectClose(actual, cpuSoftmax(data, rows, d), 1e-4);
    });

    test('small row (per-element kernel, d=8) still correct', () async {
      const rows = 4, d = 8;
      final data = seeded(rows * d, 10);
      final t = await Tensor.create([rows, d], data: data);
      final s = await t.softmax();
      final actual = await s.getData() as Float32List;
      expectClose(actual, cpuSoftmax(data, rows, d), 1e-4);
    });

    test('boundary d=64 routes to workgroup kernel', () async {
      const rows = 3, d = 64;
      final data = seeded(rows * d, 11);
      final t = await Tensor.create([rows, d], data: data);
      final s = await t.softmax();
      final actual = await s.getData() as Float32List;
      expectClose(actual, cpuSoftmax(data, rows, d), 1e-4);
    });
  });

  group('broadcasting elementwise', () {
    test('[2,3] + [3] (bias-add pattern)', () async {
      final a = await Tensor.create([2, 3],
          data: Float32List.fromList([1, 2, 3, 4, 5, 6]));
      final b =
          await Tensor.create([3], data: Float32List.fromList([10, 20, 30]));
      final c = await a.add(b);
      expect(c.shape, equals([2, 3]));
      final actual = await c.getData();
      expect(actual, equals([11, 22, 33, 14, 25, 36]));
    });

    test('[2,1] * [1,3] outer product', () async {
      final a =
          await Tensor.create([2, 1], data: Float32List.fromList([2, 3]));
      final b =
          await Tensor.create([1, 3], data: Float32List.fromList([4, 5, 6]));
      final c = await a.multiply(b);
      expect(c.shape, equals([2, 3]));
      final actual = await c.getData();
      expect(actual, equals([8, 10, 12, 12, 15, 18]));
    });

    test('[2,2,2] - [2] trailing broadcast', () async {
      final a = await Tensor.create([2, 2, 2],
          data: Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8]));
      final b = await Tensor.create([2], data: Float32List.fromList([1, 2]));
      final c = await a.subtract(b);
      expect(c.shape, equals([2, 2, 2]));
      final actual = await c.getData();
      expect(actual, equals([0, 0, 2, 2, 4, 4, 6, 6]));
    });

    test('[4] / [1] scalar-shaped divisor', () async {
      final a =
          await Tensor.create([4], data: Float32List.fromList([2, 4, 6, 8]));
      final b = await Tensor.create([1], data: Float32List.fromList([2]));
      final c = await a.divide(b);
      expect(c.shape, equals([4]));
      final actual = await c.getData();
      expect(actual, equals([1, 2, 3, 4]));
    });

    test('same-shape fast path unaffected', () async {
      final a =
          await Tensor.create([2, 2], data: Float32List.fromList([1, 2, 3, 4]));
      final b = await Tensor.create([2, 2],
          data: Float32List.fromList([10, 20, 30, 40]));
      final c = await a.add(b);
      final actual = await c.getData();
      expect(actual, equals([11, 22, 33, 44]));
    });

    test('incompatible shapes throw', () async {
      final a = await Tensor.create([2, 3],
          data: Float32List.fromList([1, 2, 3, 4, 5, 6]));
      final b = await Tensor.create([4],
          data: Float32List.fromList([1, 2, 3, 4]));
      expect(() => a.add(b), throwsA(isA<Exception>()));
    });
  });
}
