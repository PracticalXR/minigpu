import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml.dart';
import 'package:test/test.dart';

Float32List seeded(int n, int seed) {
  final rng = math.Random(seed);
  final out = Float32List(n);
  for (int i = 0; i < n; i++) {
    out[i] = rng.nextDouble() * 2 - 1;
  }
  return out;
}

void expectClose(Float32List actual, List<double> expected, double tol) {
  expect(actual.length, equals(expected.length));
  for (int i = 0; i < expected.length; i++) {
    expect(actual[i], closeTo(expected[i], tol),
        reason: 'mismatch at index $i');
  }
}

List<double> cpuRmsNorm(
  Float32List x,
  Float32List w,
  int rows,
  int d,
  double eps,
) {
  final out = List<double>.filled(rows * d, 0);
  for (int r = 0; r < rows; r++) {
    double ss = 0;
    for (int j = 0; j < d; j++) {
      ss += x[r * d + j] * x[r * d + j];
    }
    final inv = 1.0 / math.sqrt(ss / d + eps);
    for (int j = 0; j < d; j++) {
      out[r * d + j] = x[r * d + j] * inv * w[j];
    }
  }
  return out;
}

List<double> cpuRope(
  Float32List x,
  int tokens,
  int heads,
  int headDim,
  int posOffset,
  double base,
) {
  final out = List<double>.filled(x.length, 0);
  final half = headDim ~/ 2;
  for (int t = 0; t < tokens; t++) {
    for (int h = 0; h < heads; h++) {
      final rowBase = (t * heads + h) * headDim;
      for (int i = 0; i < half; i++) {
        final angle =
            (posOffset + t) * math.pow(base, -2.0 * i / headDim).toDouble();
        final c = math.cos(angle), s = math.sin(angle);
        final x0 = x[rowBase + 2 * i], x1 = x[rowBase + 2 * i + 1];
        out[rowBase + 2 * i] = x0 * c - x1 * s;
        out[rowBase + 2 * i + 1] = x0 * s + x1 * c;
      }
    }
  }
  return out;
}

Future<void> main() async {
  group('rmsNorm', () {
    test('matches CPU reference (rows=3, d=64)', () async {
      const rows = 3, d = 64;
      final xData = seeded(rows * d, 1);
      final wData = seeded(d, 2);
      final x = await Tensor.create([rows, d], data: xData);
      final w = await Tensor.create([d], data: wData);
      final y = await x.rmsNorm(w, eps: 1e-5);
      final actual = await y.getData() as Float32List;
      expectClose(actual, cpuRmsNorm(xData, wData, rows, d, 1e-5), 1e-4);
    });

    test('d not a multiple of workgroup width (d=100)', () async {
      const rows = 2, d = 100;
      final xData = seeded(rows * d, 3);
      final wData = seeded(d, 4);
      final x = await Tensor.create([rows, d], data: xData);
      final w = await Tensor.create([d], data: wData);
      final y = await x.rmsNorm(w);
      final actual = await y.getData() as Float32List;
      expectClose(actual, cpuRmsNorm(xData, wData, rows, d, 1e-5), 1e-4);
    });
  });

  group('rope', () {
    test('matches CPU reference (3 tokens, 2 heads, headDim 8)', () async {
      const tokens = 3, heads = 2, headDim = 8;
      final xData = seeded(tokens * heads * headDim, 5);
      final x = await Tensor.create([tokens, heads, headDim], data: xData);
      final y = await x.rope(
        headDim: headDim,
        heads: heads,
        positionOffset: 7,
      );
      final actual = await y.getData() as Float32List;
      expectClose(
        actual,
        cpuRope(xData, tokens, heads, headDim, 7, 10000.0),
        1e-4,
      );
    });

    test('position 0 is identity', () async {
      const heads = 2, headDim = 8;
      final xData = seeded(heads * headDim, 6);
      final x = await Tensor.create([heads, headDim], data: xData);
      final y = await x.rope(headDim: headDim, heads: heads, positionOffset: 0);
      final actual = await y.getData() as Float32List;
      expectClose(actual, xData.map((v) => v.toDouble()).toList(), 1e-6);
    });
  });

  group('silu / gelu', () {
    test('silu matches CPU', () async {
      final xData = seeded(300, 7);
      final x = await Tensor.create([300], data: xData);
      final y = await x.silu();
      final actual = await y.getData() as Float32List;
      final expected = xData
          .map((v) => v / (1.0 + math.exp(-v)))
          .toList(growable: false);
      expectClose(actual, expected, 1e-5);
    });

    test('gelu matches CPU tanh approximation', () async {
      final xData = seeded(300, 8);
      final x = await Tensor.create([300], data: xData);
      final y = await x.gelu();
      final actual = await y.getData() as Float32List;
      double tanh(double v) {
        final e2 = math.exp(2 * v);
        return (e2 - 1) / (e2 + 1);
      }

      final expected = xData
          .map((v) => 0.5 *
              v *
              (1.0 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))))
          .toList(growable: false);
      expectClose(actual, expected, 1e-5);
    });
  });
}
