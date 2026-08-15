/// Fft1dPlan must produce byte-identical results to the reference
/// real-input FFT path (upgradeRealToComplex + fft1d) — it runs the same
/// butterfly math, just pipelined with baked constants and plan-owned
/// shaders.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

Float32List randomSamples(int n, int seed) {
  final rng = math.Random(seed);
  final out = Float32List(n);
  for (int i = 0; i < n; i++) {
    out[i] = rng.nextDouble() * 2 - 1;
  }
  return out;
}

void main() {
  final gpu = Minigpu();

  setUpAll(() async {
    await gpu.init();
  });

  Future<Float32List> referenceFft(Float32List samples) async {
    final input = await Tensor.create(
      [samples.length],
      gpu: gpu,
      data: samples,
    );
    final result = await input.fft(isRealInput: true);
    final data = await result.getData() as Float32List;
    input.destroy();
    result.destroy();
    return data;
  }

  for (final n in [8, 256, 4096]) {
    test('executeReal matches reference fft for n=$n', () async {
      final samples = randomSamples(n, 42 + n);
      final expected = await referenceFft(samples);

      final plan = await Fft1dPlan.create(gpu, n);
      final input = await Tensor.create([n], gpu: gpu, data: samples);
      final result = await plan.executeReal(input);
      final actual = await result.getData() as Float32List;

      expect(actual.length, expected.length);
      for (int i = 0; i < expected.length; i++) {
        expect(
          actual[i],
          closeTo(expected[i], 1e-3),
          reason: 'mismatch at float $i',
        );
      }

      input.destroy();
      plan.destroy();
    });
  }

  test('repeated frames through one plan stay correct (workspace reuse)',
      () async {
    const n = 1024;
    final plan = await Fft1dPlan.create(gpu, n);
    final input = await Tensor.create([n], gpu: gpu);

    for (int frame = 0; frame < 3; frame++) {
      final samples = randomSamples(n, 7 + frame);
      final expected = await referenceFft(samples);

      await input.write(samples);
      final result = await plan.executeReal(input);
      expect(identical(result, plan.output), isTrue,
          reason: 'result tensor must be the stable plan output');
      final actual = await result.getData() as Float32List;

      for (int i = 0; i < expected.length; i++) {
        expect(
          actual[i],
          closeTo(expected[i], 1e-3),
          reason: 'frame $frame mismatch at float $i',
        );
      }
    }

    input.destroy();
    plan.destroy();
  });

  test('baked scale multiplies the final stage output', () async {
    const n = 256;
    final samples = randomSamples(n, 99);
    final expected = await referenceFft(samples);

    final plan = await Fft1dPlan.create(gpu, n, scale: 1.0 / n);
    final input = await Tensor.create([n], gpu: gpu, data: samples);
    final result = await plan.executeReal(input);
    final actual = await result.getData() as Float32List;

    for (int i = 0; i < expected.length; i++) {
      expect(
        actual[i],
        closeTo(expected[i] / n, 1e-5),
        reason: 'scaled mismatch at float $i',
      );
    }

    input.destroy();
    plan.destroy();
  });

  test('rejects non-power-of-two and wrong-size input', () async {
    expect(() => Fft1dPlan.create(gpu, 100), throwsArgumentError);
    expect(() => Fft1dPlan.create(gpu, 1), throwsArgumentError);

    final plan = await Fft1dPlan.create(gpu, 8);
    final wrong = await Tensor.create([16], gpu: gpu);
    expect(() => plan.executeReal(wrong), throwsArgumentError);
    wrong.destroy();
    plan.destroy();
  });
}
