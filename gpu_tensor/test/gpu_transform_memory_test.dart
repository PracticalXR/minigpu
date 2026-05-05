/// VRAM / RSS leak isolation tests for the gpu_tensor FFT layer.
///
/// Run with:
///   cd minigpu/gpu_tensor && flutter test test/gpu_transform_memory_test.dart -v
///
/// These tests are deliberately long-running so that even a small per-call
/// allocation accumulates into a measurable signal.  Each group prints
/// per-interval RSS snapshots so you can see the trend even when the
/// assertion passes.
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

String _mb(int bytes) => '${(bytes / 1024 / 1024).toStringAsFixed(2)} MB';
String _sign(double v) => v >= 0 ? '+' : '';

/// Returns (rssGrowthMB, vramGrowthMB) after [iterations] of [body].
/// Warms up [warmup] calls first, records RSS + VRAM every [interval] calls.
/// Pass [gpu] to enable VRAM tracking; omit (null) to skip it.
Future<(double rss, double vram)> _measureGrowth(
  Future<void> Function() body, {
  required int iterations,
  required Minigpu gpu,
  int warmup = 20,
  int interval = 100,
}) async {
  for (int i = 0; i < warmup; i++) {
    await body();
  }
  await Future.delayed(const Duration(milliseconds: 100));

  final int baselineRss = ProcessInfo.currentRss;
  final int baselineVram = gpu.queryVramBytes();
  final bool hasVram = baselineVram >= 0;
  print(
    '  Baseline — RSS: ${_mb(baselineRss)}'
    '${hasVram ? '  VRAM: ${_mb(baselineVram)}' : '  VRAM: n/a'}',
  );

  for (int i = 0; i < iterations; i++) {
    await body();
    if (i > 0 && i % interval == 0) {
      final rss = ProcessInfo.currentRss;
      final rssG = (rss - baselineRss) / 1024 / 1024;
      final vram = gpu.queryVramBytes();
      final vramG = hasVram ? (vram - baselineVram) / 1024 / 1024 : 0.0;
      print(
        '  iter ${i.toString().padLeft(5)}: '
        'RSS ${_mb(rss)} (${_sign(rssG)}${rssG.toStringAsFixed(2)} MB)'
        '${hasVram ? '  VRAM ${_mb(vram)} (${_sign(vramG)}${vramG.toStringAsFixed(2)} MB)' : ''}',
      );
    }
  }

  await Future.delayed(const Duration(milliseconds: 100));
  final int finalRss = ProcessInfo.currentRss;
  final int finalVram = gpu.queryVramBytes();
  final double rssGrowth = (finalRss - baselineRss) / 1024 / 1024;
  final double vramGrowth = hasVram
      ? (finalVram - baselineVram) / 1024 / 1024
      : 0.0;
  print(
    '  FINAL: RSS ${_mb(finalRss)} (${_sign(rssGrowth)}${rssGrowth.toStringAsFixed(2)} MB)'
    '${hasVram ? '  VRAM ${_mb(finalVram)} (${_sign(vramGrowth)}${vramGrowth.toStringAsFixed(2)} MB)' : ''}',
  );
  return (rssGrowth, vramGrowth);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void main() {
  late Minigpu gpu;

  setUpAll(() async {
    gpu = Minigpu();
    await gpu.init();
  });

  // ---------------------------------------------------------------------------
  // Layer 1: upgradeRealToComplexInto — must be zero-allocation per call
  // ---------------------------------------------------------------------------
  group('upgradeRealToComplexInto — zero-allocation path', () {
    // Dawn/WebGPU accumulates ~10-15 MB of internal steady-state memory
    // (command-buffer pools, descriptor heap, etc.) as the command queue grows.
    // Growth plateaus after ~1000 calls, so 20 MB is the right budget.
    test('2000 calls should not grow RSS by more than 20 MB', () async {
      const int n = 2048;
      final inputTensor = await Tensor.create([n], gpu: gpu);
      final outputTensor = await Tensor.create([n * 2], gpu: gpu);

      final data = Float32List(n);
      for (int i = 0; i < n; i++) data[i] = math.sin(i * 0.01);
      await inputTensor.buffer.write(data, n, dataType: BufferDataType.float32);

      print('\n=== upgradeRealToComplexInto x2000 ===');
      final (rssGrowth, vramGrowth) = await _measureGrowth(
        () => inputTensor.upgradeRealToComplexInto(outputTensor),
        gpu: gpu,
        iterations: 2000,
        warmup: 20,
        interval: 200,
      );

      inputTensor.destroy();
      outputTensor.destroy();

      expect(
        rssGrowth,
        lessThan(20.0),
        reason:
            'upgradeRealToComplexInto should allocate nothing per call; '
            'got +${rssGrowth.toStringAsFixed(2)} MB RSS over 2000 calls',
      );
      expect(
        vramGrowth,
        lessThan(5.0),
        reason:
            'upgradeRealToComplexInto should allocate nothing per call; '
            'got +${vramGrowth.toStringAsFixed(2)} MB VRAM over 2000 calls',
      );
    });
  });

  // ---------------------------------------------------------------------------
  // Layer 2: bitReverseReorderInto — must be zero-allocation per call
  // ---------------------------------------------------------------------------
  group('bitReverseReorderInto — zero-allocation path', () {
    test('2000 calls should not grow RSS by more than 10 MB', () async {
      const int n = 2048;
      final srcTensor = await Tensor.create([n * 2], gpu: gpu);
      final dstTensor = await Tensor.create([n * 2], gpu: gpu);

      print('\n=== bitReverseReorderInto x2000 ===');
      final (rssGrowth, vramGrowth) = await _measureGrowth(
        () => srcTensor.bitReverseReorderInto(srcTensor, dstTensor),
        gpu: gpu,
        iterations: 2000,
        warmup: 20,
        interval: 200,
      );

      srcTensor.destroy();
      dstTensor.destroy();

      expect(
        rssGrowth,
        lessThan(10.0),
        reason:
            'bitReverseReorderInto should allocate nothing per call; '
            'got +${rssGrowth.toStringAsFixed(2)} MB RSS over 2000 calls',
      );
      expect(
        vramGrowth,
        lessThan(5.0),
        reason:
            'bitReverseReorderInto should allocate nothing per call; '
            'got +${vramGrowth.toStringAsFixed(2)} MB VRAM over 2000 calls',
      );
    });
  });

  // ---------------------------------------------------------------------------
  // Layer 3: fft1dPreallocated — entire FFT pipeline with zero allocations
  // ---------------------------------------------------------------------------
  group('fft1dPreallocated — zero-allocation FFT path', () {
    test(
      '1000 full FFT cycles should not grow RSS by more than 10 MB',
      () async {
        const int n = 2048; // real samples (matches live audio)
        final inputTensor = await Tensor.create([n], gpu: gpu);
        final complexTensor = await Tensor.create([n * 2], gpu: gpu);
        final bitRevTensor = await Tensor.create([n * 2], gpu: gpu);
        final pongTensor = await Tensor.create([n * 2], gpu: gpu);
        final paramBuffer = gpu.createBuffer(16, BufferDataType.uint32);

        // Precompute twiddle factors once.
        final twiddleFactors = Float32List(n * 2);
        for (int i = 0; i < n; i++) {
          final double angle = -2.0 * math.pi * i / n;
          twiddleFactors[i * 2] = math.cos(angle);
          twiddleFactors[i * 2 + 1] = math.sin(angle);
        }
        final twiddleBuffer = gpu.createBuffer(
          n * 2 * 4,
          BufferDataType.float32,
        );
        await twiddleBuffer.write(twiddleFactors, twiddleFactors.length);

        final data = Float32List(n);
        for (int i = 0; i < n; i++) data[i] = math.sin(i * 0.01);
        await inputTensor.buffer.write(
          data,
          n,
          dataType: BufferDataType.float32,
        );

        print('\n=== fft1dPreallocated x1000 (n=$n) ===');
        final (rssGrowth, vramGrowth) = await _measureGrowth(
          () async {
            await inputTensor.upgradeRealToComplexInto(complexTensor);
            await inputTensor.bitReverseReorderInto(
              complexTensor,
              bitRevTensor,
            );
            await bitRevTensor.fft1dPreallocated(
              pong: pongTensor,
              twiddleBuffer: twiddleBuffer,
              paramBuffer: paramBuffer,
            );
          },
          gpu: gpu,
          iterations: 1000,
          warmup: 10,
          interval: 100,
        );

        inputTensor.destroy();
        complexTensor.destroy();
        bitRevTensor.destroy();
        pongTensor.destroy();
        twiddleBuffer.destroy();
        paramBuffer.destroy();

        expect(
          rssGrowth,
          lessThan(10.0),
          reason:
              'fft1dPreallocated should allocate nothing per FFT; '
              'got +${rssGrowth.toStringAsFixed(2)} MB RSS over 1000 calls',
        );
        expect(
          vramGrowth,
          lessThan(5.0),
          reason:
              'fft1dPreallocated should allocate nothing per FFT; '
              'got +${vramGrowth.toStringAsFixed(2)} MB VRAM over 1000 calls',
        );
      },
    );
  });

  // ---------------------------------------------------------------------------
  // Layer 4: fft1d() allocating path — verifies cleanup is correct
  //
  // fft1d() allocates bitReversed + pong per call and destroys them in finally.
  // After N calls, RSS should stabilize (not grow unboundedly).
  // ---------------------------------------------------------------------------
  group('fft1d() — allocating path cleanup verification', () {
    // Each fft1d() call submits 1 (bit-reverse) + 11 (butterfly) = 12 GPU
    // dispatches.  Without a readback to force a GPU→CPU sync, Dawn's
    // internal command queue grows until it crashes (~200 calls × 12 = 2400
    // queued dispatches).  We call getData() every kFlushInterval calls to
    // drain the queue — this mirrors what the live app does on every frame.
    test(
      '200 fft1d() calls with periodic flush should not grow RSS by more than 30 MB',
      () async {
        const int n = 2048; // matches live audio sample count
        const int kFlushInterval =
            20; // readback every N calls to flush Dawn queue

        final inputTensor = await Tensor.create([n * 2], gpu: gpu);

        // Provide valid complex-format data (real=sin, imag=0).
        final data = Float32List(n * 2);
        for (int i = 0; i < n; i++) {
          data[i * 2] = math.sin(i * 0.01);
          data[i * 2 + 1] = 0.0;
        }
        await inputTensor.buffer.write(
          data,
          n * 2,
          dataType: BufferDataType.float32,
        );

        print(
          '\n=== fft1d() allocating path x200 (n=$n, flush every $kFlushInterval) ===',
        );

        // Warmup
        for (int i = 0; i < 5; i++) {
          final r = await inputTensor.fft1d();
          await r.getData(); // force flush
          r.destroy();
        }
        await Future.delayed(const Duration(milliseconds: 100));
        final int baselineRss = ProcessInfo.currentRss;
        final int baselineVram = gpu.queryVramBytes();
        final bool hasVram = baselineVram >= 0;
        print(
          '  Baseline — RSS: ${_mb(baselineRss)}'
          '${hasVram ? '  VRAM: ${_mb(baselineVram)}' : '  VRAM: n/a'}',
        );

        for (int i = 0; i < 200; i++) {
          final result = await inputTensor.fft1d();
          if (i % kFlushInterval == 0) {
            // Force GPU→CPU sync so Dawn flushes its command queue.
            await result.getData();
          }
          result.destroy();

          if (i > 0 && i % 40 == 0) {
            final rss = ProcessInfo.currentRss;
            final rssG = (rss - baselineRss) / 1024 / 1024;
            final vram = gpu.queryVramBytes();
            final vramG = hasVram ? (vram - baselineVram) / 1024 / 1024 : 0.0;
            print(
              '  iter ${i.toString().padLeft(4)}: RSS ${_mb(rss)} '
              '(${_sign(rssG)}${rssG.toStringAsFixed(2)} MB)'
              '${hasVram ? '  VRAM ${_mb(vram)} (${_sign(vramG)}${vramG.toStringAsFixed(2)} MB)' : ''}',
            );
          }
        }

        await Future.delayed(const Duration(milliseconds: 100));
        final int finalRss = ProcessInfo.currentRss;
        final int finalVram = gpu.queryVramBytes();
        final double rssGrowth = (finalRss - baselineRss) / 1024 / 1024;
        final double vramGrowth = hasVram
            ? (finalVram - baselineVram) / 1024 / 1024
            : 0.0;
        print(
          '  FINAL: RSS ${_mb(finalRss)} (${_sign(rssGrowth)}${rssGrowth.toStringAsFixed(2)} MB)'
          '${hasVram ? '  VRAM ${_mb(finalVram)} (${_sign(vramGrowth)}${vramGrowth.toStringAsFixed(2)} MB)' : ''}',
        );

        inputTensor.destroy();

        // fft1d allocates/destroys per call; some Dawn page retention is normal,
        // but a 30 MB growth budget over 200 calls catches genuine leaks.
        expect(
          rssGrowth,
          lessThan(30.0),
          reason:
              'fft1d() should free all workspace in finally; '
              'got +${rssGrowth.toStringAsFixed(2)} MB RSS over 200 calls',
        );
        if (hasVram) {
          expect(
            vramGrowth,
            lessThan(10.0),
            reason:
                'fft1d() should free all GPU workspace in finally; '
                'got +${vramGrowth.toStringAsFixed(2)} MB VRAM over 200 calls',
          );
        }
      },
    );
  });

  // ---------------------------------------------------------------------------
  // Layer 5: Buffer.readDirect staging cache — verifies read doesn't grow VRAM
  //
  // Every getData() call exercises the staging buffer.  After the first call
  // the staging buffer should be reused, not reallocated.
  // ---------------------------------------------------------------------------
  group('getData() staging buffer reuse', () {
    test(
      '500 getData() calls on a fixed-size tensor should not grow RSS > 10 MB',
      () async {
        const int n = 2048;
        final tensor = await Tensor.create([n], gpu: gpu);

        final data = Float32List(n);
        for (int i = 0; i < n; i++) data[i] = i.toDouble();
        await tensor.buffer.write(data, n, dataType: BufferDataType.float32);

        print('\n=== getData() staging cache x500 (n=$n) ===');
        final (rssGrowth, vramGrowth) = await _measureGrowth(
          () async {
            await tensor.getData();
          },
          gpu: gpu,
          iterations: 500,
          warmup: 5,
          interval: 50,
        );

        tensor.destroy();

        expect(
          rssGrowth,
          lessThan(10.0),
          reason:
              'getData() staging buffer should be reused; '
              'got +${rssGrowth.toStringAsFixed(2)} MB RSS over 500 calls',
        );
        expect(
          vramGrowth,
          lessThan(5.0),
          reason:
              'getData() staging buffer should be reused (C++ cache); '
              'got +${vramGrowth.toStringAsFixed(2)} MB VRAM over 500 calls',
        );
      },
    );
  });
}
