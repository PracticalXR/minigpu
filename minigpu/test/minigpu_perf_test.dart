// Performance benchmarks for the minigpu library.
//
// Metrics captured for each benchmark:
//   - Total wall time for the operation
//   - Throughput (elements/second or GB/s where applicable)
//   - Per-iteration latency (dispatch round-trip, buffer upload/download)
//
// Run individually for clean numbers:
//   flutter test test/minigpu_perf_test.dart --timeout=120s

import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:minigpu/minigpu.dart';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

class PerfResult {
  final String label;
  final int iterations;
  final Duration total;
  final String? throughput;

  PerfResult(this.label, this.iterations, this.total, {this.throughput});

  double get msPerIter => total.inMicroseconds / 1000.0 / iterations;

  @override
  String toString() {
    final tp = throughput != null ? '  throughput: $throughput' : '';
    return '[PERF] $label\n'
        '  iterations : $iterations\n'
        '  total      : ${total.inMilliseconds} ms\n'
        '  per-iter   : ${msPerIter.toStringAsFixed(3)} ms$tp';
  }
}

/// Runs [fn] [warmup] times (discarded), then [iters] times, and returns timing.
Future<PerfResult> bench(
  String label,
  int iters,
  Future<void> Function() fn, {
  int warmup = 3,
  String Function(int iters, Duration total)? throughputFn,
}) async {
  for (int i = 0; i < warmup; i++) await fn();
  final sw = Stopwatch()..start();
  for (int i = 0; i < iters; i++) await fn();
  sw.stop();
  final tp = throughputFn?.call(iters, sw.elapsed);
  final result = PerfResult(label, iters, sw.elapsed, throughput: tp);
  // ignore: avoid_print
  print(result);
  return result;
}

// ---------------------------------------------------------------------------
// Shader snippets
// ---------------------------------------------------------------------------

const _saxpyShader = '''
@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&y)) { return; }
  y[i] = 2.0 * x[i] + y[i];
}
''';

// Each output element is the sum of `stride` consecutive input elements.
// Dispatch one thread per output element; no workgroup shared memory needed.
const _reductionShader = '''
@group(0) @binding(0) var<storage, read_write> input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outLen = arrayLength(&output);
  let inLen  = arrayLength(&input);
  let idx    = gid.x;
  if (idx >= outLen) { return; }
  let stride = (inLen + outLen - 1u) / outLen;
  let start  = idx * stride;
  var sum    = 0.0;
  for (var i = start; i < min(start + stride, inLen); i++) {
    sum += input[i];
  }
  output[idx] = sum;
}
''';

// Naive O(N^3) matrix multiply: C = A * B, row-major, square N×N.
// No workgroup shared memory; demonstrates compute / memory BW.
const _matmulShader = '''
@group(0) @binding(0) var<storage, read_write> A:   array<f32>;
@group(0) @binding(1) var<storage, read_write> B:   array<f32>;
@group(0) @binding(2) var<storage, read_write> C:   array<f32>;
@group(0) @binding(3) var<storage, read_write> dim: array<u32>; // dim[0] = N
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let N   = dim[0];
  let row = gid.y;
  let col = gid.x;
  if (row >= N || col >= N) { return; }
  var acc = 0.0;
  for (var k = 0u; k < N; k++) {
    acc += A[row * N + k] * B[k * N + col];
  }
  C[row * N + col] = acc;
}
''';

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
  late Minigpu gpu;

  setUpAll(() async {
    gpu = Minigpu();
    await gpu.init();
  });

  tearDownAll(() async {});

  // -------------------------------------------------------------------------
  // 1. Buffer upload / download throughput
  // -------------------------------------------------------------------------
  group('Buffer I/O throughput', () {
    for (final mb in [1, 16, 64]) {
      final n = (mb * 1024 * 1024) ~/ 4; // float32 elements

      test('upload ${mb}MB', () async {
        final buf = gpu.createBuffer(n * 4, BufferDataType.float32);
        final data = Float32List(n);
        for (int i = 0; i < n; i++) data[i] = i.toDouble();

        final r = await bench(
          'upload ${mb}MB',
          10,
          () => buf.write(data, n, dataType: BufferDataType.float32),
          throughputFn: (iters, dur) {
            final gbps = (iters * mb / 1024) / (dur.inMicroseconds / 1e6);
            return '${gbps.toStringAsFixed(2)} GB/s';
          },
        );
        buf.destroy();
        expect(
          r.msPerIter,
          lessThan(500),
          reason: '${mb}MB upload should finish in <500 ms',
        );
      });

      test('download ${mb}MB', () async {
        final buf = gpu.createBuffer(n * 4, BufferDataType.float32);
        final data = Float32List(n);

        final r = await bench(
          'download ${mb}MB',
          10,
          () => buf.read(data, n, dataType: BufferDataType.float32),
          throughputFn: (iters, dur) {
            final gbps = (iters * mb / 1024) / (dur.inMicroseconds / 1e6);
            return '${gbps.toStringAsFixed(2)} GB/s';
          },
        );
        buf.destroy();
        expect(
          r.msPerIter,
          lessThan(1000),
          reason: '${mb}MB download should finish in <1000 ms',
        );
      });
    }
  });

  // -------------------------------------------------------------------------
  // 2. Dispatch latency (small, already-compiled shader)
  // -------------------------------------------------------------------------
  group('Dispatch latency', () {
    test('empty workgroup dispatch round-trip', () async {
      const n = 1;
      final buf = gpu.createBuffer(n * 4, BufferDataType.float32);
      final shader = gpu.createComputeShader();
      shader.loadKernelString(_saxpyShader);
      shader.setBuffer('x', buf);
      shader.setBuffer('y', buf);

      final r = await bench(
        'dispatch round-trip (1 element)',
        100,
        () => shader.dispatch(1, 1, 1),
      );

      shader.destroy();
      buf.destroy();
      expect(
        r.msPerIter,
        lessThan(50),
        reason: 'a single dispatch should complete in <50 ms',
      );
    });

    test('dispatch + read round-trip (1 MB)', () async {
      const n = 256 * 1024; // 1 MB
      final x = gpu.createBuffer(n * 4, BufferDataType.float32);
      final y = gpu.createBuffer(n * 4, BufferDataType.float32);
      final xData = Float32List(n);
      final yData = Float32List(n);
      for (int i = 0; i < n; i++) {
        xData[i] = 1.0;
        yData[i] = 1.0;
      }
      x.write(xData, n, dataType: BufferDataType.float32);
      y.write(yData, n, dataType: BufferDataType.float32);

      final shader = gpu.createComputeShader();
      shader.loadKernelString(_saxpyShader);
      shader.setBuffer('x', x);
      shader.setBuffer('y', y);
      final out = Float32List(n);

      final r = await bench('SAXPY + read 1 MB', 20, () async {
        await shader.dispatch((n + 255) ~/ 256, 1, 1);
        await y.read(out, n, dataType: BufferDataType.float32);
      });

      shader.destroy();
      x.destroy();
      y.destroy();
      expect(r.msPerIter, lessThan(200));
      // y[i] grows by 2*x[i] = 2 each iteration; just check it's non-zero
      expect(out[0], greaterThan(0.0));
    });
  });

  // -------------------------------------------------------------------------
  // 3. Compute throughput — SAXPY at various sizes
  // -------------------------------------------------------------------------
  group('Compute throughput — SAXPY', () {
    for (final n in [64 * 1024, 1024 * 1024, 16 * 1024 * 1024]) {
      final label = n >= 1024 * 1024
          ? '${n ~/ (1024 * 1024)} M elements'
          : '${n ~/ 1024} K elements';

      test('SAXPY $label', () async {
        final x = gpu.createBuffer(n * 4, BufferDataType.float32);
        final y = gpu.createBuffer(n * 4, BufferDataType.float32);
        final xd = Float32List(n);
        final yd = Float32List(n);
        for (int i = 0; i < n; i++) {
          xd[i] = 1.0;
          yd[i] = 0.0;
        }
        x.write(xd, n, dataType: BufferDataType.float32);
        y.write(yd, n, dataType: BufferDataType.float32);

        final shader = gpu.createComputeShader();
        shader.loadKernelString(_saxpyShader);
        shader.setBuffer('x', x);
        shader.setBuffer('y', y);

        final r = await bench(
          'SAXPY $label',
          20,
          () => shader.dispatch((n + 255) ~/ 256, 1, 1),
          throughputFn: (iters, dur) {
            // 3 floats read/written per element (read x, read y, write y)
            final bytes = iters * n * 3 * 4;
            final gbps = bytes / (dur.inMicroseconds / 1e6) / 1e9;
            return '${gbps.toStringAsFixed(2)} GB/s effective memory BW';
          },
        );

        shader.destroy();
        x.destroy();
        y.destroy();
        expect(r.msPerIter, isNonNegative);
      });
    }
  });

  // -------------------------------------------------------------------------
  // 4. Reduction throughput
  // -------------------------------------------------------------------------
  group('Compute throughput — reduction', () {
    test('sum reduction 4 M elements', () async {
      const n = 4 * 1024 * 1024;
      // 16384 output elements, each sums n/16384 = 256 consecutive inputs.
      const numOutputs = n ~/ 256; // 16384
      final input = gpu.createBuffer(n * 4, BufferDataType.float32);
      final output = gpu.createBuffer(numOutputs * 4, BufferDataType.float32);

      final data = Float32List(n);
      for (int i = 0; i < n; i++) data[i] = 1.0;
      input.write(data, n, dataType: BufferDataType.float32);

      final shader = gpu.createComputeShader();
      shader.loadKernelString(_reductionShader);
      shader.setBuffer('input', input);
      shader.setBuffer('output', output);

      // One thread per output element; 256 threads per workgroup.
      const dispatchGroups = (numOutputs + 255) ~/ 256; // 64

      final r = await bench(
        'reduction 4M floats',
        20,
        () => shader.dispatch(dispatchGroups, 1, 1),
      );

      // Each output element sums 256 all-ones → 256.0
      final result = Float32List(numOutputs);
      await output.read(result, numOutputs, dataType: BufferDataType.float32);
      expect(
        result[0],
        closeTo(256.0, 1.0),
        reason: 'first partial sum should be 256',
      );

      shader.destroy();
      input.destroy();
      output.destroy();
      expect(r.msPerIter, isNonNegative);
    });
  });

  // -------------------------------------------------------------------------
  // 5. Matrix multiply (tiled 16×16)
  // -------------------------------------------------------------------------
  group('Compute throughput — matmul', () {
    for (final N in [128, 512]) {
      test('${N}×${N} matmul', () async {
        final aData = Float32List(N * N);
        final bData = Float32List(N * N);
        for (int i = 0; i < N * N; i++) {
          aData[i] = 1.0;
          bData[i] = 1.0;
        }
        final dimData = Uint32List.fromList([N]);

        final aBuf = gpu.createBuffer(N * N * 4, BufferDataType.float32);
        final bBuf = gpu.createBuffer(N * N * 4, BufferDataType.float32);
        final cBuf = gpu.createBuffer(N * N * 4, BufferDataType.float32);
        final dimBuf = gpu.createBuffer(
          dimData.lengthInBytes,
          BufferDataType.uint32,
        );

        aBuf.write(aData, N * N, dataType: BufferDataType.float32);
        bBuf.write(bData, N * N, dataType: BufferDataType.float32);
        dimBuf.write(dimData, dimData.length, dataType: BufferDataType.uint32);

        final shader = gpu.createComputeShader();
        shader.loadKernelString(_matmulShader);
        shader.setBuffer('A', aBuf);
        shader.setBuffer('B', bBuf);
        shader.setBuffer('C', cBuf);
        shader.setBuffer('dim', dimBuf);

        final groups = (N + 15) ~/ 16;
        final r = await bench(
          '${N}×${N} matmul',
          10,
          () => shader.dispatch(groups, groups, 1),
          throughputFn: (iters, dur) {
            // FLOP count: 2*N^3 per matmul (N^3 muls + N^3 adds)
            final flops = 2.0 * N * N * N * iters;
            final gflops = flops / (dur.inMicroseconds / 1e6) / 1e9;
            return '${gflops.toStringAsFixed(2)} GFLOP/s';
          },
        );

        final cData = Float32List(N * N);
        await cBuf.read(cData, N * N, dataType: BufferDataType.float32);
        // A is all-ones, B is all-ones → C[i,j] = N for every element
        expect(
          cData[0],
          closeTo(N.toDouble(), 1.0),
          reason: 'C[0,0] should be N for all-ones inputs',
        );

        shader.destroy();
        aBuf.destroy();
        bBuf.destroy();
        cBuf.destroy();
        dimBuf.destroy();
        expect(r.msPerIter, isNonNegative);
      });
    }
  });

  // -------------------------------------------------------------------------
  // 6. Pipeline overhead — many small dispatches
  // -------------------------------------------------------------------------
  group('Pipeline overhead', () {
    test('1000 consecutive 1-element dispatches', () async {
      const n = 1;
      final buf = gpu.createBuffer(n * 4, BufferDataType.float32);
      final shader = gpu.createComputeShader();
      shader.loadKernelString(_saxpyShader);
      shader.setBuffer('x', buf);
      shader.setBuffer('y', buf);

      final sw = Stopwatch()..start();
      for (int i = 0; i < 1000; i++) {
        await shader.dispatch(1, 1, 1);
      }
      sw.stop();

      final msPerDispatch = sw.elapsedMicroseconds / 1000 / 1000.0;
      // ignore: avoid_print
      print(
        '[PERF] 1000 sequential tiny dispatches: '
        '${sw.inMilliseconds} ms total, '
        '${msPerDispatch.toStringAsFixed(3)} ms/dispatch',
      );

      shader.destroy();
      buf.destroy();
      expect(
        sw.inMilliseconds,
        lessThan(30000),
        reason: '1000 dispatches should complete within 30 seconds',
      );
    });

    test('dispatch throughput — fire-and-forget 500 times', () async {
      const n = 256 * 1024;
      final x = gpu.createBuffer(n * 4, BufferDataType.float32);
      final y = gpu.createBuffer(n * 4, BufferDataType.float32);
      final shader = gpu.createComputeShader();
      shader.loadKernelString(_saxpyShader);
      shader.setBuffer('x', x);
      shader.setBuffer('y', y);

      final sw = Stopwatch()..start();
      for (int i = 0; i < 499; i++) {
        shader.dispatch((n + 255) ~/ 256, 1, 1); // fire-and-forget
      }
      await shader.dispatch((n + 255) ~/ 256, 1, 1); // final awaited
      sw.stop();

      // ignore: avoid_print
      print(
        '[PERF] 500 fire-and-forget dispatches (1 MB each): '
        '${sw.inMilliseconds} ms total',
      );

      shader.destroy();
      x.destroy();
      y.destroy();
      expect(sw.inMilliseconds, lessThan(60000));
    });
  });
}

extension on Stopwatch {
  int get inMilliseconds => elapsed.inMilliseconds;
}
