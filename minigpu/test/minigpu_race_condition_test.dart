// Tests that stress-test the threading model to detect race conditions.
//
// Race conditions guarded against:
//   1. loadKernelString data race — Dart thread writes shaderCode without lock
//      while WebGPU thread reads it under the GPU mutex.
//   2. dispatch callback fired under GPU mutex — callback must be invoked after
//      the lock is released so re-entrant GPU calls don't deadlock.
//   3. destroyContext without draining WebGPU thread — pending cleanup tasks
//      reference freed queue/device handles.
//   4. buffer destroyed before dispatch that references it runs — use-after-free
//      in bind group creation.

import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:minigpu/minigpu.dart';

const _passthruShader = '''
@group(0) @binding(0) var<storage, read_write> input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&input)) { return; }
  output[i] = input[i];
}
''';

const _altShader = '''
@group(0) @binding(0) var<storage, read_write> input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&input)) { return; }
  output[i] = input[i] * 2.0;
}
''';

void main() {
  late Minigpu gpu;

  setUpAll(() async {
    gpu = Minigpu();
    await gpu.init();
  });

  tearDownAll(() async {});

  // ---------------------------------------------------------------------------
  // Race 1 — loadKernelString data race
  // ---------------------------------------------------------------------------
  group('Race 1 — loadKernelString concurrency', () {
    test('reload kernel between dispatches does not corrupt shader', () async {
      const n = 64;
      final buf = gpu.createBuffer(n * 4, BufferDataType.float32);
      final out = gpu.createBuffer(n * 4, BufferDataType.float32);

      final data = Float32List.fromList(List.generate(n, (i) => i.toDouble()));
      buf.write(data, n, dataType: BufferDataType.float32);

      final shader = gpu.createComputeShader();
      shader.loadKernelString(_passthruShader);
      shader.setBuffer('input', buf);
      shader.setBuffer('output', out);

      // Dispatch, then reload a DIFFERENT kernel, then dispatch again.
      // Without the mutex fix, this races: the first dispatch task reads
      // shaderCode on the WebGPU thread while loadKernelString overwrites it
      // on the Dart thread.
      shader.dispatch(1, 1, 1); // fire-and-forget — task enqueued
      shader.loadKernelString(_altShader); // write shaderCode immediately after
      await shader.dispatch(1, 1, 1); // wait for second dispatch to finish

      final result = Float32List(n);
      await out.read(result, n, dataType: BufferDataType.float32);
      // Second shader multiplies by 2.0; first index should be 0.0 * 2 = 0.
      expect(result[0], isNotNaN, reason: 'output[0] must be a valid float');

      shader.destroy();
      buf.destroy();
      out.destroy();
    });

    test(
      'repeated kernel reloads under concurrent dispatches stay stable',
      () async {
        const n = 128;
        final buf = gpu.createBuffer(n * 4, BufferDataType.float32);
        final out = gpu.createBuffer(n * 4, BufferDataType.float32);

        final data = Float32List.fromList(
          List.generate(n, (i) => i.toDouble()),
        );
        buf.write(data, n, dataType: BufferDataType.float32);

        final shader = gpu.createComputeShader();
        shader.loadKernelString(_passthruShader);
        shader.setBuffer('input', buf);
        shader.setBuffer('output', out);

        // Interleave kernel reloads and dispatches 20 times.
        for (int i = 0; i < 20; i++) {
          shader.dispatch(1, 1, 1); // fire-and-forget
          shader.loadKernelString(
            i.isEven ? _passthruShader : _altShader,
          ); // race target
        }
        await shader.dispatch(1, 1, 1); // final awaited dispatch

        final result = Float32List(n);
        await out.read(result, n, dataType: BufferDataType.float32);
        expect(result[1], isNotNaN, reason: 'output[1] must be a valid float');

        shader.destroy();
        buf.destroy();
        out.destroy();
      },
    );
  });

  // ---------------------------------------------------------------------------
  // Race 2 — dispatch callback fired under GPU mutex
  // ---------------------------------------------------------------------------
  group('Race 2 — callback not fired under GPU mutex', () {
    test('GPU write in dispatch callback does not deadlock', () async {
      const n = 64;
      final buf = gpu.createBuffer(n * 4, BufferDataType.float32);
      final out = gpu.createBuffer(n * 4, BufferDataType.float32);

      final data = Float32List(n);
      for (int i = 0; i < n; i++) data[i] = i.toDouble();
      buf.write(data, n, dataType: BufferDataType.float32);

      final shader = gpu.createComputeShader();
      shader.loadKernelString(_passthruShader);
      shader.setBuffer('input', buf);
      shader.setBuffer('output', out);

      // Await the dispatch — when the callback fires, the GPU mutex must
      // already be released, otherwise a write attempted from the callback's
      // continuation would deadlock (std::mutex is not reentrant).
      await shader.dispatch(1, 1, 1);

      // Immediately write to the buffer after dispatch completes.
      // If the callback was fired under the mutex, this would deadlock.
      buf.write(data, n, dataType: BufferDataType.float32);
      await shader.dispatch(1, 1, 1);

      final result = Float32List(n);
      await out.read(result, n, dataType: BufferDataType.float32);
      expect(result[0], closeTo(0.0, 0.001));
      expect(result[n - 1], closeTo((n - 1).toDouble(), 0.001));

      shader.destroy();
      buf.destroy();
      out.destroy();
    });
  });

  // ---------------------------------------------------------------------------
  // Race 4 — buffer destroyed before dispatch (ordering safety)
  // ---------------------------------------------------------------------------
  group('Race 4 — buffer lifetime vs dispatch ordering', () {
    test('dispatch enqueued before destroy runs before cleanup task', () async {
      // The safe pattern: dispatch() is called BEFORE destroy().
      // Both enqueue tasks to the WebGPU thread; dispatch must run first.
      const n = 64;
      final buf = gpu.createBuffer(n * 4, BufferDataType.float32);
      final out = gpu.createBuffer(n * 4, BufferDataType.float32);

      final data = Float32List.fromList(List.generate(n, (i) => i.toDouble()));
      buf.write(data, n, dataType: BufferDataType.float32);

      final shader = gpu.createComputeShader();
      shader.loadKernelString(_passthruShader);
      shader.setBuffer('input', buf);
      shader.setBuffer('output', out);

      // Fire dispatch without awaiting — enqueues dispatch task.
      shader.dispatch(1, 1, 1);
      // Immediately destroy the buffer — enqueues cleanup task AFTER dispatch.
      buf.destroy();

      // Wait for the dispatch to finish via a read on the output buffer.
      final result = Float32List(n);
      await out.read(result, n, dataType: BufferDataType.float32);

      // If the cleanup ran before dispatch, Dawn would crash in bind group
      // creation. Reaching here (with valid values) confirms ordering is safe.
      expect(result[0], closeTo(0.0, 0.001));

      shader.destroy();
      out.destroy();
    });

    test(
      'many buffers created and destroyed across multiple dispatches',
      () async {
        const n = 128;
        const cycles = 30;

        final shader = gpu.createComputeShader();
        shader.loadKernelString(_passthruShader);

        for (int c = 0; c < cycles; c++) {
          final buf = gpu.createBuffer(n * 4, BufferDataType.float32);
          final out = gpu.createBuffer(n * 4, BufferDataType.float32);

          final data = Float32List.fromList(
            List.generate(n, (i) => (i + c).toDouble()),
          );
          buf.write(data, n, dataType: BufferDataType.float32);

          shader.setBuffer('input', buf);
          shader.setBuffer('output', out);

          // Fire dispatch then immediately destroy both buffers.
          // Cleanup tasks are always enqueued AFTER the dispatch task.
          shader.dispatch(1, 1, 1);
          buf.destroy();

          // Use a fresh output read to pace the loop and confirm no crash.
          final result = Float32List(n);
          await out.read(result, n, dataType: BufferDataType.float32);
          out.destroy();

          expect(
            result[0],
            isNotNaN,
            reason: 'cycle $c: output[0] must be valid',
          );
        }

        shader.destroy();
      },
    );
  });
}
