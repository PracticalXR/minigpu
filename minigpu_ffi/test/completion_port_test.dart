/// Regression test for async-completion delivery.
///
/// Async GPU work completes on the WebGPU worker thread. When that completion
/// travelled through a `NativeCallable`, the trampoline had to be `close()`d
/// when the operation finished — and the C layer cannot be told to forget a
/// pointer it already holds. Any completion arriving after the close aborted
/// the PROCESS:
///
///     runtime_entry.cc: ... Callback invoked after it has been deleted
///     ... isolate=(nil)
///
/// The `isolate=(nil)` is the tell: a thread with no isolate, i.e. the worker.
/// Isolate teardown deletes the trampolines too, so a worker isolate that
/// exited with work still queued killed the whole VM, not just itself.
///
/// Completions now travel as int64 messages on a Dart port, which is inert
/// once its isolate is gone. The isolate-exit test below is the proof: it is
/// the exact shape that used to be fatal.
///
/// THIS ORACLE HAS BEEN SHOWN TO FAIL. Run it with the callback path forced:
///
///     MGPU_UNSAFE_CALLBACK_COMPLETIONS=1 dart test test/completion_port_test.dart
///
/// and the isolate-exit test kills the test process with the abort above,
/// `isolate=(nil)` included. Without the flag it passes. If you change how
/// completions are delivered, re-run both directions — a green suite that
/// cannot go red proves nothing.
@TestOn('vm')
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:minigpu_ffi/minigpu_ffi.dart';
import 'package:minigpu_ffi/minigpu_ffi_completion.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:test/test.dart';

const String _addOne = '''
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&data)) { return; }
    data[gid.x] = data[gid.x] + 1.0;
}
''';

/// Issues real GPU work and then lets the isolate die WITHOUT awaiting it, so
/// the worker thread completes an operation whose issuing isolate is gone.
Future<void> _childAbandonsGpuWork(SendPort issued) async {
  final platform = MinigpuFfi();
  await platform.initializeContext();

  final buffer = platform.createBuffer(64 * 4, BufferDataType.float32);
  final shader = platform.createComputeShader()..loadKernelString(_addOne);
  shader.setBuffer(0, buffer);

  // Fire and DO NOT await: the isolate is about to be killed with these
  // completions still owed to it.
  for (var i = 0; i < 16; i++) {
    unawaited(shader.dispatch(1, 1, 1));
  }
  issued.send(true);
  // Do not return — the parent kills us mid-flight.
}

void main() {
  late MinigpuFfi platform;

  setUpAll(() async {
    platform = MinigpuFfi();
    await platform.initializeContext();
  });

  tearDownAll(() async {
    await platform.destroyContext();
  });

  test('native builds deliver completions by port, not by callback', () {
    expect(
      GpuCompletion.usesPort,
      isTrue,
      reason: 'the port path is what removes the abort; if this is false the '
          'library fell back to NativeCallables and the rest of this file is '
          'testing the fallback instead',
    );
  });

  test('dispatch and read complete, leaving nothing pending', () async {
    final buffer = platform.createBuffer(64 * 4, BufferDataType.float32);
    addTearDown(buffer.destroy);
    final shader = platform.createComputeShader()..loadKernelString(_addOne);
    addTearDown(shader.destroy);
    shader.setBuffer(0, buffer);

    await buffer.write(Float32List(64), 64);
    await shader.dispatch(1, 1, 1);

    final out = Float32List(64);
    await buffer.read(out, 64);
    expect(out.every((v) => v == 1.0), isTrue,
        reason: 'the dispatch and the readback both have to have really run');

    expect(GpuCompletion.pendingCount, 0,
        reason: 'every completion must be retired from the token map, or the '
            'map is a leak that also pins the isolate alive');
  });

  test('a hundred sequential ops do not accumulate pending tokens', () async {
    final buffer = platform.createBuffer(64 * 4, BufferDataType.float32);
    addTearDown(buffer.destroy);
    final shader = platform.createComputeShader()..loadKernelString(_addOne);
    addTearDown(shader.destroy);
    shader.setBuffer(0, buffer);

    for (var i = 0; i < 100; i++) {
      await shader.dispatch(1, 1, 1);
    }
    expect(GpuCompletion.pendingCount, 0);
  });

  test('an isolate exiting mid-dispatch does not take the process down',
      () async {
    // THE CRASH SHAPE. Under the callback path the child's trampolines are
    // deleted by its teardown while the worker thread still owes them
    // completions; the next one aborts the whole VM. Reaching the end of this
    // test is the entire assertion.
    final issued = ReceivePort();
    final child = await Isolate.spawn(_childAbandonsGpuWork, issued.sendPort);
    expect(await issued.first, isTrue);
    issued.close();

    final exited = ReceivePort();
    child.addOnExitListener(exited.sendPort);
    child.kill(priority: Isolate.immediate);
    await exited.first;
    exited.close();

    // Give the worker thread time to finish everything the dead isolate
    // abandoned, then keep using the device from THIS isolate.
    await Future<void>.delayed(const Duration(milliseconds: 300));

    final buffer = platform.createBuffer(64 * 4, BufferDataType.float32);
    addTearDown(buffer.destroy);
    final shader = platform.createComputeShader()..loadKernelString(_addOne);
    addTearDown(shader.destroy);
    shader.setBuffer(0, buffer);
    await buffer.write(Float32List(64), 64);
    await shader.dispatch(1, 1, 1);

    final out = Float32List(64);
    await buffer.read(out, 64);
    expect(out.first, 1.0, reason: 'the device must still be usable');
  });

  test('drain returns once the worker queue is empty', () async {
    final buffer = platform.createBuffer(64 * 4, BufferDataType.float32);
    addTearDown(buffer.destroy);
    final shader = platform.createComputeShader()..loadKernelString(_addOne);
    addTearDown(shader.destroy);
    shader.setBuffer(0, buffer);
    await buffer.write(Float32List(64), 64);

    // Fire-and-forget work, then drain. Ordering aid for teardown: after this
    // returns, nothing queued can still be holding a resource we free next.
    for (var i = 0; i < 8; i++) {
      shader.dispatchFire(1, 1, 1);
    }
    mgpuDrainWorkQueue();

    final out = Float32List(64);
    await buffer.read(out, 64);
    expect(out.first, 8.0,
        reason: 'all eight fire-and-forget dispatches must have executed '
            'before the drain returned');
  });

  test('drain is reachable through the platform API', () async {
    // The whole point of the export is that a SYNCHRONOUS caller can use it —
    // Flutter's reassemble() during hot reload. Prove the plumbing from the
    // platform interface down, not just the raw symbol.
    final buffer = platform.createBuffer(64 * 4, BufferDataType.float32);
    addTearDown(buffer.destroy);
    final shader = platform.createComputeShader()..loadKernelString(_addOne);
    addTearDown(shader.destroy);
    shader.setBuffer(0, buffer);
    await buffer.write(Float32List(64), 64);

    for (var i = 0; i < 4; i++) {
      shader.dispatchFire(1, 1, 1);
    }
    platform.drainWorkQueue();

    final out = Float32List(64);
    await buffer.read(out, 64);
    expect(out.first, 4.0);
  });
}
