/// Process-global context lifecycle under CONCURRENT ISOLATES.
///
/// `dart test` runs every suite as an isolate inside ONE process. The native
/// minigpu context is process-global, but the Dart [Minigpu] wrapper is a
/// per-isolate singleton, so N isolates each believe they must init it and
/// each believe their teardown is theirs to perform. Two failures followed,
/// and this file is the regression guard for both:
///
///   1. CONCURRENT INIT — `initializeContext()` bailed out only on
///      `ctx && ctx->initialized`. Thread A allocated a fresh Context and then
///      spent milliseconds in the adapter/device request with `initialized`
///      still false; thread B's lazy `getDevice()` saw exactly that, called
///      `initializeContext()`, fell through the guard and re-assigned `ctx` —
///      freeing A's live Context while A held raw pointers into it. The whole
///      VM died with an access violation (ExceptionCode=-1073741819), and the
///      log showed `buffer.cpp Device invalid, attempting to reinitialize
///      context` and a dozen `[mgpu] backend auto-select:` lines (one per
///      racing init) immediately before the abort.
///
///   2. PREMATURE TEARDOWN — one isolate's `destroy()` tore the device out
///      from under every other isolate. Serialized runs did not crash; they
///      went quiet instead, and a later suite's `createSharedOutputTexture()`
///      returned null on a device that no longer existed.
///
/// These tests spawn REAL isolates rather than racing futures on purpose: the
/// Dart-side singleton de-duplicates within an isolate, so a single-isolate
/// stress test cannot reach either bug. That is precisely why the suite missed
/// it for so long.
@TestOn('vm')
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';
import 'package:minigpu_ffi/minigpu_ffi_bindings.dart' as ffi;
import 'package:test/test.dart';

const _doubleShader = '''
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&inp)) { return; }
  outp[i] = inp[i] * 2.0;
}
''';

/// One real GPU round-trip on whatever context this isolate ends up sharing.
/// Returns null on success, or a description of the failure.
Future<String?> _gpuRoundTrip(Minigpu gpu, int tag) async {
  const n = 256;
  final input = Float32List(n);
  for (var i = 0; i < n; i++) {
    input[i] = (i + tag).toDouble();
  }
  final src = gpu.createBuffer(n * 4, BufferDataType.float32);
  final dst = gpu.createBuffer(n * 4, BufferDataType.float32);
  final shader = gpu.createComputeShader()..loadKernelString(_doubleShader);
  try {
    await src.write(input, n);
    shader
      ..setBufferAtSlot(0, src)
      ..setBufferAtSlot(1, dst);
    await shader.dispatch((n + 63) ~/ 64, 1, 1);
    final out = Float32List(n);
    await dst.read(out, n);
    for (var i = 0; i < n; i++) {
      if (out[i] != input[i] * 2.0) {
        return 'tag=$tag element $i: expected ${input[i] * 2.0}, got ${out[i]}';
      }
    }
    return null;
  } finally {
    shader.destroy();
    src.destroy();
    dst.destroy();
  }
}

/// Isolate body: init, do GPU work, optionally destroy, do GPU work again.
///
/// [args] is `[SendPort, tag, destroyMidway]`. A worker that destroys is the
/// interesting one — under the old code its teardown killed the device the
/// OTHER workers were mid-dispatch on.
Future<void> _worker(List<Object> args) async {
  final port = args[0] as SendPort;
  final tag = args[1] as int;
  final destroyMidway = args[2] as bool;
  try {
    final gpu = Minigpu();
    await gpu.init();
    final err1 = await _gpuRoundTrip(gpu, tag);
    if (err1 != null) {
      port.send('FAIL(before destroy): $err1');
      return;
    }
    if (destroyMidway) {
      await gpu.destroy();
      // Re-attach and prove this isolate can still work afterwards.
      await gpu.init();
    }
    final err2 = await _gpuRoundTrip(gpu, tag + 1000);
    port.send(err2 == null ? 'OK' : 'FAIL(after destroy): $err2');
  } catch (e, st) {
    port.send('THREW: $e\n$st');
  }
}

/// Runs [count] worker isolates concurrently and collects their verdicts.
Future<List<String>> _race(int count, {required Set<int> destroyers}) async {
  final rx = ReceivePort();
  final results = <String>[];
  final done = Completer<void>();
  rx.listen((msg) {
    results.add(msg as String);
    if (results.length == count && !done.isCompleted) done.complete();
  });
  final isolates = <Isolate>[];
  for (var i = 0; i < count; i++) {
    isolates.add(
      await Isolate.spawn(_worker, <Object>[
        rx.sendPort,
        i,
        destroyers.contains(i),
      ], errorsAreFatal: false),
    );
  }
  await done.future.timeout(
    const Duration(minutes: 3),
    onTimeout: () => throw StateError(
      'workers did not all report within 3 min — a context-lifecycle '
      'deadlock is the first thing to suspect (got ${results.length}/$count: '
      '$results)',
    ),
  );
  rx.close();
  for (final iso in isolates) {
    iso.kill(priority: Isolate.immediate);
  }
  return results;
}

void main() {
  group('process-global context under concurrent isolates', () {
    // FIRST on purpose: the racing tests below deliberately leave worker
    // isolates attached (an isolate that exits without destroying keeps the
    // context pinned — that is the point), so the absolute-count assertions
    // here only hold before they have run.
    test('reference count: attach/detach is balanced, teardown at zero',
        () async {
      // Single-consumer semantics must be BYTE-FOR-BYTE what they were:
      // init -> 1 -> real init, destroy -> 0 -> real teardown.
      //
      // The EXACT counts are only observable when this suite owns the process.
      // In a whole-package run other suites are isolates attaching to the SAME
      // process-global context and the absolute count moves underneath us —
      // which is the fix working, not a failure. So: assert exact counts when
      // we start from a quiet process, and assert the concurrency-safe
      // invariants (attached => >= 1, context stays usable across a paired
      // attach/detach) either way.
      final base = ffi.mgpuContextRefCount();
      final alone = base == 0;

      final gpu = Minigpu();
      await gpu.init();
      expect(ffi.mgpuContextRefCount(), greaterThanOrEqualTo(1));
      if (alone) expect(ffi.mgpuContextRefCount(), 1);

      // A second attach from the same process (what a second isolate does)
      // must not re-initialize, and the FIRST detach must not tear down.
      ffi.mgpuInitializeContext();
      if (alone) expect(ffi.mgpuContextRefCount(), 2);
      ffi.mgpuDestroyContext();
      if (alone) expect(ffi.mgpuContextRefCount(), 1);
      expect(ffi.mgpuContextRefCount(), greaterThanOrEqualTo(1));

      // The context must still be LIVE for the remaining consumer — this is
      // the assertion that fails if a detach tore the device down early, and
      // it holds regardless of what other isolates are doing.
      expect(
        await _gpuRoundTrip(gpu, 42),
        isNull,
        reason: 'a non-final detach must leave the context usable',
      );

      await gpu.destroy();
      if (alone) {
        expect(
          ffi.mgpuContextRefCount(),
          0,
          reason: 'the last detach must tear the context down',
        );
      }
    }, timeout: const Timeout(Duration(minutes: 2)));

    test('N isolates racing init + dispatch neither crash nor corrupt', () async {
      // 8 was chosen to exceed the default `dart test` concurrency on this
      // box; the original crash reproduced at roughly that width.
      final results = await _race(8, destroyers: const {});
      expect(
        results.where((r) => r != 'OK'),
        isEmpty,
        reason:
            'Every isolate must complete a correct GPU round-trip on the '
            'shared process-global context. A hang or a missing result means '
            'the process aborted (access violation from a Context freed under '
            'a live user).',
      );
      expect(results, hasLength(8));
    }, timeout: const Timeout(Duration(minutes: 4)));

    test('one isolate destroying does not break the others', () async {
      // Isolates 1 and 4 tear down while 0/2/3/5 are still dispatching. Under
      // the old unconditional teardown, this is the shape that left later
      // users on a dead device (createSharedOutputTexture -> null) or, with
      // the timing slightly different, aborted the VM outright.
      final results = await _race(6, destroyers: const {1, 4});
      expect(
        results.where((r) => r != 'OK'),
        isEmpty,
        reason:
            'A consumer that finishes and detaches must not tear down a '
            'context other consumers are still attached to.',
      );
      expect(results, hasLength(6));
    }, timeout: const Timeout(Duration(minutes: 4)));
  });
}
