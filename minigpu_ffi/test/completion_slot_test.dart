/// REGRESSION: "Callback invoked after it has been deleted."
///
/// minigpu hands raw Dart function pointers to its WebGPU worker thread and
/// has no way to cancel one once enqueued. The old code built a
/// `NativeCallable.listener` per async operation and `close()`d it in a
/// `finally` — so every dispatch, every readback and every zero-copy present
/// (10ish per encoded frame) deleted a trampoline the C layer might still
/// call. When one did, the VM aborted the PROCESS:
///
///     runtime_entry.cc: error: Callback invoked after it has been deleted.
///
/// Flutter hot reload is the reliable trigger: the isolate is paused at a
/// safepoint while native threads keep running and keep completing work.
///
/// These tests exercise the trampolines the same way native does — by calling
/// the function pointer — and the important one is [stale invocation]: under
/// the old code it killed the whole test process, so a passing run IS the
/// assertion.
library;

import 'dart:async';
import 'dart:ffi';

import 'package:minigpu_ffi/minigpu_ffi_completion.dart';
import 'package:test/test.dart';

void main() {
  group('VoidCompletionSlot', () {
    test('completes its completer when native invokes the pointer', () async {
      final completer = Completer<void>();
      final slot = VoidCompletionSlot.acquire(completer);
      slot.nativeFunction.asFunction<void Function()>()();
      await completer.future.timeout(const Duration(seconds: 5));
      slot.release();
    });

    test('stale invocation after release is a no-op, not a process abort',
        () async {
      final completer = Completer<void>();
      final slot = VoidCompletionSlot.acquire(completer);
      final fn = slot.nativeFunction.asFunction<void Function()>();
      fn();
      await completer.future.timeout(const Duration(seconds: 5));
      slot.release();

      // The C layer double-fired / fired late. The pointer MUST still be live:
      // nothing may have closed it.
      fn();
      await Future<void>.delayed(const Duration(milliseconds: 50));
      expect(completer.isCompleted, isTrue);
    });

    test('recycles slots instead of allocating a trampoline per op', () async {
      final first = VoidCompletionSlot.acquire(Completer<void>());
      final address = first.nativeFunction.address;
      first.release();

      for (var i = 0; i < 100; i++) {
        final slot = VoidCompletionSlot.acquire(Completer<void>());
        expect(slot.nativeFunction.address, address,
            reason: 'sequential ops must reuse the one idle slot');
        slot.release();
      }
    });

    test('concurrent ops get distinct slots', () async {
      final a = VoidCompletionSlot.acquire(Completer<void>());
      final b = VoidCompletionSlot.acquire(Completer<void>());
      expect(a.nativeFunction.address, isNot(b.nativeFunction.address));
      a.release();
      b.release();
    });
  });

  group('IntCompletionSlot', () {
    test('reports the native result and survives a stale invocation', () async {
      final completer = Completer<bool>();
      final slot = IntCompletionSlot.acquire(completer);
      final fn = slot.nativeFunction.asFunction<void Function(int)>();
      fn(1);
      expect(await completer.future.timeout(const Duration(seconds: 5)), isTrue);
      slot.release();

      fn(0); // late completion for an operation Dart already finished
      await Future<void>.delayed(const Duration(milliseconds: 50));
    });

    test('maps 0 to false', () async {
      final completer = Completer<bool>();
      final slot = IntCompletionSlot.acquire(completer);
      slot.nativeFunction.asFunction<void Function(int)>()(0);
      expect(
        await completer.future.timeout(const Duration(seconds: 5)),
        isFalse,
      );
      slot.release();
    });
  });
}
