/// COMPLETION DELIVERY FOR ASYNC GPU WORK.
///
/// Every async minigpu entry point starts work on the **WebGPU worker thread**
/// and signals Dart when it finishes. How that signal travels is the whole
/// subject of this file, because the obvious way is unsound.
///
/// ## Why not a NativeCallable
///
/// ```dart
/// final nc = NativeCallable<Void Function()>.listener(cb);
/// try {
///   ffi.mgpuSomethingAsync(..., nc.nativeFunction);
///   await completer.future;
/// } finally {
///   nc.close();            // ← THE BUG
/// }
/// ```
///
/// `close()` DELETES the trampoline, and the C layer cannot be told to forget
/// a pointer it has already been handed — there is no cancel. So the `finally`
/// is a bet that native will never call again, and the VM's penalty for losing
/// it is not an exception:
///
///     runtime_entry.cc: error: Callback invoked after it has been deleted.
///
/// That is an unconditional `FATAL`: the whole process dies, uncatchable, and
/// Flutter reports it as `Lost connection to device.` Worse, `close()` is not
/// the only deleter — isolate teardown deletes every callback the isolate
/// owns, so hot restart, a hard kill, and any worker isolate exiting mid-work
/// are all fatal too, and no amount of Dart-side discipline can cover them.
///
/// ## The port
///
/// Posting to a Dart port that is closed, or whose isolate is gone, is a
/// defined, silent no-op. It is thread-safe from any thread, with or without
/// an isolate — which is exactly what the WebGPU worker and Dawn's own threads
/// are. So completions travel as a port message, and the failure mode above
/// stops existing rather than being guarded against. The log stream already
/// made this move for the same reason; see `minigpu_ffi_log_port.dart`.
///
/// WIRE FORMAT: one int64 per completion, `(token << 1) | ok`. Tokens are
/// allocated here, never reused, and dropped from [_pending] as soon as they
/// resolve — so a late or duplicate completion is a map miss rather than
/// somebody else's future resolving early.
///
/// ## The fallback
///
/// A build without the vendored Dart API (the Emscripten/web library) cannot
/// post to a port, so [VoidCompletionSlot] / [IntCompletionSlot] remain: pooled
/// `NativeCallable`s that are recycled but **never closed**, which removes the
/// abort even though it cannot remove the trampoline. They are the fallback,
/// not the default.
@DefaultAsset('package:minigpu_ffi/minigpu_ffi_bindings.dart')
library;

import 'dart:async';
import 'dart:collection';
import 'dart:ffi';
import 'dart:io' show Platform;
import 'dart:isolate';

import 'package:minigpu_ffi/minigpu_ffi_bindings.dart' as ffi;
import 'package:minigpu_ffi/minigpu_ffi_log_port.dart' as log_port;

/// How an operation is started when the port path is available.
typedef PortIssue = void Function(int port, int token);

/// How it is started when it is not: with a `void()` callback pointer.
typedef VoidCallbackIssue =
    void Function(Pointer<NativeFunction<Void Function()>> callback);

/// …or with a `void(int)` callback pointer, for work that reports success.
typedef IntCallbackIssue =
    void Function(Pointer<NativeFunction<Void Function(Int)>> callback);

/// Issues async GPU work and awaits its completion.
abstract final class GpuCompletion {
  /// Runs an operation whose native completion carries no result.
  ///
  /// Returns false only when the C layer rejected the request outright (a null
  /// handle, an unknown element type). Callers that had no notion of failure
  /// before can ignore it — the point of reporting rather than staying silent
  /// is that the old code left such callers awaiting a future that would never
  /// complete.
  static Future<bool> run(PortIssue viaPort, VoidCallbackIssue viaCallback) {
    if (_portReady) return _viaPort(viaPort);
    return _viaVoidCallback(viaCallback);
  }

  /// Runs an operation whose native completion carries a success flag.
  static Future<bool> runInt(PortIssue viaPort, IntCallbackIssue viaCallback) {
    if (_portReady) return _viaPort(viaPort);
    return _viaIntCallback(viaCallback);
  }

  /// Whether completions are delivered by port (the safe path). False only on
  /// builds without the vendored Dart API.
  static bool get usesPort => _portReady;

  /// Completions still awaiting a native post. Diagnostics/tests.
  static int get pendingCount => _pending.length;

  // --- Port path -----------------------------------------------------------

  static final Map<int, Completer<bool>> _pending = <int, Completer<bool>>{};
  static RawReceivePort? _rx;
  static int _nextToken = 1;

  static Future<bool> _viaPort(PortIssue viaPort) {
    final rx = _rx ??= _openPort();
    final completer = Completer<bool>();
    final token = _nextToken++;
    _pending[token] = completer;
    // An operation is in flight, so the isolate must stay alive to receive its
    // completion — the same rule an `await` on any other event implies.
    rx.keepIsolateAlive = true;
    try {
      viaPort(rx.sendPort.nativePort, token);
    } catch (_) {
      _resolve(token, false);
      rethrow;
    }
    return completer.future;
  }

  static RawReceivePort _openPort() {
    final rx = RawReceivePort(_onCompletion, 'minigpu_ffi.completions');
    // The port outlives every individual operation (reopening one per call
    // would be pure churn on a per-frame path), so by default it would pin the
    // isolate forever. Only in-flight work may do that.
    rx.keepIsolateAlive = false;
    return rx;
  }

  static void _onCompletion(dynamic message) {
    if (message is! int) return;
    _resolve(message >> 1, (message & 1) != 0);
  }

  static void _resolve(int token, bool ok) {
    final completer = _pending.remove(token);
    if (_pending.isEmpty) _rx?.keepIsolateAlive = false;
    // A miss is a late or duplicate completion for work Dart has already
    // finished with. Dropping it is the correct and intended outcome.
    if (completer != null && !completer.isCompleted) completer.complete(ok);
  }

  /// Whether this build can deliver completions by port. Resolved once: the
  /// native library either has the vendored Dart API compiled in or it does
  /// not, and `mgpuInitDartApi` is idempotent.
  static final bool _portReady = _initPort();

  static bool _initPort() {
    // POSITIVE CONTROL ONLY. MGPU_UNSAFE_CALLBACK_COMPLETIONS=1 forces the old
    // NativeCallable path so the abort it causes can be reproduced on demand —
    // a regression test that has never been shown to fail is not a test. Under
    // it, an isolate that exits with GPU work in flight kills the process.
    // Never set it in production.
    if (Platform.environment['MGPU_UNSAFE_CALLBACK_COMPLETIONS'] == '1') {
      return false;
    }
    try {
      if (log_port.mgpuInitDartApi(NativeApi.initializeApiDLData) != 0) {
        return false; // built without the vendored Dart API
      }
      // PROBE A NEW SYMBOL BEFORE COMMITTING TO THE PORT PATH. The Dart API
      // bridge shipped before these entry points did, so "the bridge exists"
      // does not imply "the port exports exist" — a binary older than this
      // Dart code, or a stale one served out of a build cache, would resolve
      // mgpuInitDartApi and then fail at the first dispatch. Resolving the
      // address here turns that into a clean fallback instead of a throw from
      // the middle of a frame.
      Native.addressOf<NativeFunction<Void Function()>>(mgpuDrainWorkQueue);
      return true;
    } catch (_) {
      // Missing symbol: an older or stale native library. The pooled-callback
      // path still works, it just cannot survive isolate teardown.
      return false;
    }
  }

  // --- Callback fallback ---------------------------------------------------

  static Future<bool> _viaVoidCallback(VoidCallbackIssue issue) async {
    final completer = Completer<void>();
    final slot = VoidCompletionSlot.acquire(completer);
    try {
      issue(slot.nativeFunction);
      await completer.future;
      return true;
    } finally {
      slot.release();
    }
  }

  static Future<bool> _viaIntCallback(IntCallbackIssue issue) async {
    final completer = Completer<bool>();
    final slot = IntCompletionSlot.acquire(completer);
    try {
      issue(slot.nativeFunction);
      return await completer.future;
    } finally {
      slot.release();
    }
  }
}

/// A pooled `void()` completion callback for the no-port fallback.
///
/// Pooled and **never closed**: a slot is recycled for later operations, so a
/// stale native invocation lands on a live trampoline with no pending
/// completer and is a no-op rather than a process abort. FIFO recycling
/// maximises the gap between "native might still call this" and "this is
/// serving a new operation". Slots pin their isolate only while an operation
/// is actually in flight.
final class VoidCompletionSlot {
  VoidCompletionSlot._() {
    _nc = NativeCallable<Void Function()>.listener(_fire);
    _nc.keepIsolateAlive = false; // idle by construction
  }

  late final NativeCallable<Void Function()> _nc;
  Completer<void>? _completer;

  /// The pointer to hand to the C layer. Valid for the lifetime of the
  /// process — it is never closed.
  Pointer<NativeFunction<Void Function()>> get nativeFunction =>
      _nc.nativeFunction;

  void _fire() {
    final completer = _completer;
    _completer = null;
    _nc.keepIsolateAlive = false;
    if (completer != null && !completer.isCompleted) completer.complete();
  }

  static final Queue<VoidCompletionSlot> _free = Queue<VoidCompletionSlot>();

  /// Takes a slot that will complete [completer] when native calls back.
  static VoidCompletionSlot acquire(Completer<void> completer) {
    final slot = _free.isEmpty ? VoidCompletionSlot._() : _free.removeFirst();
    slot._completer = completer;
    slot._nc.keepIsolateAlive = true;
    return slot;
  }

  /// Returns the slot to the pool. Never closes it: the C layer may still hold
  /// [nativeFunction] and there is no way to ask it to forget.
  void release() {
    _completer = null;
    _nc.keepIsolateAlive = false;
    _free.addLast(this);
  }

  /// Idle trampoline count, for tests/diagnostics.
  static int get pooledCount => _free.length;
}

/// A pooled `void(int)` completion callback for the no-port fallback — the
/// shared-texture blits report success as an int. Same lifetime rules as
/// [VoidCompletionSlot].
final class IntCompletionSlot {
  IntCompletionSlot._() {
    _nc = NativeCallable<Void Function(Int)>.listener(_fire);
    _nc.keepIsolateAlive = false;
  }

  late final NativeCallable<Void Function(Int)> _nc;
  Completer<bool>? _completer;

  Pointer<NativeFunction<Void Function(Int)>> get nativeFunction =>
      _nc.nativeFunction;

  void _fire(int ok) {
    final completer = _completer;
    _completer = null;
    _nc.keepIsolateAlive = false;
    if (completer != null && !completer.isCompleted) completer.complete(ok != 0);
  }

  static final Queue<IntCompletionSlot> _free = Queue<IntCompletionSlot>();

  static IntCompletionSlot acquire(Completer<bool> completer) {
    final slot = _free.isEmpty ? IntCompletionSlot._() : _free.removeFirst();
    slot._completer = completer;
    slot._nc.keepIsolateAlive = true;
    return slot;
  }

  void release() {
    _completer = null;
    _nc.keepIsolateAlive = false;
    _free.addLast(this);
  }

  static int get pooledCount => _free.length;
}

// --- Port-delivery bindings --------------------------------------------------
// Hand-written rather than ffigen output (like minigpu_ffi_log_port.dart): the
// generated bindings file is regenerated from the headers and would lose them.
// The `@DefaultAsset` at the top of this library re-attaches them to the same
// `minigpu_ffi` code asset.

/// Element type codes for [mgpuReadAsyncToPort].
///
/// Deliberately NOT `BufferDataType.index`: the Dart and C++ `BufferDataType`
/// enums are ordered DIFFERENTLY, so an index crossing the FFI boundary is
/// ambiguous. These values match `MGPUElementType` in `src/include/minigpu.h`.
abstract final class MgpuElementType {
  static const int i8 = 0;
  static const int u8 = 1;
  static const int i16 = 2;
  static const int u16 = 3;
  static const int i32 = 4;
  static const int u32 = 5;
  static const int i64 = 6;
  static const int u64 = 7;
  static const int f32 = 8;
  static const int f64 = 9;
}

@Native<Void Function(Int64, Int64)>(symbol: 'mgpuInitializeContextAsyncToPort')
external void mgpuInitializeContextAsyncToPort(int port, int token);

@Native<Void Function(Pointer<ffi.MGPUContextHandle>, Int64, Int64)>(
  symbol: 'mgpuContextInitializeAsyncToPort',
)
external void mgpuContextInitializeAsyncToPort(
  Pointer<ffi.MGPUContextHandle> handle,
  int port,
  int token,
);

@Native<Void Function(Pointer<ffi.MGPUComputeShader>, Int, Int, Int, Int64, Int64)>(
  symbol: 'mgpuDispatchAsyncToPort',
)
external void mgpuDispatchAsyncToPort(
  Pointer<ffi.MGPUComputeShader> shader,
  int groupsX,
  int groupsY,
  int groupsZ,
  int port,
  int token,
);

@Native<
  Void Function(
    Pointer<ffi.MGPUBuffer>,
    Pointer<Void>,
    Size,
    Size,
    Int,
    Int64,
    Int64,
  )
>(symbol: 'mgpuReadAsyncToPort')
external void mgpuReadAsyncToPort(
  Pointer<ffi.MGPUBuffer> buffer,
  Pointer<Void> outputData,
  int elementCount,
  int elementOffset,
  int elementType,
  int port,
  int token,
);

@Native<
  Void Function(
    Pointer<ffi.MGPUBuffer>,
    Pointer<ffi.MGPUSharedOutputTexture>,
    Int64,
    Int64,
  )
>(symbol: 'mgpuCopyBufferToSharedOutputTextureAsyncToPort')
external void mgpuCopyBufferToSharedOutputTextureAsyncToPort(
  Pointer<ffi.MGPUBuffer> buf,
  Pointer<ffi.MGPUSharedOutputTexture> dst,
  int port,
  int token,
);

@Native<
  Void Function(
    Pointer<ffi.MGPUVideoTexture>,
    Pointer<ffi.MGPUSharedOutputTexture>,
    Int64,
    Int64,
  )
>(symbol: 'mgpuVideoTextureBGRAToRGBASharedOutputAsyncToPort')
external void mgpuVideoTextureBGRAToRGBASharedOutputAsyncToPort(
  Pointer<ffi.MGPUVideoTexture> src,
  Pointer<ffi.MGPUSharedOutputTexture> dst,
  int port,
  int token,
);

/// Blocks until every task already queued on the WebGPU worker thread has run.
///
/// Teardown ordering aid: destroying a resource an enqueued task still
/// references frees it under that task. No-op if called from the worker thread.
@Native<Void Function()>(symbol: 'mgpuDrainWorkQueue')
external void mgpuDrainWorkQueue();
