/// Native-port delivery for the minigpu C++ library's log stream.
///
/// `mgpuSetLogCallback` installs a **process-global** function pointer. A Dart
/// [NativeCallable] registered there is owned by exactly one isolate; when that
/// isolate exits the VM deletes the trampoline while the native library still
/// holds the pointer. The next `mgpu::Logger` line — and Dawn logs from its own
/// worker threads, which carry no isolate at all — then aborts the entire
/// process:
///
///     runtime_entry.cc: ... Callback invoked after it has been deleted
///
/// A whole-suite `dart test` run hits this reliably, because every test file is
/// a separate isolate inside one VM process, and it takes down unrelated suites
/// downstream of minigpu as well.
///
/// Posting to a Dart port is defined, silent and thread-safe even after the
/// port closes, so the port is the supported Dart delivery path. The
/// function-pointer API stays in the C library for non-Dart embedders, which
/// own their own function's lifetime.
///
/// These bindings live outside `minigpu_ffi_bindings.dart` because that file is
/// ffigen output; the `@DefaultAsset` below re-attaches them to the same
/// `minigpu_ffi` code asset.
@DefaultAsset('package:minigpu_ffi/minigpu_ffi_bindings.dart')
library;

import 'dart:ffi';

/// Initialises the native library's vendored Dart dynamic-linking API.
/// Returns 0 on success, non-zero when the library was built without it
/// (the Emscripten/web build) or the SDK version does not match. Idempotent.
@Native<Int Function(Pointer<Void>)>(symbol: 'mgpuInitDartApi')
external int mgpuInitDartApi(Pointer<Void> initializeApiDLData);

/// Routes formatted native log lines to [port]. Pass 0 to stop delivery.
///
/// Messages arrive as `[int32 level, Uint8List utf8Message]`. The bytes are
/// copied into the message by the native side — nothing to free, and
/// `mgpuFreeLogMessage` does not apply to this path.
///
/// PROCESS-GLOBAL, LAST WRITER WINS across isolates.
@Native<Void Function(Int64)>(symbol: 'mgpuSetLogPort')
external void mgpuSetLogPort(int port);
