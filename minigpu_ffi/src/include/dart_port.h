#pragma once

#include <cstdint>

/// DART NATIVE-PORT COMPLETIONS.
///
/// Every async entry point in this library used to take an `MGPUCallback` — a
/// raw function pointer the WebGPU worker thread calls when the work is done.
/// For a Dart embedder that pointer is a `NativeCallable` trampoline, and the
/// only way to release one is `close()`, which DELETES it. There is no way for
/// Dart to tell this library to forget a pointer it has already been handed,
/// so every close is a bet that native will not call again, and the VM's
/// penalty for losing the bet is not an exception:
///
///     runtime_entry.cc: error: Callback invoked after it has been deleted.
///
/// — an unconditional FATAL that takes down the whole process. Isolate
/// teardown (hot restart, a worker isolate exiting) deletes the trampolines
/// too, so no amount of Dart-side discipline covers it.
///
/// A Dart PORT has the property the function pointer lacks: posting to one
/// that is closed, or whose isolate is gone, is a defined, silent no-op that
/// returns false. It is also thread-safe from any thread, with or without an
/// isolate — which is what the WebGPU worker and Dawn's own threads are. The
/// log stream moved to a port for exactly this reason (see log.h); completions
/// follow.
///
/// WIRE FORMAT: one int64 per completion, `(token << 1) | ok`. The token is
/// allocated by Dart, is never reused, and identifies the operation; `ok` is
/// the success bit (always 1 for operations that cannot fail). A token Dart no
/// longer knows about is a map miss, i.e. a stale completion cannot resolve
/// somebody else's operation.
namespace mgpu {

/// Posts `(token << 1) | ok` to [port]. Safe from any thread. Silently does
/// nothing when the port is dead, when no Dart API was initialised, or in
/// builds without MINIGPU_HAVE_DART_DL (the Emscripten/web build).
void completionPost(int64_t port, int64_t token, bool ok);

/// True once mgpuInitDartApi has succeeded, i.e. completionPost can deliver.
/// Dart uses the mgpuInitDartApi return value directly; this is for native
/// callers that want to know whether to bother building a completion.
bool dartApiReady();

} // namespace mgpu
