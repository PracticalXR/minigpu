// lib/src/operation/stage_executor.dart
//
// Pluggable execution backends for CPU pipeline stages.
//
// A [CPUStageOperation] no longer hard-codes *how* its host-side processor
// runs. Instead it delegates to a [StageExecutor]. This is the seam that lets
// the same stage definition run:
//
//   * inline on the calling isolate          -> [InlineExecutor]   (default)
//   * on a long-lived worker isolate          -> [PersistentIsolateExecutor]
//     (native only — the web build gets a stub that throws; see the
//     conditional export at the bottom of this file)
//   * (future) compiled to WASM on the web / interpreted via a scripting tier
//
// The executor only ever sees raw host data (`TypedData`), never GPU tensors,
// so backends are free to live on another isolate / thread / VM.
import 'dart:async';
import 'dart:typed_data';

// PersistentIsolateExecutor: real implementation on dart:io platforms, a
// throwing stub on the web (no isolate spawning there).  This keeps
// `dart:isolate` out of web builds entirely.
export 'stage_executor_isolate_stub.dart'
    if (dart.library.io) 'stage_executor_isolate_io.dart'
    show PersistentIsolateExecutor;

/// CPU processor function signature - raw data + per-input rank (shape) info.
///
/// This mirrors the typedef in `cpu_stage_operation.dart`; it is duplicated
/// here so executors have no dependency on the stage layer.
typedef StageProcessor =
    Future<Map<String, TypedData>> Function(
      Map<String, TypedData> inputs,
      Map<String, List<int>> ranks,
      Map<String, dynamic> parameters,
    );

/// Strategy that actually runs a CPU stage's host-side work.
///
/// Implementations must be safe to call repeatedly (once per frame in a
/// streaming pipeline). [dispose] is invoked when the owning stage is torn
/// down.
abstract class StageExecutor {
  const StageExecutor();

  /// Human-readable backend name (for diagnostics / events).
  String get backend;

  /// Run the stage for a single invocation.
  Future<Map<String, TypedData>> run(
    Map<String, TypedData> inputs,
    Map<String, List<int>> ranks,
    Map<String, dynamic> parameters,
  );

  /// Release any backing resources (worker isolates, etc.). Safe to call
  /// multiple times.
  FutureOr<void> dispose() {}
}

/// Runs the processor directly on the calling isolate (today's behaviour).
///
/// This is the only correct choice for **stateful** processors — e.g. a
/// closed-loop video encoder that mutates reference-frame / entropy state
/// captured in its closure — because a worker isolate would run on a copy
/// of that state.
class InlineExecutor extends StageExecutor {
  const InlineExecutor(this.processor);

  final StageProcessor processor;

  @override
  String get backend => 'inline';

  @override
  Future<Map<String, TypedData>> run(
    Map<String, TypedData> inputs,
    Map<String, List<int>> ranks,
    Map<String, dynamic> parameters,
  ) => processor(inputs, ranks, parameters);
}
