// Native (dart:io platforms) implementation of [PersistentIsolateExecutor].
// Selected via the conditional export in `stage_executor.dart`; web builds
// get the stub in `stage_executor_isolate_stub.dart` instead, so importing
// the operation layer never drags `dart:isolate` semantics into a platform
// that cannot spawn isolates.
import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'stage_executor.dart' show StageExecutor, StageProcessor;

/// Runs the processor on a single long-lived worker isolate, keeping heavy
/// host-side compute (filtering, stateless encoding, etc.) off the isolate
/// that submits GPU work — so GPU dispatch for the next frame overlaps with
/// CPU work on the previous one.
///
/// Constraints:
///   * [processor] **must** be a top-level or static function. Closures that
///     capture state are not sendable to another isolate and will throw at
///     spawn time.
///   * Inputs/outputs are copied across the isolate boundary (Dart's standard
///     `TypedData` transfer). Zero-copy hand-off via `TransferableTypedData`
///     is a future optimisation tracked separately.
class PersistentIsolateExecutor extends StageExecutor {
  PersistentIsolateExecutor(this.processor);

  final StageProcessor processor;

  Isolate? _isolate;
  SendPort? _toWorker;
  ReceivePort? _fromWorker;
  Completer<SendPort>? _spawning;
  bool _disposed = false;

  int _seq = 0;
  final Map<int, Completer<Map<String, TypedData>>> _pending = {};

  @override
  String get backend => 'isolate';

  Future<void> _ensureSpawned() async {
    if (_disposed) {
      throw StateError('PersistentIsolateExecutor used after dispose()');
    }
    if (_toWorker != null) return;
    if (_spawning != null) {
      await _spawning!.future;
      return;
    }
    final spawning = Completer<SendPort>();
    _spawning = spawning;

    final fromWorker = ReceivePort();
    _fromWorker = fromWorker;
    fromWorker.listen(_onWorkerMessage);

    _isolate = await Isolate.spawn<_WorkerInit>(
      _workerMain,
      _WorkerInit(fromWorker.sendPort, processor),
    );
    _toWorker = await spawning.future;
  }

  void _onWorkerMessage(dynamic message) {
    if (message is SendPort) {
      // Handshake: worker published its inbound port.
      _spawning?.complete(message);
      return;
    }
    if (message is _WorkResult) {
      final completer = _pending.remove(message.id);
      if (completer == null) return;
      if (message.error != null) {
        completer.completeError(
          StateError('isolate stage failed: ${message.error}'),
          message.stack == null ? null : StackTrace.fromString(message.stack!),
        );
      } else {
        completer.complete(message.outputs!);
      }
    }
  }

  @override
  Future<Map<String, TypedData>> run(
    Map<String, TypedData> inputs,
    Map<String, List<int>> ranks,
    Map<String, dynamic> parameters,
  ) async {
    await _ensureSpawned();
    final id = _seq++;
    final completer = Completer<Map<String, TypedData>>();
    _pending[id] = completer;
    _toWorker!.send(_WorkRequest(id, inputs, ranks, parameters));
    return completer.future;
  }

  @override
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    _toWorker?.send(const _Shutdown());
    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _isolate = null;
    _toWorker = null;
    _fromWorker?.close();
    _fromWorker = null;
    for (final c in _pending.values) {
      if (!c.isCompleted) {
        c.completeError(StateError('executor disposed before completion'));
      }
    }
    _pending.clear();
  }
}

class _WorkerInit {
  const _WorkerInit(this.replyTo, this.processor);
  final SendPort replyTo;
  final StageProcessor processor;
}

class _WorkRequest {
  const _WorkRequest(this.id, this.inputs, this.ranks, this.parameters);
  final int id;
  final Map<String, TypedData> inputs;
  final Map<String, List<int>> ranks;
  final Map<String, dynamic> parameters;
}

class _WorkResult {
  const _WorkResult.ok(this.id, this.outputs) : error = null, stack = null;
  const _WorkResult.err(this.id, this.error, this.stack) : outputs = null;
  final int id;
  final Map<String, TypedData>? outputs;
  final String? error;
  final String? stack;
}

class _Shutdown {
  const _Shutdown();
}

void _workerMain(_WorkerInit init) {
  final processor = init.processor;
  final inbound = ReceivePort();
  // Publish our inbound port back to the owner.
  init.replyTo.send(inbound.sendPort);

  inbound.listen((dynamic message) async {
    if (message is _Shutdown) {
      inbound.close();
      return;
    }
    if (message is _WorkRequest) {
      try {
        final outputs = await processor(
          message.inputs,
          message.ranks,
          message.parameters,
        );
        init.replyTo.send(_WorkResult.ok(message.id, outputs));
      } catch (e, st) {
        init.replyTo.send(
          _WorkResult.err(message.id, e.toString(), st.toString()),
        );
      }
    }
  });
}
