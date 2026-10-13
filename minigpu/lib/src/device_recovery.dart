part of 'minigpu.dart';

enum GpuRecoveryPhase { idle, draining, replacing, ready, failed, cancelled }

/// Host-local progress. A failure is retained, never a timeout-to-success or
/// permission to forget readers. This contains no room or source authorization.
final class GpuRecoverySnapshot {
  const GpuRecoverySnapshot(
    this.phase, {
    required this.deviceEpoch,
    this.attempt = 0,
    this.canRetry = false,
    this.replacementEpoch,
    this.error,
  });
  final GpuRecoveryPhase phase;
  final int deviceEpoch;

  /// Monotonic within the lost epoch; used to fence stale retry actions.
  final int attempt;

  /// True only after the old device and a failed candidate fully retired.
  final bool canRetry;
  final int? replacementEpoch;
  final Object? error;
}

final class _GpuDeviceRecovery {
  _GpuDeviceRecovery(this.gpu, this.epoch, this.identity)
    : attempt = 1,
      _oldRetired = false;

  _GpuDeviceRecovery.retry(_GpuDeviceRecovery previous)
    : gpu = previous.gpu,
      epoch = previous.epoch,
      identity = previous.identity,
      attempt = previous.attempt + 1,
      _oldRetired = true;
  final Minigpu gpu;
  final int epoch;
  final DeviceRecoveryTarget? identity;
  final int attempt;
  bool _oldRetired;
  bool _candidateDiscarded = false;
  final done = Completer<int>()..future.ignore();
  late GpuRecoverySnapshot snapshot = GpuRecoverySnapshot(
    GpuRecoveryPhase.idle,
    deviceEpoch: epoch,
    attempt: attempt,
  );
  bool cancelled = false;
  bool get pending => !done.isCompleted;
  bool get failed => snapshot.phase == GpuRecoveryPhase.failed;

  void _report(GpuRecoveryPhase phase, {Object? error, int? replacement}) {
    snapshot = GpuRecoverySnapshot(
      phase,
      deviceEpoch: epoch,
      attempt: attempt,
      canRetry:
          phase == GpuRecoveryPhase.failed &&
          !cancelled &&
          _oldRetired &&
          _candidateDiscarded,
      replacementEpoch: replacement,
      error: error,
    );
    for (final entry in gpu._recoveryListeners.entries.toList(
      growable: false,
    )) {
      if (!gpu._recoveryListeners.containsKey(entry.key)) continue;
      try {
        entry.value(snapshot);
      } catch (error, stack) {
        // Diagnostics cannot turn actual retirement into failure or prevent
        // other views from observing that their old outputs are unavailable.
        scheduleMicrotask(() => Error.throwWithStackTrace(error, stack));
      }
    }
  }

  Future<void> _detach() {
    gpu._recoveryDetach = true;
    try {
      return gpu.destroy();
    } finally {
      gpu._recoveryDetach = false;
    }
  }

  Future<void> cancelAndDrain() async {
    cancelled = true;
    try {
      await done.future;
    } catch (_) {
      if (snapshot.phase == GpuRecoveryPhase.cancelled) return;
      if (!_oldRetired || !_candidateDiscarded) rethrow;
      _report(GpuRecoveryPhase.cancelled);
    }
  }

  void start() {
    _run().then(
      (value) {
        if (!done.isCompleted) done.complete(value);
      },
      onError: (Object error, StackTrace stack) {
        // Make explicit retry from a failure listener join a completed attempt.
        done.completeError(error, stack);
        if (snapshot.phase != GpuRecoveryPhase.cancelled) {
          _report(GpuRecoveryPhase.failed, error: error);
        }
      },
    );
  }

  Future<int> _run() async {
    final original = identity;
    if (original == null) {
      throw UnsupportedError('This backend cannot prove device replacement safety');
    }
    if (!_oldRetired) {
      _report(GpuRecoveryPhase.draining);
      // Register BEFORE returning our attachment. This receipt adds no owner.
      final contextRetired = gpu.observeContextRetirement(deviceEpoch: epoch);
      contextRetired.ignore();
      await _detach(); // coordinator, worker, host and external readers
      await contextRetired;
      _oldRetired = true;
    }
    if (cancelled) {
      _report(GpuRecoveryPhase.cancelled);
      throw StateError('GPU recovery was cancelled after retirement');
    }
    _report(GpuRecoveryPhase.replacing);
    gpu._retiring = null; // only the successful old receipt can reach here
    final initialized = Completer<void>();
    gpu._initializing = initialized.future;
    gpu
        ._completeInitialization(initialized, recoveryTarget: original)
        .ignore();
    try {
      await initialized.future;
    } catch (error) {
      _candidateDiscarded =
          error is DeviceRecoveryDiscarded &&
          error.target == original &&
          !gpu.isInitialized &&
          gpu.deviceEpoch == epoch;
      if (cancelled && _candidateDiscarded) {
        _report(GpuRecoveryPhase.cancelled);
      }
      rethrow;
    }
    if (cancelled) {
      final retired = gpu.observeContextRetirement(
        deviceEpoch: gpu.deviceEpoch,
      );
      retired.ignore();
      await _detach();
      await retired;
      _report(GpuRecoveryPhase.cancelled);
      throw StateError('GPU recovery was cancelled during replacement');
    }
    if (!gpu.isDeviceCurrent || gpu.recoveryTarget != original) {
      throw StateError(
        'Replacement GPU is unavailable or its recovery target changed',
      );
    }
    final replacement = gpu.deviceEpoch;
    // Complete the operation before notifying views, so their ordinary init
    // joins a finished recovery and cannot deadlock their reprepare callback.
    done.complete(replacement);
    _report(GpuRecoveryPhase.ready, replacement: replacement);
    return replacement;
  }
}
