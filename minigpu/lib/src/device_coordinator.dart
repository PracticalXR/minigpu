part of 'minigpu.dart';

/// A device drain is never a deadline or an instruction to discard live owners.
enum GpuDeviceDrainPhase { accepting, draining, drained, quarantined }

/// Read-only, host-facing progress for the exact wrapper and device epoch.
final class GpuDeviceDrainSnapshot {
  GpuDeviceDrainSnapshot._(
    this.phase,
    this.deviceEpoch,
    this.retired,
    Iterable<String> pending,
    Map<String, Object> failures,
  ) : pending = List.unmodifiable(pending),
      failures = Map.unmodifiable(failures);

  final GpuDeviceDrainPhase phase;
  final int deviceEpoch, retired;
  final List<String> pending;
  final Map<String, Object> failures;
}

/// One registry per actual device wrapper, shared by its trusted consumers.
///
/// Register before asynchronous preparation. On loss or explicit device
/// destruction, admission closes, all consumers are fenced, and their actual
/// retirement futures are joined. Failed owners remain strongly retained.
/// Independent worker devices belong to their own registries; this registry
/// neither kills workers nor selects capture sources or replacement adapters.
final class GpuDeviceCoordinator {
  GpuDeviceCoordinator._(this._gpu);

  // A removed view may be the last application reference to an independent
  // device. Pending/failed cleanup must still retain its native owners.
  static final _retained = <GpuDeviceCoordinator>{};

  final Minigpu _gpu;
  final _owners = <GpuDeviceParticipant>{};
  Future<void>? _draining;
  bool _accepting = true;
  int _nextId = 0, _retired = 0;

  bool get isAccepting =>
      _accepting && !_gpu._closing && !_gpu._deviceInvalidated;

  GpuDeviceDrainSnapshot get snapshot {
    final failures = <String, Object>{};
    final pending = <String>[];
    for (final owner in _owners) {
      if (owner._failure case final failure?) {
        failures[owner.label] = failure;
      } else {
        pending.add(owner.label);
      }
    }
    return GpuDeviceDrainSnapshot._(
      failures.isNotEmpty
          ? GpuDeviceDrainPhase.quarantined
          : _accepting
          ? GpuDeviceDrainPhase.accepting
          : pending.isEmpty
          ? GpuDeviceDrainPhase.drained
          : GpuDeviceDrainPhase.draining,
      _gpu.deviceEpoch,
      _retired,
      pending,
      failures,
    );
  }

  /// [retainedOwner] can pin a preparation's mutable resource inventory before
  /// it has a long-lived owner object. It is released only on proven retirement.
  GpuDeviceParticipant register({
    required String label,
    required void Function() fence,
    required Future<void> Function() drain,
    Object? retainedOwner,
  }) {
    if (!isAccepting) {
      throw StateError('GPU device is not accepting new owners');
    }
    final owner = GpuDeviceParticipant._(
      this,
      '${++_nextId}: $label',
      _gpu.deviceEpoch + (_gpu.isInitialized ? 0 : 1),
      fence,
      drain,
      retainedOwner,
    );
    _owners.add(owner);
    return owner;
  }

  /// Coalesced strict receipt. There is no timeout-to-success and no reset.
  Future<void> drain() {
    final existing = _draining;
    if (existing != null) return existing;
    final done = Completer<void>();
    _draining = done.future;
    _retained.add(this);
    done.future.ignore();
    _accepting = false;
    final owners = _owners.toList(growable: false);
    for (final owner in owners) {
      owner._fence();
    }
    Future.wait<void>([
      for (final owner in owners) owner._drain(),
    ]).then((_) {
      _retained.remove(this);
      done.complete();
    }, onError: done.completeError);
    return done.future;
  }

  void _initialized(int epoch) {
    if (_owners.any((owner) => owner.deviceEpoch != epoch)) {
      throw StateError('Previous GPU owners have not retired');
    }
    // An initial device may finish opening while its already-registered
    // sources are retiring. That completion must not reopen admission.
    if (_gpu._retiring != null) return;
    _draining = null;
    _accepting = true;
  }
}

/// An owner cannot unregister itself merely because its widget disappeared.
/// Only successful completion of [observeRetirement] releases the registration.
final class GpuDeviceParticipant {
  GpuDeviceParticipant._(
    this._coordinator,
    this.label,
    this.deviceEpoch,
    this._fenceCallback,
    this._drainCallback,
    Object? retainedOwner,
  ) {
    if (retainedOwner != null) _retainedOwners.add(retainedOwner);
    _receipt.future.ignore();
  }

  final GpuDeviceCoordinator _coordinator;
  final String label;
  final int deviceEpoch;
  void Function()? _fenceCallback;
  Future<void> Function()? _drainCallback;
  final _retainedOwners = <Object>[];
  final _receipt = Completer<void>();
  bool _fenced = false, _observing = false;
  Object? _failure;
  StackTrace? _failureStack;

  /// Use the same strict future as ordinary owner disposal, including failed
  /// and pending disposal. This also covers owners removed before device loss.
  void observeRetirement(Future<void> actual) {
    if (_observing) {
      actual.ignore();
      return;
    }
    _observing = true;
    GpuDeviceCoordinator._retained.add(_coordinator);
    actual.then(
      (_) {
        if (_failure == null) {
          _coordinator._owners.remove(this);
        _coordinator._retired++;
        if (_coordinator._draining == null &&
            !_coordinator._owners.any((owner) => owner._observing)) {
          GpuDeviceCoordinator._retained.remove(_coordinator);
        }
          _fenceCallback = null;
        _drainCallback = null;
        _retainedOwners.clear();
          _receipt.complete();
        } else {
          _receipt.completeError(_failure!, _failureStack);
        }
      },
      onError: (Object error, StackTrace stack) {
        _failure ??= error;
        _failureStack ??= stack;
        _receipt.completeError(_failure!, _failureStack);
      },
    );
  }

  void _fence() {
    if (_fenced || _receipt.isCompleted) return;
    _fenced = true;
    try {
      _fenceCallback?.call();
    } catch (error, stack) {
      _failure ??= error;
      _failureStack ??= stack;
    }
  }

  Future<void> _drain() {
    if (!_observing) {
      observeRetirement(Future<void>.sync(_drainCallback!));
    }
    return _receipt.future;
  }
}
