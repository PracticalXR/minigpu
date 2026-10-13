import 'dart:async';
import 'package:minigpu/minigpu.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:test/test.dart';

class _Device extends MinigpuPlatform {
  bool live = false;
  int starts = 0, detaches = 0;
  void Function()? lost;
  Completer<void>? replacementGate;
  Completer<void> replacing = Completer<void>();
  Object? replacementFailure;
  bool loseReplacement = false;
  int replacementAttempts = 0;
  Completer<void> finalContext = Completer<void>();
  @override
  DeviceAdapterIdentity? get creationAdapterIdentity =>
      const DeviceAdapterIdentity(backend: 1, kind: 1, low: 27, high: 0);
  @override
  Future<void> initializeContext() async {
    live = true;
    starts++;
  }

  @override
  Future<void> initializeContextForRecovery(
    DeviceAdapterIdentity expected,
  ) async {
    expect(expected, creationAdapterIdentity);
    replacementAttempts++;
    replacing.complete();
    await replacementGate?.future;
    final failure = replacementFailure;
    if (failure != null) throw failure;
    finalContext = Completer<void>();
    await initializeContext();
    if (loseReplacement) live = false;
  }

  @override
  Future<void> observeContextRetirement({required int deviceGeneration}) =>
      finalContext.future;
  @override
  Future<void> destroyContext() async {
    detaches++;
    live = false;
    finalContext.complete();
  }

  @override
  DeviceBufferLimits? get currentDeviceBufferLimits => !live
      ? null
      : DeviceBufferLimits(
          deviceGeneration: starts,
          maxBufferSize: 1024,
          maxStorageBufferBindingSize: 1024,
        );
  @override
  void setDeviceLostCallback(void Function()? callback) => lost = callback;
  void lose() {
    live = false;
    lost!();
  }

  @override
  PlatformBuffer createBuffer(int size, BufferDataType type) =>
      throw UnimplementedError();
  @override
  PlatformComputeShader createComputeShader() => throw UnimplementedError();
}

void main() {
  test('close still retires a replacement lost immediately after initialization', () async {
    final backend = _Device()..loseReplacement = true;
    final gpu = Minigpu.withPlatform(backend);
    await gpu.init();
    final epoch = gpu.deviceEpoch;
    backend.lose();
    await expectLater(gpu.recoverLostDevice(deviceEpoch: epoch), throwsStateError);
    expect(gpu.deviceEpoch, epoch + 1);
    expect(gpu.isInitialized, isTrue);
    expect(gpu.recoveryStatus.canRetry, isFalse);
    await gpu.destroy();
    expect(gpu.isInitialized, isFalse);
    expect(backend.detaches, 2);
    expect(backend.finalContext.isCompleted, isTrue);
  });

  test(
    'failed external retirement never detaches or replaces the GPU',
    () async {
      final backend = _Device();
      final gpu = Minigpu.withPlatform(backend);
      await gpu.init();
      final reader = Completer<void>();
      gpu.deviceCoordinator.register(
        label: 'external reader',
        fence: () {},
        drain: () => reader.future,
      );
      backend.lose();
      final recovery = gpu.recoverLostDevice(deviceEpoch: gpu.deviceEpoch);
      final failure = expectLater(recovery, throwsStateError);
      reader.completeError(StateError('reader completion unknown'));
      await failure;
      expect(
        gpu.recoverLostDevice(deviceEpoch: gpu.deviceEpoch),
        same(recovery),
      );
      await expectLater(gpu.init(), throwsStateError);
      expect(backend.detaches, 0);
      expect(backend.starts, 1);
      expect(gpu.recoveryStatus.phase, GpuRecoveryPhase.failed);
      expect(gpu.recoveryStatus.canRetry, isFalse);
      await expectLater(
        gpu.retryLostDevice(
          deviceEpoch: gpu.deviceEpoch,
          failedAttempt: gpu.recoveryStatus.attempt,
        ),
        throwsStateError,
      );
      expect(
        gpu.deviceCoordinator.snapshot.phase,
        GpuDeviceDrainPhase.quarantined,
      );
    },
  );

  test(
    'only explicit current-attempt retry replaces a discarded candidate',
    () async {
      final backend = _Device();
      backend.replacementFailure = DeviceReplacementDiscarded(
        expectedAdapter: backend.creationAdapterIdentity!,
        message: 'Candidate discarded',
      );
      final gpu = Minigpu.withPlatform(backend);
      await gpu.init();
      final epoch = gpu.deviceEpoch;
      backend.lose();
      final first = gpu.recoverLostDevice(deviceEpoch: epoch);
      await expectLater(first, throwsStateError);
      final failure = gpu.recoveryStatus;
      expect(failure.canRetry, isTrue);
      expect(failure.attempt, 1);
      await expectLater(gpu.init(), throwsStateError);
      expect(gpu.recoverLostDevice(deviceEpoch: epoch), same(first));
      await expectLater(
        gpu.retryLostDevice(deviceEpoch: epoch + 1, failedAttempt: 1),
        throwsStateError,
      );
      await expectLater(
        gpu.retryLostDevice(deviceEpoch: epoch, failedAttempt: 0),
        throwsStateError,
      );
      expect(backend.replacementAttempts, 1);

      backend.replacing = Completer<void>();
      final second = gpu.retryLostDevice(
        deviceEpoch: epoch,
        failedAttempt: failure.attempt,
      );
      expect(
        gpu.retryLostDevice(deviceEpoch: epoch, failedAttempt: 1),
        same(second),
      );
      await expectLater(second, throwsStateError);
      expect(gpu.recoveryStatus.attempt, 2);
      expect(gpu.recoveryStatus.canRetry, isTrue);
      // The first failure cannot start a third attempt.
      expect(
        gpu.retryLostDevice(deviceEpoch: epoch, failedAttempt: 1),
        same(second),
      );
      expect(backend.replacementAttempts, 2);

      backend.replacementFailure = null;
      backend.replacementGate = Completer<void>();
      backend.replacing = Completer<void>();
      final third = gpu.retryLostDevice(deviceEpoch: epoch, failedAttempt: 2);
      await backend.replacing.future;
      await expectLater(
        gpu.retryLostDevice(deviceEpoch: epoch, failedAttempt: 1),
        throwsStateError,
      );
      final joinedInit = gpu.init();
      backend.replacementGate!.complete();
      expect(await third, epoch + 1);
      await joinedInit;
      expect(gpu.recoveryStatus.canRetry, isFalse);
      expect(gpu.recoveryStatus.phase, GpuRecoveryPhase.ready);
      expect(
        backend.detaches,
        1,
      ); // Never retires the already-retired original again.
      expect(backend.starts, 2);
      expect(backend.replacementAttempts, 3);
      expect(
        gpu.deviceCoordinator.snapshot.phase,
        GpuDeviceDrainPhase.accepting,
      );
      expect(
        gpu.retryLostDevice(deviceEpoch: epoch, failedAttempt: 2),
        same(third),
      );
      await gpu.destroy();
    },
  );

  test(
    'unknown or mismatched candidate receipts cannot authorize retry',
    () async {
      for (final error in <Object>[
        StateError('Candidate cleanup unknown'),
        DeviceReplacementDiscarded(
          expectedAdapter: const DeviceAdapterIdentity(
            backend: 1,
            kind: 1,
            low: 28,
            high: 0,
          ),
          message: 'Receipt for a different adapter',
        ),
      ]) {
        final backend = _Device()..replacementFailure = error;
        final gpu = Minigpu.withPlatform(backend);
        await gpu.init();
        backend.lose();
        await expectLater(
          gpu.recoverLostDevice(deviceEpoch: gpu.deviceEpoch),
          throwsStateError,
        );
        expect(gpu.recoveryStatus.canRetry, isFalse);
        await expectLater(
          gpu.retryLostDevice(deviceEpoch: gpu.deviceEpoch, failedAttempt: 1),
          throwsStateError,
        );
        await expectLater(gpu.init(), throwsStateError);
        await expectLater(gpu.destroy(), throwsStateError);
        expect(backend.replacementAttempts, 1);
      }
    },
  );

  test(
    'close during failed retry cancels only after confirmed candidate discard',
    () async {
      final backend = _Device();
      backend.replacementFailure = DeviceReplacementDiscarded(
        expectedAdapter: backend.creationAdapterIdentity!,
        message: 'Candidate discarded',
      );
      final gpu = Minigpu.withPlatform(backend);
      await gpu.init();
      backend.lose();
      await expectLater(
        gpu.recoverLostDevice(deviceEpoch: gpu.deviceEpoch),
        throwsStateError,
      );
      backend.replacementGate = Completer<void>();
      backend.replacing = Completer<void>();
      final retry = gpu.retryLostDevice(
        deviceEpoch: gpu.deviceEpoch,
        failedAttempt: 1,
      );
      final rejected = expectLater(retry, throwsStateError);
      await backend.replacing.future;
      final close = gpu.destroy();
      expect(gpu.recoveryStatus.phase, GpuRecoveryPhase.replacing);
      backend.replacementGate!.complete();
      await close;
      await rejected;
      expect(gpu.recoveryStatus.phase, GpuRecoveryPhase.cancelled);
      expect(gpu.recoveryStatus.canRetry, isFalse);
      await gpu.destroy();
      await expectLater(
        gpu.retryLostDevice(deviceEpoch: gpu.deviceEpoch, failedAttempt: 2),
        throwsStateError,
      );
      expect(backend.replacementAttempts, 2);
    },
  );

  test(
    'close revokes a retry already offered for a discarded candidate',
    () async {
      final backend = _Device();
      backend.replacementFailure = DeviceReplacementDiscarded(
        expectedAdapter: backend.creationAdapterIdentity!,
        message: 'Candidate discarded',
      );
      final gpu = Minigpu.withPlatform(backend);
      await gpu.init();
      backend.lose();
      await expectLater(
        gpu.recoverLostDevice(deviceEpoch: gpu.deviceEpoch),
        throwsStateError,
      );
      gpu.destroySync();
      await gpu.destroy();
      expect(gpu.recoveryStatus.phase, GpuRecoveryPhase.cancelled);
      await expectLater(
        gpu.retryLostDevice(deviceEpoch: gpu.deviceEpoch, failedAttempt: 1),
        throwsStateError,
      );
      expect(backend.replacementAttempts, 1);
    },
  );

  test(
    'explicit close during replacement drains the candidate without restoring views',
    () async {
      final backend = _Device()..replacementGate = Completer<void>();
      final gpu = Minigpu.withPlatform(backend);
      await gpu.init();
      final phases = <GpuRecoveryPhase>[];
      gpu.onRecoveryChanged((state) => phases.add(state.phase));
      backend.lose();
      final recovery = gpu.recoverLostDevice(deviceEpoch: gpu.deviceEpoch);
      final failure = expectLater(recovery, throwsStateError);
      await backend.replacing.future;
      final closing = gpu.destroy();
      backend.replacementGate!.complete();
      await closing;
      await failure;
      expect(backend.detaches, 2);
      expect(gpu.isInitialized, isFalse);
      expect(phases, [
        GpuRecoveryPhase.draining,
        GpuRecoveryPhase.replacing,
        GpuRecoveryPhase.cancelled,
      ]);
    },
  );

  test(
    'healthy or stale recovery requests cannot reset the host device',
    () async {
      final backend = _Device();
      final gpu = Minigpu.withPlatform(backend);
      await gpu.init();
      await expectLater(
        gpu.recoverLostDevice(deviceEpoch: gpu.deviceEpoch),
        throwsStateError,
      );
      backend.lose();
      await expectLater(
        gpu.recoverLostDevice(deviceEpoch: gpu.deviceEpoch + 1),
        throwsStateError,
      );
      expect(backend.detaches, 0);
      expect(backend.starts, 1);
      await gpu.destroy();
    },
  );
}
