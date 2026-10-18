import 'dart:async';

import 'package:minigpu/minigpu.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:test/test.dart';

class _LimitsPlatform extends MinigpuPlatform {
  int generation = 1;
  bool live = false;
  void Function()? lost;
  Future<void>? work;
  Future<void>? opening;
  Future<void>? abortProof;
  int abortRequests = 0;

  @override
  Future<void> retireLostDevice({required int deviceGeneration}) {
    expect(deviceGeneration, generation);
    abortRequests++;
    return abortProof ?? Future<void>.value();
  }

  @override
  Future<void> initializeContext() async {
    await opening;
    live = true;
  }

  @override
  Future<void> destroyContext() async => live = false;

  @override
  DeviceBufferLimits? get currentDeviceBufferLimits => live
      ? DeviceBufferLimits(
          deviceGeneration: generation,
          maxBufferSize: 268435456,
          maxStorageBufferBindingSize: 134217728,
        )
      : null;

  @override
  void setDeviceLostCallback(void Function()? callback) => lost = callback;

  @override
  Future<void> waitForSubmittedWork() => work ?? Future<void>.value();

  @override
  PlatformBuffer createBuffer(int bufferSize, BufferDataType dataType) =>
      throw UnimplementedError();

  @override
  PlatformComputeShader createComputeShader() => throw UnimplementedError();
}

void main() {
  test('GPU retirement cannot forgive healthy errors or failed abort proof', () async {
    final platform = _LimitsPlatform();
    final gpu = Minigpu.withPlatform(platform);
    await gpu.init();
    final epoch = gpu.deviceEpoch;
    final queue = Completer<void>();
    platform.work = queue.future;
    final healthyFailure = expectLater(
      gpu.retireSubmittedWork(deviceEpoch: epoch), throwsStateError);
    queue.completeError(StateError('queue completion unavailable'));
    await healthyFailure;
    expect(platform.abortRequests, 0);
    expect(gpu.isDeviceCurrent, isTrue);

    final proof = Completer<void>();
    platform.abortProof = proof.future;
    platform.live = false;
    platform.lost!();
    final first = expectLater(
      gpu.retireSubmittedWork(deviceEpoch: epoch), throwsStateError);
    final second = expectLater(
      gpu.retireSubmittedWork(deviceEpoch: epoch), throwsStateError);
    await Future<void>.delayed(Duration.zero);
    expect(platform.abortRequests, 1);
    proof.completeError(StateError('device destruction unconfirmed'));
    await Future.wait([first, second]);
    await expectLater(gpu.retireSubmittedWork(deviceEpoch: epoch), throwsStateError);
    expect(platform.abortRequests, 1, reason: 'Failed retirement is not retried into success');
    await gpu.destroy(); // The fake owns no GPU allocations or external readers.
  });

  test('initialization finishing during drain cannot reopen resource admission', () async {
    final opening = Completer<void>();
    final platform = _LimitsPlatform()..opening = opening.future;
    final gpu = Minigpu.withPlatform(platform);
    final preparation = Completer<void>();
    gpu.deviceCoordinator.register(label: 'preparing source', fence: () {},
      drain: () => preparation.future);
    final started = gpu.init();
    final retired = gpu.destroy();
    opening.complete();
    await started;
    expect(gpu.deviceCoordinator.isAccepting, isFalse);
    expect(() => gpu.createBuffer(4, BufferDataType.uint32), throwsStateError);
    preparation.complete();
    await retired;
    expect(platform.live, isFalse);
    await gpu.init();
    expect(gpu.deviceCoordinator.isAccepting, isTrue);
    expect(gpu.deviceEpoch, 2);
    await gpu.destroy();
  });

  test(
    'device destruction fences every owner and joins real retirement',
    () async {
      final platform = _LimitsPlatform();
      final gpu = Minigpu.withPlatform(platform);
      await gpu.init();
      final coordinator = gpu.deviceCoordinator;
      final completion = Completer<void>();
      final events = <String>[];
      coordinator.register(
        label: 'presenter',
        fence: () => events.add('fence A'),
        drain: () {
          events.add('drain A');
          return completion.future;
        },
      );
      coordinator.register(
        label: 'source',
        fence: () => events.add('fence B'),
        drain: () async => events.add('drain B'),
      );
      final retired = gpu.destroy();
      expect(identical(retired, gpu.destroy()), isTrue);
      expect(events, ['fence A', 'fence B', 'drain A', 'drain B']);
      expect(platform.live, isTrue);
      expect(coordinator.isAccepting, isFalse);
      expect(
        () => gpu.createBuffer(4, BufferDataType.uint32),
        throwsStateError,
      );
      await gpu.waitForSubmittedWork(); // Active readers can still finish.
      await Future<void>.delayed(Duration.zero);
      expect(coordinator.snapshot.pending, hasLength(1));
      completion.complete();
      await retired;
      expect(platform.live, isFalse);
      expect(coordinator.snapshot.phase, GpuDeviceDrainPhase.drained);
      await gpu.init();
      expect(coordinator.isAccepting, isTrue);
      expect(gpu.deviceEpoch, 2);
      await gpu.destroy();
    },
  );

  test(
    'ordinary failed disposal remains owned through loss and repeated reset',
    () async {
      final platform = _LimitsPlatform();
      final gpu = Minigpu.withPlatform(platform);
      await gpu.init();
      final other = Minigpu.withPlatform(_LimitsPlatform());
      await other.init();
      var independentStopped = false;
      other.deviceCoordinator.register(
        label: 'other context',
        fence: () => independentStopped = true,
        drain: () async {},
      );
      final completion = Completer<void>();
      final owner = gpu.deviceCoordinator.register(
        label: 'retired view',
        fence: () {},
        drain: () => completion.future,
      );
      owner.observeRetirement(completion.future);
      platform.lost!();
      completion.completeError(StateError('borrowed storage was not released'));
      await expectLater(gpu.deviceCoordinator.drain(), throwsStateError);
      final retired = gpu.destroy();
      await expectLater(retired, throwsStateError);
      expect(identical(retired, gpu.destroy()), isTrue);
      await expectLater(gpu.init(), throwsStateError);
      expect(
        platform.live,
        isTrue,
        reason: 'No backend destruction on failed drain',
      );
      expect(gpu.deviceCoordinator.snapshot.failures, hasLength(1));
      expect(independentStopped, isFalse);
      expect(other.isDeviceCurrent, isTrue);
      await other.destroy();
    },
  );

  test('reentrant fencing and a faulty owner do not skip peers', () async {
    final gpu = Minigpu.withPlatform(_LimitsPlatform());
    await gpu.init();
    Future<void>? reentrant;
    var peerRetired = false;
    gpu.deviceCoordinator.register(
      label: 'broken fence',
      fence: () {
        reentrant = gpu.destroy();
        throw StateError('fence failed');
      },
      drain: () async {},
    );
    gpu.deviceCoordinator.register(
      label: 'peer',
      fence: () {},
      drain: () async {
        peerRetired = true;
      },
    );
    final stopped = gpu.destroy();
    expect(reentrant, same(stopped));
    await expectLater(stopped, throwsStateError);
    expect(peerRetired, isTrue);
    expect(gpu.deviceCoordinator.snapshot.retired, 1);
    expect(gpu.deviceCoordinator.snapshot.failures, hasLength(1));
  });

  test(
    'loss subscriptions fence peers despite observer failure and reentry',
    () async {
      final platform = _LimitsPlatform();
      final gpu = Minigpu.withPlatform(platform);
      await gpu.init();
      final firstCallback = platform.lost!;
      final seen = <String>[];
      final error = Completer<Object>();
      void Function()? removeLater;
      void Function()? removeAdded;
      gpu.setDeviceLostCallback((epoch) {
        seen.add('legacy:$epoch');
        expect(gpu.isDeviceCurrent, isFalse);
        throw StateError('faulty observer');
      });
      final removeFirst = gpu.onDeviceLost((epoch) {
        seen.add('first:$epoch');
        removeLater?.call();
        removeAdded ??= gpu.onDeviceLost((next) => seen.add('added:$next'));
        platform.lost
            ?.call(); // Duplicate/reentrant platform notice is ignored.
      });
      removeLater = gpu.onDeviceLost((epoch) => seen.add('removed:$epoch'));
      final removePeer = gpu.onDeviceLost((epoch) => seen.add('peer:$epoch'));
      runZonedGuarded(
        () => firstCallback(),
        (caught, _) => error.complete(caught),
      );
      expect(seen, ['legacy:1', 'first:1', 'peer:1']);
      expect(await error.future, isStateError);

      gpu.setDeviceLostCallback(null);
      removeFirst();
      removeFirst();
      await gpu.destroy();
      platform.generation++;
      await gpu.init();
      firstCallback(); // A late old-backend callback cannot retire its successor.
      expect(gpu.isDeviceCurrent, isTrue);
      platform.lost!();
      expect(seen, ['legacy:1', 'first:1', 'peer:1', 'peer:2', 'added:2']);
      removeAdded!();
      removePeer();
      await gpu.destroy();
    },
  );

  test(
    'a late queue completion cannot certify a lost or replacement device',
    () async {
      final platform = _LimitsPlatform();
      final gpu = Minigpu.withPlatform(platform);
      await gpu.init();
      final submitted = Completer<void>();
      platform.work = submitted.future;
      final completion = gpu.waitForSubmittedWork();
      final refused = expectLater(completion, throwsStateError);
      platform.lost!();
      await expectLater(gpu.waitForSubmittedWork(), throwsStateError);
      await gpu.destroy();
      platform.generation++;
      await gpu.init();
      submitted.complete();
      await refused;
      platform.work = null;
      await gpu.waitForSubmittedWork();
      await gpu.destroy();
    },
  );

  test(
    'limits are bound to an initialized device epoch and retire on loss',
    () async {
      final platform = _LimitsPlatform();
      final gpu = Minigpu.withPlatform(platform);
      expect(gpu.currentDeviceBufferLimits, isNull);
      expect(gpu.isDeviceCurrent, isFalse);

      await gpu.init();
      final first = gpu.currentDeviceBufferLimits;
      expect(first?.deviceEpoch, 1);
      expect(first?.maxBufferSize, 268435456);
      expect(first?.maxStorageBufferBindingSize, 134217728);
      expect(gpu.isDeviceCurrent, isTrue);

      final losses = <int>[];
      gpu.setDeviceLostCallback(losses.add);
      platform.live = false;
      platform.lost?.call();
      platform.lost?.call();
      expect(losses, [1]);
      expect(gpu.currentDeviceBufferLimits, isNull);

      // A native context can recover internally; its replacement is still not
      // part of this wrapper's epoch until an explicit teardown and init.
      platform.live = true;
      platform.generation++;
      expect(gpu.isDeviceCurrent, isFalse);
      await gpu.destroy();

      await gpu.init();
      expect(gpu.currentDeviceBufferLimits?.deviceEpoch, 2);
      expect(gpu.isDeviceCurrent, isTrue);
      await gpu.destroy();
      expect(gpu.currentDeviceBufferLimits, isNull);
    },
  );

  test(
    'silent native device replacement invalidates the wrapper epoch',
    () async {
      final platform = _LimitsPlatform();
      final gpu = Minigpu.withPlatform(platform);
      await gpu.init();
      final losses = <int>[];
      gpu.setDeviceLostCallback(losses.add);
      platform.generation++;
      expect(gpu.currentDeviceBufferLimits, isNull);
      expect(losses, [1]);
      expect(gpu.isDeviceCurrent, isFalse);
      await gpu.destroy();
    },
  );
}
