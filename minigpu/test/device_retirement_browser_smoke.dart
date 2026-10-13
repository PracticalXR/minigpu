// Public Dart/Wasm + WebGPU loss, retirement and replacement. Serve the packaged
// backend assets beside the compiled entry and read deviceRetirementResult.
import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart'
    show MinigpuPlatform;
import 'package:minigpu_web/bindings/minigpu_bindings.dart' as web;
import 'package:minigpu_web/minigpu_web.dart' show MinigpuWeb;

void check(bool value, String message) {
  if (!value) throw StateError(message);
}

Future<void> refuses(Future<void> Function() operation, String label) async {
  try {
    await operation();
  } on StateError {
    return;
  }
  throw StateError('$label did not refuse');
}

Future<void> availabilityIsNotLoss() async {
  // A second real module attachment observes a transient handle-lookup fault.
  // It must not destroy the first owner's still-healthy GPUDevice.
  final observer = MinigpuWeb.createForTest();
  await observer.initializeContext();
  final generation = observer.currentDeviceBufferLimits!.deviceGeneration;
  final module = globalContext.getProperty<JSObject>('MinigpuModule'.toJS);
  final key = '_mgpuGetWGPUDeviceHandle'.toJS;
  final original = module.getProperty<JSFunction>(key);
  try {
    module.setProperty(key, (() => 0.toJS).toJS);
    check(
      observer.currentDeviceBufferLimits == null,
      'Unavailable device lookup did not fence readiness',
    );
    await refuses(
      () => observer.retireLostDevice(deviceGeneration: generation),
      'Lookup failure without an actual loss receipt',
    );
  } finally {
    module.setProperty(key, original);
    await observer.destroyContext();
  }
}

Future<int> rejectedCandidateCanRetry(Minigpu gpu) async {
  final epoch = gpu.deviceEpoch, target = gpu.recoveryTarget;
  final module = globalContext.getProperty<JSObject>('MinigpuModule'.toJS);
  final key = '_mgpuWebObserveContextRetirement'.toJS;
  final original = module.getProperty<JSFunction>(key);
  JSObject? candidate;
  try {
    // Fault only the actual candidate's reported capabilities, after its real
    // module observation is captured. The production admission/retirement path
    // must discard that GPU, not label an init exception as safe to retry.
    module.setProperty(
      key,
      ((JSNumber handle) {
        final token = original.callAsFunction(module, handle);
        candidate = web.mgpuGetOwnDeviceObject()!;
        final limits = JSObject()
          ..setProperty('maxBufferSize'.toJS, 4.toJS)
          ..setProperty('maxStorageBufferBindingSize'.toJS, 4.toJS)
          ..setProperty('maxTextureDimension2D'.toJS, 1.toJS);
        final descriptor = JSObject()..setProperty('value'.toJS, limits);
        globalContext
            .getProperty<JSObject>('Object'.toJS)
            .callMethod<JSAny?>(
              'defineProperty'.toJS,
              candidate,
              'limits'.toJS,
              descriptor,
            );
        return token;
      }).toJS,
    );
    final old = web.mgpuGetOwnDeviceObject()!;
    final lost = web.mgpuOwnDeviceLost(old)!;
    web.mgpuDestroyOwnDevice(old);
    await lost;
    await refuses(() async {
      await gpu.recoverLostDevice(deviceEpoch: epoch);
    }, 'Insufficient browser capabilities');
    check(
      candidate != null &&
          !gpu.isInitialized &&
          gpu.deviceEpoch == epoch &&
          gpu.recoveryStatus.canRetry &&
          identical(gpu.recoveryTarget, target),
      'Rejected browser candidate did not produce a confirmed discard',
    );
    await web.mgpuOwnDeviceLost(candidate!)!;
  } finally {
    module.setProperty(key, original);
  }
  return gpu.retryLostDevice(
    deviceEpoch: epoch,
    failedAttempt: gpu.recoveryStatus.attempt,
  );
}

Future<void> main() async {
  final gpu = Minigpu(), platform = MinigpuPlatform.instance;
  final attachment = MinigpuWeb.createForTest();
  final readerReleased = Completer<void>(), gpuRetired = Completer<void>();
  final result = <String, Object?>{'passed': false};
  var initialized = false, readerFenced = false, readerDraining = false;
  var attachmentHeld = false;
  void Function()? removeLoss;
  try {
    await gpu.init();
    initialized = true;
    final epoch = gpu.deviceEpoch;
    final generation = platform.currentDeviceBufferLimits!.deviceGeneration;
    final device = web.mgpuGetOwnDeviceObject()!;
    final buffer = gpu.createBuffer(16, BufferDataType.uint32);
    await buffer.write(
      Uint32List.fromList([11, 22, 33, 44]),
      4,
      dataType: BufferDataType.uint32,
    );
    await gpu.waitForSubmittedWork();
    await refuses(
      () => platform.retireLostDevice(deviceGeneration: generation),
      'Healthy device retirement',
    );
    await refuses(platform.initializeContext, 'Premature context replacement');
    await availabilityIsNotLoss();
    check(
      gpu.isDeviceCurrent,
      'A refused retirement damaged the healthy device',
    );
    final target = gpu.recoveryTarget;
    check(
      target is BrowserRecoveryTarget && gpu.adapterIdentity == null,
      'Browser replacement invented a physical adapter identity',
    );
    await attachment.initializeContext();
    attachmentHeld = true;

    gpu.deviceCoordinator.register(
      label: 'GPU-only buffer',
      fence: () {},
      drain: () async {
        try {
          final retired = await gpu.retireSubmittedWork(deviceEpoch: epoch);
          check(
            retired == GpuWorkRetirement.deviceAborted,
            'Device loss was incorrectly reported as a successful frame',
          );
          buffer.destroy();
          gpuRetired.complete();
        } catch (error, stack) {
          gpuRetired.completeError(error, stack);
          rethrow;
        }
      },
    );
    gpu.deviceCoordinator.register(
      label: 'Outstanding host reader',
      fence: () => readerFenced = true,
      drain: () {
        readerDraining = true;
        return readerReleased.future;
      },
    );
    final lost = Completer<void>();
    removeLoss = gpu.onDeviceLost((lostEpoch) {
      check(lostEpoch == epoch, 'Wrong device was reported lost');
      lost.complete();
    });
    // Deliberately destroy the real creation GPUDevice, not an emulated loss.
    web.mgpuDestroyOwnDevice(device);
    await lost.future;
    await gpuRetired.future;
    check(
      !gpu.isDeviceCurrent && gpu.liveBufferCount == 0,
      'GPU-only resources did not retire after actual browser loss',
    );
    check(
      readerFenced &&
          readerDraining &&
          gpu.deviceCoordinator.snapshot.phase == GpuDeviceDrainPhase.draining,
      'GPU abortion bypassed the independent host reader',
    );
    await refuses(gpu.waitForSubmittedWork, 'Frame completion after loss');
    await refuses(
      () => platform.retireLostDevice(deviceGeneration: generation + 1),
      'Another generation retirement',
    );
    final first = platform.retireLostDevice(deviceGeneration: generation);
    check(
      identical(first, platform.retireLostDevice(deviceGeneration: generation)),
      'Repeated GPU retirement did not retain the same receipt',
    );
    await first;

    var finalRetired = false;
    final finalContext = gpu
        .observeContextRetirement(deviceEpoch: epoch)
        .then((_) => finalRetired = true);
    final recovery = gpu.recoverLostDevice(deviceEpoch: epoch);
    await Future<void>.delayed(Duration.zero);
    check(
      !finalRetired && gpu.isInitialized,
      'Context detached before its host reader finished',
    );
    readerReleased.complete();
    while (gpu.isInitialized) {
      await Future<void>.delayed(const Duration(milliseconds: 10));
    }
    initialized = false;
    check(
      !finalRetired && gpu.recoveryStatus.phase == GpuRecoveryPhase.draining,
      'A local detach was mistaken for final module retirement',
    );
    await refuses(
      () => platform.initializeContextForReplacement(target!),
      'Replacement while another real module attachment remains',
    );
    await attachment.destroyContext();
    attachmentHeld = false;
    await finalContext;
    final replacement = await recovery;
    initialized = true;
    check(
      replacement == epoch + 1 &&
          gpu.isDeviceCurrent &&
          identical(gpu.recoveryTarget, target) &&
          gpu.adapterIdentity == null,
      'Browser replacement did not retain its capability contract',
    );
    await refuses(
      () => platform.retireLostDevice(deviceGeneration: generation),
      'Detached creation device retirement',
    );
    final newDevice = web.mgpuGetOwnDeviceObject()!;
    check(
      !newDevice.strictEquals(device).toDart,
      'Browser recovery reused the lost GPUDevice',
    );

    final shader = gpu.createComputeShader();
    final output = gpu.createBuffer(16, BufferDataType.uint32);
    shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  data[id.x] = 100u + id.x;
}
''');
    shader.setBuffer('data', output);
    await shader.dispatch(1, 1, 1);
    final values = Uint32List(4);
    await output.read(values, 4, dataType: BufferDataType.uint32);
    check(
      values.join(',') == '100,101,102,103',
      'Replacement device did not execute new GPU work: $values',
    );
    web.mgpuDestroyOwnDevice(device);
    await Future<void>.delayed(Duration.zero);
    check(
      gpu.isDeviceCurrent,
      'An obsolete device callback invalidated its successor',
    );

    removeLoss();
    removeLoss = null;
    shader.destroy();
    output.destroy();
    final retryEpoch = await rejectedCandidateCanRetry(gpu);
    check(
      retryEpoch == replacement + 1 && gpu.isDeviceCurrent,
      'Explicit browser Retry did not replace the discarded candidate',
    );
    final replacementRetired = gpu.observeContextRetirement(
      deviceEpoch: retryEpoch,
    );
    await gpu.destroy();
    initialized = false;
    await replacementRetired;
    await gpu.init();
    initialized = true;
    check(
      gpu.isDeviceCurrent && gpu.deviceEpoch == retryEpoch + 1,
      'Ordinary close-then-init raced final browser retirement',
    );
    final finalClose = gpu.observeContextRetirement(
      deviceEpoch: gpu.deviceEpoch,
    );
    await gpu.destroy();
    initialized = false;
    await finalClose;
    result.addAll({
      'passed': true,
      'epoch': epoch,
      'platformGeneration': generation,
      'actualDeviceLoss': true,
      'gpuOnlyResourcesRetired': true,
      'hostReaderBlockedDetach': true,
      'staleAndHealthyRetirementRefused': true,
      'lookupFailureCannotDestroyDevice': true,
      'moduleAttachmentBlockedReplacement': true,
      'replacementEpoch': replacement,
      'replacementOutput': values.toList(),
      'rejectedCandidateRetiredBeforeRetry': true,
      'retryEpoch': retryEpoch,
      'staleLossCannotInvalidateReplacement': true,
      'healthyFinalContextRetirement': true,
      'ordinaryCloseThenInit': true,
      'automaticRecoveryQualified': true,
    });
  } catch (error, stack) {
    result['error'] = '$error\n$stack';
  } finally {
    removeLoss?.call();
    if (!readerReleased.isCompleted) readerReleased.complete();
    if (attachmentHeld) await attachment.destroyContext();
    if (initialized) {
      try {
        await gpu.destroy();
      } catch (error) {
        result['cleanupError'] = '$error';
        result['passed'] = false;
      }
    }
    globalContext.setProperty(
      'deviceRetirementResult'.toJS,
      jsonEncode(result).toJS,
    );
  }
}
