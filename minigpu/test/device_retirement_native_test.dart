@TestOn('vm')
library;

import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:minigpu/minigpu.dart';
import 'package:minigpu_ffi/minigpu_ffi.dart';
import 'package:minigpu_ffi/minigpu_ffi_bindings.dart' as ffi;
import 'package:minigpu_ffi/minigpu_ffi_completion.dart';
import 'package:test/test.dart';

void main() {
  test(
    'lost native initialization is a Dart failure, not a fresh device',
    () async {
      final platform = MinigpuFfi();
      final gpu = Minigpu.withPlatform(platform);
      final lease = calloc<Pointer<NativeContextLease>>();
      final context = calloc<Pointer<ffi.MGPUContextHandle>>();
      await gpu.init();
      final epoch = gpu.deviceEpoch;
      final retired = gpu.observeContextRetirement(deviceEpoch: epoch);
      var contextRetired = false;
      retired.then((_) => contextRetired = true).ignore();
      try {
        final generation = platform.currentDeviceBufferLimits!.deviceGeneration;
        expect(gpu.observeContextRetirement(deviceEpoch: epoch), same(retired));
        await expectLater(
          gpu.observeContextRetirement(deviceEpoch: epoch + 1),
          throwsStateError,
        );
        await expectLater(
          platform.retireLostDevice(deviceGeneration: generation),
          throwsStateError,
        );
        expect(gpu.isDeviceCurrent, isTrue);
        expect(
          await gpu.retireSubmittedWork(deviceEpoch: epoch),
          GpuWorkRetirement.completed,
        );
        expect(mgpuDefaultContextRetainLease(lease, context), 0);
        expect(
          await GpuCompletion.run(
            (port, token) => mgpuContextLeaseRetireDeviceAsyncToPort(
              lease.value,
              port,
              token,
            ),
            (_) => throw UnsupportedError('Native receipt requires Dart ports'),
          ),
          isTrue,
        );
        expect(platform.currentDeviceBufferLimits, isNull);
        await expectLater(platform.initializeContext(), throwsStateError);
        expect(platform.currentDeviceBufferLimits, isNull);
        await expectLater(
          platform.retireLostDevice(deviceGeneration: generation + 1),
          throwsStateError,
        );
        expect(
          await gpu.retireSubmittedWork(deviceEpoch: epoch),
          GpuWorkRetirement.deviceAborted,
        );
        expect(
          await gpu.retireSubmittedWork(deviceEpoch: epoch),
          GpuWorkRetirement.deviceAborted,
        );
        await expectLater(gpu.waitForSubmittedWork(), throwsStateError);
        await expectLater(
          gpu.retireSubmittedWork(deviceEpoch: epoch + 1),
          throwsStateError,
        );
        // A device-abort receipt is not a final context receipt. Detaching the
        // local wrapper must still leave the retained external owner in place.
        await gpu.destroy();
        expect(contextRetired, isFalse);
        expect(ffi.mgpuContextRefCount(), greaterThan(0));
        expect(gpu.observeContextRetirement(deviceEpoch: epoch), same(retired));
        mgpuContextReleaseLease(lease.value);
        lease.value = nullptr;
        await retired.timeout(const Duration(seconds: 5));
        expect(ffi.mgpuContextRefCount(), 0);
      } finally {
        if (lease.value != nullptr) mgpuContextReleaseLease(lease.value);
        try {
          await gpu.destroy();
          // A timeout fails this test; it can never authorize recovery.
          await retired.timeout(const Duration(seconds: 5));
          expect(ffi.mgpuContextRefCount(), 0);
        } finally {
          calloc.free(context);
          calloc.free(lease);
        }
      }
    },
  );
}
