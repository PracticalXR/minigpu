@TestOn('vm')
library;

import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:minigpu/minigpu.dart';
import 'package:minigpu_ffi/minigpu_ffi.dart';
import 'package:minigpu_ffi/minigpu_ffi_completion.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:test/test.dart';

void main() {
  test(
    'secondary recovery replaces its retired handle on the exact adapter',
    () async {
      final primary = Minigpu.withPlatform(MinigpuFfi());
      await primary.init();
      final platform =
          MinigpuFfi().createSecondaryPlatform(primary.adapterName!)!
              as FfiSecondaryPlatform;
      final gpu = Minigpu.withPlatform(platform);
      await gpu.init();
      final original = gpu.adapterIdentity;
      expect(original, isNotNull);
      final epoch = gpu.deviceEpoch;
      final pin = calloc<Pointer<NativeContextLease>>();
      try {
        platform.withLiveExternalPoolContext((_, context) {
          expect(mgpuContextRetainLease(context.cast(), pin), 0);
          return 0;
        });
        expect(
          await GpuCompletion.run(
            (port, token) =>
                mgpuContextLeaseRetireDeviceAsyncToPort(pin.value, port, token),
            (_) => throw UnsupportedError('Native ports required'),
          ),
          isTrue,
        );
        expect(gpu.isDeviceCurrent, isFalse);
        mgpuContextReleaseLease(pin.value);
        pin.value = nullptr;
        expect(await gpu.recoverLostDevice(deviceEpoch: epoch), epoch + 1);
        expect(gpu.adapterIdentity, original);
        expect(primary.isDeviceCurrent, isTrue);
        final values = Uint32List.fromList([19, 29, 31, 37]);
        final buffer = gpu.createBuffer(16, BufferDataType.uint32);
        await buffer.write(values, 4, dataType: BufferDataType.uint32);
        final read = Uint32List(4);
        await buffer.read(read, 4, dataType: BufferDataType.uint32);
        expect(read, orderedEquals(values));
        buffer.destroy();
      } finally {
        if (pin.value != nullptr) mgpuContextReleaseLease(pin.value);
        calloc.free(pin);
        final secondaryRetired = gpu.observeContextRetirement(
          deviceEpoch: gpu.deviceEpoch,
        );
        await gpu.destroy();
        await secondaryRetired.timeout(const Duration(seconds: 5));
        final primaryRetired = primary.observeContextRetirement(
          deviceEpoch: primary.deviceEpoch,
        );
        await primary.destroy();
        await primaryRetired.timeout(const Duration(seconds: 5));
      }
    },
  );

  test(
    'a different physical adapter identity never becomes a replacement',
    () async {
      final platform = MinigpuFfi();
      await platform.initializeContext();
      final actual = platform.creationAdapterIdentity!;
      final generation = platform.currentDeviceBufferLimits!.deviceGeneration;
      final retired = platform.observeContextRetirement(
        deviceGeneration: generation,
      );
      await platform.destroyContext();
      await retired;
      final wrong = DeviceAdapterIdentity(
        backend: actual.backend,
        kind: actual.kind,
        low: actual.low ^ 1,
        high: actual.high,
      );
      await expectLater(
        platform.initializeContextForRecovery(wrong),
        throwsA(
          isA<DeviceReplacementDiscarded>().having(
            (error) => error.expectedAdapter,
            'expected adapter',
            wrong,
          ),
        ),
      );
      expect(platform.currentDeviceBufferLimits, isNull);
      expect(platform.creationAdapterIdentity, isNull);
      // The receipt is emitted after final native teardown: another attempt
      // on the original physical adapter can acquire and use a fresh context.
      await platform.initializeContextForRecovery(actual);
      expect(platform.creationAdapterIdentity, actual);
      final replacement = platform.observeContextRetirement(
        deviceGeneration: platform.currentDeviceBufferLimits!.deviceGeneration,
      );
      await platform.destroyContext();
      await replacement.timeout(const Duration(seconds: 5));
    },
  );
}
