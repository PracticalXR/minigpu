import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

void main() {
  test('native query reports the initialized device limits', () async {
    final gpu = Minigpu();
    expect(gpu.currentDeviceBufferLimits, isNull);
    await gpu.init();
    try {
      final limits = gpu.currentDeviceBufferLimits;
      expect(limits, isNotNull);
      expect(limits!.deviceEpoch, gpu.deviceEpoch);
      expect(limits.maxBufferSize, greaterThan(0));
      expect(limits.maxStorageBufferBindingSize, greaterThan(0));
      expect(gpu.isDeviceCurrent, isTrue);
    } finally {
      await gpu.destroy();
    }
    expect(gpu.currentDeviceBufferLimits, isNull);
  });
}
