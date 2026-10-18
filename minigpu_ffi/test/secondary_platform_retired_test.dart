/// A secondary context whose handle has been released never hands it to C
/// again. The handle is released by destroyContext() and, since the failed-
/// init leak fix, by a failed initializeContext() too, after which the public
/// wrapper still reaches selectedAdapterName (Minigpu.adapterName) with no
/// init check: that used to read freed native memory.
///
///     dart test test/secondary_platform_retired_test.dart
@TestOn('vm')
library;

import 'package:minigpu_ffi/minigpu_ffi.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:test/test.dart';

void main() {
  test('a released secondary handle is never used again', () async {
    final secondary = MinigpuFfi().createSecondaryPlatform('');
    expect(secondary, isNotNull);
    await secondary!.destroyContext();

    expect(secondary.selectedAdapterName, isNull);
    expect(
      () => secondary.createBuffer(16, BufferDataType.float32),
      throwsStateError,
    );
    expect(secondary.createComputeShader, throwsStateError);
    await expectLater(secondary.waitForSubmittedWork(), throwsStateError);
    // A retired platform cannot be revived; a retry needs a new one.
    await expectLater(secondary.initializeContext(), throwsStateError);
    // Idempotent: a second release must not free the handle twice.
    await secondary.destroyContext();
  });
}
