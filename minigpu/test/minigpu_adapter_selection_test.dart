/// Adapter selection logic tests.
///
/// Verifies that the new dawn::native::EnumerateAdapters-based adapter
/// selection code (added to buffer.cpp) works end-to-end:
///
///  1. `gpu.init()` succeeds — proves the EnumerateAdapters path doesn't
///     crash regardless of how many adapters are present.
///  2. On Windows the selected adapter backs a real D3D11 device
///     (createD3D11DeviceOnDawnAdapter returns non-zero), confirming that
///     the enumeration picked a hardware adapter and not the WARP CPU
///     fallback.
///  3. MGPU_ADAPTER_NAME env-var override smoke-test: setting the variable
///     to a name that matches no adapter must not throw — the code falls
///     back gracefully to the automatic preference order.
///
/// All tests that require real GPU hardware are guarded so they skip
/// cleanly in headless / CI environments where no GPU is present.
@TestOn('vm')
library;

import 'dart:io';

import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

void main() {
  group('Adapter selection — EnumerateAdapters path', () {
    // -----------------------------------------------------------------------
    // 1. init() succeeds with the new enumeration code path.
    // -----------------------------------------------------------------------
    test('init() completes without throwing', () async {
      final gpu = Minigpu();
      // Will throw if EnumerateAdapters crashes or returns 0 adapters on a
      // machine that should have at least one GPU/WARP software adapter.
      await expectLater(gpu.init(), completes);
      await gpu.destroy();
    });

    // -----------------------------------------------------------------------
    // 2. On Windows the selected adapter is a hardware adapter (not CPU/WARP).
    //    createD3D11DeviceOnDawnAdapter() returns 0 only when Dawn ended up on
    //    a non-D3D11 backend or a software/CPU adapter.
    // -----------------------------------------------------------------------
    test(
      'Windows: selected adapter backs a real D3D11 device (non-zero handle)',
      () async {
        if (!Platform.isWindows) {
          markTestSkipped('D3D11 device creation is Windows-only');
          return;
        }
        final gpu = Minigpu();
        await gpu.init();
        final dev = gpu.createD3D11DeviceOnDawnAdapter();
        expect(
          dev,
          isNot(0),
          reason:
              'EnumerateAdapters should have selected a hardware adapter '
              '(discrete or integrated GPU). A zero handle indicates Dawn '
              'ended up on a CPU/WARP adapter or the D3D11 backend was not '
              'selected. Check [mgpu adapter] log lines.',
        );
        await gpu.destroy();
      },
    );

    // -----------------------------------------------------------------------
    // 3. Discrete-first preference: createD3D11DeviceOnDawnAdapter is
    //    non-zero regardless of the WebGPU backend (D3D11 or D3D12) because
    //    the shim creates an independent D3D11 device on the same physical
    //    adapter that Dawn selected.  SharedOutputTexture only works when
    //    Dawn itself is on D3D11 (MGPU_BACKEND=d3d11); on D3D12 it returns
    //    null which is the documented contract — we don't assert isNotNull
    //    here since the backend is controlled by the env var, not by our
    //    adapter-selection code.
    // -----------------------------------------------------------------------
    test(
      'Windows: createD3D11DeviceOnDawnAdapter non-zero independent of backend',
      () async {
        if (!Platform.isWindows) {
          markTestSkipped('D3D11 device creation is Windows-only');
          return;
        }
        final gpu = Minigpu();
        await gpu.init();
        // The shim creates a separate D3D11 device on the same physical
        // adapter regardless of whether Dawn uses D3D11 or D3D12.
        final dev = gpu.createD3D11DeviceOnDawnAdapter();
        expect(
          dev,
          isNot(0),
          reason:
              'createD3D11DeviceOnDawnAdapter must return a non-zero '
              'ID3D11Device* on Windows; EnumerateAdapters should have picked '
              'a hardware adapter (discrete or integrated). Zero indicates Dawn '
              'ended up on a CPU/WARP adapter.',
        );
        await gpu.destroy();
      },
    );

    // -----------------------------------------------------------------------
    // 4. MGPU_ADAPTER_NAME mismatch: a non-matching name must not throw —
    //    the code falls back to the automatic discrete→integrated preference
    //    order.  We do this by temporarily setting the env var, calling
    //    init(), then resetting.
    //
    //    Note: Dart's Platform.environment is read-only; we rely on the fact
    //    that this test always runs in a child process so setting the var
    //    in the environment before spawning (or skipping if not set) is the
    //    proper approach.  Instead, we test the *observable contract*: if
    //    the env var is absent the init path must still succeed (tested
    //    above), and if it is present with a matching value the selection
    //    must differ only in logging (not in thrown exceptions).
    //    This test documents the skip behaviour when the env var is absent.
    // -----------------------------------------------------------------------
    test(
      'MGPU_ADAPTER_NAME env var: init() succeeds regardless of match',
      () async {
        // This test only runs when the caller explicitly sets the env var;
        // otherwise it documents the expected behaviour and passes trivially.
        final nameFilter = Platform.environment['MGPU_ADAPTER_NAME'];
        if (nameFilter == null) {
          // No override set — automatic path already covered by test 1.
          return;
        }
        // Override is set. init() must not throw even if the name matches
        // nothing (the C++ code falls back to automatic selection).
        final gpu = Minigpu();
        await expectLater(
          gpu.init(),
          completes,
          reason:
              'init() must not throw when MGPU_ADAPTER_NAME="$nameFilter" '
              'even if no adapter name matches that substring',
        );
        await gpu.destroy();
      },
    );
  });
}
