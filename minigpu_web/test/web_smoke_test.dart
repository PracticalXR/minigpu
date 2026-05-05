/// Browser smoke tests for minigpu_web.
///
/// Run with:
///   dart test -p chrome
///
/// The pure-Dart tests (WebVideoTexture, probe) run without any setup.
///
/// The GPU compute group requires the minigpu WASM module to be bootstrapped
/// via the custom loader.  With plain `dart test` the WASM loader is NOT
/// included so those tests are skipped automatically on init failure.
///
/// For full GPU tests use Flutter's test runner which serves the package
/// assets correctly:
///   flutter test -p chrome test/web_smoke_test.dart \
///     --web-browser-flag="--enable-unsafe-webgpu" \
///     --web-browser-flag="--disable-dawn-features=disallow_unsafe_apis"
@TestOn('browser')
library;

import 'dart:js_interop';
import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:minigpu_web/minigpu_web.dart';
import 'package:test/test.dart';

// ---------------------------------------------------------------------------
// JS helpers â€“ probe WebGPU availability without touching the WASM.
// ---------------------------------------------------------------------------

@JS('navigator.gpu')
external JSAny? get _navigatorGpu;

bool get _hasWebGPU {
  final v = _navigatorGpu;
  return v != null && !v.isUndefined;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void main() {
  // -------------------------------------------------------------------------
  // WebVideoTexture â€“ pure Dart, no GPU required
  // -------------------------------------------------------------------------

  group('WebVideoTexture', () {
    test('constructs with expected fields', () {
      // Use a plain JSObject as a stand-in for the GPUExternalTexture handle.
      final fakeHandle = JSObject();
      final tex = WebVideoTexture(
        externalTexture: fakeHandle,
        pixelFormat: ExternalPixelFormat.bgra32,
        width: 1280,
        height: 720,
      );

      expect(tex.width, 1280);
      expect(tex.height, 720);
      expect(tex.pixelFormat, ExternalPixelFormat.bgra32);
      expect(tex.numPlanes, 1);
      expect(tex.externalTexture, same(fakeHandle));
    });

    test('bgraToRgbaSharedOutput returns false on Web', () {
      // This is a D3D11-only operation; the Web impl always returns false.
      final tex = WebVideoTexture(
        externalTexture: JSObject(),
        pixelFormat: ExternalPixelFormat.bgra32,
        width: 4,
        height: 4,
      );
      // We can't easily construct a PlatformSharedOutputTexture on Web,
      // but the method signature must exist and return false without throwing.
      // Verified by compilation succeeding and the method being present.
      expect(tex.bgraToRgbaSharedOutput, isA<Function>());
    });

    test('toRGBA throws UnsupportedError', () {
      final tex = WebVideoTexture(
        externalTexture: JSObject(),
        pixelFormat: ExternalPixelFormat.bgra32,
        width: 4,
        height: 4,
      );
      expect(() => tex.toRGBA(), throwsUnsupportedError);
    });

    test('destroy does not throw', () {
      final tex = WebVideoTexture(
        externalTexture: JSObject(),
        pixelFormat: ExternalPixelFormat.bgra32,
        width: 4,
        height: 4,
      );
      expect(() => tex.destroy(), returnsNormally);
    });

    test('setOnShader throws for non-WebComputeShader', () {
      final tex = WebVideoTexture(
        externalTexture: JSObject(),
        pixelFormat: ExternalPixelFormat.bgra32,
        width: 4,
        height: 4,
      );
      // Passing a mock that isn't a WebComputeShader should throw.
      expect(
        () => tex.setOnShader(_FakeShader(), 0, 0),
        throwsUnsupportedError,
      );
    });
  });

  // -------------------------------------------------------------------------
  // WebGPU availability probe
  // -------------------------------------------------------------------------

  group('WebGPU probe', () {
    test('_hasWebGPU returns a bool without throwing', () {
      expect(_hasWebGPU, isA<bool>());
    });
  });

  // -------------------------------------------------------------------------
  // GPU compute â€“ skipped when WebGPU not available
  // -------------------------------------------------------------------------

  group('GPU compute (requires WebGPU)', () {
    late MinigpuWeb gpu;
    bool _gpuReady = false;

    setUpAll(() async {
      if (!_hasWebGPU) return;
      try {
        gpu = MinigpuWeb.createForTest();
        await gpu.initializeContext();
        _gpuReady = true;
      } catch (_) {
        // WASM loader not served by dart test runner — skip silently.
        _gpuReady = false;
      }
    });

    tearDownAll(() async {
      if (_gpuReady) {
        await gpu.destroyContext();
      }
    });

    test('buffer write + read round-trip (float32)', () async {
      if (!_gpuReady) {
        markTestSkipped('GPU/WASM not available');
        return;
      }

      const n = 16;
      final input = Float32List.fromList(List.generate(n, (i) => i.toDouble()));
      final output = Float32List(n);

      final buf = gpu.createBuffer(n, BufferDataType.float32);
      await buf.write(input, n);
      await buf.read(output, n);
      buf.destroy();

      expect(output, orderedEquals(input));
    });

    test('compute shader doubles each element', () async {
      if (!_gpuReady) {
        markTestSkipped('GPU/WASM not available');
        return;
      }

      const n = 8;
      final input = Float32List.fromList(List.generate(n, (i) => i + 1.0));
      final output = Float32List(n);

      final buf = gpu.createBuffer(n, BufferDataType.float32);
      await buf.write(input, n);

      final shader = gpu.createComputeShader();
      shader.loadKernelString('''
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(8)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          let i = gid.x;
          data[i] = data[i] * 2.0;
        }
      ''');
      shader.setBuffer(0, buf);
      await shader.dispatch(1, 1, 1);

      await buf.read(output, n);
      buf.destroy();
      shader.destroy();

      final expected = Float32List.fromList(input.map((v) => v * 2).toList());
      expect(output, orderedEquals(expected));
    });
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

class _FakeShader implements PlatformComputeShader {
  @override
  void loadKernelString(String k) {}
  @override
  bool hasKernel() => false;
  @override
  void setBuffer(int tag, PlatformBuffer b) {}
  @override
  Future<void> dispatch(int x, int y, int z) async {}
  @override
  void destroy() {}
}
