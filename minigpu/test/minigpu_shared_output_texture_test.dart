// Tests for SharedOutputTexture and createD3D11DeviceOnDawnAdapter.
//
// On Windows these exercise the real D3D11 NT-shared-handle path. On
// non-Windows platforms `createSharedOutputTexture` returns null and
// `createD3D11DeviceOnDawnAdapter` returns 0; we assert that contract
// rather than skipping.

import 'dart:io';
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:minigpu/minigpu.dart';

bool get _isWindows => Platform.isWindows;

void main() {
  late Minigpu gpu;

  setUpAll(() async {
    gpu = Minigpu();
    await gpu.init();
  });

  tearDownAll(() async {
    gpu.destroyAllTrackedShaders();
    await gpu.destroy();
  });

  group('Cross-platform contract', () {
    test('createSharedOutputTexture returns null on non-Windows', () {
      if (_isWindows) return; // Windows path covered below.
      final tex = gpu.createSharedOutputTexture(64, 64);
      expect(tex, isNull);
    });

    test('createD3D11DeviceOnDawnAdapter returns 0 on non-Windows', () {
      if (_isWindows) return;
      expect(gpu.createD3D11DeviceOnDawnAdapter(), equals(0));
    });
  });

  group('Windows D3D11 shared output texture', () {
    test('allocation reports requested dimensions and live pointers', () {
      if (!_isWindows) return;
      final tex = gpu.createSharedOutputTexture(256, 128);
      expect(tex, isNotNull, reason: 'D3D11 shared texture creation failed');
      expect(tex!.isValid, isTrue);
      expect(tex.width, 256);
      expect(tex.height, 128);
      expect(
        tex.d3d11Handle,
        isNot(0),
        reason: 'Expected a non-null NT shared HANDLE',
      );
      expect(
        tex.d3d11TexturePtr,
        isNot(0),
        reason: 'Expected a non-null ID3D11Texture2D pointer',
      );
      tex.destroy();
      expect(tex.isValid, isFalse);
    });

    test('destroy() is idempotent', () {
      if (!_isWindows) return;
      final tex = gpu.createSharedOutputTexture(32, 32);
      expect(tex, isNotNull);
      tex!.destroy();
      // Calling destroy a second time must not throw.
      tex.destroy();
      expect(tex.isValid, isFalse);
    });

    test('createD3D11DeviceOnDawnAdapter returns a non-zero pointer', () {
      if (!_isWindows) return;
      final dev = gpu.createD3D11DeviceOnDawnAdapter();
      expect(
        dev,
        isNot(0),
        reason: 'Expected an ID3D11Device* on the Dawn adapter',
      );
      // Ownership is transferred to the caller; we deliberately leak this
      // refcount in the test (releasing requires shim access). FFmpeg / the
      // real consumer Releases it via av_buffer_unref.
    });

    test(
      'copyFromBuffer accepts an RGBA8 buffer of matching size',
      () async {
        if (!_isWindows) return;
        const w = 32;
        const h = 16;
        final tex = gpu.createSharedOutputTexture(w, h);
        expect(tex, isNotNull);

        // Build a w*h u32-packed RGBA8 buffer with a non-trivial pattern.
        final pixels = Uint32List(w * h);
        for (var i = 0; i < pixels.length; i++) {
          pixels[i] = 0xFF | (0x80 << 8) | (0x40 << 16) | (0xFF << 24);
        }

        final buf = gpu.createBuffer(w * h, BufferDataType.uint32);
        await buf.write(pixels, w * h, dataType: BufferDataType.uint32);

        final ok = tex!.copyFromBuffer(buf);
        expect(
          ok,
          isTrue,
          reason: 'copyFromBuffer should succeed on Windows D3D11 backend',
        );

        buf.destroy();
        tex.destroy();
      },
      // The copy implementation blocks on wgpuQueueOnSubmittedWorkDone
      // which requires a running event-pump on the WGPU instance. In an
      // isolated unit-test context (no example app driving the pump) the
      // callback can take several seconds to fire on some Dawn builds
      // and intermittently times out. End-to-end coverage of this path
      // lives in the screenshare_mp4 example which exercises it under
      // a real frame loop.
      skip: 'see comment: wgpuQueueOnSubmittedWorkDone pump requirement',
    );

    test('allocate / destroy 10 cycles does not leak buffers', () async {
      if (!_isWindows) return;
      final before = gpu.liveBufferCount;
      for (var i = 0; i < 10; i++) {
        final t = gpu.createSharedOutputTexture(64, 64);
        expect(t, isNotNull);
        t!.destroy();
      }
      expect(gpu.liveBufferCount, equals(before));
    });
  });
}
