import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';
import 'package:minigpu/minigpu.dart';

/// Allocate a native RGBA32 buffer filled with [r,g,b,a] repeated.
Pointer<Uint8> _allocRgba(int w, int h, int r, int g, int b, int a) {
  final ptr = malloc.allocate<Uint8>(w * h * 4);
  for (var i = 0; i < w * h; i++) {
    ptr[i * 4 + 0] = r;
    ptr[i * 4 + 1] = g;
    ptr[i * 4 + 2] = b;
    ptr[i * 4 + 3] = a;
  }
  return ptr;
}

/// Allocate a native NV12 buffer (Y plane = [y], UV plane = [u,v] interleaved).
Pointer<Uint8> _allocNv12(int w, int h, int y, int u, int v) {
  final ySize = w * h;
  final uvSize = (w ~/ 2) * (h ~/ 2) * 2;
  final ptr = malloc.allocate<Uint8>(ySize + uvSize);
  for (var i = 0; i < ySize; i++) {
    ptr[i] = y;
  }
  for (var i = 0; i < (w ~/ 2) * (h ~/ 2); i++) {
    ptr[ySize + i * 2 + 0] = u;
    ptr[ySize + i * 2 + 1] = v;
  }
  return ptr;
}

/// Read the RGBA8 output buffer back to a Uint8List.
Future<Uint8List> _readRgba(Buffer buf, int w, int h) async {
  final out = Uint8List(w * h * 4);
  await buf.read(out, w * h * 4, dataType: BufferDataType.uint8);
  return out;
}

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

  // ---------------------------------------------------------------------------
  // Capability queries
  // ---------------------------------------------------------------------------
  group('Capability queries', () {
    test('CPU content type is supported', () {
      expect(
        gpu.isExternalContentTypeSupported(ExternalContentType.cpu),
        isTrue,
      );
    });

    test('RGBA32 pixel format is supported', () {
      expect(
        gpu.isExternalPixelFormatSupported(ExternalPixelFormat.rgba32),
        isTrue,
      );
    });

    test('NV12 pixel format is supported', () {
      expect(
        gpu.isExternalPixelFormatSupported(ExternalPixelFormat.nv12),
        isTrue,
      );
    });
  });

  // ---------------------------------------------------------------------------
  // VideoTexture properties – RGBA32
  // ---------------------------------------------------------------------------
  group('VideoTexture properties – RGBA32', () {
    const W = 16, H = 12;
    Pointer<Uint8>? ptr;
    VideoTexture? tex;

    setUp(() {
      ptr = _allocRgba(W, H, 255, 0, 0, 255);
      tex = gpu.importVideoFrame(
        ExternalVideoBuffer(
          contentType: ExternalContentType.cpu,
          pixelFormat: ExternalPixelFormat.rgba32,
          width: W,
          height: H,
          planes: [
            ExternalPlane(
              dataPtr: ptr!.address,
              width: W,
              height: H,
              strideBytes: W * 4,
            ),
          ],
        ),
      );
    });

    tearDown(() {
      tex?.destroy();
      if (ptr != null) malloc.free(ptr!);
    });

    test('importVideoFrame returns non-null', () {
      expect(tex, isNotNull);
    });

    test('width and height are correct', () {
      expect(tex!.width, equals(W));
      expect(tex!.height, equals(H));
    });

    test('pixelFormat is rgba32', () {
      expect(tex!.pixelFormat, equals(ExternalPixelFormat.rgba32));
    });

    test('numPlanes is 1 for RGBA32', () {
      expect(tex!.numPlanes, equals(1));
    });

    test('isValid is true before destroy', () {
      expect(tex!.isValid, isTrue);
    });

    test('isValid is false after destroy', () {
      tex!.destroy();
      expect(tex!.isValid, isFalse);
      tex = null; // prevent double-destroy in tearDown
    });
  });

  // ---------------------------------------------------------------------------
  // toRGBA – pixel accuracy
  // ---------------------------------------------------------------------------
  group('toRGBA – pixel accuracy', () {
    const W = 4, H = 4;
    late Pointer<Uint8> ptr;
    late VideoTexture tex;
    late Buffer outBuf;

    tearDown(() {
      outBuf.destroy();
      tex.destroy();
      malloc.free(ptr);
    });

    test('solid red RGBA32 round-trips through toRGBA', () async {
      ptr = _allocRgba(W, H, 200, 0, 0, 255);
      tex = gpu.importVideoFrame(
        ExternalVideoBuffer(
          contentType: ExternalContentType.cpu,
          pixelFormat: ExternalPixelFormat.rgba32,
          width: W,
          height: H,
          planes: [
            ExternalPlane(
              dataPtr: ptr.address,
              width: W,
              height: H,
              strideBytes: W * 4,
            ),
          ],
        ),
      )!;
      outBuf = tex.toRGBA();
      final rgba = await _readRgba(outBuf, W, H);
      // Check every pixel: R≈200, G=0, B=0, A=255
      for (var i = 0; i < W * H; i++) {
        expect(rgba[i * 4 + 0], closeTo(200, 2), reason: 'R at pixel $i');
        expect(rgba[i * 4 + 1], closeTo(0, 2), reason: 'G at pixel $i');
        expect(rgba[i * 4 + 2], closeTo(0, 2), reason: 'B at pixel $i');
        expect(rgba[i * 4 + 3], closeTo(255, 2), reason: 'A at pixel $i');
      }
    });

    test('solid green RGBA32 round-trips through toRGBA', () async {
      ptr = _allocRgba(W, H, 0, 180, 0, 255);
      tex = gpu.importVideoFrame(
        ExternalVideoBuffer(
          contentType: ExternalContentType.cpu,
          pixelFormat: ExternalPixelFormat.rgba32,
          width: W,
          height: H,
          planes: [
            ExternalPlane(
              dataPtr: ptr.address,
              width: W,
              height: H,
              strideBytes: W * 4,
            ),
          ],
        ),
      )!;
      outBuf = tex.toRGBA();
      final rgba = await _readRgba(outBuf, W, H);
      for (var i = 0; i < W * H; i++) {
        expect(rgba[i * 4 + 0], closeTo(0, 2), reason: 'R at pixel $i');
        expect(rgba[i * 4 + 1], closeTo(180, 2), reason: 'G at pixel $i');
        expect(rgba[i * 4 + 2], closeTo(0, 2), reason: 'B at pixel $i');
      }
    });

    test('alpha channel preserved through toRGBA', () async {
      ptr = _allocRgba(W, H, 0, 0, 0, 128);
      tex = gpu.importVideoFrame(
        ExternalVideoBuffer(
          contentType: ExternalContentType.cpu,
          pixelFormat: ExternalPixelFormat.rgba32,
          width: W,
          height: H,
          planes: [
            ExternalPlane(
              dataPtr: ptr.address,
              width: W,
              height: H,
              strideBytes: W * 4,
            ),
          ],
        ),
      )!;
      outBuf = tex.toRGBA();
      final rgba = await _readRgba(outBuf, W, H);
      for (var i = 0; i < W * H; i++) {
        expect(rgba[i * 4 + 3], closeTo(128, 2), reason: 'A at pixel $i');
      }
    });
  });

  // ---------------------------------------------------------------------------
  // toRGBA – NV12
  // ---------------------------------------------------------------------------
  group('toRGBA – NV12', () {
    const W = 8, H = 8;
    late Pointer<Uint8> ptr;
    late VideoTexture tex;
    late Buffer outBuf;

    tearDown(() {
      outBuf.destroy();
      tex.destroy();
      malloc.free(ptr);
    });

    ExternalVideoBuffer _nv12Frame(int y, int u, int v) {
      return ExternalVideoBuffer(
        contentType: ExternalContentType.cpu,
        pixelFormat: ExternalPixelFormat.nv12,
        width: W,
        height: H,
        planes: [
          ExternalPlane(
            dataPtr: ptr.address,
            width: W,
            height: H,
            strideBytes: W,
          ),
          ExternalPlane(
            dataPtr: ptr.address + W * H,
            width: W ~/ 2,
            height: H ~/ 2,
            strideBytes: W,
          ),
        ],
      );
    }

    test('numPlanes is 2 for NV12', () {
      ptr = _allocNv12(W, H, 128, 128, 128);
      tex = gpu.importVideoFrame(_nv12Frame(128, 128, 128))!;
      outBuf = tex.toRGBA(); // needed for tearDown
      expect(tex.numPlanes, equals(2));
    });

    test('neutral grey NV12 (Y=128, UV=128) → near-grey RGB', () async {
      ptr = _allocNv12(W, H, 128, 128, 128);
      tex = gpu.importVideoFrame(_nv12Frame(128, 128, 128))!;
      outBuf = tex.toRGBA();
      final rgba = await _readRgba(outBuf, W, H);
      // BT.709 neutral grey: R≈G≈B≈128 ±10
      for (var i = 0; i < W * H; i++) {
        expect(rgba[i * 4 + 0], closeTo(128, 15), reason: 'R at $i');
        expect(rgba[i * 4 + 1], closeTo(128, 15), reason: 'G at $i');
        expect(rgba[i * 4 + 2], closeTo(128, 15), reason: 'B at $i');
      }
    });

    test('NV12 black frame (Y=16) → near-black RGB', () async {
      ptr = _allocNv12(W, H, 16, 128, 128);
      tex = gpu.importVideoFrame(_nv12Frame(16, 128, 128))!;
      outBuf = tex.toRGBA();
      final rgba = await _readRgba(outBuf, W, H);
      for (var i = 0; i < W * H; i++) {
        expect(rgba[i * 4 + 0], lessThan(30), reason: 'R at $i');
        expect(rgba[i * 4 + 1], lessThan(30), reason: 'G at $i');
        expect(rgba[i * 4 + 2], lessThan(30), reason: 'B at $i');
      }
    });

    test('NV12 white frame (Y=235) → near-white RGB', () async {
      ptr = _allocNv12(W, H, 235, 128, 128);
      tex = gpu.importVideoFrame(_nv12Frame(235, 128, 128))!;
      outBuf = tex.toRGBA();
      final rgba = await _readRgba(outBuf, W, H);
      for (var i = 0; i < W * H; i++) {
        expect(rgba[i * 4 + 0], greaterThan(220), reason: 'R at $i');
        expect(rgba[i * 4 + 1], greaterThan(220), reason: 'G at $i');
        expect(rgba[i * 4 + 2], greaterThan(220), reason: 'B at $i');
      }
    });
  });

  // ---------------------------------------------------------------------------
  // setOnShader + setBufferAtSlot – custom compute shader
  // ---------------------------------------------------------------------------
  group('Custom compute shader via setOnShader', () {
    const W = 8, H = 8;
    late Pointer<Uint8> ptr;
    late VideoTexture tex;
    late Buffer outBuf;
    late Buffer ubuf;
    late ComputeShader cs;

    const _kInvertShader = '''
@group(0) @binding(0) var in_tex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf : array<u32>;
struct Params { width: u32, height: u32 }
@group(0) @binding(2) var<storage, read_write> uni : Params;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= uni.width || gid.y >= uni.height) { return; }
  let px = textureLoad(in_tex, vec2<u32>(gid.x, gid.y), 0);
  let r = u32((1.0 - px.r) * 255.0);
  let g = u32((1.0 - px.g) * 255.0);
  let b = u32((1.0 - px.b) * 255.0);
  let a = u32(px.a * 255.0);
  out_buf[gid.y * uni.width + gid.x] = r | (g << 8u) | (b << 16u) | (a << 24u);
}
''';

    setUp(() {
      ptr = _allocRgba(W, H, 255, 0, 0, 255); // solid red
      tex = gpu.importVideoFrame(
        ExternalVideoBuffer(
          contentType: ExternalContentType.cpu,
          pixelFormat: ExternalPixelFormat.rgba32,
          width: W,
          height: H,
          planes: [
            ExternalPlane(
              dataPtr: ptr.address,
              width: W,
              height: H,
              strideBytes: W * 4,
            ),
          ],
        ),
      )!;

      outBuf = gpu.createBuffer(W * H * 4, BufferDataType.uint8);

      final params = Uint32List(2);
      params[0] = W;
      params[1] = H;
      ubuf = gpu.createBuffer(8, BufferDataType.uint32);
      ubuf.write(params, 2, dataType: BufferDataType.uint32);

      cs = gpu.createComputeShader();
      cs.loadKernelString(_kInvertShader);
    });

    tearDown(() {
      cs.destroy();
      outBuf.destroy();
      ubuf.destroy();
      tex.destroy();
      malloc.free(ptr);
    });

    test('invert shader: red input → cyan output', () async {
      tex.setOnShader(cs, 0);
      cs.setBufferAtSlot(1, outBuf);
      cs.setBufferAtSlot(2, ubuf);
      await cs.dispatch((W + 7) ~/ 8, (H + 7) ~/ 8, 1);

      final raw = Uint8List(W * H * 4);
      await outBuf.read(raw, W * H * 4, dataType: BufferDataType.uint8);
      for (var i = 0; i < W * H; i++) {
        expect(raw[i * 4 + 0], closeTo(0, 4), reason: 'R at $i'); // 255→0
        expect(raw[i * 4 + 1], closeTo(255, 4), reason: 'G at $i'); // 0→255
        expect(raw[i * 4 + 2], closeTo(255, 4), reason: 'B at $i'); // 0→255
        expect(raw[i * 4 + 3], closeTo(255, 4), reason: 'A at $i'); // 255→255
      }
    });
  });

  // ---------------------------------------------------------------------------
  // liveBufferCount – no leak across import/destroy cycles
  // ---------------------------------------------------------------------------
  group('liveBufferCount', () {
    test('single importVideoFrame + destroy does not leak buffers', () {
      const W = 4, H = 4;
      final before = gpu.liveBufferCount;
      final ptr = _allocRgba(W, H, 0, 0, 255, 255);
      final tex = gpu.importVideoFrame(
        ExternalVideoBuffer(
          contentType: ExternalContentType.cpu,
          pixelFormat: ExternalPixelFormat.rgba32,
          width: W,
          height: H,
          planes: [
            ExternalPlane(
              dataPtr: ptr.address,
              width: W,
              height: H,
              strideBytes: W * 4,
            ),
          ],
        ),
      )!;
      final outBuf = tex.toRGBA();
      outBuf.destroy();
      tex.destroy();
      malloc.free(ptr);
      expect(gpu.liveBufferCount, equals(before));
    });

    test('10 import/toRGBA/destroy cycles are stable', () async {
      const W = 4, H = 4;
      final before = gpu.liveBufferCount;
      for (var n = 0; n < 10; n++) {
        final ptr = _allocRgba(W, H, n * 20, 0, 0, 255);
        final tex = gpu.importVideoFrame(
          ExternalVideoBuffer(
            contentType: ExternalContentType.cpu,
            pixelFormat: ExternalPixelFormat.rgba32,
            width: W,
            height: H,
            planes: [
              ExternalPlane(
                dataPtr: ptr.address,
                width: W,
                height: H,
                strideBytes: W * 4,
              ),
            ],
          ),
        )!;
        final outBuf = tex.toRGBA();
        outBuf.destroy();
        tex.destroy();
        malloc.free(ptr);
      }
      expect(gpu.liveBufferCount, equals(before));
    });
  });

  // ---------------------------------------------------------------------------
  // destroyAllTrackedShaders
  // ---------------------------------------------------------------------------
  group('destroyAllTrackedShaders', () {
    test('shader count drops to zero after destroy all', () {
      final cs1 = gpu.createComputeShader();
      final cs2 = gpu.createComputeShader();
      expect(gpu.liveShaderCount, greaterThanOrEqualTo(2));
      // Suppress "used without loading kernel" warnings — just testing count.
      cs1.destroy();
      cs2.destroy();
      expect(gpu.liveShaderCount, equals(0));
    });

    test('destroyAllTrackedShaders clears any accumulated shaders', () {
      gpu.createComputeShader(); // intentionally not captured
      gpu.createComputeShader();
      gpu.destroyAllTrackedShaders();
      expect(gpu.liveShaderCount, equals(0));
    });
  });

  // ---------------------------------------------------------------------------
  // Edge cases
  // ---------------------------------------------------------------------------
  group('Edge cases', () {
    test('1×1 RGBA32 frame imports and reads back correctly', () async {
      final ptr = _allocRgba(1, 1, 77, 88, 99, 255);
      final tex = gpu.importVideoFrame(
        ExternalVideoBuffer(
          contentType: ExternalContentType.cpu,
          pixelFormat: ExternalPixelFormat.rgba32,
          width: 1,
          height: 1,
          planes: [
            ExternalPlane(
              dataPtr: ptr.address,
              width: 1,
              height: 1,
              strideBytes: 4,
            ),
          ],
        ),
      )!;
      final outBuf = tex.toRGBA();
      final rgba = await _readRgba(outBuf, 1, 1);
      expect(rgba[0], closeTo(77, 2));
      expect(rgba[1], closeTo(88, 2));
      expect(rgba[2], closeTo(99, 2));
      outBuf.destroy();
      tex.destroy();
      malloc.free(ptr);
    });

    test('double-destroy does not throw', () {
      const W = 4, H = 4;
      final ptr = _allocRgba(W, H, 0, 0, 0, 255);
      final tex = gpu.importVideoFrame(
        ExternalVideoBuffer(
          contentType: ExternalContentType.cpu,
          pixelFormat: ExternalPixelFormat.rgba32,
          width: W,
          height: H,
          planes: [
            ExternalPlane(
              dataPtr: ptr.address,
              width: W,
              height: H,
              strideBytes: W * 4,
            ),
          ],
        ),
      )!;
      tex.destroy();
      expect(() => tex.destroy(), returnsNormally);
      malloc.free(ptr);
    });
  });
}
