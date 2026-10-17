/// The planar kernel's colour on a real device, checked against the standards'
/// forward equations: an RGB colour encoded to 8-bit YUV with each matrix's
/// Kr/Kb must come back from the kernel within rounding, at either range.
library;

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:minigpu/minigpu.dart';
import 'package:minigpu_view/minigpu_view.dart';

typedef _Rgb = (int, int, int);

/// (Kr, Kb) per matrix.
const _kRKb = {
  YuvMatrix.bt709: (0.2126, 0.0722),
  YuvMatrix.bt601: (0.299, 0.114),
  YuvMatrix.bt2020: (0.2627, 0.0593),
};

const _colors = <_Rgb>[
  (255, 255, 255),
  (0, 0, 0),
  (128, 128, 128),
  (200, 60, 30),
  (40, 180, 90),
  (90, 70, 220),
  (250, 220, 20),
  (20, 200, 230),
];

/// 8-bit (Y, Cb, Cr) for an 8-bit RGB colour.
_Rgb _encode(_Rgb rgb, YuvMatrix matrix, bool full) {
  final (kr, kb) = _kRKb[matrix]!;
  final r = rgb.$1 / 255, g = rgb.$2 / 255, b = rgb.$3 / 255;
  final y = kr * r + (1 - kr - kb) * g + kb * b;
  final cb = (b - y) / (2 * (1 - kb));
  final cr = (r - y) / (2 * (1 - kr));
  int q(double v) => v.round().clamp(0, 255);
  return full
      ? (q(y * 255), q(128 + cb * 255), q(128 + cr * 255))
      : (q(16 + y * 219), q(128 + cb * 224), q(128 + cr * 224));
}

void main() {
  late Minigpu gpu;
  late ComputeShader shader;

  setUpAll(() async {
    gpu = Minigpu();
    await gpu.init();
    shader = gpu.createComputeShader()..loadKernelString(kPlanarToRgbaWgsl);
  });

  tearDownAll(() => shader.destroy());

  /// Runs the kernel over [bytes] and returns each pixel's RGB, row-major.
  Future<List<_Rgb>> convert(
    Uint8List bytes,
    int width,
    int height,
    List<int> layout, // format, yOffset, yStride, uOffset, uStride, vOffset, vStride
    YuvMatrix matrix,
    bool full,
  ) async {
    final padded = Uint8List((bytes.length + 3) & ~3)..setAll(0, bytes);
    final words = padded.buffer.asUint32List();
    final cfg = Uint32List.fromList(
        [width, height, ...layout, full ? 1 : 0, matrix.shaderCode]);
    expect(cfg.length, kPlanarCfgWords);

    final src = gpu.createBuffer(padded.length, BufferDataType.uint32);
    final cfgBuf = gpu.createBuffer(cfg.length * 4, BufferDataType.uint32);
    final dst = gpu.createBuffer(width * height * 4, BufferDataType.uint32);
    await src.write(words, words.length, dataType: BufferDataType.uint32);
    await cfgBuf.write(cfg, cfg.length, dataType: BufferDataType.uint32);
    shader
      ..setBufferAtSlot(0, src)
      ..setBufferAtSlot(1, dst)
      ..setBufferAtSlot(2, cfgBuf);
    await shader.dispatch((width + 7) ~/ 8, (height + 7) ~/ 8, 1);
    final out = Uint32List(width * height);
    await dst.read(out, out.length, dataType: BufferDataType.uint32);
    for (final b in [src, cfgBuf, dst]) {
      b.destroy();
    }
    return [for (final p in out) (p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF)];
  }

  /// Largest channel difference between [got] pixel pairs and [_colors].
  /// Each colour fills one 2x2 block, so its pixel is at column 2i.
  int worst(List<_Rgb> got, int width) {
    var worst = 0;
    for (var i = 0; i < _colors.length; i++) {
      for (final row in [0, 1]) {
        for (final col in [2 * i, 2 * i + 1]) {
          final (r, g, b) = got[row * width + col];
          final (er, eg, eb) = _colors[i];
          for (final d in [r - er, g - eg, b - eb]) {
            if (d.abs() > worst) worst = d.abs();
          }
        }
      }
    }
    return worst;
  }

  const width = 2 * 8, height = 2;

  /// One colour per 2x2 block, as NV12.
  Uint8List nv12(YuvMatrix matrix, bool full) {
    final bytes = Uint8List(width * height + width);
    for (var i = 0; i < _colors.length; i++) {
      final (y, cb, cr) = _encode(_colors[i], matrix, full);
      for (final row in [0, 1]) {
        bytes[row * width + 2 * i] = y;
        bytes[row * width + 2 * i + 1] = y;
      }
      bytes[width * height + 2 * i] = cb;
      bytes[width * height + 2 * i + 1] = cr;
    }
    return bytes;
  }

  for (final matrix in YuvMatrix.values) {
    for (final full in [false, true]) {
      test('NV12 ${matrix.name} ${full ? 'full' : 'limited'} range', () async {
        final got = await convert(nv12(matrix, full), width, height,
            [PlanarFormat.nv12.shaderCode, 0, width, width * height, width, 0, 0],
            matrix, full);
        expect(worst(got, width), lessThanOrEqualTo(3));
      });
    }
  }

  test('I420 and YUY2 unpack to the same colours', () async {
    const matrix = YuvMatrix.bt601;
    const full = false;
    final i420 = Uint8List(width * height + width);
    final yuy2 = Uint8List(width * 2 * height);
    const half = width ~/ 2;
    for (var i = 0; i < _colors.length; i++) {
      final (y, cb, cr) = _encode(_colors[i], matrix, full);
      for (final row in [0, 1]) {
        i420[row * width + 2 * i] = y;
        i420[row * width + 2 * i + 1] = y;
        yuy2.setAll(row * width * 2 + 4 * i, [y, cb, y, cr]);
      }
      i420[width * height + i] = cb;
      i420[width * height + half + i] = cr;
    }

    final fromI420 = await convert(i420, width, height, [
      PlanarFormat.i420.shaderCode,
      0, width, width * height, half, width * height + half, half,
    ], matrix, full);
    expect(worst(fromI420, width), lessThanOrEqualTo(3), reason: 'I420');

    final fromYuy2 = await convert(yuy2, width, height,
        [PlanarFormat.yuy2.shaderCode, 0, width * 2, 0, 0, 0, 0], matrix, full);
    expect(worst(fromYuy2, width), lessThanOrEqualTo(3), reason: 'YUY2');
  });
}
