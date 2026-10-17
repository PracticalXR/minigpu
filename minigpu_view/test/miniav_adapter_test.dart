import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:miniav/miniav.dart'
    show
        MiniAVBuffer,
        MiniAVBufferContentType,
        MiniAVBufferType,
        MiniAVPixelFormat,
        MiniAVVideoBuffer,
        MiniAVVideoColor,
        MiniAVYuvMatrix;
import 'package:minigpu/minigpu.dart' show Minigpu, YuvMatrix;
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart'
    show MinigpuPlatform;
import 'package:minigpu_view/minigpu_view.dart';

/// The adapters only hand the wrapper to the frames they build; nothing here
/// may reach the platform.
class _NoPlatform implements MinigpuPlatform {
  @override
  dynamic noSuchMethod(Invocation invocation) =>
      throw StateError('unexpected platform call ${invocation.memberName}');
}

MiniAVBuffer _gpuBuffer(
  MiniAVBufferContentType contentType,
  MiniAVPixelFormat pixelFormat,
  Object? handle,
) =>
    MiniAVBuffer(
      type: MiniAVBufferType.video,
      contentType: contentType,
      timestampUs: 0,
      data: MiniAVVideoBuffer(
        width: 64,
        height: 32,
        pixelFormat: pixelFormat,
        strideBytes: const [0],
        planes: [Uint8List(0)],
        nativeHandles: [handle],
      ),
      dataSizeBytes: 0,
    );

MiniAVBuffer _cpuBuffer(
  MiniAVPixelFormat pixelFormat,
  int width,
  int height,
  List<Uint8List?> planes, {
  MiniAVVideoColor? color,
}) =>
    MiniAVBuffer(
      type: MiniAVBufferType.video,
      contentType: MiniAVBufferContentType.cpu,
      timestampUs: 0,
      data: MiniAVVideoBuffer(
        width: width,
        height: height,
        pixelFormat: pixelFormat,
        // Zero strides: the producer left them to the defaults.
        strideBytes: [for (final _ in planes) 0],
        planes: planes,
        color: color,
      ),
      dataSizeBytes: 0,
    );

void main() {
  final gpu = Minigpu.withPlatform(_NoPlatform());

  group('asPreviewSource', () {
    test('refuses a Windows NT handle', () {
      final buffer = _gpuBuffer(MiniAVBufferContentType.gpuD3D11Handle,
          MiniAVPixelFormat.bgra32, 0x1234);
      expect(buffer.asPreviewSource(), isNull);
    });

    test('refuses CPU buffers and absent or opaque handles', () {
      expect(
        _cpuBuffer(MiniAVPixelFormat.bgra32, 2, 2, [Uint8List(16)])
            .asPreviewSource(),
        isNull,
      );
      for (final handle in <Object?>[null, 0, 'texture']) {
        expect(
          _gpuBuffer(MiniAVBufferContentType.gpuMetalTexture,
                  MiniAVPixelFormat.bgra32, handle)
              .asPreviewSource(),
          isNull,
          reason: 'handle $handle',
        );
      }
    });

    test('passes a texture pointer through', () {
      final source = _gpuBuffer(MiniAVBufferContentType.gpuMetalTexture,
              MiniAVPixelFormat.bgra32, 0x5678)
          .asPreviewSource()!;
      expect(source.kind, PreviewSourceKind.nativeSharedTexture);
      expect(source.toChannelMessage(), {
        'handle': 0x5678,
        'width': 64,
        'height': 32,
        'pixelFormat': 'bgra32',
        'contentType': 'gpuMetalTexture',
      });
    });
  });

  group('asGpuFrame', () {
    test('imports a D3D11 handle with its channel order', () {
      final frame = _gpuBuffer(MiniAVBufferContentType.gpuD3D11Handle,
              MiniAVPixelFormat.bgra32, 0x1234)
          .asGpuFrame(gpu);
      expect(frame, isA<SharedHandleGpuFrame>());
      final bgra = frame as SharedHandleGpuFrame;
      expect(bgra.sharedHandle, 0x1234);
      expect(bgra.bgra, isTrue);
      expect((bgra.width, bgra.height), (64, 32));

      final rgba = _gpuBuffer(MiniAVBufferContentType.gpuD3D11Handle,
              MiniAVPixelFormat.rgba32, 0x1234)
          .asGpuFrame(gpu) as SharedHandleGpuFrame;
      expect(rgba.bgra, isFalse);
    });

    test('refuses handles the BGRA blit cannot read', () {
      GpuFrame? frame(MiniAVBufferContentType type, MiniAVPixelFormat format,
              Object? handle) =>
          _gpuBuffer(type, format, handle).asGpuFrame(gpu);
      const d3d11 = MiniAVBufferContentType.gpuD3D11Handle;
      expect(frame(d3d11, MiniAVPixelFormat.nv12, 0x1234), isNull);
      expect(frame(d3d11, MiniAVPixelFormat.bgra32, 0), isNull);
      expect(frame(d3d11, MiniAVPixelFormat.bgra32, 'handle'), isNull);
      expect(
        frame(MiniAVBufferContentType.gpuMetalTexture,
            MiniAVPixelFormat.bgra32, 0x1234),
        isNull,
      );
    });

    test('odd-width NV12 defaults its chroma stride to whole pairs', () {
      final frame = _cpuBuffer(
        MiniAVPixelFormat.nv12,
        5,
        3,
        [Uint8List(15), Uint8List(12)],
      ).asGpuFrame(gpu) as PlanarCpuGpuFrame;
      expect(frame.format, PlanarFormat.nv12);
      expect(frame.planes[0].strideBytes, 5);
      expect(frame.planes[1].strideBytes, 6);
      expect(frame.planes[1].height, 2);
    });

    test('odd-width YUY2 defaults its stride to whole pixel pairs', () {
      final frame = _cpuBuffer(
        MiniAVPixelFormat.yuy2,
        5,
        2,
        [Uint8List(24)],
      ).asGpuFrame(gpu) as PlanarCpuGpuFrame;
      expect(frame.format, PlanarFormat.yuy2);
      expect(frame.planes.single.strideBytes, 12);
    });

    test('YUV frames take the colour the capture reports', () {
      PlanarCpuGpuFrame frame(MiniAVPixelFormat format, List<Uint8List> planes,
              [MiniAVVideoColor? color]) =>
          _cpuBuffer(format, 4, 2, planes, color: color).asGpuFrame(gpu)
              as PlanarCpuGpuFrame;
      final nv12 = [Uint8List(8), Uint8List(4)];
      final i420 = [Uint8List(8), Uint8List(2), Uint8List(2)];
      final yuy2 = [Uint8List(16)];

      // Unreported: BT.709, limited range.
      final plain = frame(MiniAVPixelFormat.nv12, nv12);
      expect((plain.matrix, plain.range),
          (YuvMatrix.bt709, PlanarColorRange.limited));

      const camera =
          MiniAVVideoColor(matrix: MiniAVYuvMatrix.bt601, fullRange: true);
      for (final (format, planes) in [
        (MiniAVPixelFormat.nv12, nv12),
        (MiniAVPixelFormat.i420, i420),
        (MiniAVPixelFormat.yv12, i420),
        (MiniAVPixelFormat.yuy2, yuy2),
      ]) {
        final f = frame(format, planes, camera);
        expect((f.matrix, f.range), (YuvMatrix.bt601, PlanarColorRange.full),
            reason: '$format');
      }

      final hdr = frame(MiniAVPixelFormat.nv12, nv12,
          const MiniAVVideoColor(matrix: MiniAVYuvMatrix.bt2020, fullRange: false));
      expect((hdr.matrix, hdr.range),
          (YuvMatrix.bt2020, PlanarColorRange.limited));
    });

    test('a missing CPU plane gives no frame', () {
      expect(
        _cpuBuffer(MiniAVPixelFormat.nv12, 4, 4, [Uint8List(16)])
            .asGpuFrame(gpu),
        isNull,
      );
    });
  });
}
