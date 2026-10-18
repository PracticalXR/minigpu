/// importVideoFrame validates the plane list before any native call. The
/// native importers read planes by content type and pixel format, not by
/// num_planes, so a short list handed them uninitialised slots (garbage
/// pointers and sizes), and a list longer than the struct's four was silently
/// truncated while num_planes still claimed all of them.
///
///     dart test test/import_video_frame_planes_test.dart
@TestOn('vm')
library;

import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:minigpu_ffi/minigpu_ffi.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:test/test.dart';

ExternalPlane _plane(int dataPtr, int width, int height, int strideBytes) =>
    ExternalPlane(
      dataPtr: dataPtr,
      width: width,
      height: height,
      strideBytes: strideBytes,
    );

ExternalVideoBuffer _cpuFrame(
  ExternalPixelFormat format,
  List<ExternalPlane> planes,
) => ExternalVideoBuffer(
  contentType: ExternalContentType.cpu,
  pixelFormat: format,
  width: 4,
  height: 4,
  planes: planes,
);

void main() {
  late MinigpuFfi platform;

  setUpAll(() async {
    platform = MinigpuFfi();
    await platform.initializeContext();
  });

  tearDownAll(() async {
    await platform.destroyContext();
  });

  test('a CPU NV12 frame carrying only its luma plane is refused', () {
    expect(
      () => platform.importVideoFrame(
        _cpuFrame(ExternalPixelFormat.nv12, [_plane(0, 4, 4, 4)]),
      ),
      throwsArgumentError,
    );
  });

  test('a CPU three-plane frame carrying two is refused', () {
    expect(
      () => platform.importVideoFrame(
        _cpuFrame(ExternalPixelFormat.yuv420pAsRGBPlanes, [
          _plane(0, 4, 4, 4),
          _plane(0, 2, 2, 2),
        ]),
      ),
      throwsArgumentError,
    );
  });

  test('more planes than the native struct holds are refused', () {
    expect(
      () => platform.importVideoFrame(
        _cpuFrame(
          ExternalPixelFormat.rgba32,
          List.filled(5, _plane(0, 4, 4, 16)),
        ),
      ),
      throwsArgumentError,
    );
  });

  test('complete CPU frames still import', () {
    final luma = calloc<Uint8>(16);
    final chroma = calloc<Uint8>(8);
    final rgba = calloc<Uint8>(64);
    addTearDown(() {
      calloc.free(luma);
      calloc.free(chroma);
      calloc.free(rgba);
    });

    final nv12 = platform.importVideoFrame(
      _cpuFrame(ExternalPixelFormat.nv12, [
        _plane(luma.address, 4, 4, 4),
        _plane(chroma.address, 2, 2, 4),
      ]),
    );
    expect(nv12, isNotNull);
    nv12!.destroy();

    final rgbaTexture = platform.importVideoFrame(
      _cpuFrame(ExternalPixelFormat.rgba32, [_plane(rgba.address, 4, 4, 16)]),
    );
    expect(rgbaTexture, isNotNull);
    rgbaTexture!.destroy();
  });
}
