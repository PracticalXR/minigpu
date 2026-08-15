/// Adapter: `MiniAVBuffer.asGpuFrame(gpu)` — one call from a capture buffer to
/// something a [GpuPane] can show, whatever the capture layer handed over.
///
/// ── WHY THIS EXISTS ─────────────────────────────────────────────────────────
/// A capture buffer arrives in one of several shapes depending on the platform,
/// the device, and what the driver felt able to do this second: a GPU shared
/// handle, or host memory in NV12, I420, YUY2 or BGRA. Consumers were writing a
/// branch per shape, and the CPU branches almost always ended in a per-pixel
/// Dart loop — the one thing that must not be on a display path.
///
/// This collapses all of it. The FORMAT KNOWLEDGE LIVES HERE, in the package
/// that has to know both vocabularies, rather than in every app that shows a
/// camera.
///
/// ── WHAT IT DOES NOT DO ─────────────────────────────────────────────────────
/// It does not copy the buffer. A GPU frame borrows the handle and a host
/// frame references the capture layer's own plane memory, both of which are
/// valid only until the buffer is released — so display it before releasing,
/// and do not stash the frame for later. See the scheduling note on
/// [SharedHandleGpuFrame].
library;

import 'dart:typed_data';

import 'package:miniav/miniav.dart'
    show
        MiniAVBuffer,
        MiniAVBufferContentType,
        MiniAVPixelFormat,
        MiniAVVideoBuffer;
import 'package:minigpu/minigpu.dart' show Minigpu;

import '../gpu_frame.dart';
import '../planar_convert.dart';

extension MiniavGpuFrameAdapter on MiniAVBuffer {
  /// Wrap this buffer as a displayable frame, or null when it carries nothing
  /// showable (a non-video buffer, an empty CPU plane set, a format not
  /// handled below).
  ///
  /// Returning null rather than guessing matters: a wrong guess about layout
  /// produces a picture that is skewed or miscoloured but not obviously
  /// broken, which is worse than an empty pane.
  GpuFrame? asGpuFrame(Minigpu gpu) {
    final vb = data;
    if (vb is! MiniAVVideoBuffer) return null;
    final w = vb.width, h = vb.height;
    if (w <= 0 || h <= 0) return null;

    // GPU path first: nothing to upload, just import and blit.
    if (contentType != MiniAVBufferContentType.cpu) {
      final handle = vb.nativeHandles.isEmpty ? null : vb.nativeHandles[0];
      if (handle is! int || handle == 0) return null;
      return SharedHandleGpuFrame(
        gpu: gpu,
        sharedHandle: handle,
        width: w,
        height: h,
        // NV12 handles exist, but the shared-handle blit path converts BGRA/
        // RGBA only; a planar handle needs the importer's own plane views,
        // which is a separate frame type rather than a flag here.
        bgra: vb.pixelFormat != MiniAVPixelFormat.rgba32,
      );
    }

    int strideOf(int i, int fallback) =>
        i < vb.strideBytes.length && vb.strideBytes[i] > 0
            ? vb.strideBytes[i]
            : fallback;

    List<int>? planeAt(int i) =>
        i < vb.planes.length ? vb.planes[i] : null;

    switch (vb.pixelFormat) {
      case MiniAVPixelFormat.nv12:
        final y = planeAt(0), uv = planeAt(1);
        if (y == null || uv == null) return null;
        return PlanarCpuGpuFrame(
          format: PlanarFormat.nv12,
          width: w,
          height: h,
          planes: [
            PlanarPlane(bytes: y, strideBytes: strideOf(0, w), height: h),
            // Chroma is half height, and its stride is the FULL width because
            // U and V are interleaved — a common place to halve twice.
            PlanarPlane(
                bytes: uv, strideBytes: strideOf(1, w), height: (h + 1) >> 1),
          ],
        );

      case MiniAVPixelFormat.i420:
      case MiniAVPixelFormat.yv12:
        final y = planeAt(0), u = planeAt(1), v = planeAt(2);
        if (y == null || u == null || v == null) return null;
        final chromaH = (h + 1) >> 1;
        final chromaW = (w + 1) >> 1;
        // YV12 is I420 with the chroma planes swapped; handling it by swapping
        // here costs nothing and removes a whole format from the caller.
        final swapped = vb.pixelFormat == MiniAVPixelFormat.yv12;
        return PlanarCpuGpuFrame(
          format: PlanarFormat.i420,
          width: w,
          height: h,
          planes: [
            PlanarPlane(bytes: y, strideBytes: strideOf(0, w), height: h),
            PlanarPlane(
                bytes: swapped ? v : u,
                strideBytes: strideOf(swapped ? 2 : 1, chromaW),
                height: chromaH),
            PlanarPlane(
                bytes: swapped ? u : v,
                strideBytes: strideOf(swapped ? 1 : 2, chromaW),
                height: chromaH),
          ],
        );

      case MiniAVPixelFormat.yuy2:
        final packed = planeAt(0);
        if (packed == null) return null;
        return PlanarCpuGpuFrame(
          format: PlanarFormat.yuy2,
          width: w,
          height: h,
          planes: [
            PlanarPlane(
                bytes: packed, strideBytes: strideOf(0, w * 2), height: h),
          ],
        );

      case MiniAVPixelFormat.bgra32:
        final packed = planeAt(0);
        if (packed == null) return null;
        // Through the planar kernel rather than CpuBytesGpuFrame, because that
        // one cannot honour a stride and capture rows are routinely padded.
        return PlanarCpuGpuFrame(
          format: PlanarFormat.bgra8,
          width: w,
          height: h,
          planes: [
            PlanarPlane(
                bytes: packed, strideBytes: strideOf(0, w * 4), height: h),
          ],
        );

      case MiniAVPixelFormat.rgba32:
        final packed = planeAt(0);
        if (packed == null) return null;
        final stride = strideOf(0, w * 4);
        if (stride != w * 4) return null; // padded RGBA: no unpacker yet
        return CpuBytesGpuFrame(
          rgba: _asUint8List(packed),
          width: w,
          height: h,
        );

      default:
        // Unhandled format — empty beats a guess.
        return null;
    }
  }
}

/// miniAV hands planes back as `Uint8List`; the planar path accepts the wider
/// `List<int>`, so this narrows for the one case that needs the concrete type.
Uint8List _asUint8List(List<int> bytes) =>
    bytes is Uint8List ? bytes : Uint8List.fromList(bytes);
