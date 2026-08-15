/// WHAT A PANE CAN BE MADE OF — one type for every shape a frame arrives in.
///
/// A producer hands you pixels in whatever form it has: a shared GPU texture
/// handle, a compute buffer, or plain bytes in host memory. Displaying each of
/// those is a different piece of code, and an app that shows more than one
/// ends up writing all of them — usually including a slow path that decodes an
/// image on the UI isolate, because that is the easiest one to reach for.
///
/// ⚠️ THE SLOW PATH IS NOT VIABLE AT REAL RESOLUTIONS. Reading a frame back
/// off the GPU and decoding it into a `dart:ui` image costs, per frame, more
/// than a frame interval at 4K — on the isolate that also builds the UI. It
/// does not degrade gracefully; it stalls the app. So CPU-delivered frames
/// JOIN the GPU path here ([CpuBytesGpuFrame]) rather than bypassing it: one
/// upload, then the same blit every other source uses.
///
/// A frame does not know where it will be shown. It knows how to copy itself
/// into a [GpuPaneTarget], and the target decides when — which is what lets
/// callers with very different display policies share this machinery.
library;

import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';

import 'planar_convert.dart';

/// The destination a frame copies itself into, plus scratch it may borrow.
///
/// Passed to [GpuFrame.copyInto] rather than a bare texture so a host-memory
/// frame can reach a POOLED upload buffer. Allocating one per frame would put
/// a VRAM allocation on the display path, which is exactly the per-frame churn
/// a zero-copy pipeline exists to avoid.
abstract class GpuPaneTarget {
  /// The texture being displayed.
  SharedOutputTexture get texture;

  /// A buffer of at least [byteLength], reused across frames. Owned by the
  /// target; callers must not destroy it.
  Buffer uploadBuffer(int byteLength);

  /// A SECOND pooled buffer, for conversions that cannot write in place —
  /// unpacking planar YUV needs a distinct destination from its source.
  Buffer scratchBuffer(int byteLength);

  /// Small pooled buffer for kernel constants. Separate from the others
  /// because it is tiny, rewritten every frame, and must not force the big
  /// pools to resize.
  Buffer configBuffer(int wordCount);

  /// The conversion pipeline, compiled once per device and shared by every
  /// pane. Compiling per frame (or per pane) would cost seconds on backends
  /// that go through FXC.
  ComputeShader planarConverter();
}

/// Something that can become the contents of a pane.
abstract class GpuFrame {
  const GpuFrame();

  /// Coded size, so the target can size its texture to match.
  int get width;
  int get height;

  /// Copy into [target]. Returns false when the frame could not be read — a
  /// stale handle, a cross-adapter mismatch, a short buffer — so the caller
  /// can keep showing the previous frame rather than going black.
  Future<bool> copyInto(GpuPaneTarget target);
}

/// A frame behind a platform shared handle: a D3D11 shared NT handle on
/// Windows, as produced by screen/camera capture and by GPU video decoders.
///
/// ⚠️ THE HANDLE IS BORROWED, WHICH CONSTRAINS HOW YOU MAY SCHEDULE IT.
/// Producers commonly close or recycle the handle as soon as the consumer
/// reports the frame done, so this imports and blits when asked and retains
/// nothing. Two consequences, and both bite the deferred path:
///
///  * `GpuPane.show(frame)` is safe — the copy happens before you return.
///  * `markDirty` + a LATER `flush` is not, for a handle whose validity ends
///    when the producer moves on. By flush time it may be closed (the copy
///    fails and the pane keeps its previous contents — a silently dropped
///    frame), or, worse, still open but pointing at a texture the producer has
///    since OVERWRITTEN, in which case you display a newer frame than the one
///    you marked. That second case is invisible: nothing errors.
///
/// So: pace with [CpuBytesGpuFrame] or [BufferGpuFrame], whose contents you
/// own, or call `show` immediately for a borrowed handle. A producer with a
/// PERSISTENT surface (one texture rewritten in place, rather than a handle
/// per frame) is the exception — its handle stays valid, but the overwrite
/// hazard above still applies.
class SharedHandleGpuFrame extends GpuFrame {
  const SharedHandleGpuFrame({
    required this.gpu,
    required this.sharedHandle,
    required this.width,
    required this.height,
    this.bgra = true,
  });

  final Minigpu gpu;
  final int sharedHandle;

  /// Most desktop capture and decode paths hand out BGRA; the blit swizzles
  /// into the RGBA a pane expects.
  final bool bgra;

  @override
  final int width;

  @override
  final int height;

  @override
  Future<bool> copyInto(GpuPaneTarget target) async {
    if (sharedHandle == 0) return false;
    final tex = gpu.importVideoFrame(ExternalVideoBuffer(
      contentType: ExternalContentType.d3d11SharedHandle,
      pixelFormat:
          bgra ? ExternalPixelFormat.bgra32 : ExternalPixelFormat.rgba32,
      width: width,
      height: height,
      planes: [
        ExternalPlane(
          dataPtr: sharedHandle,
          width: width,
          height: height,
          strideBytes: width * 4,
        ),
      ],
    ));
    if (tex == null) return false;
    try {
      return await tex.bgraToRgbaSharedOutputAsync(target.texture);
    } finally {
      tex.destroy();
    }
  }
}

/// A frame already resident in a minigpu [Buffer] as packed RGBA8 — one u32
/// per pixel, row major. The cheapest case: no import, no upload, a
/// GPU-to-GPU blit.
class BufferGpuFrame extends GpuFrame {
  const BufferGpuFrame({
    required this.buffer,
    required this.width,
    required this.height,
  });

  final Buffer buffer;

  @override
  final int width;

  @override
  final int height;

  @override
  Future<bool> copyInto(GpuPaneTarget target) =>
      target.texture.copyFromBufferAsync(buffer);
}

/// Packed RGBA8 bytes in host memory — a CPU capture fallback, a software
/// decoder, a generator.
///
/// This is the type that makes a CPU-delivered frame a first-class citizen
/// instead of a reason to build a second, slower display path. The bytes go UP
/// to the GPU once and are composited there.
///
/// ⚠️ TIGHTLY PACKED, RGBA8, `width * height * 4` bytes. A strided source must
/// be de-strided first — this cannot detect padding, and the symptom is a
/// skewed picture rather than an error.
class CpuBytesGpuFrame extends GpuFrame {
  const CpuBytesGpuFrame({
    required this.rgba,
    required this.width,
    required this.height,
  });

  final Uint8List rgba;

  @override
  final int width;

  @override
  final int height;

  @override
  Future<bool> copyInto(GpuPaneTarget target) async {
    final need = width * height * 4;
    if (rgba.length < need) return false;
    final buffer = target.uploadBuffer(need);
    // writeRawBytes hands the caller's own backing store to the driver, so the
    // payload is copied once (into the staging ring) rather than twice.
    await buffer.writeRawBytes(
      rgba.length == need ? rgba : Uint8List.sublistView(rgba, 0, need),
    );
    return target.texture.copyFromBufferAsync(buffer);
  }
}


/// Planar or subsampled bytes in host memory — NV12, I420, YUY2, BGRA.
///
/// THE FORMAT CAMERAS ACTUALLY PRODUCE. A capture path that cannot hand over a
/// GPU handle delivers one of these, and converting them on the CPU is the
/// mistake this type exists to prevent: a per-pixel Dart loop inside the
/// capture callback is O(pixels) on the UI isolate and stalls the app well
/// before it looks slow. Here the planes are uploaded as bytes and unpacked by
/// a compute shader.
///
/// ⚠️ STRIDES ARE NOT WIDTH. Capture APIs pad rows; pass the real
/// [PlanarPlane.strideBytes] or the picture shears progressively down the
/// frame — a failure that reads like a codec bug.
class PlanarCpuGpuFrame extends GpuFrame {
  const PlanarCpuGpuFrame({
    required this.format,
    required this.planes,
    required this.width,
    required this.height,
    this.range = PlanarColorRange.limited,
  });

  final PlanarFormat format;

  /// In layout order: NV12 = [Y, UV]; I420 = [Y, U, V]; YUY2/BGRA = [packed].
  final List<PlanarPlane> planes;

  /// Studio swing unless the producer says otherwise — see the note in
  /// planar_convert.dart about why guessing this wrong is silent.
  final PlanarColorRange range;

  @override
  final int width;

  @override
  final int height;

  @override
  Future<bool> copyInto(GpuPaneTarget target) async {
    if (planes.length < format.planeCount || width <= 0 || height <= 0) {
      return false;
    }

    // Pack every plane into ONE upload, recording each one's byte offset so
    // the kernel can find it. One transfer beats one per plane, and it keeps
    // the buffer pool to a single allocation.
    final offsets = <int>[];
    var total = 0;
    for (final p in planes) {
      offsets.add(total);
      total += p.byteLength;
      // Rows are addressed as bytes in the shader, so 4-byte alignment of each
      // plane's start keeps the u32 indexing honest.
      total = (total + 3) & ~3;
    }

    final src = target.uploadBuffer(total);
    for (var i = 0; i < planes.length; i++) {
      await src.writeRawBytes(
        Uint8List.fromList(planes[i].bytes),
        dstByteOffset: offsets[i],
      );
    }

    final rgba = target.scratchBuffer(width * height * 4);
    final cfg = Uint32List(kPlanarCfgWords)
      ..[0] = width
      ..[1] = height
      ..[2] = format.shaderCode
      ..[3] = offsets[0]
      ..[4] = planes[0].strideBytes
      ..[5] = offsets.length > 1 ? offsets[1] : 0
      ..[6] = planes.length > 1 ? planes[1].strideBytes : 0
      ..[7] = offsets.length > 2 ? offsets[2] : 0
      ..[8] = planes.length > 2 ? planes[2].strideBytes : 0
      ..[9] = range == PlanarColorRange.full ? 1 : 0;

    final cfgBuffer = target.configBuffer(kPlanarCfgWords);
    await cfgBuffer.write(cfg, kPlanarCfgWords,
        dataType: BufferDataType.uint32);

    final shader = target.planarConverter()
      ..setBufferAtSlot(0, src)
      ..setBufferAtSlot(1, rgba)
      ..setBufferAtSlot(2, cfgBuffer);
    await shader.dispatch((width + 7) ~/ 8, (height + 7) ~/ 8, 1);

    return target.texture.copyFromBufferAsync(rgba);
  }
}
