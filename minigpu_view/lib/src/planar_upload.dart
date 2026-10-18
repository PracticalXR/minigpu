/// WHERE EACH PLANE OF A HOST FRAME GOES IN ITS ONE UPLOAD, and how many of
/// its bytes are sent.
///
/// Kept apart from the GPU calls because this arithmetic fails silently when
/// it is wrong (a sheared picture, stale chroma, a write the native layer
/// drops), so it is worth checking without a device.
library;

import 'planar_convert.dart';

/// One plane's place in a [PlanarUploadPlan].
class PlanarUploadSlot {
  const PlanarUploadSlot({required this.offset, required this.writeBytes});

  /// Byte offset of the plane's first row. A multiple of 4: the upload takes
  /// whole words and the kernel indexes u32s.
  final int offset;

  /// Bytes of the caller's plane that are uploaded: its declared rows
  /// ([PlanarPlane.byteLength]) when the list holds them, never more, and
  /// never fewer than the kernel reads.
  final int writeBytes;

  /// The whole words of [writeBytes], sent straight from the caller's list.
  int get bodyBytes => writeBytes & ~3;

  /// The 1-3 bytes after [bodyBytes], sent as one zero-padded word; 0 when
  /// [writeBytes] is whole words.
  int get tailBytes => writeBytes & 3;
}

/// Upload layout of a planar frame: one slot per plane the format uses.
class PlanarUploadPlan {
  const PlanarUploadPlan._(this.slots, this.totalBytes);

  final List<PlanarUploadSlot> slots;

  /// Size the upload buffer needs; a multiple of 4.
  final int totalBytes;

  /// Lays out the first `format.planeCount` of [planes] for a [width] x
  /// [height] picture, or returns null when they cannot hold it: a plane
  /// missing, a stride shorter than a row, or a plane shorter than the rows
  /// the kernel reads. Null beats uploading: a short plane made the native
  /// write throw or drop, and a short stride shears the picture.
  static PlanarUploadPlan? of({
    required PlanarFormat format,
    required List<PlanarPlane> planes,
    required int width,
    required int height,
  }) {
    if (width <= 0 || height <= 0 || planes.length < format.planeCount) {
      return null;
    }
    final slots = <PlanarUploadSlot>[];
    var total = 0;
    for (var i = 0; i < format.planeCount; i++) {
      final plane = planes[i];
      final (:rowBytes, :rows) = planarPlaneReads(format, i, width, height);
      if (plane.strideBytes < rowBytes) return null;
      // Every row but the last at full stride, then the last row's pixels:
      // padding after the last row is optional.
      final reads = plane.strideBytes * (rows - 1) + rowBytes;
      if (plane.bytes.length < reads) return null;
      final slot = plane.byteLength > reads ? plane.byteLength : reads;
      final write = plane.bytes.length < slot ? plane.bytes.length : slot;
      slots.add(PlanarUploadSlot(offset: total, writeBytes: write));
      total = (total + slot + 3) & ~3;
    }
    return PlanarUploadPlan._(List.unmodifiable(slots), total);
  }
}

/// Bytes per row and rows of plane [index] that the conversion kernel reads
/// for a [width] x [height] picture; mirrors the indexing in
/// [kPlanarToRgbaWgsl], so change them together.
({int rowBytes, int rows}) planarPlaneReads(
  PlanarFormat format,
  int index,
  int width,
  int height,
) {
  final chromaW = (width + 1) >> 1;
  final chromaH = (height + 1) >> 1;
  return switch (format) {
    PlanarFormat.nv12 when index == 0 => (rowBytes: width, rows: height),
    PlanarFormat.nv12 => (rowBytes: chromaW * 2, rows: chromaH),
    PlanarFormat.i420 when index == 0 => (rowBytes: width, rows: height),
    PlanarFormat.i420 => (rowBytes: chromaW, rows: chromaH),
    PlanarFormat.yuy2 => (rowBytes: chromaW * 4, rows: height),
    PlanarFormat.bgra8 => (rowBytes: width * 4, rows: height),
  };
}
