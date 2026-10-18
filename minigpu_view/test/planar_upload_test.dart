import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:minigpu_view/minigpu_view.dart';
import 'package:minigpu_view/src/planar_upload.dart';

PlanarPlane _plane(int length, int stride, int height) =>
    PlanarPlane(bytes: Uint8List(length), strideBytes: stride, height: height);

void main() {
  test('odd NV12 packs both planes at word-aligned offsets', () {
    final plan = PlanarUploadPlan.of(
      format: PlanarFormat.nv12,
      planes: [_plane(801 * 601, 801, 601), _plane(802 * 301, 802, 301)],
      width: 801,
      height: 601,
    )!;
    expect([for (final s in plan.slots) s.offset], [0, 481404]);
    expect([for (final s in plan.slots) s.writeBytes], [481401, 241402]);
    expect(plan.slots[0].bodyBytes, 481400);
    expect(plan.slots[0].tailBytes, 1);
    expect(plan.slots[1].tailBytes, 2);
    expect(plan.totalBytes, 722808);
  });

  test('a plane longer than its rows uploads its rows only', () {
    // A chroma list padded past its declared rows must not spill into the
    // next slot or past the buffer.
    final plan = PlanarUploadPlan.of(
      format: PlanarFormat.nv12,
      planes: [_plane(1920 * 1080, 1920, 1080), _plane(1920 * 544, 1920, 540)],
      width: 1920,
      height: 1080,
    )!;
    expect(plan.slots[1].offset, 2073600);
    expect(plan.slots[1].writeBytes, 1036800);
    expect(plan.totalBytes, 2073600 + 1036800);
  });

  test('the last row may end without its padding', () {
    final plan = PlanarUploadPlan.of(
      format: PlanarFormat.bgra8,
      planes: [_plane(16 + 12, 16, 2)],
      width: 3,
      height: 2,
    )!;
    expect(plan.slots.single.writeBytes, 28);
    // The slot still spans the declared rows.
    expect(plan.totalBytes, 32);
  });

  test('I420 chroma planes start on whole words', () {
    final plan = PlanarUploadPlan.of(
      format: PlanarFormat.i420,
      planes: [
        _plane(852 * 270, 852, 270),
        _plane(426 * 135, 426, 135),
        _plane(426 * 135, 426, 135),
      ],
      width: 852,
      height: 270,
    )!;
    expect([for (final s in plan.slots) s.offset], [0, 230040, 287552]);
    expect(plan.slots[1].writeBytes, 57510);
    expect(plan.slots[1].bodyBytes, 57508);
    expect(plan.slots[1].tailBytes, 2);
    expect(plan.totalBytes, 345064);
    for (final s in plan.slots) {
      expect(s.offset % 4, 0);
    }
  });

  test('odd-width YUY2 reads whole pixel pairs', () {
    final plan = PlanarUploadPlan.of(
      format: PlanarFormat.yuy2,
      planes: [_plane(24, 12, 2)],
      width: 5,
      height: 2,
    )!;
    expect(plan.slots.single.writeBytes, 24);
    expect(plan.totalBytes, 24);
    // Ten bytes a row would cut the last pair in half.
    expect(
      PlanarUploadPlan.of(
        format: PlanarFormat.yuy2,
        planes: [_plane(20, 10, 2)],
        width: 5,
        height: 2,
      ),
      isNull,
    );
  });

  test('planes that cannot hold the picture are refused', () {
    PlanarUploadPlan? nv12(List<PlanarPlane> planes) => PlanarUploadPlan.of(
          format: PlanarFormat.nv12,
          planes: planes,
          width: 4,
          height: 4,
        );
    final uv = _plane(8, 4, 2);
    expect(nv12([_plane(16, 4, 4), uv]), isNotNull);
    expect(nv12([_plane(15, 4, 4), uv]), isNull, reason: 'short luma');
    expect(nv12([_plane(16, 4, 4), _plane(7, 4, 2)]), isNull,
        reason: 'short chroma');
    expect(nv12([_plane(16, 3, 4), uv]), isNull, reason: 'stride < row');
    expect(nv12([_plane(16, 4, 4)]), isNull, reason: 'missing plane');
    expect(
      PlanarUploadPlan.of(
        format: PlanarFormat.i420,
        planes: [_plane(16, 4, 4), _plane(4, 2, 2)],
        width: 4,
        height: 4,
      ),
      isNull,
      reason: 'I420 needs three planes',
    );
    expect(
      PlanarUploadPlan.of(
        format: PlanarFormat.bgra8,
        planes: [_plane(16, 16, 1)],
        width: 0,
        height: 1,
      ),
      isNull,
      reason: 'empty picture',
    );
  });

  test('kernel read extents round odd sizes up', () {
    ({int rowBytes, int rows}) reads(PlanarFormat format, int index) =>
        planarPlaneReads(format, index, 5, 3);
    expect(reads(PlanarFormat.nv12, 0), (rowBytes: 5, rows: 3));
    expect(reads(PlanarFormat.nv12, 1), (rowBytes: 6, rows: 2));
    expect(reads(PlanarFormat.i420, 0), (rowBytes: 5, rows: 3));
    expect(reads(PlanarFormat.i420, 2), (rowBytes: 3, rows: 2));
    expect(reads(PlanarFormat.yuy2, 0), (rowBytes: 12, rows: 3));
    expect(reads(PlanarFormat.bgra8, 0), (rowBytes: 20, rows: 3));
  });
}
