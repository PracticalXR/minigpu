/// Regression tests for MULTI-STREAM merging correctness.
///
/// Merging is MASTER-CLOCK driven: the first stream to deliver becomes the
/// master and every one of its frames produces a merged output that folds in
/// the LATEST snapshot of every other stream (fresh or one-frame stale).
/// Non-master frames never advance the cadence on their own — that would
/// double the merged frame rate — and a stalled master is replaced by the
/// freshest delivering stream so output degrades instead of freezing.  The
/// old all-streams barrier is exactly what these tests must NOT see again:
/// it tied cadence to pairing luck between independent device clocks, which
/// showed up as tiny gaps in multi-device spectrograms.
///
/// Content is checked with the sum strategy: 1+2=3 can only happen if both
/// streams' data reached the merge (a passthrough or aliased snapshot
/// cannot fake it).
library;

import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';
import 'mocks.dart';

/// Mock stream that feeds HOST data (TypedData), like the real AudioStream.
class HostMockMediaStream extends MockMediaStream {
  HostMockMediaStream(String id) : super(id, MediaStreamType.audio);

  void simulateHostUpdate(Float32List data, List<int> shape) {
    notifyHostData(data, shape);
  }
}

void main() {
  setUpAll(() async {
    if (!Minigpu().isInitialized) {
      await Minigpu().init();
    }
  });

  Future<Tensor> make(double v) async {
    final t = await Tensor.create([32]);
    await t.write(Float32List(32)..fillRange(0, 32, v));
    return t;
  }

  Future<Float32List?> readFinalMerged(Pipeline p) async {
    final out = p.getStageOutput('final', 'merged');
    if (out == null || out.isEmpty) return null;
    final t = out['output'] ?? out['input'] ?? out.values.first;
    return await t.getData() as Float32List;
  }

  test('master stream drives merge cadence; non-master frames ride along '
      'on the next master frame (GPU path)', () async {
    final pipeline = Pipeline(id: 'merge_content_gpu');
    pipeline.addStage(MockPipelineStage(stageId: 'pre'));
    pipeline.addMergeStage(
      StreamMergeStage(stageId: 'merge', strategy: MergeStrategy.sum),
    );

    final a = MockMediaStream('a', MediaStreamType.audio);
    final b = MockMediaStream('b', MediaStreamType.audio);
    pipeline.addStream(a);
    pipeline.addStream(b);
    await pipeline.start();

    // First deliverer becomes master and merges immediately — a slow or
    // silent second device must not stall the feed (the old barrier did).
    a.simulateTensorUpdate(await make(1.0));
    await Future.delayed(const Duration(milliseconds: 60));
    var merged = await readFinalMerged(pipeline);
    expect(merged, isNotNull, reason: 'master frame produced no merge');
    expect(
      merged!.every((v) => (v - 1.0).abs() < 1e-4),
      isTrue,
      reason: 'first merge should carry the master stream alone: '
          '${merged.take(4).toList()}',
    );

    // A non-master frame alone must NOT advance the cadence (that would
    // double the merged frame rate and distort the time axis).
    b.simulateTensorUpdate(await make(2.0));
    await Future.delayed(const Duration(milliseconds: 60));
    merged = await readFinalMerged(pipeline);
    expect(
      merged!.every((v) => (v - 1.0).abs() < 1e-4),
      isTrue,
      reason: 'non-master frame advanced the merge cadence on its own',
    );

    // The master's next frame folds in B's latest snapshot: sum(1,2)=3
    // everywhere proves both streams' data reached the merge.
    a.simulateTensorUpdate(await make(1.0));
    await Future.delayed(const Duration(milliseconds: 150));
    merged = await readFinalMerged(pipeline);
    expect(
      merged!.every((v) => (v - 3.0).abs() < 1e-4),
      isTrue,
      reason: 'merged content is not the sum of both streams: '
          '${merged.take(4).toList()}',
    );

    await pipeline.stop();
    await pipeline.dispose();
  });

  test('HOST-fed streams trigger merging (production audio path)', () async {
    final pipeline = Pipeline(id: 'merge_content_host');
    pipeline.addStage(MockPipelineStage(stageId: 'pre'));
    pipeline.addMergeStage(
      StreamMergeStage(stageId: 'merge', strategy: MergeStrategy.sum),
    );

    final a = HostMockMediaStream('ha');
    final b = HostMockMediaStream('hb');
    pipeline.addStream(a);
    pipeline.addStream(b);
    await pipeline.start();

    // a (master) merges alone, b snapshots, a's next frame merges both.
    a.simulateHostUpdate(Float32List(32)..fillRange(0, 32, 1.0), [32]);
    b.simulateHostUpdate(Float32List(32)..fillRange(0, 32, 2.0), [32]);
    await Future.delayed(const Duration(milliseconds: 100));
    a.simulateHostUpdate(Float32List(32)..fillRange(0, 32, 1.0), [32]);
    await Future.delayed(const Duration(milliseconds: 200));

    final merged = await readFinalMerged(pipeline);
    expect(
      merged,
      isNotNull,
      reason: 'host ingestion never triggered the merged stages',
    );
    expect(
      merged!.every((v) => (v - 3.0).abs() < 1e-4),
      isTrue,
      reason: 'merged content is not the sum of both host streams: '
          '${merged.take(4).toList()}',
    );

    await pipeline.stop();
    await pipeline.dispose();
  });

  test(
    'repeated frames keep merging both streams (snapshots do not alias)',
    () async {
      final pipeline = Pipeline(id: 'merge_content_repeat');
      pipeline.addStage(MockPipelineStage(stageId: 'pre'));
      pipeline.addMergeStage(
        StreamMergeStage(stageId: 'merge', strategy: MergeStrategy.sum),
      );

      final a = MockMediaStream('a', MediaStreamType.audio);
      final b = MockMediaStream('b', MediaStreamType.audio);
      pipeline.addStream(a);
      pipeline.addStream(b);
      await pipeline.start();

      // Establish 'a' as the merge master (first deliverer).
      a.simulateTensorUpdate(await make(1.0));
      await Future.delayed(const Duration(milliseconds: 100));

      // Several rounds with distinct values per round: if the per-stream
      // snapshots aliased a shared buffer, the sum would collapse to 2× one
      // stream's value instead of the true pairwise sum.  b delivers first,
      // then the master frame clocks the merge in.
      for (var round = 1; round <= 3; round++) {
        b.simulateTensorUpdate(await make(round * 10.0));
        a.simulateTensorUpdate(await make(round * 1.0));
        await Future.delayed(const Duration(milliseconds: 150));

        final merged = await readFinalMerged(pipeline);
        expect(merged, isNotNull, reason: 'round $round produced no merge');
        final expected = round * 11.0;
        expect(
          merged!.every((v) => (v - expected).abs() < 1e-3),
          isTrue,
          reason: 'round $round merged to ${merged.take(4).toList()}, '
              'expected $expected everywhere',
        );
      }

      await pipeline.stop();
      await pipeline.dispose();
    },
  );

  test('stalled master is replaced and its stale snapshot ages out', () async {
    final pipeline = Pipeline(id: 'merge_master_stall');
    pipeline.addStage(MockPipelineStage(stageId: 'pre'));
    pipeline.addMergeStage(
      StreamMergeStage(stageId: 'merge', strategy: MergeStrategy.sum),
    );

    final a = MockMediaStream('a', MediaStreamType.audio);
    final b = MockMediaStream('b', MediaStreamType.audio);
    pipeline.addStream(a);
    pipeline.addStream(b);
    await pipeline.start();

    // a becomes master, then goes silent (stopped device / silence-gated
    // loopback source).
    a.simulateTensorUpdate(await make(1.0));
    await Future.delayed(const Duration(milliseconds: 60));

    b.simulateTensorUpdate(await make(2.0));
    await Future.delayed(const Duration(milliseconds: 300));

    // b's next frame finds the master stalled past the fallback timeout:
    // b is promoted, and a's >250 ms-old snapshot is excluded — the merged
    // feed degrades to the surviving stream instead of freezing or
    // overlaying a's last spectrum forever.
    b.simulateTensorUpdate(await make(2.0));
    await Future.delayed(const Duration(milliseconds: 150));

    final merged = await readFinalMerged(pipeline);
    expect(merged, isNotNull, reason: 'promoted stream produced no merge');
    expect(
      merged!.every((v) => (v - 2.0).abs() < 1e-4),
      isTrue,
      reason: 'expected the surviving stream alone (stale master aged out), '
          'got ${merged.take(4).toList()}',
    );

    await pipeline.stop();
    await pipeline.dispose();
  });
}
