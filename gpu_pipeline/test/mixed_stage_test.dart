import 'dart:typed_data';
import 'dart:async';

import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_pipeline/src/pipeline_stage_builder.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

import 'mocks.dart';

void main() {
  group('Mixed Stage Routing Tests', () {
    late Pipeline pipeline;
    late MockMediaStream stream1;
    late MockMediaStream stream2;

    setUp(() async {
      // Ensure GPU is initialized
      if (!Minigpu().isInitialized) {
        await Minigpu().init();
      }

      pipeline = Pipeline(
        id: 'mixed_stage_pipeline',
        enableBenchmarking: true,
        debugEnabled: true,
      );

      stream1 = MockMediaStream('s1', MediaStreamType.audio);
      stream2 = MockMediaStream('s2', MediaStreamType.audio);
    });

    tearDown(() async {
      await pipeline.stop();
      await pipeline.dispose();
    });

    test(
      'Per-stream stage followed by merged stage produces merged output',
      () async {
        // Per-stream stage: simple addScalar(+1)
        final perStreamStage = StageBuilder(
          'per_stream_stage',
        ).op('addScalar', arguments: [1.0]).build();

        // Merged stage: trivial op (keeps shape); routed as merged
        // Note: merged processing sets merged 'input' to the first available stream tensor.
        final mergedStage = StageBuilder(
          'merged_stage',
        ).op('addScalar', arguments: [0.0]).build();

        pipeline.addStage(perStreamStage);
        pipeline.addMergeStage(mergedStage);

        pipeline.addStream(stream1);
        pipeline.addStream(stream2);

        await pipeline.start();

        // Prepare a tensor and push on both streams
        final N = 1024;
        final srcData = Float32List(N);
        for (int i = 0; i < N; i++) {
          srcData[i] = i.toDouble();
        }
        final t1 = await Tensor.create([N])
          ..write(srcData);
        final t2 = await Tensor.create([N])
          ..write(srcData);

        // Simulate updates on both streams
        stream1.simulateTensorUpdate(t1);
        stream2.simulateTensorUpdate(t2);

        // Give the pipeline some time to process both per-stream and merged stages
        await Future.delayed(const Duration(milliseconds: 150));

        // Check per-stream outputs exist
        final s1Out = pipeline.getStageOutput('per_stream_stage', 's1');
        final s2Out = pipeline.getStageOutput('per_stream_stage', 's2');
        expect(s1Out, isNotNull, reason: 's1 should have per-stream output');
        expect(s2Out, isNotNull, reason: 's2 should have per-stream output');

        // Check merged stage output exists under "merged"
        final mergedOut = pipeline.getStageOutput('merged_stage', 'merged');
        expect(
          mergedOut,
          isNotNull,
          reason: 'Merged stage should have produced a merged output',
        );

        // Also check final merged entry is present
        final finalMerged = pipeline.getStageOutput('final', 'merged');
        expect(
          finalMerged,
          isNotNull,
          reason: 'Final merged output should exist',
        );

        // Optional: Ensure we can enumerate per-stream outputs for per_stream_stage
        final allPerStream = pipeline.getStageOutputsForAllStreams(
          'per_stream_stage',
        );
        expect(allPerStream.keys.toSet(), containsAll(['s1', 's2']));
      },
    );

    test('Latest-only queue handles bursts without growing unbounded', () async {
      final stage = StageBuilder(
        'burst_stage',
      ).op('addScalar', arguments: [0.0]).build();

      final burstPipeline = Pipeline(
        id: 'burst_pipeline',
        enableBenchmarking: true,
        debugEnabled: true,
        queueConfig: const QueueConfig(
          strategy: QueueStrategy.latestOnly,
          maxItemsPerStream: 1,
          maxTotalItems: 8,
        ),
      );

      final s = MockMediaStream('burst', MediaStreamType.audio);

      burstPipeline.addStage(stage);
      burstPipeline.addStream(s);
      await burstPipeline.start();

      final N = 2048;
      final src = Float32List(N);
      final t = await Tensor.create([N])
        ..write(src);

      // Fire a burst of updates; latestOnly should keep at most 1 per stream queued
      for (int i = 0; i < 20; i++) {
        s.simulateTensorUpdate(t);
      }

      // Allow processing to drain
      await Future.delayed(const Duration(milliseconds: 200));

      // We only assert the pipeline is still alive and produced some output
      final out = burstPipeline.getStageOutput('burst_stage', 'burst');
      expect(out, isNotNull);

      await burstPipeline.stop();
      await burstPipeline.dispose();
    });
  });
}
