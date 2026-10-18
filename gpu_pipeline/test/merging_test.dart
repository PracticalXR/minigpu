import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_pipeline/src/pipeline_stage_builder.dart';
import 'package:gpu_pipeline/src/pipeline_event.dart';
import 'package:gpu_pipeline/src/stages/stream_merge_stage.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';
import 'dart:async';
import 'mocks.dart';

void main() {
  group('Pipeline Merging Tests', () {
    late Pipeline pipeline;
    late MockMediaStream audioStream;
    late MockMediaStream videoStream;
    late MockMediaStream screenStream;

    setUp(() async {
      await Minigpu().init();

      pipeline = Pipeline(
        id: 'merging_test',
        enableBenchmarking: true,
        debugEnabled: true,
      );

      audioStream = MockMediaStream('audio_stream', MediaStreamType.audio);
      videoStream = MockMediaStream('video_stream', MediaStreamType.video);
      screenStream = MockMediaStream('screen_stream', MediaStreamType.screen);
    });

    tearDown(() async {
      await pipeline.stop();
      pipeline.dispose();

      // Dispose streams
      audioStream.dispose();
      videoStream.dispose();
      screenStream.dispose();
    });

    group('Basic Merging', () {
      test('Compatible tensor shapes merge successfully', () async {
        final mergeStage = StreamMergeStage(
          stageId: 'merge_stage',
          strategy: MergeStrategy.average,
        );

        pipeline.addMergeStage(mergeStage);
        pipeline.addStream(audioStream);
        pipeline.addStream(videoStream);

        await pipeline.start();
        await Future.delayed(Duration(milliseconds: 50));

        // Use compatible shapes for successful merging
        final compatibleShape = [512];
        final audioTensor = await Tensor.create(
          compatibleShape,
          gpu: DefaultMinigpu.instance,
        );
        final videoTensor = await Tensor.create(
          compatibleShape,
          gpu: DefaultMinigpu.instance,
        );

        print(
          '🎯 Testing compatible tensor merge with shape: $compatibleShape',
        );

        // Send tensor updates
        audioStream.simulateTensorUpdate(audioTensor);
        await Future.delayed(Duration(milliseconds: 50));

        videoStream.simulateTensorUpdate(videoTensor);
        await Future.delayed(Duration(milliseconds: 200));

        pipeline.debugPrintAllOutputs();

        // Check for merged outputs
        final mergedOutput = pipeline.getStageOutput('merge_stage', 'merged');
        expect(
          mergedOutput,
          isNotNull,
          reason: 'Compatible tensors should merge successfully',
        );

        final finalMerged = pipeline.getStageOutput('final', 'merged');
        expect(
          finalMerged,
          isNotNull,
          reason: 'Should have final merged output',
        );

        print('✅ Compatible tensor merging successful');
      });

      test('Incompatible tensor shapes handled gracefully', () async {
        final mergeStage = StreamMergeStage(
          stageId: 'merge_stage',
          strategy: MergeStrategy.average,
        );

        pipeline.addMergeStage(mergeStage);
        pipeline.addStream(audioStream);
        pipeline.addStream(videoStream);

        final errorEvents = <PipelineEvent>[];
        pipeline.addEventListener(PipelineEventType.pipelineError, (event) {
          errorEvents.add(event);
        });

        await pipeline.start();
        await Future.delayed(Duration(milliseconds: 50));

        // Use incompatible shapes
        final audioTensor = await Tensor.create([1024]); // 1D
        final videoTensor = await Tensor.create([640, 480]); // 2D

        print('🎯 Testing incompatible tensor shapes: [1024] vs [640, 480]');

        audioStream.simulateTensorUpdate(audioTensor);
        await Future.delayed(Duration(milliseconds: 50));

        videoStream.simulateTensorUpdate(videoTensor);
        await Future.delayed(Duration(milliseconds: 200));

        pipeline.debugPrintAllOutputs();

        // Should handle the error gracefully
        // Either succeed with fallback or emit error event
        final hasOutput = pipeline.hasStageOutput('merge_stage');
        final hasErrors = errorEvents.isNotEmpty;

        expect(
          hasOutput || hasErrors,
          isTrue,
          reason: 'Should either produce fallback output or emit error event',
        );

        print('✅ Incompatible shapes handled gracefully');
      });

      test('Single stream passes through merge stage', () async {
        final mergeStage = StreamMergeStage(
          stageId: 'single_merge',
          strategy: MergeStrategy.average,
        );

        pipeline.addMergeStage(mergeStage);
        pipeline.addStream(audioStream); // Only one stream

        await pipeline.start();
        await Future.delayed(Duration(milliseconds: 50));

        final tensor = await Tensor.create([256]);
        audioStream.simulateTensorUpdate(tensor);
        await Future.delayed(Duration(milliseconds: 200));

        final output = pipeline.getStageOutput('single_merge', 'merged');
        expect(
          output,
          isNotNull,
          reason: 'Single stream should pass through merge stage',
        );
      });
    });

    // Helper method for testing different merge strategies
    Future<void> _testMergeStrategy(
      MergeStrategy strategy,
      String stageId,
    ) async {
      final mergeStage = StreamMergeStage(stageId: stageId, strategy: strategy);

      pipeline.addMergeStage(mergeStage);
      pipeline.addStream(audioStream);
      pipeline.addStream(videoStream);

      await pipeline.start();
      await Future.delayed(Duration(milliseconds: 50));

      final compatibleShape = [128];
      final tensor1 = await Tensor.create(compatibleShape);
      final tensor2 = await Tensor.create(compatibleShape);

      audioStream.simulateTensorUpdate(tensor1);
      videoStream.simulateTensorUpdate(tensor2);
      await Future.delayed(Duration(milliseconds: 200));

      final output = pipeline.getStageOutput(stageId, 'merged');
      expect(
        output,
        isNotNull,
        reason: '$strategy merge should produce output',
      );

      print('✅ ${strategy.name} merge strategy successful');
    }

    group('Merge Strategies', () {
      test('Average merge strategy', () async {
        await _testMergeStrategy(MergeStrategy.average, 'average_merge');
      });

      test('Sum merge strategy', () async {
        await _testMergeStrategy(MergeStrategy.sum, 'sum_merge');
      });

      test('Maximum merge strategy', () async {
        await _testMergeStrategy(MergeStrategy.maximum, 'max_merge');
      });

      test('Minimum merge strategy', () async {
        await _testMergeStrategy(MergeStrategy.minimum, 'min_merge');
      });
    });

    group('Stream Synchronization', () {
      test('Merge handles stream timing differences', () async {
        final timedMerge = StreamMergeStage(
          stageId: 'timed_merge',
          strategy: MergeStrategy.average,
        );

        pipeline.addMergeStage(timedMerge);
        pipeline.addStream(audioStream);
        pipeline.addStream(videoStream);

        await pipeline.start();
        await Future.delayed(Duration(milliseconds: 50));

        final tensor = await Tensor.create([64]);

        // Send updates with different timing
        audioStream.simulateTensorUpdate(tensor);
        await Future.delayed(Duration(milliseconds: 25));

        videoStream.simulateTensorUpdate(tensor);
        await Future.delayed(Duration(milliseconds: 175));

        final output = pipeline.getStageOutput('timed_merge', 'merged');
        expect(
          output,
          isNotNull,
          reason: 'Should handle different stream timing',
        );
      });
    });

    group('Mixed Routing', () {
      test('Selective then merged routing', () async {
        // Selective stage (audio only)
        final selectiveStage = StageBuilder(
          'selective_stage',
        ).op('multiplyScalar', arguments: [2.0]).build();

        // Merged stage (all available streams)
        final mergeStage = StreamMergeStage(
          stageId: 'merge_stage',
          strategy: MergeStrategy.average,
        );

        pipeline.addSelectiveStage(selectiveStage, streams: {'audio_stream'});
        pipeline.addMergeStage(mergeStage);
        pipeline.addStream(audioStream);
        pipeline.addStream(videoStream);

        await pipeline.start();
        await Future.delayed(Duration(milliseconds: 50));

        final tensor = await Tensor.create([256]);

        audioStream.simulateTensorUpdate(tensor);
        videoStream.simulateTensorUpdate(tensor);
        await Future.delayed(Duration(milliseconds: 200));

        // Selective stage should only process audio
        final selectiveAudio = pipeline.getStageOutput(
          'selective_stage',
          'audio_stream',
        );
        final selectiveVideo = pipeline.getStageOutput(
          'selective_stage',
          'video_stream',
        );

        expect(
          selectiveAudio,
          isNotNull,
          reason: 'Selective stage should process audio',
        );
        expect(
          selectiveVideo,
          isNull,
          reason: 'Selective stage should not process video',
        );

        // Merge stage should combine available streams
        final mergedOutput = pipeline.getStageOutput('merge_stage', 'merged');
        expect(
          mergedOutput,
          isNotNull,
          reason: 'Merge stage should combine available streams',
        );
      });
    });

    group('Error Handling', () {
      test('Merge stage handles processing errors', () async {
        final failingMerge = MockPipelineStage(
          stageId: 'failing_merge',
          shouldFail: true,
          failDuringProcessing: true,
        );

        pipeline.addMergeStage(failingMerge);
        pipeline.addStream(audioStream);
        pipeline.addStream(videoStream);

        final errorEvents = <PipelineEvent>[];
        pipeline.addEventListener(PipelineEventType.stageError, (event) {
          errorEvents.add(event);
        });

        await pipeline.start();
        await Future.delayed(Duration(milliseconds: 50));

        final tensor = await Tensor.create([128]);

        audioStream.simulateTensorUpdate(tensor);
        videoStream.simulateTensorUpdate(tensor);
        await Future.delayed(Duration(milliseconds: 200));

        expect(
          errorEvents.isNotEmpty,
          isTrue,
          reason: 'Should emit error events for failing stage',
        );

        final output = pipeline.getStageOutput('failing_merge', 'merged');
        expect(
          output,
          isNull,
          reason: 'Failing stage should not produce output',
        );
      });

      test('Pipeline continues after merge errors', () async {
        // First stage that works
        final workingStage = StageBuilder(
          'working_stage',
        ).op('addScalar', arguments: [1.0]).build();

        // Second stage that fails
        final failingMerge = MockPipelineStage(
          stageId: 'failing_merge',
          shouldFail: true,
          failDuringProcessing: true,
        );

        pipeline.addMergeStage(workingStage);
        pipeline.addMergeStage(failingMerge);
        pipeline.addStream(audioStream);

        await pipeline.start();
        await Future.delayed(Duration(milliseconds: 50));

        final tensor = await Tensor.create([64]);
        audioStream.simulateTensorUpdate(tensor);
        await Future.delayed(Duration(milliseconds: 200));

        // First stage should work
        final workingOutput = pipeline.getStageOutput(
          'working_stage',
          'merged',
        );
        expect(
          workingOutput,
          isNotNull,
          reason: 'Working stage should produce output',
        );

        // Pipeline should still be running
        expect(
          pipeline.isRunning,
          isTrue,
          reason: 'Pipeline should continue running after stage errors',
        );
      });
    });

    group('Performance', () {
      test('Benchmark events emitted for merged stages', () async {
        final benchmarkMerge = StageBuilder('benchmark_merge')
            .op('sqrt') // More computationally expensive
            .build();

        pipeline.addMergeStage(benchmarkMerge);
        pipeline.addStream(audioStream);
        pipeline.addStream(videoStream);

        final benchmarkEvents = <PipelineEvent>[];
        pipeline.addEventListener(PipelineEventType.benchmark, (event) {
          benchmarkEvents.add(event);
        });

        await pipeline.start();
        await Future.delayed(Duration(milliseconds: 50));

        final largeTensor = await Tensor.create([2048]);

        // Process multiple times for benchmark data
        for (int i = 0; i < 3; i++) {
          audioStream.simulateTensorUpdate(largeTensor);
          videoStream.simulateTensorUpdate(largeTensor);
          await Future.delayed(Duration(milliseconds: 100));
        }

        expect(
          benchmarkEvents.isNotEmpty,
          isTrue,
          reason: 'Should emit benchmark events',
        );

        if (benchmarkEvents.isNotEmpty) {
          final event = benchmarkEvents.first as PipelineBenchmarkEvent;
          expect(event.duration.inMicroseconds, greaterThan(0));
          print('🏁 Merge benchmark: ${event.duration.inMicroseconds}μs');
        }
      });

      test('Multiple stream merge performance', () async {
        final perfMerge = StreamMergeStage(
          stageId: 'perf_merge',
          strategy: MergeStrategy.sum,
        );

        pipeline.addMergeStage(perfMerge);
        pipeline.addStream(audioStream);
        pipeline.addStream(videoStream);
        pipeline.addStream(screenStream);

        await pipeline.start();
        await Future.delayed(Duration(milliseconds: 50));

        final tensor = await Tensor.create([1024]);
        final startTime = DateTime.now();

        // Process all three streams
        audioStream.simulateTensorUpdate(tensor);
        videoStream.simulateTensorUpdate(tensor);
        screenStream.simulateTensorUpdate(tensor);

        await Future.delayed(Duration(milliseconds: 300));

        final endTime = DateTime.now();
        final duration = endTime.difference(startTime);

        final output = pipeline.getStageOutput('perf_merge', 'merged');
        expect(output, isNotNull, reason: 'Three-stream merge should complete');

        print('🏁 Three-stream merge took: ${duration.inMilliseconds}ms');
        expect(
          duration.inMilliseconds,
          lessThan(1000),
          reason: 'Merge should complete within reasonable time',
        );
      });
    });

    group('Parameter Control', () {
      test('Dynamic parameter changes in merged stages', () async {
        final paramMerge = StageBuilder('param_merge')
            .op('multiplyScalar', arguments: ['param:factor'])
            .withParameter(
              Parameter.stageFloat(
                'factor',
                defaultValue: 1.0,
                min: 0.1,
                max: 5.0,
                description: 'Merge scaling factor',
              ),
            )
            .build();

        pipeline.addMergeStage(paramMerge);
        pipeline.addStream(audioStream);
        pipeline.addStream(videoStream);

        final paramEvents = <PipelineEvent>[];
        pipeline.addEventListener(PipelineEventType.parameterChange, (event) {
          paramEvents.add(event);
        });

        await pipeline.start();
        await Future.delayed(Duration(milliseconds: 50));

        final tensor = await Tensor.create([256]);

        // Process with default parameter
        audioStream.simulateTensorUpdate(tensor);
        videoStream.simulateTensorUpdate(tensor);
        await Future.delayed(Duration(milliseconds: 100));

        var output = pipeline.getStageOutput('param_merge', 'merged');
        expect(output, isNotNull, reason: 'Should work with default parameter');

        // Change parameter
        paramMerge.setPropertyValue('factor', 2.5);
        await Future.delayed(Duration(milliseconds: 50));

        // Process with new parameter
        audioStream.simulateTensorUpdate(tensor);
        videoStream.simulateTensorUpdate(tensor);
        await Future.delayed(Duration(milliseconds: 100));

        output = pipeline.getStageOutput('param_merge', 'merged');
        expect(output, isNotNull, reason: 'Should work with changed parameter');

        expect(
          paramEvents.isNotEmpty,
          isTrue,
          reason: 'Should emit parameter change event',
        );
      });
    });
  });
}
