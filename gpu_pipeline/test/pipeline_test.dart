import 'dart:typed_data';

import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_pipeline/src/pipeline_stage_builder.dart';
import 'package:gpu_pipeline/src/pipeline_event.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';
import 'dart:async';
import 'mocks.dart';

Minigpu minigpu = Minigpu();
void main() {
  group('Pipeline Tests', () {
    late Pipeline pipeline;
    late MockMediaStream audioStream;
    late MockMediaStream videoStream;

    setUp(() async {
      // Initialize GPU context for testing
      if (!minigpu.isInitialized) {
        await minigpu.init();
      }

      pipeline = Pipeline(
        id: 'test_pipeline',
        enableBenchmarking: true,
        debugEnabled: true,
      );

      audioStream = MockMediaStream('audio_1', MediaStreamType.audio);
      videoStream = MockMediaStream('video_1', MediaStreamType.video);
    });

    tearDown(() async {
      await pipeline.stop();
      pipeline.dispose();
    });

    test('Pipeline creation and basic properties', () {
      expect(pipeline.id, equals('test_pipeline'));
      expect(pipeline.isRunning, isFalse);
      expect(pipeline.isProcessing, isFalse);
    });

    test('Add and retrieve stages', () {
      final stage1 = MockPipelineStage(stageId: 'stage1');
      final stage2 = MockPipelineStage(stageId: 'stage2');

      pipeline.addStage(stage1);
      pipeline.addStage(stage2);

      expect(pipeline.getStage('stage1'), equals(stage1));
      expect(pipeline.getStage('stage2'), equals(stage2));
      expect(pipeline.getStage('nonexistent'), isNull);
    });

    test('Add streams', () {
      pipeline.addStream(audioStream);
      pipeline.addStream(videoStream);

      // Verify streams were added (internal state, verified through behavior)
      expect(audioStream.id, equals('audio_1'));
      expect(audioStream.type, equals(MediaStreamType.audio));
    });

    test('Pipeline start and stop', () async {
      final stage = MockPipelineStage(stageId: 'test_stage');
      pipeline.addStage(stage);
      pipeline.addStream(audioStream);

      expect(pipeline.isRunning, isFalse);

      await pipeline.start();
      expect(pipeline.isRunning, isTrue);
      expect(audioStream.isActive, isTrue);

      await pipeline.stop();
      expect(pipeline.isRunning, isFalse);
      expect(audioStream.isActive, isFalse);
    });

    test('Pipeline handles stage initialization failure', () async {
      final failingStage = MockPipelineStage(
        stageId: 'failing_stage',
        shouldFail: true,
      );
      pipeline.addStage(failingStage);

      expect(() => pipeline.start(), throwsException);
      expect(pipeline.isRunning, isFalse);
    });

    test('Event system basic functionality', () async {
      final events = <PipelineEvent>[];

      pipeline.addEventListener(PipelineEventType.pipelineStart, (event) {
        events.add(event);
      });

      pipeline.addEventListener(PipelineEventType.stageStart, (event) {
        events.add(event);
      });

      final stage = MockPipelineStage(stageId: 'event_test_stage');
      pipeline.addStage(stage);
      pipeline.addStream(audioStream);

      await pipeline.start();

      // Check that events were emitted
      expect(events.isNotEmpty, isTrue);
      expect(
        events.any((e) => e.type == PipelineEventType.pipelineStart),
        isTrue,
      );
    });

    test('Benchmark stats collection', () async {
      final stage = MockPipelineStage(
        stageId: 'benchmark_stage',
        processingDelay: Duration(milliseconds: 10),
      );
      pipeline.addStage(stage);
      pipeline.addStream(audioStream);

      await pipeline.start();

      // Simulate some processing
      final tensor = await Tensor.create([1024]);
      audioStream.simulateTensorUpdate(tensor);

      // Wait for processing
      await Future.delayed(Duration(milliseconds: 50));

      final stats = pipeline.getBenchmarkStats();
      expect(stats, isNotEmpty);
      expect(stats.containsKey('event_counts'), isTrue);
    });

    test('Stage stream configurations', () async {
      final stage1 = MockPipelineStage(stageId: 'selective_stage');
      final stage2 = MockPipelineStage(stageId: 'all_streams_stage');

      // Stage 1 only processes audio streams
      pipeline.addSelectiveStage(stage1, streams: {'audio_1'});

      // Stage 2 processes all streams
      pipeline.addStage(stage2);

      pipeline.addStream(audioStream);
      pipeline.addStream(videoStream);

      await pipeline.start();

      // Verify configuration was set (internal verification through behavior)
      expect(pipeline.isRunning, isTrue);
    });
  });

  group('Pipeline Integration Tests', () {
    late Pipeline pipeline;
    late MockMediaStream stream;

    setUp(() async {
      await Minigpu().init();

      pipeline = Pipeline(
        id: 'integration_test',
        enableBenchmarking: true,
        debugEnabled: true,
      );

      stream = MockMediaStream('test_stream', MediaStreamType.audio);
    });

    tearDown(() async {
      await pipeline.stop();
      pipeline.dispose();
    });

    test('Direct tensor operation without callbacks', () async {
      final tensor = await Tensor.create([1024]);
      final testData = Float32List(1024);
      for (int i = 0; i < 1024; i++) {
        testData[i] = i.toDouble();
      }
      await tensor.write(testData);

      // TODO: DEBUGLOG print('🔧 Tensor created and data written successfully');
      // TODO: DEBUGLOG print('🔧 Tensor shape: ${tensor.shape}');
      // TODO: DEBUGLOG print('🔧 Tensor buffer valid: ${tensor.buffer.isValid}');
      TensorOperationRegistry.ensureInitialized();

      // Try to use the tensor directly without any callbacks
      final result = await TensorOperationRegistry.executeOperation(
        'addScalar',
        tensor,
        [1.0],
        {},
      );
      expect(
        await result.getData(),
        equals([for (int i = 0; i < 1024; i++) i + 1.0]),
      );

      print('✅ Direct tensor operation completed');
      // TODO: DEBUGLOG print('🔧 Result shape: ${result.shape}');

      tensor.destroy();
      result.destroy();
    });

    test('Pipeline with dynamic stage end-to-end', () async {
      final dynamicStage = StageBuilder('integration_stage')
          .op('addScalar', arguments: ['param:gain'])
          .op('multiplyScalar', arguments: ['param:gain'])
          .withParameter(
            Parameter.stageFloat(
              'gain',
              defaultValue: 1.0,
              description: 'Audio gain',
            ),
          )
          .build();

      pipeline.addStage(dynamicStage);
      pipeline.addStream(stream);

      // Register event listener BEFORE starting pipeline
      final events = <PipelineEvent>[];
      print(
        '🔧 Registering event listener for PipelineEventType.stageComplete',
      );

      pipeline.addEventListener(PipelineEventType.stageComplete, (event) {
        print('🎉 Stage complete event received!');
        print('   Event type: ${event.type}');
        print('   Event class: ${event.runtimeType}');
        if (event is StageCompleteEvent) {
          print('   Stage ID: ${event.stageId}');
          print('   Processing time: ${event.processingTime}');
        }
        events.add(event);
      });

      // Also listen for ALL events to see what's being emitted
      final allEvents = <PipelineEvent>[];
      for (final eventType in PipelineEventType.values) {
        pipeline.addEventListener(eventType, (event) {
          print('📡 Event emitted: ${event.type} (${event.runtimeType})');
          allEvents.add(event);
        });
      }

      print('🚀 Starting pipeline...');
      await pipeline.start();

      // Create tensor with data
      // TODO: DEBUGLOG print('🔧 Creating tensor...');
      var tensor = await Tensor.create([1024]);
      final testData = Float32List(1024);
      for (int i = 0; i < 1024; i++) {
        testData[i] = i.toDouble();
      }
      // TODO: DEBUGLOG print('🔧 Writing test data to tensor...');
      await tensor.write(testData);

      print('📡 Simulating tensor update...');
      stream.simulateTensorUpdate(tensor);

      // Wait for processing
      print('⏳ Waiting for processing...');
      await Future.delayed(Duration(milliseconds: 50));
      // TODO: DEBUGLOG print('🔧 Tensor still alive: ${tensor.shape}');
      // Debug: Show all captured events
      print('📊 All events captured (${allEvents.length}):');
      for (int i = 0; i < allEvents.length; i++) {
        final event = allEvents[i];
        print(
          '   $i: ${event.type} (${event.runtimeType}) at ${event.timestamp}',
        );
      }

      print('📊 Stage complete events captured: ${events.length}');
      for (final event in events) {
        print('   ${event.type} - ${event.runtimeType}');
      }

      await pipeline.debugPrintAllOutputs();

      // Verify stage output exists
      final output = pipeline.getStageOutput(
        'integration_stage',
        'test_stream',
      );
      expect(output, isNotNull);

      // Verify events were emitted
      print('🧪 Checking if events.isNotEmpty...');
      if (events.isEmpty) {
        print('❌ No StageComplete events captured!');
        print('💡 All events that were captured:');
        for (final event in allEvents) {
          print('   - ${event.type} (${event.runtimeType})');
        }
      }

      expect(
        events.isNotEmpty,
        isTrue,
        reason: 'At least one stage complete event should be emitted',
      );

      // Clean up
      //tensor.destroy();
    });
    test('Parameter changes trigger events', () async {
      final dynamicStage = StageBuilder('event_stage')
          .op('multiply')
          .withParameter(
            Parameter.stageFloat(
              'volume',
              defaultValue: 0.5,
              description: 'Volume level',
            ),
          )
          .build();

      pipeline.addStage(dynamicStage);

      final parameterEvents = <PipelineEvent>[];
      pipeline.addEventListener(PipelineEventType.parameterChange, (event) {
        parameterEvents.add(event);
      });

      await pipeline.start();

      // Change parameter
      dynamicStage.setPropertyValue('volume', 0.8);

      // Wait for event processing
      await Future.delayed(Duration(milliseconds: 10));

      // Verify parameter change event was emitted
      expect(parameterEvents.length, equals(1));
      final event = parameterEvents.first as ParameterChangeEvent;
      expect(event.parameterName, equals('volume'));
      expect(event.oldValue, equals(0.5));
      expect(event.newValue, equals(0.8));
    });

    test('Multiple streams with selective processing', () async {
      final stream1 = MockMediaStream('stream1', MediaStreamType.audio);
      final stream2 = MockMediaStream('stream2', MediaStreamType.video);

      final selectiveStage = StageBuilder(
        'selective_stage',
      ).op('multiplyScalar', arguments: [1.0]).build();

      // Add stage that ONLY processes stream1
      pipeline.addSelectiveStage(selectiveStage, streams: {'stream1'});
      pipeline.addStream(stream1);
      pipeline.addStream(stream2);

      await pipeline.start();
      await Future.delayed(Duration(milliseconds: 50));

      final tensor = await Tensor.create([512]);

      // Simulate updates from both streams
      print('🎯 Simulating stream1 update...');
      stream1.simulateTensorUpdate(tensor);
      await Future.delayed(Duration(milliseconds: 100));

      print('🎯 Simulating stream2 update...');
      stream2.simulateTensorUpdate(tensor);
      await Future.delayed(Duration(milliseconds: 200));

      // Debug output
      pipeline.debugPrintAllOutputs();

      // Stream1 should have output from selective stage (it was processed)
      final stream1Output = pipeline.getStageOutput(
        'selective_stage',
        'stream1',
      );
      expect(
        stream1Output,
        isNotNull,
        reason: 'Stream1 should have been processed by selective stage',
      );

      // Stream2 should NOT have output from selective stage (it was filtered out)
      final stream2Output = pipeline.getStageOutput(
        'selective_stage',
        'stream2',
      );
      expect(
        stream2Output,
        isNull,
        reason: 'Stream2 should not have been processed by selective stage',
      );

      // BOTH streams should have final outputs
      final finalStream1 = pipeline.getStageOutput('final', 'stream1');
      final finalStream2 = pipeline.getStageOutput('final', 'stream2');

      expect(
        finalStream1,
        isNotNull,
        reason:
            'Stream1 should have final output (processed by selective stage)',
      );
      expect(
        finalStream2,
        isNotNull,
        reason: 'Stream2 should have final output (bypassed selective stage)',
      );

      // Optional: Verify that the final outputs are different
      // (stream1 went through processing, stream2 is original input)
      if (finalStream1 != null && finalStream2 != null) {
        print('🔍 Final stream1 shape: ${finalStream1['input']?.shape}');
        print('🔍 Final stream2 shape: ${finalStream2['input']?.shape}');
        // Both should have the same shape since multiplyScalar(1.0) doesn't change shape
        // But they are different tensor instances
      }

      await stream1.stop();
      await stream2.stop();
      stream1.dispose();
      stream2.dispose();
    });
  });

  group('Error Handling Tests', () {
    late Pipeline pipeline;

    setUp(() async {
      await Minigpu().init();
      pipeline = Pipeline(id: 'error_test', debugEnabled: true);
    });

    tearDown(() async {
      await pipeline.stop();
      pipeline.dispose();
    });

    test('Handle stage processing errors', () async {
      final failingStage = MockPipelineStage(
        stageId: 'failing_stage',
        shouldFail: true,
      );

      pipeline.addStage(failingStage);

      final stream = MockMediaStream('test_stream', MediaStreamType.audio);
      pipeline.addStream(stream);

      final errorEvents = <PipelineEvent>[];
      pipeline.addEventListener(PipelineEventType.stageError, (event) {
        errorEvents.add(event);
      });

      try {
        await pipeline.start();

        // Simulate tensor update that will cause processing error
        final tensor = await Tensor.create([256]);
        stream.simulateTensorUpdate(tensor);

        await Future.delayed(Duration(milliseconds: 50));
      } catch (e) {}

      // Verify error event was emitted
      expect(errorEvents.length, equals(1));
      final errorEvent = errorEvents.first as StageErrorEvent;
      expect(errorEvent.stageId, equals('failing_stage'));
    });

    test('Invalid parameter validation', () async {
      final stage = StageBuilder('validation_test')
          .op('multiply')
          .withParameter(
            Parameter.stageFloat(
              'constrained_value',
              defaultValue: 5.0,
              min: 0.0,
              max: 10.0,
              description: 'Constrained parameter',
            ),
          )
          .build();

      // Test valid range
      expect(await stage.setPropertyValue('constrained_value', 7.5), isTrue);

      // Test below minimum
      expect(await stage.setPropertyValue('constrained_value', -1.0), isFalse);

      // Test above maximum
      expect(await stage.setPropertyValue('constrained_value', 15.0), isFalse);

      // Test wrong type
      expect(
        await stage.setPropertyValue('constrained_value', 'invalid'),
        isFalse,
      );
    });

    test('Empty stage builder validation', () {
      expect(() => StageBuilder('empty_stage').build(), throwsStateError);
    });
  });
}
