import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_pipeline/src/pipeline_ports.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';
import 'dart:async';
import 'mocks.dart';

enum ProcessingMode { fast, quality }

void main() {
  group('Quick Guide Integration Tests', () {
    late Minigpu minigpu;

    setUpAll(() async {
      minigpu = Minigpu();
      if (!minigpu.isInitialized) {
        await minigpu.init();
      }
      EnumRegistry.register<ProcessingMode>(ProcessingMode.values);
    });

    group('Pipeline Creation and Basic Properties', () {
      test('Pipeline creation from quick guide example', () {
        final pipeline = Pipeline(
          id: 'my_pipeline',
          enableBenchmarking: true,
          debugEnabled: false,
          queueConfig: QueueConfig(strategy: QueueStrategy.latestOnly),
        );

        expect(pipeline.id, equals('my_pipeline'));
        expect(pipeline.isRunning, isFalse);
        expect(pipeline.isProcessing, isFalse);

        pipeline.dispose();
      });

      test('Pipeline key methods', () async {
        final pipeline = Pipeline(id: 'test_methods');
        final mockStage = MockPipelineStage(stageId: 'test_stage');
        final mockStream = MockMediaStream(
          'test_stream',
          MediaStreamType.audio,
        );

        pipeline.addStage(mockStage);
        pipeline.addStream(mockStream);

        // Test isRunning before start
        expect(pipeline.isRunning, isFalse);

        // Test start and isRunning
        await pipeline.start();
        expect(pipeline.isRunning, isTrue);

        // Test getStage
        expect(pipeline.getStage('test_stage'), equals(mockStage));
        expect(pipeline.getStage('nonexistent'), isNull);

        // Test getBenchmarkStats
        final stats = pipeline.getBenchmarkStats();
        expect(stats, isMap);

        // Test stop
        await pipeline.stop();
        expect(pipeline.isRunning, isFalse);

        pipeline.dispose();
      });
    });

    group('Event System', () {
      test('Direct event listener pattern from quick guide', () async {
        final pipeline = Pipeline(id: 'event_test', debugEnabled: true);
        final mockStage = MockPipelineStage(stageId: 'event_stage');
        final mockStream = MockMediaStream(
          'event_stream',
          MediaStreamType.audio,
        );

        pipeline.addStage(mockStage);
        pipeline.addStream(mockStream);

        final capturedEvents = <PipelineEvent>[];

        // Test the direct event listener pattern from the guide
        final subscription = pipeline.eventNotifier.eventStream.listen((event) {
          capturedEvents.add(event);
          if (event is PipelineRunCompleteEvent) {
            print('Pipeline completed one full processing cycle');
          } else if (event is PipelineErrorEvent) {
            print('Error: ${event.error}');
          } else if (event is PipelineBenchmarkEvent) {
            print('Performance data available');
          }
        });

        await pipeline.start();

        // Simulate processing
        final tensor = await Tensor.create([1024]);
        mockStream.simulateTensorUpdate(tensor);
        await Future.delayed(Duration(milliseconds: 100));

        // Verify events were captured
        expect(capturedEvents.isNotEmpty, isTrue);
        expect(
          capturedEvents.any((e) => e.type == PipelineEventType.pipelineStart),
          isTrue,
        );

        await subscription.cancel();
        await pipeline.stop();
        pipeline.dispose();
      });

      test('All event types from quick guide', () async {
        final pipeline = Pipeline(id: 'all_events_test');
        final mockStage = MockPipelineStage(stageId: 'test_stage');
        final mockStream = MockMediaStream(
          'test_stream',
          MediaStreamType.audio,
        );

        pipeline.addStage(mockStage);
        pipeline.addStream(mockStream);

        final eventCounts = <PipelineEventType, int>{};

        final subscription = pipeline.eventNotifier.eventStream.listen((event) {
          eventCounts[event.type] = (eventCounts[event.type] ?? 0) + 1;
        });

        await pipeline.start();

        // Trigger some processing
        final tensor = await Tensor.create([512]);
        mockStream.simulateTensorUpdate(tensor);
        await Future.delayed(Duration(milliseconds: 50));

        await pipeline.stop();

        // Verify expected event types were emitted
        expect(eventCounts[PipelineEventType.pipelineStart], greaterThan(0));

        await subscription.cancel();
        pipeline.dispose();
      });
    });

    group('StageBuilder API', () {
      test('Input ports from quick guide', () {
        final stage = StageBuilder('input_test')
            .withAudioInput('audio', description: 'Audio samples')
            .withVideoInput('video', minWidth: 640, maxWidth: 1920)
            .withFlexibleInput('flexible', minDimensions: 2, maxDimensions: 4)
            .withExactInput('exact', shape: [256, 256, 3])
            .withStaticInput('weights', assetPath: 'weights.bin')
            .withCustomInput(
              'custom',
              validator: (tensor) {
                // Return ValidationResult instead of bool
                return tensor != null && tensor.shape.isNotEmpty
                    ? ValidationResult.valid()
                    : ValidationResult.invalid('Invalid tensor');
              },
            )
            .op('normalize')
            .build();

        expect(stage.stageId, equals('input_test'));
        expect(stage.inputPorts.length, greaterThan(0));
      });

      test('Output ports from quick guide', () {
        final stage = StageBuilder('output_test')
            .withDynamicOutput(
              'dynamic',
              shapeCalculator: (inputs, params) => [512, 512],
            )
            .withFixedOutput('fixed', shape: [1024])
            .withTextureOutput('texture', width: 512, height: 512, channels: 4)
            .withHistoryOutput('history', timeSlices: 120)
            .withParameterOutput(
              'parameter',
              dependencies: ['gain', 'threshold'],
              calculator: (inputs, params) => [100],
            )
            .op('normalize')
            .build();

        expect(stage.stageId, equals('output_test'));
        expect(stage.outputPorts.length, greaterThan(0));
      });

      test('Parameters from quick guide', () {
        final stage = StageBuilder('param_test')
            .withParameter(
              Parameter.stageFloat(
                'gain',
                defaultValue: 1.0,
                min: 0.0,
                max: 2.0,
                description: 'Audio gain',
              ),
            )
            .withParameter(
              Parameter.stageInt(
                'iterations',
                defaultValue: 10,
                min: 1,
                max: 100,
                description: 'Number of iterations',
              ),
            )
            .withParameter(
              Parameter.stageBool(
                'enabled',
                defaultValue: true,
                description: 'Enable processing',
              ),
            )
            .withParameter(
              Parameter.stageVec3(
                'position',
                defaultValue: [0, 0, 0],
                description: 'Position vector',
              ),
            )
            .withParameter(
              Parameter.stageEnum<ProcessingMode>(
                'mode',
                defaultValue: ProcessingMode.fast,
                allValues:
                    ProcessingMode.values, // This provides all enum values
                description: 'Processing mode',
              ),
            )
            .op('normalize')
            .build();

        expect(stage.stageId, equals('param_test'));
        expect(stage.parameterDefinitions.length, equals(5));

        // Test parameter access
        expect(stage.getPropertyValue<double>('gain'), equals(1.0));
        expect(stage.getPropertyValue<int>('iterations'), equals(10));
        expect(stage.getPropertyValue<bool>('enabled'), equals(true));
        expect(
          stage.getPropertyValue<List<double>>('position'),
          equals([0, 0, 0]),
        );
        expect(
          stage.getPropertyValue<ProcessingMode>('mode'),
          equals(ProcessingMode.fast),
        );
      });

      test('Built-in operations from quick guide', () {
        final stage = StageBuilder('operations_test')
            .op('fft')
            .op('multiplyScalar', arguments: [2.0])
            .op('abs')
            .op('sqrt')
            .op('power', arguments: [2.0])
            .op('clamp', arguments: [0.0, 1.0])
            .op('normalize')
            .build();

        expect(stage.stageId, equals('operations_test'));
        expect(stage.operations.length, equals(7));
      });

      test('Shader execution from quick guide', () {
        final shaderCode = '''
          @group(0) @binding(0) var<storage, read_write> input: array<f32>;
          @compute @workgroup_size(64)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let index = gid.x;
            if (index >= arrayLength(&input)) { return; }
            input[index] = input[index] * 2.0;
          }
        ''';

        final stage = StageBuilder(
          'shader_test',
        ).executeShader(shaderCode).build();

        expect(stage.stageId, equals('shader_test'));
        expect(stage.operations.isNotEmpty, isTrue);
      });

      test('Shader with parameters from quick guide', () {
        final shaderCode = '''
          @group(0) @binding(0) var<storage, read_write> input: array<f32>;
          @compute @workgroup_size(64)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let index = gid.x;
            if (index >= arrayLength(&input)) { return; }
            input[index] = input[index] * GAIN;
          }
        ''';

        final stage = StageBuilder('shader_param_test')
            .withParameter(
              Parameter.stageFloat(
                'gain',
                defaultValue: 2.0,
                description: 'Shader gain',
              ),
            )
            .executeShader(
              shaderCode,
              shaderParameters: {'GAIN': 'param:gain', 'WIDTH': 'const:512'},
            )
            .build();

        expect(stage.stageId, equals('shader_param_test'));
        expect(stage.parameterDefinitions.containsKey('gain'), isTrue);
      });

      test('Generated input from quick guide', () {
        final stage = StageBuilder('generated_test')
            .withGeneratedInput(
              'noise',
              shape: [1024],
              type: 'gaussian_noise',
              description: 'Generated noise input',
              generatorParams: {'mean': 0.0, 'stddev': 1.0, 'seed': 12345},
            )
            .op('abs')
            .build();

        expect(stage.stageId, equals('generated_test'));
        expect(stage.inputPorts.length, equals(1));
        expect(stage.inputPorts.first.name, equals('noise'));
      });

      test('Asset management from quick guide', () {
        final stage = StageBuilder('asset_test')
            .withWeights('model_weights', path: 'assets/weights.bin')
            .withConfig('model_config', path: 'assets/config.json')
            .withTexture('lookup_table', path: 'assets/lut.png')
            .withLookupTable('gamma_table', path: 'assets/gamma.lut')
            .op('normalize')
            .build();

        expect(stage.stageId, equals('asset_test'));
        expect(stage.requiredAssets.length, equals(4));
      });
    });

    group('Runtime Stage Control', () {
      test('Async parameter updates from quick guide', () async {
        final stage = StageBuilder('runtime_test')
            .withParameter(
              Parameter.stageFloat(
                'gain',
                defaultValue: 1.0,
                min: 0.0,
                max: 5.0,
                description: 'Gain control',
              ),
            )
            .withParameter(
              Parameter.stageFloat(
                'threshold',
                defaultValue: 0.5,
                min: 0.0,
                max: 1.0,
                description: 'Threshold control',
              ),
            )
            .op('normalize')
            .build();

        // Test single parameter update
        final result1 = await stage.setPropertyValue('gain', 2.0);
        expect(result1, isTrue);
        expect(stage.getPropertyValue<double>('gain'), equals(2.0));

        // Test multiple parameter updates
        final result2 = await stage.setPropertyValue('gain', 3.0);
        await stage.setPropertyValue('threshold', 0.8);
        expect(result2, isTrue);
        expect(stage.getPropertyValue<double>('gain'), equals(3.0));
        expect(stage.getPropertyValue<double>('threshold'), equals(0.8));

        // Test validation
        expect(stage.isValidParameterValue('gain', 2.5), isTrue);
        expect(
          stage.isValidParameterValue('gain', 10.0),
          isFalse,
        ); // Out of range
        expect(
          stage.isValidParameterValue('gain', 'invalid'),
          isFalse,
        ); // Wrong type
      });
    });

    group('AudioStream Integration', () {
      test('AudioStream properties and methods', () {
        final audioStream = MockMediaStream(
          'audio_test',
          MediaStreamType.audio,
        );

        expect(audioStream.id, equals('audio_test'));
        expect(audioStream.type, equals(MediaStreamType.audio));
        expect(audioStream.deviceId, isNotEmpty);
        expect(audioStream.name, isNotEmpty);
        expect(audioStream.color, isNotNull);
      });

      test('AudioStream tensor notification', () async {
        final audioStream = MockMediaStream(
          'audio_notify',
          MediaStreamType.audio,
        );
        final tensor = await Tensor.create([1024]);

        var notificationReceived = false;
        audioStream.onTensorUpdate((t) {
          notificationReceived = true;
          expect(t, equals(tensor));
        });

        await audioStream.start();
        audioStream.notifyTensorUpdate(tensor);

        expect(notificationReceived, isTrue);
        await audioStream.stop();
      });
    });

    group('Tensor Operations', () {
      test('Tensor data access from quick guide', () async {
        final tensor = await Tensor.create([10]);
        final testData = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        await tensor.write(testData);

        // Get tensor data as shown in guide
        final pixels = await tensor.getData() as Float32List;
        expect(pixels.length, equals(10));
        expect(pixels[0], equals(1.0));
        expect(pixels[9], equals(10.0));

        // Check validity
        expect(tensor.buffer.isValid, isTrue);

        // Check shape
        expect(tensor.shape, equals([10]));

        tensor.destroy();
      });
    });

    group('Queue Strategies', () {
      test('All queue strategies from quick guide', () {
        final latestOnly = QueueConfig(strategy: QueueStrategy.latestOnly);
        expect(latestOnly.strategy, equals(QueueStrategy.latestOnly));

        final latestN = QueueConfig(
          strategy: QueueStrategy.latestN,
          batchSize: 5,
        );
        expect(latestN.strategy, equals(QueueStrategy.latestN));
        expect(latestN.batchSize, equals(5));

        final unlimited = QueueConfig(strategy: QueueStrategy.processAll);
        expect(unlimited.strategy, equals(QueueStrategy.processAll));

        final dropOnOverrun = QueueConfig(
          strategy: QueueStrategy.dropOnOverrun,
        );
        expect(dropOnOverrun.strategy, equals(QueueStrategy.dropOnOverrun));
      });
    });

    group('Common Pipeline Patterns', () {});

    group('Resource Management', () {
      test('Pipeline disposal pattern from quick guide', () async {
        final pipeline = Pipeline(id: 'disposal_test', debugEnabled: true);
        final mockStage = MockPipelineStage(stageId: 'test_stage');
        final mockStream = MockMediaStream(
          'test_stream',
          MediaStreamType.audio,
        );

        pipeline.addStage(mockStage);
        pipeline.addStream(mockStream);

        // Event subscription as shown in guide
        late StreamSubscription subscription;
        subscription = pipeline.eventNotifier.eventStream.listen((event) {
          // Handle events
        });

        await pipeline.start();
        expect(pipeline.isRunning, isTrue);

        // Follow the disposal pattern from the guide
        await subscription.cancel();
        await pipeline.stop();
        pipeline.dispose();

        expect(pipeline.isRunning, isFalse);
      });

      test('Image conversion pattern from quick guide', () async {
        // Create mock RGBA data
        final width = 64;
        final height = 64;
        final rgbaBytes = Uint8List(width * height * 4);

        // Fill with some test pattern
        for (int i = 0; i < rgbaBytes.length; i += 4) {
          rgbaBytes[i] = 255; // R
          rgbaBytes[i + 1] = 128; // G
          rgbaBytes[i + 2] = 64; // B
          rgbaBytes[i + 3] = 255; // A
        }

        // Follow the image conversion pattern from the guide
        final buffer = await ui.ImmutableBuffer.fromUint8List(rgbaBytes);
        final descriptor = ui.ImageDescriptor.raw(
          buffer,
          height: height,
          width: width,
          pixelFormat: ui.PixelFormat.rgba8888,
        );
        final codec = await descriptor.instantiateCodec();
        final frameInfo = await codec.getNextFrame();
        final image = frameInfo.image;

        expect(image.width, equals(width));
        expect(image.height, equals(height));

        // Clean up as shown in guide
        buffer.dispose();
        codec.dispose();
        image.dispose();
      });
    });

    group('Debugging', () {
      test('Debug mode and methods from quick guide', () async {
        final pipeline = Pipeline(id: 'debug_test', debugEnabled: true);
        final mockStage = MockPipelineStage(stageId: 'debug_stage');
        final mockStream = MockMediaStream(
          'debug_stream',
          MediaStreamType.audio,
        );

        pipeline.addStage(mockStage);
        pipeline.addStream(mockStream);

        await pipeline.start();

        // Test debugPrintAllOutputs method
        await pipeline.debugPrintAllOutputs();

        // Test getStageOutput method
        final outputs = pipeline.getStageOutput('debug_stage', 'debug_stream');
        expect(outputs, anyOf(isNull, isMap));

        // Test getBenchmarkStats method
        final stats = pipeline.getBenchmarkStats();
        expect(stats, isMap);
        expect(stats.containsKey('event_counts'), isTrue);

        await pipeline.stop();
        pipeline.dispose();
      });
    });

    group('Complete End-to-End Scenarios', () {
      test('Parameter runtime updates in complex pipeline', () async {
        final pipeline = Pipeline(id: 'param_runtime_test');

        // Create stage with parameters
        final dynamicStage = StageBuilder('dynamic_processor')
            .withParameter(
              Parameter.stageFloat(
                'gain',
                defaultValue: 1.0,
                min: 0.0,
                max: 10.0,
                description: 'Processing gain',
              ),
            )
            .withParameter(
              Parameter.stageBool(
                'enabled',
                defaultValue: true,
                description: 'Enable processing',
              ),
            )
            .withParameter(
              Parameter.stageEnum<ProcessingMode>(
                'mode',
                defaultValue: ProcessingMode.fast,
                allValues:
                    ProcessingMode.values, // This provides all enum values
                description: 'Processing mode',
              ),
            )
            .op('multiplyScalar', arguments: ['param:gain'])
            .build();

        pipeline.addStage(dynamicStage);

        final stream = MockMediaStream('param_stream', MediaStreamType.audio);
        pipeline.addStream(stream);

        // Monitor parameter changes
        final paramEvents = <ParameterChangeEvent>[];
        final subscription = pipeline.eventNotifier.eventStream.listen((event) {
          if (event is ParameterChangeEvent) {
            paramEvents.add(event);
          }
        });

        await pipeline.start();

        // Test parameter updates during runtime
        await dynamicStage.setPropertyValue('gain', 2.5);
        expect(dynamicStage.getPropertyValue<double>('gain'), equals(2.5));

        await dynamicStage.setPropertyValue('enabled', false);
        expect(dynamicStage.getPropertyValue<bool>('enabled'), equals(false));

        expect(
          dynamicStage.getPropertyValue<ProcessingMode>('mode'),
          equals(ProcessingMode.fast),
        );
        expect(
          dynamicStage.getPropertyValue<ProcessingMode>('mode'),
          equals(ProcessingMode.fast),
        );

        // Test batch parameter update
        await dynamicStage.setPropertyValue('gain', 5.0);
        await dynamicStage.setPropertyValue('enabled', true);
        expect(dynamicStage.getPropertyValue<double>('gain'), equals(5.0));
        expect(dynamicStage.getPropertyValue<bool>('enabled'), equals(true));

        // Verify parameter change events were emitted
        await Future.delayed(Duration(milliseconds: 50));
        expect(paramEvents.length, greaterThan(0));

        await subscription.cancel();
        await pipeline.stop();
        pipeline.dispose();
      });
    });
  });
}
