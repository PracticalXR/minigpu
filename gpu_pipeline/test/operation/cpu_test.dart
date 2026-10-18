import 'dart:typed_data';
import 'dart:math' as math;
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:gpu_pipeline/src/operation/cpu_stage_operation.dart';
import 'package:gpu_pipeline/src/stages/pipeline_resource_cache.dart';
import 'package:gpu_pipeline/src/pipeline_stage_builder.dart';
import 'package:gpu_pipeline/src/pipeline.dart';

void main() {
  group('CPU Operations', () {
    late ResourceCache cache;

    setUp(() {
      cache = ResourceCache(cacheId: 'default');
      CPUOperationRegistry.registerBuiltInProcessors();
    });

    group('CPUStageOperation', () {
      test(
        'should execute custom CPU processor with raw data and ranks',
        () async {
          final inputTensor = await Tensor.zeros([2, 2]);
          final inputData = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
          await inputTensor.write(inputData);

          // Create a stage with the CPU operation instead of calling it directly
          final builder = StageBuilder('cpu_test');
          builder.withFlexibleInput('input');
          builder.withDynamicOutput('output');
          builder.executeCPU((inputs, ranks, params) async {
            final data = inputs['input']! as Float32List;
            final shape = ranks['input']!;

            expect(shape, equals([2, 2]));
            expect(data.length, equals(4));

            // Double all values
            final outputData = Float32List(data.length);
            for (int i = 0; i < data.length; i++) {
              outputData[i] = data[i] * 2.0;
            }

            return {'output': outputData};
          });

          final stage = builder.build();

          // Execute through stage (uses DynamicStage's smart shape preservation)
          final result = await stage.process({'input': inputTensor});

          // Verify results
          expect(result.containsKey('output'), isTrue);
          final outputData = await result['output']!.getData() as Float32List;
          expect(outputData, equals([2.0, 4.0, 6.0, 8.0]));
          expect(result['output']!.shape, equals([2, 2])); // ✅ This will work
        },
      );

      test('should handle multiple inputs with different shapes', () async {
        // Create first input [3]
        final input1 = await Tensor.zeros([3]);
        final input1Data = Float32List.fromList([1.0, 2.0, 3.0]);
        await input1.write(input1Data);

        // Create second input [2, 3]
        final input2 = await Tensor.zeros([2, 3]);
        final input2Data = Float32List.fromList([4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        await input2.write(input2Data);

        final cpuOp = CPUStageOperation(
          operationId: 'test_multi_input',
          processor: (inputs, ranks, params) async {
            final data1 = inputs['input1']! as Float32List;
            final data2 = inputs['input2']! as Float32List;
            final shape1 = ranks['input1']!;
            final shape2 = ranks['input2']!;

            expect(shape1, equals([3]));
            expect(shape2, equals([2, 3]));
            expect(data1.length, equals(3));
            expect(data2.length, equals(6));

            // Create output by concatenating inputs
            final outputData = Float32List(9);
            outputData.setRange(0, 3, data1);
            outputData.setRange(3, 9, data2);

            return {'combined': outputData};
          },
          inputKeys: ['input1', 'input2'],
          outputKeys: ['combined'],
        );

        final result = await cpuOp.executeWithCache(
          {'input1': input1, 'input2': input2},
          {},
          cache,
        );

        final outputData = await result['combined']!.getData() as Float32List;
        expect(
          outputData,
          equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        );
      });

      test('should pass through parameters to processor', () async {
        final inputTensor = await Tensor.zeros([4]);
        final inputData = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
        await inputTensor.write(inputData);

        final cpuOp = CPUStageOperation(
          operationId: 'test_params',
          processor: (inputs, ranks, params) async {
            final data = inputs['input']! as Float32List;
            final multiplier = params['multiplier'] as double;
            final offset = params['offset'] as double;

            final outputData = Float32List(data.length);
            for (int i = 0; i < data.length; i++) {
              outputData[i] = data[i] * multiplier + offset;
            }

            return {'output': outputData};
          },
          inputKeys: ['input'],
          outputKeys: ['output'],
        );

        final result = await cpuOp.executeWithCache(
          {'input': inputTensor},
          {'multiplier': 3.0, 'offset': 10.0},
          cache,
        );

        final outputData = await result['output']!.getData() as Float32List;
        expect(outputData, equals([13.0, 16.0, 19.0, 22.0])); // (x * 3) + 10
      });

      test('should track cache statistics', () async {
        final inputTensor = await Tensor.zeros([2]);
        final inputData = Float32List.fromList([1.0, 2.0]);
        await inputTensor.write(inputData);

        final cpuOp = CPUStageOperation(
          operationId: 'test_cache_stats',
          processor: (inputs, ranks, params) async {
            final data = inputs['input']! as Float32List;
            return {'output': Float32List.fromList(data)};
          },
          inputKeys: ['input'],
          outputKeys: ['output'],
        );

        // First execution should be a cache miss
        expect(cpuOp.cacheHits, equals(0));
        expect(cpuOp.cacheMisses, equals(0));

        await cpuOp.executeWithCache({'input': inputTensor}, {}, cache);

        expect(cpuOp.cacheMisses, equals(1));
        expect(cpuOp.cacheHits, equals(0));
        expect(cpuOp.cacheHitRate, equals(0.0));

        // Second execution with same shape should be a cache hit
        await cpuOp.executeWithCache({'input': inputTensor}, {}, cache);

        expect(cpuOp.cacheMisses, equals(1));
        expect(cpuOp.cacheHits, equals(1));
        expect(cpuOp.cacheHitRate, equals(50.0));
      });
    });

    group('PrebuiltCPUOperation', () {
      test('should execute passthrough operation', () async {
        final inputTensor = await Tensor.zeros([3]);
        final inputData = Float32List.fromList([1.0, 2.0, 3.0]);
        await inputTensor.write(inputData);

        final passthroughOp = PrebuiltCPUOperation(
          operationId: 'test_passthrough',
          operation: 'passthrough',
          inputKeys: ['input'],
          outputKeys: ['output'],
        );

        final result = await passthroughOp.executeWithCache(
          {'input': inputTensor},
          {},
          cache,
        );

        final outputData = await result['output']?.getData() as Float32List;
        expect(outputData, equals([1.0, 2.0, 3.0]));
      });

      test('should execute debug_info operation', () async {
        final inputTensor = await Tensor.zeros([2, 2]);
        final inputData = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
        await inputTensor.write(inputData);

        final debugOp = PrebuiltCPUOperation(
          operationId: 'test_debug',
          operation: 'debug_info',
          inputKeys: ['input'],
          outputKeys: ['output'],
        );

        // Should not throw and should return the same data
        final result = await debugOp.executeWithCache(
          {'input': inputTensor},
          {'test_param': 'value'},
          cache,
        );

        final outputData = await result['output']?.getData() as Float32List;
        expect(outputData, equals([1.0, 2.0, 3.0, 4.0]));
      });

      test('should throw on unknown operation', () async {
        final inputTensor = await Tensor.zeros([2]);
        final inputData = Float32List.fromList([1.0, 2.0]);
        await inputTensor.write(inputData);

        final unknownOp = PrebuiltCPUOperation(
          operationId: 'test_unknown',
          operation: 'unknown_operation',
          inputKeys: ['input'],
          outputKeys: ['output'],
        );

        expect(
          () => unknownOp.executeWithCache({'input': inputTensor}, {}, cache),
          throwsA(isA<Exception>()),
        );
      });
    });

    group('StageBuilder Integration', () {
      test('should create CPU stage with executeCPU', () async {
        final builder = StageBuilder('cpu_test');
        builder.withFlexibleInput('input');
        builder.withDynamicOutput('output');
        builder.executeCPU((inputs, ranks, params) async {
          final data = inputs['input']! as Float32List;

          // Square all values
          final outputData = Float32List(data.length);
          for (int i = 0; i < data.length; i++) {
            outputData[i] = data[i] * data[i];
          }

          return {'output': outputData};
        });
        final stage = builder.build();

        expect(stage.operations.length, equals(1));
        expect(stage.operations.first, isA<CPUStageOperation>());

        // Test execution through stage.process()
        final inputTensor = await Tensor.zeros([3]);
        final inputData = Float32List.fromList([2.0, 3.0, 4.0]);
        await inputTensor.write(inputData);

        final result = await stage.process({'input': inputTensor});

        final outputData = await result['output']!.getData() as Float32List;
        expect(outputData, equals([4.0, 9.0, 16.0]));
      });

      test('should create CPU stage with cpuOp', () async {
        final builder = StageBuilder('cpu_passthrough_test');
        builder.withFlexibleInput('input');
        builder.withDynamicOutput('output');
        builder.cpuOp('passthrough');
        final stage = builder.build();

        expect(stage.operations.length, equals(1));
        expect(stage.operations.first, isA<PrebuiltCPUOperation>());

        // Test execution through stage.process()
        final inputTensor = await Tensor.zeros([3]);
        final inputData = Float32List.fromList([5.0, 6.0, 7.0]);
        await inputTensor.write(inputData);

        final result = await stage.process({'input': inputTensor});

        final outputData = await result['output']!.getData() as Float32List;
        expect(outputData, equals([5.0, 6.0, 7.0]));
      });

      test('should handle chained operations with CPU', () async {
        final builder = StageBuilder('mixed_operations');
        builder.withFlexibleInput('input');
        builder.withDynamicOutput('final_output');

        builder.op(
          'multiplyScalar',
          arguments: [2.0],
        ); // ✅ Use scalar operation
        builder.executeCPU((inputs, ranks, params) async {
          final data = inputs.values.first as Float32List;
          final outputData = Float32List(data.length);
          for (int i = 0; i < data.length; i++) {
            outputData[i] = data[i] + 1.0;
          }
          return {'output': outputData};
        });
        builder.op('addScalar', arguments: [5.0]); // ✅ Use scalar operation

        final stage = builder.build();

        // Print the operations to see what keys they expect
        for (int i = 0; i < stage.operations.length; i++) {
          final op = stage.operations[i];
          print(
            'Operation $i: ${op.operationId}, inputKeys: ${op.inputKeys}, outputKeys: ${op.outputKeys}',
          );
        }

        final inputTensor = await Tensor.zeros([3]);
        await inputTensor.write(Float32List.fromList([1.0, 2.0, 3.0]));

        final result = await stage.process({'input': inputTensor});
        final output = result['final_output']!;
        final outputData = await output.getData() as Float32List;

        expect(outputData, equals([8.0, 10.0, 12.0]));
      });
    });

    group('Direct Stage Processing', () {
      test(
        'should process data through multiple stages with CPU operations',
        () async {
          // Create individual stages
          final preprocessBuilder = StageBuilder('gpu_preprocess');
          preprocessBuilder.withFlexibleInput('input');
          preprocessBuilder.withDynamicOutput('preprocessed');
          preprocessBuilder.op('multiplyScalar', arguments: [2.0]);
          final preprocessStage = preprocessBuilder.build();

          final analysisBuilder = StageBuilder('cpu_analysis');
          analysisBuilder.withFlexibleInput('input');
          analysisBuilder.withDynamicOutput('analyzed');
          analysisBuilder.executeCPU((inputs, ranks, params) async {
            final data = inputs['input']! as Float32List;
            final shape = ranks['input']!;

            // Calculate statistics on CPU
            double sum = 0.0;
            double max = data[0];

            for (final value in data) {
              sum += value;
              if (value > max) max = value;
            }

            final mean = sum / data.length;

            // Create output with statistics
            final stats = Float32List.fromList([mean, max, sum]);

            return {
              'analyzed': stats,
              'analyzed_shape': Int32List.fromList([3]),
            };
          });
          final analysisStage = analysisBuilder.build();

          final postprocessBuilder = StageBuilder('gpu_postprocess');
          postprocessBuilder.withFlexibleInput('input');
          postprocessBuilder.withDynamicOutput('final');
          postprocessBuilder.op('addScalar', arguments: [10.0]);
          final postprocessStage = postprocessBuilder.build();

          // Process data through stages manually
          final inputTensor = await Tensor.zeros([4]);
          final inputData = Float32List.fromList([1.0, 2.0, 3.0, 4.0]);
          await inputTensor.write(inputData);

          // Stage 1: GPU preprocessing
          final preprocessResult = await preprocessStage.process({
            'input': inputTensor,
          });
          expect(preprocessResult.containsKey('preprocessed'), isTrue);

          // Stage 2: CPU analysis
          final analysisResult = await analysisStage.process({
            'input': preprocessResult['preprocessed']!,
          });
          expect(analysisResult.containsKey('analyzed'), isTrue);

          // Verify CPU analysis results
          final analysisData =
              await analysisResult['analyzed']!.getData() as Float32List;
          // [1,2,3,4] * 2 = [2,4,6,8], mean=5, max=8, sum=20
          expect(analysisData, equals([5.0, 8.0, 20.0]));

          // Stage 3: GPU postprocessing
          final finalResult = await postprocessStage.process({
            'input': analysisResult['analyzed']!,
          });
          expect(finalResult.containsKey('final'), isTrue);

          final finalData =
              await finalResult['final']!.getData() as Float32List;
          // [5,8,20] + 10 = [15,18,30]
          expect(finalData, equals([15.0, 18.0, 30.0]));
        },
      );
    });

    group('Error Handling', () {
      test('should handle processor exceptions gracefully', () async {
        final cpuOp = CPUStageOperation(
          operationId: 'test_error',
          processor: (inputs, ranks, params) async {
            throw Exception('Test CPU processor error');
          },
          inputKeys: ['input'],
          outputKeys: ['output'],
        );

        final inputTensor = await Tensor.zeros([1]);
        final inputData = Float32List.fromList([1.0]);
        await inputTensor.write(inputData);

        expect(
          () => cpuOp.executeWithCache({'input': inputTensor}, {}, cache),
          throwsA(isA<Exception>()),
        );
      });

      test('should validate input keys', () async {
        final cpuOp = CPUStageOperation(
          operationId: 'test_validation',
          processor: (inputs, ranks, params) async {
            return {'output': inputs.values.first};
          },
          inputKeys: ['required_input'],
          outputKeys: ['output'],
        );

        final inputTensor = await Tensor.zeros([1]);
        final inputData = Float32List.fromList([1.0]);
        await inputTensor.write(inputData);

        // Should validate that required input is present
        expect(
          () => cpuOp.executeWithCache({'wrong_key': inputTensor}, {}, cache),
          throwsA(isA<Exception>()),
        );
      });
    });
  });

  // The DynamicCPUOperation (dart_eval) test group was removed together
  // with the feature: its analyzer/dart:io dependency chain could not
  // compile for the web.  Per-element transforms are covered by
  // ExpressionCPUOperation (see ../expression_test.dart); the dynamic_cpu
  // opKind's fail-gracefully contract is pinned in
  // ../serialization/dynamic_cpu_smoke_test.dart.
}
