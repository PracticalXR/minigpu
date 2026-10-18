import 'dart:typed_data';
import 'package:gpu_pipeline/src/operation/operation.dart';
import 'package:test/test.dart';
import 'package:gpu_pipeline/src/pipeline_stage_builder.dart';
import 'package:gpu_pipeline/src/operation/shader_stage_operation.dart';
import 'package:gpu_pipeline/src/pipeline_properties.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  final gpu = Minigpu();

  group('DynamicStage Tests', () {
    setUpAll(() async {
      await gpu.init();
    });

    group('StageBuilder', () {
      test('should create basic stage with tensor operations', () {
        final stage = StageBuilder('test_stage')
            .op('multiplyScalar', arguments: [2.0])
            .op('addScalar', arguments: [1.0])
            .build();

        expect(stage.stageId, equals('test_stage'));
        expect(stage.operations.length, equals(2));
        expect(stage.operations[0], isA<TensorOperation>());
        expect(stage.operations[1], isA<TensorOperation>());
      });

      test('should handle custom input/output keys', () {
        final stage = StageBuilder(
          'custom_keys',
        ).op('abs', inputKey: 'signal', outputKey: 'rectified').build();

        expect(stage.operations.length, equals(1));
        final op = stage.operations[0] as TensorOperation;
        expect(op.inputKeys, equals(['signal']));
        expect(op.outputKeys, equals(['rectified']));
      });

      test('should create shader operations', () {
        final shaderCode = '''
          @group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
          @group(0) @binding(1) var<storage, read_write> output_0: array<f32>;
          
          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
            if (i >= arrayLength(&input_0)) { return; }
            output_0[i] = input_0[i] * 2.0;
          }
        ''';

        final stage = StageBuilder(
          'shader_stage',
        ).executeShader(shaderCode).build();

        expect(stage.operations.length, equals(1));
        expect(stage.operations[0], isA<ShaderStageOperation>());
      });

      test('should create shader function operations', () {
        final stage = StageBuilder('function_stage')
            .executeShaderFunction(
              'return input_val * input_val;',
              functionName: 'square',
            )
            .build();

        expect(stage.operations.length, equals(1));
        expect(stage.operations[0], isA<ShaderFunctionOperation>());
      });

      test('should handle parameters', () {
        final stage = StageBuilder('param_stage')
            .withParameter(
              Parameter.stageFloat(
                'gain',
                defaultValue: 1.0,
                min: 0.0,
                max: 10.0,
                description: 'Gain factor for the operation',
              ),
            )
            .op('multiplyScalar', arguments: [2.0])
            .build();

        expect(stage.parameterDefinitions.containsKey('gain'), isTrue);
        expect(stage.parameterDefinitions['gain']!.defaultValue, equals(1.0));
      });

      test('should fail when building empty stage', () {
        expect(() => StageBuilder('empty').build(), throwsA(isA<StateError>()));
      });
    });

    group('DynamicStage Execution', () {
      test('should execute simple tensor operations', () async {
        final stage = StageBuilder('simple_ops')
            .op('multiplyScalar', arguments: [2.0])
            .op('addScalar', arguments: [1.0])
            .build();

        await stage.initializeStage();

        final inputTensor = await Tensor.create([4]);
        await inputTensor.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));

        final inputs = {'input': inputTensor};
        final outputs = await stage.process(inputs);

        expect(outputs.containsKey('output'), isTrue);

        final resultData = await outputs['output']!.getData();
        final result = resultData as Float32List;

        // Expected: (input * 2.0) + 1.0
        expect(result[0], equals(3.0)); // (1.0 * 2.0) + 1.0
        expect(result[1], equals(5.0)); // (2.0 * 2.0) + 1.0
        expect(result[2], equals(7.0)); // (3.0 * 2.0) + 1.0
        expect(result[3], equals(9.0)); // (4.0 * 2.0) + 1.0

        inputTensor.destroy();
        outputs['output']!.destroy();
        await stage.dispose();
      });

      test('should execute shader operations', () async {
        final shaderCode = '''
          @group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
          @group(0) @binding(1) var<storage, read_write> output_0: array<f32>;
          
          @compute @workgroup_size(64)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
            if (i >= arrayLength(&input_0)) { return; }
            output_0[i] = sqrt(abs(input_0[i])) * 2.0;
          }
        ''';

        final stage = StageBuilder(
          'shader_ops',
        ).executeShader(shaderCode, workgroupSize: [64, 1, 1]).build();

        await stage.initializeStage();

        final inputTensor = await Tensor.create([8]);
        await inputTensor.write(
          Float32List.fromList([1.0, 4.0, 9.0, 16.0, 25.0, -4.0, -9.0, 0.0]),
        );

        final inputs = {'input': inputTensor};
        final outputs = await stage.process(inputs);
        final resultData = await outputs['output']?.getData();
        final result = resultData as Float32List;

        // Expected: sqrt(abs(input)) * 2.0
        expect(result[0], closeTo(2.0, 0.01)); // sqrt(1) * 2
        expect(result[1], closeTo(4.0, 0.01)); // sqrt(4) * 2
        expect(result[2], closeTo(6.0, 0.01)); // sqrt(9) * 2
        expect(result[3], closeTo(8.0, 0.01)); // sqrt(16) * 2
        expect(result[4], closeTo(10.0, 0.01)); // sqrt(25) * 2
        expect(result[5], closeTo(4.0, 0.01)); // sqrt(abs(-4)) * 2
        expect(result[6], closeTo(6.0, 0.01)); // sqrt(abs(-9)) * 2
        expect(result[7], closeTo(0.0, 0.01)); // sqrt(0) * 2

        inputTensor.destroy();
        outputs['output']!.destroy();
        await stage.dispose();
      });

      test('should execute shader function operations', () async {
        final stage = StageBuilder('function_ops')
            .executeShaderFunction(
              '''
              let squared = input_val * input_val;
              let scaled = squared * SCALE_FACTOR;
              return clamp(scaled, 0.0, MAX_VALUE);
              ''',
              functionName: 'square_and_clamp',
              functionParameters: {'SCALE_FACTOR': 0.5, 'MAX_VALUE': 8.0},
            )
            .build();

        await stage.initializeStage();

        final inputTensor = await Tensor.create([5]);
        await inputTensor.write(
          Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0]),
        );

        final inputs = {'input': inputTensor};
        final outputs = await stage.process(inputs);

        final resultData = await outputs['output']!.getData();
        final result = resultData as Float32List;

        // Expected: clamp(input^2 * 0.5, 0.0, 8.0)
        expect(result[0], closeTo(0.5, 0.01)); // 1^2 * 0.5 = 0.5
        expect(result[1], closeTo(2.0, 0.01)); // 2^2 * 0.5 = 2.0
        expect(result[2], closeTo(4.5, 0.01)); // 3^2 * 0.5 = 4.5
        expect(result[3], closeTo(8.0, 0.01)); // 4^2 * 0.5 = 8.0 (clamped)
        expect(
          result[4],
          closeTo(8.0, 0.01),
        ); // 5^2 * 0.5 = 12.5 -> 8.0 (clamped)

        inputTensor.destroy();
        outputs['output']!.destroy();
        await stage.dispose();
      });

      test('should handle mixed tensor and shader operations', () async {
        final stage = StageBuilder('mixed_ops')
            .op('multiplyScalar', arguments: [2.0])
            .executeShaderFunction(
              'return sqrt(abs(input_val));',
              functionName: 'sqrt_abs',
            )
            .op('addScalar', arguments: [0.5])
            .build();

        await stage.initializeStage();

        final inputTensor = await Tensor.create([4]);
        await inputTensor.write(Float32List.fromList([1.0, 4.0, -9.0, 16.0]));

        final inputs = {'input': inputTensor};
        final outputs = await stage.process(inputs);

        final resultData = await outputs['output']!.getData();
        final result = resultData as Float32List;

        // Expected: sqrt(abs(input * 2.0)) + 0.5
        expect(result[0], closeTo(1.914, 0.01)); // sqrt(abs(2.0)) + 0.5
        expect(result[1], closeTo(3.33, 0.01)); // sqrt(abs(8.0)) + 0.5
        expect(result[2], closeTo(4.74, 0.01)); // sqrt(abs(-18.0)) + 0.5
        expect(result[3], closeTo(6.16, 0.01)); // sqrt(abs(32.0)) + 0.5

        inputTensor.destroy();
        outputs['output']!.destroy();
        await stage.dispose();
      });

      test('should handle parameters properly', () async {
        final stage = StageBuilder('param_ops')
            .withParameter(
              Parameter.stageFloat(
                'gain',
                description: 'Gain factor for amplification',
                defaultValue: 1.0,
                min: 0.0,
                max: 10.0,
              ),
            )
            .executeShaderFunction(
              'return input_val * GAIN;',
              functionName: 'amplify',
              functionParameters: {'GAIN': 'param:gain'},
            )
            .build();

        await stage.initializeStage();

        // Test with default parameter value
        final inputTensor1 = await Tensor.create([3]);
        await inputTensor1.write(Float32List.fromList([1.0, 2.0, 3.0]));

        final inputs1 = {'input': inputTensor1};
        final outputs1 = await stage.process(inputs1);

        print('Outputs1: ${outputs1.keys}');

        final result1Data = await outputs1['output']!.getData();
        final result1 = result1Data as Float32List;

        // Default gain should be 1.0
        expect(result1[0], closeTo(1.0, 0.01));
        expect(result1[1], closeTo(2.0, 0.01));
        expect(result1[2], closeTo(3.0, 0.01));

        // Change parameter
        stage.setPropertyValue('gain', 2.5);

        final inputTensor2 = await Tensor.create([3]);
        await inputTensor2.write(Float32List.fromList([1.0, 2.0, 3.0]));

        final inputs2 = {'input': inputTensor2};
        final outputs2 = await stage.process(inputs2);

        final result2Data = await outputs2['output']!.getData();
        final result2 = result2Data as Float32List;

        // Gain should now be 2.5
        expect(result2[0], closeTo(2.5, 0.01));
        expect(result2[1], closeTo(5.0, 0.01));
        expect(result2[2], closeTo(7.5, 0.01));

        // Clean up
        inputTensor1.destroy();
        inputTensor2.destroy();
        outputs1['output']!.destroy();
        outputs2['output']!.destroy();
        await stage.dispose();
      });

      test('should handle merge operations', () async {
        final stage = StageBuilder('merge_ops')
            .merge(
              inputKeys: ['stream1', 'stream2', 'stream3'],
              mergeStrategy: 'sum',
            )
            .build();

        await stage.initializeStage();

        final tensor1 = await Tensor.create([4]);
        await tensor1.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));

        final tensor2 = await Tensor.create([4]);
        await tensor2.write(Float32List.fromList([0.5, 1.0, 1.5, 2.0]));

        final tensor3 = await Tensor.create([4]);
        await tensor3.write(Float32List.fromList([0.1, 0.2, 0.3, 0.4]));

        final inputs = {
          'stream1': tensor1,
          'stream2': tensor2,
          'stream3': tensor3,
        };

        final outputs = await stage.process(inputs);

        final resultData = await outputs['output']!.getData();
        final result = resultData as Float32List;

        // Expected: sum of all inputs
        expect(result[0], closeTo(1.6, 0.01)); // 1.0 + 0.5 + 0.1
        expect(result[1], closeTo(3.2, 0.01)); // 2.0 + 1.0 + 0.2
        expect(result[2], closeTo(4.8, 0.01)); // 3.0 + 1.5 + 0.3
        expect(result[3], closeTo(6.4, 0.01)); // 4.0 + 2.0 + 0.4

        tensor1.destroy();
        tensor2.destroy();
        tensor3.destroy();
        outputs['output']!.destroy();
        await stage.dispose();
      });

      test('should handle split operations', () async {
        final stage = StageBuilder('split_ops')
            .split(
              inputKey: 'input',
              outputKeys: ['output1', 'output2'],
              splitStrategy: 'duplicate',
            )
            .build();

        await stage.initializeStage();

        final inputTensor = await Tensor.create([3]);
        await inputTensor.write(Float32List.fromList([1.0, 2.0, 3.0]));

        final inputs = {'input': inputTensor};
        final outputs = await stage.process(inputs);

        // Both outputs should contain the same data
        final resultData = await outputs['output']!.getData();
        final result = resultData as Float32List;

        expect(result[0], equals(1.0));
        expect(result[1], equals(2.0));
        expect(result[2], equals(3.0));

        inputTensor.destroy();
        outputs['output']!.destroy();
        await stage.dispose();
      });
    });

    group('Error Handling', () {
      test('should handle invalid tensor operations', () async {
        final stage = StageBuilder(
          'invalid_ops',
        ).op('nonexistentOperation').build();

        await stage.initializeStage();

        final inputTensor = await Tensor.create([2]);
        await inputTensor.write(Float32List.fromList([1.0, 2.0]));

        final inputs = {'input': inputTensor};

        expect(() => stage.process(inputs), throwsA(isA<Exception>()));

        inputTensor.destroy();
        await stage.dispose();
      });

      test('should handle invalid shader code', () async {
        final invalidShader = '''
          invalid shader code that won't compile
        ''';

        final stage = StageBuilder(
          'invalid_shader',
        ).executeShader(invalidShader).build();

        await stage.initializeStage();

        final inputTensor = await Tensor.create([2]);
        await inputTensor.write(Float32List.fromList([1.0, 2.0]));

        final inputs = {'input': inputTensor};

        expect(() => stage.process(inputs), throwsA(isA<Exception>()));

        inputTensor.destroy();
        await stage.dispose();
      });

      test('should handle parameter validation', () async {
        final stage = StageBuilder('param_validation')
            .withParameter(
              Parameter.stageFloat(
                'limited_param',
                description: 'A parameter with limits',
                defaultValue: 5.0,
                min: 0.0,
                max: 10.0,
              ),
            )
            .op('multiplyScalar', arguments: [1.0])
            .build();

        // Valid parameter change
        expect(await stage.setPropertyValue('limited_param', 7.5), isTrue);
        expect(await stage.getPropertyValue('limited_param'), equals(7.5));

        // Invalid parameter changes
        expect(await stage.setPropertyValue('limited_param', -1.0), isFalse);
        expect(await stage.setPropertyValue('limited_param', 15.0), isFalse);
        expect(await stage.setPropertyValue('nonexistent_param', 1.0), isFalse);

        // Parameter should remain unchanged after invalid attempts
        expect(await stage.getPropertyValue('limited_param'), equals(7.5));
      });

      test('should handle missing input keys', () async {
        final stage = StageBuilder('missing_input')
            .op('multiplyScalar', inputKey: 'required_input', arguments: [2.0])
            .build();

        await stage.initializeStage();

        final inputTensor = await Tensor.create([2]);
        await inputTensor.write(Float32List.fromList([1.0, 2.0]));

        // Wrong input key
        final inputs = {'wrong_key': inputTensor};

        expect(() => stage.process(inputs), throwsA(isA<Exception>()));

        inputTensor.destroy();
        await stage.dispose();
      });

      test('should handle tensor shape mismatches in merge', () async {
        final stage = StageBuilder(
          'shape_mismatch',
        ).merge(inputKeys: ['input1', 'input2'], mergeStrategy: 'sum').build();

        await stage.initializeStage();

        final tensor1 = await Tensor.create([4]);
        await tensor1.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));

        final tensor2 = await Tensor.create([6]); // Different size
        await tensor2.write(
          Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );

        final inputs = {'input1': tensor1, 'input2': tensor2};

        expect(() => stage.process(inputs), throwsA(isA<Exception>()));

        tensor1.destroy();
        tensor2.destroy();
        await stage.dispose();
      });
    });

    group('Property Management', () {
      test('should provide property metadata', () {
        final stage = StageBuilder('metadata_test')
            .withParameters({
              'float_param': Parameter.stageFloat(
                'float_param',
                defaultValue: 1.0,
                min: 0.0,
                max: 10.0,
                description: 'A float parameter',
              ),
              'int_param': Parameter.stageInt(
                'int_param',
                defaultValue: 5,
                min: 1,
                max: 100,
                description: 'An integer parameter',
              ),
              'bool_param': Parameter.stageBool(
                'bool_param',
                defaultValue: true,
                description: 'A boolean parameter',
              ),
            })
            .op('multiplyScalar', arguments: [1.0])
            .build();

        final metadata = stage.propertyMetadata;

        expect(metadata.containsKey('float_param'), isTrue);
        expect(metadata.containsKey('int_param'), isTrue);
        expect(metadata.containsKey('bool_param'), isTrue);

        expect(
          metadata['float_param']!.description,
          equals('A float parameter'),
        );
        expect(metadata['float_param']!.min, equals(0.0));
        expect(metadata['float_param']!.max, equals(10.0));

        expect(
          metadata['int_param']!.description,
          equals('An integer parameter'),
        );
        expect(metadata['int_param']!.min, equals(1));
        expect(metadata['int_param']!.max, equals(100));

        expect(
          metadata['bool_param']!.description,
          equals('A boolean parameter'),
        );
      });

      test('should handle complex parameter scenarios', () async {
        final stage = StageBuilder('complex_params')
            .withParameter(
              Parameter.stageFloat(
                'frequency',
                defaultValue: 440.0,
                min: 20.0,
                max: 20000.0,
                description: 'Frequency in Hz',
              ),
            )
            .withParameter(
              Parameter.stageFloat(
                'amplitude',
                defaultValue: 0.5,
                min: 0.0,
                max: 1.0,
                description: 'Amplitude (0-1)',
              ),
            )
            .executeShaderFunction(
              '''
              let phase = input_val * FREQUENCY * 0.001;
              let sine = sin(phase) * AMPLITUDE;
              return sine;
              ''',
              functionName: 'sine_generator',
              functionParameters: {
                'FREQUENCY': 'param:frequency',
                'AMPLITUDE': 'param:amplitude',
              },
            )
            .build();

        await stage.initializeStage();

        // Test with default parameters
        stage.setPropertyValue('frequency', 1000.0);
        stage.setPropertyValue('amplitude', 0.8);

        final inputTensor = await Tensor.create([8]);
        final inputData = Float32List(8);
        for (int i = 0; i < 8; i++) {
          inputData[i] = i.toDouble();
        }
        await inputTensor.write(inputData);

        final inputs = {'input': inputTensor};
        final outputs = await stage.process(inputs);

        final resultData = await outputs['output']!.getData();
        final result = resultData as Float32List;

        // Results should be within amplitude bounds
        for (final value in result) {
          expect(value, greaterThanOrEqualTo(-0.8));
          expect(value, lessThanOrEqualTo(0.8));
        }

        inputTensor.destroy();
        outputs['output']!.destroy();
        await stage.dispose();
      });
    });
  });
}
