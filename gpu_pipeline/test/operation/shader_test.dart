import 'dart:typed_data';
import 'package:gpu_pipeline/src/stages/pipeline_resource_cache.dart';
import 'package:test/test.dart';
import 'package:gpu_pipeline/src/operation/shader_stage_operation.dart';
import 'package:gpu_pipeline/src/pipeline_ports.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  group('ShaderStageOperation', () {
    late ShaderStageOperation operation;

    setUpAll(() async {
      if (!DefaultMinigpu.instance.isInitialized) {
        await DefaultMinigpu.instance.init();
      }
      // Destroy all GPU shaders accumulated by previous test files.
      // Dawn/D3D12 silently fails dispatch when too many pipelines are active;
      // cleaning up between test groups prevents this.
      DefaultMinigpu.instance.destroyAllTrackedShaders();
    });

    tearDown(() {
      // Clean up any created operations
    });

    test('should create operation with basic parameters', () {
      final outputPorts = {
        'output': FixedOutputPort(name: 'output', shape: [1]),
      };
      operation = ShaderStageOperation(
        operationId: 'test_shader',
        shaderTemplate: '''
          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
          }
        ''',
        inputKeys: ['input'],
        outputKeys: ['output'],
        outputPorts: outputPorts,
      );

      expect(operation.operationId, equals('test_shader'));
      expect(operation.inputKeys, equals(['input']));
      expect(operation.outputKeys, equals(['output']));
    });

    test('should execute simple element-wise operation', () async {
      final outputPorts = {
        'output': FixedOutputPort(name: 'output', shape: [4]),
      };
      operation = ShaderStageOperation(
        operationId: 'add_one',
        shaderTemplate: '''
          @group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
          @group(0) @binding(1) var<storage, read_write> output_0: array<f32>;
          
          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
            if (i >= arrayLength(&input_0)) { return; }
            output_0[i] = input_0[i] + 1.0;
          }
        ''',
        inputKeys: ['input'],
        outputKeys: ['output'],
        outputPorts: outputPorts,
      );

      final inputTensor = await Tensor.create([4]);
      await inputTensor.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));

      final inputs = {'input': inputTensor};

      final outputs = await operation.execute(
        inputs,
        {},
        ResourceCache.forOperation(operation.operationId),
      );

      expect(outputs.containsKey('output'), isTrue);

      final resultData = await outputs['output']!.getData();
      final result = resultData as Float32List;

      expect(result[0], equals(2.0));
      expect(result[1], equals(3.0));
      expect(result[2], equals(4.0));
      expect(result[3], equals(5.0));

      inputTensor.destroy();
      outputs['output']!.destroy();
    });

    test('should handle shader parameters substitution', () async {
      final outputPorts = {
        'output': FixedOutputPort(name: 'output', shape: [3]),
      };
      operation = ShaderStageOperation(
        operationId: 'multiply_by_factor',
        shaderTemplate: '''
          @group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
          @group(0) @binding(1) var<storage, read_write> output_0: array<f32>;
          
          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
            if (i >= arrayLength(&input_0)) { return; }
            output_0[i] = input_0[i] * \${factor};
          }
        ''',
        inputKeys: ['input'],
        outputKeys: ['output'],
        shaderParameters: {'factor': 2.0},
        outputPorts: outputPorts,
      );

      final inputTensor = await Tensor.create([3]);
      await inputTensor.write(Float32List.fromList([1.0, 2.0, 3.0]));

      final inputs = {'input': inputTensor};

      final outputs = await operation.execute(
        inputs,
        {},
        ResourceCache.forOperation(operation.operationId),
      );

      final resultData = await outputs['output']!.getData();
      final result = resultData as Float32List;

      expect(result[0], equals(2.0));
      expect(result[1], equals(4.0));
      expect(result[2], equals(6.0));

      inputTensor.destroy();
      outputs['output']!.destroy();
    });

    test('should handle runtime parameters override', () async {
      final outputPorts = {
        'output': FixedOutputPort(name: 'output', shape: [3]),
      };
      operation = ShaderStageOperation(
        operationId: 'multiply_runtime',
        shaderTemplate: '''
          @group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
          @group(0) @binding(1) var<storage, read_write> output_0: array<f32>;
          
          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
            if (i >= arrayLength(&input_0)) { return; }
            output_0[i] = input_0[i] * \${multiplier};
          }
        ''',
        inputKeys: ['input'],
        outputKeys: ['output'],
        shaderParameters: {'multiplier': 2.0},
        outputPorts: outputPorts,
      );

      final inputTensor = await Tensor.create([3]);
      await inputTensor.write(Float32List.fromList([1.0, 2.0, 3.0]));

      final inputs = {'input': inputTensor};

      final outputs = await operation.execute(inputs, {
        'multiplier': 3.0,
      }, ResourceCache.forOperation(operation.operationId));

      final resultData = await outputs['output']!.getData();
      final result = resultData as Float32List;

      expect(result[0], equals(3.0));
      expect(result[1], equals(6.0));
      expect(result[2], equals(9.0));

      inputTensor.destroy();
      outputs['output']!.destroy();
    });

    test('should handle multiple inputs and outputs', () async {
      final outputPorts = {
        'sum': FixedOutputPort(name: 'sum', shape: [3]),
        'diff': FixedOutputPort(name: 'diff', shape: [3]),
      };
      operation = ShaderStageOperation(
        operationId: 'add_tensors',
        shaderTemplate: '''
          @group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
          @group(0) @binding(1) var<storage, read_write> input_1: array<f32>;
          @group(0) @binding(2) var<storage, read_write> output_0: array<f32>;
          @group(0) @binding(3) var<storage, read_write> output_1: array<f32>;
          
          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
            if (i >= arrayLength(&input_0)) { return; }
            output_0[i] = input_0[i] + input_1[i];
            output_1[i] = input_0[i] - input_1[i];
          }
        ''',
        inputKeys: ['a', 'b'],
        outputKeys: ['sum', 'diff'],
        outputPorts: outputPorts,
      );

      final inputA = await Tensor.create([3]);
      await inputA.write(Float32List.fromList([5.0, 6.0, 7.0]));

      final inputB = await Tensor.create([3]);
      await inputB.write(Float32List.fromList([1.0, 2.0, 3.0]));

      final inputs = {'a': inputA, 'b': inputB};

      final outputs = await operation.execute(
        inputs,
        {},
        ResourceCache.forOperation(operation.operationId),
      );

      print(operation.shaderTemplate);

      print(outputs);
      expect(outputs.containsKey('sum'), isTrue);
      expect(outputs.containsKey('diff'), isTrue);

      final sumData = await outputs['sum']!.getData();
      final sum = sumData as Float32List;

      final diffData = await outputs['diff']!.getData();
      final diff = diffData as Float32List;

      expect(sum[0], equals(6.0));
      expect(sum[1], equals(8.0));
      expect(sum[2], equals(10.0));

      expect(diff[0], equals(4.0));
      expect(diff[1], equals(4.0));
      expect(diff[2], equals(4.0));

      inputA.destroy();
      inputB.destroy();
      outputs['sum']!.destroy();
      outputs['diff']!.destroy();
    });

    test('should handle custom workgroup size', () async {
      final outputPorts = {
        'output': FixedOutputPort(name: 'output', shape: [3]),
      };
      operation = ShaderStageOperation(
        operationId: 'custom_workgroup',
        shaderTemplate: '''
          @group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
          @group(0) @binding(1) var<storage, read_write> output_0: array<f32>;
          
          @compute @workgroup_size(64)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
            if (i >= arrayLength(&input_0)) { return; }
            output_0[i] = input_0[i] * 2.0;
          }
        ''',
        inputKeys: ['input'],
        outputKeys: ['output'],
        workgroupSize: [64, 1, 1],
        outputPorts: outputPorts,
      );

      final inputTensor = await Tensor.create([3]);
      await inputTensor.write(Float32List.fromList([1.0, 2.0, 3.0]));

      final inputs = {'input': inputTensor};

      final outputs = await operation.execute(
        inputs,
        {},
        ResourceCache.forOperation(operation.operationId),
      );

      final resultData = await outputs['output']!.getData();
      final result = resultData as Float32List;

      expect(result[0], equals(2.0));
      expect(result[1], equals(4.0));
      expect(result[2], equals(6.0));

      inputTensor.destroy();
      outputs['output']!.destroy();
    });

    test('should serialize and deserialize configuration', () {
      final outputPorts = {
        'output1': FixedOutputPort(name: 'output1', shape: [1]),
      };
      operation = ShaderStageOperation(
        operationId: 'serialize_test',
        shaderTemplate: 'test_shader',
        inputKeys: ['input1', 'input2'],
        outputKeys: ['output1'],
        shaderParameters: {'param1': 1.0, 'param2': true},
        workgroupSize: [128, 1, 1],
        outputPorts: outputPorts,
      );

      final config = operation.toConfig();
      final restored = ShaderStageOperation.fromConfig(config);

      expect(restored.operationId, equals(operation.operationId));
      expect(restored.shaderTemplate, equals(operation.shaderTemplate));
      expect(restored.inputKeys, equals(operation.inputKeys));
      expect(restored.outputKeys, equals(operation.outputKeys));
      expect(restored.shaderParameters, equals(operation.shaderParameters));
      expect(restored.workgroupSize, equals(operation.workgroupSize));
    });

    test('should handle tensor shape validation', () async {
      final outputPorts = {
        'output': FixedOutputPort(name: 'output', shape: [2, 3]),
      };
      operation = ShaderStageOperation(
        operationId: 'shape_test',
        shaderTemplate: '''
          @group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
          @group(0) @binding(1) var<storage, read_write> output_0: array<f32>;
          
          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
            if (i >= arrayLength(&input_0)) { return; }
            output_0[i] = input_0[i];
          }
        ''',
        inputKeys: ['input'],
        outputKeys: ['output'],
        outputPorts: outputPorts,
      );

      final input2D = await Tensor.create([2, 3]);
      await input2D.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

      final inputs = {'input': input2D};

      final outputs = await operation.execute(
        inputs,
        {},
        ResourceCache.forOperation(operation.operationId),
      );

      expect(outputs['output']!.shape, equals([2, 3]));

      input2D.destroy();
      outputs['output']!.destroy();
    });

    test('should handle empty tensor', () async {
      final outputPorts = {
        'output': FixedOutputPort(name: 'output', shape: [0]),
      };
      operation = ShaderStageOperation(
        operationId: 'empty_test',
        shaderTemplate: '''
          @group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
          @group(0) @binding(1) var<storage, read_write> output_0: array<f32>;
          
          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
            if (i >= arrayLength(&input_0)) { return; }
            output_0[i] = input_0[i];
          }
        ''',
        inputKeys: ['input'],
        outputKeys: ['output'],
        outputPorts: outputPorts,
      );

      final emptyTensor = await Tensor.create([0]);
      final inputs = {'input': emptyTensor};

      final outputs = await operation.execute(
        inputs,
        {},
        ResourceCache.forOperation(operation.operationId),
      );

      expect(outputs['output']!.shape, equals([0]));

      emptyTensor.destroy();
      outputs['output']!.destroy();
    });
  });

  group('ShaderFunctionOperation', () {
    late ShaderFunctionOperation operation;

    setUpAll(() async {
      if (!DefaultMinigpu.instance.isInitialized) {
        await DefaultMinigpu.instance.init();
      }
    });

    test('should create simple function operation', () {
      operation = ShaderFunctionOperation(
        operationId: 'square_function',
        functionBody: 'return input_val * input_val;',
        functionName: 'square',
        inputKeys: ['input'],
        outputKeys: ['output'],
      );

      expect(operation.operationId, equals('square_function'));
      expect(operation.functionName, equals('square'));
    });

    test('should execute simple function', () async {
      operation = ShaderFunctionOperation(
        operationId: 'square_test',
        functionBody: 'return input_val * input_val;',
        functionName: 'square',
        inputKeys: ['input'],
        outputKeys: ['output'],
      );

      final inputTensor = await Tensor.create([3]);
      await inputTensor.write(Float32List.fromList([2.0, 3.0, 4.0]));

      final inputs = {'input': inputTensor};

      final outputs = await operation.execute(
        inputs,
        {},
        ResourceCache.forOperation(operation.operationId),
      );

      final resultData = await outputs['output']!.getData();
      final result = resultData as Float32List;

      expect(result[0], equals(4.0));
      expect(result[1], equals(9.0));
      expect(result[2], equals(16.0));

      inputTensor.destroy();
      outputs['output']!.destroy();
    });

    test('should handle function parameters', () async {
      operation = ShaderFunctionOperation(
        operationId: 'power_function',
        functionBody: 'return pow(input_val, EXPONENT);',
        functionName: 'power',
        functionParameters: {'EXPONENT': 3.0},
        inputKeys: ['input'],
        outputKeys: ['output'],
      );

      final inputTensor = await Tensor.create([2]);
      await inputTensor.write(Float32List.fromList([2.0, 3.0]));

      final inputs = {'input': inputTensor};

      final outputs = await operation.execute(
        inputs,
        {},
        ResourceCache.forOperation(operation.operationId),
      );

      final resultData = await outputs['output']!.getData();
      final result = resultData as Float32List;

      expect(result[0], closeTo(8.0, 0.001));
      expect(result[1], closeTo(27.0, 0.001));

      inputTensor.destroy();
      outputs['output']!.destroy();
    });

    test('should handle clamping function', () async {
      operation = ShaderFunctionOperation(
        operationId: 'clamp_function',
        functionBody: 'return clamp(input_val, MIN_VAL, MAX_VAL);',
        functionName: 'clamp_values',
        functionParameters: {'MIN_VAL': 0.0, 'MAX_VAL': 1.0},
        inputKeys: ['input'],
        outputKeys: ['output'],
      );

      final inputTensor = await Tensor.create([5]);
      await inputTensor.write(Float32List.fromList([-1.0, 0.5, 1.5, 0.2, 2.0]));

      final inputs = {'input': inputTensor};

      final outputs = await operation.execute(
        inputs,
        {},
        ResourceCache.forOperation(operation.operationId),
      );

      final resultData = await outputs['output']!.getData();
      final result = resultData as Float32List;

      expect(result[0], equals(0.0));
      expect(result[1], equals(0.5));
      expect(result[2], equals(1.0));
      expect(result[3], closeTo(0.2, 0.01));
      expect(result[4], equals(1.0));

      inputTensor.destroy();
      outputs['output']!.destroy();
    });

    test('should serialize and deserialize function configuration', () {
      operation = ShaderFunctionOperation(
        operationId: 'func_serialize_test',
        functionBody: 'return input_val + OFFSET;',
        functionName: 'add_offset',
        functionParameters: {'OFFSET': 1.0},
        workgroupSize: [512, 1, 1],
        inputKeys: ['input'],
        outputKeys: ['output'],
      );

      final config = operation.toConfig();
      final restored = ShaderFunctionOperation.fromConfig(config);

      expect(restored.operationId, equals(operation.operationId));
      expect(restored.functionBody, equals(operation.functionBody));
      expect(restored.functionName, equals(operation.functionName));
      expect(restored.functionParameters, equals(operation.functionParameters));
      expect(restored.workgroupSize, equals(operation.workgroupSize));
    });

    test('should handle runtime parameter override', () async {
      operation = ShaderFunctionOperation(
        operationId: 'runtime_override',
        functionBody: 'return input_val * SCALE;',
        functionName: 'modify_scale',
        functionParameters: {'SCALE': 2.0},
        inputKeys: ['input'],
        outputKeys: ['output'],
      );

      final inputTensor = await Tensor.create([3]);
      await inputTensor.write(Float32List.fromList([1.0, 2.0, 3.0]));

      final inputs = {'input': inputTensor};

      final outputs = await operation.execute(inputs, {
        'SCALE': 5.0,
      }, ResourceCache.forOperation(operation.operationId));

      final resultData = await outputs['output']!.getData();
      final result = resultData as Float32List;

      expect(result[0], equals(5.0));
      expect(result[1], equals(10.0));
      expect(result[2], equals(15.0));

      inputTensor.destroy();
      outputs['output']!.destroy();
    });

    test('should validate input keys', () async {
      operation = ShaderFunctionOperation(
        operationId: 'validation_test',
        functionBody: 'return input_val;',
        functionName: 'identity',
        inputKeys: ['required_input'],
        outputKeys: ['output'],
      );

      final inputTensor = await Tensor.create([1]);
      await inputTensor.write(Float32List.fromList([1.0]));

      expect(
        () => operation.execute(
          {'wrong_key': inputTensor},
          {},
          ResourceCache.forOperation(operation.operationId),
        ),
        throwsA(isA<Exception>()),
      );

      inputTensor.destroy();
    });
  });

  group('Error Handling', () {
    setUpAll(() async {
      await Minigpu().init();
    });

    test('should handle invalid shader code', () async {
      final outputPorts = {
        'output': FixedOutputPort(name: 'output', shape: [1]),
      };
      final operation = ShaderStageOperation(
        operationId: 'invalid_shader',
        shaderTemplate: '''
          invalid shader code that will not compile
        ''',
        inputKeys: ['input'],
        outputKeys: ['output'],
        outputPorts: outputPorts,
      );

      final inputTensor = await Tensor.create([1]);
      await inputTensor.write(Float32List.fromList([1.0]));

      final inputs = {'input': inputTensor};

      expect(
        () => operation.execute(
          inputs,
          {},
          ResourceCache.forOperation(operation.operationId),
        ),
        throwsA(isA<Exception>()),
      );

      inputTensor.destroy();
    });

    test(
      'should handle mismatched tensor shapes in multi-input operations',
      () async {
        final outputPorts = {
          'output': FixedOutputPort(name: 'output', shape: [5]),
        };
        final operation = ShaderStageOperation(
          operationId: 'mismatched_shapes',
          shaderTemplate: '''
          @group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
          @group(0) @binding(1) var<storage, read_write> input_1: array<f32>;
          @group(0) @binding(2) var<storage, read_write> output_0: array<f32>;
          
          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
            if (i >= arrayLength(&input_0)) { return; }
            output_0[i] = input_0[i] + input_1[i];
          }
        ''',
          inputKeys: ['a', 'b'],
          outputKeys: ['output'],
          outputPorts: outputPorts,
        );

        final tensor1 = await Tensor.create([3]);
        await tensor1.write(Float32List.fromList([1.0, 2.0, 3.0]));

        final tensor2 = await Tensor.create([5]);
        await tensor2.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0]));

        final inputs = {'a': tensor1, 'b': tensor2};

        try {
          final outputs = await operation.execute(
            inputs,
            {},
            ResourceCache.forOperation(operation.operationId),
          );
          outputs['output']?.destroy();
        } catch (e) {
          expect(e, isA<Exception>());
        }

        tensor1.destroy();
        tensor2.destroy();
      },
    );
  });
}
