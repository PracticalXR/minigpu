import 'dart:typed_data';
import 'package:gpu_pipeline/src/stages/pipeline_resource_cache.dart';
import 'package:test/test.dart';
import 'package:gpu_pipeline/src/operation/tensor_stage_operation.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

Minigpu minigpu = Minigpu();

void main() {
  group('TensorOperationRegistry', () {
    setUpAll(() async {
      // Initialize GPU for testing
      await minigpu.init();
    });

    setUp(() {
      // Ensure registry is initialized for each test
      TensorOperationRegistry.initialize();
    });

    group('Registry Initialization', () {
      test('should initialize with built-in operations', () {
        TensorOperationRegistry.initialize();

        expect(TensorOperationRegistry.availableOperations.isNotEmpty, isTrue);
        expect(TensorOperationRegistry.hasOperation('addScalar'), isTrue);
        expect(TensorOperationRegistry.hasOperation('multiplyScalar'), isTrue);
        expect(TensorOperationRegistry.hasOperation('abs'), isTrue);
      });

      test('should not re-initialize if already initialized', () {
        TensorOperationRegistry.initialize();
        final firstCount = TensorOperationRegistry.availableOperations.length;

        TensorOperationRegistry.initialize();
        final secondCount = TensorOperationRegistry.availableOperations.length;

        expect(firstCount, equals(secondCount));
      });

      test('should categorize operations correctly', () {
        final operations = TensorOperationRegistry.availableOperations;

        // Check for expected scalar operations
        expect(operations.contains('addScalar'), isTrue);
        expect(operations.contains('multiplyScalar'), isTrue);
        expect(operations.contains('subtractScalar'), isTrue);
        expect(operations.contains('divideScalar'), isTrue);

        // Check for expected unary operations
        expect(operations.contains('abs'), isTrue);
        expect(operations.contains('sqrt'), isTrue);
        expect(operations.contains('sin'), isTrue);
        expect(operations.contains('cos'), isTrue);
        expect(operations.contains('tanh'), isTrue);

        // Check for expected tensor operations
        expect(operations.contains('add'), isTrue);
        expect(operations.contains('multiply'), isTrue);
        expect(operations.contains('subtract'), isTrue);
        expect(operations.contains('divide'), isTrue);
      });
    });

    group('Scalar Operations', () {
      late Tensor testTensor;

      setUp(() async {
        testTensor = await Tensor.create([4]);
        await testTensor.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));
      });

      tearDown(() {
        testTensor.destroy();
      });

      test('addScalar should add scalar to all elements', () async {
        final result = await TensorOperationRegistry.executeOperation(
          'addScalar',
          testTensor,
          [5.0],
          {},
        );

        final data = await result.getData() as Float32List;
        expect(data[0], equals(6.0));
        expect(data[1], equals(7.0));
        expect(data[2], equals(8.0));
        expect(data[3], equals(9.0));

        result.destroy();
      });

      test('multiplyScalar should multiply all elements by scalar', () async {
        final result = await TensorOperationRegistry.executeOperation(
          'multiplyScalar',
          testTensor,
          [2.0],
          {},
        );

        final data = await result.getData() as Float32List;
        expect(data[0], equals(2.0));
        expect(data[1], equals(4.0));
        expect(data[2], equals(6.0));
        expect(data[3], equals(8.0));

        result.destroy();
      });

      test('subtractScalar should subtract scalar from all elements', () async {
        final result = await TensorOperationRegistry.executeOperation(
          'subtractScalar',
          testTensor,
          [1.0],
          {},
        );

        final data = await result.getData() as Float32List;
        expect(data[0], equals(0.0));
        expect(data[1], equals(1.0));
        expect(data[2], equals(2.0));
        expect(data[3], equals(3.0));

        result.destroy();
      });

      test('divideScalar should divide all elements by scalar', () async {
        final result = await TensorOperationRegistry.executeOperation(
          'divideScalar',
          testTensor,
          [2.0],
          {},
        );

        final data = await result.getData() as Float32List;
        expect(data[0], equals(0.5));
        expect(data[1], equals(1.0));
        expect(data[2], equals(1.5));
        expect(data[3], equals(2.0));

        result.destroy();
      });

      test('should throw error for missing scalar argument', () async {
        expect(
          () => TensorOperationRegistry.executeOperation(
            'addScalar',
            testTensor,
            [], // Missing argument
            {},
          ),
          throwsA(isA<ArgumentError>()),
        );
      });

      test('should throw error for invalid scalar argument type', () async {
        expect(
          () => TensorOperationRegistry.executeOperation(
            'addScalar',
            testTensor,
            ['not_a_number'], // Wrong type
            {},
          ),
          throwsA(isA<Exception>()),
        );
      });
    });

    group('Unary Operations', () {
      late Tensor testTensor;

      setUp(() async {
        testTensor = await Tensor.create([4]);
        await testTensor.write(Float32List.fromList([1.0, 4.0, 9.0, 16.0]));
      });

      tearDown(() {
        testTensor.destroy();
      });

      test('abs should return absolute values', () async {
        // Create tensor with negative values
        final negativeTensor = await Tensor.create([4]);
        await negativeTensor.write(
          Float32List.fromList([-1.0, 2.0, -3.0, 4.0]),
        );

        final result = await TensorOperationRegistry.executeOperation(
          'abs',
          negativeTensor,
          [],
          {},
        );

        final data = await result.getData() as Float32List;
        expect(data[0], equals(1.0));
        expect(data[1], equals(2.0));
        expect(data[2], equals(3.0));
        expect(data[3], equals(4.0));

        negativeTensor.destroy();
        result.destroy();
      });

      test('sqrt should return square roots', () async {
        final result = await TensorOperationRegistry.executeOperation(
          'sqrt',
          testTensor,
          [],
          {},
        );

        final data = await result.getData() as Float32List;
        expect(data[0], closeTo(1.0, 0.001));
        expect(data[1], closeTo(2.0, 0.001));
        expect(data[2], closeTo(3.0, 0.001));
        expect(data[3], closeTo(4.0, 0.001));

        result.destroy();
      });

      test('sin should return sine values', () async {
        // Create tensor with known angles
        final angleTensor = await Tensor.create([3]);
        await angleTensor.write(
          Float32List.fromList([0.0, 1.5708, 3.14159]),
        ); // 0, π/2, π

        final result = await TensorOperationRegistry.executeOperation(
          'sin',
          angleTensor,
          [],
          {},
        );

        final data = await result.getData() as Float32List;
        expect(data[0], closeTo(0.0, 0.001));
        expect(data[1], closeTo(1.0, 0.001));
        expect(data[2], closeTo(0.0, 0.001));

        angleTensor.destroy();
        result.destroy();
      });

      test('tanh should return hyperbolic tangent values', () async {
        // Create tensor with values around 0
        final inputTensor = await Tensor.create([3]);
        await inputTensor.write(Float32List.fromList([0.0, 1.0, -1.0]));

        final result = await TensorOperationRegistry.executeOperation(
          'tanh',
          inputTensor,
          [],
          {},
        );

        final data = await result.getData() as Float32List;
        expect(data[0], closeTo(0.0, 0.001));
        expect(data[1], closeTo(0.7616, 0.001)); // tanh(1.0)
        expect(data[2], closeTo(-0.7616, 0.001)); // tanh(-1.0)

        inputTensor.destroy();
        result.destroy();
      });
    });

    group('Tensor-Tensor Operations', () {
      late Tensor tensor1;
      late Tensor tensor2;

      setUp(() async {
        tensor1 = await Tensor.create([4]);
        await tensor1.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));

        tensor2 = await Tensor.create([4]);
        await tensor2.write(Float32List.fromList([5.0, 6.0, 7.0, 8.0]));
      });

      tearDown(() {
        tensor1.destroy();
        tensor2.destroy();
      });

      test('add should add corresponding elements', () async {
        final result = await TensorOperationRegistry.executeOperation(
          'add',
          tensor1,
          [tensor2],
          {},
        );

        final data = await result.getData() as Float32List;
        expect(data[0], equals(6.0));
        expect(data[1], equals(8.0));
        expect(data[2], equals(10.0));
        expect(data[3], equals(12.0));

        result.destroy();
      });

      test('multiply should multiply corresponding elements', () async {
        final result = await TensorOperationRegistry.executeOperation(
          'multiply',
          tensor1,
          [tensor2],
          {},
        );

        final data = await result.getData() as Float32List;
        expect(data[0], equals(5.0));
        expect(data[1], equals(12.0));
        expect(data[2], equals(21.0));
        expect(data[3], equals(32.0));

        result.destroy();
      });

      test('max should return element-wise maximum', () async {
        final result = await TensorOperationRegistry.executeOperation(
          'max',
          tensor1,
          [tensor2],
          {},
        );

        final data = await result.getData() as Float32List;
        expect(data[0], equals(5.0));
        expect(data[1], equals(6.0));
        expect(data[2], equals(7.0));
        expect(data[3], equals(8.0));

        result.destroy();
      });

      test('min should return element-wise minimum', () async {
        final result = await TensorOperationRegistry.executeOperation(
          'min',
          tensor1,
          [tensor2],
          {},
        );

        final data = await result.getData() as Float32List;
        expect(data[0], equals(1.0));
        expect(data[1], equals(2.0));
        expect(data[2], equals(3.0));
        expect(data[3], equals(4.0));

        result.destroy();
      });
    });

    group('Custom Operations', () {
      late Tensor testTensor;

      setUp(() async {
        testTensor = await Tensor.create([4]);
        await testTensor.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));
      });

      tearDown(() {
        testTensor.destroy();
      });

      test('should register custom operation', () {
        TensorOperationRegistry.registerCustomOperation(
          'double',
          (tensor, args, named) async {
            return await tensor.multiplyScalar(2.0);
          },
          requiredArgs: 0,
          preservesShape: true,
        );

        expect(TensorOperationRegistry.hasOperation('double'), isTrue);
      });

      test('should execute custom operation', () async {
        TensorOperationRegistry.registerCustomOperation(
          'triple',
          (tensor, args, named) async {
            return await tensor.multiplyScalar(3.0);
          },
          requiredArgs: 0,
          preservesShape: true,
        );

        final result = await TensorOperationRegistry.executeOperation(
          'triple',
          testTensor,
          [],
          {},
        );

        final data = await result.getData() as Float32List;
        expect(data[0], equals(3.0));
        expect(data[1], equals(6.0));
        expect(data[2], equals(9.0));
        expect(data[3], equals(12.0));

        result.destroy();
      });
    });

    group('Error Handling', () {
      late Tensor testTensor;

      setUp(() async {
        testTensor = await Tensor.create([4]);
        await testTensor.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));
      });

      tearDown(() {
        testTensor.destroy();
      });

      test('should throw exception for unknown operation', () async {
        expect(
          () => TensorOperationRegistry.executeOperation(
            'unknownOperation',
            testTensor,
            [],
            {},
          ),
          throwsA(isA<Exception>()),
        );
      });

      test(
        'should provide helpful error message for unknown operation',
        () async {
          try {
            await TensorOperationRegistry.executeOperation(
              'unknownOperation',
              testTensor,
              [],
              {},
            );
            fail('Should have thrown exception');
          } catch (e) {
            expect(e.toString(), contains('Unknown tensor operation'));
            expect(e.toString(), contains('Available operations'));
          }
        },
      );

      test('should throw error for insufficient arguments', () async {
        expect(
          () => TensorOperationRegistry.executeOperation(
            'addScalar',
            testTensor,
            [], // Missing required argument
            {},
          ),
          throwsA(isA<ArgumentError>()),
        );
      });
    });

    late Tensor testTensor;

    setUp(() async {
      testTensor = await Tensor.create([4]);
      await testTensor.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));
    });

    tearDown(() {
      testTensor.destroy();
    });

    test('sum with 2D tensor reduces along last dimension', () async {
      // Create 2D tensor [2, 2] with values [[1,2], [3,4]]
      final tensor2d = await Tensor.create([2, 2]);
      await tensor2d.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));

      final result = await TensorOperationRegistry.executeOperation(
        'sum',
        tensor2d,
        [],
        {},
      );

      // Should reduce [2, 2] to [2] by summing along last dimension
      expect(result.shape, equals([2]));
      final data = await result.getData() as Float32List;
      expect(data[0], equals(3.0)); // 1.0 + 2.0
      expect(data[1], equals(7.0)); // 3.0 + 4.0

      tensor2d.destroy();
      result.destroy();
    });

    test('mean with 2D tensor reduces along last dimension', () async {
      // Create 2D tensor [2, 2] with values [[1,2], [3,4]]
      final tensor2d = await Tensor.create([2, 2]);
      await tensor2d.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));

      final result = await TensorOperationRegistry.executeOperation(
        'mean',
        tensor2d,
        [],
        {},
      );

      // Should reduce [2, 2] to [2] by averaging along last dimension
      expect(result.shape, equals([2]));
      final data = await result.getData() as Float32List;
      expect(data[0], equals(1.5)); // (1.0 + 2.0) / 2
      expect(data[1], equals(3.5)); // (3.0 + 4.0) / 2

      tensor2d.destroy();
      result.destroy();
    });

    test('sum works with 2x3 tensor like in working tests', () async {
      // Test with 2x3 tensor exactly like the passing tests you showed
      final tensor2x3 = await Tensor.create([2, 3]);
      await tensor2x3.write(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
      );

      final result = await TensorOperationRegistry.executeOperation(
        'sum',
        tensor2x3,
        [],
        {},
      );

      expect(result.shape, equals([2]));
      final data = await result.getData() as Float32List;
      expect(data[0], equals(6.0)); // 1+2+3
      expect(data[1], equals(15.0)); // 4+5+6

      tensor2x3.destroy();
      result.destroy();
    });

    test('mean works with 2x3 tensor like in working tests', () async {
      // Test with 2x3 tensor exactly like the passing tests you showed
      final tensor2x3 = await Tensor.create([2, 3]);
      await tensor2x3.write(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
      );

      final result = await TensorOperationRegistry.executeOperation(
        'mean',
        tensor2x3,
        [],
        {},
      );

      expect(result.shape, equals([2]));
      final data = await result.getData() as Float32List;
      expect(data[0], closeTo(2.0, 1e-5)); // (1+2+3)/3
      expect(data[1], closeTo(5.0, 1e-5)); // (4+5+6)/3

      tensor2x3.destroy();
      result.destroy();
    });
  });

  group('TensorOperation', () {
    setUpAll(() async {
      await Minigpu().init();
      TensorOperationRegistry.initialize();
    });

    group('Construction', () {
      test('should create tensor stage operation with default keys', () {
        final operation = TensorOperation(
          operationId: 'test_op',
          operation: 'abs',
        );

        expect(operation.operationId, equals('test_op'));
        expect(operation.operation, equals('abs'));
        expect(operation.inputKeys, equals(['input']));
        expect(operation.outputKeys, equals(['output']));
      });

      test('should create tensor stage operation with custom keys', () {
        final operation = TensorOperation(
          operationId: 'test_op',
          operation: 'multiplyScalar',
          inputKeys: ['tensor_in'],
          outputKeys: ['tensor_out'],
          arguments: {'0': 2.0},
        );

        expect(operation.inputKeys, equals(['tensor_in']));
        expect(operation.outputKeys, equals(['tensor_out']));
        expect(operation.arguments, containsPair('0', 2.0));
      });
    });

    group('Execution', () {
      late Tensor testTensor;

      setUp(() async {
        testTensor = await Tensor.create([4]);
        await testTensor.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));
      });

      tearDown(() {
        testTensor.destroy();
      });

      test('should execute simple operation', () async {
        final operation = TensorOperation(
          operationId: 'test_abs',
          operation: 'abs',
        );

        final result = await operation.execute(
          {'input': testTensor},
          {},
          ResourceCache.forOperation('test_abs'),
        );

        expect(result.containsKey('output'), isTrue);
        expect(result['output']!.shape, equals([4]));

        result['output']!.destroy();
      });

      test('should execute operation with scalar argument', () async {
        final operation = TensorOperation(
          operationId: 'test_multiply',
          operation: 'multiplyScalar',
          arguments: {'0': 3.0},
        );

        final result = await operation.execute(
          {'input': testTensor},
          {},
          ResourceCache.forOperation('test_multiply'),
        );

        expect(result.containsKey('output'), isTrue);

        final data = await result['output']!.getData() as Float32List;
        expect(data[0], equals(3.0));
        expect(data[1], equals(6.0));
        expect(data[2], equals(9.0));
        expect(data[3], equals(12.0));

        result['output']!.destroy();
      });

      test('should execute operation with parameter substitution', () async {
        final operation = TensorOperation(
          operationId: 'test_param',
          operation: 'multiplyScalar',
          arguments: {'0': 'param:scale_factor'},
        );

        final result = await operation.execute(
          {'input': testTensor},
          {'scale_factor': 2.5},
          ResourceCache.forOperation('test_param'),
        );

        expect(result.containsKey('output'), isTrue);

        final data = await result['output']!.getData() as Float32List;
        expect(data[0], equals(2.5));
        expect(data[1], equals(5.0));
        expect(data[2], equals(7.5));
        expect(data[3], equals(10.0));

        result['output']!.destroy();
      });

      test('should handle multiple input-output pairs', () async {
        final tensor2 = await Tensor.create([4]);
        await tensor2.write(Float32List.fromList([5.0, 6.0, 7.0, 8.0]));

        final operation = TensorOperation(
          operationId: 'test_multi',
          operation: 'abs',
          inputKeys: ['input1', 'input2'],
          outputKeys: ['output1', 'output2'],
        );

        final result = await operation.execute(
          {'input1': testTensor, 'input2': tensor2},
          {},
          ResourceCache.forOperation('test_multi'),
        );

        expect(result.containsKey('output1'), isTrue);
        expect(result.containsKey('output2'), isTrue);
        expect(result['output1']!.shape, equals([4]));
        expect(result['output2']!.shape, equals([4]));

        tensor2.destroy();
        result['output1']!.destroy();
        result['output2']!.destroy();
      });

      test('should validate inputs', () async {
        final operation = TensorOperation(
          operationId: 'test_validate',
          operation: 'abs',
          inputKeys: ['required_input'],
        );

        expect(
          () => operation.execute(
            {'wrong_input': testTensor},
            {},
            ResourceCache.forOperation('test_validate'),
          ),
          throwsA(isA<Exception>()),
        );
      });
    });

    group('Serialization', () {
      test('should serialize to config', () {
        final operation = TensorOperation(
          operationId: 'test_serialize',
          operation: 'multiplyScalar',
          inputKeys: ['in'],
          outputKeys: ['out'],
          arguments: {'0': 2.0},
          namedArguments: {'factor': 'param:scale'},
        );

        final config = operation.toConfig();

        expect(config['type'], equals('tensor'));
        expect(config['operationId'], equals('test_serialize'));
        expect(config['operation'], equals('multiplyScalar'));
        expect(config['inputKeys'], equals(['in']));
        expect(config['outputKeys'], equals(['out']));
        expect(config['arguments'], containsPair('0', 2.0));
        expect(config['namedArguments'], containsPair('factor', 'param:scale'));
      });

      test('should deserialize from config', () {
        final config = {
          'operationId': 'test_deserialize',
          'operation': 'addScalar',
          'inputKeys': ['tensor_in'],
          'outputKeys': ['tensor_out'],
          'arguments': {'0': 5.0},
          'namedArguments': {'axis': 1},
        };

        final operation = TensorOperation.fromConfig(config);

        expect(operation.operationId, equals('test_deserialize'));
        expect(operation.operation, equals('addScalar'));
        expect(operation.inputKeys, equals(['tensor_in']));
        expect(operation.outputKeys, equals(['tensor_out']));
        expect(operation.arguments, containsPair('0', 5.0));
        expect(operation.namedArguments, containsPair('axis', 1));
      });

      test('should use defaults when deserializing incomplete config', () {
        final config = {'operationId': 'test_defaults', 'operation': 'sqrt'};

        final operation = TensorOperation.fromConfig(config);

        expect(operation.inputKeys, equals(['input']));
        expect(operation.outputKeys, equals(['output']));
        expect(operation.arguments, isEmpty);
        expect(operation.namedArguments, isEmpty);
      });
    });
  });
}
