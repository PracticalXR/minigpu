import 'dart:math' as math;
import 'dart:typed_data';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';

Future<void> main() async {
  group('Tensor Operation tests', () {
    test('Tensor elementwise addition', () async {
      var shape = [4];
      var aData = Float32List.fromList([1, 2, 3, 4]);
      var bData = Float32List.fromList([4, 3, 2, 1]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);
      var result = await tensorA.add(tensorB);
      var resultData = await result.getData();
      expect(resultData, equals(Float32List.fromList([5, 5, 5, 5])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('Tensor multiply scalar', () async {
      var shape = [4];
      var dataIn = Float32List.fromList([1, 2, 3, 4]);
      var tensor = await Tensor.create(shape, data: dataIn);
      var result = await tensor.multiplyScalar(2);
      var dataOut = await result.getData();
      expect(dataOut, equals(Float32List.fromList([2, 4, 6, 8])));
      tensor.destroy();
      result.destroy();
    });

    test('Tensor reshape carrying same data buffer', () async {
      var shape = [2, 2];
      var initialData = Float32List.fromList([1, 2, 3, 4]);
      var tensor = await Tensor.create(shape, data: initialData);
      var reshaped = tensor.reshape([4]);
      expect(reshaped.shape, equals([4]));
      var data = await reshaped.getData();
      expect(data, equals(initialData));
      tensor.destroy();
      // Assuming reshape does not require destroy on reshaped as it reuses the same buffer.
    });

    test('Tensor reshape with mismatched size throws exception', () async {
      var shape = [2, 2];
      var tensor = await Tensor.create(shape);
      expect(() => tensor.reshape([3]), throwsException);
      tensor.destroy();
    });

    test('Tensor elementwise subtraction', () async {
      var shape = [4];
      var aData = Float32List.fromList([4, 4, 4, 4]);
      var bData = Float32List.fromList([1, 1, 1, 1]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);
      var result = await tensorA.subtract(tensorB);
      var resultData = await result.getData();
      expect(resultData, equals(Float32List.fromList([3, 3, 3, 3])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('Tensor elementwise multiplication', () async {
      var shape = [4];
      var aData = Float32List.fromList([1, 2, 3, 4]);
      var bData = Float32List.fromList([2, 3, 4, 5]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);
      var result = await tensorA.multiply(tensorB);
      var resultData = await result.getData();
      expect(resultData, equals(Float32List.fromList([2, 6, 12, 20])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('Operator overloads for +, -, and *', () async {
      var shape = [3];
      var aData = Float32List.fromList([1, 2, 3]);
      var bData = Float32List.fromList([4, 5, 6]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);

      var addResult = await (tensorA + tensorB);
      var addData = await addResult.getData();
      expect(addData, equals(Float32List.fromList([5, 7, 9])));

      var subResult = await (tensorB - tensorA);
      var subData = await subResult.getData();
      expect(subData, equals(Float32List.fromList([3, 3, 3])));

      var scalarResult = await (tensorA * 3);
      var scalarData = await scalarResult.getData();
      expect(scalarData, equals(Float32List.fromList([3, 6, 9])));

      var multResult = await (tensorA * tensorB);
      var multData = await multResult.getData();
      expect(multData, equals(Float32List.fromList([4, 10, 18])));
      tensorA.destroy();
      tensorB.destroy();
      addResult.destroy();
      subResult.destroy();
      scalarResult.destroy();
      multResult.destroy();
    });

    test('Matrix multiplication (matMul) for rank-2 tensors', () async {
      var aData = Float32List.fromList([1, 2, 3, 4]);
      var bData = Float32List.fromList([5, 6, 7, 8]);
      var tensorA = await Tensor.create([2, 2], data: aData);
      var tensorB = await Tensor.create([2, 2], data: bData);
      var result = await tensorA.matMul(tensorB);
      var resultData = await result.getData();
      expect(result.shape, equals([2, 2]));
      expect(resultData, equals(Float32List.fromList([19, 22, 43, 50])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('Batched matrix multiplication (matMul) for higher-rank tensors',
        () async {
      var aData = Float32List.fromList([
        1, 2, 3, 4, 5, 6, // Batch 0: 2x3
        2, 3, 4, 5, 6, 7, // Batch 1: 2x3
      ]);
      var bData = Float32List.fromList([
        7, 8, 9, 10, 11, 12, // Batch 0: 3x2
        8, 9, 10, 11, 12, 13, // Batch 1: 3x2
      ]);
      var tensorA = await Tensor.create([2, 2, 3], data: aData);
      var tensorB = await Tensor.create([2, 3, 2], data: bData);
      var result = await tensorA.matMul(tensorB);
      var resultData = await result.getData();
      expect(result.shape, equals([2, 2, 2]));
      expect(
          resultData,
          equals(Float32List.fromList([
            58, 64, 139, 154, // Batch 0
            94, 103, 184, 202 // Batch 1
          ])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('Matrix multiplication with incompatible shapes throws exception',
        () async {
      var tensorA = await Tensor.create([2, 2]);
      var tensorB = await Tensor.create([3, 3]);
      expect(() => tensorA.matMul(tensorB), throwsException);
      tensorA.destroy();
      tensorB.destroy();
    });
  });

  // New group to cover the additional operations
  group('Additional Tensor Operations', () {
    test('powScalar computes elementwise exponentiation', () async {
      var shape = [4];
      var data = Float32List.fromList([1, 2, 3, 4]);
      var tensor = await Tensor.create(shape, data: data);
      var result = await tensor.powScalar(2);
      var resultData = await result.getData();
      expect(resultData, equals(Float32List.fromList([1, 4, 9, 16])));
      tensor.destroy();
      result.destroy();
    });

    test('log computes natural logarithm', () async {
      // Using known values: ln(1)=0, ln(e)=1
      var shape = [2];
      var data = Float32List.fromList([1.0, math.e]);
      var tensor = await Tensor.create<Float32List>(shape, data: data);
      var result = await tensor.log();
      var resultData = await result.getData();
      expect(resultData[0], closeTo(0.0, 1e-5));
      expect(resultData[1], closeTo(1.0, 1e-5));
      tensor.destroy();
      result.destroy();
    });

    test('exp computes elementwise exponential', () async {
      var shape = [2];
      var data = Float32List.fromList([0, 1]);
      var tensor = await Tensor.create(shape, data: data);
      var result = await tensor.exp();
      var resultData = await result.getData();
      expect(resultData[0], closeTo(1.0, 1e-5));
      expect(resultData[1], closeTo(math.e, 1e-5));
      tensor.destroy();
      result.destroy();
    });

    test('sqrt computes elementwise square root', () async {
      var shape = [4];
      var data = Float32List.fromList([1, 4, 9, 16]);
      var tensor = await Tensor.create(shape, data: data);
      var result = await tensor.sqrt();
      var resultData = await result.getData();
      expect(resultData, equals(Float32List.fromList([1, 2, 3, 4])));
      tensor.destroy();
      result.destroy();
    });

    test('modScalar computes elementwise modulus', () async {
      var shape = [4];
      var data = Float32List.fromList([7, 8, 9, 10]);
      var tensor = await Tensor.create(shape, data: data);
      var result = await tensor.modScalar(3);
      var resultData = await result.getData();
      // mod(7,3)=1, mod(8,3)=2, mod(9,3)=0, mod(10,3)=1
      expect(resultData, equals(Float32List.fromList([1, 2, 0, 1])));
      tensor.destroy();
      result.destroy();
    });

    test('Operator overload % computes elementwise modulus for two tensors',
        () async {
      var shape = [4];
      var aData = Float32List.fromList([7, 8, 9, 10]);
      var bData = Float32List.fromList([4, 4, 4, 4]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);
      var result = await (tensorA % tensorB);
      var resultData = await result.getData();
      // Expected: mod(7,4)=3, mod(8,4)=0, mod(9,4)=1, mod(10,4)=2
      expect(resultData, equals(Float32List.fromList([3, 0, 1, 2])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('greaterThan returns correct elementwise comparison', () async {
      var shape = [3];
      var aData = Float32List.fromList([1, 3, 5]);
      var bData = Float32List.fromList([2, 2, 6]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);
      var result = await tensorA.greaterThan(tensorB);
      var resultData = await result.getData();
      // Expected: [0, 1, 0]
      expect(resultData, equals(Float32List.fromList([0, 1, 0])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('lessThan returns correct elementwise comparison', () async {
      var shape = [3];
      var aData = Float32List.fromList([1, 3, 5]);
      var bData = Float32List.fromList([2, 2, 6]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);
      var result = await tensorA.lessThan(tensorB);
      var resultData = await result.getData();
      // Expected: [1, 0, 1]
      expect(resultData, equals(Float32List.fromList([1, 0, 1])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('equalTo returns correct elementwise equality', () async {
      var shape = [3];
      var aData = Float32List.fromList([1, 2, 3]);
      var bData = Float32List.fromList([1, 3, 3]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);
      var result = await tensorA.equalTo(tensorB);
      var resultData = await result.getData();
      // Expected: [1, 0, 1]
      expect(resultData, equals(Float32List.fromList([1, 0, 1])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('notEqualTo returns correct elementwise inequality', () async {
      var shape = [3];
      var aData = Float32List.fromList([1, 2, 3]);
      var bData = Float32List.fromList([1, 3, 3]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);
      var result = await tensorA.notEqualTo(tensorB);
      var resultData = await result.getData();
      // Expected: [0, 1, 0]
      expect(resultData, equals(Float32List.fromList([0, 1, 0])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('greaterThanOrEqual returns correct elementwise comparison', () async {
      var shape = [3];
      var aData = Float32List.fromList([1, 3, 5]);
      var bData = Float32List.fromList([1, 4, 4]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);
      var result = await tensorA.greaterThanOrEqual(tensorB);
      var resultData = await result.getData();
      // Expected: [1, 0, 1] since 1>=1 true, 3>=4 false, 5>=4 true
      expect(resultData, equals(Float32List.fromList([1, 0, 1])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('lessThanOrEqual returns correct elementwise comparison', () async {
      var shape = [3];
      var aData = Float32List.fromList([1, 3, 5]);
      var bData = Float32List.fromList([1, 4, 6]);
      var tensorA = await Tensor.create(shape, data: aData);
      var tensorB = await Tensor.create(shape, data: bData);
      var result = await tensorA.lessThanOrEqual(tensorB);
      var resultData = await result.getData();
      // Expected: [1, 1, 1] since 1<=1, 3<=4, 5<=6
      expect(resultData, equals(Float32List.fromList([1, 1, 1])));
      tensorA.destroy();
      tensorB.destroy();
      result.destroy();
    });

    test('abs returns absolute values', () async {
      var shape = [4];
      var data = Float32List.fromList([-1, -2, 3, -4]);
      var tensor = await Tensor.create(shape, data: data);
      var result = await tensor.abs();
      var resultData = await result.getData();
      expect(resultData, equals(Float32List.fromList([1, 2, 3, 4])));
      tensor.destroy();
      result.destroy();
    });

    test('Sum reduction along the last dimension', () async {
      // For a tensor of shape [2, 3]:
      // Row 1: 1+2+3 = 6, Row 2: 4+5+6 = 15
      var shape = [2, 3];
      var data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
      var tensor = await Tensor.create(shape, data: data);
      var sumTensor = await tensor.sum();
      var resultData = await sumTensor.getData();
      expect(sumTensor.shape, equals([2]));
      expect(resultData, equals(Float32List.fromList([6, 15])));
      tensor.destroy();
      sumTensor.destroy();
    });

    test('Mean reduction along the last dimension', () async {
      // For a tensor of shape [2, 3]:
      // Row 1 mean: (1+2+3)/3 = 2, Row 2 mean: (4+5+6)/3 = 5
      var shape = [2, 3];
      var data = Float32List.fromList([1, 2, 3, 4, 5, 6]);
      var tensor = await Tensor.create<Float32List>(shape, data: data);
      var meanTensor = await tensor.mean();
      var resultData = await meanTensor.getData();
      expect(meanTensor.shape, equals([2]));
      expect(resultData[0], closeTo(2.0, 1e-5));
      expect(resultData[1], closeTo(5.0, 1e-5));
      tensor.destroy();
      meanTensor.destroy();
    });

    test('Max reduction along the last dimension', () async {
      // For a tensor of shape [2, 3]:
      // Row 1: max of [1, 5, 3] is 5, Row 2: max of [4, 2, 6] is 6
      var shape = [2, 3];
      var data = Float32List.fromList([1, 5, 3, 4, 2, 6]);
      var tensor = await Tensor.create(shape, data: data);
      var maxTensor = await tensor.maxReduction();
      var resultData = await maxTensor.getData();
      expect(maxTensor.shape, equals([2]));
      expect(resultData, equals(Float32List.fromList([5, 6])));
      tensor.destroy();
      maxTensor.destroy();
    });

    test('Min reduction along the last dimension', () async {
      // For a tensor of shape [2, 3]:
      // Row 1: min of [1, 5, 3] is 1, Row 2: min of [4, 2, 6] is 2
      var shape = [2, 3];
      var data = Float32List.fromList([1, 5, 3, 4, 2, 6]);
      var tensor = await Tensor.create(shape, data: data);
      var minTensor = await tensor.minReduction();
      var resultData = await minTensor.getData();
      expect(minTensor.shape, equals([2]));
      expect(resultData, equals(Float32List.fromList([1, 2])));
      tensor.destroy();
      minTensor.destroy();
    });

    test('Argmax reduction along the last dimension', () async {
      // For a tensor of shape [2, 4]:
      // Row 1: [1, 7, 3, 5] -> argmax index 1, Row 2: [4, 6, 2, 8] -> argmax index 3.
      var shape = [2, 4];
      var data = Float32List.fromList([1, 7, 3, 5, 4, 6, 2, 8]);
      var tensor = await Tensor.create(shape, data: data);
      var argmaxTensor = await tensor.argmax();
      var resultData = await argmaxTensor.getData();
      expect(argmaxTensor.shape, equals([2]));
      // Argmax is stored as f32 values.
      expect(resultData, equals(Float32List.fromList([1, 3])));
      tensor.destroy();
      argmaxTensor.destroy();
    });
  });
}
