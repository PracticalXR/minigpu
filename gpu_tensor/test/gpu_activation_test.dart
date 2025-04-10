import 'dart:math' as math;
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

Future<void> main() async {
  group('Activation Tests', () {
    test('ReLU activation', () async {
      // Create a tensor with negative, zero, and positive values.
      List<int> shape = [4];
      Float32List inputData = Float32List.fromList([-1.0, 0.0, 1.0, 2.0]);

      Tensor<Float32List> tensor = await Tensor.create(shape, data: inputData);
      Tensor<Float32List> reluTensor = await tensor.relu();
      Float32List result = await reluTensor.getData();

      List<double> expected = [0.0, 0.0, 1.0, 2.0];
      expect(result.length, equals(expected.length));
      for (int i = 0; i < result.length; i++) {
        expect(result[i], closeTo(expected[i], 1e-5));
      }
      tensor.destroy();
      reluTensor.destroy();
    });

    test('Sigmoid activation', () async {
      List<int> shape = [4];
      Float32List inputData = Float32List.fromList([-1.0, 0.0, 1.0, 2.0]);

      Tensor<Float32List> tensor = await Tensor.create(shape, data: inputData);
      Tensor<Float32List> sigmoidTensor = await tensor.sigmoid();
      Float32List result = await sigmoidTensor.getData();

      // Expected: sigmoid(-1) ~ 0.26894, sigmoid(0) ~ 0.5, sigmoid(1) ~ 0.73106, sigmoid(2) ~ 0.88080
      List<double> expected = [0.26894, 0.5, 0.73106, 0.88080];
      expect(result.length, equals(expected.length));
      for (int i = 0; i < result.length; i++) {
        expect(result[i], closeTo(expected[i], 1e-4));
      }
      tensor.destroy();
      sigmoidTensor.destroy();
    });

    test('Sin activation', () async {
      // Test sine for known angles.
      List<int> shape = [4];
      Float32List inputData = Float32List.fromList([
        0.0,
        math.pi / 2,
        math.pi,
        3 * math.pi / 2,
      ]);

      Tensor<Float32List> tensor =
          await Tensor.create<Float32List>(shape, data: inputData);
      Tensor<Float32List> sinTensor = await tensor.sin();
      Float32List result = await sinTensor.getData();

      // Expected: sin(0)=0, sin(pi/2)=1, sin(pi)=0, sin(3pi/2)=-1.
      List<double> expected = [0.0, 1.0, 0.0, -1.0];
      expect(result.length, equals(expected.length));
      for (int i = 0; i < result.length; i++) {
        expect(result[i], closeTo(expected[i], 1e-5));
      }
      tensor.destroy();
      sinTensor.destroy();
    });

    test('Cos activation', () async {
      // Test cosine for known angles.
      List<int> shape = [4];
      Float32List inputData = Float32List.fromList([
        0.0,
        math.pi / 2,
        math.pi,
        3 * math.pi / 2,
      ]);

      Tensor<Float32List> tensor = await Tensor.create(shape, data: inputData);
      Tensor<Float32List> cosTensor = await tensor.cos();
      final result = await cosTensor.getData();

      // Expected: cos(0)=1, cos(pi/2)=0, cos(pi)=-1, cos(3pi/2)=0.
      List<double> expected = [1.0, 0.0, -1.0, 0.0];
      expect(result.length, equals(expected.length));
      for (int i = 0; i < result.length; i++) {
        expect(result[i], closeTo(expected[i], 1e-5));
      }
      tensor.destroy();
      cosTensor.destroy();
    });
  });
}
