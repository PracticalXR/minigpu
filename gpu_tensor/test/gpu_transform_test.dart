import 'dart:typed_data';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

Future<void> main() async {
  group('FFT Tests', () {
    test('1D FFT of delta signal', () async {
      // For a forward FFT, the delta input [1+0i, 0+0i, 0+0i, 0+0i]
      // should transform to [1+0i, 1+0i, 1+0i, 1+0i].
      // Here we represent each complex number as 2 floats: real then imag.
      // Thus the input data contains 8 floats.
      var inputData = Float32List.fromList([
        1, 0, // Complex 0: 1+0i
        0, 0, // Complex 1: 0+0i
        0, 0, // Complex 2: 0+0i
        0, 0 // Complex 3: 0+0i
      ]);

      // Create a tensor with shape [4] (4 complex numbers).
      var tensor = await Tensor.create([8],
          data: inputData, dataType: BufferDataType.float32);
      // Perform forward FFT.
      var fftResult = await tensor.fft();

      var resultData = await fftResult.getData();

      // Expected FFT result: each element should be (1, 0)
      var expected = Float32List.fromList([
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
      ]);

      expect(resultData, equals(expected));
    });

    test('2D FFT of delta image', () async {
      // Create a delta image of size 4x4 (i.e. 4 rows, 4 columns)
      // represented as complex numbers (2 floats per element).
      // A delta has value (1,0) in the first position and (0,0) elsewhere.
      int rows = 4;
      int cols = 4;
      var data = Float32List(rows * cols * 2);
      data[0] = 1; // real part of delta; all other values remain 0.

      // The tensor shape is [rows, cols, 2]
      var tensor = await Tensor.create([rows, cols, 2], data: data);

      // Perform 2D FFT.
      var fftResult = await tensor.fft2d();
      var resultData = await fftResult.getData();

      // For a delta input, the FFT output should be (1,0) for each complex number.
      var expected = Float32List(rows * cols * 2);
      for (int i = 0; i < rows * cols; i++) {
        expected[i * 2] = 1; // real part
        expected[i * 2 + 1] = 0; // imaginary part
      }
      expect(resultData, equals(expected));
    });

    test('3D FFT of delta volume', () async {
      // Create a delta volume of size 2x2x2 (i.e. D=2, R=2, C=2)
      // represented as complex numbers (2 floats per element).
      // A delta has value (1,0) in the first position and (0,0) elsewhere.
      int D = 2, R = 2, C = 2;
      // Total number of floats: D * R * C * 2.
      var data = Float32List(D * R * C * 2);
      data[0] = 1; // set delta

      // The tensor shape for a complex tensor is [D, R, C, 2].
      var tensor = await Tensor.create([D, R, C, 2], data: data);

      // Perform 3D FFT.
      var fftResult = await tensor.fft3d();
      var resultData = await fftResult.getData();

      // For a delta input, the FFT output should be (1,0) for each complex number.
      var expected = Float32List(D * R * C * 2);
      for (int i = 0; i < D * R * C; i++) {
        expected[i * 2] = 1; // real part
        expected[i * 2 + 1] = 0; // imaginary part
      }
      expect(resultData, equals(expected));
    });
  });
}
