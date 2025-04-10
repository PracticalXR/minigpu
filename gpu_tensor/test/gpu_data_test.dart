import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

void main() {
  group('Tensor slicing tests', () {
    test('sliceLinear slices a 1D tensor correctly', () async {
      // Create a 1D tensor with 10 elements: 0, 1, ..., 9.
      final originalData =
          Float32List.fromList(List<double>.generate(10, (i) => i.toDouble()));
      // Create the tensor (passing null for gpu for testing purposes).
      Tensor tensor = await Tensor.create([10], data: originalData);

      // Slice from flat index 2 (inclusive) to 7 (exclusive).
      Tensor sliced = await tensor.sliceLinear(start: 2, end: 7);

      final result = await sliced.getData();
      expect(result, equals([2.0, 3.0, 4.0, 5.0, 6.0]));
      expect(sliced.shape, equals([5]));
    });

    test('slice slices a 2D tensor correctly', () async {
      // Create a 2D tensor with shape [2, 3]:
      // [0, 1, 2]
      // [3, 4, 5]
      final originalData = Float32List.fromList([0, 1, 2, 3, 4, 5]);
      Tensor tensor = await Tensor.create([2, 3], data: originalData);

      // Slice: select the second row (index 1) for all columns.
      // startIndices corresponds to [row, col] = [1, 0]
      // endIndices corresponds to [row, col] = [2, 3]
      Tensor sliced = await tensor.slice(
        startIndices: [1, 0],
        endIndices: [2, 3],
      );

      final result = await sliced.getData();
      // Expected shape is [1, 3] with data from the second row.
      expect(sliced.shape, equals([1, 3]));
      expect(result, equals([3.0, 4.0, 5.0]));
    });

    test('slice throws error on invalid multidimensional indices', () async {
      // Create a 2D tensor with shape [2, 3].
      final originalData =
          Float32List.fromList(List<double>.generate(6, (i) => i.toDouble()));
      Tensor tensor = await Tensor.create([2, 3], data: originalData);

      // Provide invalid slice indices (e.g. end index equals start index for a dimension).
      expect(
        () async => await tensor.slice(
          startIndices: [0, 2],
          endIndices: [1, 2],
        ),
        throwsException,
      );
    });

    // Additional tests for higher dimensions.
    group('Multidimensional slicing tests', () {
      test('slice a 3D tensor correctly', () async {
        // Create a 3D tensor with shape [2, 2, 3] (total 12 elements).
        // Data arranged in row-major order:
        // For dimension 0 = 0, sub-tensor:
        //   [ [0, 1, 2],
        //     [3, 4, 5] ]
        // For dimension 0 = 1, sub-tensor:
        //   [ [6, 7, 8],
        //     [9, 10, 11] ]
        final originalData = Float32List.fromList(
            List<double>.generate(12, (i) => i.toDouble()));
        Tensor<Float32List> tensor =
            await Tensor.create([2, 2, 3], data: originalData);

        // Slice: select the first "plane" (dimension 0 = 0) completely.
        // New shape will be [1, 2, 3] and expected data [0, 1, 2, 3, 4, 5].
        Tensor<Float32List> sliced = await tensor.slice(
          startIndices: [0, 0, 0],
          endIndices: [1, 2, 3],
        );

        final result = await sliced.getData();
        expect(sliced.shape, equals([1, 2, 3]));
        expect(result, equals([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));
      });

      test('slice a 4D tensor correctly', () async {
        // Create a 4D tensor with shape [2, 2, 2, 2] (16 elements).
        // Fill with values 0...15 in row-major order.
        final originalData = Float32List.fromList(
            List<double>.generate(16, (i) => i.toDouble()));
        Tensor<Float32List> tensor =
            await Tensor.create([2, 2, 2, 2], data: originalData);

        // select element at indices [0, 0, :, :].
        // for dimension 0 & 1 fixed at 0, and complete slice for dimensions 2 and 3.
        // New shape: [1, 1, 2, 2] (4 elements).
        // Expected data from the original tensor:
        // [ [ [ [0, 1], [2, 3] ] ] ]
        Tensor<Float32List> sliced = await tensor.slice(
          startIndices: [0, 0, 0, 0],
          endIndices: [1, 1, 2, 2],
        );

        final result = await sliced.getData();
        expect(sliced.shape, equals([1, 1, 2, 2]));
        expect(result, equals([0.0, 1.0, 2.0, 3.0]));
      });
    });
    group('Tensor element access and print helpers', () {
      test('getElement returns the correct element for a 2D tensor', () async {
        // 2D tensor with shape [2,3]:
        // [0, 1, 2]
        // [3, 4, 5]
        final data = Float32List.fromList([0, 1, 2, 3, 4, 5]);
        Tensor<Float32List> tensor = await Tensor.create([2, 3], data: data);
        // Get element at row 1, col 1 -> Expected: 4.0
        double value = await tensor.getElement([1, 1]);
        expect(value, equals(4.0));
      });

      test('setElement updates the correct element in a 2D tensor', () async {
        // Create a 2D tensor with shape [2,3] with initial values 0...5.
        final initialData = Float32List.fromList([0, 1, 2, 3, 4, 5]);
        Tensor<Float32List> tensor =
            await Tensor.create([2, 3], data: initialData);

        // Update the element at row 1, col 1 (flat index 4) to the value 99.0.
        await tensor.setElement([1, 1], 99.0);

        // Retrieve the updated value using getElement.
        double value = await tensor.getElement([1, 1]);
        expect(value, equals(99.0));

        // Also, ensure neighboring elements remain unchanged.
        double unchanged1 = await tensor.getElement([0, 0]);
        double unchanged2 = await tensor.getElement([1, 2]);
        expect(unchanged1, equals(0.0));
        expect(unchanged2, equals(5.0));

        tensor.destroy();
      });

      test('getElement throws error on invalid indices', () async {
        final data = Float32List.fromList([0, 1, 2, 3, 4, 5]);
        Tensor<Float32List> tensor = await Tensor.create([2, 3], data: data);
        // Out-of-bound index for row dimension.
        expect(() => tensor.getElement([2, 0]), throwsException);
      });

      test('head returns correct representation for a 2D tensor', () async {
        // Create a 3x3 tensor:
        // [0, 1, 2]
        // [3, 4, 5]
        // [6, 7, 8]
        final data =
            Float32List.fromList(List<double>.generate(9, (i) => i.toDouble()));
        Tensor<Float32List> tensor = await Tensor.create([3, 3], data: data);
        // Get the first 2 rows and 2 columns.
        String headStr = await tensor.head([2, 2]);
        // Expected formatted string: "[[0.0, 1.0], [3.0, 4.0]]"
        expect(headStr, equals("[[0.0, 1.0], [3.0, 4.0]]"));
      });

      test('tail returns correct representation for a 2D tensor', () async {
        // Create a 3x3 tensor:
        // [0, 1, 2]
        // [3, 4, 5]
        // [6, 7, 8]
        final data =
            Float32List.fromList(List<double>.generate(9, (i) => i.toDouble()));
        Tensor<Float32List> tensor = await Tensor.create([3, 3], data: data);
        // Get the last 2 rows and 2 columns.
        String tailStr = await tensor.tail([2, 2]);
        // Expected formatted string: "[[4.0, 5.0], [7.0, 8.0]]"
        expect(tailStr, equals("[[4.0, 5.0], [7.0, 8.0]]"));
      });

      test('getElement returns correct element for a 4D tensor', () async {
        // 4D tensor with shape [2,2,2,2] with data 0..15.
        final data = Float32List.fromList(
            List<double>.generate(16, (i) => i.toDouble()));
        Tensor<Float32List> tensor =
            await Tensor.create([2, 2, 2, 2], data: data);
        // For indices [1, 0, 1, 1]: strides are [8,4,2,1] so flat index = 1*8+0*4+1*2+1 = 11.
        double element = await tensor.getElement([1, 0, 1, 1]);
        expect(element, equals(11.0));
      });

      test('head returns correct representation for a 4D tensor', () async {
        // 4D tensor with shape [2,2,2,2] with data 0..15.
        final data = Float32List.fromList(
            List<double>.generate(16, (i) => i.toDouble()));
        Tensor<Float32List> tensor =
            await Tensor.create([2, 2, 2, 2], data: data);
        // Use head() with counts [1,1,2,2]. This extracts the sub-tensor at indices [0,0,:,:]:
        // Expected sub-tensor (shape [1,1,2,2]) from tensor[0,0,:,:]:
        // Element indices: [0,0,0,0]=0, [0,0,0,1]=1, [0,0,1,0]=2, [0,0,1,1]=3.
        // Formatted as: "[[[0.0, 1.0], [2.0, 3.0]]]"
        String headStr = await tensor.head([1, 1, 2, 2]);
        expect(headStr, equals("[[[[0.0, 1.0], [2.0, 3.0]]]]"));
      });

      test('tail returns correct representation for a 4D tensor', () async {
        // 4D tensor with shape [2,2,2,2] with data 0..15.
        final data = Float32List.fromList(
            List<double>.generate(16, (i) => i.toDouble()));
        Tensor<Float32List> tensor =
            await Tensor.create([2, 2, 2, 2], data: data);
        // Use tail() with counts [1,1,2,2]. Fallback tail computes startIndices = [2-1, 2-1, 2-2, 2-2] = [1,1,0,0],
        // then slices to get sub-tensor of shape [1,1,2,2] from tensor[1,1,:,:]:
        // Elements at tensor[1,1,0,0]=12, [1,1,0,1]=13, [1,1,1,0]=14, [1,1,1,1]=15.
        // Expected formatted output: "[[[12.0, 13.0], [14.0, 15.0]]]"
        String tailStr = await tensor.tail([1, 1, 2, 2]);
        expect(tailStr, equals("[[[[12.0, 13.0], [14.0, 15.0]]]]"));
      });
    });
  });
}
