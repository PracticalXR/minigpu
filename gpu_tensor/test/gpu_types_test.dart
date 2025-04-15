import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:minigpu/minigpu.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

void main() {
  group('Tensor Types Tests', () {
    test('Tensor with type Uint8List and nonzero input', () async {
      final shape = [2, 5];

      // Provide nonzero data to check that FFI setData works.
      final input = Int8List.fromList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      Tensor<Int8List> tensor = await Tensor.create<Int8List>(
        shape,
        data: input,
        dataType: BufferDataType.int8,
      );
      final data = await tensor.getData();
      // Expect that we retrieve the non-zero values.
      expect((data as List).toList(), equals(input.toList()));
    });
    test('Default tensor type defaults to Float32List', () async {
      final shape = [2, 5];

      // When no type is specified it should default to Float32List.
      Tensor tensor = await Tensor.create(shape);
      final data = await tensor.getData();
      expect(data, isA<Float32List>());
      expect((data as List).length, equals(10));
    });

    test('Tensor with type Int16List', () async {
      final shape = [2, 5]; // total 6 elements
      Tensor<Int16List> tensor = await Tensor.create<Int16List>(
        shape,
        dataType: BufferDataType.int16,
      );
      final data = await tensor.getData();
      expect(data, isA<Int16List>());
      expect(data.length, equals(10));
    });

    test('Tensor with type Int32List', () async {
      final shape = [2, 5];
      Tensor<Int32List> tensor = await Tensor.create<Int32List>(
        shape,
        dataType: BufferDataType.int32,
      );
      final data = await tensor.getData();
      expect(data, isA<Int32List>());
      expect(data.length, equals(10));
    });

    test('Tensor with type Int64List', () async {
      final shape = [2, 5];
      Tensor<Int64List> tensor = await Tensor.create<Int64List>(
        shape,
        dataType: BufferDataType.int64,
      );
      final data = await tensor.getData();
      expect(data, isA<Int64List>());
      expect(data.length, equals(10));
    });

    test('Tensor with type Uint16List', () async {
      final shape = [2, 5];
      Tensor<Uint16List> tensor = await Tensor.create<Uint16List>(
        shape,
        dataType: BufferDataType.uint16,
      );
      final data = await tensor.getData();
      expect(data, isA<Uint16List>());
      expect(data.length, equals(10));
    });

    test('Tensor with type Uint32List', () async {
      final shape = [2, 5];
      Tensor<Uint32List> tensor = await Tensor.create<Uint32List>(
        shape,
        dataType: BufferDataType.uint32,
      );
      final data = await tensor.getData();
      expect(data, isA<Uint32List>());
      expect(data.length, equals(10));
    });

    test('Tensor with type Float64List', () async {
      final shape = [2, 5];
      Tensor<Float64List> tensor = await Tensor.create<Float64List>(
        shape,
        dataType: BufferDataType.double,
      );
      final data = await tensor.getData();
      expect(data, isA<Float64List>());
      expect(data.length, equals(10));
    });
  });
}
