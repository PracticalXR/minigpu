import 'dart:typed_data';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';

Future<void> main() async {
  group('Tensor tests', () {
    test('Tensor creation with zeros', () async {
      var shape = [2, 3];
      var tensor = await Tensor.create<Float32List>(shape);
      var data = await tensor.getData();
      expect(data.length, equals(6));
      expect(data.every((v) => v == 0), isTrue);
      tensor.destroy();
    });

    test('Tensor creation with initial data', () async {
      var shape = [3];
      var initialData = Float32List.fromList([1, 2, 3]);
      var tensor = await Tensor.create(shape, data: initialData);
      var data = await tensor.getData();
      expect(data, equals(initialData));
      tensor.destroy();
    });
  });
}
