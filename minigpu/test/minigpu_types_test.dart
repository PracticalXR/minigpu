import 'dart:typed_data';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

void main() {
  group('Buffer Data Types Tests', () {
    late Minigpu minigpu;

    setUpAll(() async {
      minigpu = Minigpu();
      await minigpu.init();
    });

    tearDownAll(() async {
      await minigpu.destroy();
    });

    test('Float32 read/write', () async {
      // Allocate 16 floats.
      final int count = 16;
      final int byteSize = count * Float32List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize, BufferDataType.float32);

      // Create known Float32List
      final Float32List input = Float32List.fromList(
        List.generate(count, (i) => i.toDouble()),
      );
      buffer.write(input, input.length);

      final Float32List output = Float32List(count);
      await buffer.read(output, count, dataType: BufferDataType.float32);

      expect(output, equals(input));
    });

    test('Float64 read/write', () async {
      final int count = 16;
      // Allocate based on the packed type (floats)
      final int bufferSize = count * Float64List.bytesPerElement;
      final buffer = minigpu.createBuffer(bufferSize, BufferDataType.float64);

      final Float64List input = Float64List.fromList(
        List<double>.generate(count, (i) => i.toDouble(), growable: false),
      );
      // Pass the element count (not the byte count)
      buffer.write(input, input.length, dataType: BufferDataType.float64);

      final Float64List output = Float64List(count);
      await buffer.read(output, count, dataType: BufferDataType.float64);

      // Since the GPU packs doubles as floats, compare with a tolerance.
      for (int i = 0; i < count; i++) {
        double expected = input[i];
        expect(
          output[i],
          closeTo(expected, 1e-4),
          reason: 'Element $i should be close to $expected',
        );
      }
    });

    test('Int8 read/write', () async {
      final int count = 16;
      final int byteSize = count * Int8List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize, BufferDataType.int8);

      final Int8List input = Int8List.fromList(
        List.generate(count, (i) => i - 8),
      );
      buffer.write(input, input.lengthInBytes, dataType: BufferDataType.int8);

      final Int8List output = Int8List(count);
      await buffer.read(output, count, dataType: BufferDataType.int8);

      expect(output, equals(input));
    });

    test('Int16 read/write', () async {
      final int count = 16;
      final int byteSize = count * Int16List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize, BufferDataType.int16);

      final Int16List input = Int16List.fromList(
        List.generate(count, (i) => i * 10 - 50),
      );
      buffer.write(input, input.length, dataType: BufferDataType.int16);

      final Int16List output = Int16List(count);
      await buffer.read(output, count, dataType: BufferDataType.int16);

      expect(output, equals(input));
    });

    test('Int32 read/write', () async {
      final int count = 16;
      final int byteSize = count * Int32List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize, BufferDataType.int32);

      final Int32List input = Int32List.fromList(
        List.generate(count, (i) => i * 100 - 500),
      );
      buffer.write(input, input.length, dataType: BufferDataType.int32);

      final Int32List output = Int32List(count);
      await buffer.read(output, count, dataType: BufferDataType.int32);

      expect(output, equals(input));
    });

    test('Int64 read/write', () async {
      final int count = 16;
      final int byteSize = count * Int64List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize, BufferDataType.int64);

      // Use Int64List directly.
      final Int64List input = Int64List.fromList(
        List.generate(count, (i) => i * 1000 - 5000, growable: false),
      );
      // Pass the element count (not the byte count)
      buffer.write(input, input.length, dataType: BufferDataType.int64);

      final Int64List output = Int64List(count);
      await buffer.read(output, count, dataType: BufferDataType.int64);
      expect(output, equals(input));
    });

    test('Uint8 read/write', () async {
      final int count = 16;
      final int byteSize = count * Uint8List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize, BufferDataType.uint8);

      final Uint8List input = Uint8List.fromList(
        List.generate(count, (i) => i),
      );
      buffer.write(input, input.length, dataType: BufferDataType.uint8);

      final Uint8List output = Uint8List(count);
      await buffer.read(output, count, dataType: BufferDataType.uint8);

      expect(output, equals(input));
    });

    test('Uint16 read/write', () async {
      final int count = 16;
      final int byteSize = count * Uint16List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize, BufferDataType.uint16);

      final Uint16List input = Uint16List.fromList(
        List.generate(count, (i) => i * 2),
      );
      buffer.write(input, input.length, dataType: BufferDataType.uint16);

      final Uint16List output = Uint16List(count);
      await buffer.read(output, count, dataType: BufferDataType.uint16);

      expect(output, equals(input));
    });

    test('Uint32 read/write', () async {
      final int count = 16;
      final int byteSize = count * Uint32List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize, BufferDataType.uint32);

      final Uint32List input = Uint32List.fromList(
        List.generate(count, (i) => (i + 1) * 100),
      );
      buffer.write(input, input.length, dataType: BufferDataType.uint32);

      final Uint32List output = Uint32List(count);
      await buffer.read(output, count, dataType: BufferDataType.uint32);

      expect(output, equals(input));
    });
  });

  group('Compute Shader Roundtrip Tests', () {
    late Minigpu minigpu;

    setUpAll(() async {
      minigpu = Minigpu();
      await minigpu.init();
    });

    tearDownAll(() async {
      await minigpu.destroy();
    });

    test('Float32 Roundtrip', () async {
      final int count = 16;
      final int bufSize = getBufferSizeForType(BufferDataType.float32, count);
      final inputBuffer = minigpu.createBuffer(bufSize, BufferDataType.float32);
      final outputBuffer = minigpu.createBuffer(
        bufSize,
        BufferDataType.float32,
      );

      final Float32List input = Float32List.fromList(
        List.generate(count, (i) => i.toDouble(), growable: false),
      );
      // No packing concerns at the test level.
      inputBuffer.write(input, input.length, dataType: BufferDataType.float32);

      final String shaderCode =
          '''
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx < ${count}u) {
    out[idx] = inp[idx];
  }
}
''';
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);
      final int workgroups = ((count + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Float32List output = Float32List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.float32);
      expect(output, equals(input));
    });

    test('Float64 Roundtrip', () async {
      final int count = 16;
      // For doubles the API automatically packs/unpacks.
      final int bufSize = getBufferSizeForType(BufferDataType.float64, count);
      final inputBuffer = minigpu.createBuffer(bufSize, BufferDataType.float64);
      final outputBuffer = minigpu.createBuffer(
        bufSize,
        BufferDataType.float64,
      );

      final Float64List input = Float64List.fromList(
        List.generate(count, (i) => i.toDouble(), growable: false),
      );
      inputBuffer.write(input, input.length, dataType: BufferDataType.float64);

      // WGSL doesn’t support 64-bit floats so the shader sees them as i32 pairs.
      final String shaderCode =
          '''
@group(0) @binding(0) var<storage, read_write> inp: array<i32>;
@group(0) @binding(1) var<storage, read_write> out: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx < ${count * 2}u) {
    out[idx] = inp[idx];
  }
}
''';
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);
      final int workgroups = ((count * 2 + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Float64List output = Float64List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.float64);
      for (int i = 0; i < count; i++) {
        expect(
          output[i],
          closeTo(input[i], 1e-12),
          reason: 'Double element \$i should match',
        );
      }
    });

    test('Int32 Roundtrip', () async {
      final int count = 16;
      final int bufSize = getBufferSizeForType(BufferDataType.int32, count);
      final inputBuffer = minigpu.createBuffer(bufSize, BufferDataType.int32);
      final outputBuffer = minigpu.createBuffer(bufSize, BufferDataType.int32);

      final Int32List input = Int32List.fromList(
        List.generate(count, (i) => i * 100 - 500, growable: false),
      );
      inputBuffer.write(input, input.length, dataType: BufferDataType.int32);

      final String shaderCode =
          '''
@group(0) @binding(0) var<storage, read_write> inp: array<i32>;
@group(0) @binding(1) var<storage, read_write> out: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx < ${count}u) {
    out[idx] = inp[idx];
  }
}
''';
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);
      final int workgroups = ((count + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Int32List output = Int32List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.int32);
      expect(output, equals(input));
    });

    test('Int64 Roundtrip', () async {
      final int count = 16;
      // The API packs Int64 into 2 Int32; the user simply passes Int64List.
      final int bufSize = getBufferSizeForType(BufferDataType.int64, count);
      final inputBuffer = minigpu.createBuffer(bufSize, BufferDataType.int64);
      final outputBuffer = minigpu.createBuffer(bufSize, BufferDataType.int64);

      final Int64List input = Int64List.fromList(
        List.generate(count, (i) => i * 1000 - 5000, growable: false),
      );
      inputBuffer.write(input, input.length, dataType: BufferDataType.int64);

      final String shaderCode =
          '''
@group(0) @binding(0) var<storage, read_write> inp: array<i32>;
@group(0) @binding(1) var<storage, read_write> out: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx < ${count * 2}u) {
    out[idx] = inp[idx];
  }
}
''';
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);
      final int workgroups = ((count * 2 + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Int64List output = Int64List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.int64);
      expect(output, equals(input));
    });

    test('Uint8 Roundtrip', () async {
      final int count = 16;
      // The API packs 4 Uint8 per Uint32.
      final int bufSize = getBufferSizeForType(BufferDataType.uint8, count);
      final inputBuffer = minigpu.createBuffer(count, BufferDataType.uint8);
      final outputBuffer = minigpu.createBuffer(count, BufferDataType.uint8);

      final Uint8List input = Uint8List.fromList(
        List.generate(count, (i) => i, growable: false),
      );
      inputBuffer.write(input, input.length, dataType: BufferDataType.uint8);

      final String shaderCode = '''
@group(0) @binding(0) var<storage, read_write> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  // The array length is managed by the API.
  if (idx < arrayLength(&inp)) {
    out[idx] = inp[idx];
  }
}
''';
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);
      final int workgroups =
          (((bufSize ~/ Uint32List.bytesPerElement) + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Uint8List output = Uint8List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.uint8);
      expect(output, equals(input));
    });

    test('Uint16 Roundtrip', () async {
      final int count = 16;
      // The API packs 2 Uint16 per Uint32.
      final int bufSize = getBufferSizeForType(BufferDataType.uint16, count);
      final inputBuffer = minigpu.createBuffer(count, BufferDataType.uint16);
      final outputBuffer = minigpu.createBuffer(count, BufferDataType.uint16);

      final Uint16List input = Uint16List.fromList(
        List.generate(count, (i) => i * 2, growable: false),
      );
      inputBuffer.write(input, input.length, dataType: BufferDataType.uint16);

      final String shaderCode = '''
@group(0) @binding(0) var<storage, read_write> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx < arrayLength(&inp)) {
    out[idx] = inp[idx];
  }
}
''';
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);
      final int workgroups =
          (((bufSize ~/ Uint32List.bytesPerElement) + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Uint16List output = Uint16List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.uint16);
      expect(output, equals(input));
    });
  });
}
