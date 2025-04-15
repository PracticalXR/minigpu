import 'dart:typed_data';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

Future<Minigpu> _initMinigpu() async {
  final minigpu = Minigpu();
  await minigpu.init();
  return minigpu;
}

void main() {
  group('Buffer Data Types Tests', () {
    test('Float32 read/write', () async {
      final minigpu = await _initMinigpu();
      // Allocate 16 floats.
      final int count = 16;
      final int byteSize = count * Float32List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize);

      // Create known Float32List
      final Float32List input = Float32List.fromList(
        List.generate(count, (i) => i.toDouble()),
      );
      buffer.setData(input, input.lengthInBytes);

      final Float32List output = Float32List(count);
      await buffer.read(output, count, dataType: BufferDataType.float);

      expect(output, equals(input));
    });

    test('Float64 read/write', () async {
      final minigpu = await _initMinigpu();
      final int count = 16;
      // Allocate based on the packed type (floats)
      final int bufferSize = count * Float64List.bytesPerElement;
      final buffer = minigpu.createBuffer(bufferSize);

      final Float64List input = Float64List.fromList(
        List<double>.generate(count, (i) => i.toDouble(), growable: false),
      );
      // Pass the element count (not the byte count)
      buffer.setData(input, input.length, dataType: BufferDataType.double);

      final Float64List output = Float64List(count);
      await buffer.read(output, count, dataType: BufferDataType.double);

      // Since the GPU packs doubles as floats, compare with a tolerance.
      for (int i = 0; i < count; i++) {
        double expected = input[i];
        expect(output[i], closeTo(expected, 1e-4),
            reason: 'Element $i should be close to $expected');
      }
    });

    test('Int8 read/write', () async {
      final minigpu = await _initMinigpu();
      final int count = 16;
      final int byteSize = count * Int8List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize);

      final Int8List input = Int8List.fromList(
        List.generate(count, (i) => i - 8),
      );
      buffer.setData(input, input.lengthInBytes, dataType: BufferDataType.int8);

      final Int8List output = Int8List(count);
      await buffer.read(output, count, dataType: BufferDataType.int8);

      expect(output, equals(input));
    });

    test('Int16 read/write', () async {
      final minigpu = await _initMinigpu();
      final int count = 16;
      final int byteSize = count * Int16List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize);

      final Int16List input = Int16List.fromList(
        List.generate(count, (i) => i * 10 - 50),
      );
      buffer.setData(input, input.lengthInBytes,
          dataType: BufferDataType.int16);

      final Int16List output = Int16List(count);
      await buffer.read(output, count, dataType: BufferDataType.int16);

      expect(output, equals(input));
    });

    test('Int32 read/write', () async {
      final minigpu = await _initMinigpu();
      final int count = 16;
      final int byteSize = count * Int32List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize);

      final Int32List input = Int32List.fromList(
        List.generate(count, (i) => i * 100 - 500),
      );
      buffer.setData(input, input.lengthInBytes,
          dataType: BufferDataType.int32);

      final Int32List output = Int32List(count);
      await buffer.read(output, count, dataType: BufferDataType.int32);

      expect(output, equals(input));
    });

    test('Int64 read/write', () async {
      final minigpu = await _initMinigpu();
      final int count = 16;
      final int byteSize = count * Int64List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize);

      // Use Int64List directly.
      final Int64List input = Int64List.fromList(
        List.generate(count, (i) => i * 1000 - 5000, growable: false),
      );
      // Pass the element count (not the byte count)
      buffer.setData(input, input.length, dataType: BufferDataType.int64);

      final Int64List output = Int64List(count);
      await buffer.read(output, count, dataType: BufferDataType.int64);
      expect(output, equals(input));
    });

    test('Uint8 read/write', () async {
      final minigpu = await _initMinigpu();
      final int count = 16;
      final int byteSize = count * Uint8List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize);

      final Uint8List input = Uint8List.fromList(
        List.generate(count, (i) => i),
      );
      buffer.setData(input, input.lengthInBytes,
          dataType: BufferDataType.uint8);

      final Uint8List output = Uint8List(count);
      await buffer.read(output, count, dataType: BufferDataType.uint8);

      expect(output, equals(input));
    });

    test('Uint16 read/write', () async {
      final minigpu = await _initMinigpu();
      final int count = 16;
      final int byteSize = count * Uint16List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize);

      final Uint16List input = Uint16List.fromList(
        List.generate(count, (i) => i * 2),
      );
      buffer.setData(input, input.lengthInBytes,
          dataType: BufferDataType.uint16);

      final Uint16List output = Uint16List(count);
      await buffer.read(output, count, dataType: BufferDataType.uint16);

      expect(output, equals(input));
    });

    test('Uint32 read/write', () async {
      final minigpu = await _initMinigpu();
      final int count = 16;
      final int byteSize = count * Uint32List.bytesPerElement;
      final buffer = minigpu.createBuffer(byteSize);

      final Uint32List input = Uint32List.fromList(
        List.generate(count, (i) => (i + 1) * 100),
      );
      buffer.setData(input, input.lengthInBytes,
          dataType: BufferDataType.uint32);

      final Uint32List output = Uint32List(count);
      await buffer.read(output, count, dataType: BufferDataType.uint32);

      expect(output, equals(input));
    });
  });

  group('Compute Shader Roundtrip Tests', () {
    test('Float32 Roundtrip', () async {
      final minigpu = await _initMinigpu();
      final int count = 16;
      final int bufSize = getBufferSizeForType(BufferDataType.float, count);
      final inputBuffer = minigpu.createBuffer(bufSize);
      final outputBuffer = minigpu.createBuffer(bufSize);

      final Float32List input = Float32List.fromList(
          List.generate(count, (i) => i.toDouble(), growable: false));
      // No packing concerns at the test level.
      inputBuffer.setData(input, input.lengthInBytes,
          dataType: BufferDataType.float);

      final String shaderCode = '''
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
      await outputBuffer.read(output, count, dataType: BufferDataType.float);
      expect(output, equals(input));
    });

    test('Float64 Roundtrip', () async {
      final minigpu = await _initMinigpu();
      final int count = 16;
      // For doubles the API automatically packs/unpacks.
      final int bufSize = getBufferSizeForType(BufferDataType.double, count);
      final inputBuffer = minigpu.createBuffer(bufSize);
      final outputBuffer = minigpu.createBuffer(bufSize);

      final Float64List input = Float64List.fromList(
          List.generate(count, (i) => i.toDouble(), growable: false));
      inputBuffer.setData(input, input.length, dataType: BufferDataType.double);

      // WGSL doesn’t support 64-bit floats so the shader sees them as i32 pairs.
      final String shaderCode = '''
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
      await outputBuffer.read(output, count, dataType: BufferDataType.double);
      for (int i = 0; i < count; i++) {
        expect(output[i], closeTo(input[i], 1e-12),
            reason: 'Double element \$i should match');
      }
    });

    test('Int32 Roundtrip', () async {
      final minigpu = await _initMinigpu();
      final int count = 16;
      final int bufSize = getBufferSizeForType(BufferDataType.int32, count);
      final inputBuffer = minigpu.createBuffer(bufSize);
      final outputBuffer = minigpu.createBuffer(bufSize);

      final Int32List input = Int32List.fromList(
          List.generate(count, (i) => i * 100 - 500, growable: false));
      inputBuffer.setData(input, input.lengthInBytes,
          dataType: BufferDataType.int32);

      final String shaderCode = '''
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
      final minigpu = await _initMinigpu();
      final int count = 16;
      // The API packs Int64 into 2 Int32; the user simply passes Int64List.
      final int bufSize = getBufferSizeForType(BufferDataType.int64, count);
      final inputBuffer = minigpu.createBuffer(bufSize);
      final outputBuffer = minigpu.createBuffer(bufSize);

      final Int64List input = Int64List.fromList(
          List.generate(count, (i) => i * 1000 - 5000, growable: false));
      inputBuffer.setData(input, input.length, dataType: BufferDataType.int64);

      final String shaderCode = '''
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
      final minigpu = await _initMinigpu();
      final int count = 16;
      // The API packs 4 Uint8 per Uint32.
      final int bufSize = getBufferSizeForType(BufferDataType.uint8, count);
      final inputBuffer = minigpu.createBuffer(bufSize);
      final outputBuffer = minigpu.createBuffer(bufSize);

      final Uint8List input =
          Uint8List.fromList(List.generate(count, (i) => i, growable: false));
      inputBuffer.setData(input, input.length, dataType: BufferDataType.uint8);

      final String shaderCode = '''
@group(0) @binding(0) var<storage, read_write> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  // The array length is managed by the API.
  if (idx < ${bufSize ~/ Uint32List.bytesPerElement}u) {
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
      final minigpu = await _initMinigpu();
      final int count = 16;
      // The API packs 2 Uint16 per Uint32.
      final int bufSize = getBufferSizeForType(BufferDataType.uint16, count);
      final inputBuffer = minigpu.createBuffer(bufSize);
      final outputBuffer = minigpu.createBuffer(bufSize);

      final Uint16List input = Uint16List.fromList(
          List.generate(count, (i) => i * 2, growable: false));
      inputBuffer.setData(input, input.length, dataType: BufferDataType.uint16);

      final String shaderCode = '''
@group(0) @binding(0) var<storage, read_write> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx < ${bufSize ~/ Uint32List.bytesPerElement}u) {
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
