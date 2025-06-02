import 'dart:typed_data';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

// Removed getBufferSizeForType helper

void main() {
  group('Buffer Data Types Read/Write Tests', () {
    late Minigpu minigpu;

    setUpAll(() async {
      minigpu = Minigpu();
      await minigpu.init();
    });
    tearDownAll(() async {
      await minigpu.destroy();
    });

    test('Float32 read/write', () async {
      final int count = 16;
      final int byteSize = count * Float32List.bytesPerElement; // Physical size
      final buffer = minigpu.createBuffer(
          byteSize, BufferDataType.float32); // Pass physical size

      final Float32List input = Float32List.fromList(
        List.generate(count, (i) => i.toDouble() * 1.1),
      );
      buffer.setData(input, count,
          dataType: BufferDataType.float32); // Pass element count

      final Float32List output = Float32List(count);
      await buffer.read(output, count,
          dataType: BufferDataType.float32); // Pass element count

      expect(output, equals(input));
      buffer.destroy();
    });

    test('Int8 read/write', () async {
      final int count = 17;
      // For i8, createBuffer expects logical element count
      final buffer = minigpu.createBuffer(
          count, BufferDataType.int8); // Pass logical count

      final Int8List input = Int8List.fromList(
        List.generate(count, (i) => (i % 256) - 128),
      );
      buffer.setData(input, count,
          dataType: BufferDataType.int8); // Pass element count

      final Int8List output = Int8List(count);
      await buffer.read(output, count,
          dataType: BufferDataType.int8); // Pass element count

      expect(output, equals(input));
      buffer.destroy();
    });

    test('Int16 read/write', () async {
      final int count = 17;
      // For i16, createBuffer expects logical element count
      final buffer = minigpu.createBuffer(
          count, BufferDataType.int16); // Pass logical count

      final Int16List input = Int16List.fromList(
        List.generate(count, (i) => (i * 100) - 15000),
      );
      buffer.setData(input, count,
          dataType: BufferDataType.int16); // Pass element count

      final Int16List output = Int16List(count);
      await buffer.read(output, count,
          dataType: BufferDataType.int16); // Pass element count

      expect(output, equals(input));
      buffer.destroy();
    });

    test('Int32 read/write', () async {
      final int count = 16;
      final int byteSize = count * Int32List.bytesPerElement; // Physical size
      final buffer = minigpu.createBuffer(
          byteSize, BufferDataType.int32); // Pass physical size

      final Int32List input = Int32List.fromList(
        List.generate(count, (i) => (i * 100000) - 500000),
      );
      buffer.setData(input, count,
          dataType: BufferDataType.int32); // Pass element count

      final Int32List output = Int32List(count);
      await buffer.read(output, count,
          dataType: BufferDataType.int32); // Pass element count

      expect(output, equals(input));
      buffer.destroy();
    });

    test('Int64 read/write', () async {
      final int count = 16;
      final int byteSize = count * Int64List.bytesPerElement; // Physical size
      final buffer = minigpu.createBuffer(
          byteSize, BufferDataType.int64); // Pass physical size

      final Int64List input = Int64List.fromList(
        List.generate(count, (i) => (i - 8) * 0x100000000 + i),
      );
      buffer.setData(input, count,
          dataType: BufferDataType.int64); // Pass element count

      final Int64List output = Int64List(count);
      await buffer.read(output, count,
          dataType: BufferDataType.int64); // Pass element count

      expect(output, equals(input));
      buffer.destroy();
    });

    test('Uint8 read/write', () async {
      final int count = 17;
      // For u8, createBuffer expects logical element count
      final buffer = minigpu.createBuffer(
          count, BufferDataType.uint8); // Pass logical count

      final Uint8List input = Uint8List.fromList(
        List.generate(count, (i) => i % 256),
      );
      buffer.setData(input, count,
          dataType: BufferDataType.uint8); // Pass element count

      final Uint8List output = Uint8List(count);
      await buffer.read(output, count,
          dataType: BufferDataType.uint8); // Pass element count

      expect(output, equals(input));
      buffer.destroy();
    });

    test('Uint16 read/write', () async {
      final int count = 17;
      // For u16, createBuffer expects logical element count
      final buffer = minigpu.createBuffer(
          count, BufferDataType.uint16); // Pass logical count

      final Uint16List input = Uint16List.fromList(
        List.generate(count, (i) => (i * 100) % 65536),
      );
      buffer.setData(input, count,
          dataType: BufferDataType.uint16); // Pass element count

      final Uint16List output = Uint16List(count);
      await buffer.read(output, count,
          dataType: BufferDataType.uint16); // Pass element count

      expect(output, equals(input));
      buffer.destroy();
    });

    test('Uint32 read/write', () async {
      final int count = 16;
      final int byteSize = count * Uint32List.bytesPerElement; // Physical size
      final buffer = minigpu.createBuffer(
          byteSize, BufferDataType.uint32); // Pass physical size

      final Uint32List input = Uint32List.fromList(
        List.generate(count, (i) => (i + 1) * 100000),
      );
      buffer.setData(input, count,
          dataType: BufferDataType.uint32); // Pass element count

      final Uint32List output = Uint32List(count);
      await buffer.read(output, count,
          dataType: BufferDataType.uint32); // Pass element count

      expect(output, equals(input));
      buffer.destroy();
    });

    test('Uint64 read/write', () async {
      final int count = 16;
      final int byteSize = count * Uint64List.bytesPerElement; // Physical size
      final buffer = minigpu.createBuffer(
          byteSize, BufferDataType.uint64); // Pass physical size

      final Uint64List input = Uint64List.fromList(
        List.generate(count, (i) => i * 0x100000000 + i),
      );
      buffer.setData(input, count,
          dataType: BufferDataType.uint64); // Pass element count

      final Uint64List output = Uint64List(count);
      await buffer.read(output, count,
          dataType: BufferDataType.uint64); // Pass element count

      expect(output, equals(input));
      buffer.destroy();
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
      final int byteSize = count * Float32List.bytesPerElement; // Physical size
      final inputBuffer = minigpu.createBuffer(
          byteSize, BufferDataType.float32); // Pass physical size
      final outputBuffer = minigpu.createBuffer(
          byteSize, BufferDataType.float32); // Pass physical size

      final Float32List input = Float32List.fromList(
          List.generate(count, (i) => i.toDouble() * 1.1, growable: false));
      inputBuffer.setData(input, count, dataType: BufferDataType.float32);

      final String shaderCode = '''
        @group(0) @binding(0) var<storage, read_write> inp: array<f32>;
        @group(0) @binding(1) var<storage, read_write> out: array<f32>;

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

      // Dispatch based on logical element count for direct types
      final int workgroups = ((count + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Float32List output = Float32List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.float32);
      expect(output, equals(input));

      inputBuffer.destroy();
      outputBuffer.destroy();
      shader.destroy();
    });

    test('Float64 Roundtrip', () async {
      final int count = 16;
      // For f64, createBuffer expects logical element count (C++ handles expansion)
      final inputBuffer = minigpu.createBuffer(count, BufferDataType.float64);
      final outputBuffer = minigpu.createBuffer(count, BufferDataType.float64);

      final Float64List input = Float64List.fromList(List.generate(
          count, (i) => i.toDouble() * 1.23456789e10,
          growable: false));
      inputBuffer.setData(input, count, dataType: BufferDataType.float64);

      // Shader operates on the internal u32 pairs representation
      final int internalElementCount = count * 2; // u32 count
      final String shaderCode = '''
        @group(0) @binding(0) var<storage, read_write> inp: array<u32>; // Matches C++ internal type for f64
        @group(0) @binding(1) var<storage, read_write> out: array<u32>; // Matches C++ internal type for f64

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x; // Index into the u32 array
          if (idx < arrayLength(&inp)) {
            out[idx] = inp[idx]; // Copy u32 element
          }
        }
      ''';
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);

      // Dispatch based on internal u32 element count
      final int workgroups = ((internalElementCount + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Float64List output = Float64List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.float64);
      expect(output, equals(input)); // C++ read handles re-combining u32 pairs

      inputBuffer.destroy();
      outputBuffer.destroy();
      shader.destroy();
    });

    test('Int8 Roundtrip', () async {
      final int count = 7;
      // For i8, createBuffer expects logical element count
      final inputBuffer = minigpu.createBuffer(
          count, BufferDataType.int8); // Pass logical count
      final outputBuffer = minigpu.createBuffer(
          count, BufferDataType.int8); // Pass logical count

      final Int8List input = Int8List.fromList(
          List.generate(count, (i) => (i % 256) - 128, growable: false));
      inputBuffer.setData(input, count, dataType: BufferDataType.int8);

      // Shader operates on internal i32 representation
      final int internalElementCount =
          count; // i32 count matches logical i8 count
      final String shaderCode = '''
        @group(0) @binding(0) var<storage, read_write> inp: array<i32>;
        @group(0) @binding(1) var<storage, read_write> out: array<i32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x; // Index into the i32 array
          if (idx < arrayLength(&inp)) {
            out[idx] = inp[idx];
          }
        }
      ''';
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);

      // Dispatch based on internal i32 element count
      final int workgroups = ((internalElementCount + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Int8List output = Int8List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.int8);
      expect(output, equals(input));

      inputBuffer.destroy();
      outputBuffer.destroy();
      shader.destroy();
    });

    test('Int16 Roundtrip', () async {
      final int count = 17;
      // For i16, createBuffer expects logical element count
      final inputBuffer = minigpu.createBuffer(
          count, BufferDataType.int16); // Pass logical count
      final outputBuffer = minigpu.createBuffer(
          count, BufferDataType.int16); // Pass logical count

      final Int16List input = Int16List.fromList(
          List.generate(count, (i) => (i * 100) - 15000, growable: false));
      inputBuffer.setData(input, count, dataType: BufferDataType.int16);

      final String shaderCode = '''
        @group(0) @binding(0) var<storage, read_write> inp: array<i32>;
        @group(0) @binding(1) var<storage, read_write> out: array<i32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x; // Index into the i32 array
          if (idx < arrayLength(&inp)) {
            out[idx] = inp[idx];
          }
        }
      ''';
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);

      // Dispatch based on internal i32 element count
      final int workgroups = ((count + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Int16List output = Int16List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.int16);
      expect(output, equals(input));

      inputBuffer.destroy();
      outputBuffer.destroy();
      shader.destroy();
    });

    test('Int32 Roundtrip', () async {
      final int count = 16;
      final int byteSize = count * Int32List.bytesPerElement; // Physical size
      final inputBuffer = minigpu.createBuffer(
          byteSize, BufferDataType.int32); // Pass physical size
      final outputBuffer = minigpu.createBuffer(
          byteSize, BufferDataType.int32); // Pass physical size

      final Int32List input = Int32List.fromList(
          List.generate(count, (i) => (i * 100000) - 500000, growable: false));
      inputBuffer.setData(input, count, dataType: BufferDataType.int32);

      // Shader operates directly on i32
      final String shaderCode = '''
        @group(0) @binding(0) var<storage, read_write> inp: array<i32>;
        @group(0) @binding(1) var<storage, read_write> out: array<i32>;

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

      // Dispatch based on logical element count
      final int workgroups = ((count + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Int32List output = Int32List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.int32);
      expect(output, equals(input));

      inputBuffer.destroy();
      outputBuffer.destroy();
      shader.destroy();
    });

    test('Int64 Roundtrip', () async {
      final int count = 16;
      final int byteSize = count * Int64List.bytesPerElement; // Physical size
      final inputBuffer = minigpu.createBuffer(
          byteSize, BufferDataType.int64); // Pass physical size
      final outputBuffer = minigpu.createBuffer(
          byteSize, BufferDataType.int64); // Pass physical size

      final Int64List input = Int64List.fromList(List.generate(
          count, (i) => (i - 8) * 0x100000000 + i,
          growable: false));
      inputBuffer.setData(input, count, dataType: BufferDataType.int64);

      final String shaderCode = '''
        @group(0) @binding(0) var<storage, read_write> inp: array<i32>;
        @group(0) @binding(1) var<storage, read_write> out: array<i32>;

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

      // Dispatch based on logical i64 element count
      final int workgroups = ((count + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Int64List output = Int64List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.int64);
      expect(output, equals(input));

      inputBuffer.destroy();
      outputBuffer.destroy();
      shader.destroy();
    });

    test('Uint8 Roundtrip', () async {
      final int count = 17;
      // For u8, createBuffer expects logical element count
      final inputBuffer = minigpu.createBuffer(
          count, BufferDataType.uint8); // Pass logical count
      final outputBuffer = minigpu.createBuffer(
          count, BufferDataType.uint8); // Pass logical count

      final Uint8List input = Uint8List.fromList(
          List.generate(count, (i) => i % 256, growable: false));
      inputBuffer.setData(input, count, dataType: BufferDataType.uint8);

      // Shader operates on internal u32 representation
      final int internalElementCount =
          count; // u32 count matches logical u8 count
      final String shaderCode = '''
        @group(0) @binding(0) var<storage, read_write> inp: array<u32>;
        @group(0) @binding(1) var<storage, read_write> out: array<u32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x; // Index into the u32 array
          if (idx < arrayLength(&inp)) {
            out[idx] = inp[idx];
          }
        }
      ''';
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);

      // Dispatch based on internal u32 element count
      final int workgroups = ((internalElementCount + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Uint8List output = Uint8List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.uint8);
      expect(output, equals(input));

      inputBuffer.destroy();
      outputBuffer.destroy();
      shader.destroy();
    });

    test('Uint16 Roundtrip', () async {
      final int count = 17;
      // For u16, createBuffer expects logical element count
      final inputBuffer = minigpu.createBuffer(
          count, BufferDataType.uint16); // Pass logical count
      final outputBuffer = minigpu.createBuffer(
          count, BufferDataType.uint16); // Pass logical count

      final Uint16List input = Uint16List.fromList(
          List.generate(count, (i) => (i * 100) % 65536, growable: false));
      inputBuffer.setData(input, count, dataType: BufferDataType.uint16);

      // Shader operates on internal u32 representation
      final int internalElementCount =
          count; // u32 count matches logical u16 count
      final String shaderCode = '''
        @group(0) @binding(0) var<storage, read_write> inp: array<u32>;
        @group(0) @binding(1) var<storage, read_write> out: array<u32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x; // Index into the u32 array
          let internal_length = ${internalElementCount}u; // Use internal count
          if (idx < internal_length) {
            out[idx] = inp[idx];
          }
        }
      ''';
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);

      // Dispatch based on internal u32 element count
      final int workgroups = ((internalElementCount + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Uint16List output = Uint16List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.uint16);
      expect(output, equals(input));

      inputBuffer.destroy();
      outputBuffer.destroy();
      shader.destroy();
    });

    test('Uint32 Roundtrip', () async {
      final int count = 16;
      final int byteSize = count * Uint32List.bytesPerElement; // Physical size
      final inputBuffer = minigpu.createBuffer(
          byteSize, BufferDataType.uint32); // Pass physical size
      final outputBuffer = minigpu.createBuffer(
          byteSize, BufferDataType.uint32); // Pass physical size

      final Uint32List input = Uint32List.fromList(
          List.generate(count, (i) => (i + 1) * 100000, growable: false));
      inputBuffer.setData(input, count, dataType: BufferDataType.uint32);

      // Shader operates directly on u32
      final String shaderCode = '''
        @group(0) @binding(0) var<storage, read_write> inp: array<u32>;
        @group(0) @binding(1) var<storage, read_write> out: array<u32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let idx = global_id.x;
          let logical_length = ${count}u; // Use logical count
          if (idx < logical_length) {
            out[idx] = inp[idx];
          }
        }
      ''';
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);

      // Dispatch based on logical element count
      final int workgroups = ((count + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Uint32List output = Uint32List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.uint32);
      expect(output, equals(input));

      inputBuffer.destroy();
      outputBuffer.destroy();
      shader.destroy();
    });

    test('Uint64 Roundtrip', () async {
      final int count = 16;
      final int byteSize = count * Uint64List.bytesPerElement; // Physical size
      final inputBuffer = minigpu.createBuffer(
          byteSize, BufferDataType.uint64); // Pass physical size
      final outputBuffer = minigpu.createBuffer(
          byteSize, BufferDataType.uint64); // Pass physical size

      final Uint64List input = Uint64List.fromList(
          List.generate(count, (i) => i * 0x100000000 + i, growable: false));
      inputBuffer.setData(input, count, dataType: BufferDataType.uint64);

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

      // Dispatch based on logical u64 element count
      final int workgroups = ((count + 255) ~/ 256);
      await shader.dispatch(workgroups, 1, 1);

      final Uint64List output = Uint64List(count);
      await outputBuffer.read(output, count, dataType: BufferDataType.uint64);
      expect(output, equals(input));

      inputBuffer.destroy();
      outputBuffer.destroy();
      shader.destroy();
    });
  });
}
