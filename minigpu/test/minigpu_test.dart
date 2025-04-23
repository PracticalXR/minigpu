import 'dart:typed_data';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

void main() {
  group('Minigpu Library Tests', () {
    late Minigpu minigpu;

    setUp(() async {
      // Each test gets a new instance.
      minigpu = Minigpu();
      await minigpu.init();
    });

    test('Context Initialization', () {
      // isInitialized should be true after init.
      expect(minigpu.isInitialized, isTrue);
    });

    test('Buffer creation and data transfer', () async {
      const int bufferSize = 100;
      final int memorySize = bufferSize * 4; // 4 bytes per float

      // Create a buffer and set known data.
      final buffer = minigpu.createBuffer(memorySize, BufferDataType.float32);
      final inputData =
          Float32List.fromList(List.generate(bufferSize, (i) => i.toDouble()));
      buffer.setData(inputData, bufferSize);

      // Read back the data.
      final outputData = Float32List(bufferSize);
      await buffer.read(outputData, bufferSize);

      // Check that the buffer returns the original data.
      expect(outputData, equals(inputData));

      // Clean up.
      buffer.destroy();
    });

    test('Compute Shader: adds 0.2 to each element', () async {
      const int numFloats = 100;
      final int memorySize = numFloats * 4;

      // Create input and output buffers.
      final inputBuffer =
          minigpu.createBuffer(memorySize, BufferDataType.float32);
      final outputBuffer =
          minigpu.createBuffer(memorySize, BufferDataType.float32);

      // Initialize input data.
      final inputData =
          Float32List.fromList(List.generate(numFloats, (i) => i.toDouble()));
      inputBuffer.setData(inputData, numFloats);

      // WGSL shader code which adds 0.2 to each input element.
      final shaderTemplate = '''
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654;
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < ${numFloats}u) {
        let x: f32 = inp[i];
        out[i] = x + 0.2;
    }
}
''';

      // Create and configure the compute shader.
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);

      // Calculate workgroups (assuming 256 threads per group).
      final int workgroups = ((numFloats + 255) / 256).floor();
      await shader.dispatch(workgroups, 1, 1);

      // Read output data.
      final outputData = Float32List(numFloats);
      await outputBuffer.read(outputData, numFloats);

      // Verify that each element was increased by 0.2.
      for (int i = 0; i < numFloats; i++) {
        expect(outputData[i], closeTo(inputData[i] + 0.2, 1e-4));
      }

      // Clean up.
      shader.destroy();
      inputBuffer.destroy();
      outputBuffer.destroy();
    });
  });
}
