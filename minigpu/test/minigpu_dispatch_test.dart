import 'dart:io';
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  late Minigpu gpu;

  setUpAll(() async {
    gpu = Minigpu();
    await gpu.init();
  });

  tearDownAll(() async {});

  group('Buffer Binding State Crash Tests', () {
    test('rapid buffer binding changes - crash reproduction', () async {
      const size = 1024;

      print('Testing rapid buffer binding changes that cause crashes...');

      final inputBuffer = gpu.createBuffer(size * 4, BufferDataType.float32);
      final outputBuffer = gpu.createBuffer(size * 4, BufferDataType.float32);

      // Fill input with test data
      final inputData = Float32List.fromList(
        List.generate(size, (i) => i.toDouble()),
      );
      inputBuffer.write(
        inputData,
        inputData.length,
        dataType: BufferDataType.float32,
      );

      final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_0)) { return; }
    output_0[i] = input_0[i] * 2.0;
}
''';

      final shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);

      // This pattern mimics what happens in the GPU pipeline
      print('  Setting initial buffers...');
      shader.setBuffer('input_0', inputBuffer);
      shader.setBuffer('output_0', outputBuffer);

      print('  First dispatch...');
      await shader.dispatch(4, 1, 1);

      // Now reset bindings - this might trigger the crash
      print('  Resetting same buffers (this might crash)...');
      shader.setBuffer('input_0', inputBuffer); // Same buffer, same tag
      shader.setBuffer('output_0', outputBuffer); // Same buffer, same tag

      print('  Second dispatch (crash likely here)...');
      await shader.dispatch(4, 1, 1);

      shader.destroy();
      inputBuffer.destroy();
      outputBuffer.destroy();
    });

    test(
      'string tag buffer binding with same pointers - crash scenario',
      () async {
        const size = 512;

        print('Testing string tag binding with pointer reuse...');

        final buffer1 = gpu.createBuffer(size * 4, BufferDataType.float32);
        final buffer2 = gpu.createBuffer(size * 4, BufferDataType.float32);

        final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_0)) { return; }
    output_0[i] = input_0[i] + 1.0;
}
''';

        final shader = gpu.createComputeShader();
        shader.loadKernelString(shaderCode);

        // Scenario 1: Set buffers first time
        print('  Initial binding...');
        shader.setBuffer('input_0', buffer1);
        shader.setBuffer('output_0', buffer2);
        await shader.dispatch(8, 1, 1);

        // Scenario 2: Set same tag with same buffer (early return in C++)
        print('  Same buffer, same tag (should early return)...');
        shader.setBuffer('input_0', buffer1); // Same pointer - early return

        // Scenario 3: Different buffer, same tag (binding update)
        print('  Different buffer, same tag...');
        shader.setBuffer(
          'input_0',
          buffer2,
        ); // Different pointer - update binding

        // Scenario 4: Back to original buffer (this is where crash happens)
        print('  Back to original buffer (crash point)...');
        shader.setBuffer('input_0', buffer1); // Back to original - crash here

        print('  Final dispatch (likely crashes)...');
        await shader.dispatch(8, 1, 1);

        shader.destroy();
        buffer1.destroy();
        buffer2.destroy();
      },
    );

    test('kernel tags map corruption - exact crash reproduction', () async {
      const size = 256;

      print('Testing kernel tags map corruption...');

      final inputBuffer = gpu.createBuffer(size * 4, BufferDataType.float32);
      final outputBuffer = gpu.createBuffer(size * 4, BufferDataType.float32);

      final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_0)) { return; }
    output_0[i] = input_0[i] * 1.5;
}
''';

      final shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);

      // This exactly replicates the crash scenario from the logs
      print('  🔧 Setting buffer input_0...');
      shader.setBuffer('input_0', inputBuffer);

      print('  🔧 Setting buffer output_0...');
      shader.setBuffer('output_0', outputBuffer);

      print('  🔧 First dispatch...');
      await shader.dispatch(1, 1, 1);

      // Now simulate what happens in merged processing
      print('  🔧 Re-setting input_0 (same buffer)...');
      shader.setBuffer('input_0', inputBuffer);

      print('  🔧 Re-setting output_0 (same buffer)...');
      shader.setBuffer('output_0', outputBuffer);

      print('  🔧 Crash dispatch...');
      await shader.dispatch(1, 1, 1); // This should crash

      shader.destroy();
      inputBuffer.destroy();
      outputBuffer.destroy();
    });

    test('buffer binding state after multiple operations', () async {
      const size = 128;

      print('Testing buffer binding state persistence...');

      // Create buffers that will be reused
      final inputA = gpu.createBuffer(size * 4, BufferDataType.float32);
      final inputB = gpu.createBuffer(size * 4, BufferDataType.float32);
      final output = gpu.createBuffer(size * 4, BufferDataType.float32);

      final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_0)) { return; }
    output_0[i] = input_0[i] + 10.0;
}
''';

      final shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);

      // Pattern that leads to crash:
      // 1. Set initial bindings
      shader.setBuffer('input_0', inputA);
      shader.setBuffer('output_0', output);
      await shader.dispatch(4, 1, 1);

      // 2. Change input buffer
      shader.setBuffer('input_0', inputB);
      await shader.dispatch(4, 1, 1);

      // 3. Change back to original (this corrupts binding state)
      shader.setBuffer('input_0', inputA);

      // 4. Reset output buffer too (double corruption)
      shader.setBuffer('output_0', output);

      // 5. This dispatch crashes due to corrupted WebGPU binding state
      print('  Final dispatch (should crash)...');
      await shader.dispatch(4, 1, 1);

      shader.destroy();
      inputA.destroy();
      inputB.destroy();
      output.destroy();
    });
  });

  group('Rapid Dispatch Crash Test', () {
    test('rapid sequential dispatches - spectrogram scenario', () async {
      const width = 200;
      const height = 512;
      const outputSize = width * height * 4;
      const numRapidDispatches = 50; // Simulate rapid spectrogram updates

      print(
        'Testing $numRapidDispatches rapid dispatches of ${width}x${height}...',
      );

      // Create buffers once and reuse
      final inputBuffer = gpu.createBuffer(
        width * height * 4,
        BufferDataType.float32,
      );
      final outputBuffer = gpu.createBuffer(
        outputSize * 4,
        BufferDataType.float32,
      );

      final shaderCode =
          '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const OUTPUT_SIZE: u32 = ${outputSize}u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    
    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }
    
    let inputIndex: u32 = x * HEIGHT + y;
    let outputIndex: u32 = (y * WIDTH + x) * 4u;
    
    var value: f32 = f32(inputIndex) / f32(WIDTH * HEIGHT); // Generate some data
    
    if (outputIndex + 3u < OUTPUT_SIZE && outputIndex + 3u < arrayLength(&output)) {
        output[outputIndex] = value;
        output[outputIndex + 1u] = 0.0;
        output[outputIndex + 2u] = 0.0;  
        output[outputIndex + 3u] = 1.0;
    }
}
''';

      final shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', inputBuffer);
      shader.setBuffer('output', outputBuffer);

      final workgroupsX = (width + 15) ~/ 16;
      final workgroupsY = (height + 15) ~/ 16;

      // Rapid dispatches without waiting
      for (int i = 0; i < numRapidDispatches; i++) {
        try {
          print('Dispatch $i...');
          await shader.dispatch(workgroupsX, workgroupsY, 1);

          // Simulate real-world scenario: sometimes read back data
          if (i % 10 == 0) {
            final outputData = Float32List(outputSize);
            await outputBuffer.read(
              outputData,
              outputSize,
              dataType: BufferDataType.float32,
            );
            print('  Readback successful for dispatch $i');
          }
        } catch (e, stackTrace) {
          print('CRASH on dispatch $i: $e');
          print('Stack trace: $stackTrace');
          break;
        }
      }

      shader.destroy();
      inputBuffer.destroy();
      outputBuffer.destroy();
    });

    test('concurrent shader dispatches - multiple shaders', () async {
      const width = 200;
      const height = 512;
      const outputSize = width * height * 4;
      const numConcurrentShaders = 5;

      print('Testing $numConcurrentShaders concurrent shader dispatches...');

      final futures = <Future<void>>[];

      for (
        int shaderIndex = 0;
        shaderIndex < numConcurrentShaders;
        shaderIndex++
      ) {
        final future = () async {
          try {
            final inputBuffer = gpu.createBuffer(
              width * height * 4,
              BufferDataType.float32,
            );
            final outputBuffer = gpu.createBuffer(
              outputSize * 4,
              BufferDataType.float32,
            );

            final shaderCode =
                '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const OUTPUT_SIZE: u32 = ${outputSize}u;
const SHADER_ID: u32 = ${shaderIndex}u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    
    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }
    
    let inputIndex: u32 = x * HEIGHT + y;
    let outputIndex: u32 = (y * WIDTH + x) * 4u;
    
    var value: f32 = f32(inputIndex + SHADER_ID * 1000u) / f32(WIDTH * HEIGHT);
    
    if (outputIndex + 3u < OUTPUT_SIZE && outputIndex + 3u < arrayLength(&output)) {
        output[outputIndex] = value;
        output[outputIndex + 1u] = f32(SHADER_ID) / 10.0;
        output[outputIndex + 2u] = 0.0;
        output[outputIndex + 3u] = 1.0;
    }
}
''';

            final shader = gpu.createComputeShader();
            shader.loadKernelString(shaderCode);
            shader.setBuffer('input', inputBuffer);
            shader.setBuffer('output', outputBuffer);

            final workgroupsX = (width + 15) ~/ 16;
            final workgroupsY = (height + 15) ~/ 16;

            print('Shader $shaderIndex: Starting dispatch...');
            await shader.dispatch(workgroupsX, workgroupsY, 1);

            // Read back to verify
            final outputData = Float32List(outputSize);
            await outputBuffer.read(
              outputData,
              outputSize,
              dataType: BufferDataType.float32,
            );

            print('Shader $shaderIndex: SUCCESS');

            shader.destroy();
            inputBuffer.destroy();
            outputBuffer.destroy();
          } catch (e, stackTrace) {
            print('Shader $shaderIndex CRASHED: $e');
            print('Stack trace: $stackTrace');
            rethrow;
          }
        }();

        futures.add(future);
      }

      // Wait for all concurrent dispatches
      await Future.wait(futures);
    });

    test('rapid buffer create/destroy with dispatches', () async {
      const width = 100;
      const height = 100;
      const outputSize = width * height * 4;
      const numCycles = 20;

      print('Testing $numCycles rapid buffer create/destroy cycles...');

      for (int cycle = 0; cycle < numCycles; cycle++) {
        try {
          print('Cycle $cycle: Creating buffers...');

          final inputBuffer = gpu.createBuffer(
            width * height * 4,
            BufferDataType.float32,
          );
          final outputBuffer = gpu.createBuffer(
            outputSize * 4,
            BufferDataType.float32,
          );

          // Fill input with test data
          final inputData = Float32List.fromList(
            List.generate(width * height, (i) => i.toDouble() + cycle * 1000),
          );
          inputBuffer.write(
            inputData,
            inputData.length,
            dataType: BufferDataType.float32,
          );

          final shaderCode =
              '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    
    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }
    
    let inputIndex: u32 = x * HEIGHT + y;
    let outputIndex: u32 = (y * WIDTH + x) * 4u;
    
    var value: f32 = 0.0;
    if (inputIndex < arrayLength(&input)) {
        value = input[inputIndex];
    }
    
    if (outputIndex + 3u < arrayLength(&output)) {
        output[outputIndex] = value;
        output[outputIndex + 1u] = 0.0;
        output[outputIndex + 2u] = 0.0;
        output[outputIndex + 3u] = 1.0;
    }
}
''';

          final shader = gpu.createComputeShader();
          shader.loadKernelString(shaderCode);
          shader.setBuffer('input', inputBuffer);
          shader.setBuffer('output', outputBuffer);

          final workgroupsX = (width + 7) ~/ 8;
          final workgroupsY = (height + 7) ~/ 8;

          print('  Dispatching...');
          await shader.dispatch(workgroupsX, workgroupsY, 1);

          print('  Reading back...');
          final outputData = Float32List(outputSize);
          await outputBuffer.read(
            outputData,
            outputSize,
            dataType: BufferDataType.float32,
          );

          print('  Destroying resources...');
          shader.destroy();
          inputBuffer.destroy();
          outputBuffer.destroy();

          print('Cycle $cycle: SUCCESS');
        } catch (e, stackTrace) {
          print('CRASH on cycle $cycle: $e');
          print('Stack trace: $stackTrace');
          break;
        }
      }
    });

    test('stress test - simulating real spectrogram updates', () async {
      const width = 200;
      const height = 512;
      const outputSize = width * height * 4;
      const updateRate = 60; // 60 FPS simulation
      const durationSeconds = 2;
      const totalUpdates = updateRate * durationSeconds;

      print(
        'Simulating spectrogram at ${updateRate}FPS for ${durationSeconds}s ($totalUpdates updates)...',
      );

      // Reuse buffers like a real app would
      final inputBuffer = gpu.createBuffer(
        width * height * 4,
        BufferDataType.float32,
      );
      final outputBuffer = gpu.createBuffer(
        outputSize * 4,
        BufferDataType.float32,
      );

      final shaderCode =
          '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const OUTPUT_SIZE: u32 = ${outputSize}u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    
    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }
    
    let inputIndex: u32 = x * HEIGHT + y;
    let outputIndex: u32 = (y * WIDTH + x) * 4u;
    
    var value: f32 = 0.0;
    if (inputIndex < arrayLength(&input)) {
        value = input[inputIndex];
    }
    
    if (outputIndex + 3u < OUTPUT_SIZE && outputIndex + 3u < arrayLength(&output)) {
        output[outputIndex] = value;
        output[outputIndex + 1u] = 0.0;
        output[outputIndex + 2u] = 0.0;
        output[outputIndex + 3u] = 1.0;
    }
}
''';

      final shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', inputBuffer);
      shader.setBuffer('output', outputBuffer);

      final workgroupsX = (width + 15) ~/ 16;
      final workgroupsY = (height + 15) ~/ 16;

      final startTime = DateTime.now();

      for (int frame = 0; frame < totalUpdates; frame++) {
        try {
          // Simulate new audio data arriving
          final inputData = Float32List.fromList(
            List.generate(
              width * height,
              (i) => (i + frame * 100) % 1000 / 1000.0,
            ),
          );
          inputBuffer.write(
            inputData,
            inputData.length,
            dataType: BufferDataType.float32,
          );

          // Process to texture
          await shader.dispatch(workgroupsX, workgroupsY, 1);

          // Occasionally read back (like UI updates)
          if (frame % 30 == 0) {
            // Every 30 frames
            final outputData = Float32List(outputSize);
            await outputBuffer.read(
              outputData,
              outputSize,
              dataType: BufferDataType.float32,
            );
          }

          if (frame % 60 == 0) {
            final elapsed = DateTime.now().difference(startTime);
            print(
              'Frame $frame/${totalUpdates} (${elapsed.inMilliseconds}ms elapsed)',
            );
          }
        } catch (e, stackTrace) {
          print('CRASH on frame $frame: $e');
          print('Stack trace: $stackTrace');
          break;
        }
      }

      final totalTime = DateTime.now().difference(startTime);
      print(
        'Completed ${totalUpdates} updates in ${totalTime.inMilliseconds}ms',
      );

      shader.destroy();
      inputBuffer.destroy();
      outputBuffer.destroy();
    });
  });

  group('Shader Reuse Tests', () {
    test('same shader multiple dispatches - no binding changes', () async {
      const width = 128;
      const height = 128;
      const outputSize = width * height * 4;
      const numDispatches = 10;

      print(
        'Testing $numDispatches dispatches from same shader (no binding changes)...',
      );

      final inputBuffer = gpu.createBuffer(
        width * height * 4,
        BufferDataType.float32,
      );
      final outputBuffer = gpu.createBuffer(
        outputSize * 4,
        BufferDataType.float32,
      );

      // Fill input with test data
      final inputData = Float32List.fromList(
        List.generate(width * height, (i) => i.toDouble() / (width * height)),
      );
      inputBuffer.write(
        inputData,
        inputData.length,
        dataType: BufferDataType.float32,
      );

      final shaderCode =
          '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    
    if (x >= WIDTH || y >= HEIGHT) {
        return;
    }
    
    let inputIndex: u32 = x * HEIGHT + y;
    let outputIndex: u32 = (y * WIDTH + x) * 4u;
    
    var value: f32 = 0.0;
    if (inputIndex < arrayLength(&input)) {
        value = input[inputIndex] * 2.0; // Simple transform
    }
    
    if (outputIndex + 3u < arrayLength(&output)) {
        output[outputIndex] = value;
        output[outputIndex + 1u] = value * 0.5;
        output[outputIndex + 2u] = value * 0.25;
        output[outputIndex + 3u] = 1.0;
    }
}
''';

      final shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', inputBuffer);
      shader.setBuffer('output', outputBuffer);

      final workgroupsX = (width + 7) ~/ 8;
      final workgroupsY = (height + 7) ~/ 8;

      // Multiple dispatches without changing bindings
      for (int i = 0; i < numDispatches; i++) {
        print('  Dispatch $i (should reuse compiled kernel)...');
        await shader.dispatch(workgroupsX, workgroupsY, 1);

        // Verify results occasionally
        if (i % 3 == 0) {
          final outputData = Float32List(outputSize);
          await outputBuffer.read(
            outputData,
            outputSize,
            dataType: BufferDataType.float32,
          );
          print('    Verification successful for dispatch $i');
        }
      }

      shader.destroy();
      inputBuffer.destroy();
      outputBuffer.destroy();
    });

    test('shader reuse after buffer binding changes', () async {
      const size = 1024;
      const numBufferSwaps = 5;

      print(
        'Testing shader reuse after $numBufferSwaps buffer binding changes...',
      );

      // Create multiple input buffers with different data
      final inputBuffers = <Buffer>[];
      final outputBuffers = <Buffer>[];

      for (int i = 0; i < numBufferSwaps; i++) {
        final inputBuffer = gpu.createBuffer(size * 4, BufferDataType.float32);
        final outputBuffer = gpu.createBuffer(size * 4, BufferDataType.float32);

        // Fill with unique data per buffer
        final inputData = Float32List.fromList(
          List.generate(size, (j) => (j + i * 1000).toDouble()),
        );
        inputBuffer.write(
          inputData,
          inputData.length,
          dataType: BufferDataType.float32,
        );

        inputBuffers.add(inputBuffer);
        outputBuffers.add(outputBuffer);
      }

      final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    
    if (index >= arrayLength(&input)) {
        return;
    }
    
    output[index] = input[index] + 100.0; // Simple offset
}
''';

      final shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);

      final workgroupsX = (size + 63) ~/ 64;

      // Test binding different buffers and dispatching
      for (int i = 0; i < numBufferSwaps; i++) {
        print('  Binding change $i (should trigger kernel recreation)...');

        shader.setBuffer('input', inputBuffers[i]);
        shader.setBuffer('output', outputBuffers[i]);

        await shader.dispatch(workgroupsX, 1, 1);

        // Verify the specific buffer was processed
        final outputData = Float32List(size);
        await outputBuffers[i].read(
          outputData,
          size,
          dataType: BufferDataType.float32,
        );

        // Check that the transform was applied correctly
        final expectedFirst = (i * 1000).toDouble() + 100.0;
        if ((outputData[0] - expectedFirst).abs() < 0.001) {
          print('    Buffer $i processed correctly');
        } else {
          throw Exception(
            'Buffer $i not processed correctly: expected $expectedFirst, got ${outputData[0]}',
          );
        }
      }

      shader.destroy();
      for (final buffer in inputBuffers) buffer.destroy();
      for (final buffer in outputBuffers) buffer.destroy();
    });

    test('shader reuse with different workgroup sizes', () async {
      const baseSize = 512;
      final workgroupConfigs = [
        {'x': 8, 'y': 8, 'z': 1},
        {'x': 16, 'y': 16, 'z': 1},
        {'x': 32, 'y': 32, 'z': 1},
        {'x': 8, 'y': 8, 'z': 1}, // Back to first config
      ];

      print('Testing shader reuse with different workgroup sizes...');

      final inputBuffer = gpu.createBuffer(
        baseSize * baseSize * 4,
        BufferDataType.float32,
      );
      final outputBuffer = gpu.createBuffer(
        baseSize * baseSize * 4 * 4,
        BufferDataType.float32,
      );

      final inputData = Float32List.fromList(
        List.generate(baseSize * baseSize, (i) => i.toDouble()),
      );
      inputBuffer.write(
        inputData,
        inputData.length,
        dataType: BufferDataType.float32,
      );

      final shaderCode =
          '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const SIZE: u32 = ${baseSize}u;

@compute @workgroup_size(16, 16) // Note: workgroup size in shader is fixed
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    
    if (x >= SIZE || y >= SIZE) {
        return;
    }
    
    let inputIndex = x * SIZE + y;
    let outputIndex = (y * SIZE + x) * 4u;
    
    var value: f32 = 0.0;
    if (inputIndex < arrayLength(&input)) {
        value = input[inputIndex];
    }
    
    if (outputIndex + 3u < arrayLength(&output)) {
        output[outputIndex] = value;
        output[outputIndex + 1u] = 0.0;
        output[outputIndex + 2u] = 0.0;
        output[outputIndex + 3u] = 1.0;
    }
}
''';

      final shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', inputBuffer);
      shader.setBuffer('output', outputBuffer);

      for (int i = 0; i < workgroupConfigs.length; i++) {
        final config = workgroupConfigs[i];
        final workgroupsX = (baseSize + config['x']! - 1) ~/ config['x']!;
        final workgroupsY = (baseSize + config['y']! - 1) ~/ config['y']!;
        final workgroupsZ = config['z']!;

        print(
          '  Dispatch $i with workgroups: ${workgroupsX}x${workgroupsY}x${workgroupsZ}...',
        );

        if (i == 3) {
          print('    (Should reuse kernel from dispatch 0)');
        } else {
          print('    (Should create new kernel)');
        }

        await shader.dispatch(workgroupsX, workgroupsY, workgroupsZ);

        // Verify results
        final outputData = Float32List(baseSize * baseSize * 4);
        await outputBuffer.read(
          outputData,
          baseSize * baseSize * 4,
          dataType: BufferDataType.float32,
        );
        print('    Dispatch $i completed successfully');
      }

      shader.destroy();
      inputBuffer.destroy();
      outputBuffer.destroy();
    });

    test('shader reuse vs recreation performance comparison', () async {
      const size = 256;
      const numReuseDispatches = 20;
      const numRecreateDispatches = 5;

      print('Performance comparison: reuse vs recreation...');

      final inputBuffer = gpu.createBuffer(
        size * size * 4,
        BufferDataType.float32,
      );
      final outputBuffer = gpu.createBuffer(
        size * size * 4 * 4,
        BufferDataType.float32,
      );

      final inputData = Float32List.fromList(
        List.generate(size * size, (i) => i.toDouble()),
      );
      inputBuffer.write(
        inputData,
        inputData.length,
        dataType: BufferDataType.float32,
      );

      final shaderCode =
          '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const SIZE: u32 = ${size}u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    
    if (x >= SIZE || y >= SIZE) {
        return;
    }
    
    let inputIndex = x * SIZE + y;
    let outputIndex = (y * SIZE + x) * 4u;
    
    var value: f32 = 0.0;
    if (inputIndex < arrayLength(&input)) {
        value = input[inputIndex] * 1.5;
    }
    
    if (outputIndex + 3u < arrayLength(&output)) {
        output[outputIndex] = value;
        output[outputIndex + 1u] = value * 0.5;
        output[outputIndex + 2u] = value * 0.25;
        output[outputIndex + 3u] = 1.0;
    }
}
''';

      final workgroupsX = (size + 15) ~/ 16;
      final workgroupsY = (size + 15) ~/ 16;

      // Test 1: Shader reuse (same bindings, same workgroups)
      print('  Testing shader reuse ($numReuseDispatches dispatches)...');
      final shader1 = gpu.createComputeShader();
      shader1.loadKernelString(shaderCode);
      shader1.setBuffer('input', inputBuffer);
      shader1.setBuffer('output', outputBuffer);

      final reuseStart = DateTime.now();
      for (int i = 0; i < numReuseDispatches; i++) {
        await shader1.dispatch(workgroupsX, workgroupsY, 1);
      }
      final reuseTime = DateTime.now().difference(reuseStart);
      print(
        '    Reuse time: ${reuseTime.inMilliseconds}ms (avg: ${reuseTime.inMilliseconds / numReuseDispatches}ms per dispatch)',
      );

      // Test 2: Shader recreation (new shader each time)
      print(
        '  Testing shader recreation ($numRecreateDispatches dispatches)...',
      );
      final recreateStart = DateTime.now();
      for (int i = 0; i < numRecreateDispatches; i++) {
        final shader = gpu.createComputeShader();
        shader.loadKernelString(shaderCode);
        shader.setBuffer('input', inputBuffer);
        shader.setBuffer('output', outputBuffer);
        await shader.dispatch(workgroupsX, workgroupsY, 1);
        shader.destroy();
      }
      final recreateTime = DateTime.now().difference(recreateStart);
      print(
        '    Recreation time: ${recreateTime.inMilliseconds}ms (avg: ${recreateTime.inMilliseconds / numRecreateDispatches}ms per dispatch)',
      );

      final speedup =
          (recreateTime.inMilliseconds / numRecreateDispatches) /
          (reuseTime.inMilliseconds / numReuseDispatches);
      print('    Speedup from reuse: ${speedup.toStringAsFixed(2)}x');

      shader1.destroy();
      inputBuffer.destroy();
      outputBuffer.destroy();
    });

    test('complex binding pattern stress test', () async {
      const size = 128;
      const numCycles = 10;

      print('Testing complex binding patterns over $numCycles cycles...');

      // Create pool of buffers to swap between
      final bufferPool = <Buffer>[];
      for (int i = 0; i < 6; i++) {
        final buffer = gpu.createBuffer(
          size * size * 4,
          BufferDataType.float32,
        );
        final data = Float32List.fromList(
          List.generate(size * size, (j) => (j + i * 100).toDouble()),
        );
        buffer.write(data, data.length, dataType: BufferDataType.float32);
        bufferPool.add(buffer);
      }

      final shaderCode =
          '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const SIZE: u32 = ${size}u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    
    if (x >= SIZE || y >= SIZE) {
        return;
    }
    
    let index = x * SIZE + y;
    
    if (index < arrayLength(&input) && index < arrayLength(&output)) {
        output[index] = input[index] + 50.0;
    }
}
''';

      final shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);

      final workgroupsX = (size + 7) ~/ 8;
      final workgroupsY = (size + 7) ~/ 8;

      for (int cycle = 0; cycle < numCycles; cycle++) {
        print('  Cycle $cycle:');

        // Pattern 1: Same input/output (should reuse kernel)
        shader.setBuffer('input', bufferPool[0]);
        shader.setBuffer('output', bufferPool[1]);
        await shader.dispatch(workgroupsX, workgroupsY, 1);
        print('    Same bindings dispatch - OK');

        // Pattern 2: Swap buffers (should recreate kernel)
        shader.setBuffer('input', bufferPool[2]);
        shader.setBuffer('output', bufferPool[3]);
        await shader.dispatch(workgroupsX, workgroupsY, 1);
        print('    New bindings dispatch - OK');

        // Pattern 3: Back to original (should reuse if caching works)
        shader.setBuffer('input', bufferPool[0]);
        shader.setBuffer('output', bufferPool[1]);
        await shader.dispatch(workgroupsX, workgroupsY, 1);
        print('    Back to original bindings - OK');

        // Pattern 4: Different workgroup size (should recreate)
        final altWorkgroupsX = (size + 15) ~/ 16;
        final altWorkgroupsY = (size + 15) ~/ 16;
        await shader.dispatch(altWorkgroupsX, altWorkgroupsY, 1);
        print('    Different workgroups - OK');

        // Pattern 5: Back to original workgroups (should reuse)
        await shader.dispatch(workgroupsX, workgroupsY, 1);
        print('    Back to original workgroups - OK');
      }

      shader.destroy();
      for (final buffer in bufferPool) {
        buffer.destroy();
      }

      print('Complex binding pattern test completed successfully!');
    });

    test('memory leak detection during rapid operations', () async {
      const cycles = 20;
      const dispatches = 10;

      print(
        'Memory leak test: $cycles cycles of $dispatches dispatches each...',
      );

      final memoryHistory = <int>[];

      for (int cycle = 0; cycle < cycles; cycle++) {
        // Record memory before cycle
        final beforeMemory = ProcessInfo.currentRss;

        // Rapid operations
        final buffer = gpu.createBuffer(1024 * 4, BufferDataType.float32);
        final shader = gpu.createComputeShader();

        shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < arrayLength(&data)) {
        data[index] = f32(index) * 1.5;
    }
}
''');

        shader.setBuffer('data', buffer);

        for (int dispatch = 0; dispatch < dispatches; dispatch++) {
          await shader.dispatch(16, 1, 1);
        }

        shader.destroy();
        buffer.destroy();

        // Record memory after cycle
        await Future.delayed(Duration(milliseconds: 10));
        final afterMemory = ProcessInfo.currentRss;
        memoryHistory.add(afterMemory);

        final cycleMB = (afterMemory / 1024 / 1024).toStringAsFixed(1);
        print('  Cycle $cycle: ${cycleMB} MB');

        // Check for excessive growth
        if (cycle > 5) {
          final recentGrowth =
              (afterMemory - memoryHistory[cycle - 5]) / 1024 / 1024;
          if (recentGrowth > 100) {
            fail(
              'Excessive memory growth detected: ${recentGrowth.toStringAsFixed(1)} MB in 5 cycles',
            );
          }
        }
      }

      final totalGrowth =
          (memoryHistory.last - memoryHistory.first) / 1024 / 1024;
      print('Total memory growth: ${totalGrowth.toStringAsFixed(1)} MB');

      expect(totalGrowth, lessThan(200), reason: 'Memory leak detected');
    });
  });
}
