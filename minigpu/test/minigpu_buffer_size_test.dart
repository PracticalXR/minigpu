import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  late Minigpu gpu;

  setUpAll(() async {
    gpu = Minigpu();
    await gpu.init();
  });

  tearDownAll(() async {
    // Clean up GPU resources if needed
  });

  group('Minigpu Buffer Tests', () {
    test('progressive buffer size crash diagnosis', () async {
      // Test buffer sizes that match our spectrogram scenarios
      final testSizes = [
        {'name': '1MB', 'elements': 256 * 1024},
        {'name': '5MB', 'elements': 1280 * 1024},
        {'name': '10MB', 'elements': 2560 * 1024},
        {'name': '25MB', 'elements': 6400 * 1024}, // This is where it crashes
        {'name': '50MB', 'elements': 12800 * 1024},
      ];

      for (final testCase in testSizes) {
        final sizeName = testCase['name'] as String;
        final elements = testCase['elements'] as int;
        final sizeBytes = elements * 4;

        print(
          '\n=== Testing $sizeName buffer (${elements} elements, ${sizeBytes} bytes) ===',
        );

        try {
          // Step 1: Buffer creation
          print('  Creating buffer...');
          final buffer = gpu.createBuffer(sizeBytes, BufferDataType.float32);
          print('   Buffer created successfully');

          // Step 2: Data preparation
          print('  Preparing test data...');
          final testData = Float32List(elements);
          for (int i = 0; i < elements; i++) {
            testData[i] = (i % 1000).toDouble();
          }
          print('   Test data prepared');

          // Step 3: Data writing
          print('  Writing data to buffer...');
          buffer.write(testData, elements, dataType: BufferDataType.float32);
          print('   Data written successfully');

          // Step 4: Data reading (this is where crashes often occur)
          print('  Reading data from buffer...');
          final readback = Float32List(elements);
          await buffer.read(
            readback,
            elements,
            dataType: BufferDataType.float32,
          );
          print('   Data read successfully');

          // Step 5: Verification
          print('  Verifying data integrity...');
          expect(readback.length, equals(elements));
          expect(readback[0], equals(0.0));
          expect(readback[999], equals(999.0));
          expect(readback[1000], equals(0.0)); // Pattern repeats
          print('   Data verification passed');

          // Step 6: Cleanup
          print('  Destroying buffer...');
          buffer.destroy();
          print('   Buffer destroyed successfully');
        } catch (e) {
          print('   FAILED at $sizeName: $e');
          print('  This is likely our crash threshold!');
          break; // Stop testing larger sizes once we hit the crash
        }
      }
    });

    test('compute shader with large buffers - spectrogram simulation', () async {
      // Simulate the exact scenario from our spectrogram
      final scenarios = [
        {'name': 'Small (1024 FFT)', 'width': 200, 'height': 1024},
        {'name': 'Medium (2048 FFT)', 'width': 200, 'height': 2048},
        {'name': 'Large (4096 FFT)', 'width': 200, 'height': 4096},
        {
          'name': 'XLarge (8192 FFT)',
          'width': 200,
          'height': 8192,
        }, // Crash point
      ];

      for (final scenario in scenarios) {
        final name = scenario['name'] as String;
        final width = scenario['width'] as int;
        final height = scenario['height'] as int;

        // Calculate buffer sizes like our spectrogram does
        final inputElements = width * height;
        final outputElements =
            (width * 2) * height * 4; // 2x width upscaling, RGBA
        final inputSizeBytes = inputElements * 4;
        final outputSizeBytes = outputElements * 4;

        print('\n=== $name Spectrogram Simulation ===');
        print(
          '  Input: ${width}x${height} = ${inputElements} elements (${(inputSizeBytes / 1024 / 1024).toStringAsFixed(1)}MB)',
        );
        print(
          '  Output: ${width * 2}x${height} RGBA = ${outputElements} elements (${(outputSizeBytes / 1024 / 1024).toStringAsFixed(1)}MB)',
        );

        try {
          // Create buffers
          print('  Creating input buffer...');
          final inputBuffer = gpu.createBuffer(
            inputSizeBytes,
            BufferDataType.float32,
          );

          print('  Creating output buffer...');
          final outputBuffer = gpu.createBuffer(
            outputSizeBytes,
            BufferDataType.float32,
          );

          // Prepare input data
          print('  Preparing input data...');
          final inputData = Float32List(inputElements);
          for (int i = 0; i < inputElements; i++) {
            inputData[i] =
                (i % 100).toDouble() / 100.0; // Normalized test pattern
          }

          // Write input data
          print('  Writing input data...');
          inputBuffer.write(
            inputData,
            inputElements,
            dataType: BufferDataType.float32,
          );

          // Create and setup compute shader
          print('  Creating compute shader...');
          final shader = gpu.createComputeShader();

          print('  Loading shader kernel...');
          shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const INPUT_WIDTH: u32 = ${width}u;
const INPUT_HEIGHT: u32 = ${height}u;
const OUTPUT_WIDTH: u32 = ${width * 2}u;
const OUTPUT_HEIGHT: u32 = ${height}u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    
    if (x >= OUTPUT_WIDTH || y >= OUTPUT_HEIGHT) {
        return;
    }
    
    // Simple upscaling: sample from input with nearest neighbor
    let inputX: u32 = x * INPUT_WIDTH / OUTPUT_WIDTH;
    let inputY: u32 = y;
    
    var value: f32 = 0.0;
    if (inputX < INPUT_WIDTH && inputY < INPUT_HEIGHT) {
        let inputIndex: u32 = inputY * INPUT_WIDTH + inputX;
        if (inputIndex < arrayLength(&input)) {
            value = input[inputIndex];
        }
    }
    
    // Write RGBA output
    let outputIndex: u32 = (y * OUTPUT_WIDTH + x) * 4u;
    if (outputIndex + 3u < arrayLength(&output)) {
        output[outputIndex] = value;     // R
        output[outputIndex + 1u] = value; // G
        output[outputIndex + 2u] = value; // B
        output[outputIndex + 3u] = 1.0;  // A
    }
}
''');

          print('  Setting input buffer...');
          shader.setBuffer('input', inputBuffer);

          print('  Setting output buffer...');
          shader.setBuffer('output', outputBuffer);

          // Calculate workgroups
          final workgroupsX = ((width * 2) + 7) ~/ 8;
          final workgroupsY = (height + 7) ~/ 8;

          print('  Dispatching ${workgroupsX}x${workgroupsY} workgroups...');
          await shader.dispatch(workgroupsX, workgroupsY, 1);
          print('   Dispatch completed successfully');

          // Read back output (this is often where the crash occurs)
          print('  Reading output buffer...');
          final outputData = Float32List(outputElements);
          await outputBuffer.read(
            outputData,
            outputElements,
            dataType: BufferDataType.float32,
          );
          print('   Output read successfully');

          // Basic verification
          print('  Verifying output...');
          expect(outputData.length, equals(outputElements));
          expect(outputData[3], equals(1.0)); // Alpha channel should be 1.0
          print('   Output verification passed');

          // Cleanup
          print('  Cleaning up...');
          shader.destroy();
          inputBuffer.destroy();
          outputBuffer.destroy();
          print('   Cleanup completed');
        } catch (e) {
          print('   FAILED: $e');
          print('  Stack trace: ${StackTrace.current}');
          rethrow; // Re-throw to see the full error in test output
        }
      }
    });

    test('memory pressure and descriptor table limits', () async {
      // Test if the issue is related to multiple large buffers or descriptor limits
      print('\n=== Testing Multiple Large Buffers ===');

      final buffers = <Buffer>[];
      const bufferSize = 5 * 1024 * 1024; // 5MB each
      const maxBuffers = 10;

      try {
        for (int i = 0; i < maxBuffers; i++) {
          print('  Creating buffer ${i + 1}/${maxBuffers} (5MB each)...');
          final buffer = gpu.createBuffer(
            bufferSize * 4,
            BufferDataType.float32,
          );
          buffers.add(buffer);

          // Fill with data to ensure actual allocation
          final data = Float32List(bufferSize);
          for (int j = 0; j < bufferSize; j++) {
            data[j] = (j + i * 1000).toDouble();
          }

          print('    Writing data to buffer ${i + 1}...');
          buffer.write(data, bufferSize, dataType: BufferDataType.float32);

          // Test immediate readback
          print('    Reading back from buffer ${i + 1}...');
          final readback = Float32List(bufferSize);
          await buffer.read(
            readback,
            bufferSize,
            dataType: BufferDataType.float32,
          );

          expect(readback[0], equals(i * 1000.0));
          print('   Buffer ${i + 1} created and verified successfully');
        }

        print('  All buffers created successfully!');

        // Test using them all in a compute shader
        print('  Testing compute shader with multiple large buffers...');
        final shader = gpu.createComputeShader();

        shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> buffer0: array<f32>;
@group(0) @binding(1) var<storage, read_write> buffer1: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&buffer0) && idx < arrayLength(&buffer1)) {
        buffer1[idx] = buffer0[idx] * 2.0;
    }
}
''');

        shader.setBuffer('buffer0', buffers[0]);
        shader.setBuffer('buffer1', buffers[1]);

        await shader.dispatch((bufferSize + 63) ~/ 64, 1, 1);
        print('   Compute shader with multiple large buffers succeeded');

        shader.destroy();
      } catch (e) {
        print('   Failed with ${buffers.length} buffers: $e');
        rethrow;
      } finally {
        // Cleanup all buffers
        for (int i = 0; i < buffers.length; i++) {
          buffers[i].destroy();
        }
        print('  Cleaned up ${buffers.length} buffers');
      }
    });

    test('exact crash reproduction - 25MB buffer dispatch', () async {
      // This should reproduce the exact crash scenario
      print('\n=== Exact Crash Reproduction Test ===');

      const width = 400; // 2x upscaled from 200
      const height = 4096; // 8192 FFT size / 2
      const elements = width * height * 4; // RGBA
      const sizeBytes = elements * 4; // Float32 bytes

      print(
        'Target: ${width}x${height} RGBA = ${elements} elements (${(sizeBytes / 1024 / 1024).toStringAsFixed(1)}MB)',
      );

      try {
        // This is the exact sequence that crashes
        print('  1. Creating large output buffer...');
        final buffer = gpu.createBuffer(sizeBytes, BufferDataType.float32);

        print('  2. Creating compute shader...');
        final shader = gpu.createComputeShader();

        print('  3. Loading simple shader kernel...');
        shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 8u;
    if (idx * 4u + 3u < arrayLength(&output)) {
        output[idx * 4u] = 1.0;     // R
        output[idx * 4u + 1u] = 0.5; // G
        output[idx * 4u + 2u] = 0.0; // B
        output[idx * 4u + 3u] = 1.0; // A
    }
}
''');

        print('  4. Setting buffer binding...');
        shader.setBuffer('output', buffer);

        print('  5. Dispatching workgroups...');
        final workgroupsX = (width + 7) ~/ 8;
        final workgroupsY = (height + 7) ~/ 8;
        print(
          '     Workgroups: ${workgroupsX}x${workgroupsY} = ${workgroupsX * workgroupsY} total',
        );

        // THIS IS WHERE THE CRASH SHOULD OCCUR
        await shader.dispatch(workgroupsX, workgroupsY, 1);
        print('   Dispatch completed successfully (unexpected!)');

        print('  6. Reading back data...');
        final readback = Float32List(elements);
        await buffer.read(readback, elements, dataType: BufferDataType.float32);
        print('   Readback completed successfully');

        // Cleanup
        shader.destroy();
        buffer.destroy();
        print('   Test passed - no crash occurred');
      } catch (e) {
        print('   CRASH REPRODUCED: $e');
        print('  This confirms the issue with large buffer dispatch');
        rethrow;
      }
    });

    test('workgroup size vs buffer size correlation', () async {
      // Test if the crash is related to the number of workgroups
      print('\n=== Workgroup vs Buffer Size Test ===');

      const bufferSize = 6400 * 1024; // 25MB worth of elements
      const scenarios = [
        {
          'name': 'Few Large Workgroups',
          'wgX': 100,
          'wgY': 100,
        }, // 10K workgroups
        {
          'name': 'Many Small Workgroups',
          'wgX': 500,
          'wgY': 500,
        }, // 250K workgroups
        {
          'name': 'Extreme Workgroups',
          'wgX': 1000,
          'wgY': 1000,
        }, // 1M workgroups
      ];

      for (final scenario in scenarios) {
        final name = scenario['name'] as String;
        final wgX = scenario['wgX'] as int;
        final wgY = scenario['wgY'] as int;

        print('  Testing $name: ${wgX}x${wgY} = ${wgX * wgY} workgroups');

        try {
          final buffer = gpu.createBuffer(
            bufferSize * 4,
            BufferDataType.float32,
          );
          final shader = gpu.createComputeShader();

          shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * ${wgX * 8}u;
    if (idx < arrayLength(&data)) {
        data[idx] = f32(idx);
    }
}
''');

          shader.setBuffer('data', buffer);

          await shader.dispatch(wgX, wgY, 1);
          print('     $name completed successfully');

          shader.destroy();
          buffer.destroy();
        } catch (e) {
          print('     $name failed: $e');
        }
      }
    });
  });
}
