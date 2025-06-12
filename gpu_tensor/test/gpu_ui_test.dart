import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
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

  group('GPU Buffer Readback Tests', () {
    test('progressively larger buffer sizes', () async {
      final testSizes = [
        1024, // 4KB
        16384, // 64KB
        65536, // 256KB
        102400, // 400KB (similar to texture data)
        409600, // 1.6MB (exact size: 200*512*4)
        1048576, // 4MB
      ];

      for (final size in testSizes) {
        print(
          'Testing buffer size: $size floats (${(size * 4 / 1024).toStringAsFixed(1)}KB)',
        );

        // Create tensor with test data
        final testData = Float32List(size);
        for (int i = 0; i < size; i++) {
          testData[i] = (i % 256) / 255.0; // Simple test pattern
        }

        final tensor = await Tensor.create([size], gpu: gpu, data: testData);

        // Try to read back the data
        final readbackData = await tensor.getData() as Float32List;
        expect(readbackData.length, equals(size));

        // Verify first few values
        for (int i = 0; i < 10 && i < size; i++) {
          expect(
            (testData[i] - readbackData[i]).abs(),
            lessThan(0.001),
            reason: 'Data mismatch at index $i',
          );
        }

        // Cleanup
        tensor.destroy();
      }
    });

    test('spectrogram scenario recreation', () async {
      const tensorTimeSlices = 200;
      const tensorFreqBins = 512;
      const outputSize = tensorTimeSlices * tensorFreqBins * 4; // RGBA

      // Create input tensor
      final inputData = Float32List(tensorTimeSlices * tensorFreqBins);
      for (int i = 0; i < inputData.length; i++) {
        inputData[i] = (i % 100) / 100.0; // Test pattern
      }

      final inputTensor = await Tensor.create(
        [tensorTimeSlices, tensorFreqBins],
        gpu: gpu,
        data: inputData,
      );

      // Create output tensor
      final outputTensor = await Tensor.create([outputSize], gpu: gpu);

      // Compute shader to fill output with test pattern
      final shaderCode =
          '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const TEXTURE_WIDTH: u32 = ${tensorTimeSlices}u;
const TEXTURE_HEIGHT: u32 = ${tensorFreqBins}u;
const OUTPUT_SIZE: u32 = ${outputSize}u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    
    if (x >= TEXTURE_WIDTH || y >= TEXTURE_HEIGHT) {
        return;
    }
    
    let inputIndex: u32 = x * TEXTURE_HEIGHT + y;
    let outputIndex: u32 = (y * TEXTURE_WIDTH + x) * 4u;
    
    var value: f32 = 0.0;
    if (inputIndex < arrayLength(&input)) {
        value = input[inputIndex];
    }
    
    // Write RGBA
    if (outputIndex + 3u < OUTPUT_SIZE && outputIndex + 3u < arrayLength(&output)) {
        output[outputIndex] = value;        // R
        output[outputIndex + 1u] = 0.0;     // G
        output[outputIndex + 2u] = 0.0;     // B
        output[outputIndex + 3u] = 1.0;     // A
    }
}
''';

      final shader = gpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('input', inputTensor.buffer);
      shader.setBuffer('output', outputTensor.buffer);

      final workgroupsX = (tensorTimeSlices + 15) ~/ 16;
      final workgroupsY = (tensorFreqBins + 15) ~/ 16;

      await shader.dispatch(workgroupsX, workgroupsY, 1);
      shader.destroy();

      // Critical test - readback the large output buffer
      final outputData = await outputTensor.getData() as Float32List;
      expect(outputData.length, equals(outputSize));

      // Verify some RGBA values
      expect(outputData[3], equals(1.0)); // Alpha should be 1.0
      expect(outputData[7], equals(1.0)); // Alpha should be 1.0

      // Cleanup
      inputTensor.destroy();
      outputTensor.destroy();
    });

    test('memory allocation limits', () async {
      final sizes = [
        1024 * 1024, // 4MB
        2 * 1024 * 1024, // 8MB
        4 * 1024 * 1024, // 16MB
        8 * 1024 * 1024, // 32MB
        16 * 1024 * 1024, // 64MB
      ];

      Tensor? lastSuccessfulTensor;
      int maxSuccessfulSize = 0;

      for (final size in sizes) {
        try {
          print(
            'Testing ${(size * 4 / 1024 / 1024).toStringAsFixed(1)}MB allocation',
          );

          final tensor = await Tensor.create([size], gpu: gpu);

          // Try to read a small portion back
          final data = await tensor.getData() as Float32List;
          expect(data.length, equals(size));

          lastSuccessfulTensor?.destroy();
          lastSuccessfulTensor = tensor;
          maxSuccessfulSize = size;
        } catch (e) {
          print(
            'Failed at ${(size * 4 / 1024 / 1024).toStringAsFixed(1)}MB: $e',
          );
          break;
        }
      }

      lastSuccessfulTensor?.destroy();

      // Expect at least 4MB to work
      expect(
        maxSuccessfulSize,
        greaterThanOrEqualTo(1024 * 1024),
        reason: 'Should be able to allocate at least 4MB',
      );
    });

    test('buffer readback consistency', () async {
      const size = 102400; // 400KB test

      // Create test data with known pattern
      final testData = Float32List(size);
      for (int i = 0; i < size; i++) {
        testData[i] = i.toDouble();
      }

      final tensor = await Tensor.create([size], gpu: gpu, data: testData);

      // Read back multiple times to check consistency
      for (int attempt = 0; attempt < 3; attempt++) {
        final readbackData = await tensor.getData() as Float32List;
        expect(readbackData.length, equals(size));

        // Check every 100th element for performance
        for (int i = 0; i < size; i += 100) {
          expect(
            readbackData[i],
            equals(testData[i]),
            reason: 'Mismatch at index $i on attempt $attempt',
          );
        }
      }

      tensor.destroy();
    });

    test('concurrent buffer operations', () async {
      const size = 50000; // Moderate size for concurrent test

      // Create multiple tensors concurrently
      final futures = <Future<Tensor>>[];
      for (int i = 0; i < 5; i++) {
        final data = Float32List.fromList(
          List.generate(size, (index) => (index + i).toDouble()),
        );
        futures.add(Tensor.create([size], gpu: gpu, data: data));
      }

      final tensors = await Future.wait(futures);

      // Read back all tensors concurrently
      final readbackFutures = tensors.map((t) => t.getData()).toList();
      final results = await Future.wait(readbackFutures);

      // Verify each result
      for (int i = 0; i < results.length; i++) {
        final data = results[i] as Float32List;
        expect(data.length, equals(size));
        expect(data[0], equals(i.toDouble()));
        expect(data[size - 1], equals((size - 1 + i).toDouble()));
      }

      // Cleanup
      for (final tensor in tensors) {
        tensor.destroy();
      }
    });
  });
}
