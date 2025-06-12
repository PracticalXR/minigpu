import 'dart:async';
import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:isolate';
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  late Minigpu gpu;

  setUpAll(() async {
    gpu = Minigpu();
    await gpu.init();
  });

  group('Tensor Destruction Crash Tests', () {
    // Add crash detection to the aggressive test
    test(
      'aggressive concurrent tensor destruction with crash detection',
      () async {
        print(
          '\n=== Aggressive Concurrent Tensor Destruction (With Crash Detection) ===',
        );

        // Reduced parameters to isolate the crash point
        const numStreams = 3; // Fewer streams to pinpoint the issue
        const iterations = 5; // Fewer iterations to crash faster
        final fftSizes = [2048, 4096]; // Focus on sizes that crash

        final streamTensors = <String, Tensor>{};

        try {
          for (int i = 0; i < iterations; i++) {
            print('Aggressive iteration ${i + 1}/$iterations...');

            // Add checkpoint before each operation
            print('  Checkpoint: Starting iteration ${i + 1}');

            final futures = <Future<void>>[];

            for (int streamIdx = 0; streamIdx < numStreams; streamIdx++) {
              final streamId = 'aggressive_stream_$streamIdx';
              final fftSize = fftSizes[(i + streamIdx) % fftSizes.length];

              print('    Launching $streamId with FFT size $fftSize');

              // Wrap each operation to catch individual failures
              futures.add(
                _safeAggressiveTensorOperation(
                  gpu,
                  streamId,
                  fftSize,
                  streamTensors,
                  i,
                ).catchError((e, stackTrace) {
                  print('     Operation $streamId failed: $e');
                  print('    Stack trace: $stackTrace');
                  throw e;
                }),
              );
            }

            print('  Checkpoint: Waiting for ${futures.length} operations...');

            // Add timeout to detect hangs/deadlocks
            await Future.wait(futures).timeout(
              Duration(seconds: 10),
              onTimeout: () {
                print('   TIMEOUT: Operations are hanging - likely deadlock');
                throw TimeoutException(
                  'GPU operations timed out',
                  Duration(seconds: 10),
                );
              },
            );

            print('   Aggressive iteration ${i + 1} completed');

            // Brief checkpoint delay
            await Future.delayed(Duration(milliseconds: 50));
          }

          print(' All aggressive operations completed - crash was avoided');
        } catch (e, stackTrace) {
          print(' CRASH in aggressive operations: $e');
          print('Full stack trace: $stackTrace');

          // Analyze the crash type
          if (e is TimeoutException) {
            print('  -> DEADLOCK detected: GPU operations are blocking');
          } else if (e.toString().contains('CommandList') ||
              e.toString().contains('mD3d12CommandList') ||
              e.toString().contains('dawn')) {
            print(
              '  -> D3D12 VALIDATION ERROR: Command list destroyed while active',
            );
          } else {
            print('  -> UNKNOWN GPU ERROR: ${e.runtimeType}');
          }

          rethrow;
        } finally {
          print('Cleanup: ${streamTensors.length} tensors remaining');
          for (final entry in streamTensors.entries) {
            try {
              print('  Cleaning up ${entry.key}');
              entry.value.buffer.destroy();
            } catch (e) {
              print('  Cleanup error for ${entry.key}: $e');
            }
          }
        }
      },
    );

    // Minimal test to find the exact crash point
    test('minimal crash reproduction - single operation at a time', () async {
      print('\n=== Minimal Crash Reproduction ===');

      const fftSize = 4096; // Size that causes crashes
      var currentTensor = <String, Tensor>{};

      try {
        // Step 1: Create initial tensor
        print('Step 1: Creating initial tensor...');
        await _safeAggressiveTensorOperation(
          gpu,
          'test',
          fftSize,
          currentTensor,
          0,
        );
        print('   Initial tensor created');

        // Step 2: Destroy and recreate (first potential crash point)
        print('Step 2: First destroy/recreate cycle...');
        await _safeAggressiveTensorOperation(
          gpu,
          'test',
          fftSize,
          currentTensor,
          1,
        );
        print('   First cycle completed');

        // Step 3: Rapid destroy/recreate (second potential crash point)
        print('Step 3: Rapid destroy/recreate cycle...');
        await _safeAggressiveTensorOperation(
          gpu,
          'test',
          fftSize,
          currentTensor,
          2,
        );
        print('   Rapid cycle completed');

        // Step 4: Concurrent operations (most likely crash point)
        print('Step 4: Concurrent operations on same tensor...');
        final tensor = currentTensor['test']!;

        final futures = [
          _createSpectrogram(tensor, fftSize),
          _createSpectrogram2(tensor, fftSize),
        ];

        await Future.wait(futures);
        print('   Concurrent operations completed');

        print(' Minimal reproduction completed without crash');
      } catch (e, stackTrace) {
        print(' CRASH in minimal reproduction: $e');
        print('Stack trace: $stackTrace');
        rethrow;
      } finally {
        for (final tensor in currentTensor.values) {
          try {
            tensor.buffer.destroy();
          } catch (e) {
            print('Cleanup error: $e');
          }
        }
      }
    });

    // Test to isolate the exact destroy timing
    test('destroy timing isolation', () async {
      print('\n=== Destroy Timing Isolation ===');

      const fftSize = 4096;
      Tensor? tensorA, tensorB;

      try {
        // Create two tensors
        print('Creating tensor A...');
        tensorA = await Tensor.create([200, fftSize], gpu: gpu);

        print('Creating tensor B...');
        tensorB = await Tensor.create([200, fftSize], gpu: gpu);

        // Start operation on A
        print('Starting operation on tensor A...');
        final futureA = _createSpectrogram(tensorA, fftSize);

        // Wait just a bit to ensure A's operation has started
        await Future.delayed(Duration(milliseconds: 10));

        // Destroy B while A is running (should be safe)
        print('Destroying tensor B while A is active...');
        tensorB.buffer.destroy();
        tensorB = null;

        // Wait for A to complete
        await futureA;
        print('   Tensor A operation completed');

        // Now destroy A and immediately create new one
        print('Destroying tensor A and creating new one...');
        tensorA.buffer.destroy();

        tensorA = await Tensor.create([
          200,
          fftSize * 2,
        ], gpu: gpu); // Different size

        // Use the new tensor immediately
        await _createSpectrogram(tensorA, fftSize * 2);

        print(' Destroy timing test completed');
      } catch (e, stackTrace) {
        print(' CRASH in destroy timing test: $e');
        print('Stack trace: $stackTrace');
        rethrow;
      } finally {
        try {
          tensorA?.buffer.destroy();
        } catch (e) {
          print('Cleanup A error: $e');
        }
        try {
          tensorB?.buffer.destroy();
        } catch (e) {
          print('Cleanup B error: $e');
        }
      }
    });
  });
}

// Safer version with better error handling
Future<void> _safeAggressiveTensorOperation(
  Minigpu gpu,
  String streamId,
  int fftSize,
  Map<String, Tensor> tensors,
  int iteration,
) async {
  print(
    '      Safe aggressive operation: $streamId -> $fftSize (iter $iteration)',
  );

  try {
    // Checkpoint before destruction
    final oldTensor = tensors[streamId];
    if (oldTensor != null) {
      print('        Checkpoint: About to destroy old tensor for $streamId');
      oldTensor.buffer.destroy(); // CRASH POINT 1
      print('        Checkpoint: Old tensor destroyed for $streamId');
    }

    // Checkpoint before creation
    print('        Checkpoint: About to create new tensor for $streamId');
    final bufferData = List.generate(
      200 * fftSize,
      (i) => math.sin(i * 0.001 + iteration * 0.1) * 10.0,
    );

    final newTensor = await Tensor.create(
      [200, fftSize],
      gpu: gpu,
      data: Float32List.fromList(bufferData),
    );
    print('        Checkpoint: New tensor created for $streamId');

    tensors[streamId] = newTensor;

    // Checkpoint before GPU operations
    print('        Checkpoint: About to start GPU operations for $streamId');

    // Only do one operation at a time to isolate crash
    await _createSpectrogram(newTensor, fftSize); // CRASH POINT 2

    print('        Checkpoint: GPU operations completed for $streamId');
    print('       Safe aggressive $streamId completed');
  } catch (e, stackTrace) {
    print('       Safe aggressive $streamId failed at checkpoint: $e');
    print('      Stack trace: $stackTrace');
    rethrow;
  }
}

// Keep the existing helper functions
Future<void> _createSpectrogram(Tensor tensor, int fftSize) async {
  final outputTensor = await Tensor.create([
    400 * fftSize * 4,
  ], gpu: tensor.gpu);
  final shader = tensor.gpu.createComputeShader();

  shader.loadKernelString('''
    @group(0) @binding(0) var<storage, read_write> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x + gid.y * 50u;
        if (idx * 4u + 3u < arrayLength(&output)) {
            let inputIdx = idx % arrayLength(&input);
            var value: f32 = 0.5;
            if (inputIdx < arrayLength(&input)) {
                value = input[inputIdx] * 0.1;
            }
            output[idx * 4u] = value;
            output[idx * 4u + 1u] = value;
            output[idx * 4u + 2u] = value;
            output[idx * 4u + 3u] = 1.0;
        }
    }
  ''');

  shader.setBuffer('input', tensor.buffer);
  shader.setBuffer('output', outputTensor.buffer);

  await shader.dispatch((400 + 7) ~/ 8, (fftSize + 7) ~/ 8, 1);

  // Cleanup
  shader.destroy();
  outputTensor.buffer.destroy();
}

Future<void> _createSpectrogram2(Tensor tensor, int fftSize) async {
  final outputTensor = await Tensor.create([200 * fftSize], gpu: tensor.gpu);
  final shader = tensor.gpu.createComputeShader();

  shader.loadKernelString('''
    @group(0) @binding(0) var<storage, read_write> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    
    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x + gid.y * 13u;
        if (idx < arrayLength(&output)) {
            let inputIdx = idx % arrayLength(&input);
            if (inputIdx < arrayLength(&input)) {
                output[idx] = input[inputIdx] * 0.2;
            }
        }
    }
  ''');

  shader.setBuffer('input', tensor.buffer);
  shader.setBuffer('output', outputTensor.buffer);

  await shader.dispatch((200 + 15) ~/ 16, (fftSize + 15) ~/ 16, 1);

  // Cleanup
  shader.destroy();
  outputTensor.buffer.destroy();
}
