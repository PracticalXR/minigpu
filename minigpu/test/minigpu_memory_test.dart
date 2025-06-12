import 'dart:typed_data';
import 'dart:io';
import 'package:test/test.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  late Minigpu gpu;

  setUpAll(() async {
    gpu = Minigpu();
    await gpu.init();
  });

  tearDownAll(() async {});

  group('Memory Leak Tests', () {
    test('shader creation/destruction memory leak test', () async {
      const numCycles = 100;
      const memoryCheckInterval = 10;

      print(
        'Testing shader creation/destruction for memory leaks over $numCycles cycles...',
      );

      final memorySnapshots = <int>[];

      for (int cycle = 0; cycle < numCycles; cycle++) {
        // Create and immediately destroy shader
        final shader = gpu.createComputeShader();
        shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < arrayLength(&data)) {
        data[index] = f32(index);
    }
}
''');
        shader.destroy();

        // Check memory usage periodically
        if (cycle % memoryCheckInterval == 0) {
          // Force garbage collection
          await Future.delayed(Duration(milliseconds: 10));

          final memoryUsage = ProcessInfo.currentRss;
          memorySnapshots.add(memoryUsage);
          print(
            '  Cycle $cycle: Memory usage: ${(memoryUsage / 1024 / 1024).toStringAsFixed(2)} MB',
          );

          // Check for significant memory growth
          if (memorySnapshots.length > 3) {
            final firstSnapshot = memorySnapshots[0];
            final currentSnapshot = memorySnapshots.last;
            final growth = currentSnapshot - firstSnapshot;
            final growthMB = growth / 1024 / 1024;

            if (growthMB > 50) {
              // Alert if memory grew by more than 50MB
              print(
                'WARNING: Potential memory leak detected! Growth: ${growthMB.toStringAsFixed(2)} MB',
              );
            }
          }
        }
      }

      // Final analysis
      if (memorySnapshots.length >= 2) {
        final initialMemory = memorySnapshots.first;
        final finalMemory = memorySnapshots.last;
        final totalGrowth = (finalMemory - initialMemory) / 1024 / 1024;

        print('Memory analysis:');
        print(
          '  Initial: ${(initialMemory / 1024 / 1024).toStringAsFixed(2)} MB',
        );
        print('  Final: ${(finalMemory / 1024 / 1024).toStringAsFixed(2)} MB');
        print('  Growth: ${totalGrowth.toStringAsFixed(2)} MB');

        // Fail test if memory growth is excessive
        expect(
          totalGrowth,
          lessThan(100),
          reason:
              'Memory leak detected: ${totalGrowth.toStringAsFixed(2)} MB growth',
        );
      }
    });

    test('dispatch memory leak test', () async {
      const numDispatches = 500;
      const memoryCheckInterval = 50;

      print(
        'Testing dispatch operations for memory leaks over $numDispatches dispatches...',
      );

      final buffer = gpu.createBuffer(1024 * 4, BufferDataType.float32);
      final data = Float32List.fromList(
        List.generate(1024, (i) => i.toDouble()),
      );
      buffer.write(data, 1024, dataType: BufferDataType.float32);

      final shader = gpu.createComputeShader();
      shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < arrayLength(&data)) {
        data[index] = data[index] * 2.0;
    }
}
''');
      shader.setBuffer('data', buffer);

      final memorySnapshots = <int>[];

      for (int dispatch = 0; dispatch < numDispatches; dispatch++) {
        await shader.dispatch(16, 1, 1);

        if (dispatch % memoryCheckInterval == 0) {
          await Future.delayed(Duration(milliseconds: 5));

          final memoryUsage = ProcessInfo.currentRss;
          memorySnapshots.add(memoryUsage);
          print(
            '  Dispatch $dispatch: Memory usage: ${(memoryUsage / 1024 / 1024).toStringAsFixed(2)} MB',
          );
        }
      }

      shader.destroy();
      buffer.destroy();

      // Analyze dispatch memory growth
      if (memorySnapshots.length >= 2) {
        final initialMemory = memorySnapshots.first;
        final finalMemory = memorySnapshots.last;
        final totalGrowth = (finalMemory - initialMemory) / 1024 / 1024;

        print('Dispatch memory analysis:');
        print(
          '  Initial: ${(initialMemory / 1024 / 1024).toStringAsFixed(2)} MB',
        );
        print('  Final: ${(finalMemory / 1024 / 1024).toStringAsFixed(2)} MB');
        print('  Growth: ${totalGrowth.toStringAsFixed(2)} MB');

        expect(
          totalGrowth,
          lessThan(50),
          reason:
              'Dispatch memory leak detected: ${totalGrowth.toStringAsFixed(2)} MB growth',
        );
      }
    });

    test('mixed operations memory stress test', () async {
      const numCycles = 50;

      print('Mixed operations memory stress test...');

      final initialMemory = ProcessInfo.currentRss;
      print(
        'Initial memory: ${(initialMemory / 1024 / 1024).toStringAsFixed(2)} MB',
      );

      for (int cycle = 0; cycle < numCycles; cycle++) {
        // Create multiple resources
        final buffers = <Buffer>[];
        final shaders = <ComputeShader>[];

        // Create buffers
        for (int i = 0; i < 5; i++) {
          final buffer = gpu.createBuffer(512 * 4, BufferDataType.float32);
          final data = Float32List.fromList(
            List.generate(512, (j) => (j + i).toDouble()),
          );
          buffer.write(data, 512, dataType: BufferDataType.float32);
          buffers.add(buffer);
        }

        // Create shaders and dispatch
        for (int i = 0; i < 3; i++) {
          final shader = gpu.createComputeShader();
          shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < arrayLength(&input) && index < arrayLength(&output)) {
        output[index] = input[index] + ${i.toDouble()};
    }
}
''');
          shader.setBuffer('input', buffers[i]);
          shader.setBuffer('output', buffers[i + 1]);
          await shader.dispatch(16, 1, 1);
          shaders.add(shader);
        }

        // Clean up
        for (final shader in shaders) {
          shader.destroy();
        }
        for (final buffer in buffers) {
          buffer.destroy();
        }

        if (cycle % 10 == 0) {
          final currentMemory = ProcessInfo.currentRss;
          final growth = (currentMemory - initialMemory) / 1024 / 1024;
          print(
            '  Cycle $cycle: Memory: ${(currentMemory / 1024 / 1024).toStringAsFixed(2)} MB (growth: ${growth.toStringAsFixed(2)} MB)',
          );
        }
      }

      await Future.delayed(Duration(milliseconds: 100)); // Let things settle

      final finalMemory = ProcessInfo.currentRss;
      final totalGrowth = (finalMemory - initialMemory) / 1024 / 1024;

      print('Mixed operations final analysis:');
      print(
        '  Initial: ${(initialMemory / 1024 / 1024).toStringAsFixed(2)} MB',
      );
      print('  Final: ${(finalMemory / 1024 / 1024).toStringAsFixed(2)} MB');
      print('  Growth: ${totalGrowth.toStringAsFixed(2)} MB');

      expect(
        totalGrowth,
        lessThan(100),
        reason:
            'Mixed operations memory leak: ${totalGrowth.toStringAsFixed(2)} MB growth',
      );
    });

    test('kernel compilation memory leak test', () async {
      const numKernels = 100;
      const memoryCheckInterval = 10;

      print(
        'Testing kernel compilation memory leaks with $numKernels different kernels...',
      );

      final memorySnapshots = <int>[];
      final shader = gpu.createComputeShader();
      final buffer = gpu.createBuffer(1024 * 4, BufferDataType.float32);

      for (int i = 0; i < numKernels; i++) {
        // Create a unique kernel each time to force recompilation
        final kernelCode =
            '''
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

const MULTIPLIER: f32 = ${(i + 1).toDouble()};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < arrayLength(&data)) {
        data[index] = f32(index) * MULTIPLIER;
    }
}
''';

        shader.loadKernelString(kernelCode);
        shader.setBuffer('data', buffer);

        // Dispatch to trigger kernel compilation
        await shader.dispatch(16, 1, 1);

        if (i % memoryCheckInterval == 0) {
          await Future.delayed(Duration(milliseconds: 10));

          final memoryUsage = ProcessInfo.currentRss;
          memorySnapshots.add(memoryUsage);
          print(
            '  Kernel $i: Memory usage: ${(memoryUsage / 1024 / 1024).toStringAsFixed(2)} MB',
          );
        }
      }

      shader.destroy();
      buffer.destroy();

      // Analyze kernel compilation memory growth
      if (memorySnapshots.length >= 2) {
        final initialMemory = memorySnapshots.first;
        final finalMemory = memorySnapshots.last;
        final totalGrowth = (finalMemory - initialMemory) / 1024 / 1024;

        print('Kernel compilation memory analysis:');
        print(
          '  Initial: ${(initialMemory / 1024 / 1024).toStringAsFixed(2)} MB',
        );
        print('  Final: ${(finalMemory / 1024 / 1024).toStringAsFixed(2)} MB');
        print('  Growth: ${totalGrowth.toStringAsFixed(2)} MB');

        expect(
          totalGrowth,
          lessThan(150),
          reason:
              'Kernel compilation memory leak: ${totalGrowth.toStringAsFixed(2)} MB growth',
        );
      }
    });

    test('rapid binding changes memory leak test', () async {
      const numBindingCycles = 200;
      const buffersPerCycle = 8;
      const memoryCheckInterval = 20;

      print(
        'Testing rapid binding changes for memory leaks over $numBindingCycles cycles...',
      );

      final memorySnapshots = <int>[];
      final shader = gpu.createComputeShader();

      shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < arrayLength(&input) && index < arrayLength(&output)) {
        output[index] = input[index] + 1.0;
    }
}
''');

      for (int cycle = 0; cycle < numBindingCycles; cycle++) {
        // Create temporary buffers for this cycle
        final tempBuffers = <Buffer>[];

        for (int b = 0; b < buffersPerCycle; b++) {
          final inputBuffer = gpu.createBuffer(512 * 4, BufferDataType.float32);
          final outputBuffer = gpu.createBuffer(
            512 * 4,
            BufferDataType.float32,
          );

          // Fill input with test data
          final inputData = Float32List.fromList(
            List.generate(512, (i) => (i + cycle + b).toDouble()),
          );
          inputBuffer.write(inputData, 512, dataType: BufferDataType.float32);

          // Change bindings rapidly
          shader.setBuffer('input', inputBuffer);
          shader.setBuffer('output', outputBuffer);

          // Dispatch with new bindings
          await shader.dispatch(16, 1, 1);

          tempBuffers.addAll([inputBuffer, outputBuffer]);
        }

        // Clean up all buffers from this cycle
        for (final buffer in tempBuffers) {
          buffer.destroy();
        }

        if (cycle % memoryCheckInterval == 0) {
          await Future.delayed(Duration(milliseconds: 10));

          final memoryUsage = ProcessInfo.currentRss;
          memorySnapshots.add(memoryUsage);
          print(
            '  Cycle $cycle: Memory usage: ${(memoryUsage / 1024 / 1024).toStringAsFixed(2)} MB',
          );

          // Check for rapid growth
          if (memorySnapshots.length > 3) {
            final recentGrowth =
                (memorySnapshots.last -
                    memorySnapshots[memorySnapshots.length - 3]) /
                1024 /
                1024;
            if (recentGrowth > 100) {
              print(
                'WARNING: Rapid memory growth detected: ${recentGrowth.toStringAsFixed(2)} MB in recent cycles',
              );
            }
          }
        }
      }

      shader.destroy();

      // Final analysis
      if (memorySnapshots.length >= 2) {
        final initialMemory = memorySnapshots.first;
        final finalMemory = memorySnapshots.last;
        final totalGrowth = (finalMemory - initialMemory) / 1024 / 1024;

        print('Binding changes memory analysis:');
        print(
          '  Initial: ${(initialMemory / 1024 / 1024).toStringAsFixed(2)} MB',
        );
        print('  Final: ${(finalMemory / 1024 / 1024).toStringAsFixed(2)} MB');
        print('  Growth: ${totalGrowth.toStringAsFixed(2)} MB');

        expect(
          totalGrowth,
          lessThan(200),
          reason:
              'Binding changes memory leak: ${totalGrowth.toStringAsFixed(2)} MB growth',
        );
      }
    });

    test('kernel cache invalidation memory leak test', () async {
      const numIterations = 50;
      const bindingChangesPerIteration = 10;

      print('Testing kernel cache invalidation memory leaks...');

      final memorySnapshots = <int>[];
      final shader = gpu.createComputeShader();

      // Create a pool of buffers to rotate through
      final bufferPool = <Buffer>[];
      for (int i = 0; i < 20; i++) {
        final buffer = gpu.createBuffer(1024 * 4, BufferDataType.float32);
        final data = Float32List.fromList(
          List.generate(1024, (j) => (j + i * 100).toDouble()),
        );
        buffer.write(data, 1024, dataType: BufferDataType.float32);
        bufferPool.add(buffer);
      }

      for (int iteration = 0; iteration < numIterations; iteration++) {
        // Load a slightly different kernel to force recompilation
        shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const ITERATION: f32 = ${iteration.toDouble()};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < arrayLength(&input) && index < arrayLength(&output)) {
        output[index] = input[index] + ITERATION;
    }
}
''');

        // Rapidly change bindings to trigger cache invalidation
        for (int b = 0; b < bindingChangesPerIteration; b++) {
          final inputIndex =
              (iteration * bindingChangesPerIteration + b) % bufferPool.length;
          final outputIndex = (inputIndex + 1) % bufferPool.length;

          shader.setBuffer('input', bufferPool[inputIndex]);
          shader.setBuffer('output', bufferPool[outputIndex]);

          await shader.dispatch(16, 1, 1);
        }

        if (iteration % 5 == 0) {
          await Future.delayed(Duration(milliseconds: 10));

          final memoryUsage = ProcessInfo.currentRss;
          memorySnapshots.add(memoryUsage);
          print(
            '  Iteration $iteration: Memory usage: ${(memoryUsage / 1024 / 1024).toStringAsFixed(2)} MB',
          );
        }
      }

      shader.destroy();
      for (final buffer in bufferPool) {
        buffer.destroy();
      }

      // Analyze cache invalidation memory growth
      if (memorySnapshots.length >= 2) {
        final initialMemory = memorySnapshots.first;
        final finalMemory = memorySnapshots.last;
        final totalGrowth = (finalMemory - initialMemory) / 1024 / 1024;

        print('Cache invalidation memory analysis:');
        print(
          '  Initial: ${(initialMemory / 1024 / 1024).toStringAsFixed(2)} MB',
        );
        print('  Final: ${(finalMemory / 1024 / 1024).toStringAsFixed(2)} MB');
        print('  Growth: ${totalGrowth.toStringAsFixed(2)} MB');

        expect(
          totalGrowth,
          lessThan(300),
          reason:
              'Cache invalidation memory leak: ${totalGrowth.toStringAsFixed(2)} MB growth',
        );
      }
    });

    test('massive binding permutation stress test', () async {
      const numShaders = 5;
      const numBuffers = 10;
      const numPermutations = 100;

      print('Massive binding permutation stress test...');

      final initialMemory = ProcessInfo.currentRss;
      print(
        'Initial memory: ${(initialMemory / 1024 / 1024).toStringAsFixed(2)} MB',
      );

      // Create shader pool
      final shaders = <ComputeShader>[];
      for (int s = 0; s < numShaders; s++) {
        final shader = gpu.createComputeShader();
        shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const SHADER_ID: f32 = ${s.toDouble()};

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < arrayLength(&input) && index < arrayLength(&output)) {
        output[index] = input[index] * SHADER_ID;
    }
}
''');
        shaders.add(shader);
      }

      // Create buffer pool
      final buffers = <Buffer>[];
      for (int b = 0; b < numBuffers; b++) {
        final buffer = gpu.createBuffer(512 * 4, BufferDataType.float32);
        final data = Float32List.fromList(
          List.generate(512, (i) => (i + b * 10).toDouble()),
        );
        buffer.write(data, 512, dataType: BufferDataType.float32);
        buffers.add(buffer);
      }

      // Create many different binding permutations
      for (int perm = 0; perm < numPermutations; perm++) {
        final shaderIndex = perm % numShaders;
        final inputIndex = perm % numBuffers;
        final outputIndex = (perm + 1) % numBuffers;

        final shader = shaders[shaderIndex];
        shader.setBuffer('input', buffers[inputIndex]);
        shader.setBuffer('output', buffers[outputIndex]);

        await shader.dispatch(16, 1, 1);

        if (perm % 20 == 0) {
          final currentMemory = ProcessInfo.currentRss;
          final growth = (currentMemory - initialMemory) / 1024 / 1024;
          print(
            '  Permutation $perm: Memory growth: ${growth.toStringAsFixed(2)} MB',
          );

          if (growth > 500) {
            fail(
              'Excessive memory growth during permutation test: ${growth.toStringAsFixed(2)} MB',
            );
          }
        }
      }

      // Clean up
      for (final shader in shaders) {
        shader.destroy();
      }
      for (final buffer in buffers) {
        buffer.destroy();
      }

      await Future.delayed(Duration(milliseconds: 100));

      final finalMemory = ProcessInfo.currentRss;
      final totalGrowth = (finalMemory - initialMemory) / 1024 / 1024;

      print('Binding permutation final analysis:');
      print(
        '  Initial: ${(initialMemory / 1024 / 1024).toStringAsFixed(2)} MB',
      );
      print('  Final: ${(finalMemory / 1024 / 1024).toStringAsFixed(2)} MB');
      print('  Growth: ${totalGrowth.toStringAsFixed(2)} MB');

      expect(
        totalGrowth,
        lessThan(400),
        reason:
            'Binding permutation memory leak: ${totalGrowth.toStringAsFixed(2)} MB growth',
      );
    });
  });
}
