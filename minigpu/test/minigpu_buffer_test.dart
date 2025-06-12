import 'dart:io';
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:minigpu/minigpu.dart';

class MemoryTracker {
  static int _initialMemory = 0;
  static final Map<String, int> checkpoints = {};

  static void start(String testName) {
    _initialMemory = ProcessInfo.currentRss;
    checkpoints.clear();
    print(
      '[$testName] Starting memory tracking: ${(_initialMemory / 1024 / 1024).toStringAsFixed(2)} MB',
    );
  }

  static void checkpoint(String operation) {
    final currentMemory = ProcessInfo.currentRss;
    final growth = currentMemory - _initialMemory;
    checkpoints[operation] = growth;
    print(
      '  [$operation] Memory: ${(currentMemory / 1024 / 1024).toStringAsFixed(2)} MB (+${(growth / 1024 / 1024).toStringAsFixed(2)} MB)',
    );
  }

  static void analyze() {
    print('\nMemory Growth Analysis:');
    final sortedCheckpoints = checkpoints.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    for (final entry in sortedCheckpoints) {
      final growthMB = entry.value / 1024 / 1024;
      print('  ${entry.key}: +${growthMB.toStringAsFixed(2)} MB');
    }
  }
}

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
    test('basic buffer creation and readback', () async {
      const size = 1024;
      final buffer = gpu.createBuffer(size * 4, BufferDataType.float32);

      // Create test data
      final testData = Float32List(size);
      for (int i = 0; i < size; i++) {
        testData[i] = i.toDouble();
      }

      // Write data to buffer
      buffer.write(testData, size, dataType: BufferDataType.float32);

      // Read back data
      final readbackData = Float32List(size);
      await buffer.read(readbackData, size, dataType: BufferDataType.float32);

      // Verify
      expect(readbackData.length, equals(size));
      for (int i = 0; i < 10; i++) {
        expect(readbackData[i], equals(testData[i]));
      }

      buffer.destroy();
    });

    test('concurrent buffer operations - should fail', () async {
      const size = 50000; // Moderate size
      const numBuffers = 5;

      // Create multiple buffers concurrently
      final buffers = <Buffer>[];
      final testDataSets = <Float32List>[];

      for (int i = 0; i < numBuffers; i++) {
        final buffer = gpu.createBuffer(size * 4, BufferDataType.float32);
        buffers.add(buffer);

        final testData = Float32List(size);
        for (int j = 0; j < size; j++) {
          testData[j] = (j + i * 1000).toDouble(); // Unique pattern per buffer
        }
        testDataSets.add(testData);

        // Write data to buffer
        buffer.write(testData, size, dataType: BufferDataType.float32);
      }

      // Read back all buffers concurrently
      final readbackFutures = <Future<Float32List>>[];
      for (int i = 0; i < numBuffers; i++) {
        final future = () async {
          final readbackData = Float32List(size);
          await buffers[i].read(
            readbackData,
            size,
            dataType: BufferDataType.float32,
          );
          return readbackData;
        }();
        readbackFutures.add(future);
      }

      final results = await Future.wait(readbackFutures);

      // Verify each result matches its expected pattern
      for (int i = 0; i < numBuffers; i++) {
        final readbackData = results[i];
        final expectedData = testDataSets[i];

        expect(readbackData.length, equals(size));

        // Check first and last elements
        expect(
          readbackData[0],
          equals(expectedData[0]),
          reason: 'Buffer $i: First element mismatch',
        );
        expect(
          readbackData[size - 1],
          equals(expectedData[size - 1]),
          reason: 'Buffer $i: Last element mismatch',
        );

        // Sample check throughout the buffer
        for (int j = 0; j < size; j += 1000) {
          expect(
            readbackData[j],
            equals(expectedData[j]),
            reason: 'Buffer $i: Mismatch at index $j',
          );
        }
      }

      // Cleanup
      for (final buffer in buffers) {
        buffer.destroy();
      }
    });

    test('sequential vs concurrent buffer read timing', () async {
      const size = 100000;
      const numBuffers = 3;

      // Setup buffers with test data
      final buffers = <Buffer>[];
      for (int i = 0; i < numBuffers; i++) {
        final buffer = gpu.createBuffer(size * 4, BufferDataType.float32);
        buffers.add(buffer);

        final testData = Float32List.fromList(
          List.generate(size, (index) => (index + i * 1000).toDouble()),
        );
        buffer.write(testData, size, dataType: BufferDataType.float32);
      }

      // Sequential reads
      final sequentialStart = DateTime.now();
      for (int i = 0; i < numBuffers; i++) {
        final readbackData = Float32List(size);
        await buffers[i].read(
          readbackData,
          size,
          dataType: BufferDataType.float32,
        );
      }
      final sequentialTime = DateTime.now().difference(sequentialStart);
      print('Sequential read time: ${sequentialTime.inMilliseconds}ms');

      // Concurrent reads
      final concurrentStart = DateTime.now();
      final futures = buffers.map((buffer) async {
        final readbackData = Float32List(size);
        await buffer.read(readbackData, size, dataType: BufferDataType.float32);
        return readbackData;
      }).toList();

      final results = await Future.wait(futures);
      final concurrentTime = DateTime.now().difference(concurrentStart);
      print('Concurrent read time: ${concurrentTime.inMilliseconds}ms');

      // Verify results are correct
      for (int i = 0; i < numBuffers; i++) {
        expect(results[i][0], equals(i * 1000.0));
        expect(results[i][size - 1], equals(size - 1 + i * 1000.0));
      }

      // Cleanup
      for (final buffer in buffers) {
        buffer.destroy();
      }

      // Concurrent should be faster or at least not much slower
      expect(
        concurrentTime.inMilliseconds,
        lessThanOrEqualTo(sequentialTime.inMilliseconds * 2),
        reason:
            'Concurrent reads should not be significantly slower than sequential',
      );
    });

    test('large buffer stress test', () async {
      final sizes = [
        256 * 1024, // 1MB
        512 * 1024, // 2MB
        1024 * 1024, // 4MB
        2048 * 1024, // 8MB
      ];

      for (final size in sizes) {
        print(
          'Testing ${(size * 4 / 1024 / 1024).toStringAsFixed(1)}MB buffer',
        );

        final buffer = gpu.createBuffer(size * 4, BufferDataType.float32);

        // Create pattern that's easy to verify
        final testData = Float32List(size);
        for (int i = 0; i < size; i++) {
          testData[i] = (i % 1000).toDouble();
        }

        buffer.write(testData, size, dataType: BufferDataType.float32);

        final readbackData = Float32List(size);
        await buffer.read(readbackData, size, dataType: BufferDataType.float32);

        // Verify pattern
        expect(readbackData.length, equals(size));
        expect(readbackData[0], equals(0.0));
        expect(readbackData[999], equals(999.0));
        expect(readbackData[1000], equals(0.0)); // Pattern repeats

        buffer.destroy();
      }
    });

    test('buffer read/write data type consistency', () async {
      const size = 1000;

      final testCases = [
        BufferDataType.float32,
        BufferDataType.int32,
        BufferDataType.uint32,
      ];

      for (final dataType in testCases) {
        final buffer = gpu.createBuffer(size * 4, dataType);

        switch (dataType) {
          case BufferDataType.float32:
            final testData = Float32List.fromList(
              List.generate(size, (i) => i * 0.5),
            );
            buffer.write(testData, size, dataType: dataType);

            final readback = Float32List(size);
            await buffer.read(readback, size, dataType: dataType);

            expect(readback[0], equals(0.0));
            expect(readback[10], equals(5.0));
            break;

          case BufferDataType.int32:
            final testData = Int32List.fromList(
              List.generate(size, (i) => i * 2),
            );
            buffer.write(testData, size, dataType: dataType);

            final readback = Int32List(size);
            await buffer.read(readback, size, dataType: dataType);

            expect(readback[0], equals(0));
            expect(readback[10], equals(20));
            break;

          case BufferDataType.uint32:
            final testData = Uint32List.fromList(
              List.generate(size, (i) => i * 3),
            );
            buffer.write(testData, size, dataType: dataType);

            final readback = Uint32List(size);
            await buffer.read(readback, size, dataType: dataType);

            expect(readback[0], equals(0));
            expect(readback[10], equals(30));
            break;

          default:
            // Skip other types for this test
            continue;
        }

        buffer.destroy();
      }
    });

    test('buffer creation/destruction memory leak test', () async {
      const numCycles = 400;
      const bufferSize = 1024 * 1024; // 1MB per buffer
      const memoryCheckInterval = 20;

      print(
        'Testing buffer creation/destruction for memory leaks over $numCycles cycles...',
      );

      final memorySnapshots = <int>[];

      for (int cycle = 0; cycle < numCycles; cycle++) {
        // Create and immediately destroy buffer
        final buffer = gpu.createBuffer(bufferSize * 4, BufferDataType.float32);

        // Fill with data to ensure allocation
        final data = Float32List(bufferSize);
        for (int i = 0; i < bufferSize; i++) {
          data[i] = i.toDouble();
        }
        buffer.write(data, bufferSize, dataType: BufferDataType.float32);

        buffer.destroy();

        if (cycle % memoryCheckInterval == 0) {
          await Future.delayed(Duration(milliseconds: 10));

          final memoryUsage = ProcessInfo.currentRss;
          memorySnapshots.add(memoryUsage);
          print(
            '  Cycle $cycle: Memory usage: ${(memoryUsage / 1024 / 1024).toStringAsFixed(2)} MB',
          );
        }
      }

      // Analyze memory growth
      if (memorySnapshots.length >= 2) {
        final initialMemory = memorySnapshots.first;
        final finalMemory = memorySnapshots.last;
        final totalGrowth = (finalMemory - initialMemory) / 1024 / 1024;

        print('Buffer memory analysis:');
        print(
          '  Initial: ${(initialMemory / 1024 / 1024).toStringAsFixed(2)} MB',
        );
        print('  Final: ${(finalMemory / 1024 / 1024).toStringAsFixed(2)} MB');
        print('  Growth: ${totalGrowth.toStringAsFixed(2)} MB');

        expect(
          totalGrowth,
          lessThan(200),
          reason:
              'Buffer memory leak detected: ${totalGrowth.toStringAsFixed(2)} MB growth',
        );
      }
    });

    test('granular memory leak detection', () async {
      MemoryTracker.start('Granular Test');

      // Test shader creation
      MemoryTracker.checkpoint('Before shader creation');
      final shader = gpu.createComputeShader();
      MemoryTracker.checkpoint('After shader creation');

      // Test kernel loading
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
      MemoryTracker.checkpoint('After kernel loading');

      // Test buffer creation
      final buffer = gpu.createBuffer(1024 * 4, BufferDataType.float32);
      MemoryTracker.checkpoint('After buffer creation');

      // Test buffer binding
      shader.setBuffer('data', buffer);
      MemoryTracker.checkpoint('After buffer binding');

      // Test dispatch
      await shader.dispatch(16, 1, 1);
      MemoryTracker.checkpoint('After first dispatch');

      // Test repeated dispatch (should reuse kernel)
      await shader.dispatch(16, 1, 1);
      MemoryTracker.checkpoint('After second dispatch');

      // Test binding change (should invalidate kernel)
      final buffer2 = gpu.createBuffer(1024 * 4, BufferDataType.float32);
      shader.setBuffer('data', buffer2);
      MemoryTracker.checkpoint('After binding change');

      await shader.dispatch(16, 1, 1);
      MemoryTracker.checkpoint('After dispatch with new binding');

      // Test cleanup
      buffer2.destroy();
      MemoryTracker.checkpoint('After buffer2 destroy');

      buffer.destroy();
      MemoryTracker.checkpoint('After buffer destroy');

      shader.destroy();
      MemoryTracker.checkpoint('After shader destroy');

      MemoryTracker.analyze();
    });

    test('comprehensive buffer memory analysis', () async {
      const bufferSize = 1024 * 1024; // 1MB per buffer
      const numBuffers = 50;

      MemoryTracker.start('Comprehensive Buffer Analysis');

      // Test 1: Buffer creation only
      final buffers = <Buffer>[];
      for (int i = 0; i < numBuffers; i++) {
        final buffer = gpu.createBuffer(bufferSize * 4, BufferDataType.float32);
        buffers.add(buffer);

        if ((i + 1) % 10 == 0) {
          MemoryTracker.checkpoint('Created ${i + 1} buffers');
        }
      }

      // Test 2: Data writing
      final testData = Float32List(bufferSize);
      for (int i = 0; i < bufferSize; i++) {
        testData[i] = i.toDouble();
      }

      for (int i = 0; i < numBuffers; i++) {
        buffers[i].write(
          testData,
          bufferSize,
          dataType: BufferDataType.float32,
        );

        if (i % 10 == 0) {
          MemoryTracker.checkpoint('Filled ${i + 1} buffers with data');
        }
      }

      // Test 3: Data reading
      for (int i = 0; i < numBuffers; i++) {
        final readback = Float32List(bufferSize);
        await buffers[i].read(
          readback,
          bufferSize,
          dataType: BufferDataType.float32,
        );

        if (i % 10 == 0) {
          MemoryTracker.checkpoint('Read from ${i + 1} buffers');
        }
      }

      // Test 4: Buffer destruction
      for (int i = 0; i < numBuffers; i++) {
        buffers[i].destroy();

        if (i % 10 == 0) {
          MemoryTracker.checkpoint('Destroyed ${i + 1} buffers');
        }
      }

      // Final memory check
      await Future.delayed(Duration(milliseconds: 50));
      MemoryTracker.checkpoint('After cleanup delay');

      MemoryTracker.analyze();

      // Check that memory returns to reasonable levels after cleanup
      final creationGrowth =
          MemoryTracker.checkpoints['Created 50 buffers']! / 1024 / 1024;
      final finalGrowth =
          MemoryTracker.checkpoints['After cleanup delay']! / 1024 / 1024;

      print('Buffer creation peak: ${creationGrowth.toStringAsFixed(2)} MB');
      print('Final growth after cleanup: ${finalGrowth.toStringAsFixed(2)} MB');

      // Memory should return to reasonable levels after destruction
      expect(
        finalGrowth,
        lessThan(
          creationGrowth * 0.5,
        ), // Should drop to less than half peak usage
        reason:
            'Buffers not properly cleaned up: final=${finalGrowth.toStringAsFixed(2)}MB, peak=${creationGrowth.toStringAsFixed(2)}MB',
      );
    });

    test('buffer lifecycle memory tracking', () async {
      const bufferSize = 512 * 1024; // 512KB per buffer
      const numCycles = 100;

      MemoryTracker.start('Buffer Lifecycle');

      for (int cycle = 0; cycle < numCycles; cycle++) {
        // Creation
        final buffer = gpu.createBuffer(bufferSize * 4, BufferDataType.float32);

        // Writing
        final data = Float32List.fromList(
          List.generate(bufferSize, (i) => (i + cycle).toDouble()),
        );
        buffer.write(data, bufferSize, dataType: BufferDataType.float32);

        // Reading
        final readback = Float32List(bufferSize);
        await buffer.read(
          readback,
          bufferSize,
          dataType: BufferDataType.float32,
        );

        // Destruction
        buffer.destroy();

        // Track every 10 cycles
        if (cycle % 10 == 0) {
          MemoryTracker.checkpoint('Completed $cycle cycles');
        }
      }

      MemoryTracker.checkpoint('All cycles complete');
      MemoryTracker.analyze();

      // Should show minimal growth over time
      final totalGrowth =
          MemoryTracker.checkpoints['All cycles complete']! / 1024 / 1024;
      expect(
        totalGrowth,
        lessThan(50),
        reason:
            'Excessive memory growth in buffer lifecycle: ${totalGrowth.toStringAsFixed(2)} MB',
      );
    });

    test('rapid buffer operations stress test', () async {
      const smallBufferSize = 1024; // 1KB buffers for rapid operations
      const numRapidOps = 1000;

      MemoryTracker.start('Rapid Buffer Operations');

      for (int op = 0; op < numRapidOps; op++) {
        // Create, write, read, destroy in rapid succession
        final buffer = gpu.createBuffer(
          smallBufferSize * 4,
          BufferDataType.float32,
        );

        final data = Float32List.fromList(
          List.generate(smallBufferSize, (i) => (i + op).toDouble()),
        );
        buffer.write(data, smallBufferSize, dataType: BufferDataType.float32);

        final readback = Float32List(smallBufferSize);
        await buffer.read(
          readback,
          smallBufferSize,
          dataType: BufferDataType.float32,
        );

        buffer.destroy();

        if (op % 100 == 0) {
          MemoryTracker.checkpoint('Rapid ops: $op');
        }
      }

      MemoryTracker.checkpoint('Rapid ops complete');
      MemoryTracker.analyze();

      final totalGrowth =
          MemoryTracker.checkpoints['Rapid ops complete']! / 1024 / 1024;
      expect(
        totalGrowth,
        lessThan(25),
        reason:
            'Memory leak in rapid buffer operations: ${totalGrowth.toStringAsFixed(2)} MB',
      );
    });
  });
}
