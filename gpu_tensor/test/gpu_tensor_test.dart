import 'dart:typed_data';
import 'dart:async';
import 'dart:math' as math;
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';

Future<void> main() async {
  group('GPU Tensor Stress Tests', () {
    test('Rapid FFT operations - single tensor', () async {
      final tensor = await Tensor.create([1024]);
      final testData = Float32List(1024);

      // Generate test signal
      for (int i = 0; i < 1024; i++) {
        testData[i] = math.sin(2 * math.pi * i / 64).toFloat();
      }

      tensor.buffer.setData(
        testData,
        testData.length,
        dataType: BufferDataType.float32,
      );

      // Rapid FFT operations
      for (int i = 0; i < 100; i++) {
        try {
          final fftResult = await tensor.fft();
          final data = await fftResult.getData();

          // Immediately dispose result
          fftResult.destroy();

          // Small delay to prevent overwhelming GPU
          if (i % 10 == 0) {
            await Future.delayed(Duration(microseconds: 100));
          }
        } catch (e) {
          print('FFT failed at iteration $i: $e');
          fail('FFT operation failed at iteration $i');
        }
      }

      tensor.destroy();
    });

    test('Multiple tensors rapid FFT - simulating audio streams', () async {
      const int numStreams = 4;
      const int fftSize = 1024;
      final tensors = <Tensor>[];
      final testData = Float32List(fftSize);

      // Create test signal
      for (int i = 0; i < fftSize; i++) {
        testData[i] = math.sin(2 * math.pi * i / 64).toFloat();
      }

      // Create multiple tensors (simulating multiple audio streams)
      for (int i = 0; i < numStreams; i++) {
        final tensor = await Tensor.create([fftSize]);
        tensor.buffer.setData(
          testData,
          testData.length,
          dataType: BufferDataType.float32,
        );
        tensors.add(tensor);
      }

      // Rapid processing of all streams
      for (int iteration = 0; iteration < 50; iteration++) {
        final futures = <Future>[];

        for (int streamIndex = 0; streamIndex < numStreams; streamIndex++) {
          futures.add(() async {
            try {
              final fftResult = await tensors[streamIndex].fft();
              final data = await fftResult.getData();
              fftResult.destroy();
            } catch (e) {
              print('Stream $streamIndex failed at iteration $iteration: $e');
              throw e;
            }
          }());
        }

        await Future.wait(futures);

        // Simulate different processing rates
        if (iteration % 5 == 0) {
          await Future.delayed(Duration(milliseconds: 10));
        }
      }

      // Cleanup
      for (final tensor in tensors) {
        tensor.destroy();
      }
    });

    test('Memory pressure test - large FFT operations', () async {
      const int fftSize = 4096; // Larger FFT
      final tensor = await Tensor.create([fftSize]);
      final testData = Float32List(fftSize);

      for (int i = 0; i < fftSize; i++) {
        testData[i] = math.sin(2 * math.pi * i / 256).toFloat();
      }

      tensor.buffer.setData(
        testData,
        testData.length,
        dataType: BufferDataType.float32,
      );

      // Test with minimal delay (similar to your 50ms issue)
      final timer = Timer.periodic(Duration(milliseconds: 20), (timer) async {
        try {
          final fftResult = await tensor.fft();
          final data = await fftResult.getData();
          fftResult.destroy();
        } catch (e) {
          print('Timer FFT failed: $e');
          timer.cancel();
          fail('Timer-based FFT failed');
        }
      });

      // Run for 2 seconds
      await Future.delayed(Duration(seconds: 2));
      timer.cancel();

      tensor.destroy();
    });

    test('Resource cleanup timing test', () async {
      final tensor = await Tensor.create([1024]);
      final testData = Float32List(1024);

      for (int i = 0; i < 1024; i++) {
        testData[i] = math.sin(2 * math.pi * i / 64).toFloat();
      }

      tensor.buffer.setData(
        testData,
        testData.length,
        dataType: BufferDataType.float32,
      );

      // Test immediate disposal vs delayed disposal
      for (int i = 0; i < 50; i++) {
        final fftResult = await tensor.fft();
        final data = await fftResult.getData();

        if (i % 2 == 0) {
          // Immediate disposal
          fftResult.destroy();
        } else {
          // Delayed disposal (simulating real-world scenario)
          Future.delayed(Duration(milliseconds: 1), () {
            fftResult.destroy();
          });
        }
      }

      // Wait for delayed disposals
      await Future.delayed(Duration(milliseconds: 100));
      tensor.destroy();
    });
  });
}
