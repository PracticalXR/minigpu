// Create this as a Flutter integration test
// filepath: test_driver/gpu_spectrogram_integration_test.dart

import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('GPU Spectrogram Flutter Integration Tests', () {
    late Minigpu gpu;

    setUpAll(() async {
      gpu = Minigpu();
      await gpu.init();
    });

    testWidgets('large texture to Flutter Image conversion', (tester) async {
      // Test the exact Flutter UI integration that crashes
      final sizes = [
        {'name': 'Small', 'width': 400, 'height': 1024},
        {'name': 'Medium', 'width': 400, 'height': 2048},
        {'name': 'Large', 'width': 400, 'height': 4096},
        {'name': 'XLarge', 'width': 400, 'height': 8192}, // This should crash
      ];

      for (final size in sizes) {
        final name = size['name'] as String;
        final width = size['width'] as int;
        final height = size['height'] as int;
        final rgbaCount = width * height * 4;

        print('\n--- Testing $name Flutter Image: ${width}x${height} ---');

        try {
          // Create large tensor with GPU
          final tensor = await Tensor.create([rgbaCount], gpu: gpu);

          // Fill with GPU compute shader (like your spectrogram)
          final shader = gpu.createComputeShader();
          shader.loadKernelString('''
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            
            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let idx = gid.x + gid.y * ${(width + 7) ~/ 8 * 8}u;
                if (idx * 4u + 3u < arrayLength(&data)) {
                    data[idx * 4u] = 1.0;     // R
                    data[idx * 4u + 1u] = 0.5; // G
                    data[idx * 4u + 2u] = 0.0; // B
                    data[idx * 4u + 3u] = 1.0; // A
                }
            }
          ''');

          shader.setBuffer('data', tensor.buffer);
          await shader.dispatch((width + 7) ~/ 8, (height + 7) ~/ 8, 1);
          shader.destroy();

          // Read back data
          final pixels = await tensor.getData() as Float32List;

          // Convert to Uint8List (like your spectrogram)
          final uint8Pixels = Uint8List(rgbaCount);
          for (int i = 0; i < rgbaCount; i++) {
            uint8Pixels[i] = (pixels[i] * 255).clamp(0, 255).toInt();
          }

          // THIS IS WHERE THE CRASH LIKELY OCCURS
          print('  Creating Flutter Image from ${rgbaCount} pixels...');
          final completer = Completer<ui.Image>();
          ui.decodeImageFromPixels(
            uint8Pixels,
            width,
            height,
            ui.PixelFormat.rgba8888,
            completer.complete,
          );

          final image = await completer.future;
          print('   Image created: ${image.width}x${image.height}');

          // Try to use the image in a Flutter widget
          await tester.pumpWidget(
            MaterialApp(
              home: Scaffold(
                body: RawImage(
                  image: image,
                  width: 200,
                  height: 200,
                  fit: BoxFit.contain,
                ),
              ),
            ),
          );

          await tester.pump();
          print('   Widget rendered successfully');

          // Cleanup
          tensor.buffer.destroy();
          image.dispose();
        } catch (e) {
          print('   $name FAILED: $e');
          if (e.toString().contains('decodeImageFromPixels') ||
              e.toString().contains('Image') ||
              e.toString().contains('texture')) {
            print('  -> CRASH IS IN FLUTTER IMAGE HANDLING!');
          }
          rethrow;
        }
      }
    });

    testWidgets('concurrent spectrogram simulation', (tester) async {
      // Simulate the concurrent processing that happens in your real app
      print('\n--- Testing Concurrent Spectrogram Processing ---');

      const numConcurrent = 3; // Process 3 spectrograms simultaneously
      const width = 400;
      const height = 4096; // Large enough to potentially cause issues

      final futures = <Future<ui.Image>>[];

      for (int i = 0; i < numConcurrent; i++) {
        futures.add(_createSpectrogramImage(gpu, width, height, i));
      }

      try {
        final images = await Future.wait(futures);
        print(' All ${numConcurrent} concurrent spectrograms completed');

        // Try to render them all at once
        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              body: Row(
                children: images
                    .map(
                      (img) => Expanded(
                        child: RawImage(image: img, fit: BoxFit.contain),
                      ),
                    )
                    .toList(),
              ),
            ),
          ),
        );

        await tester.pump();
        print(' All concurrent images rendered successfully');

        // Cleanup
        for (final img in images) {
          img.dispose();
        }
      } catch (e) {
        print(' Concurrent processing failed: $e');
        rethrow;
      }
    });
  });
}

Future<ui.Image> _createSpectrogramImage(
  Minigpu gpu,
  int width,
  int height,
  int seed,
) async {
  final rgbaCount = width * height * 4;

  // Create tensor
  final tensor = await Tensor.create([rgbaCount], gpu: gpu);

  // Process with shader
  final shader = gpu.createComputeShader();
  shader.loadKernelString('''
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;
    
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x + gid.y * ${(width + 7) ~/ 8 * 8}u;
        if (idx * 4u + 3u < arrayLength(&data)) {
            let color = f32($seed) / 10.0;
            data[idx * 4u] = color;       // R
            data[idx * 4u + 1u] = color;   // G
            data[idx * 4u + 2u] = color;   // B
            data[idx * 4u + 3u] = 1.0;     // A
        }
    }
  ''');

  shader.setBuffer('data', tensor.buffer);
  await shader.dispatch((width + 7) ~/ 8, (height + 7) ~/ 8, 1);
  shader.destroy();

  // Read and convert
  final pixels = await tensor.getData() as Float32List;
  final uint8Pixels = Uint8List(rgbaCount);
  for (int i = 0; i < rgbaCount; i++) {
    uint8Pixels[i] = (pixels[i] * 255).clamp(0, 255).toInt();
  }

  // Create image
  final completer = Completer<ui.Image>();
  ui.decodeImageFromPixels(
    uint8Pixels,
    width,
    height,
    ui.PixelFormat.rgba8888,
    completer.complete,
  );

  final image = await completer.future;
  tensor.buffer.destroy();

  return image;
}
