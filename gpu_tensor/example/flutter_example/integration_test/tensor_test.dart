import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:math' as math;
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

    testWidgets('exact spectrogram widget usage reproduction', (tester) async {
      print('\n--- Testing Exact Widget Usage Pattern ---');

      // Simulate the exact timer-driven, widget-integrated usage
      const iterations = 20; // Enough to trigger the issue
      final spectrograms = <ui.Image>[];

      // Create mock streams like your real app
      final mockStreams = [
        MockAudioStream('stream1', 4096), // 8192 FFT size
        MockAudioStream('stream2', 4096),
      ];

      try {
        for (int i = 0; i < iterations; i++) {
          print('Widget cycle ${i + 1}/$iterations...');

          // Simulate timer update with new FFT data
          for (final stream in mockStreams) {
            stream.updateWithNewFFT(i);
          }

          // This mimics your _updateSpectrogram() method
          final image = await _createSpectrogramWithExactPattern(
            mockStreams,
            tester,
            i,
          );

          spectrograms.add(image);

          // Simulate widget updates with setState timing
          await tester.pumpWidget(
            MaterialApp(
              home: Scaffold(
                body: RawImage(image: image, fit: BoxFit.contain),
              ),
            ),
          );

          // This rapid pump/update cycle might trigger the crash
          await tester.pump();
          await tester.pump(Duration(milliseconds: 10));

          // Simulate your resource management pattern
          if (i > 0) {
            // Dispose previous image (like your setState does)
            spectrograms[i - 1].dispose();
          }

          // Force brief delay like your timer
          await Future.delayed(Duration(milliseconds: 100));
        }

        print(' All widget cycles completed - crash was not in basic pattern');
      } catch (e) {
        print(' CRASH in widget cycle: $e');
        print('  This reproduces your exact crash scenario!');
        rethrow;
      } finally {
        // Cleanup remaining images
        for (final img in spectrograms) {
          img.dispose();
        }
      }
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
  testWidgets('memory leak simulation - repeated spectrogram creation', (
    tester,
  ) async {
    print(
      '\n--- Testing Repeated Spectrogram Creation (Memory Leak Detection) ---',
    );

    const width = 400;
    const height = 4096; // Large enough to cause issues if leaking
    const iterations = 50; // Simulate "running for a while"

    final images = <ui.Image>[];

    try {
      for (int i = 0; i < iterations; i++) {
        print('Iteration ${i + 1}/$iterations...');

        // Create spectrogram exactly like your real code
        final image = await _createRealisticSpectrogramImage(width, height, i);

        // Store images to simulate your app keeping references
        images.add(image);

        // Occasionally render the images (like your real app would)
        if (i % 10 == 0) {
          await tester.pumpWidget(
            MaterialApp(
              home: Scaffold(
                body: RawImage(image: image, fit: BoxFit.contain),
              ),
            ),
          );
          await tester.pump();
          print('  Rendered iteration $i');
        }

        // This is where the crash should eventually happen
        if (i % 5 == 0) {
          print('  Memory checkpoint at iteration $i');
          // Force a brief pause to simulate real app timing
          await Future.delayed(Duration(milliseconds: 10));
        }
      }

      print(' All $iterations iterations completed without crash');
      print('  If your real app crashes, the issue might be:');
      print('  - Different cleanup timing');
      print('  - Audio thread interference');
      print('  - Real FFT data causing issues');
    } catch (e) {
      print(' CRASH AFTER ${images.length} iterations: $e');
      print('  This confirms a memory leak or resource accumulation issue!');
      rethrow;
    } finally {
      // Cleanup all images
      for (final img in images) {
        img.dispose();
      }
      print('Disposed ${images.length} images');
    }
  });

  testWidgets('aggressive resource cycling test', (tester) async {
    print('\n--- Testing Aggressive Resource Cycling ---');

    const cycles = 100;
    const width = 400;
    const height = 8192; // Maximum size that crashes in your app

    for (int cycle = 0; cycle < cycles; cycle++) {
      print('Cycle ${cycle + 1}/$cycles...');

      try {
        // Create and immediately destroy resources (no accumulation)
        final image = await _createRealisticSpectrogramImage(
          width,
          height,
          cycle,
        );

        // Use the image briefly
        await tester.pumpWidget(
          MaterialApp(
            home: Scaffold(
              body: RawImage(image: image, fit: BoxFit.contain),
            ),
          ),
        );
        await tester.pump();

        // Immediately dispose (like proper cleanup should do)
        image.dispose();

        // Simulate rapid cycling like real audio processing
        if (cycle % 10 == 0) {
          print('  Checkpoint: ${cycle + 1} cycles completed');
        }
      } catch (e) {
        print(' CRASH at cycle ${cycle + 1}: $e');
        print('  This suggests the issue is not simple accumulation');
        rethrow;
      }
    }

    print(' All $cycles aggressive cycles completed');
  });
}

Future<ui.Image> _createRealisticSpectrogramImage(
  int width,
  int height,
  int iteration,
) async {
  final tensorTimeSlices = 200;
  final tensorFreqBins = height;
  final inputElements = tensorTimeSlices * tensorFreqBins;
  final rgbaComponentCount = width * height * 4;

  // Create realistic FFT-like input data
  final inputData = Float32List(inputElements);
  for (int i = 0; i < inputElements; i++) {
    // Simulate real FFT data with varying patterns per iteration
    final freq = i % tensorFreqBins;
    final time = i ~/ tensorFreqBins;
    final magnitude =
        math.sin((freq + iteration * 10) * 0.01) *
            math.cos((time + iteration * 5) * 0.02) *
            10.0 +
        iteration * 0.1;
    inputData[i] = magnitude.abs();
  }

  final inputTensor = await Tensor.create([
    tensorTimeSlices,
    tensorFreqBins,
  ], data: inputData);
  final outputTensor = await Tensor.create([rgbaComponentCount]);

  // Use the EXACT shader from your spectrogram
  final shader = outputTensor.gpu.createComputeShader();
  final shaderTemplate =
      '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const TEXTURE_WIDTH: u32 = ${width}u;
const TEXTURE_HEIGHT: u32 = ${height}u;
const TENSOR_TIME_SLICES: u32 = ${tensorTimeSlices}u;
const TENSOR_FREQ_BINS: u32 = ${tensorFreqBins}u;
const MAX_MAGNITUDE: f32 = 15.0;
const MIN_FREQ: f32 = 20.0;
const MAX_FREQ: f32 = 5000.0;
const SAMPLE_RATE: f32 = 44100.0;
const NYQUIST_FREQ: f32 = SAMPLE_RATE / 2.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    
    if (x >= TEXTURE_WIDTH || y >= TEXTURE_HEIGHT) {
        return;
    }
    
    let outputIndex: u32 = (y * TEXTURE_WIDTH + x) * 4u;
    
    if (outputIndex + 3u >= arrayLength(&output)) {
        return;
    }
    
    let timeSlice: u32 = x * TENSOR_TIME_SLICES / TEXTURE_WIDTH;
    let freqRatio: f32 = f32(TEXTURE_HEIGHT - 1u - y) / f32(TEXTURE_HEIGHT - 1u);
    let targetFreq: f32 = MIN_FREQ + freqRatio * (MAX_FREQ - MIN_FREQ);
    let freqBin: u32 = u32(targetFreq * f32(TENSOR_FREQ_BINS) / NYQUIST_FREQ);
    
    var magnitude: f32 = 0.0;
    
    if (timeSlice < TENSOR_TIME_SLICES && freqBin < TENSOR_FREQ_BINS) {
        let inputIndex: u32 = timeSlice * TENSOR_FREQ_BINS + freqBin;
        
        if (inputIndex < arrayLength(&input)) {
            magnitude = input[inputIndex];
        }
    }
    
    let normalizedMag: f32 = clamp(magnitude / MAX_MAGNITUDE, 0.0, 1.0);
    
    var color: vec4<f32>;
    if (normalizedMag <= 0.001) {
        color = vec4<f32>(0.0, 0.0, 0.1, 1.0);
    } else {
        let logMag: f32 = clamp(log(normalizedMag * 9.0 + 1.0) / log(10.0), 0.0, 1.0);
        
        if (logMag < 0.5) {
            let t: f32 = logMag * 2.0;
            color = vec4<f32>(0.0, t, 1.0, 1.0);
        } else {
            let t: f32 = (logMag - 0.5) * 2.0;
            color = vec4<f32>(t, 1.0, 1.0 - t, 1.0);
        }
    }
    
    output[outputIndex] = color.r;
    output[outputIndex + 1u] = color.g;
    output[outputIndex + 2u] = color.b;
    output[outputIndex + 3u] = color.a;
}
''';

  shader.loadKernelString(shaderTemplate);
  shader.setBuffer('input', inputTensor.buffer);
  shader.setBuffer('output', outputTensor.buffer);

  final workgroupsX = (width + 7) ~/ 8;
  final workgroupsY = (height + 7) ~/ 8;

  await shader.dispatch(workgroupsX, workgroupsY, 1);

  // Read back and convert
  final pixels = await outputTensor.getData() as Float32List;
  final uint8Pixels = Uint8List(rgbaComponentCount);
  for (int i = 0; i < rgbaComponentCount; i++) {
    uint8Pixels[i] = (pixels[i] * 255).clamp(0, 255).toInt();
  }

  // Create Flutter image
  final completer = Completer<ui.Image>();
  ui.decodeImageFromPixels(
    uint8Pixels,
    width,
    height,
    ui.PixelFormat.rgba8888,
    completer.complete,
  );

  final image = await completer.future;

  // CRITICAL: Cleanup resources immediately
  shader.destroy();
  inputTensor.buffer.destroy();
  outputTensor.buffer.destroy();

  return image;
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

// Mock audio stream that generates realistic FFT data
class MockAudioStream {
  final String id;
  final int fftSize;
  List<double> fftData = [];

  MockAudioStream(this.id, this.fftSize);

  void updateWithNewFFT(int iteration) {
    // Generate FFT data that changes over time (like real audio)
    fftData = List.generate(fftSize, (i) {
      return math.sin(i * 0.01 + iteration * 0.1) *
              math.cos(i * 0.001 + iteration * 0.05) *
              10.0 +
          iteration * 0.1;
    });
  }
}

// Recreate your exact spectrogram creation pattern
Future<ui.Image> _createSpectrogramWithExactPattern(
  List<MockAudioStream> streams,
  WidgetTester widget,
  int iteration,
) async {
  const maxTimeSlices = 200;
  const sampleRate = 44100.0;
  const minFreq = 20.0;
  const maxFreq = 5000.0;

  final fftSize = streams.first.fftSize;

  // Create spectrogram data exactly like your SpectrogramData class
  final spectrogramData = <List<double>>[];

  // Build rolling buffer of recent FFT data (like your _createRollingBufferData)
  for (int timeSlice = 0; timeSlice < maxTimeSlices; timeSlice++) {
    if (timeSlice < iteration + 1) {
      // Use real FFT data for recent slices
      spectrogramData.add(List.from(streams.first.fftData));
    } else {
      // Pad with zeros for future slices
      spectrogramData.add(List.filled(fftSize, 0.0));
    }
  }

  // Flatten to match your tensor creation
  final bufferData = <double>[];
  for (final slice in spectrogramData) {
    bufferData.addAll(slice);
  }

  // THE CRITICAL PATTERN: destroy old tensor, create new one with explicit GPU
  // This matches your _ensureGpuTensor method exactly
  print('  Creating tensor with explicit GPU reference...');
  final inputTensor = await Tensor.create([
    maxTimeSlices,
    fftSize,
  ], data: Float32List.fromList(bufferData));

  // Create output tensor exactly like your code
  final outputWidth = 400;
  final outputHeight = fftSize;
  final rgbaComponentCount = outputWidth * outputHeight * 4;

  print('  Creating output tensor with explicit GPU reference...');
  final outputTensor = await Tensor.create([rgbaComponentCount]);

  // Use your exact shader template
  final shader = outputTensor.gpu
      .createComputeShader(); // Note: using global gpu, not tensor.gpu

  final shaderTemplate =
      '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const TEXTURE_WIDTH: u32 = ${outputWidth}u;
const TEXTURE_HEIGHT: u32 = ${outputHeight}u;
const TENSOR_TIME_SLICES: u32 = ${maxTimeSlices}u;
const TENSOR_FREQ_BINS: u32 = ${fftSize}u;
const MAX_MAGNITUDE: f32 = 15.0;
const MIN_FREQ: f32 = ${minFreq};
const MAX_FREQ: f32 = ${maxFreq};
const SAMPLE_RATE: f32 = ${sampleRate};
const NYQUIST_FREQ: f32 = SAMPLE_RATE / 2.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    
    if (x >= TEXTURE_WIDTH || y >= TEXTURE_HEIGHT) {
        return;
    }
    
    let outputIndex: u32 = (y * TEXTURE_WIDTH + x) * 4u;
    
    if (outputIndex + 3u >= arrayLength(&output)) {
        return;
    }
    
    let timeSlice: u32 = x * TENSOR_TIME_SLICES / TEXTURE_WIDTH;
    let freqRatio: f32 = f32(TEXTURE_HEIGHT - 1u - y) / f32(TEXTURE_HEIGHT - 1u);
    let targetFreq: f32 = MIN_FREQ + freqRatio * (MAX_FREQ - MIN_FREQ);
    let freqBin: u32 = u32(targetFreq * f32(TENSOR_FREQ_BINS) / NYQUIST_FREQ);
    
    var magnitude: f32 = 0.0;
    
    if (timeSlice < TENSOR_TIME_SLICES && freqBin < TENSOR_FREQ_BINS) {
        let inputIndex: u32 = timeSlice * TENSOR_FREQ_BINS + freqBin;
        
        if (inputIndex < arrayLength(&input)) {
            magnitude = input[inputIndex];
        }
    }
    
    let normalizedMag: f32 = clamp(magnitude / 15.0, 0.0, 1.0);
    
    var color: vec4<f32>;
    if (normalizedMag <= 0.001) {
        color = vec4<f32>(0.0, 0.0, 0.1, 1.0);
    } else {
        let logMag: f32 = clamp(log(normalizedMag * 9.0 + 1.0) / log(10.0), 0.0, 1.0);
        
        if (logMag < 0.5) {
            let t: f32 = logMag * 2.0;
            color = vec4<f32>(0.0, t, 1.0, 1.0);
        } else {
            let t: f32 = (logMag - 0.5) * 2.0;
            color = vec4<f32>(t, 1.0, 1.0 - t, 1.0);
        }
    }
    
    output[outputIndex] = color.r;
    output[outputIndex + 1u] = color.g;
    output[outputIndex + 2u] = color.b;
    output[outputIndex + 3u] = color.a;
}
''';

  shader.loadKernelString(shaderTemplate);

  // Your exact buffer binding pattern
  shader.setBuffer('input', inputTensor.buffer);
  shader.setBuffer('output', outputTensor.buffer);

  final workgroupsX = (outputWidth + 7) ~/ 8;
  final workgroupsY = (outputHeight + 7) ~/ 8;

  // THE DISPATCH THAT SHOULD CRASH
  await shader.dispatch(workgroupsX, workgroupsY, 1);

  // Read back data
  final pixels = await outputTensor.getData() as Float32List;

  // Convert to image exactly like your code
  final uint8Pixels = Uint8List(rgbaComponentCount);
  for (int i = 0; i < rgbaComponentCount; i++) {
    uint8Pixels[i] = (pixels[i] * 255).clamp(0, 255).toInt();
  }

  final completer = Completer<ui.Image>();
  ui.decodeImageFromPixels(
    uint8Pixels,
    outputWidth,
    outputHeight,
    ui.PixelFormat.rgba8888,
    completer.complete,
  );

  final image = await completer.future;

  // Your exact cleanup pattern
  shader.destroy();
  inputTensor.buffer.destroy();
  outputTensor.buffer.destroy();

  return image;
}
