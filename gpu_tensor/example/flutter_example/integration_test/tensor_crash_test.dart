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

  late Minigpu gpu;

  setUpAll(() async {
    gpu = Minigpu();
    await gpu.init();
  });

  testWidgets('no flutter sync tensor operations', (tester) async {
    print('\n--- Testing No Flutter Sync Tensor Operations ---');

    late Minigpu gpu;
    gpu = Minigpu();
    await gpu.init();

    try {
      await tester.pumpWidget(MaterialApp(home: Scaffold(body: Container())));

      final tensors = <String, Tensor>{};

      for (int i = 0; i < 3; i++) {
        print('No sync iteration ${i + 1}...');

        // REMOVE ALL FLUTTER SYNCHRONIZATION:
        // await tester.pumpAndSettle();           // REMOVED
        // await WidgetsBinding.instance.endOfFrame; // REMOVED - THIS CAUSES HANG
        // await Future.delayed(Duration(milliseconds: 50)); // REMOVED

        print('  Starting tensor operations immediately');

        // Just do tensor operations with backend queue flush
        final oldTensor = tensors['test'];
        if (oldTensor != null) {
          print('  Destroying old tensor...');
          oldTensor.buffer
              .destroy(); // Your backend queue flush should handle this
        }

        final fftSize = 2048 * (i + 1);
        final bufferData = List.generate(
          200 * fftSize,
          (i) => math.sin(i * 0.001) * 10.0,
        );

        print('  Creating new tensor...');
        final newTensor = await Tensor.create(
          [200, fftSize],
          gpu: gpu,
          data: Float32List.fromList(bufferData),
        );
        tensors['test'] = newTensor;

        final image = await _createSpectrogramFromTensor(newTensor, fftSize, i);
        image.dispose();

        print('No sync iteration ${i + 1} completed');
      }

      print('No sync operations completed without hang');
    } catch (e) {
      print('CRASH in no sync operations: $e');
      rethrow;
    }
  });

  testWidgets('synchronized flutter tensor operations', (tester) async {
    print('\n--- Testing Synchronized Flutter Tensor Operations ---');

    late Minigpu gpu;
    gpu = Minigpu();
    await gpu.init();

    try {
      await tester.pumpWidget(MaterialApp(home: Scaffold(body: Container())));

      final tensors = <String, Tensor>{};

      for (int i = 0; i < 3; i++) {
        print('Synchronized iteration ${i + 1}...');

        // CRITICAL: Wait for Flutter to finish ALL operations
        await tester.pumpAndSettle();
        await WidgetsBinding.instance.endOfFrame;
        await Future.delayed(Duration(milliseconds: 50)); // Extra safety margin

        print('  Flutter fully settled and GPU idle');

        // Now safe to do tensor operations
        final oldTensor = tensors['test'];
        if (oldTensor != null) {
          print('  Destroying old tensor after Flutter sync...');
          oldTensor.buffer.destroy();
          await Future.delayed(
            Duration(milliseconds: 10),
          ); // Wait after destroy
        }

        final fftSize = 2048 * (i + 1);
        final bufferData = List.generate(
          200 * fftSize,
          (i) => math.sin(i * 0.001) * 10.0,
        );

        print('  Creating new tensor...');
        final newTensor = await Tensor.create(
          [200, fftSize],
          gpu: gpu,
          data: Float32List.fromList(bufferData),
        );
        tensors['test'] = newTensor;

        final image = await _createSpectrogramFromTensor(newTensor, fftSize, i);
        image.dispose();

        print('   Synchronized iteration ${i + 1} completed');
      }

      print(' Synchronized operations completed without crash');
    } catch (e) {
      print(' CRASH in synchronized operations: $e');
      rethrow;
    }
  });

  testWidgets('FFT window size switch reproduction', (tester) async {
    print('\n--- Testing FFT Window Size Switch (The Real Crash) ---');

    late Minigpu gpu;
    gpu = Minigpu();
    await gpu.init();

    // Simulate your exact FFT window switch scenario
    final fftSizes = [512, 1024, 2048, 4096, 8192]; // Your progression
    final streamId = 'test_stream';

    // Mock the persistent data and tensor maps like your real code
    final persistentSpectrogramData = <String, MockSpectrogramData>{};
    final spectrogramTensors = <String, Tensor>{};

    try {
      for (int i = 0; i < fftSizes.length; i++) {
        final fftSize = fftSizes[i];
        final sizeBytes = (400 * fftSize * 4 * 4) / (1024 * 1024); // RGBA MB

        print(
          '\n--- Switching to FFT size: $fftSize (${sizeBytes.toStringAsFixed(1)}MB) ---',
        );

        // Simulate your _persistentSpectrogramData update when FFT size changes
        persistentSpectrogramData[streamId] = MockSpectrogramData(
          maxTimeSlices: 200,
          frequencyBins: fftSize,
          sampleRate: 44100,
        );

        // Add some realistic FFT data
        final mockData = persistentSpectrogramData[streamId]!;
        for (int slice = 0; slice < 5; slice++) {
          final fftData = List.generate(
            fftSize,
            (i) => math.sin(i * 0.01 + slice * 0.1) * 10.0 + i * 0.001,
          );
          mockData.addFFTData(fftData);
        }

        // THIS IS WHERE THE CRASH HAPPENS - exact reproduction of your _ensureGpuTensor
        print('  Destroying old tensor...');
        final oldTensor = spectrogramTensors[streamId];
        if (oldTensor != null) {
          // This might destroy a tensor that's still being used by GPU!
          oldTensor.buffer.destroy();
          print('    Old tensor destroyed');
        }

        print('  Creating new tensor for FFT size $fftSize...');
        final bufferData = _createRollingBufferData(mockData, 200);

        // This is your exact tensor creation pattern
        final newTensor = await Tensor.create(
          [200, fftSize],
          gpu: gpu, // Explicit GPU reference like your code
          data: Float32List.fromList(bufferData),
        );
        spectrogramTensors[streamId] = newTensor;
        print('    New tensor created');

        // Now try to use it immediately (like your real code does)
        print('  Creating spectrogram texture...');
        final image = await _createSpectrogramFromTensor(
          newTensor,
          fftSize,
          i, // iteration for variety
        );

        print('     Texture created successfully');

        // Brief delay to simulate real app timing
        await Future.delayed(Duration(milliseconds: 50));

        // Cleanup image
        image.dispose();
      }

      print('\n All FFT size switches completed successfully');
      print('  If this test passes but your real app crashes, the issue is:');
      print('  - GPU command timing in real audio processing');
      print('  - Multiple streams switching simultaneously');
      print('  - Audio thread interference with GPU operations');
    } catch (e) {
      print('\n CRASH during FFT size switch: $e');
      print('  This reproduces your exact crash scenario!');

      // Check if it's the specific D3D12 command list error
      if (e.toString().contains('CommandList') ||
          e.toString().contains('mD3d12CommandList') ||
          e.toString().contains('dawn') ||
          e.toString().contains('d3d12')) {
        print(
          '  -> CONFIRMED: This is the D3D12 command list validation error!',
        );
        print(
          '  -> CAUSE: Destroying tensors while GPU commands are still active',
        );
      }

      rethrow;
    } finally {
      // Cleanup remaining tensors
      for (final tensor in spectrogramTensors.values) {
        try {
          tensor.buffer.destroy();
        } catch (e) {
          print('Cleanup error: $e');
        }
      }
    }
  });

  testWidgets('flutter gpu context isolation test', (tester) async {
    print('\n--- Testing Flutter GPU Context Isolation ---');

    late Minigpu gpu;
    gpu = Minigpu();
    await gpu.init();

    try {
      // Force Flutter to render something GPU-intensive
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Colors.red,
                    Colors.blue,
                    Colors.green,
                    Colors.yellow,
                  ],
                  stops: [0.0, 0.33, 0.66, 1.0],
                ),
              ),
              child: CustomPaint(painter: _TestPainter(), size: Size(400, 400)),
            ),
          ),
        ),
      );

      // Let Flutter finish its rendering
      await tester.pumpAndSettle();
      print('Flutter rendering completed');

      // Now try tensor operations while Flutter is active
      print('Starting tensor operations with active Flutter context...');

      final fftSizes = [2048, 4096, 8192];
      final tensors = <String, Tensor>{};

      for (int i = 0; i < fftSizes.length; i++) {
        final fftSize = fftSizes[i];
        print('  Creating tensor with FFT size $fftSize...');

        // Destroy old tensor while Flutter might be using GPU
        final oldTensor = tensors['test'];
        if (oldTensor != null) {
          print('    Destroying old tensor during Flutter rendering...');
          oldTensor.buffer.destroy(); // This should crash in Flutter context
        }

        // Create new tensor
        final bufferData = List.generate(
          200 * fftSize,
          (i) => math.sin(i * 0.001) * 10.0,
        );
        final newTensor = await Tensor.create(
          [200, fftSize],
          gpu: gpu,
          data: Float32List.fromList(bufferData),
        );
        tensors['test'] = newTensor;

        // Use tensor immediately
        final image = await _createSpectrogramFromTensor(newTensor, fftSize, i);

        // Force Flutter to render again while tensor operation completes
        await tester.pump();

        image.dispose();

        print('     FFT size $fftSize completed');
      }

      print(' Flutter GPU context test completed');
    } catch (e) {
      print(' CRASH in Flutter GPU context test: $e');

      if (e.toString().contains('CommandList') ||
          e.toString().contains('mD3d12CommandList')) {
        print('  -> CONFIRMED: Flutter GPU context interference!');
        print(
          '  -> Flutter and tensor operations are sharing GPU resources unsafely',
        );
      }

      rethrow;
    }
  });

  // Add a test that synchronizes with Flutter's rendering
  testWidgets('synchronized flutter tensor operations', (tester) async {
    print('\n--- Testing Synchronized Flutter Tensor Operations ---');

    late Minigpu gpu;
    gpu = Minigpu();
    await gpu.init();

    try {
      await tester.pumpWidget(MaterialApp(home: Scaffold(body: Container())));

      final tensors = <String, Tensor>{};

      for (int i = 0; i < 3; i++) {
        print('Synchronized iteration ${i + 1}...');

        // Wait for Flutter to finish any pending operations
        await tester.pumpAndSettle();
        print('  Flutter settled');

        // Brief delay to ensure GPU is idle
        await Future.delayed(Duration(milliseconds: 100));

        // Now do tensor operations
        final oldTensor = tensors['test'];
        if (oldTensor != null) {
          print('  Destroying old tensor after Flutter sync...');
          oldTensor.buffer.destroy();
        }

        final fftSize = 2048 * (i + 1);
        final bufferData = List.generate(
          200 * fftSize,
          (i) => math.sin(i * 0.001) * 10.0,
        );
        final newTensor = await Tensor.create(
          [200, fftSize],
          gpu: gpu,
          data: Float32List.fromList(bufferData),
        );
        tensors['test'] = newTensor;

        final image = await _createSpectrogramFromTensor(newTensor, fftSize, i);
        image.dispose();

        print('   Synchronized iteration ${i + 1} completed');
      }

      print(' Synchronized operations completed without crash');
    } catch (e) {
      print(' CRASH in synchronized operations: $e');
      rethrow;
    }
  });

  // Test that forces GPU context conflicts
  testWidgets('force gpu context conflict', (tester) async {
    print('\n--- Forcing GPU Context Conflict ---');

    late Minigpu gpu;
    gpu = Minigpu();
    await gpu.init();

    try {
      // Create a widget that continuously updates (forces Flutter GPU usage)
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: AnimatedBuilder(
              animation: AlwaysStoppedAnimation(0.5),
              builder: (context, child) {
                return Container(
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        Color.lerp(Colors.red, Colors.blue, 0.5)!,
                        Color.lerp(Colors.green, Colors.yellow, 0.5)!,
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ),
      );

      // Start continuous Flutter rendering
      final renderFuture = _continuousFlutterRender(tester);

      // Simultaneously do aggressive tensor operations
      final tensorFuture = _aggressiveTensorOperationsWithFlutter(gpu);

      // Wait for either to complete (or crash)
      await Future.any([renderFuture, tensorFuture]);

      print(' GPU context conflict test completed without crash - unexpected!');
    } catch (e) {
      print(' CRASH in GPU context conflict: $e');

      if (e.toString().contains('CommandList') ||
          e.toString().contains('mD3d12CommandList')) {
        print(
          '  -> CONFIRMED: GPU context conflict between Flutter and tensors!',
        );
      }

      rethrow;
    }
  });

  testWidgets('enhanced concurrent FFT size switches (guaranteed crash)', (
    tester,
  ) async {
    print('\n--- Testing Enhanced Concurrent FFT Size Switches ---');

    late Minigpu gpu;
    gpu = Minigpu();
    await gpu.init();

    // More aggressive concurrent scenario
    const numStreams = 5; // More streams = more contention
    const switchIterations = 20; // More iterations = higher crash probability
    final fftSizes = [1024, 2048, 4096, 8192]; // Include the largest size

    final allTensors = <String, Tensor>{};
    final activeFutures = <Future<void>>[];

    try {
      for (int iteration = 0; iteration < switchIterations; iteration++) {
        print('Iteration ${iteration + 1}/$switchIterations...');

        activeFutures.forEach((future) {
          future.whenComplete(() {
            activeFutures.remove(future);
          });
        });

        // Launch ALL streams concurrently with NO delays
        for (int streamIdx = 0; streamIdx < numStreams; streamIdx++) {
          final streamId = 'stream_$streamIdx';
          final fftSize = fftSizes[(iteration + streamIdx) % fftSizes.length];

          // Don't await - let them all run truly concurrently
          final future = _aggressiveSwitchStreamFFTSize(
            gpu,
            streamId,
            fftSize,
            allTensors,
            iteration,
          );
          activeFutures.add(future);
        }

        // Add more concurrent operations in the same iteration
        for (int extraIdx = 0; extraIdx < 2; extraIdx++) {
          final streamId = 'extra_${iteration}_$extraIdx';
          final fftSize = fftSizes[extraIdx % fftSizes.length];

          final future = _aggressiveSwitchStreamFFTSize(
            gpu,
            streamId,
            fftSize,
            allTensors,
            iteration,
          );
          activeFutures.add(future);
        }

        // Wait for this batch with a timeout
        try {
          await Future.wait(activeFutures).timeout(Duration(seconds: 5));
          print(
            '   Iteration ${iteration + 1} completed with ${activeFutures.length} concurrent operations',
          );
        } catch (timeoutError) {
          print(
            '   Iteration ${iteration + 1} timed out - this might indicate a deadlock',
          );
          rethrow;
        }

        // NO delay between iterations - maximum pressure
        // await Future.delayed(Duration(milliseconds: 100)); // REMOVED
      }

      print(' All enhanced concurrent FFT switches completed - unexpected!');
    } catch (e) {
      print(' CRASH in enhanced concurrent FFT switch: $e');
      print('Stack trace: ${StackTrace.current}');

      // More detailed error analysis
      if (e.toString().contains('CommandList') ||
          e.toString().contains('mD3d12CommandList') ||
          e.toString().contains('dawn') ||
          e.toString().contains('d3d12')) {
        print('  -> CONFIRMED: D3D12 command list validation error!');
        print(
          '  -> ROOT CAUSE: Race condition in concurrent tensor destruction',
        );
      } else if (e.toString().contains('timeout')) {
        print('  -> DEADLOCK: GPU operations are blocking each other');
      } else {
        print('  -> OTHER GPU ERROR: ${e.runtimeType}');
      }

      rethrow;
    } finally {
      // Cleanup
      print('Cleaning up ${allTensors.length} tensors...');
      for (final tensor in allTensors.values) {
        try {
          tensor.buffer.destroy();
        } catch (e) {
          print('Cleanup error: $e');
        }
      }
    }
  });

  // Add this test to isolate the exact moment of the crash
  testWidgets('minimal reproduction - two tensors destroying each other', (
    tester,
  ) async {
    print('\n--- Minimal Reproduction: Tensor Destruction Race ---');

    late Minigpu gpu;
    gpu = Minigpu();
    await gpu.init();

    try {
      // Create two tensors
      print('Creating tensor A...');
      final tensorA = await Tensor.create([200, 4096], gpu: gpu);

      print('Creating tensor B...');
      final tensorB = await Tensor.create([200, 4096], gpu: gpu);

      // Start operation on tensor A
      print('Starting operation on tensor A...');
      final futureA = _longRunningOperation(tensorA);

      // Immediately destroy tensor B and create new one
      print('Destroying tensor B during tensor A operation...');
      tensorB.buffer.destroy(); // This might affect shared GPU state

      final tensorC = await Tensor.create([
        200,
        8192,
      ], gpu: gpu); // Different size

      // Start operation on tensor C
      print('Starting operation on tensor C...');
      final futureC = _longRunningOperation(tensorC);

      // Wait for both operations
      await Future.wait([futureA, futureC]);

      print(' Minimal reproduction completed without crash');

      // Cleanup
      tensorA.buffer.destroy();
      tensorC.buffer.destroy();
    } catch (e) {
      print(' CRASH in minimal reproduction: $e');
      rethrow;
    }
  });
}

Future<void> _longRunningOperation(Tensor tensor) async {
  final shader = tensor.gpu.createComputeShader();
  final outputTensor = await Tensor.create([
    1000000,
  ], gpu: tensor.gpu); // 1M elements

  shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 1000u;
    if (idx < arrayLength(&output)) {
        // Long computation
        var sum: f32 = 0.0;
        for (var i: u32 = 0u; i < 1000u; i++) {
            sum += sin(f32(i + idx) * 0.001);
        }
        output[idx] = sum;
    }
}
''');

  shader.setBuffer('input', tensor.buffer);
  shader.setBuffer('output', outputTensor.buffer);

  await shader.dispatch(125, 125, 1); // 125*125*64 = 1M workgroups

  shader.destroy();
  outputTensor.buffer.destroy();
}

// Helper function that exactly mimics your _ensureGpuTensor
Future<void> _switchStreamFFTSize(
  Minigpu gpu,
  String streamId,
  int fftSize,
  Map<String, Tensor> tensors,
) async {
  // This is your exact pattern from _ensureGpuTensor
  final oldTensor = tensors[streamId];
  if (oldTensor != null) {
    // CRITICAL: This destroys tensor while it might still be in use!
    oldTensor.buffer.destroy();
  }

  // Create new tensor
  final bufferData = List.generate(
    200 * fftSize,
    (i) => math.sin(i * 0.001) * 10.0,
  );

  final newTensor = await Tensor.create(
    [200, fftSize],
    gpu: gpu,
    data: Float32List.fromList(bufferData),
  );

  tensors[streamId] = newTensor;

  // Immediately use the tensor (like your real code)
  final image = await _createSpectrogramFromTensor(newTensor, fftSize, 0);
  image.dispose();
}

// Async version to increase concurrency
Future<ui.Image> _createSpectrogramFromTensorAsync(
  Tensor tensor,
  int fftSize,
  int iteration,
) async {
  final outputWidth = 400;
  final outputHeight = fftSize;
  final rgbaComponentCount = outputWidth * outputHeight * 4;

  final outputTensor = await Tensor.create([
    rgbaComponentCount,
  ], gpu: tensor.gpu);
  final shader = tensor.gpu.createComputeShader();

  // More complex shader to increase GPU processing time
  shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 50u;
    
    // Add more computation to increase processing time
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < 100u; i++) {
        sum += sin(f32(i + idx) * 0.01);
    }
    
    if (idx * 4u + 3u < arrayLength(&output)) {
        let inputIdx = idx % arrayLength(&input);
        var value: f32 = 0.5;
        if (inputIdx < arrayLength(&input)) {
            value = input[inputIdx] * 0.1 + sum * 0.001;
        }
        
        output[idx * 4u] = value;
        output[idx * 4u + 1u] = value * 0.8;
        output[idx * 4u + 2u] = value * 0.6;
        output[idx * 4u + 3u] = 1.0;
    }
}
''');

  shader.setBuffer('input', tensor.buffer);
  shader.setBuffer('output', outputTensor.buffer);

  await shader.dispatch((outputWidth + 7) ~/ 8, (outputHeight + 7) ~/ 8, 1);

  final pixels = await outputTensor.getData() as Float32List;
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

  // Cleanup
  shader.destroy();
  outputTensor.buffer.destroy();

  return image;
}

// Mock spectrogram data class
class MockSpectrogramData {
  final int maxTimeSlices;
  final int frequencyBins;
  final double sampleRate;
  final List<List<double>> timeSlices = [];

  MockSpectrogramData({
    required this.maxTimeSlices,
    required this.frequencyBins,
    required this.sampleRate,
  });

  void addFFTData(List<double> fftData) {
    timeSlices.add(List.from(fftData));

    // Keep only recent slices
    while (timeSlices.length > maxTimeSlices) {
      timeSlices.removeAt(0);
    }
  }

  int get currentTimeSlices => timeSlices.length;
}

List<double> _createRollingBufferData(
  MockSpectrogramData data,
  int maxTimeSlices,
) {
  final bufferData = <double>[];
  final dataSlices = data.timeSlices;
  final totalSlices = dataSlices.length;

  final startIndex = math.max(0, totalSlices - maxTimeSlices);

  for (int bufferIndex = 0; bufferIndex < maxTimeSlices; bufferIndex++) {
    final dataIndex = startIndex + bufferIndex;

    if (dataIndex < totalSlices) {
      bufferData.addAll(dataSlices[dataIndex]);
    } else {
      bufferData.addAll(List.filled(data.frequencyBins, 0.0));
    }
  }

  return bufferData;
}

Future<ui.Image> _createSpectrogramFromTensor(
  Tensor tensor,
  int fftSize,
  int iteration,
) async {
  final outputWidth = 400;
  final outputHeight = fftSize;
  final rgbaComponentCount = outputWidth * outputHeight * 4;

  final outputTensor = await Tensor.create([
    rgbaComponentCount,
  ], gpu: tensor.gpu);
  final shader = tensor.gpu.createComputeShader();

  // Use simplified shader that should trigger the command list error
  shader.loadKernelString('''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 50u; // 400/8 = 50
    if (idx * 4u + 3u < arrayLength(&output)) {
        output[idx * 4u] = 1.0;
        output[idx * 4u + 1u] = 0.5;
        output[idx * 4u + 2u] = 0.0;
        output[idx * 4u + 3u] = 1.0;
    }
}
''');

  shader.setBuffer('input', tensor.buffer);
  shader.setBuffer('output', outputTensor.buffer);

  await shader.dispatch((outputWidth + 7) ~/ 8, (outputHeight + 7) ~/ 8, 1);

  final pixels = await outputTensor.getData() as Float32List;
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

  // Cleanup
  shader.destroy();
  outputTensor.buffer.destroy();

  return image;
}

// More aggressive version that should crash more reliably
Future<void> _aggressiveSwitchStreamFFTSize(
  Minigpu gpu,
  String streamId,
  int fftSize,
  Map<String, Tensor> tensors,
  int iteration,
) async {
  print(
    '    Starting aggressive switch: $streamId -> $fftSize (iteration $iteration)',
  );

  try {
    // CRITICAL: Destroy old tensor immediately without any synchronization
    final oldTensor = tensors[streamId];
    if (oldTensor != null) {
      print('      Destroying old tensor for $streamId...');
      oldTensor.buffer.destroy(); // This should cause the crash
    }

    // Create new tensor with maximum size to increase memory pressure
    final bufferData = List.generate(
      200 * fftSize,
      (i) => math.sin(i * 0.001 + iteration * 0.1) * 10.0,
    );

    print('      Creating new tensor for $streamId...');
    final newTensor = await Tensor.create(
      [200, fftSize],
      gpu: gpu,
      data: Float32List.fromList(bufferData),
    );

    tensors[streamId] = newTensor;

    // Immediately start multiple concurrent operations on the new tensor
    final futures = <Future<void>>[];

    // Operation 1: Create spectrogram
    futures.add(
      _createSpectrogramFromTensorAsync(newTensor, fftSize, iteration).then((
        image,
      ) {
        image.dispose();
      }),
    );

    // Operation 2: Another spectrogram with different parameters
    futures.add(
      _createSpectrogramFromTensorAsync(
        newTensor,
        fftSize,
        iteration + 100,
      ).then((image) {
        image.dispose();
      }),
    );

    // Wait for both operations - this increases the chance of overlapping GPU commands
    await Future.wait(futures);

    print('       Aggressive switch completed: $streamId');
  } catch (e) {
    print('       Aggressive switch failed: $streamId - $e');
    rethrow;
  }
}

class _TestPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint();

    // Draw something GPU-intensive
    for (int i = 0; i < 100; i++) {
      paint.color = Color.lerp(Colors.red, Colors.blue, i / 100.0)!;
      canvas.drawCircle(
        Offset(size.width * math.sin(i * 0.1), size.height * math.cos(i * 0.1)),
        10 + i * 0.5,
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

Future<void> _continuousFlutterRender(WidgetTester tester) async {
  for (int i = 0; i < 50; i++) {
    await tester.pump(Duration(milliseconds: 50));
    print('    Flutter render cycle $i');
  }
}

Future<void> _aggressiveTensorOperationsWithFlutter(Minigpu gpu) async {
  final tensors = <String, Tensor>{};

  for (int i = 0; i < 20; i++) {
    print('    Tensor operation $i during Flutter rendering...');

    final oldTensor = tensors['test'];
    if (oldTensor != null) {
      oldTensor.buffer.destroy(); // This should crash with Flutter active
    }

    final fftSize = 2048 + (i * 100);
    final bufferData = List.generate(
      200 * fftSize,
      (j) => math.sin(j * 0.001) * 10.0,
    );
    final newTensor = await Tensor.create(
      [200, fftSize],
      gpu: gpu,
      data: Float32List.fromList(bufferData),
    );
    tensors['test'] = newTensor;

    // Brief operation
    final outputTensor = await Tensor.create([1000], gpu: gpu);
    final shader = gpu.createComputeShader();

    shader.loadKernelString('''
      @group(0) @binding(0) var<storage, read_write> input: array<f32>;
      @group(0) @binding(1) var<storage, read_write> output: array<f32>;
      
      @compute @workgroup_size(8, 8)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          let idx = gid.x + gid.y * 8u;
          if (idx < arrayLength(&output)) {
              output[idx] = 1.0;
          }
      }
    ''');

    shader.setBuffer('input', newTensor.buffer);
    shader.setBuffer('output', outputTensor.buffer);

    await shader.dispatch(5, 5, 1);

    shader.destroy();
    outputTensor.buffer.destroy();
  }
}
