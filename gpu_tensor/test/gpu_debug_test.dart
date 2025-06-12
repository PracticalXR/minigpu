import 'dart:async';
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:minigpu/minigpu.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

void main() {
  late Minigpu gpu;

  setUpAll(() async {
    gpu = Minigpu();
    await gpu.init();
  });

  tearDownAll(() async {});

  group('Tensor Lifecycle Crash Tests', () {
    test(
      'rapid tensor create/destroy with shaders - exact spectrogram pattern',
      () async {
        const tensorTimeSlices = 200;
        const tensorFreqBins = 512;
        const actualTimeSlices = 1;
        const maxMagnitude = 0.09470604943271473; // Your exact value
        const numCycles = 20;

        print('Testing rapid tensor lifecycle like spectrogram...');

        for (int cycle = 0; cycle < numCycles; cycle++) {
          try {
            print('Cycle $cycle: Creating tensors...');

            // Create full spectrogram data like your code
            final fullSpectrogramData = <double>[];
            for (int t = 0; t < tensorTimeSlices; t++) {
              for (int f = 0; f < tensorFreqBins; f++) {
                if (t < actualTimeSlices) {
                  fullSpectrogramData.add((f * t + cycle) % 100 / 100.0);
                } else {
                  fullSpectrogramData.add(0.0);
                }
              }
            }

            // Create input tensor EXACTLY like your code
            print(
              '  Creating input tensor [${tensorTimeSlices}, ${tensorFreqBins}]...',
            );
            final inputTensor = await Tensor.create(
              [tensorTimeSlices, tensorFreqBins],
              gpu: gpu,
              data: Float32List.fromList(fullSpectrogramData),
            );

            // Create output tensor EXACTLY like your code
            final outputSize = tensorTimeSlices * tensorFreqBins * 4;
            print('  Creating output tensor [$outputSize]...');
            final outputTensor = await Tensor.create([outputSize], gpu: gpu);

            // Use your EXACT shader template with dynamic values
            final shaderTemplate =
                '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const TEXTURE_WIDTH: u32 = ${tensorTimeSlices}u;
const TEXTURE_HEIGHT: u32 = ${tensorFreqBins}u;
const ACTUAL_TIME_SLICES: u32 = ${actualTimeSlices}u;
const MAX_MAGNITUDE: f32 = ${maxMagnitude.toStringAsFixed(6)};
const INPUT_SIZE: u32 = ${tensorTimeSlices * tensorFreqBins}u;
const OUTPUT_SIZE: u32 = ${outputSize}u;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: u32 = gid.x;
    let y: u32 = gid.y;
    
    if (x >= TEXTURE_WIDTH || y >= TEXTURE_HEIGHT) {
        return;
    }
    
    let inputIndex: u32 = x * TEXTURE_HEIGHT + y;
    var magnitude: f32 = 0.0;
    
    if (x < ACTUAL_TIME_SLICES && 
        inputIndex < INPUT_SIZE && 
        inputIndex < arrayLength(&input)) {
        magnitude = input[inputIndex];
    }
    
    let normalizedMag: f32 = clamp(magnitude / MAX_MAGNITUDE, 0.0, 1.0);
    
    var color: vec4<f32>;
    if (normalizedMag <= 0.01) {
        color = vec4<f32>(0.0, 0.0, 0.1, 1.0);
    } else {
        let t: f32 = normalizedMag;
        if (t < 0.33) {
            let s: f32 = t * 3.0;
            color = vec4<f32>(0.0, s, 1.0 - s * 0.5, 1.0);
        } else if (t < 0.66) {
            let s: f32 = (t - 0.33) * 3.0;
            color = vec4<f32>(s, 1.0, 0.0, 1.0);
        } else {
            let s: f32 = (t - 0.66) * 3.0;
            color = vec4<f32>(1.0, 1.0 - s * 0.5, 0.0, 1.0);
        }
    }
    
    let outputY: u32 = TEXTURE_HEIGHT - 1u - y;
    let outputIndex: u32 = (outputY * TEXTURE_WIDTH + x) * 4u;
    
    if (outputIndex + 3u < OUTPUT_SIZE && 
        outputIndex + 3u < arrayLength(&output)) {
        output[outputIndex] = color.r;
        output[outputIndex + 1u] = color.g;
        output[outputIndex + 2u] = color.b;
        output[outputIndex + 3u] = color.a;
    }
}
''';

            print('  Creating compute shader...');
            final shader = gpu.createComputeShader();

            print('  Loading shader...');
            shader.loadKernelString(shaderTemplate);

            print('  Setting buffers...');
            shader.setBuffer('input', inputTensor.buffer);
            shader.setBuffer('output', outputTensor.buffer);

            final workgroupsX = (tensorTimeSlices + 15) ~/ 16;
            final workgroupsY = (tensorFreqBins + 15) ~/ 16;

            print('  Dispatching ${workgroupsX}x${workgroupsY} workgroups...');
            await shader.dispatch(workgroupsX, workgroupsY, 1);

            print('  Getting data...');
            final pixels = await outputTensor.getData() as Float32List;
            print('  Got ${pixels.length} pixels');

            print('  Destroying resources...');
            shader.destroy();
            inputTensor.buffer.destroy();
            outputTensor.buffer.destroy();

            print('Cycle $cycle: SUCCESS');
          } catch (e, stackTrace) {
            print('CRASH on cycle $cycle: $e');
            print('Stack trace: $stackTrace');
            break;
          }
        }
      },
    );

    test('timer-driven updates like real spectrogram widget', () async {
      const tensorTimeSlices = 200;
      const tensorFreqBins = 512;
      var actualTimeSlices = 1;
      const updateIntervalMs = 16;
      const numUpdates = 1000;

      print('Testing timer-driven updates every ${updateIntervalMs}ms...');

      // Persistent state like your widget
      final Map<String, Tensor> spectrogramTensors = {};
      final streamId = 'test_stream';

      final timer = Timer.periodic(Duration(milliseconds: updateIntervalMs), (
        timer,
      ) async {
        if (timer.tick > numUpdates) {
          timer.cancel();
          return;
        }

        try {
          print('Timer update ${timer.tick}/$numUpdates...');

          // Clean up previous tensor like your code
          if (spectrogramTensors.containsKey(streamId)) {
            spectrogramTensors[streamId]!.buffer.destroy();
          }

          // Create new tensor with updated data
          final fullSpectrogramData = <double>[];
          for (int t = 0; t < tensorTimeSlices; t++) {
            for (int f = 0; f < tensorFreqBins; f++) {
              if (t < actualTimeSlices) {
                fullSpectrogramData.add(
                  (f * t + timer.tick * 1000) % 100 / 100.0,
                );
              } else {
                fullSpectrogramData.add(0.0);
              }
            }
          }

          spectrogramTensors[streamId] = await Tensor.create(
            [tensorTimeSlices, tensorFreqBins],
            gpu: gpu,
            data: Float32List.fromList(fullSpectrogramData),
          );

          // Process like your createSpectrogramTexture
          final outputSize = tensorTimeSlices * tensorFreqBins * 4;
          final outputTensor = await Tensor.create([outputSize], gpu: gpu);

          final maxMagnitude = 0.1 + timer.tick * 0.01; // Changing value

          final shaderCode =
              '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const TEXTURE_WIDTH: u32 = ${tensorTimeSlices}u;
const TEXTURE_HEIGHT: u32 = ${tensorFreqBins}u;
const ACTUAL_TIME_SLICES: u32 = ${actualTimeSlices}u;
const MAX_MAGNITUDE: f32 = ${maxMagnitude.toStringAsFixed(6)};

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
    if (x < ACTUAL_TIME_SLICES && inputIndex < arrayLength(&input)) {
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
          shader.setBuffer('input', spectrogramTensors[streamId]!.buffer);
          shader.setBuffer('output', outputTensor.buffer);

          await shader.dispatch(
            (tensorTimeSlices + 15) ~/ 16,
            (tensorFreqBins + 15) ~/ 16,
            1,
          );

          final pixels = await outputTensor.getData() as Float32List;

          shader.destroy();
          outputTensor.buffer.destroy();

          // Simulate time slices growing
          if (timer.tick % 3 == 0) actualTimeSlices++;

          print('Timer update ${timer.tick}: SUCCESS');
        } catch (e, stackTrace) {
          print('Timer update ${timer.tick} CRASHED: $e');
          timer.cancel();
        }
      });

      // Wait for timer to finish
      await Future.delayed(
        Duration(milliseconds: updateIntervalMs * (numUpdates + 2)),
      );

      // Cleanup
      for (final tensor in spectrogramTensors.values) {
        tensor.buffer.destroy();
      }
    });

    test('test float precision in shader constants', () async {
      // Test if the dynamic float values are causing issues
      final problematicValues = [
        0.09470604943271473, // Your exact crash value
        0.000000000001, // Very small number
        999999999.999999, // Very large number
        double.nan, // NaN
        double.infinity, // Infinity
        -double.infinity, // -Infinity
      ];

      for (final maxMag in problematicValues) {
        try {
          print('Testing MAX_MAGNITUDE: $maxMag');

          final tensorData = await Tensor.create([100], gpu: gpu);
          final outputData = await Tensor.create([400], gpu: gpu);

          final shaderCode =
              '''
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const MAX_MAGNITUDE: f32 = ${maxMag.toStringAsFixed(6)};

@compute @workgroup_size(10, 10)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if (i >= 100u) return;
    
    let value: f32 = f32(i) / MAX_MAGNITUDE;
    let outputIndex: u32 = i * 4u;
    
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
          shader.setBuffer('input', tensorData.buffer);
          shader.setBuffer('output', outputData.buffer);

          await shader.dispatch(10, 10, 1);

          shader.destroy();
          tensorData.buffer.destroy();
          outputData.buffer.destroy();

          print('  SUCCESS with $maxMag');
        } catch (e) {
          print('  CRASH with $maxMag: $e');
        }
      }
    });
  });
}
