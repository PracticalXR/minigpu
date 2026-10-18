import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_pipeline/src/pipeline_stage_builder.dart';
import 'package:gpu_pipeline/src/operation/shader_stage_operation.dart';
import 'package:gpu_pipeline/src/operation/tensor_stage_operation.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  group('Cache Debugging Tests', () {
    setUpAll(() async {
      await Minigpu().init();
    });

    test('FFT operation should process different inputs differently', () async {
      print('🔬 Testing FFT caching behavior');

      final stage = StageBuilder('fft_cache_test').op('fft').build();

      await stage.initializeStage();

      // Test 1: Process first input
      final input1 = await Tensor.create([8]);
      await input1.write(
        Float32List.fromList([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
      );

      print('📊 Input 1: ${await input1.getData()}');

      final result1 = await stage.process({'input': input1});
      final output1Data = await result1['output']!.getData() as Float32List;

      print('📊 Output 1: ${output1Data.take(8).toList()}');

      // Test 2: Process different input
      final input2 = await Tensor.create([8]);
      await input2.write(
        Float32List.fromList([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
      );

      print('📊 Input 2: ${await input2.getData()}');

      final result2 = await stage.process({'input': input2});
      final output2Data = await result2['output']!.getData() as Float32List;

      print('📊 Output 2: ${output2Data.take(8).toList()}');

      // Test 3: Process first input again
      final input3 = await Tensor.create([8]);
      await input3.write(
        Float32List.fromList([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
      );

      print('📊 Input 3 (same as 1): ${await input3.getData()}');

      final result3 = await stage.process({'input': input3});
      final output3Data = await result3['output']!.getData() as Float32List;

      print('📊 Output 3: ${output3Data.take(8).toList()}');

      // Verify outputs are different for different inputs
      bool outputs1And2AreDifferent = false;
      for (int i = 0; i < 8; i++) {
        if ((output1Data[i] - output2Data[i]).abs() > 0.001) {
          outputs1And2AreDifferent = true;
          break;
        }
      }

      // Verify outputs are same for same inputs
      bool outputs1And3AreSame = true;
      for (int i = 0; i < 8; i++) {
        if ((output1Data[i] - output3Data[i]).abs() > 0.001) {
          outputs1And3AreSame = false;
          break;
        }
      }

      print('📊 Outputs 1&2 different: $outputs1And2AreDifferent');
      print('📊 Outputs 1&3 same: $outputs1And3AreSame');

      expect(
        outputs1And2AreDifferent,
        isTrue,
        reason: 'Different inputs should produce different outputs',
      );
      expect(
        outputs1And3AreSame,
        isTrue,
        reason: 'Same inputs should produce same outputs',
      );

      // Clean up
      input1.destroy();
      input2.destroy();
      input3.destroy();
      result1['output']!.destroy();
      result2['output']!.destroy();
      result3['output']!.destroy();
      await stage.dispose();
    });

    test(
      'Shader function should process different inputs differently',
      () async {
        print('🔬 Testing shader function caching behavior');

        final stage = StageBuilder('shader_cache_test')
            .executeShaderFunction(
              'return input_val * 2.0;',
              functionName: 'double_value',
            )
            .build();

        await stage.initializeStage();

        // Test 1: Process first input
        final input1 = await Tensor.create([4]);
        await input1.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));

        print('📊 Shader Input 1: ${await input1.getData()}');

        final result1 = await stage.process({'input': input1});
        final output1Data = await result1['output']!.getData() as Float32List;

        print('📊 Shader Output 1: ${output1Data.toList()}');

        // Test 2: Process different input
        final input2 = await Tensor.create([4]);
        await input2.write(Float32List.fromList([5.0, 6.0, 7.0, 8.0]));

        print('📊 Shader Input 2: ${await input2.getData()}');

        final result2 = await stage.process({'input': input2});
        final output2Data = await result2['output']!.getData() as Float32List;

        print('📊 Shader Output 2: ${output2Data.toList()}');

        // Check cache statistics
        final op = stage.operations[0] as ShaderFunctionOperation;
        print(
          '📊 Shader Cache hits: ${op.cacheHits}, misses: ${op.cacheMisses}',
        );

        // Verify outputs are different for different inputs
        expect(
          output1Data[0],
          equals(2.0),
          reason: '1.0 * 2.0 should equal 2.0',
        );
        expect(
          output1Data[1],
          equals(4.0),
          reason: '2.0 * 2.0 should equal 4.0',
        );
        expect(
          output2Data[0],
          equals(10.0),
          reason: '5.0 * 2.0 should equal 10.0',
        );
        expect(
          output2Data[1],
          equals(12.0),
          reason: '6.0 * 2.0 should equal 12.0',
        );

        // Clean up
        input1.destroy();
        input2.destroy();
        result1['output']!.destroy();
        result2['output']!.destroy();
        await stage.dispose();
      },
    );

    test('Window + FFT pipeline should handle changing inputs', () async {
      print('🔬 Testing window+FFT pipeline caching behavior');

      // Create window stage
      final windowStage = StageBuilder('window_test').executeShaderFunction('''
            let n = f32(arrayLength(&input_0));
            let i = f32(index);
            let pi = 3.14159265359;
            let window_val = 0.5 * (1.0 - cos(2.0 * pi * i / (n - 1.0)));
            return input_val * window_val;
            ''', functionName: 'applyHannWindow').build();

      // Create FFT stage
      final fftStage = StageBuilder('fft_test').op('fft').build();

      await windowStage.initializeStage();
      await fftStage.initializeStage();

      // Test with first input
      final input1 = await Tensor.create([8]);
      await input1.write(
        Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
      );

      print('📊 Pipeline Input 1: ${await input1.getData()}');

      // Window stage
      final windowResult1 = await windowStage.process({'input': input1});
      final windowOutput1 =
          await windowResult1['output']!.getData() as Float32List;
      print('📊 Window Output 1: ${windowOutput1.toList()}');

      // FFT stage
      final fftResult1 = await fftStage.process({
        'input': windowResult1['output']!,
      });
      final fftOutput1 = await fftResult1['output']!.getData() as Float32List;
      print('📊 FFT Output 1: ${fftOutput1.take(8).toList()}');

      // Test with second input (different data)
      final input2 = await Tensor.create([8]);
      await input2.write(
        Float32List.fromList([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
      );

      print('📊 Pipeline Input 2: ${await input2.getData()}');

      // Window stage
      final windowResult2 = await windowStage.process({'input': input2});
      final windowOutput2 =
          await windowResult2['output']!.getData() as Float32List;
      print('📊 Window Output 2: ${windowOutput2.toList()}');

      // FFT stage
      final fftResult2 = await fftStage.process({
        'input': windowResult2['output']!,
      });
      final fftOutput2 = await fftResult2['output']!.getData() as Float32List;
      print('📊 FFT Output 2: ${fftOutput2.take(8).toList()}');

      // Check cache statistics
      final windowOp = windowStage.operations[0] as ShaderFunctionOperation;
      final fftOp = fftStage.operations[0] as TensorOperation;
      print(
        '📊 Window Cache hits: ${windowOp.cacheHits}, misses: ${windowOp.cacheMisses}',
      );
      print(
        '📊 FFT Cache hits: ${fftOp.cacheHits}, misses: ${fftOp.cacheMisses}',
      );

      // Verify window outputs are different
      bool windowOutputsDifferent = false;
      for (int i = 0; i < 8; i++) {
        if ((windowOutput1[i] - windowOutput2[i]).abs() > 0.001) {
          windowOutputsDifferent = true;
          break;
        }
      }

      // Verify FFT outputs are different
      bool fftOutputsDifferent = false;
      for (int i = 0; i < 8; i++) {
        if ((fftOutput1[i] - fftOutput2[i]).abs() > 0.001) {
          fftOutputsDifferent = true;
          break;
        }
      }

      print('📊 Window outputs different: $windowOutputsDifferent');
      print('📊 FFT outputs different: $fftOutputsDifferent');

      expect(
        windowOutputsDifferent,
        isTrue,
        reason:
            'Window stage should produce different outputs for different inputs',
      );
      expect(
        fftOutputsDifferent,
        isTrue,
        reason:
            'FFT stage should produce different outputs for different inputs',
      );

      // Clean up
      input1.destroy();
      input2.destroy();
      windowResult1['output']!.destroy();
      windowResult2['output']!.destroy();
      fftResult1['output']!.destroy();
      fftResult2['output']!.destroy();
      await windowStage.dispose();
      await fftStage.dispose();
    });

    test('Check if tensor operations are cached incorrectly', () async {
      print('🔬 Testing tensor operation result caching');

      final stage = StageBuilder('tensor_cache_test').op('abs').build();

      await stage.initializeStage();

      // Test 1: Process positive values
      final input1 = await Tensor.create([4]);
      await input1.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));

      final result1 = await stage.process({'input': input1});
      final output1Data = await result1['output']!.getData() as Float32List;

      print('📊 Abs Input 1: [1.0, 2.0, 3.0, 4.0]');
      print('📊 Abs Output 1: ${output1Data.toList()}');

      // Test 2: Process negative values
      final input2 = await Tensor.create([4]);
      await input2.write(Float32List.fromList([-1.0, -2.0, -3.0, -4.0]));

      final result2 = await stage.process({'input': input2});
      final output2Data = await result2['output']!.getData() as Float32List;

      print('📊 Abs Input 2: [-1.0, -2.0, -3.0, -4.0]');
      print('📊 Abs Output 2: ${output2Data.toList()}');

      // Check cache statistics
      final op = stage.operations[0] as TensorOperation;
      print(
        '📊 Tensor Op Cache hits: ${op.cacheHits}, misses: ${op.cacheMisses}',
      );

      // Both outputs should be [1.0, 2.0, 3.0, 4.0] since abs of negatives = positives
      expect(output1Data[0], equals(1.0));
      expect(output1Data[1], equals(2.0));
      expect(output2Data[0], equals(1.0));
      expect(output2Data[1], equals(2.0));

      // But they should be computed fresh each time, not cached
      expect(
        op.cacheMisses,
        greaterThanOrEqualTo(2),
        reason:
            'Should have at least 2 cache misses for 2 different executions',
      );

      // Clean up
      input1.destroy();
      input2.destroy();
      result1['output']!.destroy();
      result2['output']!.destroy();
      await stage.dispose();
    });
  });
}
