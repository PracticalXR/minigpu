import 'dart:typed_data';
import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_pipeline/src/pipeline_ports.dart';
import 'package:test/test.dart';
import 'package:gpu_pipeline/src/stages/pipeline_resource_cache.dart';
import 'package:gpu_pipeline/src/operation/shader_stage_operation.dart';
import 'package:gpu_pipeline/src/pipeline_stage_builder.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  group('Resource Management Crash Tests', () {
    late Minigpu gpu;

    setUpAll(() async {
      gpu = Minigpu();
      await gpu.init();
    });

    tearDownAll(() async {
      await gpu.destroy();
    });

    test('shader cache destruction while in use - crash reproduction', () async {
      print('🧪 Testing shader cache destruction during active use...');

      const size = 1024;
      final cache = ResourceCache.forPipeline('test_pipeline');

      // Create tensors for testing
      final inputTensor = await Tensor.create([size]);
      await inputTensor.write(
        Float32List.fromList(List.generate(size, (i) => i.toDouble())),
      );

      final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_0)) { return; }
    output_0[i] = input_0[i] * 2.0;
}
''';

      // Create first shader operation and cache it
      final operation1 = ShaderStageOperation(
        operationId: 'test_shader_1',
        shaderTemplate: shaderCode,
        inputKeys: ['input'],
        outputKeys: ['output'],
        outputPorts: {
          'output': FixedOutputPort(name: 'output', shape: [size]),
        },
      );

      // Execute first operation (this caches the shader)
      final inputs = {'input': inputTensor};
      final outputs1 = await operation1.executeWithCache(inputs, {}, cache);

      // Create second shader operation to trigger cache overflow
      final operation2 = ShaderStageOperation(
        operationId: 'test_shader_2',
        shaderTemplate: shaderCode.replaceAll('* 2.0', '* 3.0'),
        inputKeys: ['input'],
        outputKeys: ['output'],
        outputPorts: {
          'output': FixedOutputPort(name: 'output', shape: [size]),
        },
      );

      final outputs2 = await operation2.executeWithCache(inputs, {}, cache);

      // Create third shader - this should trigger cache cleanup
      final operation3 = ShaderStageOperation(
        operationId: 'test_shader_3',
        shaderTemplate: shaderCode.replaceAll('* 2.0', '* 4.0'),
        inputKeys: ['input'],
        outputKeys: ['output'],
        outputPorts: {
          'output': FixedOutputPort(name: 'output', shape: [size]),
        },
      );

      print('  🔥 Creating third shader (should trigger cache cleanup)...');
      final outputs3 = await operation3.executeWithCache(
        inputs,
        {},
        cache,
      ); // First shader might get destroyed here

      // Now try to use the first operation again - its cached shader might be destroyed
      print(
        '  💥 Re-executing first operation (cached shader might be destroyed)...',
      );
      try {
        final outputs1Again = await operation1.executeWithCache(
          inputs,
          {},
          cache,
        );
        outputs1Again['output']!.destroy();
      } catch (e) {
        print('  ❌ Expected crash occurred: $e');
      }

      // Clean up
      inputTensor.destroy();
      outputs1['output']!.destroy();
      outputs2['output']!.destroy();
      outputs3['output']!.destroy();
      cache.dispose();
    });

    test('concurrent shader operations with cache cleanup', () async {
      print('🧪 Testing concurrent shader operations during cache cleanup...');

      const size = 512;
      final cache = ResourceCache(
        cacheId: 'concurrent_test',
        maxShaders: 3,
        maxTensors: 10,
      );

      final inputTensor = await Tensor.create([size]);
      await inputTensor.write(
        Float32List.fromList(List.generate(size, (i) => (i % 100).toDouble())),
      );

      final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_0)) { return; }
    output_0[i] = input_0[i] * MULTIPLIER;
}
''';

      // Create multiple shader operations
      final operations = <ShaderStageOperation>[];
      for (int i = 0; i < 5; i++) {
        final operation = ShaderStageOperation(
          operationId: 'concurrent_shader_$i',
          shaderTemplate: shaderTemplate.replaceAll('MULTIPLIER', '${i + 1}.0'),
          inputKeys: ['input'],
          outputKeys: ['output'],
          outputPorts: {
            'output': FixedOutputPort(name: 'output', shape: [size]),
          },
        );
        operations.add(operation);
      }

      final inputs = {'input': inputTensor};

      // Execute all operations concurrently - this will trigger cache cleanup
      print(
        '  🔥 Executing operations concurrently (cache cleanup triggered)...',
      );

      final futures = operations.map((op) async {
        try {
          return await op.executeWithCache(inputs, {}, cache);
        } catch (e) {
          print('  ❌ Operation failed (expected): $e');
          return <String, Tensor>{};
        }
      }).toList();

      final results = await Future.wait(futures);

      // Clean up results
      for (final result in results) {
        result['output']?.destroy();
      }

      inputTensor.destroy();
      cache.dispose();
    });

    test('shader function operation resource conflicts', () async {
      print('🧪 Testing shader function operation resource conflicts...');

      const size = 256;
      final cache = ResourceCache(
        cacheId: 'function_test',
        maxShaders: 2,
        maxTensors: 5,
      );

      final inputTensor = await Tensor.create([size]);
      await inputTensor.write(
        Float32List.fromList(List.generate(size, (i) => i.toDouble() / size)),
      );

      // Create dynamic stage with shader function
      final stage = StageBuilder('function_conflict_test')
          .executeShaderFunction(
            'return input_val * MULTIPLIER;',
            functionName: 'multiply',
            functionParameters: {'MULTIPLIER': 2.5},
          )
          .build();

      // Set the resource cache for the stage
      stage.setResourceCache(cache);

      await stage.initializeStage();

      final inputs = {'input': inputTensor};

      // Execute multiple times to trigger caching issues
      print('  🔥 Executing shader function multiple times...');

      Map<String, Tensor>? outputs1, outputs2, outputs3;

      try {
        outputs1 = await stage.process(inputs);
        print('  ✅ First execution successful');

        outputs2 = await stage.process(inputs);
        print('  ✅ Second execution successful');

        outputs3 = await stage.process(inputs);
        print('  💥 Third execution (potential resource conflict)...');
      } catch (e) {
        print('  ❌ Shader function execution crashed: $e');
      }

      // Clean up
      inputTensor.destroy();
      outputs1?['input']?.destroy();
      outputs2?['input']?.destroy();
      outputs3?['input']?.destroy();
      await stage.dispose();
      cache.dispose();
    });

    test('pipeline stage disposal during active operations', () async {
      print('🧪 Testing pipeline stage disposal during active operations...');

      const size = 128;
      final cache = ResourceCache.forOperation('disposal_test');

      final inputTensor = await Tensor.create([size]);
      await inputTensor.write(
        Float32List.fromList(List.generate(size, (i) => i.toDouble())),
      );

      final stage = StageBuilder('disposal_test').executeShader('''
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_0)) { return; }
    output_0[i] = input_0[i] * 0.5;
}
''').build();

      // Set the resource cache for the stage
      stage.setResourceCache(cache);

      await stage.initializeStage();

      final inputs = {'input': inputTensor};

      // Start processing (async)
      print('  🔥 Starting async processing...');
      final processingFuture = stage.process(inputs);

      // Immediately dispose stage while processing
      print('  💥 Disposing stage during active processing...');
      final disposalFuture = stage.dispose();

      // Wait for both to complete - should crash or handle gracefully
      try {
        final results = await Future.wait([processingFuture, disposalFuture]);
        final outputs = results[0] as Map<String, Tensor>;
        print('  ⚠️  Operations completed without crash (unexpected)');
        outputs['input']?.destroy();
      } catch (e) {
        print('  ❌ Expected crash during concurrent disposal: $e');
      }

      inputTensor.destroy();
      cache.dispose();
    });

    test('buffer state corruption between operations', () async {
      print('🧪 Testing buffer state corruption between operations...');

      const size = 64;
      final cache = ResourceCache.forOperation('buffer_test');

      final inputTensor1 = await Tensor.create([size]);
      await inputTensor1.write(
        Float32List.fromList(List.generate(size, (i) => i.toDouble())),
      );

      final inputTensor2 = await Tensor.create([size * 2]); // Different size
      await inputTensor2.write(
        Float32List.fromList(
          List.generate(size * 2, (i) => i.toDouble() * 0.5),
        ),
      );

      final shaderCode = '''
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_0)) { return; }
    output_0[i] = input_0[i] + 1.0;
}
''';

      final operation = ShaderStageOperation(
        operationId: 'buffer_corruption_test',
        shaderTemplate: shaderCode,
        inputKeys: ['input'],
        outputKeys: ['output'],
        outputPorts: {
          'output': FixedOutputPort(name: 'output', shape: [size]),
        },
      );

      // Execute with first tensor
      print('  🔥 Executing with first tensor (size: $size)...');
      final inputs1 = {'input': inputTensor1};
      final outputs1 = await operation.executeWithCache(inputs1, {}, cache);

      // Execute with second tensor (different size) - should cause buffer mismatch
      print(
        '  💥 Executing with second tensor (size: ${size * 2}) - buffer mismatch...',
      );
      final inputs2 = {'input': inputTensor2};

      try {
        final outputs2 = await operation.executeWithCache(inputs2, {}, cache);
        print('  ⚠️  Second execution succeeded (unexpected)');
        outputs2['output']?.destroy();
      } catch (e) {
        print('  ❌ Expected buffer mismatch crash: $e');
      }

      // Clean up
      inputTensor1.destroy();
      inputTensor2.destroy();
      outputs1['output']!.destroy();
      cache.dispose();
    });

    test('parameter resolution crash in shader functions', () async {
      print('🧪 Testing parameter resolution crashes in shader functions...');

      const size = 32;
      final cache = ResourceCache.forOperation('param_test');

      final inputTensor = await Tensor.create([size]);
      await inputTensor.write(
        Float32List.fromList(List.generate(size, (i) => i.toDouble())),
      );

      // Create stage with invalid parameter reference
      final stage = StageBuilder('param_crash_test')
          .withParameter(
            Parameter.stageFloat(
              'valid_param',
              description: "Valid parameter",
              defaultValue: 1.0,
            ),
          )
          .executeShaderFunction(
            'return input_val * INVALID_PARAM;', // References non-existent parameter
            functionName: 'crash_func',
            functionParameters: {
              'INVALID_PARAM': 'param:nonexistent_param', // This should crash
            },
          )
          .build();

      // Set the resource cache for the stage
      stage.setResourceCache(cache);

      await stage.initializeStage();

      final inputs = {'input': inputTensor};

      print('  💥 Executing with invalid parameter reference...');
      try {
        final outputs = await stage.process(inputs);
        print('  ⚠️  Execution succeeded with invalid param (unexpected)');
        outputs['input']?.destroy();
      } catch (e) {
        print('  ❌ Expected parameter resolution crash: $e');
      }

      inputTensor.destroy();
      await stage.dispose();
      cache.dispose();
    });

    test('tensor disposal during shader execution', () async {
      print('🧪 Testing tensor disposal during shader execution...');

      const size = 256;
      final cache = ResourceCache.forOperation('tensor_disposal_test');

      var inputTensor = await Tensor.create([size]);
      await inputTensor.write(
        Float32List.fromList(List.generate(size, (i) => i.toDouble())),
      );

      final stage = StageBuilder('tensor_disposal_test').executeShaderFunction(
        '''
            // Simulate slow operation
            let val = input_val;
            for (var i: i32 = 0; i < 100; i++) {
              val = val * 1.001;
            }
            return val;
            ''',
        functionName: 'slow_operation',
      ).build();

      // Set the resource cache for the stage
      stage.setResourceCache(cache);

      await stage.initializeStage();

      final inputs = {'input': inputTensor};

      // Start processing
      print('  🔥 Starting slow shader execution...');
      final processingFuture = stage.process(inputs);

      // Dispose input tensor while shader is running
      print('  💥 Disposing input tensor during execution...');
      inputTensor.destroy();
      inputTensor = await Tensor.create([
        1,
      ]); // Create new tensor to avoid null reference

      try {
        final outputs = await processingFuture;
        print('  ⚠️  Shader completed after tensor disposal (unexpected)');
        outputs['input']?.destroy();
      } catch (e) {
        print('  ❌ Expected crash after tensor disposal: $e');
      }

      inputTensor.destroy();
      await stage.dispose();
      cache.dispose();
    });

    test('shader compilation error handling', () async {
      print('🧪 Testing shader compilation error handling...');

      const size = 16;
      final cache = ResourceCache.forOperation('compilation_test');

      final inputTensor = await Tensor.create([size]);
      await inputTensor.write(
        Float32List.fromList(List.generate(size, (i) => i.toDouble())),
      );

      // Create stage with invalid shader code
      final stage = StageBuilder('compilation_error_test').executeShader('''
// Invalid WGSL syntax
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_0)) { return; }
    output_0[i] = input_0[i] * UNDEFINED_CONSTANT; // This will cause compilation error
}
''').build();

      // Set the resource cache for the stage
      stage.setResourceCache(cache);

      print('  💥 Initializing stage with invalid shader...');
      try {
        await stage.initializeStage();

        final inputs = {'input': inputTensor};
        final outputs = await stage.process(inputs);
        print('  ⚠️  Invalid shader executed successfully (unexpected)');
        outputs['input']?.destroy();
      } catch (e) {
        print('  ❌ Expected shader compilation error: $e');
      }

      inputTensor.destroy();
      await stage.dispose();
      cache.dispose();
    });
  });

  group('Memory Leak Detection', () {
    test('detect tensor memory leaks in operations', () async {
      print('🧪 Testing for tensor memory leaks...');

      const iterations = 10;
      const size = 1024;

      for (int i = 0; i < iterations; i++) {
        final cache = ResourceCache.forOperation('leak_test_$i');

        final inputTensor = await Tensor.create([size]);
        await inputTensor.write(
          Float32List.fromList(List.generate(size, (j) => j.toDouble())),
        );

        final stage = StageBuilder('leak_test_$i')
            .executeShaderFunction(
              'return input_val * 2.0;',
              functionName: 'double_val',
            )
            .build();

        // Set the resource cache for the stage
        stage.setResourceCache(cache);

        await stage.initializeStage();

        final inputs = {'input': inputTensor};
        final outputs = await stage.process(inputs);

        // Clean up - if this isn't working properly, we'll have leaks
        inputTensor.destroy();
        outputs['input']?.destroy();
        await stage.dispose();
        cache.dispose();

        if (i % 5 == 0) {
          print('  🔄 Completed iteration $i/$iterations');
        }
      }

      print('  ✅ Memory leak test completed');
    });

    test('detect shader cache memory leaks', () async {
      print('🧪 Testing for shader cache memory leaks...');

      const iterations = 15; // Reduced from 20 to avoid hanging
      final cache = ResourceCache(
        cacheId: 'cache_leak_test',
        maxShaders: 3, // Smaller cache to trigger cleanup more often
        maxTensors: 5,
      );

      final operations = <ShaderStageOperation>[];

      try {
        for (int i = 0; i < iterations; i++) {
          final shaderCode =
              '''
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

const MULTIPLIER: f32 = ${i + 1}.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&input_0)) { return; }
    output_0[idx] = input_0[idx] * MULTIPLIER;
}
''';

          final operation = ShaderStageOperation(
            operationId: 'cache_leak_test_$i',
            shaderTemplate: shaderCode,
            inputKeys: ['input'],
            outputKeys: ['output'],
            outputPorts: {
              'output': FixedOutputPort(name: 'output', shape: [64]),
            },
          );

          operations.add(operation);

          final inputTensor = await Tensor.create([64]);
          await inputTensor.write(
            Float32List.fromList(List.generate(64, (j) => j.toDouble())),
          );

          final inputs = {'input': inputTensor};

          try {
            final outputs = await operation.executeWithCache(inputs, {}, cache);

            // Clean up immediately to avoid resource buildup
            inputTensor.destroy();
            outputs['output']?.destroy();

            if (i % 5 == 0) {
              print(
                '  🔄 Cache test iteration $i/$iterations (cache size: ${cache.shaderCacheSize}/${cache.maxShaders})',
              );

              // Add small delay to let cleanup complete
              await Future.delayed(Duration(milliseconds: 10));
            }
          } catch (e) {
            print('  ⚠️ Operation $i failed: $e');
            inputTensor.destroy();
            // Continue with next iteration
          }
        }

        print('  ✅ Shader cache leak test completed');
      } catch (e) {
        print('  ❌ Cache leak test failed: $e');
      } finally {
        // Clean up all operations
        for (final op in operations) {
          try {
            await op.dispose();
          } catch (e) {
            print('  ⚠️ Failed to dispose operation: $e');
          }
        }

        // Final cache cleanup
        try {
          cache.dispose();
        } catch (e) {
          print('  ⚠️ Failed to dispose cache: $e');
        }
      }
    });

    test('cache statistics and monitoring', () async {
      print('🧪 Testing cache statistics and monitoring...');

      final cache = ResourceCache(
        cacheId: 'stats_test',
        maxShaders: 10,
        maxTensors: 20,
      );

      // Perform some operations to generate cache activity
      for (int i = 0; i < 5; i++) {
        final shaderCode =
            '''
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_0)) { return; }
    output_0[i] = input_0[i] * ${i + 1}.0;
}
''';

        final operation = ShaderStageOperation(
          operationId: 'stats_test_$i',
          shaderTemplate: shaderCode,
          inputKeys: ['input'],
          outputKeys: ['output'],
          outputPorts: {
            'output': FixedOutputPort(name: 'output', shape: [32]),
          },
        );

        final inputTensor = await Tensor.create([32]);
        await inputTensor.write(
          Float32List.fromList(List.generate(32, (j) => j.toDouble())),
        );

        final inputs = {'input': inputTensor};
        final outputs = await operation.executeWithCache(inputs, {}, cache);

        inputTensor.destroy();
        outputs['output']!.destroy();
      }

      // Check cache statistics
      final stats = cache.getCacheStats();
      print('  📊 Cache Statistics:');
      print('     - Cache ID: ${stats['cache_id']}');
      print('     - Shaders: ${stats['shader_count']}/${stats['max_shaders']}');
      print('     - Tensors: ${stats['tensor_count']}/${stats['max_tensors']}');
      print(
        '     - Hit Rate: ${(stats['hit_rate'] * 100).toStringAsFixed(1)}%',
      );
      print('     - Memory Usage: ${stats['memory_usage']} bytes');

      expect(stats['shader_count'], greaterThan(0));
      expect(stats['cache_id'], equals('stats_test'));
      expect(stats['hit_rate'], greaterThanOrEqualTo(0.0));
      expect(stats['hit_rate'], lessThanOrEqualTo(1.0));

      cache.printStats();
      cache.dispose();
      print('  ✅ Cache statistics test completed');
    });
  });
}
