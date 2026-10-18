import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:gpu_pipeline/src/stages/pipeline_resource_cache.dart';
import 'package:gpu_pipeline/src/pipeline_ports.dart';
import 'package:gpu_pipeline/src/operation/shader_stage_operation.dart';
import 'package:gpu_pipeline/src/operation/cpu_stage_operation.dart';

void main() {
  group('Multi-binding (ShaderStageOperation)', () {
    setUpAll(() async {
      if (!DefaultMinigpu.instance.isInitialized) {
        await DefaultMinigpu.instance.init();
      }
    });

    test('binds 3 inputs to 2 outputs (elementwise)', () async {
      final outputPorts = {
        'sum': FixedOutputPort(name: 'sum', shape: [4]),
        'mix': FixedOutputPort(name: 'mix', shape: [4]),
      };

      final op = ShaderStageOperation(
        operationId: 'multi_in_3_out_2',
        shaderTemplate: '''
          @group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
          @group(0) @binding(1) var<storage, read_write> input_1: array<f32>;
          @group(0) @binding(2) var<storage, read_write> input_2: array<f32>;
          @group(0) @binding(3) var<storage, read_write> output_0: array<f32>;
          @group(0) @binding(4) var<storage, read_write> output_1: array<f32>;

          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
            if (i >= arrayLength(&input_0)) { return; }
            // output_0: sum of all three inputs
            output_0[i] = input_0[i] + input_1[i] + input_2[i];
            // output_1: a simple mix: a*b - c
            output_1[i] = input_0[i] * input_1[i] - input_2[i];
          }
        ''',
        inputKeys: ['a', 'b', 'c'],
        outputKeys: ['sum', 'mix'],
        outputPorts: outputPorts,
      );

      final a = await Tensor.create([4])
        ..write(Float32List.fromList([1, 2, 3, 4]));
      final b = await Tensor.create([4])
        ..write(Float32List.fromList([10, 20, 30, 40]));
      final c = await Tensor.create([4])
        ..write(Float32List.fromList([100, 200, 300, 400]));

      final outputs = await op.execute(
        {'a': a, 'b': b, 'c': c},
        {},
        ResourceCache.forOperation(op.operationId),
      );

      final sum = (await outputs['sum']!.getData()) as Float32List;
      final mix = (await outputs['mix']!.getData()) as Float32List;

      expect(sum.toList(), equals([111, 222, 333, 444]));
      expect(mix.toList(), equals([-90, -160, -210, -240]));

      a.destroy();
      b.destroy();
      c.destroy();
      outputs['sum']!.destroy();
      outputs['mix']!.destroy();
    });

    test('binds 2 inputs to 3 outputs (sum, diff, prod)', () async {
      final outputPorts = {
        'sum': FixedOutputPort(name: 'sum', shape: [3]),
        'diff': FixedOutputPort(name: 'diff', shape: [3]),
        'prod': FixedOutputPort(name: 'prod', shape: [3]),
      };

      final op = ShaderStageOperation(
        operationId: 'multi_out_3',
        shaderTemplate: '''
          @group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
          @group(0) @binding(1) var<storage, read_write> input_1: array<f32>;
          @group(0) @binding(2) var<storage, read_write> output_0: array<f32>;
          @group(0) @binding(3) var<storage, read_write> output_1: array<f32>;
          @group(0) @binding(4) var<storage, read_write> output_2: array<f32>;

          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
            if (i >= arrayLength(&input_0)) { return; }
            output_0[i] = input_0[i] + input_1[i]; // sum
            output_1[i] = input_0[i] - input_1[i]; // diff
            output_2[i] = input_0[i] * input_1[i]; // prod
          }
        ''',
        inputKeys: ['x', 'y'],
        outputKeys: ['sum', 'diff', 'prod'],
        outputPorts: outputPorts,
      );

      final x = await Tensor.create([3])
        ..write(Float32List.fromList([5, 6, 7]));
      final y = await Tensor.create([3])
        ..write(Float32List.fromList([1, 2, 3]));

      final outputs = await op.execute(
        {'x': x, 'y': y},
        {},
        ResourceCache.forOperation(op.operationId),
      );

      final sum = (await outputs['sum']!.getData()) as Float32List;
      final diff = (await outputs['diff']!.getData()) as Float32List;
      final prod = (await outputs['prod']!.getData()) as Float32List;

      expect(sum.toList(), equals([6, 8, 10]));
      expect(diff.toList(), equals([4, 4, 4]));
      expect(prod.toList(), equals([5, 12, 21]));

      x.destroy();
      y.destroy();
      outputs['sum']!.destroy();
      outputs['diff']!.destroy();
      outputs['prod']!.destroy();
    });

    test('input map order is independent of binding order', () async {
      final outputPorts = {
        'out': FixedOutputPort(name: 'out', shape: [3]),
      };

      final op = ShaderStageOperation(
        operationId: 'key_order_independent',
        shaderTemplate: '''
          @group(0) @binding(0) var<storage, read_write> input_0: array<f32>; // 'x'
          @group(0) @binding(1) var<storage, read_write> input_1: array<f32>; // 'y'
          @group(0) @binding(2) var<storage, read_write> output_0: array<f32>;

          @compute @workgroup_size(256)
          fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i: u32 = gid.x;
            if (i >= arrayLength(&input_0)) { return; }
            // Validate binding correctness: output = y - x
            output_0[i] = input_1[i] - input_0[i];
          }
        ''',
        inputKeys: ['x', 'y'],
        outputKeys: ['out'],
        outputPorts: outputPorts,
      );

      final x = await Tensor.create([3])
        ..write(Float32List.fromList([1, 2, 3]));
      final y = await Tensor.create([3])
        ..write(Float32List.fromList([3, 4, 5]));

      // Provide map in reversed order; binding should still follow inputKeys
      final outputs = await op.execute(
        {'y': y, 'x': x},
        {},
        ResourceCache.forOperation(op.operationId),
      );

      final out = (await outputs['out']!.getData()) as Float32List;
      expect(out.toList(), equals([2, 2, 2]));

      x.destroy();
      y.destroy();
      outputs['out']!.destroy();
    });
  });

  group('Multi-binding (CPUStageOperation)', () {
    late ResourceCache cache;

    setUp(() {
      cache = ResourceCache(cacheId: 'multi_cpu');
    });

    tearDown(() {
      cache.dispose();
    });

    test(
      'multiple inputs -> multiple outputs (concatenate and pairwise)',
      () async {
        final op = CPUStageOperation(
          operationId: 'cpu_multi_in_out',
          processor: (inputs, ranks, params) async {
            final a = inputs['a']! as Float32List;
            final b = inputs['b']! as Float32List;

            // out_cat: [a..., b...]
            final outCat = Float32List(a.length + b.length)
              ..setRange(0, a.length, a)
              ..setRange(a.length, a.length + b.length, b);

            // out_pair: elementwise a+b (assume same length)
            final n = a.length < b.length ? a.length : b.length;
            final outPair = Float32List(n);
            for (var i = 0; i < n; i++) {
              outPair[i] = a[i] + b[i];
            }

            return {'out_cat': outCat, 'out_pair': outPair};
          },
          inputKeys: ['a', 'b'],
          outputKeys: ['out_cat', 'out_pair'],
        );

        final ta = await Tensor.zeros([3])
          ..write(Float32List.fromList([1, 2, 3]));
        final tb = await Tensor.zeros([3])
          ..write(Float32List.fromList([4, 5, 6]));

        final outputs = await op.executeWithCache(
          {'a': ta, 'b': tb},
          {},
          cache,
        );

        final outCat = (await outputs['out_cat']!.getData()) as Float32List;
        final outPair = (await outputs['out_pair']!.getData()) as Float32List;

        expect(outCat.toList(), equals([1, 2, 3, 4, 5, 6]));
        expect(outPair.toList(), equals([5, 7, 9]));
      },
    );
  });
}
