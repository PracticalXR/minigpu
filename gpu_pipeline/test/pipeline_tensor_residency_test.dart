import 'dart:typed_data';

import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';

/// A [PipelineTensor] keeps a host copy and a GPU copy. Whichever side was
/// written last is current, and a read from the other side must return it.
///
/// These pin the two orders that used to return stale data: a host write
/// followed by a host read while a GPU copy exists (the old GPU contents were
/// read back over the new host data), and a GPU write followed by a host read
/// while a host copy exists (the old host copy was returned). The stage tests
/// reach the same two orders through ops that write a key in place.
void main() {
  Future<Tensor> gpuTensor(List<double> values) async {
    final t = await Tensor.zeros([values.length]);
    await t.write(Float32List.fromList(values));
    return t;
  }

  Float32List plusOne(TypedData data) =>
      Float32List.fromList([for (final v in data as Float32List) v + 1]);

  group('PipelineTensor residency', () {
    test('a host write is what the next host read returns', () async {
      final pt = PipelineTensor.gpu(await gpuTensor([1, 2, 3, 4]), [4]);
      expect(await pt.asCPU(), equals([1, 2, 3, 4]));

      pt.setHostData(Float32List.fromList([5, 6, 7, 8]));
      expect(await pt.asCPU(), equals([5, 6, 7, 8]));
      // The GPU copy is refreshed from it on the next GPU read.
      expect(await (await pt.asGPU()).getData(), equals([5, 6, 7, 8]));
    });

    test('a GPU write is what the next host read returns', () async {
      final pt = PipelineTensor.cpu(Float32List.fromList([1, 2, 3, 4]), [4]);
      await pt.asGPU();

      pt.setGpuTensor(await gpuTensor([9, 8, 7, 6]));
      expect(await pt.asCPU(), equals([9, 8, 7, 6]));
    });
  });

  group('DynamicStage ops writing a key in place', () {
    test('a CPU op reads the previous CPU op\'s write', () async {
      final stage = StageBuilder('inplace_cpu_cpu')
          .executeCPU(
            (inputs, ranks, params) async => {
              'input': plusOne(inputs['input']!),
            },
            inputKey: 'input',
            outputKey: 'input',
          )
          .executeCPU(
            (inputs, ranks, params) async => {
              'result': plusOne(inputs['input']!),
            },
            inputKey: 'input',
            outputKey: 'result',
          )
          .build();

      final out = await stage.process({
        'input': await gpuTensor([1, 2, 3, 4]),
      });
      expect(await out['output']!.getData(), equals([3, 4, 5, 6]));
    });

    // The CPU write lands in the stage's own input tensor, a Tensor<TypedData>,
    // and the GPU op uploads it there. The upload used to cast the tensor to
    // Tensor<Float32List> and throw.
    test('a GPU op reads the previous CPU op\'s write', () async {
      final stage = StageBuilder('inplace_cpu_gpu')
          .executeCPU(
            (inputs, ranks, params) async => {
              'input': plusOne(inputs['input']!),
            },
            inputKey: 'input',
            outputKey: 'input',
          )
          .op(
            'multiplyScalar',
            arguments: [10.0],
            inputKey: 'input',
            outputKey: 'result',
          )
          .build();

      final out = await stage.process({
        'input': await gpuTensor([1, 2, 3, 4]),
      });
      expect(await out['output']!.getData(), equals([20, 30, 40, 50]));
    });

    test('a CPU op reads the previous GPU op\'s write', () async {
      final stage = StageBuilder('inplace_cpu_gpu_cpu')
          .executeCPU(
            (inputs, ranks, params) async => {'x': plusOne(inputs['input']!)},
            inputKey: 'input',
            outputKey: 'x',
          )
          .op(
            'multiplyScalar',
            arguments: [10.0],
            inputKey: 'x',
            outputKey: 'x',
          )
          .executeCPU(
            (inputs, ranks, params) async => {'result': plusOne(inputs['x']!)},
            inputKey: 'x',
            outputKey: 'result',
          )
          .build();

      final out = await stage.process({
        'input': await gpuTensor([1, 2, 3, 4]),
      });
      expect(await out['output']!.getData(), equals([21, 31, 41, 51]));
    });
  });
}
