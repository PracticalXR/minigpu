import 'dart:typed_data';

import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';

/// Counts the runs it is handed and runs each one inline.
class _CountingExecutor extends StageExecutor {
  _CountingExecutor(this.processor);

  final StageProcessor processor;
  int runs = 0;

  @override
  String get backend => 'counting';

  @override
  Future<Map<String, TypedData>> run(
    Map<String, TypedData> inputs,
    Map<String, List<int>> ranks,
    Map<String, dynamic> parameters,
  ) {
    runs++;
    return processor(inputs, ranks, parameters);
  }
}

void main() {
  // A DynamicStage used to call a CPU operation's processor itself, so the
  // executor the operation was built with (a worker isolate, say) never ran.
  test('a DynamicStage runs a CPU operation through its executor', () async {
    Future<Map<String, TypedData>> twice(
      Map<String, TypedData> inputs,
      Map<String, List<int>> ranks,
      Map<String, dynamic> parameters,
    ) async => {
      'output': Float32List.fromList([
        for (final v in inputs['input']! as Float32List) v * 2,
      ]),
    };

    final executor = _CountingExecutor(twice);
    final stage = DynamicStage(
      stageId: 'executor_cpu',
      operations: [
        CPUStageOperation(
          operationId: 'twice',
          processor: twice,
          executor: executor,
          inputKeys: ['input'],
          outputKeys: ['output'],
        ),
      ],
      parameterDefinitions: const {},
    );

    final input = await Tensor.zeros([4]);
    await input.write(Float32List.fromList([1, 2, 3, 4]));
    final out = await stage.process({'input': input});

    expect(await out['output']!.getData(), equals([2, 4, 6, 8]));
    expect(executor.runs, 1);
    await stage.dispose();
  });
}
