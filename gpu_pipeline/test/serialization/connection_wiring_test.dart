import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_loader.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

void main() {
  test('explicit connection feeds next stage input', () async {
    final net = NetworkConfig(
      schemaVersion: 1,
      pipelines: [
        PipelineConfig(
          id: 'p',
          queue: {'strategy': 'latest_only'},
          stages: [
            StageConfig(
              stageId: 's1',
              stageKind: 'DynamicStage',
              operations: [
                OperationConfig(
                  opKind: 'tensor_op',
                  operation: 'addScalar',
                  inputKeys: ['input'],
                  outputKeys: ['s1_out'],
                  arguments: {
                    'args': [1.0],
                  },
                ),
              ],
            ),
            StageConfig(
              stageId: 's2',
              stageKind: 'DynamicStage',
              operations: [
                OperationConfig(
                  opKind: 'tensor_op',
                  operation: 'addScalar',
                  inputKeys: ['mid'],
                  outputKeys: ['output'],
                  arguments: {
                    'args': [2.0],
                  },
                ),
              ],
            ),
          ],
          connections: [
            ConnectionConfig(
              fromStage: 's1',
              toStage: 's2',
              fromOutput: 'output',
              toInput: 'mid',
            ),
          ],
        ),
      ],
    );

    final loader = PipelineLoader();
    final load = loader.loadFromJson(net.encode());
    final pipeline = load.runtime.pipelines.first;

    final t = await Tensor.zeros([2]);
    await t.write(Float32List.fromList([10, 20]));

    final outputs = await pipeline.runOnce({'input': t});
    final out = outputs['output']!;
    final data = await out.getData() as Float32List;
    expect(data.toList(), equals([13, 23])); // stage1 +1, stage2 +2
  });
}
