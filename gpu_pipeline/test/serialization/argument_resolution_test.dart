import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_loader.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

void main() {
  group('Argument Resolution', () {
    test('param reference resolves in tensor operation', () async {
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
                parameters: {'delta': 2.0},
                operations: [
                  OperationConfig(
                    opKind: 'tensor_op',
                    operation: 'addScalar',
                    inputKeys: ['input'],
                    outputKeys: ['output'],
                    arguments: {
                      'args': ['param:delta'], // should resolve to 2.0
                    },
                  ),
                ],
              ),
            ],
            connections: [],
          ),
        ],
      );

      final loader = PipelineLoader();
      final load = loader.loadFromJson(net.encode());

      expect(load.validation.ok, isTrue);

      final pipeline = load.runtime.pipelines.first;

      final t = await Tensor.zeros([4]);
      await t.write(Float32List.fromList([1, 1, 1, 1]));

      // Assuming pipeline runOnce or a similar API; adjust if different
      final outputs = await pipeline.runOnce({'input': t});
      final outTensor = outputs['output']!;
      final data = await outTensor.getData() as Float32List;
      expect(data.toList(), equals([3, 3, 3, 3])); // now only added once
    });
  });
}
