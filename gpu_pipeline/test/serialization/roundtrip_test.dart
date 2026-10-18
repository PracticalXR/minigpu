import 'dart:convert';
import 'package:test/test.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';

void main() {
  group('Serialization Roundtrip', () {
    test('network encodes and decodes identically (structural)', () {
      final net = NetworkConfig(
        schemaVersion: 1,
        pipelines: [
          PipelineConfig(
            id: 'p1',
            queue: {'strategy': 'latest_only'},
            stages: [
              StageConfig(
                stageId: 's1',
                stageKind: 'DynamicStage',
                parameters: {'gain': 1.0},
                operations: [
                  OperationConfig(
                    opKind: 'tensor_op',
                    operation: 'addScalar',
                    inputKeys: ['input'],
                    outputKeys: ['output'],
                    arguments: {
                      'args': [1.5],
                    },
                  ),
                ],
              ),
            ],
            connections: [],
          ),
        ],
      );

      final jsonStr = jsonEncode(net.toJson());
      final decoded = NetworkConfig.fromJson(
        jsonDecode(jsonStr) as Map<String, dynamic>,
      );

      expect(decoded.schemaVersion, equals(1));
      expect(decoded.pipelines.first.id, equals('p1'));
      expect(
        decoded.pipelines.first.stages.first.operations.first.operation,
        equals('addScalar'),
      );
    });
  });
}
