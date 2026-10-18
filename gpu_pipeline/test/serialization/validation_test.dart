import 'package:test/test.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_validator.dart';

void main() {
  group('Validation', () {
    test('duplicate stage id detected', () {
      final net = NetworkConfig(
        schemaVersion: 1,
        pipelines: [
          PipelineConfig(
            id: 'p',
            queue: {'strategy': 'latest_only'},
            stages: [
              StageConfig(stageId: 's', stageKind: 'DynamicStage'),
              StageConfig(stageId: 's', stageKind: 'DynamicStage'),
            ],
            connections: [],
          ),
        ],
      );
      final report = PipelineValidator().validate(net);
      expect(report.ok, isFalse);
      expect(report.errors.any((e) => e.code == 'DUP_STAGE_ID'), isTrue);
    });

    test('unsupported schema version', () {
      final net = NetworkConfig(schemaVersion: 99, pipelines: []);
      final report = PipelineValidator().validate(net);
      expect(report.ok, isFalse);
      expect(report.errors.first.code, equals('UNSUPPORTED_SCHEMA_VERSION'));
    });

    test('missing op_kind', () {
      final stage = StageConfig(
        stageId: 's1',
        stageKind: 'DynamicStage',
        operations: [
          OperationConfig(opKind: ''), // invalid
        ],
      );
      final net = NetworkConfig(
        schemaVersion: 1,
        pipelines: [
          PipelineConfig(
            id: 'p',
            queue: {'strategy': 'latest_only'},
            stages: [stage],
            connections: [],
          ),
        ],
      );
      final report = PipelineValidator().validate(net);
      expect(report.ok, isFalse);
      expect(report.errors.any((e) => e.code == 'OP_KIND_MISSING'), isTrue);
    });
  });
}
