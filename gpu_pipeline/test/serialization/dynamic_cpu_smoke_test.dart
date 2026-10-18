/// Pins the contract of the deprecated `dynamic_cpu` opKind.
///
/// Dart-source CPU stages (dart_eval) were removed because the analyzer /
/// dart:io import graph cannot compile for the web. A serialized network with
/// a `dynamic_cpu` step fails validation, and the operation factory refuses
/// to build one; both point at the replacement (opKind 'expression_cpu'). A
/// `DynamicCPUOperation` built in Dart still constructs, and fails at
/// EXECUTION with an error that points at `ExpressionCPUOperation`.
library;

import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_pipeline/src/serialization/operation_factory.dart';

const _dartCode = '''
Map<String, TypedData> main(
  Map<String, TypedData> inputs,
  Map<String, List<int>> ranks,
  Map<String, dynamic> parameters
) {
  return {'out': inputs['input'] as Float32List};
}
''';

final _config = OperationConfig(
  opKind: 'dynamic_cpu',
  inputKeys: ['input'],
  outputKeys: ['out'],
  dartCode: _dartCode,
);

void main() {
  test('a dynamic_cpu step fails validation', () {
    final net = NetworkConfig(
      schemaVersion: 1,
      pipelines: [
        PipelineConfig(
          id: 'p',
          queue: {'strategy': 'latest_only'},
          stages: [
            StageConfig(
              stageId: 's',
              stageKind: 'DynamicStage',
              operations: [_config],
            ),
          ],
          connections: [],
        ),
      ],
    );

    expect(
      () => PipelineLoader().loadFromJson(net.encode()),
      throwsA(
        isA<PipelineValidationException>().having(
          (e) => e.report.errors.map((issue) => issue.code),
          'error codes',
          contains('OP_KIND_UNSUPPORTED'),
        ),
      ),
    );
  });

  test('the operation factory refuses a dynamic_cpu config', () {
    expect(
      () => OperationFactory.create(_config),
      throwsA(
        isA<UnsupportedError>().having(
          (e) => e.message,
          'message',
          contains('expression_cpu'),
        ),
      ),
    );
  });

  test('a DynamicCPUOperation constructs, and execution points to the '
      'expression replacement', () async {
    // ignore: deprecated_member_use_from_same_package
    final op = DynamicCPUOperation(
      operationId: 'dynamic_cpu_smoke',
      dartCode: _dartCode,
      inputKeys: ['input'],
      outputKeys: ['out'],
    );

    await expectLater(
      op.executor.run(
        {
          'input': Float32List.fromList([1, 2, 3]),
        },
        {
          'input': [3],
        },
        {},
      ),
      throwsA(
        isA<UnimplementedError>().having(
          (e) => e.message,
          'message',
          contains('ExpressionCPUOperation'),
        ),
      ),
    );
  });
}
