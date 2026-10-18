/// Pins the fail-gracefully contract of the retired `dynamic_cpu` opKind.
///
/// Dart-source CPU stages (dart_eval) were removed because the analyzer /
/// dart:io import graph cannot compile for the web.  Old serialized configs
/// may still carry `opKind: 'dynamic_cpu'` — they must keep DESERIALIZING
/// (construction succeeds) and fail at EXECUTION with an error that points
/// at the replacement (ExpressionCPUOperation, opKind 'expression_cpu').
library;

import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_pipeline/src/serialization/operation_factory.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';

void main() {
  test('dynamic_cpu configs still construct, and execution points to the '
      'expression replacement', () async {
    final op = OperationFactory.create(
      OperationConfig(
        opKind: 'dynamic_cpu',
        inputKeys: ['input'],
        outputKeys: ['out'],
        dartCode: '''
Map<String, TypedData> main(
  Map<String, TypedData> inputs,
  Map<String, List<int>> ranks,
  Map<String, dynamic> parameters
) {
  return {'out': inputs['input'] as Float32List};
}
''',
      ),
    );
    expect(op, isA<DynamicCPUOperation>());

    await expectLater(
      (op as DynamicCPUOperation).executor.run(
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
