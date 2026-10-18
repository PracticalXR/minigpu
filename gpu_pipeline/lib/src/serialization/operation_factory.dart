import 'package:gpu_pipeline/src/pipeline_ports.dart';

import '../operation/tensor_stage_operation.dart';
import '../operation/cpu_stage_operation.dart';
import '../operation/expression_cpu_operation.dart';
import '../operation/shader_stage_operation.dart';
import '../operation/stage_operation.dart';
import 'pipeline_models.dart';

class OperationFactory {
  /// Builds the operation [c] describes.
  ///
  /// Throws an [ArgumentError] for a `tensor_op` or `cpu` operation without
  /// an `operation` name, and an [UnsupportedError] for an op_kind that
  /// cannot run: `dynamic_cpu`, `merge`, `split` or an unknown one.
  static StageOperation create(
    OperationConfig c, {
    Map<String, OutputPort>? outputPorts,
  }) {
    switch (c.opKind) {
      case 'tensor_op':
        final operation = _operationName(c);
        return TensorOperation(
          operationId:
              'tensor_${operation}_${DateTime.now().microsecondsSinceEpoch}',
          operation: operation,
          // Omitted keys default to ['input'] and ['output'].
          inputKeys: c.inputKeys.isEmpty ? null : c.inputKeys,
          outputKeys: c.outputKeys.isEmpty ? null : c.outputKeys,
          arguments: c.arguments,
          namedArguments: c.namedArguments,
        );
      case 'cpu':
        final operation = _operationName(c);
        return PrebuiltCPUOperation(
          operationId:
              'cpu_${operation}_${DateTime.now().microsecondsSinceEpoch}',
          operation: operation,
          operationParameters: c.namedArguments,
          arguments: c.arguments,
          inputKeys: c.inputKeys,
          outputKeys: c.outputKeys,
        );
      case 'expression_cpu':
        // Per-element math expressions compiled once at build (web+native —
        // no interpreter, no dart_eval).  The expressions map travels in
        // namedArguments: {outputKey: expression}; initial parameters in
        // arguments.
        return ExpressionCPUOperation(
          operationId: 'expr_cpu_${DateTime.now().microsecondsSinceEpoch}',
          expressions: c.namedArguments.map(
            (k, v) => MapEntry(k, v.toString()),
          ),
          initialParameters: c.arguments,
          inputKeys: c.inputKeys.isEmpty ? const ['input'] : c.inputKeys,
          outputKeys: c.outputKeys.isEmpty ? null : c.outputKeys,
        );
      case 'shader':
        return ShaderStageOperation(
          operationId: 'shader_${DateTime.now().microsecondsSinceEpoch}',
          shaderTemplate: c.wgsl ?? '',
          inputKeys: c.inputKeys,
          outputKeys: c.outputKeys,
          shaderParameters: c.shaderParameters ?? const {},
          workgroupSize: c.workgroupSize,
          arguments: c.arguments,
          namedArguments: c.namedArguments,
          outputPorts: outputPorts ?? const {},
        );
      case 'dynamic_cpu':
        throw UnsupportedError(
          'dynamic_cpu operations cannot run, because Dart source is not '
          'executed; use expression_cpu for per-element math',
        );
      case 'merge' || 'split':
        throw UnsupportedError(
          '${c.opKind} operations have no implementation; write the step as '
          'a shader operation',
        );
      default:
        throw UnsupportedError('Unknown op_kind ${c.opKind}');
    }
  }

  static String _operationName(OperationConfig c) {
    final name = c.operation;
    if (name == null || name.isEmpty) {
      throw ArgumentError('A ${c.opKind} operation needs an operation name');
    }
    return name;
  }
}
