import 'package:gpu_pipeline/src/pipeline_ports.dart';

import '../operation/tensor_stage_operation.dart';
import '../operation/cpu_stage_operation.dart';
import '../operation/expression_cpu_operation.dart';
import '../operation/shader_stage_operation.dart';
import '../operation/stage_operation.dart';
import 'pipeline_models.dart';

class OperationFactory {
  static StageOperation create(
    OperationConfig c, {
    Map<String, OutputPort>? outputPorts,
  }) {
    switch (c.opKind) {
      case 'tensor_op':
        return TensorOperation(
          operationId:
              'tensor_${c.operation}_${DateTime.now().microsecondsSinceEpoch}',
          operation: c.operation!,
          inputKeys: c.inputKeys,
          outputKeys: c.outputKeys,
          arguments: c.arguments,
          namedArguments: c.namedArguments,
        );
      case 'cpu':
        return PrebuiltCPUOperation(
          operationId:
              'cpu_${c.operation}_${DateTime.now().microsecondsSinceEpoch}',
          operation: c.operation!,
          operationParameters: c.namedArguments,
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
      case 'dynamic_cpu':
        var normalized = _normalizeDynamicCpuSource(c.dartCode ?? '');
        // Ensure import
        if (RegExp(
              r'\b(TypedData|Float32List|Int32List)\b',
            ).hasMatch(normalized) &&
            !RegExp(r"import\s+'dart:typed_data'").hasMatch(normalized)) {
          normalized = "import 'dart:typed_data';\n$normalized";
        }
        return DynamicCPUOperation(
          operationId: 'dyn_cpu_${DateTime.now().microsecondsSinceEpoch}',
          dartCode: normalized,
          inputKeys: c.inputKeys,
          outputKeys: c.outputKeys,
          initialParameters: c.namedArguments ?? const {},
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
      case 'merge':
        return ShaderStageOperation(
          operationId: 'merge_${DateTime.now().microsecondsSinceEpoch}',
          shaderTemplate: '// TODO merge shader',
          outputPorts: outputPorts ?? const {},
          inputKeys: c.inputKeys,
          outputKeys: c.outputKeys,
        );
      case 'split':
        return ShaderStageOperation(
          operationId: 'split_${DateTime.now().microsecondsSinceEpoch}',
          shaderTemplate: '// TODO split shader',
          outputPorts: outputPorts ?? const {},
          inputKeys: c.inputKeys,
          outputKeys: c.outputKeys,
        );
      default:
        throw UnsupportedError('Unknown op_kind ${c.opKind}');
    }
  }

  static String _normalizeDynamicCpuSource(String raw) {
    var body = raw.trim();
    if (body.isEmpty) {
      return '''
import 'dart:typed_data';

Map<String, TypedData> main(
  Map<String, TypedData> inputs,
  Map<String, List<int>> ranks,
  Map<String, dynamic> parameters
) {
  return <String, TypedData>{};
}
'''
          .trim();
    }
    final needsImport = RegExp(
      r'\b(TypedData|Float32List|Int32List)\b',
    ).hasMatch(body);
    final hasImport = RegExp(r"import\s+'dart:typed_data'").hasMatch(body);
    if (needsImport && !hasImport) {
      body = "import 'dart:typed_data';\n$body";
    }
    final hasMainSig = RegExp(
      r'Map<String,\s*TypedData>\s+main\s*\(\s*Map<String,\s*TypedData>\s*inputs,',
    ).hasMatch(body);
    if (hasMainSig) return body;
    if (body.contains('main(')) return body;
    return '''
${hasImport ? '' : "import 'dart:typed_data';"}

Map<String, TypedData> main(
  Map<String, TypedData> inputs,
  Map<String, List<int>> ranks,
  Map<String, dynamic> parameters
) {
  $body
}
'''
        .trim();
  }
}
