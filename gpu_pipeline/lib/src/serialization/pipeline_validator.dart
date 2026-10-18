import 'pipeline_models.dart';
import 'shape_expr.dart';

class ValidationIssue {
  final String code;
  final String message;
  final String path;
  final bool warning;
  ValidationIssue({
    required this.code,
    required this.message,
    required this.path,
    this.warning = false,
  });
  Map<String, dynamic> toJson() => {
    'code': code,
    'message': message,
    'path': path,
    'severity': warning ? 'warning' : 'error',
  };
}

class ValidationReport {
  final List<ValidationIssue> errors;
  final List<ValidationIssue> warnings;
  ValidationReport({this.errors = const [], this.warnings = const []});
  bool get ok => errors.isEmpty;
  Map<String, dynamic> toJson() => {
    'ok': ok,
    'errors': errors.map((e) => e.toJson()).toList(),
    'warnings': warnings.map((w) => w.toJson()).toList(),
  };
}

class PipelineValidator {
  final ShapeExprEvaluator _shapeEval = ShapeExprEvaluator();

  ValidationReport validate(NetworkConfig net) {
    final errors = <ValidationIssue>[];
    final warnings = <ValidationIssue>[];

    // Schema version
    if (net.schemaVersion != 1) {
      errors.add(
        ValidationIssue(
          code: 'UNSUPPORTED_SCHEMA_VERSION',
          message: 'Unsupported schema_version ${net.schemaVersion}',
          path: 'schema_version',
        ),
      );
    }

    // Unique pipeline ids and collect port names
    final pipelineIds = <String>{};
    final inputsByPipeline = <String, Set<String>>{};
    final outputsByPipeline = <String, Set<String>>{};

    for (int pi = 0; pi < net.pipelines.length; pi++) {
      final p = net.pipelines[pi];
      final ppath = 'pipelines[$pi]';
      if (!pipelineIds.add(p.id)) {
        errors.add(
          ValidationIssue(
            code: 'DUP_PIPELINE_ID',
            message: 'Duplicate pipeline id ${p.id}',
            path: '$ppath.id',
          ),
        );
      }

      final stageIds = <String>{};
      final inNames = <String>{};
      final outNames = <String>{};

      for (int si = 0; si < p.stages.length; si++) {
        final s = p.stages[si];
        final spath = '$ppath.stages[$si]';
        if (!stageIds.add(s.stageId)) {
          errors.add(
            ValidationIssue(
              code: 'DUP_STAGE_ID',
              message: 'Duplicate stage id ${s.stageId}',
              path: '$spath.stage_id',
            ),
          );
        }

        // Collect port names for link validation later
        for (final port in s.inputs) {
          inNames.add(port.name);
        }
        for (final port in s.outputs) {
          outNames.add(port.name);
        }

        // Ports shape expr
        for (int oi = 0; oi < s.outputs.length; oi++) {
          final o = s.outputs[oi];
          if (o.shapeExpr != null) {
            try {
              _shapeEval.evaluate(
                o.shapeExpr!.raw,
                ShapeEvalContext(
                  inputShapes: const {},
                  parameters: s.parameters,
                ),
                path: '$spath.outputs[$oi].shape_expr',
              );
            } catch (e) {
              errors.add(
                ValidationIssue(
                  code: e is ShapeEvalError ? e.code : 'SHAPE_EVAL_ERROR',
                  message: e.toString(),
                  path: '$spath.outputs[$oi].shape_expr',
                ),
              );
            }
          }
        }

        // Operations basic checks
        for (int oi = 0; oi < s.operations.length; oi++) {
          final op = s.operations[oi];
          if (op.opKind.isEmpty) {
            errors.add(
              ValidationIssue(
                code: 'OP_KIND_MISSING',
                message: 'Missing op_kind',
                path: '$spath.operations[$oi].op_kind',
              ),
            );
          }
          if (op.opKind == 'tensor_op' && (op.operation == null)) {
            errors.add(
              ValidationIssue(
                code: 'OP_NAME_MISSING',
                message: 'tensor_op missing operation name',
                path: '$spath.operations[$oi].operation',
              ),
            );
          }
        }
      }

      // Stage connections sanity
      for (int ci = 0; ci < p.connections.length; ci++) {
        final c = p.connections[ci];
        final cpath = '$ppath.connections[$ci]';
        if (!stageIds.contains(c.fromStage)) {
          errors.add(
            ValidationIssue(
              code: 'CONNECTION_UNKNOWN_STAGE',
              message: 'from_stage ${c.fromStage} not found',
              path: '$cpath.from_stage',
            ),
          );
        }
        if (!stageIds.contains(c.toStage)) {
          errors.add(
            ValidationIssue(
              code: 'CONNECTION_UNKNOWN_STAGE',
              message: 'to_stage ${c.toStage} not found',
              path: '$cpath.to_stage',
            ),
          );
        }
      }

      inputsByPipeline[p.id] = inNames;
      outputsByPipeline[p.id] = outNames;
    }

    // Inter-pipeline links validation
    for (int li = 0; li < net.interPipelineLinks.length; li++) {
      final l = net.interPipelineLinks[li];
      final lpath = 'inter_pipeline_links[$li]';

      if (!pipelineIds.contains(l.fromPipeline)) {
        errors.add(
          ValidationIssue(
            code: 'LINK_UNKNOWN_FROM_PIPELINE',
            message: 'from_pipeline ${l.fromPipeline} not found',
            path: '$lpath.from_pipeline',
          ),
        );
      }
      if (!pipelineIds.contains(l.toPipeline)) {
        errors.add(
          ValidationIssue(
            code: 'LINK_UNKNOWN_TO_PIPELINE',
            message: 'to_pipeline ${l.toPipeline} not found',
            path: '$lpath.to_pipeline',
          ),
        );
      }

      final outs = outputsByPipeline[l.fromPipeline] ?? const {};
      final ins = inputsByPipeline[l.toPipeline] ?? const {};

      if (outs.isNotEmpty && !outs.contains(l.fromOutput)) {
        errors.add(
          ValidationIssue(
            code: 'LINK_UNKNOWN_FROM_OUTPUT',
            message:
                'Output ${l.fromOutput} not found in pipeline ${l.fromPipeline}',
            path: '$lpath.from_output',
          ),
        );
      }
      if (ins.isNotEmpty && !ins.contains(l.toInput)) {
        errors.add(
          ValidationIssue(
            code: 'LINK_UNKNOWN_TO_INPUT',
            message: 'Input ${l.toInput} not found in pipeline ${l.toPipeline}',
            path: '$lpath.to_input',
          ),
        );
      }

      // Mode warning if future modes appear
      switch (l.mode) {
        case InterPipelineLinkMode.latest:
        case InterPipelineLinkMode.latestAutorun:
          break;
      }
    }

    return ValidationReport(errors: errors, warnings: warnings);
  }
}
