import '../operation/cpu_stage_operation.dart';
import '../pipeline_assets.dart' show assetSourceProblem;
import '../operation/expression.dart';
import '../operation/tensor_stage_operation.dart';
import 'connection_graph.dart';
import 'pipeline_models.dart';
import 'shape_expr.dart';

/// One problem [PipelineValidator] found, with the JSON path it is at.
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

/// Checks a [NetworkConfig] before it is built.
///
/// An error describes a network that would fail to build or run, and
/// `PipelineLoader.loadFromJson` refuses it. A warning describes something
/// the network runs without: a field that is accepted but has no effect, or
/// an optional asset that cannot load, which `Pipeline.start` skips.
class PipelineValidator {
  final ShapeExprEvaluator _shapeEval = ShapeExprEvaluator();

  static const _queueStrategies = {
    'latest_only',
    'latest_n',
    'drop_on_overrun',
    'batched',
    'process_all',
  };

  static const _assetStrategies = {'preload', 'lazy', 'per_call', 'stream'};

  ValidationReport validate(NetworkConfig net) {
    final issues = _Issues();

    if (net.schemaVersion != 1) {
      issues.error(
        'UNSUPPORTED_SCHEMA_VERSION',
        'Unsupported schema_version ${net.schemaVersion}',
        'schema_version',
      );
    }

    final pipelines = <String, _PipelineInfo>{};
    for (var pi = 0; pi < net.pipelines.length; pi++) {
      final p = net.pipelines[pi];
      final ppath = 'pipelines[$pi]';
      final duplicate = pipelines.containsKey(p.id);
      if (duplicate) {
        issues.error(
          'DUP_PIPELINE_ID',
          'Duplicate pipeline id ${p.id}',
          '$ppath.id',
        );
      }
      final info = _validatePipeline(p, ppath, issues);
      if (!duplicate) pipelines[p.id] = info;
    }

    _validateLinks(net.interPipelineLinks, pipelines, issues);
    _validateAssets(net, issues);
    return ValidationReport(errors: issues.errors, warnings: issues.warnings);
  }

  void _validateAssets(NetworkConfig net, _Issues issues) {
    final names = <String>{};
    for (var ai = 0; ai < net.assets.length; ai++) {
      final a = net.assets[ai];
      final apath = 'assets[$ai]';
      if (!names.add(a.name)) {
        issues.error(
          'DUP_ASSET_NAME',
          'Duplicate asset name ${a.name}',
          '$apath.name',
        );
      }
      if (!_assetStrategies.contains(a.strategy)) {
        issues.error(
          'ASSET_STRATEGY_UNKNOWN',
          'Unknown asset strategy "${a.strategy}"; expected '
              '${_assetStrategies.join(', ')}',
          '$apath.strategy',
        );
      }
      final problem = assetSourceProblem(a);
      if (problem == null) continue;
      // A pipeline skips an optional asset that fails to load.
      if (a.required) {
        issues.error('ASSET_SOURCE_INVALID', problem, '$apath.source');
      } else {
        issues.warning('ASSET_SOURCE_INVALID', problem, '$apath.source');
      }
    }

    for (var pi = 0; pi < net.pipelines.length; pi++) {
      final stages = net.pipelines[pi].stages;
      for (var si = 0; si < stages.length; si++) {
        final listed = stages[si].assets;
        for (var k = 0; k < listed.length; k++) {
          if (names.contains(listed[k])) continue;
          issues.warning(
            'ASSET_UNDECLARED',
            'Stage ${stages[si].stageId} lists asset "${listed[k]}", which '
                'the network does not declare',
            'pipelines[$pi].stages[$si].assets[$k]',
          );
        }
      }
    }
  }

  _PipelineInfo _validatePipeline(
    PipelineConfig p,
    String ppath,
    _Issues issues,
  ) {
    _validateQueue(p.queue, '$ppath.queue', issues);

    final stages = <String, StageConfig>{};
    for (var si = 0; si < p.stages.length; si++) {
      final s = p.stages[si];
      final spath = '$ppath.stages[$si]';
      if (stages.containsKey(s.stageId)) {
        issues.error(
          'DUP_STAGE_ID',
          'Duplicate stage id ${s.stageId}',
          '$spath.stage_id',
        );
      } else {
        stages[s.stageId] = s;
      }
      _validateStage(s, spath, issues);
    }

    final fed = <String>{};
    final order = _validateConnections(p, ppath, stages, fed, issues);
    return _PipelineInfo(stages, order, fed, p.connections);
  }

  void _validateQueue(Map<String, dynamic> queue, String path, _Issues issues) {
    final strategy = queue['strategy'];
    if (strategy != null && !_queueStrategies.contains(strategy)) {
      issues.error(
        'QUEUE_STRATEGY_UNKNOWN',
        'Unknown queue strategy "$strategy"; expected '
            '${_queueStrategies.join(', ')}',
        '$path.strategy',
      );
    }
    for (final key in const [
      'max_items_per_stream',
      'max_total_items',
      'batch_size',
    ]) {
      final value = queue[key];
      if (value != null && (value is! int || value < 1)) {
        issues.error(
          'QUEUE_INVALID',
          '$key must be a positive integer, got $value',
          '$path.$key',
        );
      }
    }
  }

  void _validateStage(StageConfig s, String spath, _Issues issues) {
    if (s.stageKind != 'DynamicStage') {
      issues.warning(
        'STAGE_KIND_UNKNOWN',
        'stage_kind "${s.stageKind}" has no implementation of its own; the '
            'stage is built as a DynamicStage from its ports, parameters and '
            'operations',
        '$spath.stage_kind',
      );
    }

    _validatePorts(s.inputs, '$spath.inputs', const [
      'flexible_input',
      'exact_input',
    ], issues);
    _validatePorts(s.outputs, '$spath.outputs', const [
      'dynamic_output',
    ], issues);
    for (var oi = 0; oi < s.outputs.length; oi++) {
      _validateOutputShape(
        s,
        s.outputs[oi],
        '$spath.outputs[$oi].shape_expr',
        issues,
      );
    }

    for (var oi = 0; oi < s.operations.length; oi++) {
      _validateOperation(
        s,
        s.operations[oi],
        '$spath.operations[$oi]',
        issues,
      );
    }
  }

  void _validatePorts(
    List<PortConfig> ports,
    String path,
    List<String> portTypes,
    _Issues issues,
  ) {
    final names = <String>{};
    for (var i = 0; i < ports.length; i++) {
      final port = ports[i];
      final ppath = '$path[$i]';
      if (!names.add(port.name)) {
        issues.error(
          'DUP_PORT_NAME',
          'Duplicate port name ${port.name}',
          '$ppath.name',
        );
      }
      if (!portTypes.contains(port.portType)) {
        issues.error(
          'PORT_TYPE_UNKNOWN',
          'Port ${port.name} has port_type "${port.portType}"; expected '
              '${portTypes.join(' or ')}',
          '$ppath.port_type',
        );
      }
      if (port.portType == 'exact_input' && exactPortShape(port) == null) {
        issues.error(
          'PORT_SHAPE_INVALID',
          'exact_input port ${port.name} needs a shape_expr '
              '{"op": "fixed", "shape": [...]} whose dims are positive '
              'integers',
          '$ppath.shape_expr',
        );
      }
      final min = port.minDimensions;
      final max = port.maxDimensions;
      if ((min != null && min < 0) ||
          (max != null && max < 0) ||
          (min != null && max != null && min > max)) {
        issues.error(
          'PORT_DIMENSIONS_INVALID',
          'Port ${port.name} has min_dimensions $min and max_dimensions $max',
          ppath,
        );
      }
    }
  }

  void _validateOutputShape(
    StageConfig s,
    PortConfig port,
    String path,
    _Issues issues,
  ) {
    final expr = port.shapeExpr;
    if (expr == null) return;
    final raw = expr.raw;
    switch (expr.op) {
      case 'fixed':
        try {
          _shapeEval.evaluate(
            raw,
            ShapeEvalContext(inputShapes: const {}, parameters: s.parameters),
            path: path,
          );
        } on ShapeEvalError catch (e) {
          issues.error(e.code, e.message, path);
        } catch (e) {
          issues.error('SHAPE_EVAL_ERROR', 'Cannot evaluate: $e', path);
        }
      case 'same_as' || 'reshape_flat':
        _validateShapeInputs(s, port, [raw['input']], path, issues);
      case 'broadcast_pair':
        _validateShapeInputs(s, port, [raw['a'], raw['b']], path, issues);
      case 'concat_along':
        final inputs = raw['inputs'];
        if (inputs is List && inputs.isNotEmpty) {
          _validateShapeInputs(s, port, inputs, path, issues);
        } else {
          issues.error(
            'SHAPE_RESULT_EMPTY',
            'concat_along needs a non-empty inputs list',
            path,
          );
        }
        final axis = raw['axis'] ?? 0;
        if (axis is! int || axis < 0) {
          issues.error(
            'SHAPE_AXIS_OOB',
            'concat_along axis must be a non-negative integer, got $axis',
            path,
          );
        }
      default:
        issues.error(
          'SHAPE_OP_UNKNOWN',
          'Unknown shape_expr op "${expr.op}"; expected fixed, same_as, '
              'concat_along, broadcast_pair or reshape_flat',
          path,
        );
    }
  }

  /// Checks that the inputs a shape expression names are ones its
  /// calculation receives: the input keys of the shader operations that
  /// write [port], or, for a port no shader operation writes, any input or
  /// key of the stage.
  void _validateShapeInputs(
    StageConfig s,
    PortConfig port,
    List<dynamic> names,
    String path,
    _Issues issues,
  ) {
    final writers = s.operations.where(
      (o) => o.opKind == 'shader' && o.outputKeys.contains(port.name),
    );
    final visible = writers.isNotEmpty
        ? {for (final o in writers) ...o.inputKeys}
        : {
            'input',
            for (final p in s.inputs) p.name,
            for (final o in s.operations) ...o.inputKeys,
            for (final o in s.operations) ...o.outputKeys,
          };
    for (final name in names) {
      if (name is! String || !visible.contains(name)) {
        issues.error(
          'SHAPE_INPUT_UNKNOWN',
          'The shape of port ${port.name} refers to input $name, which its '
              'calculation does not receive; it receives '
              '${visible.isEmpty ? 'no inputs' : visible.join(', ')}',
          path,
        );
      }
    }
  }

  void _validateOperation(
    StageConfig s,
    OperationConfig op,
    String path,
    _Issues issues,
  ) {
    switch (op.opKind) {
      case '':
        issues.error('OP_KIND_MISSING', 'Missing op_kind', '$path.op_kind');
      case 'tensor_op':
        final name = _operationName(op, path, issues);
        if (name != null) {
          TensorOperationRegistry.ensureInitialized();
          if (!TensorOperationRegistry.hasOperation(name)) {
            issues.error(
              'OP_NAME_UNKNOWN',
              'No tensor operation "$name" is registered; '
                  'TensorOperationRegistry.availableOperations lists them',
              '$path.operation',
            );
          }
        }
      case 'cpu':
        final name = _operationName(op, path, issues);
        if (name != null && !CPUOperationRegistry.has(name)) {
          issues.error(
            'OP_NAME_UNKNOWN',
            'No CPU operation "$name" is registered; '
                'CPUOperationRegistry.availableOperations lists them',
            '$path.operation',
          );
        }
      case 'expression_cpu':
        _validateExpressions(s, op, path, issues);
      case 'shader':
        if ((op.wgsl ?? '').trim().isEmpty) {
          issues.error(
            'SHADER_SOURCE_MISSING',
            'A shader operation needs its WGSL source in wgsl',
            '$path.wgsl',
          );
        }
        if (op.arguments.isNotEmpty || op.namedArguments.isNotEmpty) {
          issues.warning(
            'ARGUMENTS_UNUSED',
            'Shader operations ignore arguments and named_arguments; they '
                'read shader_parameters and the stage parameters',
            path,
          );
        }
      case 'dynamic_cpu':
        issues.error(
          'OP_KIND_UNSUPPORTED',
          'dynamic_cpu operations cannot run, because Dart source is not '
              'executed; use expression_cpu for per-element math',
          '$path.op_kind',
        );
      case 'merge' || 'split':
        issues.error(
          'OP_KIND_UNSUPPORTED',
          '${op.opKind} operations have no implementation; write the step '
              'as a shader operation',
          '$path.op_kind',
        );
      default:
        issues.error(
          'OP_KIND_UNKNOWN',
          'Unknown op_kind "${op.opKind}"; expected tensor_op, cpu, '
              'expression_cpu or shader',
          '$path.op_kind',
        );
    }
  }

  String? _operationName(OperationConfig op, String path, _Issues issues) {
    final name = op.operation;
    if (name == null || name.isEmpty) {
      issues.error(
        'OP_NAME_MISSING',
        '${op.opKind} missing operation name',
        '$path.operation',
      );
      return null;
    }
    return name;
  }

  /// Compiles each expression, and checks that every name it uses is bound
  /// when it runs: a numeric stage parameter, a key of `arguments`, or an
  /// input other than the one bound to `x`.
  void _validateExpressions(
    StageConfig s,
    OperationConfig op,
    String path,
    _Issues issues,
  ) {
    if (op.namedArguments.isEmpty) {
      issues.error(
        'EXPRESSION_MISSING',
        'An expression_cpu operation needs named_arguments mapping each '
            'output key to an expression',
        '$path.named_arguments',
      );
      return;
    }
    final inputKeys = op.inputKeys.isEmpty ? const ['input'] : op.inputKeys;
    final primary = inputKeys.contains('input') ? 'input' : inputKeys.first;
    final bound = <String>{
      for (final e in s.parameters.entries)
        if (e.value is num) e.key,
      ...op.arguments.keys,
      for (final key in inputKeys)
        if (key != primary) key,
    };
    for (final e in op.namedArguments.entries) {
      final epath = '$path.named_arguments.${e.key}';
      final CompiledExpression compiled;
      try {
        compiled = CompiledExpression.compile(e.value.toString());
      } on FormatException catch (err) {
        issues.error(
          'EXPRESSION_INVALID',
          'Expression for ${e.key}: ${err.message}',
          epath,
        );
        continue;
      }
      for (final name in compiled.identifiers) {
        if (!bound.contains(name)) {
          issues.error(
            'EXPRESSION_UNKNOWN_NAME',
            'Expression for ${e.key} uses $name, which is not a numeric '
                'stage parameter, an argument or a second input',
            epath,
          );
        }
      }
    }
  }

  /// Checks [p]'s connections, records in [fed] each stage input they feed,
  /// and returns the stage run order, or null when the connections cannot
  /// order the stages.
  List<String>? _validateConnections(
    PipelineConfig p,
    String ppath,
    Map<String, StageConfig> stages,
    Set<String> fed,
    _Issues issues,
  ) {
    var stagesKnown = true;
    for (var ci = 0; ci < p.connections.length; ci++) {
      final c = p.connections[ci];
      final cpath = '$ppath.connections[$ci]';
      final from = stages[c.fromStage];
      final to = stages[c.toStage];
      if (from == null) {
        stagesKnown = false;
        issues.error(
          'CONNECTION_UNKNOWN_STAGE',
          'from_stage ${c.fromStage} not found',
          '$cpath.from_stage',
        );
      } else {
        _validateConnectionSource(from, c, cpath, issues);
      }
      if (to == null) {
        stagesKnown = false;
        issues.error(
          'CONNECTION_UNKNOWN_STAGE',
          'to_stage ${c.toStage} not found',
          '$cpath.to_stage',
        );
      } else {
        final input = _validateConnectionTarget(to, c, cpath, issues);
        if (input != null && !fed.add(_key(to.stageId, input))) {
          issues.error(
            'CONNECTION_DUPLICATE_INPUT',
            'Input $input of stage ${to.stageId} is fed by more than one '
                'connection',
            cpath,
          );
        }
      }
    }
    if (!stagesKnown) return null;

    try {
      return ConnectionGraph.build(p).topologicalOrder(stages.keys);
    } on StateError catch (e) {
      issues.error('CONNECTION_CYCLE', e.message, '$ppath.connections');
      return null;
    }
  }

  void _validateConnectionSource(
    StageConfig from,
    ConnectionConfig c,
    String cpath,
    _Issues issues,
  ) {
    final outputs = _outputNames(from);
    final name = c.fromOutput;
    if (name != null) {
      if (!outputs.contains(name)) {
        issues.error(
          'CONNECTION_UNKNOWN_OUTPUT',
          'Stage ${from.stageId} has no output $name; its outputs are '
              '${outputs.join(', ')}',
          '$cpath.from_output',
        );
      }
    } else if (defaultConnectionOutput(outputs) == null) {
      issues.error(
        'CONNECTION_AMBIGUOUS_OUTPUT',
        'Stage ${from.stageId} has several outputs (${outputs.join(', ')}) '
            'and none is named output; set from_output',
        cpath,
      );
    }
  }

  /// Returns the input [c] feeds on [to], or null when there is none.
  String? _validateConnectionTarget(
    StageConfig to,
    ConnectionConfig c,
    String cpath,
    _Issues issues,
  ) {
    final ports = [for (final port in to.inputs) port.name];
    final name = c.toInput;
    if (name != null) {
      if (ports.isNotEmpty && !ports.contains(name)) {
        issues.error(
          'CONNECTION_UNKNOWN_INPUT',
          'Stage ${to.stageId} has no input port $name; its input ports are '
              '${ports.join(', ')}',
          '$cpath.to_input',
        );
        return null;
      }
      return name;
    }
    final input = defaultConnectionInput(ports);
    if (input == null) {
      issues.error(
        'CONNECTION_AMBIGUOUS_INPUT',
        'Stage ${to.stageId} has several input ports (${ports.join(', ')}) '
            'and none is named input; set to_input',
        cpath,
      );
    }
    return input;
  }

  void _validateLinks(
    List<InterPipelineLinkConfig> links,
    Map<String, _PipelineInfo> pipelines,
    _Issues issues,
  ) {
    final fedInputs = <String>{};
    final autorun = <String, Set<String>>{};
    for (var li = 0; li < links.length; li++) {
      final l = links[li];
      final lpath = 'inter_pipeline_links[$li]';
      final from = pipelines[l.fromPipeline];
      final to = pipelines[l.toPipeline];

      if (from == null) {
        issues.error(
          'LINK_UNKNOWN_FROM_PIPELINE',
          'from_pipeline ${l.fromPipeline} not found',
          '$lpath.from_pipeline',
        );
      } else {
        final last = from.finalStage;
        final outputs = last == null ? null : _outputNames(last);
        if (outputs != null && !outputs.contains(l.fromOutput)) {
          issues.error(
            'LINK_UNKNOWN_FROM_OUTPUT',
            'Pipeline ${l.fromPipeline} ends with stage ${last!.stageId}, '
                'which has no output ${l.fromOutput}; its outputs are '
                '${outputs.join(', ')}',
            '$lpath.from_output',
          );
        }
      }

      if (to == null) {
        issues.error(
          'LINK_UNKNOWN_TO_PIPELINE',
          'to_pipeline ${l.toPipeline} not found',
          '$lpath.to_pipeline',
        );
      } else if (!to.readsRunInput(l.toInput)) {
        issues.error(
          'LINK_UNKNOWN_TO_INPUT',
          'No stage of pipeline ${l.toPipeline} reads an input named '
              '${l.toInput}',
          '$lpath.to_input',
        );
      }

      if (!fedInputs.add(_key(l.toPipeline, l.toInput))) {
        issues.warning(
          'LINK_DUPLICATE_INPUT',
          'Input ${l.toInput} of pipeline ${l.toPipeline} is fed by more '
              'than one link; the source that finishes last wins',
          lpath,
        );
      }
      if (l.mode == InterPipelineLinkMode.latestAutorun) {
        autorun.putIfAbsent(l.fromPipeline, () => <String>{}).add(l.toPipeline);
      }
    }

    final loop = _loop(autorun);
    if (loop != null) {
      issues.error(
        'LINK_AUTORUN_CYCLE',
        'latest_autorun links form a loop (${loop.join(' -> ')}), so each '
            'run would start the next forever',
        'inter_pipeline_links',
      );
    }
  }

  /// A loop in [edges], as the ids along it with the first repeated at the
  /// end, or null when there is none.
  static List<String>? _loop(Map<String, Set<String>> edges) {
    final done = <String>{};
    final path = <String>[];
    List<String>? visit(String id) {
      final at = path.indexOf(id);
      if (at >= 0) return [...path.sublist(at), id];
      if (!done.add(id)) return null;
      path.add(id);
      for (final next in edges[id] ?? const <String>{}) {
        final loop = visit(next);
        if (loop != null) return loop;
      }
      path.removeLast();
      return null;
    }

    for (final id in edges.keys) {
      final loop = visit(id);
      if (loop != null) return loop;
    }
    return null;
  }
}

/// The shape of an `exact_input` port: its shape_expr's `shape`, when that
/// is a non-empty list of positive integers; otherwise null.
List<int>? exactPortShape(PortConfig port) {
  // Decoded JSON lists are List<dynamic>, so check the elements, not the type.
  final shape = port.shapeExpr?.raw['shape'];
  if (shape is! List || shape.isEmpty) return null;
  if (!shape.every((d) => d is int && d > 0)) return null;
  return List<int>.from(shape);
}

/// The outputs a built stage returns: its output ports, or `output` for a
/// stage that declares none.
List<String> _outputNames(StageConfig s) =>
    s.outputs.isEmpty ? const ['output'] : [for (final p in s.outputs) p.name];

String _key(String owner, String name) => '$owner\u0000$name';

class _Issues {
  final errors = <ValidationIssue>[];
  final warnings = <ValidationIssue>[];

  void error(String code, String message, String path) =>
      errors.add(ValidationIssue(code: code, message: message, path: path));

  void warning(String code, String message, String path) => warnings.add(
    ValidationIssue(code: code, message: message, path: path, warning: true),
  );
}

class _PipelineInfo {
  _PipelineInfo(this.stages, this.order, this.fedInputs, this.connections);

  final Map<String, StageConfig> stages;

  /// The stage run order, or null when the connections cannot order them.
  final List<String>? order;

  /// The stage inputs that connections feed, as [_key]s.
  final Set<String> fedInputs;

  final List<ConnectionConfig> connections;

  StageConfig? get finalStage {
    final o = order;
    return o == null || o.isEmpty ? null : stages[o.last];
  }

  /// Whether an input set on the pipeline under [name] reaches a stage.
  bool readsRunInput(String name) {
    for (final s in stages.values) {
      if (s.inputs.isEmpty) {
        // A stage without input ports reads the run inputs unless a
        // connection feeds it.
        if (!connections.any((c) => c.toStage == s.stageId)) return true;
      } else if (s.inputs.any((port) => port.name == name) &&
          !fedInputs.contains(_key(s.stageId, name))) {
        return true;
      }
    }
    return false;
  }
}
