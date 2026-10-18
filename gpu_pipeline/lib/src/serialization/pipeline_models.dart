import 'dart:convert';

// Define link modes
enum InterPipelineLinkMode {
  latest, // stash latest into target external input
  latestAutorun, // stash + trigger target.runOnce({})
}

class NetworkConfig {
  final int schemaVersion;
  final List<PipelineConfig> pipelines;
  final List<InterPipelineLinkConfig> interPipelineLinks;
  final List<AssetConfig> assets;
  final String? notes;

  NetworkConfig({
    required this.schemaVersion,
    required this.pipelines,
    this.interPipelineLinks = const [],
    this.assets = const [],
    this.notes,
  });

  Map<String, dynamic> toJson() => {
    'schema_version': schemaVersion,
    'pipelines': pipelines.map((p) => p.toJson()).toList(),
    if (interPipelineLinks.isNotEmpty)
      'inter_pipeline_links': interPipelineLinks
          .map((l) => l.toJson())
          .toList(),
    if (assets.isNotEmpty) 'assets': assets.map((a) => a.toJson()).toList(),
    if (notes != null) 'notes': notes,
  };

  static NetworkConfig fromJson(Map<String, dynamic> json) {
    return NetworkConfig(
      schemaVersion: json['schema_version'] as int,
      pipelines: (json['pipelines'] as List)
          .map((e) => PipelineConfig.fromJson(e))
          .toList(),
      interPipelineLinks: (json['inter_pipeline_links'] as List? ?? [])
          .map((e) => InterPipelineLinkConfig.fromJson(e))
          .toList(),
      assets: (json['assets'] as List? ?? [])
          .map((e) => AssetConfig.fromJson(e))
          .toList(),
      notes: json['notes'] as String?,
    );
  }

  String encode() => jsonEncode(toJson());
  static NetworkConfig decode(String s) =>
      fromJson(jsonDecode(s) as Map<String, dynamic>);
}

class PipelineConfig {
  final String id;
  final Map<String, dynamic> queue;
  final bool? enableBenchmarking;
  final bool? debugEnabled;
  final List<StageConfig> stages;
  final List<ConnectionConfig> connections;
  final Map<String, dynamic> parameters;
  final Map<String, dynamic>? metadata;

  PipelineConfig({
    required this.id,
    required this.queue,
    required this.stages,
    required this.connections,
    this.enableBenchmarking,
    this.debugEnabled,
    this.parameters = const {},
    this.metadata,
  });

  Map<String, dynamic> toJson() => {
    'type': 'pipeline',
    'id': id,
    'queue': queue,
    if (enableBenchmarking != null) 'enable_benchmarking': enableBenchmarking,
    if (debugEnabled != null) 'debug_enabled': debugEnabled,
    'stages': stages.map((s) => s.toJson()).toList(),
    'connections': connections.map((c) => c.toJson()).toList(),
    if (parameters.isNotEmpty) 'parameters': parameters,
    if (metadata != null) 'metadata': metadata,
  };

  static PipelineConfig fromJson(Map<String, dynamic> j) {
    return PipelineConfig(
      id: j['id'],
      queue: (j['queue'] as Map<String, dynamic>? ?? const {}),
      enableBenchmarking: j['enable_benchmarking'] as bool?,
      debugEnabled: j['debug_enabled'] as bool?,
      stages: (j['stages'] as List)
          .map((e) => StageConfig.fromJson(e))
          .toList(),
      connections: (j['connections'] as List)
          .map((e) => ConnectionConfig.fromJson(e))
          .toList(),
      parameters:
          (j['parameters'] as Map<String, dynamic>? ??
          const <String, dynamic>{}),
      metadata: j['metadata'] as Map<String, dynamic>?,
    );
  }
}

class StageConfig {
  final String stageId;
  final String stageKind;
  final int? order;
  final String? executionTarget;
  final Map<String, dynamic>? resource;
  final List<PortConfig> inputs;
  final List<PortConfig> outputs;
  final Map<String, dynamic> parameters;
  final List<OperationConfig> operations; // composite/dynamic
  final List<String> assets;

  StageConfig({
    required this.stageId,
    required this.stageKind,
    this.order,
    this.executionTarget,
    this.resource,
    this.inputs = const [],
    this.outputs = const [],
    this.parameters = const {},
    this.operations = const [],
    this.assets = const [],
  });

  Map<String, dynamic> toJson() => {
    'type': 'stage',
    'stage_id': stageId,
    'stage_kind': stageKind,
    if (order != null) 'order': order,
    if (executionTarget != null) 'execution_target': executionTarget,
    if (resource != null) 'resource': resource,
    if (inputs.isNotEmpty) 'inputs': inputs.map((p) => p.toJson()).toList(),
    if (outputs.isNotEmpty) 'outputs': outputs.map((p) => p.toJson()).toList(),
    if (parameters.isNotEmpty) 'parameters': parameters,
    if (operations.isNotEmpty)
      'operations': operations.map((o) => o.toJson()).toList(),
    if (assets.isNotEmpty) 'assets': assets,
  };

  static StageConfig fromJson(Map<String, dynamic> j) {
    return StageConfig(
      stageId: j['stage_id'],
      stageKind: j['stage_kind'],
      order: j['order'] as int?,
      executionTarget: j['execution_target'] as String?,
      resource: j['resource'] as Map<String, dynamic>?,
      inputs: (j['inputs'] as List? ?? [])
          .map((e) => PortConfig.fromJson(e))
          .toList(),
      outputs: (j['outputs'] as List? ?? [])
          .map((e) => PortConfig.fromJson(e))
          .toList(),
      parameters:
          (j['parameters'] as Map<String, dynamic>? ??
          const <String, dynamic>{}),
      operations: (j['operations'] as List? ?? [])
          .map((e) => OperationConfig.fromJson(e))
          .toList(),
      assets: (j['assets'] as List? ?? []).cast<String>(),
    );
  }
}

class PortConfig {
  final String name;
  final String portType;
  final String? shapePolicy;
  final String? sizePolicy;
  final ShapeExprConfig? shapeExpr;
  final int? minDimensions;
  final int? maxDimensions;
  final bool? required;

  PortConfig({
    required this.name,
    required this.portType,
    this.shapePolicy,
    this.sizePolicy,
    this.shapeExpr,
    this.minDimensions,
    this.maxDimensions,
    this.required,
  });

  Map<String, dynamic> toJson() => {
    'name': name,
    'port_type': portType,
    if (shapePolicy != null) 'shape_policy': shapePolicy,
    if (sizePolicy != null) 'size_policy': sizePolicy,
    if (shapeExpr != null) 'shape_expr': shapeExpr!.toJson(),
    if (minDimensions != null) 'min_dimensions': minDimensions,
    if (maxDimensions != null) 'max_dimensions': maxDimensions,
    if (required != null) 'required': required,
  };

  static PortConfig fromJson(Map<String, dynamic> j) => PortConfig(
    name: j['name'],
    portType: j['port_type'],
    shapePolicy: j['shape_policy'] as String?,
    sizePolicy: j['size_policy'] as String?,
    shapeExpr: j['shape_expr'] != null
        ? ShapeExprConfig.fromJson(j['shape_expr'])
        : null,
    minDimensions: j['min_dimensions'] as int?,
    maxDimensions: j['max_dimensions'] as int?,
    required: j['required'] as bool?,
  );
}

class OperationConfig {
  final String opKind; // tensor_op, shader, dynamic_cpu, cpu, merge, split
  final String? operation; // tensor_op name
  final List<String> inputKeys;
  final List<String> outputKeys;
  final Map<String, dynamic> arguments;
  final Map<String, dynamic> namedArguments;
  final String? dartCode;
  final String? wgsl;
  final Map<String, dynamic>? shaderParameters;
  final List<int>? workgroupSize;

  OperationConfig({
    required this.opKind,
    this.operation,
    this.inputKeys = const [],
    this.outputKeys = const [],
    this.arguments = const {},
    this.namedArguments = const {},
    this.dartCode,
    this.wgsl,
    this.shaderParameters,
    this.workgroupSize,
  });

  Map<String, dynamic> toJson() => {
    'type': 'operation',
    'op_kind': opKind,
    if (operation != null) 'operation': operation,
    if (inputKeys.isNotEmpty) 'input_keys': inputKeys,
    if (outputKeys.isNotEmpty) 'output_keys': outputKeys,
    if (arguments.isNotEmpty) 'arguments': arguments,
    if (namedArguments.isNotEmpty) 'named_arguments': namedArguments,
    if (dartCode != null) 'dart_code': dartCode,
    if (wgsl != null) 'wgsl': wgsl,
    if (shaderParameters != null) 'shader_parameters': shaderParameters,
    if (workgroupSize != null) 'workgroup_size': workgroupSize,
  };

  static OperationConfig fromJson(Map<String, dynamic> j) => OperationConfig(
    opKind: j['op_kind'],
    operation: j['operation'] as String?,
    inputKeys: (j['input_keys'] as List? ?? const <dynamic>[]).cast<String>(),
    outputKeys: (j['output_keys'] as List? ?? const <dynamic>[]).cast<String>(),
    arguments:
        (j['arguments'] as Map<String, dynamic>? ?? const <String, dynamic>{}),
    namedArguments:
        (j['named_arguments'] as Map<String, dynamic>? ??
        const <String, dynamic>{}),
    dartCode: j['dart_code'] as String?,
    wgsl: j['wgsl'] as String?,
    shaderParameters: j['shader_parameters'] as Map<String, dynamic>?,
    workgroupSize: (j['workgroup_size'] as List?)
        ?.map((e) => e as int)
        .toList(),
  );
}

class ShapeExprConfig {
  final String op;
  final Map<String, dynamic> raw; // store all fields for evaluator

  ShapeExprConfig({required this.op, required this.raw});

  Map<String, dynamic> toJson() => raw;

  static ShapeExprConfig fromJson(Map<String, dynamic> j) {
    return ShapeExprConfig(op: j['op'], raw: Map<String, dynamic>.from(j));
  }
}

class ConnectionConfig {
  final String fromStage;
  final String toStage;
  final String? fromOutput; // optional specific output name
  final String? toInput; // optional specific input port name

  ConnectionConfig({
    required this.fromStage,
    required this.toStage,
    this.fromOutput,
    this.toInput,
  });

  Map<String, dynamic> toJson() => {
    'from_stage': fromStage,
    'to_stage': toStage,
    if (fromOutput != null) 'from_output': fromOutput,
    if (toInput != null) 'to_input': toInput,
  };

  static ConnectionConfig fromJson(Map<String, dynamic> j) => ConnectionConfig(
    fromStage: j['from_stage'],
    toStage: j['to_stage'],
    fromOutput: j['from_output'] as String?,
    toInput: j['to_input'] as String?,
  );
}

class AssetConfig {
  /// Logical name used by stages/pipeline to request this asset.
  final String name;

  /// Serialized source descriptor.
  /// Examples:
  /// { "type": "file", "path": "models/mobilenet_v2.bin" }
  /// { "type": "url", "url": "https://..." }
  /// { "type": "embedded", "data": "<base64 or small literal>" }
  /// { "type": "preloaded" }
  final Map<String, dynamic> source;

  /// If true, pipeline start should fail if this asset is unavailable.
  final bool required;

  /// "preload" | "lazy" | "per_call" | "stream"
  final String strategy;

  /// Whether AssetManager should keep this asset after first load.
  /// Defaults to true for preload, false for per_call.
  final bool cache;

  /// Optional UI/validation hint: "weights" | "shader" | "texture" | "lut" | ...
  final String? typeHint;

  /// Optional extra metadata for loaders or validation.
  final Map<String, dynamic>? metadata;

  AssetConfig({
    required this.name,
    required this.source,
    this.required = false,
    this.strategy = 'preload',
    bool? cache,
    this.typeHint,
    this.metadata,
  }) : cache = cache ?? (strategy == 'preload');

  factory AssetConfig.fromJson(Map<String, dynamic> j) => AssetConfig(
    name: j['name'] as String,
    source: (j['source'] as Map).cast<String, dynamic>(),
    required: (j['required'] as bool?) ?? false,
    strategy: (j['strategy'] as String?) ?? 'preload',
    cache: j['cache'] as bool?,
    typeHint: j['type_hint'] as String?,
    metadata: (j['metadata'] as Map?)?.cast<String, dynamic>(),
  );

  Map<String, dynamic> toJson() => {
    'name': name,
    'source': source,
    if (required) 'required': true,
    if (strategy != 'preload') 'strategy': strategy,
    if (cache != (strategy == 'preload')) 'cache': cache,
    if (typeHint != null) 'type_hint': typeHint,
    if (metadata != null) 'metadata': metadata,
  };
}

class InterPipelineLinkConfig {
  final String fromPipeline;
  final String toPipeline;
  final String fromOutput;
  final String toInput;
  final InterPipelineLinkMode mode;

  const InterPipelineLinkConfig({
    required this.fromPipeline,
    required this.toPipeline,
    required this.fromOutput,
    required this.toInput,
    this.mode = InterPipelineLinkMode.latest,
  });

  factory InterPipelineLinkConfig.fromJson(Map<String, dynamic> json) {
    // Accept both flat and nested forms for compatibility with docs:
    // flat:   { from_pipeline, to_pipeline, from_output, to_input, mode }
    // nested: { from: { pipeline, output }, to: { pipeline, input }, mode }
    String? fromPipeline;
    String? toPipeline;
    String? fromOutput;
    String? toInput;

    if (json.containsKey('from') || json.containsKey('to')) {
      final from = (json['from'] as Map?)?.cast<String, dynamic>() ?? const {};
      final to = (json['to'] as Map?)?.cast<String, dynamic>() ?? const {};
      fromPipeline = from['pipeline'] as String?;
      fromOutput = from['output'] as String?;
      toPipeline = to['pipeline'] as String?;
      toInput = to['input'] as String?;
    } else {
      fromPipeline = json['from_pipeline'] as String?;
      fromOutput = json['from_output'] as String?;
      toPipeline = json['to_pipeline'] as String?;
      toInput = json['to_input'] as String?;
    }

    final rawMode = (json['mode'] as String?)?.toLowerCase() ?? 'latest';
    final m = switch (rawMode) {
      'latest_autorun' => InterPipelineLinkMode.latestAutorun,
      _ => InterPipelineLinkMode.latest,
    };

    return InterPipelineLinkConfig(
      fromPipeline: fromPipeline ?? '',
      toPipeline: toPipeline ?? '',
      fromOutput: fromOutput ?? '',
      toInput: toInput ?? '',
      mode: m,
    );
  }

  Map<String, dynamic> toJson() => {
    'from_pipeline': fromPipeline,
    'to_pipeline': toPipeline,
    'from_output': fromOutput,
    'to_input': toInput,
    'mode': switch (mode) {
      InterPipelineLinkMode.latestAutorun => 'latest_autorun',
      InterPipelineLinkMode.latest => 'latest',
    },
  };
}
