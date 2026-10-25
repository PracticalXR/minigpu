import 'dart:async';

import 'package:gpu_pipeline/src/pipeline_ports.dart';
import 'package:gpu_pipeline/src/serialization/connection_graph.dart';
import 'package:gpu_pipeline/src/serialization/operation_factory.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_validator.dart';
import 'package:gpu_pipeline/src/serialization/shape_expr.dart';

import '../pipeline.dart';
import '../pipeline_dynamic_stage.dart';
import '../pipeline_stage.dart';
import '../pipeline_properties.dart';
import '../operation/stage_operation.dart';
import '../pipeline_assets.dart';
import 'pipeline_models.dart';
import '../pipeline_event.dart';

class RuntimeBuildResult {
  final List<Pipeline> pipelines;
  final AssetManager assets;
  final _InterPipelineRouter? _router;

  RuntimeBuildResult({
    required this.pipelines,
    required this.assets,
    _InterPipelineRouter? router,
  }) : _router = router;

  /// Stops the links between the pipelines, then disposes the pipelines.
  /// The asset manager is not disposed.
  Future<void> dispose() async {
    await _router?.dispose();
    for (final pipeline in pipelines) {
      await pipeline.dispose();
    }
  }
}

// Router: on source completion, stash latest output into target external inputs.
// If mode == latestAutorun, also trigger target.runOnce({}).
class _InterPipelineRouter {
  final Map<String, Pipeline> _byId;
  final List<InterPipelineLinkConfig> _links;
  final List<StreamSubscription> _subs = [];
  final Map<String, Future<void>> _chains = {}; // serialize autoruns per target

  _InterPipelineRouter(this._byId, this._links);

  void install() {
    if (_links.isEmpty) return;

    final bySrc = <String, List<InterPipelineLinkConfig>>{};
    for (final l in _links) {
      bySrc.putIfAbsent(l.fromPipeline, () => []).add(l);
    }

    for (final entry in bySrc.entries) {
      final src = _byId[entry.key];
      if (src == null) continue;

      final sub = src.eventNotifier.eventStream.listen((e) {
        if (e is! PipelineRunCompleteEvent) return;

        final outputs = src.getLatestFinalOutputs();
        if (outputs == null || outputs.isEmpty) return;

        for (final link in entry.value) {
          final tgt = _byId[link.toPipeline];
          if (tgt == null) continue;

          // Only the named output: the validator checked that the source's
          // final stage has it, so a run that lacks it has nothing to pass.
          final tensor = outputs[link.fromOutput];
          if (tensor == null) continue;
          tgt.setExternalInput(link.toInput, tensor);

          if (link.mode == InterPipelineLinkMode.latestAutorun &&
              tgt.isRunning) {
            final previous = _chains[tgt.id] ?? Future.value();
            _chains[tgt.id] = previous
                .then((_) => tgt.runOnce({}))
                .then((_) {
                  // Clear consumed inputs after autorun to avoid stale reuse
                  tgt.clearExternalInputs();
                })
                .catchError((Object error, StackTrace stackTrace) {
                  _reportAutorunError(tgt, link, error, stackTrace);
                });
          }
        }
      });

      _subs.add(sub);
    }
  }

  /// Reports a failed autorun as a [PipelineErrorEvent] on the target, so
  /// the next autorun still runs.
  void _reportAutorunError(
    Pipeline target,
    InterPipelineLinkConfig link,
    Object error,
    StackTrace stackTrace,
  ) {
    try {
      target.eventNotifier.emit(
        PipelineErrorEvent(
          error: '$error',
          stackTrace: stackTrace,
          pipelineId: target.id,
          metadata: {
            'source': 'inter_pipeline_link',
            'from_pipeline': link.fromPipeline,
          },
        ),
      );
    } catch (_) {
      // The target was disposed while the run was in flight.
    }
  }

  Future<void> dispose() async {
    for (final s in _subs) {
      try {
        await s.cancel();
      } catch (_) {}
    }
    _subs.clear();
  }
}

class RuntimeBuilder {
  final AssetManager? sharedAssets;
  const RuntimeBuilder({this.sharedAssets});

  QueueStrategy _parseQueueStrategy(Map<String, dynamic> q) {
    final strat = (q['strategy'] as String?) ?? 'latest_only';
    switch (strat) {
      case 'latest_only':
        return QueueStrategy.latestOnly;
      case 'latest_n':
        return QueueStrategy.latestN;
      case 'drop_on_overrun':
        return QueueStrategy.dropOnOverrun;
      case 'batched':
        return QueueStrategy.batched;
      case 'process_all':
        return QueueStrategy.processAll;
      default:
        return QueueStrategy.latestOnly;
    }
  }

  Pipeline _buildPipeline(PipelineConfig pc) {
    final queueConfig = QueueConfig(
      strategy: _parseQueueStrategy(pc.queue),
      maxItemsPerStream: pc.queue['max_items_per_stream'] ?? 1,
      maxTotalItems: pc.queue['max_total_items'] ?? 10,
      batchSize: pc.queue['batch_size'] ?? 1,
    );
    final pipeline = Pipeline(
      id: pc.id,
      queueConfig: queueConfig,
      enableBenchmarking: pc.enableBenchmarking ?? false,
      debugEnabled: pc.debugEnabled ?? false,
      assetManager: sharedAssets,
    );

    // Build stages (unordered map first)
    final stages = <String, PipelineStage>{};
    for (final sc in pc.stages) {
      stages[sc.stageId] = _buildStage(sc);
    }

    // Build graph if connections present
    List<String> executionOrder;
    if (pc.connections.isNotEmpty) {
      final graph = ConnectionGraph.build(pc);
      executionOrder = graph.topologicalOrder(stages.keys);
    } else {
      executionOrder = pc.stages.map((s) => s.stageId).toList();
    }

    for (final id in executionOrder) {
      pipeline.addStage(stages[id]!);
    }

    // The pipeline routes each stage's inputs along these connections.
    pipeline.setSerializedConnections(pc.connections);

    return pipeline;
  }

  /// Builds [sc] as a [DynamicStage] from its ports, parameters and
  /// operations, whatever its stage_kind.
  PipelineStage _buildStage(StageConfig sc) {
    final outputPorts = <String, OutputPort>{
      for (final portCfg in sc.outputs) portCfg.name: _makeOutputPort(portCfg),
    };

    final ops = <StageOperation>[
      for (final oc in sc.operations)
        OperationFactory.create(oc, outputPorts: outputPorts),
    ];
    final paramDefs = <String, Parameter>{
      for (final entry in sc.parameters.entries)
        entry.key: Parameter(
          name: entry.key,
          defaultValue: entry.value,
          description: '',
        ),
    };
    // The constructor sets each parameter to its default value.
    final stage = DynamicStage(
      stageId: sc.stageId,
      operations: ops,
      parameterDefinitions: paramDefs,
    );
    for (final portCfg in sc.inputs) {
      stage.addInputPort(_makeInputPort(portCfg));
    }
    outputPorts.values.forEach(stage.addOutputPort);
    return stage;
  }

  RuntimeBuildResult build(NetworkConfig net) {
    final assets = sharedAssets ?? AssetManager();

    final built = <Pipeline>[];
    for (final p in net.pipelines) {
      final pipe = _buildPipeline(p);
      pipe.setSerializedAssets(net.assets);
      built.add(pipe);
    }

    // Wire inter-pipeline links
    final byId = {for (final p in built) p.id: p};
    final router = _InterPipelineRouter(byId, net.interPipelineLinks);
    router.install();

    return RuntimeBuildResult(pipelines: built, assets: assets, router: router);
  }
}

InputPort _makeInputPort(PortConfig cfg) {
  switch (cfg.portType) {
    case 'flexible_input':
      return FlexibleInputPort(
        name: cfg.name,
        isRequired: cfg.required ?? true,
        minDimensions: cfg.minDimensions,
        maxDimensions: cfg.maxDimensions,
      );
    case 'exact_input':
      final shape = exactPortShape(cfg);
      if (shape == null) {
        throw ArgumentError(
          'exact_input port ${cfg.name} needs a shape_expr '
          '{"op": "fixed", "shape": [...]} whose dims are positive integers',
        );
      }
      return ExactInputPort(name: cfg.name, expectedShape: shape);
  }
  throw ArgumentError('Unknown input port type: ${cfg.portType}');
}

OutputPort _makeOutputPort(PortConfig cfg) {
  if (cfg.portType == 'dynamic_output') {
    // If shapeExpr is a fixed shape, use FixedOutputPort
    final shapeRaw = cfg.shapeExpr?.raw['shape'];
    if (shapeRaw is List && shapeRaw.every((e) => e is int)) {
      return FixedOutputPort(name: cfg.name, shape: List<int>.from(shapeRaw));
    }
    // If shapeExpr exists, use it for shapeCalculator
    if (cfg.shapeExpr != null) {
      return DynamicOutputPort(
        name: cfg.name,
        shapeCalculator: (inputs, parameters) {
          final inputShapes = <String, List<int>>{};
          inputs.forEach((k, v) => inputShapes[k] = v.shape);
          final ctx = ShapeEvalContext(
            inputShapes: inputShapes,
            parameters: parameters,
          );
          return ShapeExprEvaluator().evaluate(cfg.shapeExpr!.raw, ctx);
        },
      );
    }
    // Fallback: mirror first input shape
    return DynamicOutputPort(
      name: cfg.name,
      shapeCalculator: (inputs, _) =>
          inputs.isNotEmpty ? List<int>.from(inputs.values.first.shape) : [1],
    );
  }
  throw ArgumentError('Unknown output port type: ${cfg.portType}');
}
