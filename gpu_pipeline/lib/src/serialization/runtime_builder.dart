import 'dart:async';

import 'package:gpu_pipeline/src/pipeline_ports.dart';
import 'package:gpu_pipeline/src/serialization/connection_graph.dart';
import 'package:gpu_pipeline/src/serialization/operation_factory.dart';
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

  Future<void> dispose() async {
    await _router?.dispose();
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

          final tensor =
              outputs[link.fromOutput] ??
              outputs['output'] ??
              outputs.values.first;
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
                .catchError((_, __) {});
          }
        }
      });

      _subs.add(sub);
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

    // Store raw connections for runtime routing (add a field or method in Pipeline)
    pipeline.setSerializedConnections(pc.connections);

    return pipeline;
  }

  PipelineStage _buildStage(StageConfig sc) {
    // If stage_kind signals a dynamic composite stage:
    if (sc.stageKind == 'DynamicStage' || sc.operations.isNotEmpty) {
      final outputPortsMap = <String, OutputPort>{};
      for (final portCfg in sc.outputs) {
        final port = _makeOutputPort(portCfg);
        outputPortsMap[portCfg.name] = port;
      }

      final ops = <StageOperation>[];
      for (final oc in sc.operations) {
        ops.add(OperationFactory.create(oc, outputPorts: outputPortsMap));
      }
      final paramDefs = <String, Parameter>{};
      for (final entry in sc.parameters.entries) {
        // Minimal param metadata (could be enriched later)
        paramDefs[entry.key] = Parameter(
          name: entry.key,
          defaultValue: entry.value,
          description: '',
        );
      }
      final dyn = DynamicStage(
        stageId: sc.stageId,
        operations: ops,
        parameterDefinitions: paramDefs,
      );
      // Add input ports from config
      for (final portCfg in sc.inputs) {
        dyn.addInputPort(_makeInputPort(portCfg));
      }
      // Add output ports from config
      for (final portCfg in sc.outputs) {
        dyn.addOutputPort(_makeOutputPort(portCfg));
      }
      // Initialize parameter current values (defaults already set)
      sc.parameters.forEach((k, v) => dyn.setPropertyValue(k, v));
      return dyn;
    }

    // TODO: build specific concrete stage kinds (WindowStage, FFTStage, etc.)
    // Placeholder: treat as empty dynamic stage so pipeline stays consistent.
    return DynamicStage(
      stageId: sc.stageId,
      operations: const [],
      parameterDefinitions: {},
    );
  }

  RuntimeBuildResult build(NetworkConfig net) {
    final assets = sharedAssets ?? AssetManager();

    final built = <Pipeline>[];
    for (final p in net.pipelines) {
      final pipe = _buildPipeline(p);
      // If you added setSerializedAssets earlier:
      pipe.setSerializedAssets(net.assets);
      built.add(pipe);
    }

    // Wire inter-pipeline links (no-op today)
    final byId = {for (final p in built) p.id: p};
    final router = _InterPipelineRouter(byId, net.interPipelineLinks);
    router.install();

    return RuntimeBuildResult(pipelines: built, assets: assets, router: router);
  }
}

InputPort _makeInputPort(PortConfig cfg) {
  // Example: choose based on config
  if (cfg.portType == 'flexible_input') {
    return FlexibleInputPort(name: cfg.name);
  }
  if (cfg.portType == 'exact_input' &&
      cfg.shapeExpr?.raw['shape'] is List<int>) {
    return ExactInputPort(
      name: cfg.name,
      expectedShape: List<int>.from(cfg.shapeExpr!.raw['shape']),
    );
  }
  // Add more cases as needed
  throw Exception('Unknown input port type: ${cfg.portType}');
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
  // Add more cases as needed
  throw Exception('Unknown output port type: ${cfg.portType}');
}
