import 'pipeline_models.dart';

/// The output that a connection without `from_output` reads from a stage
/// whose outputs are [outputs]: the only one, else the one named `output`.
/// Returns null when neither applies.
String? defaultConnectionOutput(Iterable<String> outputs) {
  final names = outputs.toList();
  if (names.length == 1) return names.single;
  return names.contains('output') ? 'output' : null;
}

/// The input that a connection without `to_input` feeds on a stage whose
/// input ports are [inputPorts]: `input` for a stage without ports, else the
/// only port, else the port named `input`. Returns null when none applies.
String? defaultConnectionInput(Iterable<String> inputPorts) {
  final names = inputPorts.toList();
  if (names.isEmpty) return 'input';
  if (names.length == 1) return names.single;
  return names.contains('input') ? 'input' : null;
}

class ConnectionEdge {
  final String fromStage;
  final String toStage;
  final String? fromOutput;
  final String? toInput;
  ConnectionEdge(this.fromStage, this.toStage, this.fromOutput, this.toInput);
}

class ConnectionGraph {
  final Map<String, List<ConnectionEdge>> outgoing = {};
  final Map<String, List<ConnectionEdge>> incoming = {};
  final List<ConnectionEdge> edges;

  /// The pipeline the connections belong to, for error messages.
  final String pipelineId;

  ConnectionGraph._(this.edges, this.pipelineId);

  static ConnectionGraph build(PipelineConfig config) {
    final edges = <ConnectionEdge>[];
    for (final c in config.connections) {
      final e = ConnectionEdge(c.fromStage, c.toStage, c.fromOutput, c.toInput);
      edges.add(e);
    }
    final g = ConnectionGraph._(edges, config.id);
    for (final e in edges) {
      g.outgoing.putIfAbsent(e.fromStage, () => []).add(e);
      g.incoming.putIfAbsent(e.toStage, () => []).add(e);
    }
    return g;
  }

  /// Returns the order to run [stageIds] in: each stage after every stage
  /// connected into it, and otherwise in the order [stageIds] lists them.
  ///
  /// Throws a [StateError] if a connection names a stage that is not in
  /// [stageIds], or if the connections form a cycle.
  List<String> topologicalOrder(Iterable<String> stageIds) {
    final remaining = stageIds.toList();
    final known = remaining.toSet();
    for (final e in edges) {
      for (final id in [e.fromStage, e.toStage]) {
        if (!known.contains(id)) {
          throw StateError(
            'Pipeline $pipelineId: connection ${e.fromStage} -> ${e.toStage} '
            'names unknown stage "$id"',
          );
        }
      }
    }
    final inDegree = {for (final id in remaining) id: 0};
    for (final e in edges) {
      inDegree[e.toStage] = inDegree[e.toStage]! + 1;
    }

    final order = <String>[];
    while (remaining.isNotEmpty) {
      final next = remaining.indexWhere((id) => inDegree[id] == 0);
      if (next < 0) {
        throw StateError(
          'Pipeline $pipelineId: connections form a cycle, so these stages '
          'cannot be ordered: ${remaining.join(', ')}',
        );
      }
      final id = remaining.removeAt(next);
      order.add(id);
      for (final e in outgoing[id] ?? const <ConnectionEdge>[]) {
        inDegree[e.toStage] = inDegree[e.toStage]! - 1;
      }
    }
    return order;
  }
}
