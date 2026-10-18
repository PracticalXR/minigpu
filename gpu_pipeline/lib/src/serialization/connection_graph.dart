import 'pipeline_models.dart';

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

  ConnectionGraph._(this.edges);

  static ConnectionGraph build(PipelineConfig config) {
    final edges = <ConnectionEdge>[];
    for (final c in config.connections) {
      final e = ConnectionEdge(c.fromStage, c.toStage, c.fromOutput, c.toInput);
      edges.add(e);
    }
    final g = ConnectionGraph._(edges);
    for (final e in edges) {
      g.outgoing.putIfAbsent(e.fromStage, () => []).add(e);
      g.incoming.putIfAbsent(e.toStage, () => []).add(e);
    }
    return g;
  }

  /// Returns execution order or throws on cycle.
  List<String> topologicalOrder(Iterable<String> stageIds) {
    final inDegree = <String, int>{};
    for (final id in stageIds) {
      inDegree[id] = 0;
    }
    for (final list in incoming.values) {
      for (final e in list) {
        inDegree[e.toStage] = (inDegree[e.toStage] ?? 0) + 1;
      }
    }
    final queue = <String>[];
    inDegree.forEach((k, v) {
      if (v == 0) queue.add(k);
    });

    final order = <String>[];
    while (queue.isNotEmpty) {
      final n = queue.removeLast();
      order.add(n);
      for (final e in outgoing[n] ?? const []) {
        inDegree[e.toStage] = (inDegree[e.toStage]! - 1);
        if (inDegree[e.toStage] == 0) {
          queue.add(e.toStage);
        }
      }
    }
    if (order.length != stageIds.length) {
      throw StateError('CYCLE_DETECTED');
    }
    return order;
  }
}
