import 'pipeline_models.dart';

abstract class PatchOp {
  String get op;
  Map<String, dynamic> toJson();
  static PatchOp fromJson(Map<String, dynamic> j) {
    switch (j['op']) {
      case 'set_param':
        return SetParamOp(
          stageId: j['stage_id'] as String,
          name: j['name'] as String,
          value: j['value'],
        );
      case 'replace_stage':
        return ReplaceStageOp(
          targetStageId: j['target_stage_id'] as String,
          stage: StageConfig.fromJson(j['stage'] as Map<String, dynamic>),
        );
      case 'add_operation':
        return AddOperationOp(
          stageId: j['stage_id'] as String,
          operation: OperationConfig.fromJson(
            j['operation'] as Map<String, dynamic>,
          ),
          index: j['index'] as int?,
        );
      default:
        throw UnsupportedError('Unknown patch op ${j['op']}');
    }
  }
}

class SetParamOp extends PatchOp {
  final String stageId;
  final String name;
  final dynamic value;
  SetParamOp({required this.stageId, required this.name, required this.value});
  @override
  String get op => 'set_param';
  @override
  Map<String, dynamic> toJson() => {
    'op': op,
    'stage_id': stageId,
    'name': name,
    'value': value,
  };
}

class ReplaceStageOp extends PatchOp {
  final String targetStageId;
  final StageConfig stage; // new stage (may have different stage_id)
  ReplaceStageOp({required this.targetStageId, required this.stage});
  @override
  String get op => 'replace_stage';
  @override
  Map<String, dynamic> toJson() => {
    'op': op,
    'target_stage_id': targetStageId,
    'stage': stage.toJson(),
  };
}

class AddOperationOp extends PatchOp {
  final String stageId;
  final OperationConfig operation;
  final int? index; // if omitted, append
  AddOperationOp({required this.stageId, required this.operation, this.index});
  @override
  String get op => 'add_operation';
  @override
  Map<String, dynamic> toJson() => {
    'op': op,
    'stage_id': stageId,
    'operation': operation.toJson(),
    if (index != null) 'index': index,
  };
}

class PatchEnvelope {
  final String pipelineId;
  final List<PatchOp> ops;
  PatchEnvelope({required this.pipelineId, required this.ops});

  /// Reads the JSON that [toJson] writes.
  factory PatchEnvelope.fromJson(Map<String, dynamic> j) => PatchEnvelope(
    pipelineId: j['pipeline_id'] as String,
    ops: [
      for (final o in j['ops'] as List? ?? const <dynamic>[])
        PatchOp.fromJson((o as Map).cast<String, dynamic>()),
    ],
  );

  Map<String, dynamic> toJson() => {
    'type': 'patch',
    'pipeline_id': pipelineId,
    'ops': ops.map((o) => o.toJson()).toList(),
  };
}

class PatchApplier {
  /// Returns a copy of [net] with [patch] applied; [net] is not changed.
  ///
  /// Throws an [ArgumentError], and applies nothing, if the patch names a
  /// pipeline, stage or parameter that the network does not have, or if a
  /// `replace_stage` gives the stage the id of another stage.
  NetworkConfig apply(NetworkConfig net, PatchEnvelope patch) {
    if (!net.pipelines.any((p) => p.id == patch.pipelineId)) {
      throw ArgumentError(
        'The patch names pipeline ${patch.pipelineId}, which the network '
        'does not have',
      );
    }
    final pipelines = <PipelineConfig>[];

    for (final p in net.pipelines) {
      if (p.id != patch.pipelineId) {
        pipelines.add(p);
        continue;
      }

      // Work on JSON to preserve unknown/extensible fields.
      final pj = Map<String, dynamic>.from(p.toJson());
      final stagesJ = (pj['stages'] as List)
          .cast<Map<String, dynamic>>()
          .toList();
      final connsJ = (pj['connections'] as List)
          .cast<Map<String, dynamic>>()
          .toList();

      int stageIndex(String stageId) {
        final idx = stagesJ.indexWhere((s) => s['stage_id'] == stageId);
        if (idx < 0) {
          throw ArgumentError(
            'The patch names stage $stageId, which pipeline ${p.id} does '
            'not have',
          );
        }
        return idx;
      }

      for (final op in patch.ops) {
        if (op is SetParamOp) {
          final idx = stageIndex(op.stageId);
          final sj = Map<String, dynamic>.from(stagesJ[idx]);
          final params = Map<String, dynamic>.from(
            (sj['parameters'] as Map?) ?? const <String, dynamic>{},
          );
          if (!params.containsKey(op.name)) {
            throw ArgumentError(
              'The patch sets parameter ${op.name}, which stage ${op.stageId} '
              'does not have',
            );
          }
          params[op.name] = op.value;
          sj['parameters'] = params;
          stagesJ[idx] = sj;
        } else if (op is ReplaceStageOp) {
          final idx = stageIndex(op.targetStageId);

          final oldId = op.targetStageId;
          final newJ = op.stage.toJson();
          final newId = (newJ['stage_id'] as String?) ?? oldId;
          if (newId != oldId &&
              stagesJ.any((s) => s['stage_id'] == newId)) {
            throw ArgumentError(
              'The patch replaces stage $oldId with stage $newId, but '
              'pipeline ${p.id} already has a stage $newId',
            );
          }

          // Replace stage JSON
          stagesJ[idx] = newJ;

          // Rewrite connections if stage_id changed
          if (newId != oldId) {
            for (var i = 0; i < connsJ.length; i++) {
              final cj = Map<String, dynamic>.from(connsJ[i]);
              if (cj['from_stage'] == oldId) cj['from_stage'] = newId;
              if (cj['to_stage'] == oldId) cj['to_stage'] = newId;
              connsJ[i] = cj;
            }
          }
        } else if (op is AddOperationOp) {
          final idx = stageIndex(op.stageId);
          final sj = Map<String, dynamic>.from(stagesJ[idx]);
          final opsJ = (sj['operations'] as List? ?? const [])
              .map((e) => (e as Map).cast<String, dynamic>())
              .toList();
          final opJ = op.operation.toJson();
          if (op.index == null || op.index! >= opsJ.length) {
            opsJ.add(opJ);
          } else {
            final insertAt = op.index! < 0 ? 0 : op.index!;
            opsJ.insert(insertAt, opJ);
          }
          sj['operations'] = opsJ;
          stagesJ[idx] = sj;
        }
      }

      pj['stages'] = stagesJ;
      pj['connections'] = connsJ;
      pipelines.add(PipelineConfig.fromJson(pj));
    }

    return NetworkConfig(
      schemaVersion: net.schemaVersion,
      pipelines: pipelines,
      interPipelineLinks: net.interPipelineLinks,
      assets: net.assets,
      notes: net.notes,
    );
  }
}
