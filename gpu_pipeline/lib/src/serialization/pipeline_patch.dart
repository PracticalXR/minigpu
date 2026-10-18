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
  Map<String, dynamic> toJson() => {
    'type': 'patch',
    'pipeline_id': pipelineId,
    'ops': ops.map((o) => o.toJson()).toList(),
  };
}

class PatchApplier {
  NetworkConfig apply(NetworkConfig net, PatchEnvelope patch) {
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

      for (final op in patch.ops) {
        if (op is SetParamOp) {
          final idx = stagesJ.indexWhere((s) => s['stage_id'] == op.stageId);
          if (idx < 0) continue;
          final sj = Map<String, dynamic>.from(stagesJ[idx]);
          final params = Map<String, dynamic>.from(
            (sj['parameters'] as Map?) ?? const <String, dynamic>{},
          );
          params[op.name] = op.value;
          sj['parameters'] = params;
          stagesJ[idx] = sj;
        } else if (op is ReplaceStageOp) {
          final idx = stagesJ.indexWhere(
            (s) => s['stage_id'] == op.targetStageId,
          );
          if (idx < 0) continue;

          final oldId = op.targetStageId;
          final newJ = op.stage.toJson();
          final newId = (newJ['stage_id'] as String?) ?? oldId;

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
          final idx = stagesJ.indexWhere((s) => s['stage_id'] == op.stageId);
          if (idx < 0) continue;
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
