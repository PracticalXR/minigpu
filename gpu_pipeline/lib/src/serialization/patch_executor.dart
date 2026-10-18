import 'dart:async';

import 'pipeline_loader.dart';
import 'pipeline_models.dart';
import 'pipeline_patch.dart';
import 'pipeline_validator.dart';
import 'runtime_builder.dart';
import '../pipeline.dart';
import '../pipeline_assets.dart';
import '../pipeline_stage.dart';

/// Applies patches to live pipelines.
/// Tip: Structural patches rebuild the entire runtime so inter-pipeline links keep working.
class RuntimePatchExecutor {
  final Map<String, Pipeline> livePipelines;
  RuntimePatchExecutor(this.livePipelines);

  /// Applies a patch of `set_param` ops to the live stages in place, and
  /// returns whether it did.
  ///
  /// It changes nothing and returns false when the patch has a structural
  /// op, which needs [applyAndRebuild], or when an op names a pipeline,
  /// stage or parameter the pipelines do not have, or a value the parameter
  /// rejects. [currentConfig] is not changed; apply the patch to it with
  /// [PatchApplier] to keep it in step.
  bool apply(NetworkConfig currentConfig, PatchEnvelope env) {
    final pipeline = livePipelines[env.pipelineId];
    if (pipeline == null) return false;

    final changes = <(PipelineStage, SetParamOp)>[];
    for (final op in env.ops) {
      if (op is! SetParamOp) return false;
      final stage = pipeline.getStage(op.stageId);
      if (stage == null ||
          !stage.hasParameter(op.name) ||
          !stage.isValidParameterValue(op.name, op.value)) {
        return false;
      }
      changes.add((stage, op));
    }
    for (final (stage, op) in changes) {
      // The value is stored before setPropertyValue first awaits; the
      // future carries only the change notification.
      unawaited(stage.setPropertyValue(op.name, op.value));
    }
    return true;
  }

  /// Applies [env] to [currentConfig], validates and builds the result, and
  /// replaces [livePipelines] with the new pipelines.
  ///
  /// Throws, leaving [livePipelines] as they were, if the patch names
  /// something the network does not have ([ArgumentError]) or the patched
  /// network has validation errors ([PipelineValidationException]). Dispose
  /// the previous runtime once the new one is in use.
  RuntimePatchApplyResult applyAndRebuild(
    NetworkConfig currentConfig,
    PatchEnvelope env,
  ) {
    // Apply patch to config JSON-wise
    final applier = PatchApplier();
    final newConfig = applier.apply(currentConfig, env);
    final report = PipelineValidator().validate(newConfig);
    if (!report.ok) throw PipelineValidationException(report);

    // Use the same shared AssetManager all pipelines were using (from any live pipeline)
    final AssetManager am = livePipelines.values.isNotEmpty
        ? livePipelines.values.first.assetManager
        : AssetManager();

    // Build new runtime (pipelines + router wired for inter-pipeline links)
    final builder = RuntimeBuilder(sharedAssets: am);
    final build = builder.build(newConfig);

    // Replace all live pipelines
    livePipelines
      ..clear()
      ..addEntries(build.pipelines.map((p) => MapEntry(p.id, p)));

    return RuntimePatchApplyResult(config: newConfig, runtime: build);
  }
}

class RuntimePatchApplyResult {
  final NetworkConfig config;
  final RuntimeBuildResult runtime;
  RuntimePatchApplyResult({required this.config, required this.runtime});
}
