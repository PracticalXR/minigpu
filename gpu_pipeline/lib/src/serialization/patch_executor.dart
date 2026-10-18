import 'pipeline_models.dart';
import 'pipeline_patch.dart';
import 'runtime_builder.dart';
import '../pipeline.dart';
import '../pipeline_assets.dart';

/// Applies patches to live pipelines.
/// Tip: Structural patches rebuild the entire runtime so inter-pipeline links keep working.
class RuntimePatchExecutor {
  final Map<String, Pipeline> livePipelines;
  RuntimePatchExecutor(this.livePipelines);

  /// Fast param-only application in place. Returns true if applied fully.
  bool apply(NetworkConfig currentConfig, PatchEnvelope env) {
    final pipeline = livePipelines[env.pipelineId];
    if (pipeline == null) return false;

    final hasStructural = env.ops.any(
      (o) => o is ReplaceStageOp || o is AddOperationOp,
    );

    if (!hasStructural) {
      for (final op in env.ops) {
        if (op is SetParamOp) {
          final stage = pipeline.getStage(op.stageId);
          if (stage == null) continue;
          try {
            stage.setPropertyValue(op.name, op.value);
          } catch (_) {
            return false;
          }
        } else {
          return false;
        }
      }
      return true;
    }

    // Structural present: signal caller to use applyAndRebuild so router is refreshed.
    return false;
  }

  /// Structural rebuild that returns a new runtime and replaces livePipelines wholesale.
  /// Caller should dispose the previous runtime/router.
  RuntimePatchApplyResult applyAndRebuild(
    NetworkConfig currentConfig,
    PatchEnvelope env,
  ) {
    // Apply patch to config JSON-wise
    final applier = PatchApplier();
    final newConfig = applier.apply(currentConfig, env);

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
