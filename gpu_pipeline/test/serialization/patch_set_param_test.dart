import 'package:test/test.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_loader.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_patch.dart';
import 'package:gpu_pipeline/src/serialization/patch_executor.dart';

void main() {
  test('set_param patch updates live pipeline stage parameter', () {
    final net = NetworkConfig(
      schemaVersion: 1,
      pipelines: [
        PipelineConfig(
          id: 'p',
          queue: {'strategy': 'latest_only'},
          stages: [
            StageConfig(
              stageId: 's1',
              stageKind: 'DynamicStage',
              parameters: {'gain': 1.0},
            ),
          ],
          connections: [],
        ),
      ],
    );

    final loader = PipelineLoader();
    final load = loader.loadFromJson(net.encode());
    final pipeline = load.runtime.pipelines.first;

    final patch = PatchEnvelope(
      pipelineId: 'p',
      ops: [SetParamOp(stageId: 's1', name: 'gain', value: 2.5)],
    );

    final exec = RuntimePatchExecutor({'p': pipeline});
    final applied = exec.apply(load.config, patch);
    expect(applied, isTrue);

    final stage = pipeline.getStage('s1');
    expect(stage?.getPropertyValue('gain'), equals(2.5));
  });
}
