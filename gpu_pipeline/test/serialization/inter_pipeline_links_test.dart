import 'dart:async';
import 'dart:typed_data';
import 'package:gpu_pipeline/src/pipeline_event.dart';
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

import 'package:gpu_pipeline/src/serialization/runtime_builder.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';
import 'package:gpu_pipeline/src/pipeline_assets.dart';

void main() {
  ShapeExprConfig fixedShape(int len) => ShapeExprConfig(
    op: 'fixed',
    raw: {
      'op': 'fixed',
      'shape': [len],
    },
  );

  const copyWgsl = '''
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&dst)) { return; }
  dst[i] = src[i];
}
''';

  group('Inter-pipeline links', () {
    test('latest_autorun triggers B on A completion', () async {
      const N = 8;

      final net = NetworkConfig(
        schemaVersion: 1,
        assets: const [],
        pipelines: [
          PipelineConfig(
            id: 'pA',
            queue: const {'strategy': 'latest_only'},
            stages: [
              StageConfig(
                stageId: 'copyA',
                stageKind: 'DynamicStage',
                inputs: [
                  PortConfig(
                    name: 'a',
                    portType: 'flexible_input',
                    shapeExpr: fixedShape(N),
                  ),
                ],
                outputs: [
                  PortConfig(
                    name: 'b',
                    portType: 'dynamic_output',
                    shapeExpr: fixedShape(N),
                  ),
                ],
                operations: [
                  OperationConfig(
                    opKind: 'shader',
                    inputKeys: const ['a'],
                    outputKeys: const ['b'],
                    wgsl: copyWgsl,
                  ),
                ],
              ),
            ],
            connections: const [],
          ),
          PipelineConfig(
            id: 'pB',
            queue: const {'strategy': 'latest_only'},
            stages: [
              StageConfig(
                stageId: 'copyB',
                stageKind: 'DynamicStage',
                inputs: [
                  PortConfig(
                    name: 'x',
                    portType: 'flexible_input',
                    shapeExpr: fixedShape(N),
                  ),
                ],
                outputs: [
                  PortConfig(
                    name: 'y',
                    portType: 'dynamic_output',
                    shapeExpr: fixedShape(N),
                  ),
                ],
                operations: [
                  OperationConfig(
                    opKind: 'shader',
                    inputKeys: const ['x'],
                    outputKeys: const ['y'],
                    wgsl: copyWgsl,
                  ),
                ],
              ),
            ],
            connections: const [],
          ),
        ],
        interPipelineLinks: const [
          InterPipelineLinkConfig(
            fromPipeline: 'pA',
            toPipeline: 'pB',
            fromOutput: 'b',
            toInput: 'x',
            mode: InterPipelineLinkMode.latestAutorun,
          ),
        ],
      );

      final shared = AssetManager();
      final builder = RuntimeBuilder(sharedAssets: shared);
      final runtime = builder.build(net);

      final pA = runtime.pipelines.firstWhere((p) => p.id == 'pA');
      final pB = runtime.pipelines.firstWhere((p) => p.id == 'pB');

      await pA.start();
      await pB.start();

      // Prepare input for A
      final a = await Tensor.zeros([N]);
      final src = Float32List.fromList(
        List<double>.generate(N, (i) => i.toDouble()),
      );
      await a.write(src);

      // Listen for B completion before firing A
      final completer = Completer<void>();
      final sub = pB.eventNotifier.eventStream.listen((e) {
        if (e is PipelineRunCompleteEvent) completer.complete();
      });

      // Run A; router should stash b into B.x and autorun B
      final outA = await pA.runOnce({'a': a});
      expect(outA['b'], isNotNull);

      // Await B autorun completion
      await completer.future.timeout(const Duration(seconds: 2));
      await sub.cancel();

      final latestB = pB.getLatestFinalOutputs()!;
      final y = latestB['y']!;
      final yData = await y.getData() as Float32List;

      expect(yData.length, N);
      for (int i = 0; i < N; i++) {
        expect(yData[i], closeTo(src[i], 1e-5));
      }

      await pA.stop();
      await pB.stop();
      await runtime.dispose(); // disposes router
      await shared.dispose();
    });
  });
}
