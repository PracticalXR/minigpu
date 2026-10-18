import 'dart:io';
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:gpu_pipeline/src/serialization/runtime_builder.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';
import 'package:gpu_pipeline/src/pipeline_assets.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

void main() {
  group('ML linear layer (elementwise) — weights from file', () {
    late Directory tempDir;
    late File weightsFile;
    late Float32List weightsF32;
    const N = 32;

    ShapeExprConfig fixedShape(int len) => ShapeExprConfig(
      op: 'fixed',
      raw: {
        'op': 'fixed',
        'shape': [len],
      },
    );

    setUp(() async {
      tempDir = await Directory.systemTemp.createTemp('ml_linear_');
      weightsFile = File('${tempDir.path}\\w.bin');
      weightsF32 = Float32List.fromList(
        List<double>.generate(N, (i) => 0.5 + i / N),
      );
      final bytes = Uint8List.view(weightsF32.buffer);
      await weightsFile.writeAsBytes(bytes, flush: true);
    });

    tearDown(() async {
      if (await tempDir.exists()) {
        await tempDir.delete(recursive: true);
      }
    });

    test(
      'pipeline runs one shader pass and uses weights loaded from file',
      () async {
        const wgsl = '''
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&y)) { return; }
  y[i] = x[i] * w[i];
}
''';

        final net = NetworkConfig(
          schemaVersion: 1,
          assets: [
            AssetConfig(
              name: 'model_weights',
              source: {'type': 'file', 'path': weightsFile.path},
              typeHint: 'weights',
              required: true,
            ),
          ],
          pipelines: [
            PipelineConfig(
              id: 'lin',
              queue: const {'strategy': 'latest_only'},
              stages: [
                StageConfig(
                  stageId: 'mul',
                  stageKind: 'DynamicStage',
                  inputs: [
                    PortConfig(
                      name: 'x',
                      portType: 'flexible_input',
                      shapeExpr: fixedShape(N),
                    ),
                    PortConfig(
                      name: 'w',
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
                      inputKeys: const ['x', 'w'],
                      outputKeys: const ['y'],
                      wgsl: wgsl,
                    ),
                  ],
                ),
              ],
              connections: const [],
            ),
          ],
          interPipelineLinks: const [],
        );

        final assets = AssetManager();
        final builder = RuntimeBuilder(sharedAssets: assets);
        final runtime = builder.build(net);
        final pipeline = runtime.pipelines.single;

        await pipeline.start();

        // Prepare inputs: x from values, w from loaded asset bytes.
        final x = await Tensor.zeros([N]);
        final xData = Float32List.fromList(
          List<double>.generate(N, (i) => i / (N - 1)),
        );
        await x.write(xData);

        final wAsset = assets.get('model_weights')!;
        final w = await Tensor.zeros([N]);
        final wBytes = wAsset.bytes!;
        final wData = Float32List.view(wBytes.buffer, wBytes.offsetInBytes, N);
        await w.write(wData);

        final out = await pipeline.runOnce({'x': x, 'w': w});
        final y = out['y'];
        expect(y, isNotNull);

        final yData = await y!.getData() as Float32List;
        expect(yData.length, N);

        for (int i = 0; i < N; i++) {
          expect(yData[i], closeTo(xData[i] * wData[i], 1e-5));
        }

        await pipeline.stop();
        await assets.dispose();
      },
    );
  });
}
