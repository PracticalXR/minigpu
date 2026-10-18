import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

import 'package:gpu_pipeline/src/serialization/runtime_builder.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';
import 'package:gpu_pipeline/src/pipeline_assets.dart';
import 'package:gpu_pipeline/src/pipeline_event.dart';

void main() {
  ShapeExprConfig fixedShape(int len) => ShapeExprConfig(
    op: 'fixed',
    raw: {
      'op': 'fixed',
      'shape': [len],
    },
  );

  const mul2Wgsl = '''
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&dst)) { return; }
  dst[i] = src[i] * 2.0;
}
''';

  const squareWgsl = '''
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&dst)) { return; }
  let v = src[i];
  dst[i] = v * v;
}
''';

  const packWgsl = '''
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> outbuf: array<f32>;

fn clamp01(v: f32) -> f32 {
  if (v < 0.0) { return 0.0; }
  if (v > 1.0) { return 1.0; }
  return v;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = arrayLength(&src);
  if (i >= n) { return; }
  let r = clamp01(src[i]);
  let base = i * 4u;
  if (base + 3u >= arrayLength(&outbuf)) { return; }
  outbuf[base]     = r;
  outbuf[base + 1] = r * 0.5;
  outbuf[base + 2] = 1.0 - r;
  outbuf[base + 3] = 1.0;
}
''';

  group('Inter-pipeline chain (A -> B -> C) with latest_autorun', () {
    late Directory tempDir;
    late File weightsFile;
    late Uint8List weightsBytes;

    setUp(() async {
      tempDir = await Directory.systemTemp.createTemp('chain_autorun_');
      weightsFile = File('${tempDir.path}\\w.bin');
      weightsBytes = Uint8List.fromList(
        List<int>.generate(64, (i) => (i * 3) & 0xFF),
      );
      await weightsFile.writeAsBytes(weightsBytes, flush: true);
    });

    tearDown(() async {
      if (await tempDir.exists()) {
        await tempDir.delete(recursive: true);
      }
    });

    test(
      'A doubles, B squares, C packs RGBA; links autorun through chain',
      () async {
        const N = 32;

        final net = NetworkConfig(
          schemaVersion: 1,
          // Verify assets preload alongside links
          assets: [
            AssetConfig(
              name: 'model_weights',
              source: {'type': 'file', 'path': weightsFile.path},
              typeHint: 'weights',
              required: true,
            ),
          ],
          pipelines: [
            // Pipeline A: x -> a_out (mul2)
            PipelineConfig(
              id: 'pA',
              queue: const {'strategy': 'latest_only'},
              stages: [
                StageConfig(
                  stageId: 'mul2',
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
                      name: 'a_out',
                      portType: 'dynamic_output',
                      shapeExpr: fixedShape(N),
                    ),
                  ],
                  operations: [
                    OperationConfig(
                      opKind: 'shader',
                      inputKeys: const ['x'],
                      outputKeys: const ['a_out'],
                      wgsl: mul2Wgsl,
                    ),
                  ],
                ),
              ],
              connections: const [],
            ),
            // Pipeline B: b_in -> b_out (square)
            PipelineConfig(
              id: 'pB',
              queue: const {'strategy': 'latest_only'},
              stages: [
                StageConfig(
                  stageId: 'square',
                  stageKind: 'DynamicStage',
                  inputs: [
                    PortConfig(
                      name: 'b_in',
                      portType: 'flexible_input',
                      shapeExpr: fixedShape(N),
                    ),
                  ],
                  outputs: [
                    PortConfig(
                      name: 'b_out',
                      portType: 'dynamic_output',
                      shapeExpr: fixedShape(N),
                    ),
                  ],
                  operations: [
                    OperationConfig(
                      opKind: 'shader',
                      inputKeys: const ['b_in'],
                      outputKeys: const ['b_out'],
                      wgsl: squareWgsl,
                    ),
                  ],
                ),
              ],
              connections: const [],
            ),
            // Pipeline C: c_in -> c_out (pack RGBA)
            PipelineConfig(
              id: 'pC',
              queue: const {'strategy': 'latest_only'},
              stages: [
                StageConfig(
                  stageId: 'pack',
                  stageKind: 'DynamicStage',
                  inputs: [
                    PortConfig(
                      name: 'c_in',
                      portType: 'flexible_input',
                      shapeExpr: fixedShape(N),
                    ),
                  ],
                  outputs: [
                    PortConfig(
                      name: 'c_out',
                      portType: 'dynamic_output',
                      shapeExpr: fixedShape(N * 4),
                    ),
                  ],
                  operations: [
                    OperationConfig(
                      opKind: 'shader',
                      inputKeys: const ['c_in'],
                      outputKeys: const ['c_out'],
                      wgsl: packWgsl,
                    ),
                  ],
                ),
              ],
              connections: const [],
            ),
          ],
          interPipelineLinks: const [
            // pA.a_out -> pB.b_in
            InterPipelineLinkConfig(
              fromPipeline: 'pA',
              toPipeline: 'pB',
              fromOutput: 'a_out',
              toInput: 'b_in',
              mode: InterPipelineLinkMode.latestAutorun,
            ),
            // pB.b_out -> pC.c_in
            InterPipelineLinkConfig(
              fromPipeline: 'pB',
              toPipeline: 'pC',
              fromOutput: 'b_out',
              toInput: 'c_in',
              mode: InterPipelineLinkMode.latestAutorun,
            ),
          ],
        );

        final shared = AssetManager();
        final builder = RuntimeBuilder(sharedAssets: shared);
        final runtime = builder.build(net);

        final pA = runtime.pipelines.firstWhere((p) => p.id == 'pA');
        final pB = runtime.pipelines.firstWhere((p) => p.id == 'pB');
        final pC = runtime.pipelines.firstWhere((p) => p.id == 'pC');

        await pA.start();
        await pB.start();
        await pC.start();

        // Asset preloaded
        final w = shared.get('model_weights');
        expect(w, isNotNull);
        expect(w!.bytes, isNotNull);

        // Input for A: linearly spaced 0..1
        final x = await Tensor.zeros([N]);
        final xData = Float32List.fromList(
          List<double>.generate(N, (i) => i / (N - 1)),
        );
        await x.write(xData);

        // Subscribe to C completion
        final completer = Completer<void>();
        final sub = pC.eventNotifier.eventStream.listen((e) {
          if (e is PipelineRunCompleteEvent) completer.complete();
        });

        // Kick the chain by running A
        final outA = await pA.runOnce({'x': x});
        expect(outA['a_out'], isNotNull);

        // Wait for C autorun
        await completer.future.timeout(const Duration(seconds: 3));
        await sub.cancel();

        // Validate final C output
        final latestC = pC.getLatestFinalOutputs()!;
        final c = latestC['c_out']!;
        final cData = await c.getData() as Float32List;

        // Expected: r = clamp01((2x)^2) = clamp01(4 x^2)
        double clamp01(double v) => v < 0 ? 0 : (v > 1 ? 1 : v);
        for (int i = 0; i < N; i++) {
          final r = clamp01(4.0 * xData[i] * xData[i]);
          final base = i * 4;
          expect(cData[base + 0], closeTo(r, 1e-5));
          expect(cData[base + 1], closeTo(r * 0.5, 1e-5));
          expect(cData[base + 2], closeTo(1.0 - r, 1e-5));
          expect(cData[base + 3], closeTo(1.0, 1e-5));
        }

        await pA.stop();
        await pB.stop();
        await pC.stop();
        await runtime.dispose(); // dispose router
        await shared.dispose();
      },
    );
  });
}
