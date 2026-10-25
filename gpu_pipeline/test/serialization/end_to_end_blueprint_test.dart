import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

import 'package:gpu_pipeline/src/serialization/pipeline_loader.dart';
import 'package:gpu_pipeline/src/pipeline_event.dart';

void main() {
  const mul2Wgsl = '''
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> a_out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&a_out)) { return; }
  a_out[i] = x[i] * 2.0;
}
''';

  const squareWgsl = '''
@group(0) @binding(0) var<storage, read_write> b_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> b_out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&b_out)) { return; }
  let v = b_in[i];
  b_out[i] = v * v;
}
''';

  const packWgsl = '''
@group(0) @binding(0) var<storage, read_write> c_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> c_out: array<f32>;

fn clamp01(v: f32) -> f32 {
  if (v < 0.0) { return 0.0; }
  if (v > 1.0) { return 1.0; }
  return v;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&c_in)) { return; }
  let r = clamp01(c_in[i]);
  let base = i * 4u;
  if (base + 3u >= arrayLength(&c_out)) { return; }
  c_out[base + 0u] = r;
  c_out[base + 1u] = r * 0.5;
  c_out[base + 2u] = 1.0 - r;
  c_out[base + 3u] = 1.0;
}
''';

  group('End-to-end: JSON -> build -> autorun chain -> output', () {
    late Directory tempDir;
    late File weightsFile;
    late Uint8List weightsBytes;

    setUp(() async {
      tempDir = await Directory.systemTemp.createTemp('e2e_blueprint_');
      weightsFile = File('${tempDir.path}\\w.bin');
      weightsBytes = Uint8List.fromList(
        List<int>.generate(64, (i) => (i * 5) & 0xFF),
      );
      await weightsFile.writeAsBytes(weightsBytes, flush: true);
    });

    tearDown(() async {
      if (await tempDir.exists()) {
        await tempDir.delete(recursive: true);
      }
    });

    test(
      'builds from JSON, preloads assets, routes A->B->C with latest_autorun',
      () async {
        const N = 32;

        Map<String, dynamic> fixedShape(int n) => {
          'op': 'fixed',
          'shape': [n],
        };

        final jsonMap = {
          'schema_version': 1,
          'assets': [
            {
              'name': 'model_weights',
              'source': {'type': 'file', 'path': weightsFile.path},
              'type_hint': 'weights',
              'required': true,
            },
          ],
          'pipelines': [
            {
              'id': 'pA',
              'queue': {'strategy': 'latest_only'},
              'stages': [
                {
                  'stage_id': 'mul2',
                  'stage_kind': 'DynamicStage',
                  'inputs': [
                    {
                      'name': 'x',
                      'port_type': 'flexible_input',
                      'shape_expr': fixedShape(N),
                    },
                  ],
                  'outputs': [
                    {
                      'name': 'a_out',
                      'port_type': 'dynamic_output',
                      'shape_expr': fixedShape(N),
                    },
                  ],
                  'operations': [
                    {
                      'op_kind': 'shader',
                      'input_keys': ['x'],
                      'output_keys': ['a_out'],
                      'wgsl': mul2Wgsl,
                    },
                  ],
                },
              ],
              'connections': [],
            },
            {
              'id': 'pB',
              'queue': {'strategy': 'latest_only'},
              'stages': [
                {
                  'stage_id': 'square',
                  'stage_kind': 'DynamicStage',
                  'inputs': [
                    {
                      'name': 'b_in',
                      'port_type': 'flexible_input',
                      'shape_expr': fixedShape(N),
                    },
                  ],
                  'outputs': [
                    {
                      'name': 'b_out',
                      'port_type': 'dynamic_output',
                      'shape_expr': fixedShape(N),
                    },
                  ],
                  'operations': [
                    {
                      'op_kind': 'shader',
                      'input_keys': ['b_in'],
                      'output_keys': ['b_out'],
                      'wgsl': squareWgsl,
                    },
                  ],
                },
              ],
              'connections': [],
            },
            {
              'id': 'pC',
              'queue': {'strategy': 'latest_only'},
              'stages': [
                {
                  'stage_id': 'pack',
                  'stage_kind': 'DynamicStage',
                  'inputs': [
                    {
                      'name': 'c_in',
                      'port_type': 'flexible_input',
                      'shape_expr': fixedShape(N),
                    },
                  ],
                  'outputs': [
                    {
                      'name': 'c_out',
                      'port_type': 'dynamic_output',
                      'shape_expr': fixedShape(N * 4),
                    },
                  ],
                  'operations': [
                    {
                      'op_kind': 'shader',
                      'input_keys': ['c_in'],
                      'output_keys': ['c_out'],
                      'wgsl': packWgsl,
                    },
                  ],
                },
              ],
              'connections': [],
            },
          ],
          'inter_pipeline_links': [
            {
              'from': {'pipeline': 'pA', 'output': 'a_out'},
              'to': {'pipeline': 'pB', 'input': 'b_in'},
              'mode': 'latest_autorun',
            },
            {
              'from': {'pipeline': 'pB', 'output': 'b_out'},
              'to': {'pipeline': 'pC', 'input': 'c_in'},
              'mode': 'latest_autorun',
            },
          ],
        };

        final loader = PipelineLoader();
        final loaded = loader.loadFromJson(jsonEncode(jsonMap));

        // Sanity: validation should have no errors
        expect(loaded.validation.errors, isEmpty);

        // Runtime built with shared assets
        final runtime = loaded.runtime;
        expect(runtime.pipelines.length, 3);

        final pA = runtime.pipelines.firstWhere((p) => p.id == 'pA');
        final pB = runtime.pipelines.firstWhere((p) => p.id == 'pB');
        final pC = runtime.pipelines.firstWhere((p) => p.id == 'pC');

        await pA.start();
        await pB.start();
        await pC.start();

        // Asset preloaded
        final weights = loaded.assets.get('model_weights');
        expect(weights, isNotNull);
        expect(weights!.bytes, isNotNull);
        expect(weights.bytes!.length, weightsBytes.length);

        // Prepare input for A
        final xData = Float32List.fromList(
          List<double>.generate(N, (i) => i / (N - 1)),
        );
        final x = await Tensor.create([N], data: xData);

        // Subscribe to C completion before firing A
        final doneC = Completer<void>();
        final sub = pC.eventNotifier.eventStream.listen((e) {
          if (e is PipelineRunCompleteEvent) doneC.complete();
        });

        // Kick the chain by running A; B and C should autorun
        final outA = await pA.runOnce({'x': x});
        expect(outA['a_out'], isNotNull);

        await doneC.future.timeout(const Duration(seconds: 3));
        await sub.cancel();

        // Verify final packed output from C
        final latestC = pC.getLatestFinalOutputs()!;
        final c = latestC['c_out']!;
        final cData = await c.getData() as Float32List;

        double clamp01(double v) => v < 0 ? 0 : (v > 1 ? 1 : v);
        for (int i = 0; i < N; i++) {
          final r = clamp01(4.0 * xData[i] * xData[i]); // (2x)^2
          final base = i * 4;
          expect(cData[base + 0], closeTo(r, 1e-5));
          expect(cData[base + 1], closeTo(r * 0.5, 1e-5));
          expect(cData[base + 2], closeTo(1.0 - r, 1e-5));
          expect(cData[base + 3], closeTo(1.0, 1e-5));
        }

        await pA.stop();
        await pB.stop();
        await pC.stop();
        await runtime.dispose(); // disposes router
        await loaded.assets.dispose();
      },
    );
  });
}
