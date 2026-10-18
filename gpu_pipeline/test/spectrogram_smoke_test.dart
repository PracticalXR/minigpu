import 'dart:math' as math;
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_loader.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';

void main() {
  test('spectrogram pipeline with CPU window stage smoke test', () async {
    const windowSize = 8;

    // Hann window as a per-element EXPRESSION (the dart_eval dynamic-CPU
    // stage this test used before is gone — its analyzer dependency broke
    // web builds; ExpressionCPUOperation is the web+native replacement).
    const windowExpression =
        'n > 1 ? x * (0.5 - 0.5 * cos(2 * pi * i / (n - 1))) : x';

    // Helper fixed shape expr builders
    ShapeExprConfig fixedShape(int len) => ShapeExprConfig(
      op: 'fixed',
      raw: {
        'op': 'fixed',
        'shape': [len],
      },
    );

    final magnitudeWgsl = '''
@group(0) @binding(0) var<storage, read>  fft_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> magnitude: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&fft_in)) { return; }
  // (If fft output is complex interleaved, replace with sqrt(re*re+im*im))
  magnitude[i] = abs(fft_in[i]);
}
''';

    final heightmapWgsl = '''
@group(0) @binding(0) var<storage, read>  mag_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> heights: array<f32>;
@group(0) @binding(2) var<storage, read_write> magnitudes: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&mag_in)) { return; }
  let v = mag_in[i];
  heights[i] = v;
  magnitudes[i] = v;
}
''';

    final renderWgsl = '''
@group(0) @binding(0) var<storage, read> heights: array<f32>;
@group(0) @binding(1) var<storage, read_write> magnitudes: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

fn clamp01(v: f32) -> f32 {
  if (v < 0.0) { return 0.0; }
  if (v > 1.0) { return 1.0; }
  return v;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&heights)) { return; }
  let vv = clamp01(heights[i]);
  let base = i * 4u;
  if (base + 3u >= arrayLength(&output)) { return; }
  output[base] = vv;
  output[base + 1u] = vv * 0.5;
  output[base + 2u] = 1.0 - vv;
  output[base + 3u] = 1.0;
}
''';

    final net = NetworkConfig(
      schemaVersion: 1,
      pipelines: [
        PipelineConfig(
          id: 'spec3d',
          queue: {'strategy': 'latest_only'},
          stages: [
            // CPU window stage (unchanged logic)
            StageConfig(
              stageId: 'window',
              stageKind: 'DynamicStage',
              inputs: [
                PortConfig(
                  name: 'input',
                  portType: 'flexible_input',
                  shapeExpr: fixedShape(windowSize),
                ),
              ],
              outputs: [
                PortConfig(
                  name: 'windowed',
                  portType: 'dynamic_output',
                  shapeExpr: fixedShape(windowSize),
                ),
              ],
              operations: [
                OperationConfig(
                  opKind: 'expression_cpu',
                  inputKeys: ['input'],
                  outputKeys: ['windowed'],
                  namedArguments: {'windowed': windowExpression},
                ),
              ],
            ),
            // FFT (tensor op)
            StageConfig(
              stageId: 'fft',
              stageKind: 'DynamicStage',
              inputs: [
                PortConfig(
                  name: 'windowed',
                  portType: 'flexible_input',
                  shapeExpr: fixedShape(windowSize),
                ),
              ],
              outputs: [
                PortConfig(
                  name: 'fft',
                  portType: 'dynamic_output',
                  shapeExpr: fixedShape(windowSize),
                ),
              ],
              operations: [
                OperationConfig(
                  opKind: 'tensor_op',
                  operation: 'fft',
                  inputKeys: ['windowed'],
                  outputKeys: ['fft'],
                ),
              ],
            ),
            // GPU magnitude
            StageConfig(
              stageId: 'magnitude',
              stageKind: 'DynamicStage',
              inputs: [
                PortConfig(
                  name: 'fft',
                  portType: 'flexible_input',
                  shapeExpr: fixedShape(windowSize),
                ),
              ],
              outputs: [
                PortConfig(
                  name: 'magnitude',
                  portType: 'dynamic_output',
                  shapeExpr: fixedShape(windowSize),
                ),
              ],
              operations: [
                OperationConfig(
                  opKind: 'shader',
                  inputKeys: ['fft'],
                  outputKeys: ['magnitude'],
                  // Assuming field name 'wgslCode'; adjust if API differs
                  wgsl: magnitudeWgsl,
                ),
              ],
            ),
            // GPU heightmap (duplicating)
            StageConfig(
              stageId: 'heightmap',
              stageKind: 'DynamicStage',
              inputs: [
                PortConfig(
                  name: 'magnitude',
                  portType: 'flexible_input',
                  shapeExpr: fixedShape(windowSize),
                ),
              ],
              outputs: [
                PortConfig(
                  name: 'heights',
                  portType: 'dynamic_output',
                  shapeExpr: fixedShape(windowSize),
                ),
                PortConfig(
                  name: 'magnitudes',
                  portType: 'dynamic_output',
                  shapeExpr: fixedShape(windowSize),
                ),
              ],
              operations: [
                OperationConfig(
                  opKind: 'shader',
                  inputKeys: ['magnitude'],
                  outputKeys: ['heights', 'magnitudes'],
                  wgsl: heightmapWgsl,
                ),
              ],
            ),
            // GPU render
            StageConfig(
              stageId: 'render',
              stageKind: 'DynamicStage',
              inputs: [
                PortConfig(
                  name: 'heights',
                  portType: 'flexible_input',
                  shapeExpr: fixedShape(windowSize),
                ),
                PortConfig(
                  name: 'magnitudes',
                  portType: 'flexible_input',
                  shapeExpr: fixedShape(windowSize),
                ),
              ],
              outputs: [
                PortConfig(
                  name: 'output',
                  portType: 'dynamic_output',
                  shapeExpr: fixedShape(windowSize * 4),
                ),
              ],
              operations: [
                OperationConfig(
                  opKind: 'shader',
                  inputKeys: ['heights', 'magnitudes'],
                  outputKeys: ['output'],
                  wgsl: renderWgsl,
                ),
              ],
            ),
          ],
          connections: [
            ConnectionConfig(
              fromStage: 'window',
              toStage: 'fft',
              fromOutput: 'windowed',
              toInput: 'windowed',
            ),
            ConnectionConfig(
              fromStage: 'fft',
              toStage: 'magnitude',
              fromOutput: 'fft',
              toInput: 'fft',
            ),
            ConnectionConfig(
              fromStage: 'magnitude',
              toStage: 'heightmap',
              fromOutput: 'magnitude',
              toInput: 'magnitude',
            ),
            ConnectionConfig(
              fromStage: 'heightmap',
              toStage: 'render',
              fromOutput: 'heights',
              toInput: 'heights',
            ),
            ConnectionConfig(
              fromStage: 'heightmap',
              toStage: 'render',
              fromOutput: 'magnitudes',
              toInput: 'magnitudes',
            ),
          ],
        ),
      ],
    );

    final loader = PipelineLoader();
    final load = loader.loadFromJson(net.encode());
    expect(load.validation.ok, isTrue);

    final pipeline = load.runtime.pipelines.first;

    final input = await Tensor.zeros([windowSize]);
    final src = List<double>.generate(windowSize, (i) => (i + 1).toDouble());
    await input.write(Float32List.fromList(src));

    final outputs = await pipeline.runOnce({'input': input});

    pipeline.debugPrintAllOutputs();
    // Window output
    final windowStageOut = pipeline.getStageOutput('window');
    print('Window stage output: $windowStageOut');
    final windowed = windowStageOut!['windowed'];
    expect(windowed, isNotNull);
    final windowedData = (await windowed!.getData() as Float32List).toList();

    // Expected Hann
    final expected = <double>[];
    for (var n = 0; n < windowSize; n++) {
      final w = 0.5 - 0.5 * math.cos(2 * math.pi * n / (windowSize - 1));
      expected.add(src[n] * w);
    }
    for (var i = 0; i < expected.length; i++) {
      expect(windowedData[i], closeTo(expected[i], 1e-5));
    }

    // Render output
    await pipeline.debugPrintAllOutputs();
    final renderOut = outputs['output'];
    expect(renderOut, isNotNull);
    final renderData = await renderOut!.getData() as Float32List;
    expect(renderData.length, windowSize * 4);

    pipeline.debugPrintAllOutputs(getOutputs: true);

    // --- Added texture validity + channel invariant checks ---
    // Interpret as RGBA tuples
    bool anyNonZero = false;
    final distinctValues = <double>{};

    for (int i = 0; i < windowSize; i++) {
      final base = i * 4;
      final r = renderData[base];
      final g = renderData[base + 1];
      final b = renderData[base + 2];
      final a = renderData[base + 3];

      // Basic finite / range checks
      expect(r.isFinite, isTrue, reason: 'R not finite at $i');
      expect(g.isFinite, isTrue, reason: 'G not finite at $i');
      expect(b.isFinite, isTrue, reason: 'B not finite at $i');
      expect(a.isFinite, isTrue, reason: 'A not finite at $i');

      expect(r, inInclusiveRange(0.0, 1.0), reason: 'R out of [0,1] at $i');
      expect(g, inInclusiveRange(0.0, 1.0), reason: 'G out of [0,1] at $i');
      expect(b, inInclusiveRange(0.0, 1.0), reason: 'B out of [0,1] at $i');
      expect(a, inInclusiveRange(0.0, 1.0), reason: 'A out of [0,1] at $i');

      // Shader invariants from renderWgsl:
      // g == r * 0.5, b == 1 - r, a == 1
      expect(g, closeTo(r * 0.5, 1e-5), reason: 'G != R*0.5 at $i');
      expect(b, closeTo(1.0 - r, 1e-5), reason: 'B != 1-R at $i');
      expect(a, closeTo(1.0, 1e-5), reason: 'A != 1 at $i');

      if (r > 0.0) anyNonZero = true;
      if (r > 1e-6) distinctValues.add(double.parse(r.toStringAsFixed(5)));
    }

    expect(anyNonZero, isTrue, reason: 'All red (magnitude) values are zero');
    // Heuristic: ensure more than one distinct magnitude value so we did not just fill a single pixel (texture sliver issue)
    expect(
      distinctValues.length,
      greaterThan(1),
      reason:
          'Expected multiple distinct magnitude samples, got ${distinctValues.length}: $distinctValues',
    );

    // Optional: ensure last pixel written (guards against only first portion populated)
    final lastBase = (windowSize - 1) * 4;
    expect(
      renderData[lastBase + 3],
      closeTo(1.0, 1e-5),
      reason: 'Alpha of last pixel not written (possible partial dispatch)',
    );
  });
}
