import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';

ShapeExprConfig _fixedShape(int len) => ShapeExprConfig(
  op: 'fixed',
  raw: {
    'op': 'fixed',
    'shape': [len],
  },
);

class CustomCodecPipelines {
  // Encoder: RGBA Float32 [0..255] -> luma Float32 [0..255] (packet)
  static NetworkConfig buildEncoderNetwork({
    required int width,
    required int height,
  }) {
    final numPixels = width * height;
    final rgbaLen = numPixels * 4;

    const encoderWgsl = '''
@group(0) @binding(0) var<storage, read> rgba: array<f32>;
@group(0) @binding(1) var<storage, read_write> packet: array<f32>;

fn clamp255(v: f32) -> f32 {
  return max(0.0, min(255.0, v));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let base = i * 4u;

  if (base + 3u >= arrayLength(&rgba)) { return; }
  if (i >= arrayLength(&packet)) { return; }

  let r = rgba[base + 0u];
  let g = rgba[base + 1u];
  let b = rgba[base + 2u];

  // Rec.709 luma approx
  let y = clamp255(0.2126 * r + 0.7152 * g + 0.0722 * b);
  packet[i] = y;
}
''';

    return NetworkConfig(
      schemaVersion: 1,
      pipelines: [
        PipelineConfig(
          id: 'encoder',
          queue: {'strategy': 'latest_only'},
          stages: [
            StageConfig(
              stageId: 'encode',
              stageKind: 'DynamicStage',
              parameters: {'width': width, 'height': height},
              inputs: [
                PortConfig(
                  name: 'rgba',
                  portType: 'flexible_input',
                  shapeExpr: _fixedShape(rgbaLen),
                ),
              ],
              outputs: [
                PortConfig(
                  name: 'packet',
                  portType: 'dynamic_output',
                  shapeExpr: _fixedShape(numPixels),
                ),
              ],
              operations: [
                OperationConfig(
                  opKind: 'shader',
                  inputKeys: ['rgba'],
                  outputKeys: ['packet'],
                  wgsl: encoderWgsl,
                ),
              ],
            ),
          ],
          connections: const [],
        ),
      ],
    );
  }

  // Decoder: luma Float32 [0..255] -> grayscale RGBA Float32 [0..255]
  static NetworkConfig buildDecoderNetwork({
    required int width,
    required int height,
  }) {
    final numPixels = width * height;
    final rgbaLen = numPixels * 4;

    const decoderWgsl = '''
@group(0) @binding(0) var<storage, read> packet: array<f32>;
@group(0) @binding(1) var<storage, read_write> rgba: array<f32>;

fn clamp255(v: f32) -> f32 {
  return max(0.0, min(255.0, v));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&packet)) { return; }
  let v = clamp255(packet[i]);
  let base = i * 4u;
  if (base + 3u >= arrayLength(&rgba)) { return; }

  rgba[base + 0u] = v;
  rgba[base + 1u] = v;
  rgba[base + 2u] = v;
  rgba[base + 3u] = 255.0;
}
''';

    return NetworkConfig(
      schemaVersion: 1,
      pipelines: [
        PipelineConfig(
          id: 'decoder',
          queue: {'strategy': 'latest_only'},
          stages: [
            StageConfig(
              stageId: 'decode',
              stageKind: 'DynamicStage',
              parameters: {'width': width, 'height': height},
              inputs: [
                PortConfig(
                  name: 'packet',
                  portType: 'flexible_input',
                  shapeExpr: _fixedShape(numPixels),
                ),
              ],
              outputs: [
                PortConfig(
                  name: 'rgba',
                  portType: 'dynamic_output',
                  shapeExpr: _fixedShape(rgbaLen),
                ),
              ],
              operations: [
                OperationConfig(
                  opKind: 'shader',
                  inputKeys: ['packet'],
                  outputKeys: ['rgba'],
                  wgsl: decoderWgsl,
                ),
              ],
            ),
          ],
          connections: const [],
        ),
      ],
    );
  }
}
