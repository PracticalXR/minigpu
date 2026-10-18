// Shared public-Dart raster acceptance scenario.
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';

const _source = '''
struct Camera { scale: vec4<f32> }
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var image: texture_2d<f32>;
@group(0) @binding(2) var imageSampler: sampler;
struct Output {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
}
@vertex fn vsMain(@location(0) position: vec2<f32>,
  @location(1) offset: vec2<f32>, @location(2) color: vec4<f32>,
  @location(3) depth: f32) -> Output {
  var output: Output;
  output.position = vec4<f32>(position * camera.scale.xy + offset, depth, 1.);
  output.color = color;
  return output;
}
@fragment fn fsMain(input: Output) -> @location(0) vec4<f32> {
  return input.color * textureSample(image, imageSampler, vec2<f32>(.5));
}
''';

void check(bool condition, String message) {
  if (!condition) throw StateError(message);
}

Future<void> refuses(Future<void> Function() action, String message) async {
  try {
    await action();
  } on ArgumentError {
    return;
  } on StateError {
    return;
  }
  throw StateError(message);
}

Future<void> runRasterScenario() async {
  final gpu = Minigpu();
  try {
    await gpu.init();
    final color = gpu.createTexture(
      TextureDescriptor(
        width: 8,
        height: 8,
        format: TextureFormat.rgba8Unorm,
        usage: TextureUsage.renderAttachment | TextureUsage.textureBinding,
      ),
    );
    final colorView = color.createView();
    final depth = gpu.createTexture(
      TextureDescriptor(
        width: 8,
        height: 8,
        format: TextureFormat.depth32Float,
        usage: TextureUsage.renderAttachment,
      ),
    );
    final depthView = depth.createView();
    final image = gpu.createTexture(
      TextureDescriptor(
        width: 1,
        height: 1,
        format: TextureFormat.rgba8Unorm,
        usage: TextureUsage.textureBinding | TextureUsage.copyDst,
      ),
    );
    await image.write(Uint8List.fromList([255, 255, 255, 255]));
    final imageView = image.createView();
    final sampler = gpu.createSampler(SamplerDescriptor());
    Future<Buffer> buffer(Float32List values, int usage) async {
      final value = gpu.createBufferFromDescriptor(
        BufferDescriptor(
          sizeBytes: values.lengthInBytes,
          usage: usage | BufferUsage.copyDst,
        ),
      );
      await value.write(values, values.length);
      return value;
    }

    final vertices = await buffer(
      Float32List.fromList([-1, -1, 1, -1, 1, 1, -1, 1]),
      BufferUsage.vertex,
    );
    final instances = await buffer(
      Float32List.fromList([
        -.5,
        0,
        1,
        0,
        0,
        1,
        .25,
        .5,
        0,
        0,
        1,
        0,
        1,
        .25,
        0,
        0,
        0,
        0,
        1,
        1,
        .75,
      ]),
      BufferUsage.vertex,
    );
    final overlays = await buffer(
      Float32List.fromList([-.5, 0, 0, 0, 1, .5, .1, .5, 0, 0, 0, 1, .5, .1]),
      BufferUsage.vertex,
    );
    final camera = await buffer(
      Float32List.fromList([.45, .8, 0, 0]),
      BufferUsage.uniform,
    );
    final indices = gpu.createBufferFromDescriptor(
      BufferDescriptor(
        sizeBytes: 12,
        usage: BufferUsage.index | BufferUsage.copyDst,
      ),
    );
    await indices.writeRawBytes(
      Uint16List.fromList([0, 1, 2, 0, 2, 3]).buffer.asUint8List(),
    );
    RenderPipelineDescriptor descriptor(RenderBlendMode blend) =>
        RenderPipelineDescriptor(
          code: _source,
          colorFormat: TextureFormat.rgba8Unorm,
          depthEnabled: true,
          depthWrite: blend == RenderBlendMode.opaque,
          blend: blend,
          vertexBuffers: [
            VertexBufferLayout(
              arrayStride: 8,
              attributes: [
                VertexAttribute(
                  shaderLocation: 0,
                  offset: 0,
                  format: VertexFormat.float32x2,
                ),
              ],
            ),
            VertexBufferLayout(
              arrayStride: 28,
              stepMode: VertexStepMode.instance,
              attributes: [
                VertexAttribute(
                  shaderLocation: 1,
                  offset: 0,
                  format: VertexFormat.float32x2,
                ),
                VertexAttribute(
                  shaderLocation: 2,
                  offset: 8,
                  format: VertexFormat.float32x4,
                ),
                VertexAttribute(
                  shaderLocation: 3,
                  offset: 24,
                  format: VertexFormat.float32,
                ),
              ],
            ),
          ],
          bindings: [
            RenderBindingLayout(
              slot: 0,
              kind: RenderBindingKind.uniformBuffer,
              visibility: 1,
              minBindingSize: 16,
            ),
            RenderBindingLayout(
              slot: 1,
              kind: RenderBindingKind.sampledFloat,
              visibility: 2,
            ),
            RenderBindingLayout(
              slot: 2,
              kind: RenderBindingKind.nonFilteringSampler,
              visibility: 2,
            ),
          ],
        );
    final opaque = await gpu.createRenderPipeline(
      descriptor(RenderBlendMode.opaque),
    );
    final alpha = await gpu.createRenderPipeline(
      descriptor(RenderBlendMode.alpha),
    );
    void bind(GpuRenderPass pass, Buffer instanceBuffer) {
      pass.setVertexBuffer(0, vertices);
      pass.setVertexBuffer(1, instanceBuffer);
      pass.setIndexBuffer(indices, IndexFormat.uint16);
      pass.setBuffer(0, camera);
      pass.setTexture(1, imageView);
      pass.setSampler(2, sampler);
      pass.setViewport(0, 0, 8, 8);
      pass.setScissorRect(0, 0, 8, 8);
    }

    final first = opaque.createPass(
      color: colorView,
      depth: depthView,
      clearColor: [0, 0, 0, 1],
    );
    await refuses(() async => first.drawIndexed(6), 'Unbound draw accepted');
    bind(first, instances);
    await refuses(
      () async => first.drawIndexed(7),
      'Out-of-range indices accepted',
    );
    first.drawIndexed(6, instanceCount: 2);
    first.drawIndexed(6, firstInstance: 2); // Behind the first two quads.
    await first.submit();
    await refuses(first.submit, 'Consumed pass resubmitted');
    final repeated = opaque.createPass(
      color: colorView,
      depth: depthView,
      loadColor: true,
      loadDepth: true,
    );
    bind(repeated, instances);
    repeated.drawIndexed(6, instanceCount: 2);
    await repeated.submit();
    await gpu.waitForSubmittedWork();
    final bindingCache = opaque.bindingCacheSnapshot;
    check(bindingCache != null, 'Render binding cache diagnostics unavailable');
    check(
      bindingCache!.created == 1 &&
          bindingCache.reused == 1 &&
          bindingCache.retired == 0 &&
          bindingCache.live == 1,
      'Stable render bindings were not retained: ${bindingCache.toJson()}',
    );
    final second = alpha.createPass(
      color: colorView,
      depth: depthView,
      loadColor: true,
      loadDepth: true,
    );
    bind(second, overlays);
    second.drawIndexed(6, instanceCount: 2);
    await second.submit();

    final output = gpu.createBufferFromDescriptor(
      BufferDescriptor(
        sizeBytes: 8 * 8 * 16,
        usage: BufferUsage.storage | BufferUsage.copySrc,
      ),
    );
    final read = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
let pixel = vec4<u32>(round(textureLoad(src, vec2<i32>(id.xy), 0) * 255.));
let index = (id.y * 8u + id.x) * 4u;
dst[index] = pixel.r; dst[index + 1u] = pixel.g;
dst[index + 2u] = pixel.b; dst[index + 3u] = pixel.a;
}
''');
    read.setTexture(0, colorView, TextureBindingKind.sampledUnfilterableFloat);
    read.setBufferRange(
      1,
      output,
      BufferBindingDescriptor(kind: BufferBindingKind.storage, sizeBytes: 1024),
    );
    await read.dispatch(8, 8, 1);
    final pixels = Uint32List(256);
    await output.read(pixels, pixels.length, dataType: BufferDataType.uint32);
    void pixel(int x, int y, List<int> expected) {
      final actual = pixels.sublist((y * 8 + x) * 4, (y * 8 + x) * 4 + 4);
      check(
        List.generate(
          4,
          (i) => (actual[i] - expected[i]).abs() <= 1,
        ).every((value) => value),
        'Pixel $x,$y: $actual != $expected',
      );
    }

    pixel(1, 4, [128, 0, 128, 255]);
    pixel(3, 4, [128, 0, 128, 255]); // Depth rejects the blue quad.
    pixel(6, 4, [0, 128, 128, 255]);
    pixel(4, 0, [0, 0, 0, 255]);
    final stale = opaque.createPass(color: colorView, depth: depthView);
    bind(stale, instances);
    stale.drawIndexed(6);
    colorView.destroy();
    await refuses(stale.submit, 'Destroyed attachment accepted');
    stale.discard();
    await gpu.waitForSubmittedWork();
    await gpu.destroy();
    check(
      !opaque.isValid && !alpha.isValid && !depth.isValid,
      'Owned render resources survived context close',
    );
  } finally {
    if (gpu.isInitialized) await gpu.destroy();
  }
}
