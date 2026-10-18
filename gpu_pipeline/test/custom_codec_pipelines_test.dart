import 'dart:typed_data';

import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_pipeline/src/web/custom_codec_pipelines.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';

// The web demos load these networks; they must keep passing the loader's
// validation and the pipeline's port checks.
void main() {
  test('the codec demo networks turn RGBA into luma and back', () async {
    const width = 16, height = 4, pixels = width * height;
    final rgba = Float32List(pixels * 4);
    final luma = <double>[];
    for (var i = 0; i < pixels; i++) {
      final r = (i * 3) % 256, g = (i * 7) % 256, b = (i * 11) % 256;
      rgba.setAll(i * 4, [r.toDouble(), g.toDouble(), b.toDouble(), 255]);
      luma.add(0.2126 * r + 0.7152 * g + 0.0722 * b);
    }

    final encoder = PipelineLoader().loadFromJson(
      CustomCodecPipelines.buildEncoderNetwork(
        width: width,
        height: height,
      ).encode(),
    );
    final decoder = PipelineLoader().loadFromJson(
      CustomCodecPipelines.buildDecoderNetwork(
        width: width,
        height: height,
      ).encode(),
    );
    expect(encoder.validation.ok, isTrue);
    expect(decoder.validation.ok, isTrue);
    final encode = encoder.runtime.pipelines.single;
    final decode = decoder.runtime.pipelines.single;

    final input = await Tensor.zeros([rgba.length]);
    await input.write(rgba);
    final packet = (await encode.runOnce({'rgba': input}))['packet']!;
    final y = await packet.getData() as Float32List;
    expect(y.length, pixels);
    for (var i = 0; i < pixels; i++) {
      expect(y[i], closeTo(luma[i], 1e-3), reason: 'luma of pixel $i');
    }

    final gray = (await decode.runOnce({'packet': packet}))['rgba']!;
    final out = await gray.getData() as Float32List;
    expect(out.length, pixels * 4);
    for (var i = 0; i < pixels; i++) {
      expect(out.sublist(i * 4, i * 4 + 4), [
        y[i],
        y[i],
        y[i],
        255,
      ], reason: 'pixel $i');
    }

    await encode.dispose();
    await decode.dispose();
    await encoder.dispose();
    await decoder.dispose();
  });
}
