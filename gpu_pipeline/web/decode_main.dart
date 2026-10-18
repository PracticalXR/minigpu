import 'dart:async' show unawaited;
import 'dart:js_interop';
import 'package:minigpu/minigpu.dart';
import 'package:web/web.dart' as web;
import 'dart:typed_data';

import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_loader.dart';
import 'package:gpu_pipeline/src/web/custom_codec_pipelines.dart';

Future<void> main() async {
  await Minigpu().init();
  final canvas =
      web.document.getElementById('decCanvas') as web.HTMLCanvasElement;
  final ctx = (canvas.getContext('2d') as web.CanvasRenderingContext2D?)!;
  final logEl = web.document.getElementById('decLog') as web.HTMLPreElement;
  final runBtn = web.document.getElementById('run') as web.HTMLButtonElement;

  final width = canvas.width;
  final height = canvas.height;
  final numPixels = width * height;

  // Build decoder
  final decNet = CustomCodecPipelines.buildDecoderNetwork(
    width: width,
    height: height,
  );
  final loader = PipelineLoader();
  final decLoad = loader.loadFromJson(decNet.encode());
  if (!decLoad.validation.ok) {
    logEl.text = 'Validation failed: ${decLoad.validation.errors}';
    return;
  }
  final decoder = decLoad.runtime.pipelines.first;

  Future<void> runDemo() async {
    logEl.text = 'Running...';

    // Synthesize a packet by encoding a test pattern locally
    final srcU8 = Uint8ClampedList(numPixels * 4);
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final i = (y * width + x) * 4;
        final gx = (255.0 * x / (width - 1)).round();
        final gy = (255.0 * y / (height - 1)).round();
        final stripe = ((y ~/ 8) % 2 == 0) ? 64 : 0;
        srcU8[i] = gx;
        srcU8[i + 1] = gy;
        srcU8[i + 2] = stripe;
        srcU8[i + 3] = 255;
      }
    }
    final rgbaF32 = Float32List(numPixels * 4);
    for (int i = 0; i < srcU8.length; i++) {
      rgbaF32[i] = srcU8[i].toDouble();
    }

    final encNet = CustomCodecPipelines.buildEncoderNetwork(
      width: width,
      height: height,
    );
    final encLoad = loader.loadFromJson(encNet.encode());
    final encoder = encLoad.runtime.pipelines.first;

    final inTensor = await Tensor.zeros([rgbaF32.length]);
    await inTensor.write(rgbaF32);
    final encOut = await encoder.runOnce({'rgba': inTensor});
    final packetF32 = await encOut['packet']!.getData() as Float32List;

    // Decode
    final packetTensor = await Tensor.zeros([packetF32.length]);
    await packetTensor.write(packetF32);
    final decOut = await decoder.runOnce({'packet': packetTensor});
    final rgba = await decOut['rgba']!.getData() as Float32List;

    // Convert Float32 [0..255] -> Uint8ClampedList and draw
    final dst = Uint8ClampedList(numPixels * 4);
    for (int i = 0; i < dst.length; i++) {
      final v = rgba[i].round();
      dst[i] = v.clamp(0, 255);
    }
    final img = web.ImageData(dst.toJS, width, height.toJS);
    ctx.putImageData(img, 0, 0);

    logEl.text = 'Done.';
  }

  // Use a JS-typed EventListener
  final web.EventListener onClick = ((web.Event _) {
    unawaited(runDemo());
  }).toJS;

  runBtn.addEventListener('click', onClick);
}
