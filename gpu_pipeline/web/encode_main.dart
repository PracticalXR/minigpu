import 'dart:async';
import 'dart:js_interop';
import 'package:minigpu/minigpu.dart';
import 'package:web/web.dart' as web;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_loader.dart';
import 'package:gpu_pipeline/src/web/custom_codec_pipelines.dart';

Future<void> main() async {
  print('Starting encode demo...');
  await Minigpu().init();
  final canvas =
      web.document.getElementById('encCanvas') as web.HTMLCanvasElement;
  final ctx = (canvas.getContext('2d') as web.CanvasRenderingContext2D?)!;
  final logEl = web.document.getElementById('encLog') as web.HTMLPreElement;

  final width = canvas.width;
  final height = canvas.height;
  final numPixels = width * height;

  // Build a test pattern into a Dart Uint8ClampedList, then wrap in ImageData
  final data = Uint8ClampedList(numPixels * 4);
  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      final i = (y * width + x) * 4;
      final gx = (255.0 * x / (width - 1)).round();
      final gy = (255.0 * y / (height - 1)).round();
      final stripe = ((y ~/ 8) % 2 == 0) ? 64 : 0;
      data[i] = gx; // R
      data[i + 1] = gy; // G
      data[i + 2] = stripe; // B
      data[i + 3] = 255; // A
    }
  }
  final img = web.ImageData(data.toJS, width, height.toJS);
  ctx.putImageData(img, 0, 0);

  // Convert to Float32 RGBA [0..255]
  final rgbaF32 = Float32List(numPixels * 4);
  for (int i = 0; i < data.length; i++) {
    rgbaF32[i] = data[i].toDouble();
  }

  // Build encoder pipeline
  final net = CustomCodecPipelines.buildEncoderNetwork(
    width: width,
    height: height,
  );
  final loader = PipelineLoader();
  final load = loader.loadFromJson(net.encode());
  if (!load.validation.ok) {
    logEl.text = 'Validation failed: ${load.validation.errors}';
    return;
  }
  final pipeline = load.runtime.pipelines.first;

  // Feed RGBA tensor
  final inTensor = await Tensor.zeros([rgbaF32.length]);
  await inTensor.write(rgbaF32);

  final t0 = web.window.performance.now();
  final out = await pipeline.runOnce({'rgba': inTensor});
  final t1 = web.window.performance.now();

  final packet = out['packet']!;
  final packetData = await packet.getData() as Float32List;

  // Simple stats
  final minV = packetData.reduce(math.min);
  final maxV = packetData.reduce(math.max);
  logEl.text = [
    'Encoded ${rgbaF32.length} floats -> ${packetData.length} floats',
    'Time: ${(t1 - t0).toStringAsFixed(2)} ms',
    'Packet range: [$minV, $maxV]',
  ].join('\n');
}
