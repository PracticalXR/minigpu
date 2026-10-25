import 'dart:typed_data';
import 'dart:async';
import 'package:test/test.dart';
import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_pipeline/src/pipeline_stage_builder.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'mocks.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  setUpAll(() async {
    if (!Minigpu().isInitialized) {
      await Minigpu().init();
    }
  });

  test('merged stage runs after both streams provide tensors', () async {
    final pipeline = Pipeline(
      id: 'merge_pipeline',
      enableBenchmarking: true,
      debugEnabled: true,
    );

    final pre = StageBuilder(
      'pre',
    ).op('multiplyScalar', arguments: [1.0]).build();
    final merged = StageBuilder(
      'merged',
    ).op('addScalar', arguments: [0.0]).build();

    // Pre-merge per-stream stage
    pipeline.addStage(pre);
    // First merged routing stage
    pipeline.addMergeStage(merged);

    final a = MockMediaStream('a', MediaStreamType.audio);
    final b = MockMediaStream('b', MediaStreamType.audio);
    pipeline.addStream(a);
    pipeline.addStream(b);

    await pipeline.start();

    // Send tensors on both streams
    Future<Tensor> make(int n) async {
      final t = await Tensor.create([32]);
      await t.write(Float32List(32)..fillRange(0, 32, n.toDouble()));
      return t;
    }

    a.simulateTensorUpdate(await make(1));
    b.simulateTensorUpdate(await make(2));

    // Allow processing and merge
    await Future.delayed(const Duration(milliseconds: 80));

    // After merged processing, final_merged output should exist
    final out = pipeline.getStageOutput('final', 'merged');
    expect(out, isNotNull);

    await pipeline.stop();
    await pipeline.dispose();
  });
}
