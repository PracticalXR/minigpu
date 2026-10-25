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

  group('Queue strategies', () {
    late Pipeline pipeline;
    late MockMediaStream stream;

    setUp(() async {
      stream = MockMediaStream('s', MediaStreamType.audio);
    });

    tearDown(() async {
      await pipeline.stop();
      await pipeline.dispose();
    });

    test('latestN keeps only N per stream', () async {
      // Slow mock to force queue trimming; echoes input payload to output
      final pass = MockPipelineStage(
        stageId: 'pass',
        processingDelay: const Duration(milliseconds: 6),
      );

      pipeline = Pipeline(
        id: 'queue_test_latestN',
        enableBenchmarking: true,
        queueConfig: const QueueConfig(
          strategy: QueueStrategy.latestN,
          maxItemsPerStream: 2,
          maxTotalItems: 100,
        ),
      );
      pipeline.addStage(pass);
      pipeline.addStream(stream);
      await pipeline.start();

      Future<Tensor> make(int v) async {
        final t = await Tensor.create([16]);
        final data = Float32List(16)..fillRange(0, 16, v.toDouble());
        await t.write(data);
        return t;
      }

      for (int i = 1; i <= 5; i++) {
        stream.simulateTensorUpdate(await make(i));
        await Future.delayed(const Duration(milliseconds: 1));
      }

      await Future.delayed(const Duration(milliseconds: 150));

      expect(pipeline.hasStageOutput('pass'), isTrue);

      final out = pipeline.getStageOutput('pass', 's')!;
      final outTensor = out.values.first;
      final outData = await outTensor.getData() as Float32List;
      final lastVal = outData[0].toInt();
      expect(lastVal, inInclusiveRange(4, 5));
    });

    test('dropOnOverrun drops when processing slower than input', () async {
      final slow = MockPipelineStage(
        stageId: 'slow',
        processingDelay: const Duration(milliseconds: 10),
      );
      pipeline = Pipeline(
        id: 'queue_drop',
        enableBenchmarking: true,
        queueConfig: const QueueConfig(
          strategy: QueueStrategy.dropOnOverrun,
          maxTotalItems: 2,
        ),
      );
      pipeline.addStage(slow);
      pipeline.addStream(stream);
      await pipeline.start();

      for (int i = 0; i < 10; i++) {
        final t = await Tensor.create([64]);
        await t.write(Float32List(64)..[0] = i.toDouble());
        stream.simulateTensorUpdate(t);
        await Future.delayed(const Duration(milliseconds: 1));
      }

      await Future.delayed(const Duration(milliseconds: 150));
      expect(pipeline.hasStageOutput('slow'), isTrue);
    });

    test('batched waits for batchSize or maxAge', () async {
      pipeline = Pipeline(
        id: 'queue_batch',
        enableBenchmarking: true,
        queueConfig: const QueueConfig(
          strategy: QueueStrategy.batched,
          batchSize: 3,
          maxAge: Duration(milliseconds: 30),
        ),
      );
      // Use a simple stage that produces output
      final stage = StageBuilder(
        'batch_stage',
      ).op('multiplyScalar', arguments: [1.0]).build();
      pipeline.addStage(stage);
      pipeline.addStream(stream);
      await pipeline.start();

      for (int i = 0; i < 2; i++) {
        final t = await Tensor.create([16]);
        await t.write(Float32List(16)..[0] = i.toDouble());
        stream.simulateTensorUpdate(t);
      }

      // Not enough to trigger batchSize; wait for maxAge to flush
      await Future.delayed(const Duration(milliseconds: 80));

      expect(pipeline.hasStageOutput('batch_stage'), isTrue);
    });
  });
}
