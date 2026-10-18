import 'dart:typed_data';

import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';

/// The README's stream example.
class SampleStream extends MediaStream {
  SampleStream()
    : super(
        id: 'samples',
        deviceId: 'generated',
        name: 'Samples',
        color: 0xFF2196F3,
        type: MediaStreamType.audio,
        context: null,
      );

  @override
  Future<void> startCapture() async {
    // Open the device here.
  }

  @override
  Future<void> stopCapture() async {
    // Close it here.
  }

  void push(Float32List samples) => notifyHostData(samples, [samples.length]);
}

/// Runs the README's examples as written, so the published snippets keep
/// working. Keep the two in step.
void main() {
  test('build and run a stage', () async {
    // Double every value on the GPU, then clip to [-1, 1] on the CPU.
    final stage = StageBuilder('gain_and_clip')
        .op(
          'multiplyScalar',
          arguments: [2.0],
          inputKey: 'input',
          outputKey: 'gained',
        )
        .executeCPU(
          (inputs, shapes, params) async {
            final data = inputs['gained']! as Float32List;
            return {
              'clipped': Float32List.fromList([
                for (final v in data) v.clamp(-1.0, 1.0),
              ]),
            };
          },
          inputKey: 'gained',
          outputKey: 'clipped',
        )
        .build();

    final input = await Tensor.zeros([4]);
    await input.write(Float32List.fromList([0.25, 0.375, -0.75, 0.625]));

    final out = await stage.process({'input': input});
    expect(await out['output']!.getData(), equals([0.5, 0.75, -1.0, 1.0]));
  });

  test('a shader function step', () async {
    final stage = StageBuilder('scaled_square')
        .executeShaderFunction(
          'return gain * input_val * input_val;',
          functionParameters: {'gain': 0.5},
        )
        .build();

    final input = await Tensor.zeros([4]);
    await input.write(Float32List.fromList([1, 2, 3, 4]));

    final out = await stage.process({'input': input});
    expect(await out['output']!.getData(), equals([0.5, 2.0, 4.5, 8.0]));
  });

  test('run stages over a stream', () async {
    final pipeline = Pipeline(id: 'gain');
    pipeline.addStage(
      StageBuilder('half').op('multiplyScalar', arguments: [0.5]).build(),
    );
    final stream = SampleStream();
    pipeline.addStream(stream);

    await pipeline.start();
    expect(stream.isActive, isTrue);
    stream.push(Float32List.fromList([2, 4, 6, 8]));

    Tensor? latest;
    for (var i = 0; i < 250 && latest == null; i++) {
      await Future<void>.delayed(const Duration(milliseconds: 20));
      latest = pipeline.getStageOutput('half', stream.id)?['output'];
    }
    expect(latest, isNotNull, reason: 'the stage produced no output');
    expect(await latest!.getData(), equals([1.0, 2.0, 3.0, 4.0]));

    await pipeline.stop();
    expect(stream.isActive, isFalse);
    await pipeline.dispose();
  });
}
