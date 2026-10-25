# gpu_pipeline

A GPU-accelerated streaming pipeline for Dart and Flutter. A stage chains WGSL compute shaders, built-in tensor ops and Dart functions; a pipeline runs its stages on every frame a media stream emits. Data moves as [`Tensor`](https://pub.dev/packages/gpu_tensor)s on [minigpu](https://pub.dev/packages/minigpu), or as host typed data between CPU steps.

Part of the [minigpu](https://github.com/PracticalXR/minigpu) package family.

## Features

- **GPU and CPU steps in one stage**: `StageBuilder` chains built-in tensor ops (`op('multiplyScalar', ...)`), WGSL compute shaders (`executeShader`, `executeShaderFunction`) and Dart functions over host `TypedData` (`executeCPU`) into one `DynamicStage`.
- **Data stays where it was written**: inside a stage, a value written by a CPU step stays on the host until a GPU step reads it, and the reverse, so back-to-back CPU steps never round-trip through the GPU.
- **Streaming**: a `Pipeline` runs its stages, in the order they were added, on every frame a `MediaStream` emits, either a GPU `Tensor` or a host `Float32List`. `QueueConfig` sets how frames queue per stream.
- **Multi-stream routing**: `StageStreamConfig` decides whether a stage runs once per stream, on all streams merged (`addMergeStage`), or only on selected streams (`addSelectiveStage`).
- **Pluggable CPU executors**: a `CPUStageOperation` runs its function through a `StageExecutor`. `InlineExecutor` is the default, and the only choice for a function that keeps state between frames. `PersistentIsolateExecutor` runs a stateless top-level function on a long-lived worker isolate, off the isolate that submits GPU work. It is native-only and throws on the web.
- **Assets and parameters**: stages declare `StaticAsset`s (weights, textures, lookup tables), which the pipeline's `AssetManager` preloads and caches at `start()`, and typed, range-checked `Parameter`s.
- **Serialization**: `PipelineLoader().loadFromJson(...)` builds the pipelines a JSON network config describes, and `PipelineSerializer` validates and re-encodes one.
- **Events**: `addEventListener` delivers stage, operation, stream and pipeline events, including errors.

The resource and scheduling hints (`ResourceRequirements`, `MemoryStrategy`, `RealTimeConstraints` and `PipelineStage.stageConcurrency`) are deprecated: a stage stores them, but the pipeline reads none of them. So is `Pipeline.resourceGovernor`, a placeholder that reports invented devices. All of them will be removed in 2.0.

## Getting started

Add to `pubspec.yaml`:

```yaml
dependencies:
  gpu_pipeline: ^1.9.3
  gpu_tensor: ^1.9.0
  minigpu: ^1.9.0
```

The first tensor you create initializes the default minigpu context. To choose when that happens, initialize it yourself:

```dart
import 'package:minigpu/minigpu.dart';

await Minigpu().init();
```

## Usage

### Build and run a stage

Each step reads the key named by `inputKey` and writes the key named by `outputKey`; leave them out and each step reads the previous step's result. `process` returns the last step's result under `'output'`.

```dart
import 'dart:typed_data';

import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

// Double every value on the GPU, then clip to [-1, 1] on the CPU.
final stage = StageBuilder('gain_and_clip')
    .op('multiplyScalar', arguments: [2.0], inputKey: 'input', outputKey: 'gained')
    .executeCPU(
      (inputs, shapes, params) async {
        final data = inputs['gained']! as Float32List;
        return {
          'clipped': Float32List.fromList([for (final v in data) v.clamp(-1.0, 1.0)]),
        };
      },
      inputKey: 'gained',
      outputKey: 'clipped',
    )
    .build();

final input = await Tensor.zeros([4]);
await input.write(Float32List.fromList([0.25, 0.375, -0.75, 0.625]));

final out = await stage.process({'input': input});
print(await out['output']!.getData()); // [0.5, 0.75, -1.0, 1.0]
```

### Run stages over a stream

A `MediaStream` subclass emits frames with `notifyTensorUpdate` (a GPU tensor) or `notifyHostData` (host samples and a shape). Put your capture logic in `startCapture` and `stopCapture`; `start` and `stop` call each once and keep `isActive` up to date.

```dart
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

final pipeline = Pipeline(id: 'gain');
pipeline.addStage(
  StageBuilder('half').op('multiplyScalar', arguments: [0.5]).build(),
);
final stream = SampleStream();
pipeline.addStream(stream);

pipeline.addEventListener(PipelineEventType.stageError, (event) {
  print('stage failed: ${(event as StageErrorEvent).error}');
});

await pipeline.start();
stream.push(Float32List(480));

// Later, for example once per rendered frame: the latest result the stage
// produced for this stream.
final latest = pipeline.getStageOutput('half', stream.id)?['output'];

await pipeline.stop();
await pipeline.dispose();
```

### Shader steps

`executeShaderFunction` wraps a WGSL function body in a compute shader that runs once per element. The body sees `index: u32` and `input_val: f32` and returns the element's new value; each entry of `functionParameters` becomes an `f32` constant:

```dart
final stage = StageBuilder('scaled_square')
    .executeShaderFunction(
      'return gain * input_val * input_val;',
      functionParameters: {'gain': 0.5},
    )
    .build();
```

For a full shader with several bindings, use `executeShader`.

### Guides

- [Pipelines from JSON](doc/serialization.md): the network format, operation kinds, how stages pass tensors, links between pipelines, validation and live patches.
- [Assets](doc/assets.md): assets in a JSON network and in stages, reading them, per-call assets and lifetimes.
- [CPU steps](doc/cpu_steps.md): Dart steps, worker isolates and expression steps.

Every code example in the guides runs as a test in `test/doc_examples_test.dart`.

## Architecture

```text
MediaStream  ──►  Stage 1  ──►  Stage 2  ──►  Stage 3  ──►  getStageOutput()
                  (GPU ops)     (CPU)         (WGSL)
```

A pipeline runs its stages in the order they were added. Each stage receives the previous stage's outputs as a `Map<String, Tensor>` and returns its own. Errors are reported through `addEventListener`.

Inside a stage, each value is a `PipelineTensor` holding a host copy, a GPU copy, or both. A CPU step reads and writes host data, a GPU step reads and writes GPU tensors, and data crosses between them only when the next step needs the other side. For readbacks in your own code, `Tensor.getData(into:)` from gpu_tensor fills a buffer you already own instead of allocating one per frame.

## Platform support

| Platform | Status |
|----------|--------|
| Windows  | ✅     |
| Linux    | ✅     |
| macOS    | ✅     |
| Android  | ✅     |
| iOS      | ✅     |
| Web      | ✅ (dart2wasm / dart2js) |

## Additional information

- Source: [github.com/PracticalXR/minigpu](https://github.com/PracticalXR/minigpu)
- Issues: [github.com/PracticalXR/minigpu/issues](https://github.com/PracticalXR/minigpu/issues)
- Related packages: [minigpu](https://pub.dev/packages/minigpu), [gpu_tensor](https://pub.dev/packages/gpu_tensor)
