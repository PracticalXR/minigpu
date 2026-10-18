# gpu_pipeline

A GPU-accelerated streaming pipeline for Dart and Flutter. Define directed graphs of processing stages that route [`Tensor`](https://pub.dev/packages/gpu_tensor) and host typed-data through WGSL compute shaders via [minigpu](https://pub.dev/packages/minigpu), with typed I/O ports, resource management, and real-time constraints.

Part of the [minigpu](https://github.com/PracticalXR/minigpu) package family.

## Features

- **Directed stage graph** — connect `PipelineStage` nodes via typed `InputPort`/`OutputPort` pairs; the pipeline routes data automatically
- **GPU + CPU stages** — `ShaderStageOperation` runs WGSL compute shaders on `Tensor` data; `CPUStageOperation` processes host `TypedData`; mix freely in the same graph
- **Pluggable CPU executors** — CPU stages delegate to a `StageExecutor`: `InlineExecutor` (default, always correct for stateful stages) or `PersistentIsolateExecutor` (long-lived worker isolate for stateless/heavy compute; GPU submission overlaps CPU work automatically)
- **Stage concurrency hints** — each stage carries a `StageConcurrency` flag (`ordered` / `asyncSafe`) so the scheduler knows whether frames must execute strictly in order or may overlap
- **Zero-allocation readback / upload** — `Tensor.getData({into:})` lets callers supply a pre-allocated `TypedData` buffer to avoid per-frame heap churn; the FFI layer pools its native scratch buffer and `NativeCallable` across calls
- **Multi-stream routing** — each stage declares a `StageStreamConfig` to receive all streams, a merged view, or only selected stream IDs
- **Resource-aware scheduling** — `ResourceRequirements`, `MemoryStrategy`, and `RealTimeConstraints` let stages express their resource needs; the pipeline honours them during execution
- **Dynamic stages** — `PipelineDynamicStage` can modify the graph topology at runtime
- **Asset management** — stages declare `StaticAsset` requirements; the built-in `AssetManager` loads and caches them before first execution
- **Event bus** — `PipelineEventNotifier` delivers lifecycle and error events to any observer

## Getting started

Add to `pubspec.yaml`:

```yaml
dependencies:
  gpu_pipeline: ^1.0.0
  gpu_tensor: ^1.3.0
  minigpu: ^1.3.0
```

Initialise minigpu once before creating pipelines:

```dart
import 'package:minigpu/minigpu.dart';

await Minigpu.initialize();
```

## Usage

### Define a stage

```dart
import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_tensor/gpu_tensor.dart';

class NormalizeStage extends PipelineStage {
  NormalizeStage() : super(stageId: 'normalize') {
    addInputPort(InputPort('input', formats: ['tensor']));
    addOutputPort(OutputPort('output', format: 'tensor'));
  }

  @override
  Future<PipelineEvent> process(Map<String, dynamic> inputs) async {
    final tensor = inputs['input'] as Tensor;
    // run a WGSL shader or transform tensors here
    return PipelineEvent.data({'output': tensor});
  }
}
```

### Build and run a pipeline

```dart
final pipeline = Pipeline();
pipeline.addStage(NormalizeStage());
pipeline.addStage(MyOutputStage());
pipeline.connect('normalize.output', 'output.input');

await pipeline.initialize();

final result = await pipeline.process({'normalize.input': myTensor});
```

### Shader stage

Use `ShaderStageOperation` to dispatch a WGSL compute shader directly:

```dart
final op = ShaderStageOperation(
  shader: myWgslSource,
  workgroupSize: (8, 8, 1),
  bindingLayout: [...],
);
```

## Architecture

```
MediaStream  ──►  Stage A  ──►  Stage B  ──►  Stage C  ──►  output
                  (WGSL)        (CPU)          (WGSL)
```

Each stage receives a `Map<String, Tensor>` of named inputs and returns a `Map<String, Tensor>` of outputs. The `Pipeline` scheduler resolves the connection graph, dispatches stages in dependency order, and propagates errors via the event bus.

### CPU↔GPU boundary

CPU stages (`CPUStageOperation`) read `TypedData` from GPU tensors, run Dart logic, then upload results back. The boundary has three performance layers:

1. **FFI scratch pooling** — `FfiBuffer` reuses its native scratch memory and `NativeCallable` across `read`/`write` calls, eliminating per-frame `malloc` and callback registration overhead.
2. **Buffer reuse** — `Tensor.getData({T? into})` accepts a caller-supplied `TypedData` to avoid allocating a new buffer every frame.
3. **Isolate offload** — stateless CPU work can run on a `PersistentIsolateExecutor`, freeing the GPU-submission isolate to queue the next frame while the previous frame's CPU work is still in flight.

### Concurrency model

```dart
// Stateful stage: must observe frame order (closed-loop encoder, etc.)
myStage.stageConcurrency = StageConcurrency.ordered;   // default

// Stateless stage: frames may overlap
myStage.stageConcurrency = StageConcurrency.asyncSafe;
```

`ordered` is the safe default. Only mark a stage `asyncSafe` if its processor holds no cross-frame state.

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
