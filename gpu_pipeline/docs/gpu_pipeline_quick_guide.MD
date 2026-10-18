# MiniGPU Pipeline - Complete Quick Reference v2

## Overview

GPU-accelerated media processing framework for Dart/Flutter. Build real-time audio/video pipelines using specialized stages, a fluent builder API, GPU compute shaders, and CPU processing stages with explicit CPU↔GPU data transfers.

## Core Concepts

### Pipeline

```dart
Pipeline(
  id: 'my_pipeline',
  enableBenchmarking: true,
  debugEnabled: false,
  queueConfig: QueueConfig(strategy: QueueStrategy.latestOnly),
)
  .addStage(stage1)
  .addStage(stage2)
  .addMergeStage(mergeStage)  // Special handling for stream merging
  .addStream(audioStream)
  .start();  // Returns Future<void>
```

**Key Methods:**

- `pipeline.isRunning` - Check if pipeline is active
- `pipeline.isProcessing` - Check if currently processing data
- `pipeline.stop()` - Async stop (returns Future<void>)
- `pipeline.dispose()` - Clean up resources
- `pipeline.getStageOutput('stageId', 'streamId')` - Get output for specific stream
- `pipeline.getStage('stageId')` - Retrieve stage instance
- `pipeline.debugPrintAllOutputs()` - Debug tensor states
- `pipeline.getBenchmarkStats()` - Performance metrics

### Event System

```dart
// Direct event listener pattern
final subscription = pipeline.eventNotifier.eventStream.listen((event) {
  if (event is PipelineRunCompleteEvent) {
    print('Pipeline completed one full processing cycle');
  } else if (event is PipelineErrorEvent) {
    print('Error: ${event.error}');
  } else if (event is PipelineBenchmarkEvent) {
    print('Performance data available');
  }
});

// Don't forget to cancel subscription
await subscription.cancel();
```

**Event Types:**

- `PipelineRunCompleteEvent` - Full processing cycle complete
- `PipelineErrorEvent` - Pipeline error occurred
- `PipelineBenchmarkEvent` - Performance metrics
- `StageStartEvent` / `StageCompleteEvent`
- `StreamStartEvent` / `StreamStopEvent`
- `TensorUpdateEvent`
- `ParameterChangeEvent` - Parameter value changed

### Stage Types

#### GPU Stages (Specialized Audio/Video)

##### WindowStage

```dart
WindowStage(
  stageId: 'window',
  windowType: WindowType.hann,  // Also: .hamming, .blackman, .rectangular
)
```

##### FFTStage

```dart
FFTStage(stageId: 'fft')
```

##### MagnitudeStage

```dart
MagnitudeStage(
  stageId: 'magnitude',
  useLogScale: true,
  normalizeOutput: true,
  minDb: -80.0,      // For log scale
  maxDb: 0.0,        // For log scale
)
```

##### SmoothingStage

```dart
SmoothingStage(
  stageId: 'smoothing',
  smoothingFactor: 0.25,  // 0.0 to 1.0
  smoothingType: SmoothingType.exponential,  // Also: .linear
)
```

##### HistoryStage

```dart
HistoryStage(
  stageId: 'history',
  maxTimeSlices: 120,
  historyType: HistoryType.spectrogram,  // Also: .waveform, .generic
)
```

##### StreamMergeStage

```dart
StreamMergeStage(
  stageId: 'merge',
  strategy: MergeStrategy.maximum,  // Also: .average, .sum, .minimum
)
```

##### SpectrogramVisualizationStage

```dart
SpectrogramVisualizationStage(
  stageId: 'visualization',
  minFreq: 20.0,
  maxFreq: 20000.0,
  sampleRate: 48000.0,
)
```

#### CPU Stages

##### CPUStage (Custom Dart Processing)

```dart
CPUStage(
  stageId: 'cpu_processor',
  processor: (inputs, parameters) async {
    // Custom Dart processing logic - runs on CPU
    final input = inputs['audio'] as Tensor;
    final gain = parameters['gain'] as double;
    
    // Automatically reads data from GPU tensor to CPU
    final data = await input.getData() as Float32List;
    
    // Process on CPU
    for (int i = 0; i < data.length; i++) {
      data[i] *= gain;
    }
    
    // Create output tensor - automatically uploads to GPU if next stage is GPU
    final output = await Tensor.create(input.shape);
    await output.write(data);
    
    return {'processed': output};
  },
)
```

##### DynamicCPUStage (Runtime Dart Code)

```dart
DynamicCPUStage(
  stageId: 'dynamic_cpu',
  dartCode: '''
    Future<Map<String, Tensor>> process(Map<String, Tensor> inputs, Map<String, dynamic> params) async {
      final audio = inputs['audio']!;
      final threshold = params['threshold'] as double;
      
      // Pipeline automatically handles GPU→CPU data transfer
      final data = await audio.getData() as Float32List;
      
      // Apply threshold on CPU
      for (int i = 0; i < data.length; i++) {
        if (data[i].abs() < threshold) {
          data[i] = 0.0;
        }
      }
      
      final output = await Tensor.create(audio.shape);
      await output.write(data);
      return {'thresholded': output};
    }
  ''',
  isolateEnabled: true,  // Run in separate isolate for better performance
)
```

##### PrebuiltCPUStage (Common CPU Operations)

```dart
PrebuiltCPUStage(
  stageId: 'prebuilt_cpu',
  operation: CPUOperation.normalize,  // Built-in CPU operations
  parameters: {
    'min': -1.0,
    'max': 1.0,
  },
)
```

**Available CPU Operations:**
- `CPUOperation.normalize` - Normalize data to range
- `CPUOperation.threshold` - Apply threshold
- `CPUOperation.convolve` - 1D convolution
- `CPUOperation.correlate` - Cross-correlation
- `CPUOperation.interpolate` - Data interpolation
- `CPUOperation.filter` - Digital filtering
- `CPUOperation.statistics` - Statistical analysis

### CPU↔GPU Data Flow

When CPU stages are mixed with GPU stages, the pipeline automatically handles data transfers:

```dart
final pipeline = Pipeline(id: 'mixed_pipeline')
  .addStage(WindowStage(stageId: 'gpu_window'))     // GPU
  .addStage(CPUStage(                               // CPU - auto GPU→CPU transfer
    stageId: 'cpu_analysis',
    processor: analyzeCPU,
  ))
  .addStage(FFTStage(stageId: 'gpu_fft'));          // GPU - auto CPU→GPU transfer
```

### StageBuilder API

**Support for both CPU and GPU stages:**

```dart
StageBuilder('stage_id')
  // Input Ports - use name: parameter
  .withAudioInput(name: 'audio', description: 'Audio samples')
  .withVideoInput(name: 'video', minWidth: 640, maxWidth: 1920)
  .withFlexibleInput(name: 'flexible', minDimensions: 2, maxDimensions: 4)
  .withExactInput(name: 'exact', shape: [256, 256, 3])
  .withStaticInput(name: 'weights', assetPath: 'weights.bin')
  .withGeneratedInput(
    name: 'noise',
    shape: [1024],
    type: 'gaussian_noise',
    description: 'Generated noise input',
    generatorParams: {'mean': 0.0, 'stddev': 1.0, 'seed': 12345},
  )
  .withCustomInput(
    name: 'custom',
    validator: (tensor) {
      return tensor != null && tensor.shape.isNotEmpty
          ? ValidationResult.valid()
          : ValidationResult.invalid('Invalid tensor');
    },
  )
  
  // Output Ports - use name: parameter
  .withDynamicOutput(
    name: 'dynamic',
    shapeCalculator: (inputs, params) => [512, 512],
  )
  .withFixedOutput(name: 'fixed', shape: [1024])
  .withTextureOutput(name: 'texture', width: 512, height: 512, channels: 4)
  .withHistoryOutput(name: 'history', timeSlices: 120)
  .withParameterOutput(
    name: 'parameter',
    dependencies: ['gain', 'threshold'],
    calculator: (inputs, params) => [100],
  )
  
  // Parameters - use Parameter.stageType() constructors
  .withParameter(
    Parameter.stageFloat(
      name: 'gain',
      defaultValue: 1.0,
      min: 0.0,
      max: 2.0,
      description: 'Audio gain',
    ),
  )
  .withParameter(
    Parameter.stageInt(
      name: 'iterations',
      defaultValue: 10,
      min: 1,
      max: 100,
      description: 'Number of iterations',
    ),
  )
  .withParameter(
    Parameter.stageBool(
      name: 'enabled',
      defaultValue: true,
      description: 'Enable processing',
    ),
  )
  .withParameter(
    Parameter.stageVec3(
      name: 'position',
      defaultValue: [0, 0, 0],
      description: 'Position vector',
    ),
  )
  .withParameter(
    Parameter.stageString(
      name: 'mode',
      defaultValue: 'fast',
      allowedValues: ['fast', 'quality'],
      description: 'Processing mode',
    ),
  )
  
  // GPU Operations - Built-in operations
  .op('fft')
  .op('multiplyScalar', arguments: [2.0])
  .op('abs')
  .op('sqrt')
  .op('power', arguments: [2.0])
  .op('clamp', arguments: [0.0, 1.0])
  .op('normalize')
  
  // GPU Shader Execution
  .executeShader('''
    @group(0) @binding(0) var<storage, read_write> input: array<f32>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let index = gid.x;
      if (index >= arrayLength(&input)) { return; }
      input[index] = input[index] * 2.0;
    }
  ''')
  
  // CPU Processing
  .executeCPU((inputs, parameters) async {
    // Custom Dart processing
    final audio = inputs['audio'] as Tensor;
    final gain = parameters['gain'] as double;
    
    final data = await audio.getData() as Float32List;
    for (int i = 0; i < data.length; i++) {
      data[i] *= gain;
    }
    
    final output = await Tensor.create(audio.shape);
    await output.write(data);
    return {'output': output};
  })
  
  // Dynamic CPU (Runtime Dart Code)
  .executeDynamicCPU('''
    Future<Map<String, Tensor>> process(Map<String, Tensor> inputs, Map<String, dynamic> params) async {
      // Runtime-evaluated Dart code
      final input = inputs['audio']!;
      final factor = params['factor'] as double;
      
      final data = await input.getData() as Float32List;
      
      // Custom processing logic
      for (int i = 0; i < data.length; i++) {
        data[i] = data[i] * factor + math.sin(i * 0.01);
      }
      
      final output = await Tensor.create(input.shape);
      await output.write(data);
      return {'processed': output};
    }
  ''', isolateEnabled: true)
  
  // Execution Target (explicit)
  .setExecutionTarget(ExecutionTarget.gpu)    // Force GPU execution
  // .setExecutionTarget(ExecutionTarget.cpu)  // Force CPU execution
  
  // Asset Management
  .withWeights(name: 'model_weights', path: 'assets/weights.bin')
  .withConfig(name: 'model_config', path: 'assets/config.json')
  .withTexture(name: 'lookup_table', path: 'assets/lut.png')
  .withLookupTable(name: 'gamma_table', path: 'assets/gamma.lut')
  
  .build();
```

### Pipeline Serialization

**Save and load pipeline configurations:**

```dart
// Create pipeline configuration
final pipelineConfig = PipelineConfig(
  id: 'audio_visualizer',
  enableBenchmarking: true,
  stages: [
    StageConfig(
      stageId: 'window',
      type: 'WindowStage',
      parameters: {
        'windowType': 'hann',
      },
    ),
    StageConfig(
      stageId: 'fft',
      type: 'FFTStage',
      parameters: {},
    ),
    StageConfig(
      stageId: 'custom_cpu',
      type: 'CPUStage',
      executionTarget: ExecutionTarget.cpu,
      dartCode: '''
        Future<Map<String, Tensor>> process(inputs, params) async {
          // Custom processing
          return {'output': processedTensor};
        }
      ''',
      parameters: {
        'gain': 1.0,
        'threshold': 0.1,
      },
    ),
  ],
  connections: [
    StageConnection(from: 'window', to: 'fft'),
    StageConnection(from: 'fft', to: 'custom_cpu'),
  ],
);

// Serialize to JSON
final json = pipelineConfig.toJson();

// Save to file or send to frontend
await File('pipeline_config.json').writeAsString(jsonEncode(json));

// Load from JSON
final loadedJson = jsonDecode(await File('pipeline_config.json').readAsString());
final loadedConfig = PipelineConfig.fromJson(loadedJson);

// Build pipeline from config
final pipeline = await PipelineBuilder.fromConfig(loadedConfig);
```

## Serialization (v1 Draft)

### Full Network Example
```json
{
  "schema_version": 1,
  "pipelines": [
    {
      "type": "pipeline",
      "id": "audio_visualizer",
      "queue": { "strategy": "latest_only" },
      "enable_benchmarking": true,
      "stages": [
        {
          "type": "stage",
          "stage_id": "window",
            "stage_kind": "WindowStage",
          "parameters": { "window_type": "hann" }
        },
        {
          "type": "stage",
          "stage_id": "fft",
          "stage_kind": "FFTStage",
          "parameters": {}
        },
        {
          "type": "stage",
          "stage_id": "custom_cpu",
          "stage_kind": "DynamicCPUStage",
          "execution_target": "cpu",
          "parameters": { "gain": 1.0, "threshold": 0.1 },
          "operations": [
            {
              "type": "operation",
              "op_kind": "dynamic_cpu",
              "input_keys": ["windowed"],
              "output_keys": ["processed"],
              "dart_code": "Future<Map<String,TypedData>> main(...) { /* user code */ }"
            }
          ]
        }
      ],
      "connections": [
        { "from_stage": "window", "to_stage": "fft" },
        { "from_stage": "fft", "to_stage": "custom_cpu" }
      ]
    }
  ],
  "assets": [
    {
      "type": "asset",
      "name": "model_weights",
      "asset_type": "weights",
      "source": { "source_type": "file", "path": "assets/weights.bin" }
    }
  ],
  "inter_pipeline_links": []
}
```

### Runtime Patch Example
```json
{
  "type": "patch",
  "pipeline_id": "audio_visualizer",
  "ops": [
    { "op": "set_param", "stage_id": "custom_cpu", "name": "gain", "value": 1.25 }
  ]
}
```

### Error / Validation Result
```json
{
  "ok": false,
  "errors": [
    { "code": "UNKNOWN_OPERATION", "message": "op_kind=foo_op", "path": "pipelines[0].stages[1].operations[0]" }
  ],
  "warnings": []
}
```

### Loader Flow
1. Parse JSON → check `schema_version`.
2. Validate structural schema & registry entries.
3. Build stage graph (array order for execution).
4. Register dynamic code & compile (collect compile errors).
5. Emit validation summary event; proceed only if `errors` empty (or allow run with warnings).

### Supported op_kind values
- tensor_op
- cpu
- dynamic_cpu
- shader
- merge
- split

### Notes
- Initial tensors not serialized; produced from streams or assets.
- All `parameters` included for now (simpler diff UI).
- WGSL / Dart code stored inline (future: external references + hash).

## Inter-Pipeline Communication

**Pipelines can send/receive data to other pipelines:**

```dart
// Source pipeline (produces data)
final sourcePipeline = Pipeline(id: 'audio_source')
  .addStage(WindowStage(stageId: 'window'))
  .addStage(FFTStage(stageId: 'fft'))
  .addOutputSink(PipelineOutputSink(
    sinkId: 'fft_output',
    targetPipelineId: 'audio_effects',
    targetInputId: 'spectrum_input',
  ));

// Consumer pipeline (receives data)
final consumerPipeline = Pipeline(id: 'audio_effects')
  .addInputSource(PipelineInputSource(
    sourceId: 'spectrum_input',
    fromPipelineId: 'audio_source',
    fromOutputId: 'fft_output',
  ))
  .addStage(MagnitudeStage(stageId: 'magnitude'))
  .addStage(SmoothingStage(stageId: 'smoothing'));

// Pipeline orchestrator manages multiple pipelines
final orchestrator = PipelineOrchestrator()
  .addPipeline(sourcePipeline)
  .addPipeline(consumerPipeline)
  .establishConnection(
    from: 'audio_source.fft_output',
    to: 'audio_effects.spectrum_input',
  );

await orchestrator.startAll();
```

### Working with Tensors

```dart
// Create and populate tensor
final tensor = await Tensor.create([10]);
final testData = Float32List.fromList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
await tensor.write(testData);

// Get tensor data
final pixels = await tensor.getData() as Float32List;

// Check validity
if (tensor.buffer.isValid) {
  // Safe to use tensor
}

// Tensor shape
final shape = tensor.shape; // [height, width, channels]

// Don't forget to clean up
tensor.destroy();
```

### Queue Strategies

```dart
QueueConfig(strategy: QueueStrategy.latestOnly)     // Keep only newest
QueueConfig(strategy: QueueStrategy.latestN, batchSize: 5)  // Keep last N
QueueConfig(strategy: QueueStrategy.processAll)     // Process all (unlimited)
QueueConfig(strategy: QueueStrategy.dropOnOverrun)  // Drop if full
```

### Runtime Stage Control

```dart
// Async parameter updates work for both CPU and GPU stages
await stage.setPropertyValue('gain', 2.0);
await stage.setPropertyValue('threshold', 0.8);

// Synchronous parameter reads
final gain = stage.getPropertyValue<double>('gain');
final enabled = stage.getPropertyValue<bool>('enabled');
final mode = stage.getPropertyValue<String>('mode');
final position = stage.getPropertyValue<List<double>>('position');

// For enum parameters
final enumMode = stage.getPropertyValue<ProcessingMode>('mode');

// Validation
final isValid = stage.isValidParameterValue('gain', 2.5);

// Check execution target
final isGPUStage = stage.executionTarget == ExecutionTarget.gpu;
final isCPUStage = stage.executionTarget == ExecutionTarget.cpu;
```

### Parameter Types

- `Parameter.stageFloat()` - Floating point with min/max
- `Parameter.stageInt()` - Integer with min/max
- `Parameter.stageBool()` - Boolean parameter
- `Parameter.stageVec3()` - 3-component vector
- `Parameter.stageString()` - String with allowedValues
- `Parameter.stageEnum<T>()` - Enum type with allValues

### Validation Results

```dart
// Custom input validators must return ValidationResult
validator: (tensor) {
  return tensor != null && tensor.shape.isNotEmpty
      ? ValidationResult.valid()
      : ValidationResult.invalid('Invalid tensor');
}
```

### Enum Registration

```dart
// Register enums before using them (in setUpAll)
enum ProcessingMode { fast, quality }

setUpAll(() async {
  EnumRegistry.register<ProcessingMode>(ProcessingMode.values);
});
```

## Performance

### Zero-allocation readback

Supply a pre-allocated `TypedData` buffer to `getData` to skip per-frame heap allocation:

```dart
// Allocate once, reuse every frame
final scratch = Float32List(width * height);

// Every subsequent call writes into scratch instead of allocating
final data = await tensor.getData(into: scratch) as Float32List;
```

This is especially useful in tight render loops where GC pressure is a concern.

### Pluggable CPU executors

`CPUStageOperation` delegates its host-side work to a `StageExecutor`. Pick the backend that matches your stage's characteristics:

| Executor | When to use |
|---|---|
| `InlineExecutor` (default) | Any stage; **required** for stateful stages (closed-loop encoders, accumulators) |
| `PersistentIsolateExecutor` | Stateless, CPU-heavy stages that can run independently of frame order |

```dart
// Stateless filter — offload to worker isolate
// Processor MUST be a top-level or static function (Dart isolate requirement)
CPUStageOperation(
  operationId: 'denoise',
  processor: denoiseTopLevel,        // top-level / static function
  executor: PersistentIsolateExecutor(denoiseTopLevel),
)

// Stateful encoder — must stay inline (closed-loop state in closure)
CPUStageOperation(
  operationId: 'av1_pack',
  processor: _packFrameClosure,      // captures state — inline only
  // no executor: argument → defaults to InlineExecutor
)
```

The worker isolate is spawned lazily on the first call and kept alive for the lifetime of the stage, so you pay the isolate spawn cost only once.

### Stage concurrency hints

Each `PipelineStage` carries a `stageConcurrency` property that tells the scheduler whether frames must be serialised:

```dart
// Default: frame N completes before frame N+1 starts.
// Use for any stage with cross-frame state, or when output order matters.
stage.stageConcurrency = StageConcurrency.ordered;

// The stage is a pure per-frame function; frames may overlap on the
// worker isolate while the GPU-submit isolate queues the next frame.
stage.stageConcurrency = StageConcurrency.asyncSafe;
```

Combining `asyncSafe` with `PersistentIsolateExecutor` gives you automatic GPU/CPU overlap at zero cost to the pipeline engine.

## Platform Support

- **Desktop**: Vulkan, D3D12, Metal + Multi-core CPU
- **Mobile**: Metal, Vulkan + ARM CPU optimization  
- **Web**: WebGPU + Web Workers for CPU stages

## API Parameter Conventions

1. **Named Parameters Required**: 
   - All port methods (input/output): `name:`, `shape:`, `description:`, etc.
   - Asset methods: `name:`, `path:`
   - Parameter constructors: `name:`, `defaultValue:`, `description:`

2. **Positional Parameters**: 
   - Operations: `.op('normalize')`, `.op('multiplyScalar', arguments: [2.0])`
   - Shader execution: `.executeShader(shaderCode)` or `.executeShader(shaderCode, shaderParameters: {...})`
   - CPU execution: `.executeCPU(processorFunction)`

3. **Execution Targets**: 
   - `.setExecutionTarget(ExecutionTarget.gpu)` - Force GPU execution
   - `.setExecutionTarget(ExecutionTarget.cpu)` - Force CPU execution
   - Automatic GPU↔CPU data transfers when stages have different targets

4. **Pipeline Serialization**: JSON-based configuration for dynamic pipeline creation

5. **Inter-Pipeline Communication**: Pipelines can send/receive data to other pipelines

## Shape & Dependency Expressions (Quick Reference)

| Use Case | JSON Snippet |
|----------|--------------|
| Fixed shape | `"shape_expr": { "op": "fixed", "shape": [512] }` |
| Same as input | `"shape_expr": { "op": "same_as", "input": "input_a" }` |
| Concat | `{ "op": "concat_along", "axis": 0, "inputs": ["a","b"] }` |
| Broadcast pair | `{ "op": "broadcast_pair", "a": "lhs", "b": "rhs" }` |
| Scale dim by param | `{ "op": "scale_dim", "input": "img", "axis": 1, "factor": "param:scale_y" }` |
| Stack | `{ "op": "stack", "inputs": ["f1","f2","f3"], "axis": 0 }` |
| Reduce axes | `{ "op": "reduce_axes", "input": "feat", "axes": [1,2] }` |

Argument references:
```
"arguments": {
  "gain": "param:gain",
  "lhs_tensor": "tensor:input_a",
  "rhs_shape": "shape:input_b"
}
```

Validation fails early if any reference cannot be resolved.