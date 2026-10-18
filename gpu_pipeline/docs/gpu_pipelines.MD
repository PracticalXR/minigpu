# GPU Pipeline Examples and Use Cases

This document provides comprehensive examples of creating GPU pipelines for different scenarios, including all features from the quick guide: CPU stages, serialization, inter-pipeline communication, and dynamic pipeline creation.

## Table of Contents

1. Audio Processing Pipeline
2. Video Computer Vision Pipeline
3. CPU-GPU Mixed Processing
4. Dynamic Pipeline Creation
5. Pipeline Serialization
6. Inter-Pipeline Communication
7. Real-time Video Effects
8. Audio Analysis Pipeline
9. Screen Capture & Streaming
10. Multi-Stream Audio Mixer

---

## Audio Processing Pipeline

### Basic Audio Visualizer with All StageBuilder Features

````dart
import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

Future<void> createCompleteAudioVisualizerPipeline() async {
  await Minigpu().init();

  final pipeline = Pipeline(
    id: 'complete_audio_visualizer',
    enableBenchmarking: true,
    debugEnabled: false,
    queueConfig: QueueConfig(strategy: QueueStrategy.latestOnly),
  );

  // Stage 1: Input preprocessing with all input port types
  final inputStage = StageBuilder('input_stage')
      // Audio input port
      .withAudioInput(name: 'audio', description: 'Audio samples')
      
      // Flexible input for variable size data
      .withFlexibleInput(name: 'metadata', minDimensions: 1, maxDimensions: 3)
      
      // Static input for window coefficients
      .withStaticInput(name: 'window_coeffs', assetPath: 'window_hann_1024.bin')
      
      // Generated noise input for testing
      .withGeneratedInput(
        name: 'test_noise',
        shape: [1024],
        type: 'gaussian_noise',
        description: 'Test noise signal',
        generatorParams: {
          'mean': 0.0,
          'stddev': 0.1,
          'seed': 42,
        },
      )
      
      // Custom input with validation
      .withCustomInput(
        name: 'control_signal',
        validator: (tensor) {
          if (tensor == null) return ValidationResult.invalid('Tensor is null');
          if (tensor.shape.isEmpty) return ValidationResult.invalid('Empty shape');
          if (tensor.shape[0] != 1024) return ValidationResult.invalid('Must be 1024 samples');
          return ValidationResult.valid();
        },
      )
      
      // Parameters with all types
      .withParameter(
        Parameter.stageFloat(
          name: 'input_gain',
          defaultValue: 1.0,
          min: 0.0,
          max: 4.0,
          description: 'Input gain control',
        ),
      )
      .withParameter(
        Parameter.stageInt(
          name: 'window_size',
          defaultValue: 1024,
          min: 256,
          max: 4096,
          description: 'Window size for processing',
        ),
      )
      .withParameter(
        Parameter.stageBool(
          name: 'use_window',
          defaultValue: true,
          description: 'Apply windowing function',
        ),
      )
      .withParameter(
        Parameter.stageVec3(
          name: 'eq_gains',
          defaultValue: [1.0, 1.0, 1.0],
          description: 'EQ gains for low/mid/high',
        ),
      )
      .withParameter(
        Parameter.stageString(
          name: 'processing_mode',
          defaultValue: 'realtime',
          allowedValues: ['realtime', 'quality', 'fast'],
          description: 'Processing mode',
        ),
      )
      
      // GPU operations
      .op('multiplyScalar', arguments: ['param:input_gain'])
      .op('highpass', arguments: [80.0])
      
      .build();

  // Stage 2: Windowing with output ports
  final windowStage = StageBuilder('windowing')
      // Fixed output
      .withFixedOutput(name: 'windowed', shape: [1024])
      
      // Dynamic output based on input
      .withDynamicOutput(
        name: 'window_stats',
        shapeCalculator: (inputs, params) {
          final windowSize = params['window_size'] as int;
          return [windowSize ~/ 4]; // Statistics are 1/4 the window size
        },
      )
      
      // History output for tracking
      .withHistoryOutput(name: 'window_history', timeSlices: 60)
      
      // Parameter output for analysis
      .withParameterOutput(
        name: 'window_energy',
        dependencies: ['input_gain', 'use_window'],
        calculator: (inputs, params) => [1], // Single energy value
      )
      
      .withParameter(
        Parameter.stageFloat(
          name: 'window_overlap',
          defaultValue: 0.5,
          min: 0.0,
          max: 0.9,
          description: 'Window overlap factor',
        ),
      )
      
      // Apply windowing
      .op('applyWindow', arguments: ['hann'])
      .build();

  // Stage 3: FFT Transform
  final fftStage = FFTStage(stageId: 'fft');

  // Stage 4: Magnitude calculation
  final magnitudeStage = MagnitudeStage(
    stageId: 'magnitude',
    useLogScale: true,
    normalizeOutput: true,
    minDb: -80.0,
    maxDb: 0.0,
  );

  // Stage 5: Custom GPU shader stage
  final customShaderStage = StageBuilder('custom_gpu_shader')
      .withParameter(
        Parameter.stageFloat(
          name: 'spectral_gain',
          defaultValue: 2.0,
          min: 0.1,
          max: 10.0,
          description: 'Spectral enhancement gain',
        ),
      )
      .executeShader('''
        @group(0) @binding(0) var<storage, read_write> spectrum: array<f32>;
        @group(1) @binding(0) var<uniform> params: Parameters;
        
        struct Parameters {
          spectral_gain: f32,
          frequency_bins: u32,
        }
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          let index = gid.x;
          if (index >= arrayLength(&spectrum)) { return; }
          
          // Apply spectral enhancement
          let freq_ratio = f32(index) / f32(params.frequency_bins);
          let enhancement = 1.0 + (params.spectral_gain - 1.0) * (1.0 - freq_ratio);
          spectrum[index] = spectrum[index] * enhancement;
        }
      ''')
      .build();

  // Stage 6: Smoothing
  final smoothingStage = SmoothingStage(
    stageId: 'smoothing',
    smoothingFactor: 0.25,
    smoothingType: SmoothingType.exponential,
  );

  // Stage 7: History for spectrogram
  final historyStage = HistoryStage(
    stageId: 'history',
    maxTimeSlices: 512,
    historyType: HistoryType.spectrogram,
  );

  // Stage 8: Visualization
  final visualizationStage = SpectrogramVisualizationStage(
    stageId: 'visualization',
    minFreq: 20.0,
    maxFreq: 20000.0,
    sampleRate: 48000.0,
  );

  // Add stages to pipeline
  pipeline.addStage(inputStage);
  pipeline.addStage(windowStage);
  pipeline.addStage(fftStage);
  pipeline.addStage(magnitudeStage);
  pipeline.addStage(customShaderStage);
  pipeline.addStage(smoothingStage);
  pipeline.addStage(historyStage);
  pipeline.addStage(visualizationStage);

  // Create audio stream
  final audioStream = AudioStream.fromDevice(
    await AudioManager.getDefaultInputDevice(),
    windowSize: 1024,
    sampleRate: 48000,
  );
  
  pipeline.addStream(audioStream);

  // Event handling
  final subscription = pipeline.eventNotifier.eventStream.listen((event) {
    if (event is PipelineRunCompleteEvent) {
      final visualOutput = pipeline.getStageOutput('visualization', audioStream.id);
      if (visualOutput != null) {
        displaySpectrogram(visualOutput['texture']!);
      }
    } else if (event is PipelineErrorEvent) {
      print('Pipeline error: ${event.error}');
    } else if (event is PipelineBenchmarkEvent) {
      print('Performance metrics available');
    }
  });

  await pipeline.start();
  print('Complete audio visualizer pipeline started!');
}
````

---

## CPU-GPU Mixed Processing

### Audio Analysis with CPU Post-Processing

````dart
Future<void> createMixedCPUGPUPipeline() async {
  final pipeline = Pipeline(id: 'mixed_cpu_gpu_audio');

  // Stage 1: GPU preprocessing
  final gpuPreprocess = StageBuilder('gpu_preprocess')
      .setExecutionTarget(ExecutionTarget.gpu)
      .withParameter(
        Parameter.stageFloat(
          name: 'gain',
          defaultValue: 1.0,
          min: 0.0,
          max: 4.0,
          description: 'Input gain',
        ),
      )
      .op('multiplyScalar', arguments: ['param:gain'])
      .op('highpass', arguments: [80.0])
      .build();

  // Stage 2: GPU FFT
  final gpuFFT = FFTStage(stageId: 'gpu_fft');

  // Stage 3: GPU magnitude
  final gpuMagnitude = MagnitudeStage(
    stageId: 'gpu_magnitude',
    useLogScale: true,
  );

  // Stage 4: CPU analysis (automatic GPU→CPU transfer)
  final cpuAnalysis = CPUStage(
    stageId: 'cpu_analysis',
    processor: (inputs, parameters) async {
      // Input tensor automatically transferred from GPU to CPU
      final spectrum = inputs['input'] as Tensor;
      final threshold = parameters['analysis_threshold'] as double;
      
      final spectrumData = await spectrum.getData() as Float32List;
      
      // Perform complex analysis on CPU
      final peaks = findSpectralPeaks(spectrumData, threshold);
      final harmonics = analyzeHarmonics(spectrumData, peaks);
      final centroid = calculateSpectralCentroid(spectrumData);
      final rolloff = calculateSpectralRolloff(spectrumData, 0.85);
      
      // Create analysis results tensor
      final analysisResults = Float32List.fromList([
        centroid,
        rolloff,
        peaks.length.toDouble(),
        harmonics.length.toDouble(),
        ...peaks.take(10), // Top 10 peak frequencies
      ]);
      
      final outputTensor = await Tensor.create([analysisResults.length]);
      await outputTensor.write(analysisResults);
      
      return {
        'analysis': outputTensor,
        'peak_count': await createScalarTensor(peaks.length.toDouble()),
      };
    },
  );

  // Add CPU analysis parameters
  cpuAnalysis.addParameter(
    Parameter.stageFloat(
      name: 'analysis_threshold',
      defaultValue: 0.1,
      min: 0.01,
      max: 1.0,
      description: 'Peak detection threshold',
    ),
  );

  // Stage 5: GPU visualization (automatic CPU→GPU transfer)
  final gpuVisualization = StageBuilder('gpu_visualization')
      .setExecutionTarget(ExecutionTarget.gpu)
      .withTextureOutput(name: 'analysis_display', width: 512, height: 256, channels: 4)
      .executeShader('''
        @group(0) @binding(0) var<storage, read> analysis_data: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output_texture: array<vec4<f32>>;
        
        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
          let x = gid.x;
          let y = gid.y;
          if (x >= 512u || y >= 256u) { return; }
          
          let pixel_index = y * 512u + x;
          
          // Visualize analysis results
          let centroid = analysis_data[0];
          let rolloff = analysis_data[1];
          let peak_count = analysis_data[2];
          
          let freq_pos = f32(x) / 512.0;
          let time_pos = f32(y) / 256.0;
          
          var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
          
          // Draw centroid line
          if (abs(freq_pos - centroid / 20000.0) < 0.01) {
            color.r = 1.0;
          }
          
          // Draw rolloff line
          if (abs(freq_pos - rolloff / 20000.0) < 0.01) {
            color.g = 1.0;
          }
          
          // Peak count intensity
          color.b = peak_count / 20.0;
          
          output_texture[pixel_index] = color;
        }
      ''')
      .build();

  // Stage 6: Dynamic CPU stage for real-time adjustments
  final dynamicCPU = DynamicCPUStage(
    stageId: 'dynamic_analysis',
    dartCode: '''
      Future<Map<String, Tensor>> process(Map<String, Tensor> inputs, Map<String, dynamic> params) async {
        final analysisData = inputs['analysis']!;
        final adaptiveMode = params['adaptive_mode'] as bool;
        
        final data = await analysisData.getData() as Float32List;
        
        if (adaptiveMode) {
          // Adaptive processing based on spectral content
          final centroid = data[0];
          final peakCount = data[2].toInt();
          
          if (peakCount > 5) {
            // High harmonic content - apply different processing
            for (int i = 4; i < data.length; i++) {
              data[i] *= 1.2; // Enhance peak frequencies
            }
          }
          
          if (centroid > 2000.0) {
            // Bright spectrum - reduce high frequencies
            for (int i = data.length ~/ 2; i < data.length; i++) {
              data[i] *= 0.8;
            }
          }
        }
        
        final output = await Tensor.create(data.length);
        await output.write(data);
        return {'enhanced_analysis': output};
      }
    ''',
    isolateEnabled: true,
  );

  // Add dynamic parameters
  dynamicCPU.addParameter(
    Parameter.stageBool(
      name: 'adaptive_mode',
      defaultValue: true,
      description: 'Enable adaptive processing',
    ),
  );

  // Prebuilt CPU stage for statistical analysis
  final statsCPU = PrebuiltCPUStage(
    stageId: 'statistics',
    operation: CPUOperation.statistics,
    parameters: {
      'include_variance': true,
      'include_skewness': true,
      'include_kurtosis': true,
    },
  );

  // Add all stages
  pipeline.addStage(gpuPreprocess);
  pipeline.addStage(gpuFFT);
  pipeline.addStage(gpuMagnitude);
  pipeline.addStage(cpuAnalysis);      // Auto GPU→CPU transfer here
  pipeline.addStage(gpuVisualization); // Auto CPU→GPU transfer here
  pipeline.addStage(dynamicCPU);       // Stays CPU
  pipeline.addStage(statsCPU);         // Stays CPU

  // Add audio stream
  final audioStream = await createAudioInputStream();
  pipeline.addStream(audioStream);

  await pipeline.start();
  print('Mixed CPU-GPU pipeline started!');
}

// Helper functions for CPU analysis
List<double> findSpectralPeaks(Float32List spectrum, double threshold) {
  final peaks = <double>[];
  for (int i = 1; i < spectrum.length - 1; i++) {
    if (spectrum[i] > threshold && 
        spectrum[i] > spectrum[i-1] && 
        spectrum[i] > spectrum[i+1]) {
      peaks.add(i.toDouble());
    }
  }
  return peaks;
}

List<double> analyzeHarmonics(Float32List spectrum, List<double> peaks) {
  // Simplified harmonic analysis
  final harmonics = <double>[];
  for (final peak in peaks) {
    for (int harmonic = 2; harmonic <= 5; harmonic++) {
      final harmonicFreq = peak * harmonic;
      if (harmonicFreq < spectrum.length) {
        harmonics.add(harmonicFreq);
      }
    }
  }
  return harmonics;
}

double calculateSpectralCentroid(Float32List spectrum) {
  double weightedSum = 0.0;
  double magnitudeSum = 0.0;
  
  for (int i = 0; i < spectrum.length; i++) {
    weightedSum += i * spectrum[i];
    magnitudeSum += spectrum[i];
  }
  
  return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0.0;
}

double calculateSpectralRolloff(Float32List spectrum, double threshold) {
  final totalEnergy = spectrum.fold(0.0, (sum, value) => sum + value);
  final rolloffEnergy = totalEnergy * threshold;
  
  double cumulativeEnergy = 0.0;
  for (int i = 0; i < spectrum.length; i++) {
    cumulativeEnergy += spectrum[i];
    if (cumulativeEnergy >= rolloffEnergy) {
      return i.toDouble();
    }
  }
  return spectrum.length.toDouble();
}

Future<Tensor> createScalarTensor(double value) async {
  final tensor = await Tensor.create([1]);
  await tensor.write(Float32List.fromList([value]));
  return tensor;
}
````

---

## Dynamic Pipeline Creation

### Runtime Pipeline Assembly

````dart
// Pipeline factory for dynamic creation
class DynamicPipelineFactory {
  static Pipeline createFromDescription(Map<String, dynamic> description) {
    final pipeline = Pipeline(
      id: description['id'] as String,
      enableBenchmarking: description['enableBenchmarking'] as bool? ?? false,
      debugEnabled: description['debugEnabled'] as bool? ?? false,
    );

    // Add stages based on description
    final stages = description['stages'] as List<dynamic>;
    for (final stageDesc in stages) {
      final stage = createStageFromDescription(stageDesc);
      pipeline.addStage(stage);
    }

    // Add streams
    final streams = description['streams'] as List<dynamic>? ?? [];
    for (final streamDesc in streams) {
      final stream = createStreamFromDescription(streamDesc);
      pipeline.addStream(stream);
    }

    return pipeline;
  }

  static PipelineStage createStageFromDescription(Map<String, dynamic> desc) {
    final type = desc['type'] as String;
    final stageId = desc['stageId'] as String;
    final parameters = desc['parameters'] as Map<String, dynamic>? ?? {};

    switch (type) {
      case 'WindowStage':
        return WindowStage(
          stageId: stageId,
          windowType: WindowType.values.firstWhere(
            (e) => e.toString().split('.').last == parameters['windowType'],
            orElse: () => WindowType.hann,
          ),
        );

      case 'FFTStage':
        return FFTStage(stageId: stageId);

      case 'MagnitudeStage':
        return MagnitudeStage(
          stageId: stageId,
          useLogScale: parameters['useLogScale'] as bool? ?? false,
          normalizeOutput: parameters['normalizeOutput'] as bool? ?? false,
          minDb: parameters['minDb'] as double? ?? -80.0,
          maxDb: parameters['maxDb'] as double? ?? 0.0,
        );

      case 'CustomStage':
        return createCustomStageFromDescription(desc);

      case 'CPUStage':
        return createCPUStageFromDescription(desc);

      default:
        throw UnsupportedError('Unknown stage type: $type');
    }
  }

  static PipelineStage createCustomStageFromDescription(Map<String, dynamic> desc) {
    final stageId = desc['stageId'] as String;
    final operations = desc['operations'] as List<dynamic>? ?? [];
    final parameters = desc['parameters'] as Map<String, dynamic>? ?? {};
    final shaderCode = desc['shaderCode'] as String?;
    final cpuCode = desc['cpuCode'] as String?;
    final executionTarget = desc['executionTarget'] as String?;

    final builder = StageBuilder(stageId);

    // Set execution target
    if (executionTarget != null) {
      switch (executionTarget) {
        case 'gpu':
          builder.setExecutionTarget(ExecutionTarget.gpu);
          break;
        case 'cpu':
          builder.setExecutionTarget(ExecutionTarget.cpu);
          break;
      }
    }

    // Add parameters
    for (final entry in parameters.entries) {
      final paramDesc = entry.value as Map<String, dynamic>;
      final parameter = createParameterFromDescription(entry.key, paramDesc);
      builder.withParameter(parameter);
    }

    // Add operations
    for (final opDesc in operations) {
      final operation = opDesc['operation'] as String;
      final arguments = opDesc['arguments'] as List<dynamic>?;
      if (arguments != null) {
        builder.op(operation, arguments: arguments);
      } else {
        builder.op(operation);
      }
    }

    // Add shader code
    if (shaderCode != null) {
      builder.executeShader(shaderCode);
    }

    // Add CPU code
    if (cpuCode != null) {
      builder.executeDynamicCPU(cpuCode);
    }

    return builder.build();
  }

  static Parameter createParameterFromDescription(String name, Map<String, dynamic> desc) {
    final type = desc['type'] as String;
    final defaultValue = desc['defaultValue'];
    final description = desc['description'] as String? ?? '';
    final min = desc['min'];
    final max = desc['max'];
    final allowedValues = desc['allowedValues'] as List<dynamic>?;

    switch (type) {
      case 'float':
        return Parameter.stageFloat(
          name: name,
          defaultValue: defaultValue as double,
          min: min as double?,
          max: max as double?,
          description: description,
        );

      case 'int':
        return Parameter.stageInt(
          name: name,
          defaultValue: defaultValue as int,
          min: min as int?,
          max: max as int?,
          description: description,
        );

      case 'bool':
        return Parameter.stageBool(
          name: name,
          defaultValue: defaultValue as bool,
          description: description,
        );

      case 'string':
        return Parameter.stageString(
          name: name,
          defaultValue: defaultValue as String,
          allowedValues: allowedValues?.map((e) => e as String).toList(),
          description: description,
        );

      case 'vec3':
        return Parameter.stageVec3(
          name: name,
          defaultValue: (defaultValue as List<dynamic>).map((e) => e as double).toList(),
          description: description,
        );

      default:
        throw UnsupportedError('Unknown parameter type: $type');
    }
  }

  static CPUStage createCPUStageFromDescription(Map<String, dynamic> desc) {
    final stageId = desc['stageId'] as String;
    final processorType = desc['processorType'] as String;
    final parameters = desc['parameters'] as Map<String, dynamic>? ?? {};

    switch (processorType) {
      case 'prebuilt':
        final operation = desc['operation'] as String;
        return PrebuiltCPUStage(
          stageId: stageId,
          operation: CPUOperation.values.firstWhere(
            (e) => e.toString().split('.').last == operation,
          ),
          parameters: parameters,
        );

      case 'dynamic':
        final dartCode = desc['dartCode'] as String;
        return DynamicCPUStage(
          stageId: stageId,
          dartCode: dartCode,
          isolateEnabled: desc['isolateEnabled'] as bool? ?? false,
        );

      case 'custom':
        // This would require a way to serialize/deserialize custom processor functions
        // For now, we'll use a registry pattern
        final processorName = desc['processorName'] as String;
        final processor = CPUProcessorRegistry.getProcessor(processorName);
        return CPUStage(
          stageId: stageId,
          processor: processor,
        );

      default:
        throw UnsupportedError('Unknown CPU processor type: $processorType');
    }
  }

  static MediaStream createStreamFromDescription(Map<String, dynamic> desc) {
    final type = desc['type'] as String;
    final id = desc['id'] as String;
    final parameters = desc['parameters'] as Map<String, dynamic>? ?? {};

    switch (type) {
      case 'audio':
        return AudioStream.fromDevice(
          MockAudioDevice(id),
          windowSize: parameters['windowSize'] as int? ?? 1024,
          sampleRate: parameters['sampleRate'] as int? ?? 44100,
        );

      case 'video':
        return VideoStream.fromCamera(
          MockVideoDevice(id),
          width: parameters['width'] as int? ?? 640,
          height: parameters['height'] as int? ?? 480,
          fps: parameters['fps'] as int? ?? 30,
        );

      default:
        throw UnsupportedError('Unknown stream type: $type');
    }
  }
}

// Example usage of dynamic pipeline creation
Future<void> demonstrateDynamicPipelineCreation() async {
  // Pipeline description (could come from JSON, UI, etc.)
  final pipelineDescription = {
    'id': 'dynamic_audio_pipeline',
    'enableBenchmarking': true,
    'debugEnabled': false,
    'stages': [
      {
        'type': 'WindowStage',
        'stageId': 'window',
        'parameters': {
          'windowType': 'hann',
        },
      },
      {
        'type': 'FFTStage',
        'stageId': 'fft',
      },
      {
        'type': 'CustomStage',
        'stageId': 'custom_processing',
        'executionTarget': 'gpu',
        'operations': [
          {
            'operation': 'multiplyScalar',
            'arguments': ['param:gain'],
          },
          {
            'operation': 'clamp',
            'arguments': [0.0, 1.0],
          },
        ],
        'parameters': {
          'gain': {
            'type': 'float',
            'defaultValue': 2.0,
            'min': 0.1,
            'max': 10.0,
            'description': 'Processing gain',
          },
        },
      },
      {
        'type': 'CPUStage',
        'stageId': 'cpu_analysis',
        'processorType': 'dynamic',
        'dartCode': '''
          Future<Map<String, Tensor>> process(Map<String, Tensor> inputs, Map<String, dynamic> params) async {
            final input = inputs['input']!;
            final data = await input.getData() as Float32List;
            
            // Calculate RMS
            double sum = 0.0;
            for (final sample in data) {
              sum += sample * sample;
            }
            final rms = math.sqrt(sum / data.length);
            
            final output = await Tensor.create([1]);
            await output.write(Float32List.fromList([rms]));
            return {'rms': output};
          }
        ''',
        'isolateEnabled': true,
      },
    ],
    'streams': [
      {
        'type': 'audio',
        'id': 'main_audio',
        'parameters': {
          'windowSize': 1024,
          'sampleRate': 44100,
        },
      },
    ],
  };

  // Create pipeline from description
  final pipeline = DynamicPipelineFactory.createFromDescription(pipelineDescription);

  await pipeline.start();
  print('Dynamic pipeline created and started!');

  // Modify pipeline at runtime
  await modifyPipelineAtRuntime(pipeline);
}

Future<void> modifyPipelineAtRuntime(Pipeline pipeline) async {
  // Add a new stage dynamically
  final newStage = StageBuilder('runtime_added')
      .withParameter(
        Parameter.stageFloat(
          name: 'runtime_param',
          defaultValue: 1.0,
          min: 0.0,
          max: 5.0,
          description: 'Runtime added parameter',
        ),
      )
      .op('multiplyScalar', arguments: ['param:runtime_param'])
      .build();

  pipeline.addStage(newStage);

  // Update parameter values
  final customStage = pipeline.getStage('custom_processing');
  if (customStage != null) {
    await customStage.setPropertyValue('gain', 3.0);
  }

  print('Pipeline modified at runtime!');
}

// Registry for custom CPU processors
class CPUProcessorRegistry {
  static final Map<String, CPUProcessor> _processors = {};

  static void registerProcessor(String name, CPUProcessor processor) {
    _processors[name] = processor;
  }

  static CPUProcessor getProcessor(String name) {
    final processor = _processors[name];
    if (processor == null) {
      throw ArgumentError('Unknown CPU processor: $name');
    }
    return processor;
  }

  static void registerBuiltInProcessors() {
    registerProcessor('rms_calculator', (inputs, parameters) async {
      final input = inputs['input'] as Tensor;
      final data = await input.getData() as Float32List;
      
      double sum = 0.0;
      for (final sample in data) {
        sum += sample * sample;
      }
      final rms = math.sqrt(sum / data.length);
      
      final output = await Tensor.create([1]);
      await output.write(Float32List.fromList([rms]));
      return {'rms': output};
    });

    registerProcessor('peak_detector', (inputs, parameters) async {
      final input = inputs['input'] as Tensor;
      final threshold = parameters['threshold'] as double? ?? 0.1;
      final data = await input.getData() as Float32List;
      
      final peaks = <double>[];
      for (int i = 1; i < data.length - 1; i++) {
        if (data[i] > threshold && 
            data[i] > data[i-1] && 
            data[i] > data[i+1]) {
          peaks.add(i.toDouble());
        }
      }
      
      final output = await Tensor.create([peaks.length]);
      await output.write(Float32List.fromList(peaks));
      return {'peaks': output};
    });
  }
}

// Mock classes for demonstration
class MockAudioDevice extends AudioDevice {
  final String deviceId;
  MockAudioDevice(this.deviceId);
  
  @override
  String get id => deviceId;
  
  @override
  String get name => 'Mock Audio Device $deviceId';
}

class MockVideoDevice extends VideoDevice {
  final String deviceId;
  MockVideoDevice(this.deviceId);
  
  @override
  String get id => deviceId;
  
  @override
  String get name => 'Mock Video Device $deviceId';
}
````

---

## Pipeline Serialization

### Complete Serialization Example

````dart
// Pipeline configuration classes for serialization
class PipelineConfig {
  final String id;
  final bool enableBenchmarking;
  final bool debugEnabled;
  final QueueConfigData queueConfig;
  final List<StageConfigData> stages;
  final List<StreamConfigData> streams;
  final List<ConnectionData> connections;

  PipelineConfig({
    required this.id,
    this.enableBenchmarking = false,
    this.debugEnabled = false,
    required this.queueConfig,
    required this.stages,
    required this.streams,
    required this.connections,
  });

  Map<String, dynamic> toJson() => {
    'id': id,
    'enableBenchmarking': enableBenchmarking,
    'debugEnabled': debugEnabled,
    'queueConfig': queueConfig.toJson(),
    'stages': stages.map((s) => s.toJson()).toList(),
    'streams': streams.map((s) => s.toJson()).toList(),
    'connections': connections.map((c) => c.toJson()).toList(),
  };

  factory PipelineConfig.fromJson(Map<String, dynamic> json) => PipelineConfig(
    id: json['id'] as String,
    enableBenchmarking: json['enableBenchmarking'] as bool? ?? false,
    debugEnabled: json['debugEnabled'] as bool? ?? false,
    queueConfig: QueueConfigData.fromJson(json['queueConfig'] as Map<String, dynamic>),
    stages: (json['stages'] as List<dynamic>)
        .map((s) => StageConfigData.fromJson(s as Map<String, dynamic>))
        .toList(),
    streams: (json['streams'] as List<dynamic>)
        .map((s) => StreamConfigData.fromJson(s as Map<String, dynamic>))
        .toList(),
    connections: (json['connections'] as List<dynamic>)
        .map((c) => ConnectionData.fromJson(c as Map<String, dynamic>))
        .toList(),
  );
}

class StageConfigData {
  final String stageId;
  final String type;
  final String? executionTarget;
  final Map<String, dynamic> parameters;
  final List<OperationData> operations;
  final List<InputPortData> inputPorts;
  final List<OutputPortData> outputPorts;
  final String? shaderCode;
  final String? cpuCode;
  final Map<String, String>? shaderParameters;

  StageConfigData({
    required this.stageId,
    required this.type,
    this.executionTarget,
    this.parameters = const {},
    this.operations = const [],
    this.inputPorts = const [],
    this.outputPorts = const [],
    this.shaderCode,
    this.cpuCode,
    this.shaderParameters,
  });

  Map<String, dynamic> toJson() => {
    'stageId': stageId,
    'type': type,
    if (executionTarget != null) 'executionTarget': executionTarget,
    'parameters': parameters,
    'operations': operations.map((o) => o.toJson()).toList(),
    'inputPorts': inputPorts.map((p) => p.toJson()).toList(),
    'outputPorts': outputPorts.map((p) => p.toJson()).toList(),
    if (shaderCode != null) 'shaderCode': shaderCode,
    if (cpuCode != null) 'cpuCode': cpuCode,
    if (shaderParameters != null) 'shaderParameters': shaderParameters,
  };

  factory StageConfigData.fromJson(Map<String, dynamic> json) => StageConfigData(
    stageId: json['stageId'] as String,
    type: json['type'] as String,
    executionTarget: json['executionTarget'] as String?,
    parameters: json['parameters'] as Map<String, dynamic>? ?? {},
    operations: (json['operations'] as List<dynamic>? ?? [])
        .map((o) => OperationData.fromJson(o as Map<String, dynamic>))
        .toList(),
    inputPorts: (json['inputPorts'] as List<dynamic>? ?? [])
        .map((p) => InputPortData.fromJson(p as Map<String, dynamic>))
        .toList(),
    outputPorts: (json['outputPorts'] as List<dynamic>? ?? [])
        .map((p) => OutputPortData.fromJson(p as Map<String, dynamic>))
        .toList(),
    shaderCode: json['shaderCode'] as String?,
    cpuCode: json['cpuCode'] as String?,
    shaderParameters: (json['shaderParameters'] as Map<String, dynamic>?)
        ?.map((k, v) => MapEntry(k, v as String)),
  );
}

class OperationData {
  final String operation;
  final List<dynamic>? arguments;
  final Map<String, dynamic>? namedArguments;

  OperationData({
    required this.operation,
    this.arguments,
    this.namedArguments,
  });

  Map<String, dynamic> toJson() => {
    'operation': operation,
    if (arguments != null) 'arguments': arguments,
    if (namedArguments != null) 'namedArguments': namedArguments,
  };

  factory OperationData.fromJson(Map<String, dynamic> json) => OperationData(
    operation: json['operation'] as String,
    arguments: json['arguments'] as List<dynamic>?,
    namedArguments: json['namedArguments'] as Map<String, dynamic>?,
  );
}

class InputPortData {
  final String name;
  final String type;
  final Map<String, dynamic> properties;

  InputPortData({
    required this.name,
    required this.type,
    this.properties = const {},
  });

  Map<String, dynamic> toJson() => {
    'name': name,
    'type': type,
    'properties': properties,
  };

  factory InputPortData.fromJson(Map<String, dynamic> json) => InputPortData(
    name: json['name'] as String,
    type: json['type'] as String,
    properties: json['properties'] as Map<String, dynamic>? ?? {},
  );
}

class OutputPortData {
  final String name;
  final String type;
  final Map<String, dynamic> properties;

  OutputPortData({
    required this.name,
    required this.type,
    this.properties = const {},
  });

  Map<String, dynamic> toJson() => {
    'name': name,
    'type': type,
    'properties': properties,
  };

  factory OutputPortData.fromJson(Map<String, dynamic> json) => OutputPortData(
    name: json['name'] as String,
    type: json['type'] as String,
    properties: json['properties'] as Map<String, dynamic>? ?? {},
  );
}

class QueueConfigData {
  final String strategy;
  final int? batchSize;
  final int? maxSize;

  QueueConfigData({
    required this.strategy,
    this.batchSize,
    this.maxSize,
  });

  Map<String, dynamic> toJson() => {
    'strategy': strategy,
    if (batchSize != null) 'batchSize': batchSize,
    if (maxSize != null) 'maxSize': maxSize,
  };

  factory QueueConfigData.fromJson(Map<String, dynamic> json) => QueueConfigData(
    strategy: json['strategy'] as String,
    batchSize: json['batchSize'] as int?,
    maxSize: json['maxSize'] as int?,
  );
}

class StreamConfigData {
  final String id;
  final String type;
  final Map<String, dynamic> properties;

  StreamConfigData({
    required this.id,
    required this.type,
    this.properties = const {},
  });

  Map<String, dynamic> toJson() => {
    'id': id,
    'type': type,
    'properties': properties,
  };

  factory StreamConfigData.fromJson(Map<String, dynamic> json) => StreamConfigData(
    id: json['id'] as String,
    type: json['type'] as String,
    properties: json['properties'] as Map<String, dynamic>? ?? {},
  );
}

class ConnectionData {
  final String fromStage;
  final String toStage;
  final String? fromPort;
  final String? toPort;

  ConnectionData({
    required this.fromStage,
    required this.toStage,
    this.fromPort,
    this.toPort,
  });

  Map<String, dynamic> toJson() => {
        'fromStage': fromStage,
    'toStage': toStage,
    if (fromPort != null) 'fromPort': fromPort,
    if (toPort != null) 'toPort': toPort,
  };

  factory ConnectionData.fromJson(Map<String, dynamic> json) => ConnectionData(
    fromStage: json['fromStage'] as String,
    toStage: json['toStage'] as String,
    fromPort: json['fromPort'] as String?,
    toPort: json['toPort'] as String?,
  );
}

// Pipeline serialization service
class PipelineSerializer {
  static String toJson(Pipeline pipeline) {
    final config = _pipelineToConfig(pipeline);
    return jsonEncode(config.toJson());
  }

  static Pipeline fromJson(String json) {
    final configData = jsonDecode(json) as Map<String, dynamic>;
    final config = PipelineConfig.fromJson(configData);
    return _configToPipeline(config);
  }

  static PipelineConfig _pipelineToConfig(Pipeline pipeline) {
    final stages = <StageConfigData>[];
    final connections = <ConnectionData>[];

    for (final stage in pipeline.stages) {
      stages.add(_stageToConfig(stage));
      
      // Extract connections from stage relationships
      for (final outputPort in stage.outputPorts) {
        for (final connection in outputPort.connections) {
          connections.add(ConnectionData(
            fromStage: stage.stageId,
            toStage: connection.targetStage.stageId,
            fromPort: outputPort.name,
            toPort: connection.targetPort?.name,
          ));
        }
      }
    }

    final streams = pipeline.streams.map((stream) => StreamConfigData(
      id: stream.id,
      type: stream.type.toString().split('.').last,
      properties: _extractStreamProperties(stream),
    )).toList();

    return PipelineConfig(
      id: pipeline.id,
      enableBenchmarking: pipeline.enableBenchmarking,
      debugEnabled: pipeline.debugEnabled,
      queueConfig: QueueConfigData(
        strategy: pipeline.queueConfig.strategy.toString().split('.').last,
        batchSize: pipeline.queueConfig.batchSize,
        maxSize: pipeline.queueConfig.maxSize,
      ),
      stages: stages,
      streams: streams,
      connections: connections,
    );
  }

  static StageConfigData _stageToConfig(PipelineStage stage) {
    final inputPorts = stage.inputPorts.map((port) => InputPortData(
      name: port.name,
      type: port.runtimeType.toString(),
      properties: _extractPortProperties(port),
    )).toList();

    final outputPorts = stage.outputPorts.map((port) => OutputPortData(
      name: port.name,
      type: port.runtimeType.toString(),
      properties: _extractPortProperties(port),
    )).toList();

    String? executionTarget;
    if (stage is GPUStage) {
      executionTarget = 'gpu';
    } else if (stage is CPUStage) {
      executionTarget = 'cpu';
    }

    return StageConfigData(
      stageId: stage.stageId,
      type: stage.runtimeType.toString(),
      executionTarget: executionTarget,
      parameters: _extractStageParameters(stage),
      operations: _extractStageOperations(stage),
      inputPorts: inputPorts,
      outputPorts: outputPorts,
      shaderCode: _extractShaderCode(stage),
      cpuCode: _extractCPUCode(stage),
      shaderParameters: _extractShaderParameters(stage),
    );
  }

  static Pipeline _configToPipeline(PipelineConfig config) {
    final pipeline = Pipeline(
      id: config.id,
      enableBenchmarking: config.enableBenchmarking,
      debugEnabled: config.debugEnabled,
      queueConfig: QueueConfig(
        strategy: QueueStrategy.values.firstWhere(
          (e) => e.toString().split('.').last == config.queueConfig.strategy,
        ),
        batchSize: config.queueConfig.batchSize,
        maxSize: config.queueConfig.maxSize,
      ),
    );

    // Create stages
    final stageMap = <String, PipelineStage>{};
    for (final stageConfig in config.stages) {
      final stage = _configToStage(stageConfig);
      pipeline.addStage(stage);
      stageMap[stage.stageId] = stage;
    }

    // Create streams
    for (final streamConfig in config.streams) {
      final stream = _configToStream(streamConfig);
      pipeline.addStream(stream);
    }

    // Establish connections
    for (final connection in config.connections) {
      final fromStage = stageMap[connection.fromStage];
      final toStage = stageMap[connection.toStage];
      
      if (fromStage != null && toStage != null) {
        // Connect stages using pipeline's connection system
        pipeline.connectStages(
          fromStage: fromStage,
          toStage: toStage,
          fromPort: connection.fromPort,
          toPort: connection.toPort,
        );
      }
    }

    return pipeline;
  }

  static PipelineStage _configToStage(StageConfigData config) {
    switch (config.type) {
      case 'WindowStage':
        return WindowStage(
          stageId: config.stageId,
          windowType: WindowType.values.firstWhere(
            (e) => e.toString().split('.').last == config.parameters['windowType'],
          ),
        );

      case 'FFTStage':
        return FFTStage(stageId: config.stageId);

      case 'MagnitudeStage':
        return MagnitudeStage(
          stageId: config.stageId,
          useLogScale: config.parameters['useLogScale'] as bool? ?? false,
          normalizeOutput: config.parameters['normalizeOutput'] as bool? ?? false,
          minDb: config.parameters['minDb'] as double? ?? -80.0,
          maxDb: config.parameters['maxDb'] as double? ?? 0.0,
        );

      case 'CPUStage':
        return _createCPUStageFromConfig(config);

      case 'CustomStage':
      default:
        return _createCustomStageFromConfig(config);
    }
  }

  static CPUStage _createCPUStageFromConfig(StageConfigData config) {
    if (config.cpuCode != null) {
      return DynamicCPUStage(
        stageId: config.stageId,
        dartCode: config.cpuCode!,
        isolateEnabled: config.parameters['isolateEnabled'] as bool? ?? false,
      );
    } else if (config.parameters.containsKey('operation')) {
      final operation = config.parameters['operation'] as String;
      return PrebuiltCPUStage(
        stageId: config.stageId,
        operation: CPUOperation.values.firstWhere(
          (e) => e.toString().split('.').last == operation,
        ),
        parameters: Map<String, dynamic>.from(config.parameters)..remove('operation'),
      );
    } else {
      // Custom processor from registry
      final processorName = config.parameters['processorName'] as String;
      return CPUStage(
        stageId: config.stageId,
        processor: CPUProcessorRegistry.getProcessor(processorName),
      );
    }
  }

  static PipelineStage _createCustomStageFromConfig(StageConfigData config) {
    final builder = StageBuilder(config.stageId);

    // Set execution target
    if (config.executionTarget == 'gpu') {
      builder.setExecutionTarget(ExecutionTarget.gpu);
    } else if (config.executionTarget == 'cpu') {
      builder.setExecutionTarget(ExecutionTarget.cpu);
    }

    // Add parameters
    for (final entry in config.parameters.entries) {
      if (entry.value is Map<String, dynamic>) {
        final paramData = entry.value as Map<String, dynamic>;
        final parameter = _createParameterFromData(entry.key, paramData);
        builder.withParameter(parameter);
      }
    }

    // Add input ports
    for (final portData in config.inputPorts) {
      _addInputPortToBuilder(builder, portData);
    }

    // Add output ports
    for (final portData in config.outputPorts) {
      _addOutputPortToBuilder(builder, portData);
    }

    // Add operations
    for (final opData in config.operations) {
      if (opData.arguments != null) {
        builder.op(opData.operation, arguments: opData.arguments!);
      } else {
        builder.op(opData.operation);
      }
    }

    // Add shader code
    if (config.shaderCode != null) {
      if (config.shaderParameters != null) {
        builder.executeShader(config.shaderCode!, shaderParameters: config.shaderParameters!);
      } else {
        builder.executeShader(config.shaderCode!);
      }
    }

    // Add CPU code
    if (config.cpuCode != null) {
      builder.executeDynamicCPU(config.cpuCode!);
    }

    return builder.build();
  }

  // Helper methods for extraction and creation
  static Map<String, dynamic> _extractStageParameters(PipelineStage stage) {
    final parameters = <String, dynamic>{};
    for (final entry in stage.parameterDefinitions.entries) {
      parameters[entry.key] = _parameterToData(entry.value);
    }
    return parameters;
  }

  static Map<String, dynamic> _parameterToData(Parameter parameter) {
    return {
      'type': _getParameterType(parameter),
      'defaultValue': parameter.defaultValue,
      'description': parameter.description,
      if (parameter.min != null) 'min': parameter.min,
      if (parameter.max != null) 'max': parameter.max,
      if (parameter.allowedValues != null) 'allowedValues': parameter.allowedValues,
    };
  }

  static String _getParameterType(Parameter parameter) {
    if (parameter.defaultValue is double) return 'float';
    if (parameter.defaultValue is int) return 'int';
    if (parameter.defaultValue is bool) return 'bool';
    if (parameter.defaultValue is String) return 'string';
    if (parameter.defaultValue is List<double>) return 'vec3';
    return 'unknown';
  }

  static Parameter _createParameterFromData(String name, Map<String, dynamic> data) {
    final type = data['type'] as String;
    final defaultValue = data['defaultValue'];
    final description = data['description'] as String? ?? '';

    switch (type) {
      case 'float':
        return Parameter.stageFloat(
          name: name,
          defaultValue: defaultValue as double,
          min: data['min'] as double?,
          max: data['max'] as double?,
          description: description,
        );
      case 'int':
        return Parameter.stageInt(
          name: name,
          defaultValue: defaultValue as int,
          min: data['min'] as int?,
          max: data['max'] as int?,
          description: description,
        );
      case 'bool':
        return Parameter.stageBool(
          name: name,
          defaultValue: defaultValue as bool,
          description: description,
        );
      case 'string':
        return Parameter.stageString(
          name: name,
          defaultValue: defaultValue as String,
          allowedValues: (data['allowedValues'] as List<dynamic>?)?.cast<String>(),
          description: description,
        );
      case 'vec3':
        return Parameter.stageVec3(
          name: name,
          defaultValue: (defaultValue as List<dynamic>).cast<double>(),
          description: description,
        );
      default:
        throw UnsupportedError('Unknown parameter type: $type');
    }
  }

  // Additional helper methods would go here for:
  // - _extractStageOperations
  // - _extractShaderCode
  // - _extractCPUCode
  // - _extractShaderParameters
  // - _extractStreamProperties
  // - _extractPortProperties
  // - _addInputPortToBuilder
  // - _addOutputPortToBuilder
  // - _configToStream
}

// Example usage of pipeline serialization
Future<void> demonstratePipelineSerialization() async {
  // Create a complex pipeline
  final originalPipeline = Pipeline(id: 'serialization_demo');
  
  // Add stages
  originalPipeline.addStage(WindowStage(stageId: 'window', windowType: WindowType.hann));
  originalPipeline.addStage(FFTStage(stageId: 'fft'));
  
  final cpuStage = CPUStage(
    stageId: 'cpu_analysis',
    processor: (inputs, params) async {
      // CPU processing logic
      return inputs;
    },
  );
  originalPipeline.addStage(cpuStage);

  // Serialize pipeline to JSON
  final json = PipelineSerializer.toJson(originalPipeline);
  print('Serialized pipeline: $json');

  // Save to file
  await savePipelineToFile(json, 'my_pipeline.json');

  // Load from file
  final loadedJson = await loadPipelineFromFile('my_pipeline.json');
  final restoredPipeline = PipelineSerializer.fromJson(loadedJson);

  print('Pipeline restored with ${restoredPipeline.stages.length} stages');
}

Future<void> savePipelineToFile(String json, String filename) async {
  // Implementation depends on platform
  // For Flutter: use path_provider
  // For web: use local storage or download
  print('Saved pipeline to $filename');
}

Future<String> loadPipelineFromFile(String filename) async {
  // Implementation depends on platform
  return '{}'; // Placeholder
}

// Pipeline network for complex processing
class PipelineNetwork {
  final Map<String, Pipeline> _pipelines = {};
  final Map<String, StreamSubscription> _subscriptions = {};
  final Map<String, List<PipelineConnection>> _connections = {};

  void addPipeline(String id, Pipeline pipeline) {
    _pipelines[id] = pipeline;
    _connections[id] = [];
  }

  void connectPipelines({
    required String fromPipelineId,
    required String toPipelineId,
    required String fromStageId,
    required String toStreamId,
    String? fromOutputPort,
    TransferMode transferMode = TransferMode.realtime,
  }) {
    final connection = PipelineConnection(
      fromPipelineId: fromPipelineId,
      toPipelineId: toPipelineId,
      fromStageId: fromStageId,
      toStreamId: toStreamId,
      fromOutputPort: fromOutputPort,
      transferMode: transferMode,
    );

    _connections[fromPipelineId]!.add(connection);
    _setupConnection(connection);
  }

  void _setupConnection(PipelineConnection connection) {
    final fromPipeline = _pipelines[connection.fromPipelineId]!;
    final toPipeline = _pipelines[connection.toPipelineId]!;

    final subscription = fromPipeline.eventNotifier.eventStream.listen((event) {
      if (event is PipelineRunCompleteEvent) {
        _transferData(connection, fromPipeline, toPipeline);
      }
    });

    _subscriptions['${connection.fromPipelineId}->${connection.toPipelineId}'] = subscription;
  }

  Future<void> _transferData(
    PipelineConnection connection,
    Pipeline fromPipeline,
    Pipeline toPipeline,
  ) async {
    final fromStage = fromPipeline.getStage(connection.fromStageId);
    if (fromStage == null) return;

    // Get output data from source stage
    final outputData = fromPipeline.getStageOutput(
      connection.fromStageId,
      fromPipeline.streams.first.id,
    );

    if (outputData == null) return;

    // Create inter-pipeline stream for data transfer
    final transferStream = InterPipelineStream(
      id: connection.toStreamId,
      sourceData: outputData,
      transferMode: connection.transferMode,
    );

    // Add to target pipeline
    toPipeline.addStream(transferStream);

    // Trigger processing in target pipeline
    if (connection.transferMode == TransferMode.realtime) {
      await transferStream.pushData(outputData);
    }
  }

  Future<void> startAll() async {
    for (final pipeline in _pipelines.values) {
      await pipeline.start();
    }
  }

  Future<void> stopAll() async {
    for (final subscription in _subscriptions.values) {
      await subscription.cancel();
    }
    for (final pipeline in _pipelines.values) {
      await pipeline.stop();
    }
  }

  void dispose() {
    for (final pipeline in _pipelines.values) {
      pipeline.dispose();
    }
    _pipelines.clear();
    _connections.clear();
  }
}

class PipelineConnection {
  final String fromPipelineId;
  final String toPipelineId;
  final String fromStageId;
  final String toStreamId;
  final String? fromOutputPort;
  final TransferMode transferMode;

  PipelineConnection({
    required this.fromPipelineId,
    required this.toPipelineId,
    required this.fromStageId,
    required this.toStreamId,
    this.fromOutputPort,
    required this.transferMode,
  });
}

enum TransferMode {
  realtime,    // Transfer data immediately
  buffered,    // Buffer and transfer in batches
  onDemand,    // Transfer only when requested
  synchronized // Wait for both pipelines to be ready
}

class InterPipelineStream extends MediaStream {
  final Map<String, Tensor> sourceData;
  final TransferMode transferMode;
  final Queue<Map<String, Tensor>> _buffer = Queue();

  InterPipelineStream({
    required String id,
    required this.sourceData,
    required this.transferMode,
  }) : super(id: id, type: MediaStreamType.data);

  Future<void> pushData(Map<String, Tensor> data) async {
    switch (transferMode) {
      case TransferMode.realtime:
        notifyTensorUpdate(data.values.first);
        break;
      case TransferMode.buffered:
        _buffer.add(data);
        if (_buffer.length >= 10) { // Buffer size
          await _flushBuffer();
        }
        break;
      case TransferMode.onDemand:
        _buffer.add(data);
        break;
      case TransferMode.synchronized:
        await _synchronizedTransfer(data);
        break;
    }
  }

  Future<void> _flushBuffer() async {
    while (_buffer.isNotEmpty) {
      final data = _buffer.removeFirst();
      notifyTensorUpdate(data.values.first);
      await Future.delayed(Duration(milliseconds: 1)); // Small delay
    }
  }

  Future<void> _synchronizedTransfer(Map<String, Tensor> data) async {
    // Wait for target pipeline to be ready
    // Implementation would check target pipeline state
    notifyTensorUpdate(data.values.first);
  }
}

// Example: Multi-stage audio processing network
Future<void> createAudioProcessingNetwork() async {
  final network = PipelineNetwork();

  // Pipeline 1: Input processing
  final inputPipeline = Pipeline(id: 'input_processor');
  inputPipeline.addStage(StageBuilder('input_gain')
      .withParameter(Parameter.stageFloat(
        name: 'gain',
        defaultValue: 1.0,
        min: 0.0,
        max: 4.0,
        description: 'Input gain',
      ))
      .op('multiplyScalar', arguments: ['param:gain'])
      .build());
  
  inputPipeline.addStage(WindowStage(stageId: 'window', windowType: WindowType.hann));
  
  // Add input stream
  final audioInput = await createAudioInputStream();
  inputPipeline.addStream(audioInput);

  // Pipeline 2: Frequency analysis
  final freqPipeline = Pipeline(id: 'frequency_analyzer');
  freqPipeline.addStage(FFTStage(stageId: 'fft'));
  freqPipeline.addStage(MagnitudeStage(stageId: 'magnitude', useLogScale: true));
  
  // CPU analysis stage
  freqPipeline.addStage(CPUStage(
    stageId: 'spectral_analysis',
    processor: (inputs, params) async {
      final spectrum = inputs['input'] as Tensor;
      final data = await spectrum.getData() as Float32List;
      
      // Perform detailed spectral analysis
      final centroid = calculateSpectralCentroid(data);
      final rolloff = calculateSpectralRolloff(data, 0.85);
      final flatness = calculateSpectralFlatness(data);
      
      final analysis = await Tensor.create([3]);
      await analysis.write(Float32List.fromList([centroid, rolloff, flatness]));
      
      return {'spectral_features': analysis};
    },
  ));

  // Pipeline 3: Visualization
  final visualPipeline = Pipeline(id: 'visualizer');
  visualPipeline.addStage(HistoryStage(
    stageId: 'history',
    maxTimeSlices: 256,
    historyType: HistoryType.spectrogram,
  ));
  
  visualPipeline.addStage(SpectrogramVisualizationStage(
    stageId: 'visualization',
    minFreq: 20.0,
    maxFreq: 20000.0,
    sampleRate: 48000.0,
  ));

  // Pipeline 4: Feature aggregation (CPU)
  final aggregationPipeline = Pipeline(id: 'feature_aggregator');
  aggregationPipeline.addStage(DynamicCPUStage(
    stageId: 'feature_aggregation',
    dartCode: '''
      Future<Map<String, Tensor>> process(Map<String, Tensor> inputs, Map<String, dynamic> params) async {
        final features = inputs['spectral_features']!;
        final data = await features.getData() as Float32List;
        
        // Aggregate features over time
        final windowSize = params['window_size'] as int? ?? 10;
        
        // Simple moving average aggregation
        final aggregated = List<double>.filled(data.length, 0.0);
        for (int i = 0; i < data.length; i++) {
          aggregated[i] = data[i]; // Simplified - would actually aggregate over time
        }
        
        final output = await Tensor.create(aggregated.length);
        await output.write(Float32List.fromList(aggregated));
        return {'aggregated_features': output};
      }
    ''',
    isolateEnabled: true,
  ));

  // Add pipelines to network
  network.addPipeline('input', inputPipeline);
  network.addPipeline('frequency', freqPipeline);
  network.addPipeline('visual', visualPipeline);
  network.addPipeline('aggregation', aggregationPipeline);

  // Connect pipelines
  network.connectPipelines(
    fromPipelineId: 'input',
    toPipelineId: 'frequency',
    fromStageId: 'window',
    toStreamId: 'windowed_audio',
    transferMode: TransferMode.realtime,
  );

  network.connectPipelines(
    fromPipelineId: 'frequency',
    toPipelineId: 'visual',
    fromStageId: 'magnitude',
    toStreamId: 'spectrum_data',
    transferMode: TransferMode.realtime,
  );

  network.connectPipelines(
    fromPipelineId: 'frequency',
    toPipelineId: 'aggregation',
    fromStageId: 'spectral_analysis',
    toStreamId: 'feature_data',
    transferMode: TransferMode.buffered,
  );

  // Start the entire network
  await network.startAll();

  // Monitor the network
  monitorPipelineNetwork(network);

  print('Audio processing network started!');
}

void monitorPipelineNetwork(PipelineNetwork network) {
  // Set up monitoring for each pipeline in the network
  for (final entry in network._pipelines.entries) {
    final pipelineId = entry.key;
    final pipeline = entry.value;
    
    pipeline.eventNotifier.eventStream.listen((event) {
      if (event is PipelineErrorEvent) {
        print('Error in pipeline $pipelineId: ${event.error}');
      } else if (event is PipelineBenchmarkEvent) {
        final stats = pipeline.getBenchmarkStats();
        print('Pipeline $pipelineId performance: $stats');
      }
    });
  }
}

// Helper function for spectral flatness calculation
double calculateSpectralFlatness(Float32List spectrum) {
  double geometricMean = 1.0;
  double arithmeticMean = 0.0;
  int validBins = 0;

  for (final value in spectrum) {
    if (value > 0) {
      geometricMean *= value;
      arithmeticMean += value;
      validBins++;
    }
  }

  if (validBins == 0) return 0.0;

  geometricMean = math.pow(geometricMean, 1.0 / validBins) as double;
  arithmeticMean /= validBins;

  return arithmeticMean > 0 ? geometricMean / arithmeticMean : 0.0;
}
````

---

## Serialization Use Case: Live Editing

### Scenario

User edits a dynamic CPU operation code snippet; UI sends patch with:

```json
{
  "type": "patch",
  "pipeline_id": "audio_visualizer",
  "ops": [
    {
      "op": "replace_operation",
      "stage_id": "custom_cpu",
      "operation_index": 0,
      "operation": {
        "type": "operation",
        "op_kind": "dynamic_cpu",
        "input_keys": ["windowed"],
        "output_keys": ["processed"],
        "dart_code": "Future<Map<String,TypedData>> main(...) { /* new code */ }"
      }
    }
  ]
}
```

Backend (runtime):
1. Validate op shape & keys.
2. Recompile dynamic code (collect compile errors).
3. If success, hot-swap operation; else emit error to log panel.

### Inter-Pipeline Future Skeleton
```json
{
  "inter_pipeline_links": [
    {
      "from": { "pipeline": "audio_source", "output": "fft_output" },
      "to":   { "pipeline": "audio_effects", "input": "spectrum_input" },
      "mode": "realtime"
    }
  ]
}
```

## Logging & Error Panel
Every failure emits:
```json
{
  "event": "runtime_error",
  "pipeline_id": "audio_visualizer",
  "stage_id": "custom_cpu",
  "operation_index": 0,
  "severity": "error",
  "code": "DART_COMPILE_ERROR",
  "message": "Unknown identifier x",
  "snippet": "result[i] = x + 1;",
  "line": 12,
  "column": 15
}
```
UI appends event to result panel with filtering (error/warning/info).

---

## Example: Declarative Shapes

Stage excerpt:
```json
{
  "type": "stage",
  "stage_id": "mels",
  "stage_kind": "MelStage",
  "inputs": [{ "name": "fft_out", "port_type": "tensor", "shape_policy": "flexible" }],
  "outputs": [
    {
      "name": "mel",
      "size_policy": "computed",
      "shape_expr": { "op": "fixed", "shape": [128] }
    }
  ],
  "parameters": { "mel_bins": 128 }
}
```

Runtime shape transform:
- Loader sees `fixed` → no dependency analysis required.
- If `shape_expr` absent and policy is input_dependent: default is first input’s shape.

## Example: Broadcast + Concat

```json
{
  "outputs": [
    {
      "name": "merged",
      "size_policy": "computed",
      "shape_expr": {
        "op": "concat_along",
        "axis": 0,
        "inputs": ["path_a", "path_b"]
      }
    }
  ]
}
```

## Shape Validation Failures (Samples)

| Error | Example |
|-------|---------|
| SHAPE_AXIS_OOB | axis=3 on rank-3 concat_along |
| SHAPE_INCOMPATIBLE | broadcast_pair with shapes [8,16] & [10,16] |
| SHAPE_PARAM_MISSING | scale_dim referencing missing param:scale_y |
| SHAPE_OP_UNKNOWN | op=tile_repeat (not implemented yet) |
