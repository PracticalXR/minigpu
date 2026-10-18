import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

/// Advanced signal processing pipeline demonstrating new architecture features
/// Shows per-stream processing, stream merging, and selective routing
Future<void> main() async {
  print('🚀 GPU Pipeline Advanced Signal Processing Example');
  print('=' * 60);

  try {
    // Initialize GPU with error checking
    // TODO: DEBUGLOG print('🔧 Initializing GPU...');
    await Minigpu().init();
    print('✅ GPU initialized successfully');

    // Test basic tensor creation before proceeding
    print('🧪 Testing basic tensor operations...');
    await testBasicTensorOperations();
    print('✅ Basic tensor operations working');

    // Initialize tensor operation registry
    // TODO: DEBUGLOG print('🔧 Initializing tensor operation registry...');
    TensorOperationRegistry.initialize();

    // Run examples with more conservative approach
    await runSafeTensorExample();
    //Only run other examples if the first one succeeds
    await runBasicSignalProcessingExample();
    await runMultiStreamMergingExample();
    await runSelectiveProcessingExample();
  } catch (e, stackTrace) {
    print('❌ Error: $e');
    print('Stack trace: $stackTrace');
  }
}

/// Test basic tensor operations to ensure GPU is working properly
Future<void> testBasicTensorOperations() async {
  try {
    // Test 1: Simple tensor creation
    print('  📊 Creating test tensor...');
    final testTensor = await Tensor.create([4]);

    // Test 2: Set simple data
    print('  📝 Setting tensor data...');
    await testTensor.write(Float32List.fromList([1.0, 2.0, 3.0, 4.0]));

    // Test 3: Read data back
    print('  📖 Reading tensor data...');
    final data = await testTensor.getData() as Float32List;
    print('  📊 Tensor data: ${data.take(4).toList()}');

    // Test 4: Simple scalar operation
    print('  🔢 Testing scalar multiplication...');
    final result = await testTensor.multiplyScalar(2.0);
    final resultData = await result.getData() as Float32List;
    print('  📊 Result data: ${resultData.take(4).toList()}');

    // Clean up
    testTensor.destroy();
    result.destroy();
  } catch (e, stackTrace) {
    print('❌ Basic tensor test failed: $e');
    rethrow;
  }
}

/// Safer version of the signal processing example
Future<void> runSafeTensorExample() async {
  print('\n🛡️  Safe Tensor Processing Example');
  print('-' * 50);

  final pipeline = Pipeline(
    id: 'safe_processing',
    enableBenchmarking: true,
    debugEnabled: true,
  );

  // Use only the safest operations
  final safeStage = StageBuilder('safe_operations')
      .op('multiplyScalar', arguments: [1.1]) // Very safe scalar op
      .op('abs') // Safe unary operation
      .build();

  pipeline.addStage(safeStage);

  // Create a very simple stream
  final testStream = SafeMockStream(id: 'test_stream', name: 'Test Stream');
  pipeline.addStream(testStream);

  // Set up error monitoring
  pipeline.addEventListener(PipelineEventType.operationError, (event) {
    print('🚨 Pipeline error: ${event}');
  });

  pipeline.addEventListener(PipelineEventType.stageError, (event) {
    final stageError = event as StageErrorEvent;
    print('🚨 Stage error in ${stageError.stageId}: ${stageError.error}');
  });

  try {
    print('🚀 Starting safe pipeline...');
    await pipeline.start();

    print('🧪 Processing safe test data...');
    for (int i = 0; i < 3; i++) {
      print('  📊 Processing buffer ${i + 1}/3...');
      await testStream.generateSafeBuffer();
      await Future.delayed(Duration(milliseconds: 100));
      print('  ✅ Buffer ${i + 1} completed');
    }

    await Future.delayed(Duration(milliseconds: 200));
    print('✅ Safe processing completed successfully');

    // Check outputs
    final output = pipeline.getStageOutput('safe_operations', 'test_stream');
    if (output != null && output.containsKey('input')) {
      print('📊 Final output shape: ${output['input']!.shape}');
    }
  } catch (e, stackTrace) {
    print('❌ Safe pipeline failed: $e');
    print('Stack trace: $stackTrace');
  } finally {
    await pipeline.stop();
    pipeline.dispose();
    testStream.dispose();
  }
}

/// Basic signal processing pipeline using safer operations
Future<void> runBasicSignalProcessingExample() async {
  print('\n📊 Example 1: Basic Signal Processing Pipeline');
  print('-' * 50);

  final pipeline = Pipeline(
    id: 'basic_signal_processing',
    enableBenchmarking: true,
    debugEnabled: true,
  );

  // Simplified generation stage - avoid complex shader functions for now
  final generationStage = StageBuilder('signal_generation')
      .op('multiplyScalar', arguments: [0.8]) // Simple scaling
      .op('addScalar', arguments: [0.1]) // Add DC offset
      .build();

  // Simplified processing stage
  final processingStage = StageBuilder('signal_processing')
      .op('abs') // Rectification
      .op('multiplyScalar', arguments: [1.2]) // Gain
      .build();

  // Analysis stage using only safe operations
  final analysisStage = StageBuilder('frequency_analysis')
      .op('abs') // Get absolute values
      .op('multiplyScalar', arguments: [2.0]) // Scale up
      .build();

  // Add stages with per-stream processing (default behavior)
  pipeline.addStage(generationStage);
  pipeline.addStage(processingStage);
  pipeline.addStage(analysisStage);

  // Create safer coordinate stream
  final coordinateStream = SafeCoordinateStream(
    id: 'coordinates',
    bufferSize: 256, // Smaller buffer size for safety
  );
  pipeline.addStream(coordinateStream);

  // Set up benchmarking
  final benchmarkResults = <String, List<Duration>>{};
  pipeline.addEventListener(PipelineEventType.stageComplete, (event) {
    final stageEvent = event as StageCompleteEvent;
    benchmarkResults
        .putIfAbsent(stageEvent.stageId, () => [])
        .add(stageEvent.processingTime);
  });

  try {
    await pipeline.start();

    print('🎵 Processing 5 signal buffers...');
    for (int i = 0; i < 5; i++) {
      print('  Processing buffer ${i + 1}/5...');
      await coordinateStream.generateNextBuffer();
      await Future.delayed(Duration(milliseconds: 100));
    }

    await Future.delayed(Duration(milliseconds: 200));

    // Print results
    printBenchmarkResults('Basic Processing', benchmarkResults);
    await validateOutput(pipeline, 'frequency_analysis', 'audio_1');
  } finally {
    await pipeline.stop();
    pipeline.dispose();
    coordinateStream.dispose();
  }
}

/// Very safe mock stream that creates minimal tensors
class SafeMockStream extends MediaStream {
  SafeMockStream({required String id, required String name})
    : super(
        id: id,
        type: MediaStreamType.audio,
        deviceId: 'safe_mock',
        name: name,
        color: 0xFF0000FF,
        context: 'safe_test',
      );

  Future<void> generateSafeBuffer() async {
    try {
      print('    🔧 Creating tensor...');
      final shape = [8]; // Very small tensor
      final tensor = await Tensor.create(shape);

      print('    📝 Setting safe data...');
      final data = Float32List(8);
      for (int i = 0; i < 8; i++) {
        data[i] = (i + 1).toDouble() * 0.1; // Simple, safe values
      }

      await tensor.write(data);
      print('    📡 Notifying tensor update...');
      notifyTensorUpdate(tensor);
      print('    ✅ Safe buffer generated');
    } catch (e, stackTrace) {
      print('    ❌ Failed to generate safe buffer: $e');
      print('    Stack trace: $stackTrace');
      rethrow;
    }
  }

  @override
  Future<void> start() async {
    _isActive = true;
    print('📐 Safe stream started');
  }

  @override
  Future<void> stop() async {
    _isActive = false;
  }

  @override
  void dispose() {
    _isActive = false;
  }

  bool _isActive = false;
  @override
  bool get isActive => _isActive;
}

/// Safer coordinate stream with minimal operations
class SafeCoordinateStream extends MediaStream {
  final int bufferSize;
  int _bufferCount = 0;

  SafeCoordinateStream({required String id, required this.bufferSize})
    : super(
        id: id,
        type: MediaStreamType.audio,
        deviceId: 'safe_coordinate_generator',
        name: 'Safe Coordinates',
        color: 0xFF00FF00,
        context: 'safe_coordinate',
      );

  Future<void> generateNextBuffer() async {
    try {
      print('    🔧 Creating coordinate tensor (size: $bufferSize)...');
      final shape = [bufferSize];
      final tensor = await Tensor.create(shape);

      print('    📝 Generating coordinate data...');
      final data = Float32List(bufferSize);
      final baseOffset = _bufferCount * bufferSize;

      for (int i = 0; i < bufferSize; i++) {
        // Simple, safe coordinate generation
        data[i] =
            ((baseOffset + i) % 1000).toDouble() / 1000.0; // Normalized 0-1
      }

      print('    📡 Setting tensor data...');
      await tensor.write(data);

      print('    📨 Notifying update...');
      _bufferCount++;
      notifyTensorUpdate(tensor);

      print('    ✅ Coordinate buffer ${_bufferCount} generated successfully');
    } catch (e, stackTrace) {
      print('    ❌ Failed to generate coordinate buffer: $e');
      print('    Stack trace: $stackTrace');
      rethrow;
    }
  }

  @override
  Future<void> start() async {
    _isActive = true;
    print('📐 Safe coordinate stream started');
  }

  @override
  Future<void> stop() async {
    _isActive = false;
  }

  @override
  void dispose() {
    _isActive = false;
  }

  bool _isActive = false;
  @override
  bool get isActive => _isActive;
}

/// Print benchmark results
void printBenchmarkResults(
  String testName,
  Map<String, List<Duration>> results,
) {
  print('\n📈 $testName Benchmark Results:');
  print('=' * 40);

  for (final entry in results.entries) {
    final stageName = entry.key;
    final timings = entry.value;

    if (timings.isEmpty) continue;

    final microseconds = timings.map((d) => d.inMicroseconds).toList();
    final avgTime = microseconds.reduce((a, b) => a + b) / microseconds.length;
    final minTime = microseconds.reduce(math.min);
    final maxTime = microseconds.reduce(math.max);

    print('🔹 $stageName:');
    print('   Avg: ${avgTime.toStringAsFixed(1)}μs');
    print('   Min: ${minTime}μs, Max: ${maxTime}μs');
    print('   Samples: ${timings.length}');
  }
}

/// Selective processing example
Future<void> runSelectiveProcessingExample() async {
  print('\n🎯 Example 3: Selective Stream Processing');
  print('-' * 50);

  final pipeline = Pipeline(
    id: 'selective_processor',
    enableBenchmarking: true,
    debugEnabled: true,
  );

  // Stage 1: Process all streams
  final preProcessStage = StageBuilder('preprocess')
      .op('multiplyScalar', arguments: [0.9]) // Slight attenuation
      .build();

  // Stage 2: Only process audio streams (selective)
  final audioEnhanceStage = StageBuilder('audio_enhance')
      .executeShaderFunction(
        '''
        // Audio-specific enhancement
        let enhanced = input_val * ENHANCEMENT_FACTOR;
        return tanh(enhanced); // Soft saturation
        ''',
        functionParameters: {'ENHANCEMENT_FACTOR': 'param:enhance_factor'},
      )
      .withParameter(
        Parameter.stageFloat(
          'enhance_factor',
          defaultValue: 1.3,
          min: 0.8,
          max: 2.0,
          description: 'Audio enhancement factor',
        ),
      )
      .build();

  // Stage 3: Only process video streams (selective)
  final videoProcessStage = StageBuilder('video_process')
      .executeShaderFunction(
        '''
        // Video-specific processing (brightness adjustment)
        return clamp(input_val * BRIGHTNESS_FACTOR, 0.0, 1.0);
        ''',
        functionParameters: {'BRIGHTNESS_FACTOR': 'param:brightness'},
      )
      .withParameter(
        Parameter.stageFloat(
          'brightness',
          defaultValue: 1.1,
          min: 0.5,
          max: 2.0,
          description: 'Video brightness factor',
        ),
      )
      .build();

  // Stage 4: Final processing on all remaining streams
  final finalStage = StageBuilder('final_process')
      .op('abs') // Ensure positive values
      .build();

  // Add stages with different routing
  pipeline.addStage(preProcessStage); // Process all streams

  // Selective processing
  pipeline.addSelectiveStage(
    audioEnhanceStage,
    streams: {'audio1', 'audio2'}, // Only audio streams
  );

  pipeline.addSelectiveStage(
    videoProcessStage,
    streams: {'video1'}, // Only video streams
  );

  pipeline.addStage(finalStage); // Process all remaining streams

  // Create mixed media streams
  final audio1Stream = MockAudioStream(
    id: 'audio1',
    name: 'Audio Channel 1',
    baseFrequency: 440.0,
    amplitude: 0.7,
  );

  final audio2Stream = MockAudioStream(
    id: 'audio2',
    name: 'Audio Channel 2',
    baseFrequency: 660.0,
    amplitude: 0.5,
  );

  final video1Stream = MockVideoStream(
    id: 'video1',
    name: 'Video Feed',
    baseValue: 0.5,
    amplitude: 0.3,
  );

  pipeline.addStream(audio1Stream);
  pipeline.addStream(audio2Stream);
  pipeline.addStream(video1Stream);

  // Benchmarking
  final benchmarkResults = <String, List<Duration>>{};
  pipeline.addEventListener(PipelineEventType.stageComplete, (event) {
    final stageEvent = event as StageCompleteEvent;
    benchmarkResults
        .putIfAbsent(stageEvent.stageId, () => [])
        .add(stageEvent.processingTime);
  });

  await pipeline.start();

  print('🎭 Processing selective streams...');
  for (int i = 0; i < 6; i++) {
    await Future.wait([
      audio1Stream.generateNextBuffer(),
      audio2Stream.generateNextBuffer(),
      video1Stream.generateNextBuffer(),
    ]);
    await Future.delayed(Duration(milliseconds: 80));
  }

  await Future.delayed(Duration(milliseconds: 200));

  printBenchmarkResults('Selective Processing', benchmarkResults);

  await pipeline.debugPrintAllOutputs();

  // Validate that each stream was processed correctly
  await validateOutput(pipeline, 'final', 'audio1');
  await validateOutput(pipeline, 'final', 'audio2');
  await validateOutput(pipeline, 'final', 'video1');

  await pipeline.stop();
  pipeline.dispose();
  audio1Stream.dispose();
  audio2Stream.dispose();
  video1Stream.dispose();
}

/// Multi-stream merging example
Future<void> runMultiStreamMergingExample() async {
  print('\n🔄 Example 2: Multi-Stream Merging Pipeline');
  print('-' * 50);

  final pipeline = Pipeline(
    id: 'multi_stream_mixer',
    enableBenchmarking: true,
    debugEnabled: true,
  );

  // Stage 1: Individual stream normalization (per-stream processing)
  final normalizeStage = StageBuilder('normalize')
      .op('divideScalar', arguments: [32768.0]) // Normalize 16-bit audio
      .build();

  // Stage 2: Stream merging using weighted sum
  final mixerStage = StageBuilder('audio_mixer').executeShader('''
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_0)) { return; }
    output_0[i] = input_0[i]; // Just copy input to output
}
''').build();

  // Stage 3: Post-processing (processes merged stream)
  final limiterStage = StageBuilder('limiter')
      .executeShaderFunction(
        '''
        // Soft limiting with tanh
        return tanh(input_val * DRIVE) / DRIVE;
        ''',
        functionName: 'applyLimiter',
        functionParameters: {'DRIVE': 'param:limiter_drive'},
      )
      .withParameter(
        Parameter.stageFloat(
          'limiter_drive',
          defaultValue: 1.2,
          min: 0.5,
          max: 3.0,
          description: 'Limiter drive amount',
        ),
      )
      .build();

  // Add stages with different processing modes
  // pipeline.addStage(normalizeStage); // Per-stream processing

  // Merge streams with weighted sum
  pipeline.addMergeStage(mixerStage);

  pipeline.addStage(limiterStage); // Processes merged result

  // Create multiple audio streams
  final micStream = MockAudioStream(
    id: 'microphone',
    name: 'Microphone',
    baseFrequency: 440.0, // A4
    amplitude: 0.8,
  );

  final musicStream = MockAudioStream(
    id: 'music',
    name: 'Background Music',
    baseFrequency: 220.0, // A3
    amplitude: 0.6,
  );

  final effectsStream = MockAudioStream(
    id: 'effects',
    name: 'Sound Effects',
    baseFrequency: 880.0, // A5
    amplitude: 0.4,
  );

  pipeline.addStream(micStream);
  pipeline.addStream(musicStream);
  pipeline.addStream(effectsStream);

  // Benchmarking
  final benchmarkResults = <String, List<Duration>>{};
  pipeline.addEventListener(PipelineEventType.stageComplete, (event) {
    final stageEvent = event as StageCompleteEvent;
    benchmarkResults
        .putIfAbsent(stageEvent.stageId, () => [])
        .add(stageEvent.processingTime);
  });

  await pipeline.start();

  print('🎵 Processing mixed audio streams...');
  for (int i = 0; i < 8; i++) {
    // Generate audio on all streams
    await Future.wait([
      micStream.generateNextBuffer(),
      musicStream.generateNextBuffer(),
      effectsStream.generateNextBuffer(),
    ]);
    await Future.delayed(Duration(milliseconds: 60));
  }

  await Future.delayed(Duration(milliseconds: 200));

  printBenchmarkResults('Stream Merging', benchmarkResults);
  await validateOutput(pipeline, 'limiter', 'merged');

  await pipeline.stop();
  pipeline.dispose();
  micStream.dispose();
  musicStream.dispose();
  effectsStream.dispose();
}

/// Validate pipeline output
Future<void> validateOutput(
  Pipeline pipeline,
  String stageId, [
  String? streamId,
]) async {
  final output = pipeline.getStageOutput(stageId, streamId!);
  if (output == null) {
    print(
      '❌ No output found for stage: $stageId${streamId != null ? ' (stream: $streamId)' : ''}',
    );
    return;
  }

  Tensor? tensor = output['output'] ?? output['input'];
  if (tensor == null) {
    print('❌ No input tensor in output for stage: $stageId');
    return;
  }

  print(
    '✅ Stage $stageId${streamId != null ? ' ($streamId)' : ''} output: ${tensor.shape}',
  );

  // Try to read a few values safely
  try {
    final data = await tensor.getData() as Float32List;
    final sample = data.take(math.min(4, data.length)).toList();
    print('   📊 Sample values: $sample');
  } catch (e) {
    print('   ⚠️  Could not read tensor data: $e');
  }
}

/// Mock audio stream
class MockAudioStream extends MediaStream {
  final double baseFrequency;
  final double amplitude;
  final int bufferSize;
  int _sampleOffset = 0;

  MockAudioStream({
    required String id,
    required String name,
    required this.baseFrequency,
    required this.amplitude,
    this.bufferSize = 1024,
  }) : super(
         id: id,
         type: MediaStreamType.audio,
         deviceId: 'mock_audio_$id',
         name: name,
         color: 0xFF00FF00,
         context: 'mock_audio',
       );

  Future<void> generateNextBuffer() async {
    final shape = [bufferSize];
    final tensor = await Tensor.create(shape);

    // Generate audio samples
    final data = Float32List(bufferSize);
    const sampleRate = 44100.0;

    for (int i = 0; i < bufferSize; i++) {
      final sampleIndex = _sampleOffset + i;
      final time = sampleIndex / sampleRate;
      data[i] = amplitude * math.sin(2.0 * math.pi * baseFrequency * time);
    }

    print('    📡 Writing audio data to tensor (offset: $_sampleOffset)');

    await tensor.write(data);
    _sampleOffset += bufferSize;

    notifyTensorUpdate(tensor);
  }

  @override
  Future<void> start() async {
    _isActive = true;
  }

  @override
  Future<void> stop() async {
    _isActive = false;
  }

  @override
  void dispose() {
    _isActive = false;
  }

  bool _isActive = false;
  @override
  bool get isActive => _isActive;
}

/// Mock video stream
class MockVideoStream extends MediaStream {
  final double baseValue;
  final double amplitude;
  final int bufferSize;
  int _frameCount = 0;

  MockVideoStream({
    required String id,
    required String name,
    required this.baseValue,
    required this.amplitude,
    this.bufferSize = 256,
  }) : super(
         id: id,
         type: MediaStreamType.video,
         deviceId: 'mock_video_$id',
         name: name,
         color: 0xFFFF0000,
         context: 'mock_video',
       );

  Future<void> generateNextBuffer() async {
    final shape = [bufferSize];
    final tensor = await Tensor.create(shape);

    // Generate video-like data (brightness values)
    final data = Float32List(bufferSize);

    for (int i = 0; i < bufferSize; i++) {
      // Simple pattern that changes over time
      final pattern = math.sin((_frameCount * 0.1) + (i * 0.01));
      data[i] = (baseValue + amplitude * pattern).clamp(0.0, 1.0);
    }

    await tensor.write(data);
    _frameCount++;

    notifyTensorUpdate(tensor);
  }

  @override
  Future<void> start() async {
    _isActive = true;
  }

  @override
  Future<void> stop() async {
    _isActive = false;
  }

  @override
  void dispose() {
    _isActive = false;
  }

  bool _isActive = false;
  @override
  bool get isActive => _isActive;
}
