import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_pipeline/src/pipeline_ports.dart';
import 'package:gpu_pipeline/src/stages/pipeline_resource_cache.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'dart:async';
import 'dart:ui';

// --- Mock MediaStream ---
class MockMediaStream extends MediaStream {
  bool _captureStarted = false;
  bool _isActive = false;

  @override
  bool get isActive => _isActive;

  MockMediaStream(
    String id,
    MediaStreamType type, {
    String? deviceId,
    String? name,
    int? color,
    dynamic context,
  }) : super(
         id: id,
         deviceId: deviceId ?? 'mock_device_$id',
         name: name ?? 'Mock ${type.name} Stream',
         color: color ?? const Color(0xFF2196F3).toARGB32(),
         type: type,
         context: context ?? 'mock_context',
       );

  @override
  Future<void> start() async {
    await _startCapture();
    _isActive = true;
  }

  @override
  Future<void> stop() async {
    await _stopCapture();
    _isActive = false;
  }

  Future<void> _startCapture() async {
    _captureStarted = true;
    await Future.delayed(const Duration(milliseconds: 10));
  }

  Future<void> _stopCapture() async {
    _captureStarted = false;
    await Future.delayed(const Duration(milliseconds: 5));
  }

  void simulateTensorUpdate(Tensor tensor) {
    if (_captureStarted && _isActive) {
      print(
        '🔍 MockMediaStream $id: Simulating tensor update with shape ${tensor.shape}',
      );
      notifyTensorUpdate(tensor);
    } else {
      print(
        '⚠️ MockMediaStream $id: Cannot simulate tensor update - not started or active',
      );
    }
  }

  bool get isCaptureStarted => _captureStarted;
}

// --- Mock TensorOperation ---
class MockTensorOperation extends TensorOperation {
  final String operation;
  final double factor;

  MockTensorOperation({
    required this.operation,
    this.factor = 1.0,
    String inputKey = 'input',
    String outputKey = 'output',
    List<dynamic> arguments = const [],
    Map<String, dynamic> namedArguments = const {},
  }) : super(
         operationId: 'mock_${operation}_operation',
         inputKeys: [inputKey],
         outputKeys: [outputKey],
         operation: operation,
         arguments: arguments.asMap().map((i, v) => MapEntry('arg$i', v)),
       );

  static MockTensorOperation fromConfig(Map<String, dynamic> config) {
    return MockTensorOperation(
      operation: config['operation'] as String,
      factor: (config['factor'] as num?)?.toDouble() ?? 1.0,
      inputKey: config['inputKey'] as String? ?? 'input',
      outputKey: config['outputKey'] as String? ?? 'output',
      arguments: config['arguments'] as List<dynamic>? ?? const [],
      namedArguments:
          config['namedArguments'] as Map<String, dynamic>? ?? const {},
    );
  }

  @override
  Future<Map<String, Tensor>> execute(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache cache,
  ) async {
    final input = inputs[inputKeys.first];
    if (input == null)
      throw ArgumentError('Input tensor not found for key: ${inputKeys.first}');
    Tensor output;
    switch (operation) {
      case 'multiply':
        output = await input.multiplyScalar(factor);
        break;
      case 'add':
        output = await input.addScalar(factor);
        break;
      case 'normalize':
        output = await input.multiplyScalar(1.0 / 255.0);
        break;
      case 'resize':
        final newShape =
            parameters['output_shape'] as List<int>? ?? input.shape;
        output = await Tensor.create(newShape, gpu: input.gpu);
        break;
      default:
        output = await Tensor.create(input.shape, gpu: input.gpu);
    }
    return {outputKeys.first: output};
  }
}

// --- Mock ShaderOperation ---
class MockShaderOperation extends ShaderStageOperation {
  final String mockShaderCode;
  final Duration? processingDelay;

  MockShaderOperation({
    required String operationId,
    required this.mockShaderCode,
    this.processingDelay,
    List<String> inputKeys = const ['input'],
    List<String> outputKeys = const ['output'],
    Map<String, OutputPort> outputPorts = const {
      'output': FixedOutputPort(name: 'output', shape: [1]),
    },
    Map<String, dynamic> arguments = const {},
  }) : super(
         operationId: operationId,
         inputKeys: inputKeys,
         outputKeys: outputKeys,
         shaderTemplate: mockShaderCode,
         outputPorts: outputPorts,
         arguments: arguments,
       );

  @override
  Future<Map<String, Tensor>> execute(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache cache,
  ) async {
    if (processingDelay != null) await Future.delayed(processingDelay!);
    final input = inputs[inputKeys.first];
    if (input == null)
      throw ArgumentError('Input tensor not found for key: ${inputKeys.first}');
    final output = await Tensor.create(input.shape, gpu: input.gpu);
    print(
      '🖥️ MockShaderOperation $operationId: Executed shader on ${input.shape}',
    );
    return {outputKeys.first: output};
  }
}

// --- Mock PipelineStage ---
class MockPipelineStage extends PipelineStage {
  final Duration? processingDelay;
  final bool shouldFail;
  final bool failDuringProcessing;
  final Map<String, dynamic> mockParameters;

  MockPipelineStage({
    required String stageId,
    this.processingDelay,
    this.shouldFail = false,
    this.failDuringProcessing = false,
    this.mockParameters = const {},
    bool isPersistent = false,
  }) : super(stageId: stageId, isPersistent: isPersistent) {
    for (final entry in mockParameters.entries) {
      addParameter(
        Parameter.stageFloat(
          entry.key,
          defaultValue: (entry.value as num).toDouble(),
          description: 'Mock parameter ${entry.key}',
        ),
      );
    }
  }

  @override
  Future<void> initializeStage() async {
    if (processingDelay != null) await Future.delayed(processingDelay!);
    if (shouldFail && !failDuringProcessing)
      throw Exception('Mock stage initialization failed');
    print(
      '🔧 MockPipelineStage $stageId: Initialized with ${parameterDefinitions.length} parameters',
    );
  }

  @override
  Future<Map<String, Tensor>> process(
    Map<String, Tensor> inputs, [
    Map<String, Tensor>? outputs,
  ]) async {
    if (shouldFail && failDuringProcessing)
      throw Exception('Mock stage processing failed');
    if (processingDelay != null) await Future.delayed(processingDelay!);
    final input = inputs['input'] ?? inputs.values.first;
    final output = await Tensor.create(input.shape, gpu: input.gpu);

    // Echo input payload into output so tests can assert content.
    try {
      final inData = await input.getData();
      await output.write(inData);
    } catch (_) {
      // If dtype mismatch or not readable, keep zeroed output.
    }

    print(
      '🔧 MockPipelineStage $stageId: Processing tensor ${input.shape} -> ${output.shape}',
    );
    if (outputs != null && outputs.isNotEmpty) {
      final outputKey = outputs.keys.first;
      return {outputKey: output};
    }
    return {'output': output};
  }

  @override
  Future<Map<String, Tensor>> handleError(
    Exception error,
    Map<String, Tensor> inputs,
  ) async {
    print('🚨 MockPipelineStage $stageId: Handling error: $error');
    if (shouldFail) return {};
    final recoveryTensor = await Tensor.create([1]);
    return {'output': recoveryTensor};
  }

  @override
  bool shouldExecute(Map<String, Tensor> inputs, Map<String, dynamic> metrics) {
    final executionCount = metrics['execution_count'] as int? ?? 0;
    return executionCount % 5 != 4;
  }
}

// --- MockEnhancedStage ---
class MockEnhancedStage extends PipelineStage {
  final Duration? processingDelay;
  final bool shouldFail;

  MockEnhancedStage({
    required String stageId,
    this.processingDelay,
    this.shouldFail = false,
  }) : super(stageId: stageId) {
    addInputPort(
      FlexibleInputPort(name: 'input', description: 'Main input tensor'),
    );
    addOutputPort(
      FixedOutputPort(
        name: 'output',
        description: 'Processed output tensor',
        shape: [224, 224, 3],
        format: 'processed_tensor',
      ),
    );
    addParameter(
      Parameter.stageFloat(
        'gain',
        defaultValue: 1.0,
        min: 0.0,
        max: 10.0,
        description: 'Processing gain factor',
      ),
    );
    addParameter(
      Parameter.stageInt(
        'iterations',
        defaultValue: 1,
        min: 1,
        max: 100,
        description: 'Number of processing iterations',
      ),
    );
    addAsset(
      StaticAsset.weights('mock_weights', path: 'assets/mock_weights.bin'),
    );
  }

  @override
  Future<Map<String, Tensor>> process(
    Map<String, Tensor> inputs, [
    Map<String, Tensor>? outputs,
  ]) async {
    if (shouldFail) throw Exception('Mock enhanced stage processing failed');
    if (processingDelay != null) await Future.delayed(processingDelay!);
    final input = inputs['input'];
    if (input == null) throw ArgumentError('Input tensor not found');
    final gain = getPropertyValue<double>('gain') ?? 1.0;
    final iterations = getPropertyValue<int>('iterations') ?? 1;
    print(
      '🎯 MockEnhancedStage $stageId: Processing with gain=$gain, iterations=$iterations',
    );
    var processedTensor = input;
    for (int i = 0; i < iterations; i++) {
      processedTensor = await processedTensor.multiplyScalar(gain);
    }
    final outputPort = outputPorts.first as FixedOutputPort;
    final outputTensor = await Tensor.create(outputPort.shape, gpu: input.gpu);
    return {'output': outputTensor};
  }
}

// --- MockDynamicStage ---
class MockDynamicStage extends DynamicStage {
  MockDynamicStage({
    required String stageId,
    List<StageOperation>? operations,
    Map<String, Parameter>? parameters,
  }) : super(
         stageId: stageId,
         operations: operations ?? _createDefaultOperations(),
         parameterDefinitions: parameters ?? _createDefaultParameters(),
       );

  static List<StageOperation> _createDefaultOperations() => [
    MockTensorOperation(
      operation: 'normalize',
      inputKey: 'input',
      outputKey: 'normalized',
    ),
    MockTensorOperation(
      operation: 'multiply',
      factor: 2.0,
      inputKey: 'normalized',
      outputKey: 'amplified',
    ),
    MockTensorOperation(
      operation: 'add',
      factor: 0.1,
      inputKey: 'amplified',
      outputKey: 'output',
    ),
  ];

  static Map<String, Parameter> _createDefaultParameters() => {
    'gain': Parameter.stageFloat(
      'gain',
      defaultValue: 1.0,
      min: 0.0,
      max: 5.0,
      description: 'Processing gain',
    ),
    'offset': Parameter.stageFloat(
      'offset',
      defaultValue: 0.0,
      min: -1.0,
      max: 1.0,
      description: 'DC offset',
    ),
  };
}

// --- MockAssetManager ---
class MockAssetManager extends AssetManager {
  final Map<String, Tensor> _mockTensors = {};
  final Map<String, Map<String, dynamic>> _mockConfigs = {};

  MockAssetManager() {
    _mockConfigs['mock_config'] = {
      'version': '1.0',
      'settings': {'enabled': true, 'threshold': 0.5},
    };
  }

  void addMockTensor(String name, Tensor tensor) => _mockTensors[name] = tensor;
  void addMockConfig(String name, Map<String, dynamic> config) =>
      _mockConfigs[name] = config;

  @override
  Future<void> preload(StaticAsset asset) async {
    await Future.delayed(const Duration(milliseconds: 10));
    print('📦 MockAssetManager: Preloaded ${asset.name}');
  }

  @override
  Tensor? getPreloadedTensor(String name) => _mockTensors[name];

  @override
  Map<String, dynamic>? getPreloadedConfig(String name) => _mockConfigs[name];
}
