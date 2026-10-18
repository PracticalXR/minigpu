import 'package:gpu_pipeline/src/operation/cpu_stage_operation.dart';
import 'package:gpu_pipeline/src/operation/stage_operation.dart';
import 'package:gpu_pipeline/src/operation/tensor_stage_operation.dart';
import 'package:gpu_pipeline/src/operation/shader_stage_operation.dart';
import 'package:gpu_pipeline/src/pipeline_ports.dart';
import 'package:gpu_pipeline/src/pipeline_properties.dart';
import 'package:gpu_pipeline/src/pipeline_assets.dart';
import 'package:gpu_pipeline/src/pipeline_resources.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'pipeline_dynamic_stage.dart';

class StageBuilder {
  final String _stageId;
  final List<StageOperation> _operations = [];
  final List<Parameter> _parameters = [];
  final List<InputPort> _inputPorts = [];
  final List<OutputPort> _outputPorts = [];
  final List<StaticAsset> _assets = [];

  ResourceRequirements? _resourceRequirements;
  MemoryStrategy? _memoryStrategy;
  RealTimeConstraints? _realTimeConstraints;

  int _tempCounter = 0;

  // Track the current output key for chaining
  String? _currentOutputKey;

  StageBuilder(this._stageId) {
    // Initialize tensor operation registry when builder is created
    TensorOperationRegistry.ensureInitialized();
  }

  // ============ I/O PORT METHODS ============

  /// Add flexible input port
  StageBuilder withFlexibleInput(
    String name, {
    String description = '',
    bool isRequired = true,
    int? minDimensions,
    int? maxDimensions,
  }) {
    _inputPorts.add(
      FlexibleInputPort(
        name: name,
        description: description,
        isRequired: isRequired,
        minDimensions: minDimensions,
        maxDimensions: maxDimensions,
      ),
    );
    return this;
  }

  /// Add audio input port
  StageBuilder withAudioInput(
    String name, {
    String description = '',
    bool isRequired = true,
    int? minSamples,
    int? maxSamples,
    int? expectedChannels,
  }) {
    _inputPorts.add(
      AudioInputPort(
        name: name,
        description: description,
        isRequired: isRequired,
        minSamples: minSamples,
        maxSamples: maxSamples,
        expectedChannels: expectedChannels,
      ),
    );
    return this;
  }

  /// Add video input port
  StageBuilder withVideoInput(
    String name, {
    String description = '',
    bool isRequired = true,
    int? minWidth,
    int? maxWidth,
    int? minHeight,
    int? maxHeight,
  }) {
    _inputPorts.add(
      VideoInputPort(
        name: name,
        description: description,
        isRequired: isRequired,
        minWidth: minWidth,
        maxWidth: maxWidth,
        minHeight: minHeight,
        maxHeight: maxHeight,
      ),
    );
    return this;
  }

  /// Add exact shape input port
  StageBuilder withExactInput(
    String name, {
    required List<int> shape,
    String description = '',
  }) {
    _inputPorts.add(
      ExactInputPort(
        name: name,
        expectedShape: shape,
        description: description,
      ),
    );
    return this;
  }

  /// Add custom input port
  StageBuilder withCustomInput(
    String name, {
    required ValidationResult Function(Tensor?) validator,
    String description = '',
    bool isRequired = true,
  }) {
    _inputPorts.add(
      CustomInputPort(
        name: name,
        validator: validator, // Use ValidationResult directly
        description: description,
        isRequired: isRequired,
      ),
    );
    return this;
  }

  /// Add static asset input
  StageBuilder withStaticInput(
    String name, {
    required String assetPath,
    String description = '',
  }) {
    _inputPorts.add(
      StaticInputPort(
        name: name,
        assetPath: assetPath,
        description: description,
      ),
    );
    return this;
  }

  /// Add generated input (creates noise, patterns, etc.)
  StageBuilder withGeneratedInput(
    String name, {
    required List<int> shape,
    required String type,
    String description = '',
    Map<String, dynamic> generatorParams = const {},
  }) {
    // Create a generated asset that produces the input
    final asset = StaticAsset.generated(
      name: '${name}_generator',
      type: AssetType.binary,
      generatorParams: {
        'shape': shape,
        'generator_type': type,
        ...generatorParams,
      },
    );

    _assets.add(asset);

    // Add generated input port (not static input port)
    _inputPorts.add(
      GeneratedInputPort(
        name: name,
        shape: shape,
        generatorType: type,
        generatorParameters: generatorParams,
        description: description.isEmpty
            ? 'Generated $type input'
            : description,
      ),
    );
    return this;
  }

  /// Add fixed size output port
  StageBuilder withFixedOutput(
    String name, {
    required List<int> shape,
    String description = '',
    String format = 'tensor',
  }) {
    _outputPorts.add(
      FixedOutputPort(
        name: name,
        shape: shape,
        description: description,
        format: format,
      ),
    );
    return this;
  }

  /// Add texture output port - matches TextureOutputPort constructor
  StageBuilder withTextureOutput(
    String name, {
    required int width,
    required int height,
    int channels = 4,
    String description = '',
  }) {
    _outputPorts.add(
      TextureOutputPort(
        name: name,
        width: width,
        height: height,
        channels: channels,
        description: description,
      ),
    );
    return this;
  }

  /// Add dynamic output port with custom sizing
  StageBuilder withDynamicOutput(
    String name, {
    String description = '',
    String format = 'tensor',
    bool persistent = false,
    List<int> Function(Map<String, Tensor>, Map<String, dynamic>)?
    shapeCalculator,
    List<int>? fixedShape,
  }) {
    _outputPorts.add(
      DynamicOutputPort(
        name: name,
        description: description,
        format: format,
        persistent: persistent,
        shapeCalculator: shapeCalculator,
        fixedShape: fixedShape,
      ),
    );
    return this;
  }

  /// Add history buffer output - matches HistoryOutputPort constructor
  StageBuilder withHistoryOutput(
    String name, {
    required int timeSlices,
    String description = '',
  }) {
    _outputPorts.add(
      HistoryOutputPort(
        name: name,
        timeSlices: timeSlices,
        description: description,
      ),
    );
    return this;
  }

  /// Add parameter output (computed output based on other parameters)
  StageBuilder withParameterOutput(
    String name, {
    required List<String> dependencies,
    required List<int> Function(Map<String, Tensor>, Map<String, dynamic>)
    calculator,
    String description = '',
    String format = 'tensor',
  }) {
    _outputPorts.add(
      ParameterOutputPort(
        name: name,
        parameterDependencies: dependencies,
        shapeCalculator: calculator,
        description: description,
        format: format,
      ),
    );
    return this;
  }

  // ============ PARAMETER METHODS ============

  /// Add parameter directly using Parameter class
  StageBuilder withParameter(Parameter parameter) {
    _parameters.add(parameter);
    return this;
  }

  // ============ ASSET METHODS - Use actual StaticAsset factory methods ============

  /// Add weights asset using StaticAsset.weights factory
  StageBuilder withWeights(
    String name, {
    required String path,
    bool isRequired = true,
  }) {
    _assets.add(StaticAsset.weights(name, path: path, isRequired: isRequired));
    return this;
  }

  /// Add texture asset using StaticAsset.texture factory
  StageBuilder withTexture(
    String name, {
    required String path,
    bool isRequired = true,
  }) {
    _assets.add(
      StaticAsset.texture(name: name, path: path, isRequired: isRequired),
    );
    return this;
  }

  /// Add config asset using StaticAsset.config factory
  StageBuilder withConfig(
    String name, {
    required String path,
    bool isRequired = true,
  }) {
    _assets.add(
      StaticAsset.config(name: name, path: path, isRequired: isRequired),
    );
    return this;
  }

  /// Add lookup table asset using StaticAsset.lookupTable factory
  StageBuilder withLookupTable(
    String name, {
    required String path,
    bool required = true,
  }) {
    _assets.add(
      StaticAsset.lookupTable(name: name, path: path, isRequired: required),
    );
    return this;
  }

  /// Add generated asset using StaticAsset.generated factory
  StageBuilder withGeneratedAsset(
    String name, {
    required AssetType type,
    Map<String, dynamic> generatorParams = const {},
    bool isRequired = true,
  }) {
    _assets.add(
      StaticAsset.generated(
        name: name,
        type: type,
        generatorParams: generatorParams,
        isRequired: isRequired,
      ),
    );
    return this;
  }

  /// Add asset directly
  StageBuilder withAsset({required StaticAsset asset}) {
    _assets.add(asset);
    return this;
  }

  // ============ RESOURCE METHODS ============

  /// Set resource requirements
  StageBuilder withResourceRequirements({
    required ResourceRequirements requirements,
  }) {
    _resourceRequirements = requirements;
    return this;
  }

  /// Set memory strategy
  StageBuilder withMemoryStrategy({required MemoryStrategy strategy}) {
    _memoryStrategy = strategy;
    return this;
  }

  /// Set real-time constraints
  StageBuilder withRealTimeConstraints({
    required RealTimeConstraints constraints,
  }) {
    _realTimeConstraints = constraints;
    return this;
  }

  /// Set GPU affinity - create ResourceRequirements if needed
  StageBuilder withGPUAffinity({required List<int> gpuIds}) {
    // This would need to match the actual ResourceRequirements constructor
    if (_resourceRequirements == null) {
      _resourceRequirements = ResourceRequirements(preferredGPUs: gpuIds);
    } else {
      // Update existing requirements with GPU affinity
      _resourceRequirements = _resourceRequirements!.copyWith(
        preferredGPUs: gpuIds,
      );
    }
    return this;
  }

  // ============ OPERATION METHODS ============

  /// Add a tensor operation using the simplified API
  StageBuilder op(
    String operationName, {
    List<dynamic>? arguments,
    Map<String, dynamic>? namedArguments,
    String? inputKey,
    String? outputKey,
  }) {
    final opId = '${_stageId}_op_${_operations.length}';

    // Smart input key resolution
    String resolvedInputKey;
    if (inputKey != null) {
      resolvedInputKey = inputKey;
    } else if (_currentOutputKey != null) {
      resolvedInputKey = _currentOutputKey!;
    } else {
      resolvedInputKey = 'input';
    }

    // Smart output key resolution
    String resolvedOutputKey;
    if (outputKey != null) {
      resolvedOutputKey = outputKey;
    } else if (_outputPorts.any((port) => port is DynamicOutputPort)) {
      resolvedOutputKey = 'final_output';
    } else {
      resolvedOutputKey = 'temp_${_operations.length}';
    }

    // Update tracking for next operation
    _currentOutputKey = resolvedOutputKey;

    // Convert arguments list to map format for TensorOperation constructor
    final argumentsMap = <String, dynamic>{};
    if (arguments != null && arguments.isNotEmpty) {
      argumentsMap['args'] = arguments;
    }
    // Add any named arguments directly to the map
    argumentsMap.addAll(namedArguments ?? {});

    final operation = TensorOperation(
      operationId: opId,
      operation: operationName,
      arguments: argumentsMap, // Map format
      namedArguments: <String, dynamic>{}, // Empty since we merged them above
      inputKeys: [resolvedInputKey],
      outputKeys: [resolvedOutputKey],
    );

    _operations.add(operation);
    return this;
  }

  /// Execute custom CPU processing function
  StageBuilder executeCPU(
    CPUProcessor processor, {
    String? inputKey,
    String? outputKey,
  }) {
    final opId = '${_stageId}_cpu_${_operations.length}';

    // Smart input key resolution
    String resolvedInputKey;
    if (inputKey != null) {
      resolvedInputKey = inputKey;
    } else if (_currentOutputKey != null) {
      resolvedInputKey = _currentOutputKey!;
    } else {
      resolvedInputKey = 'input';
    }

    // Smart output key resolution
    String resolvedOutputKey;
    if (outputKey != null) {
      resolvedOutputKey = outputKey;
    } else {
      resolvedOutputKey = 'temp_${_operations.length}';
    }

    // Update tracking for next operation
    _currentOutputKey = resolvedOutputKey;

    final operation = CPUStageOperation(
      operationId: opId,
      processor: processor,
      inputKeys: [resolvedInputKey],
      outputKeys: [resolvedOutputKey],
    );

    _operations.add(operation);
    return this;
  }

  /// Execute dynamic CPU code (for future implementation)
  StageBuilder executeDynamicCPU(
    String dartCode, {
    bool isolateEnabled = false,
    String? inputKey,
    String? outputKey,
  }) {
    final actualInputKey = inputKey ?? _currentOutputKey;
    final actualOutputKey = outputKey ?? _getNextTensorKey();

    final dynamicCpuOp = DynamicCPUOperation(
      operationId: '${_stageId}_dynamic_cpu_${_operations.length}',
      dartCode: dartCode,
      isolateEnabled: isolateEnabled,
      inputKeys: [actualInputKey!],
      outputKeys: [actualOutputKey],
    );

    _operations.add(dynamicCpuOp);
    _currentOutputKey = actualOutputKey;
    return this;
  }

  /// Execute prebuilt CPU operation
  StageBuilder cpuOp(
    String operation, {
    Map<String, dynamic> parameters = const {},
    String? inputKey,
    String? outputKey,
  }) {
    final opId = '${_stageId}_cpu_op_${_operations.length}';

    // Smart input key resolution - handle null _currentOutputKey
    String resolvedInputKey;
    if (inputKey != null) {
      resolvedInputKey = inputKey;
    } else if (_currentOutputKey != null) {
      resolvedInputKey = _currentOutputKey!;
    } else {
      resolvedInputKey = 'input'; // Safe fallback
    }

    // Smart output key resolution
    String resolvedOutputKey;
    if (outputKey != null) {
      resolvedOutputKey = outputKey;
    } else {
      resolvedOutputKey = 'temp_${_operations.length}';
    }

    // Update tracking for next operation
    _currentOutputKey = resolvedOutputKey;

    final prebuiltCpuOp = PrebuiltCPUOperation(
      operationId: opId,
      operation: operation,
      operationParameters: parameters,
      inputKeys: [resolvedInputKey], // Use resolved key
      outputKeys: [resolvedOutputKey], // Use resolved key
    );

    _operations.add(prebuiltCpuOp);
    return this;
  }

  /// Add a complete custom shader
  StageBuilder executeShader(
    String shaderCode, {
    List<String>? inputKeys,
    List<String>? outputKeys,
    Map<String, String> shaderParameters = const {},
    List<int>? workgroupSize,
  }) {
    // Handle input keys - use port names if defined
    List<String> finalInputKeys;
    if (inputKeys != null) {
      finalInputKeys = inputKeys;
    } else if (_inputPorts.isNotEmpty) {
      finalInputKeys = _inputPorts.map((p) => p.name).toList();
    } else if (_currentOutputKey != null) {
      finalInputKeys = [_currentOutputKey!];
    } else {
      finalInputKeys = ['input']; // Default input key
    }

    // Handle output keys - use port names if defined
    List<String> finalOutputKeys;
    if (outputKeys != null) {
      finalOutputKeys = outputKeys;
    } else if (_outputPorts.isNotEmpty) {
      finalOutputKeys = _outputPorts.map((p) => p.name).toList();
    } else {
      finalOutputKeys = [_getNextTensorKey()];
    }

    final shaderOp = ShaderStageOperation(
      operationId: '${_stageId}_shader_${_operations.length}',
      shaderTemplate: shaderCode,
      outputPorts: _outputPorts.asMap().map(
        (index, port) => MapEntry(port.name, port),
      ),
      inputKeys: finalInputKeys,
      outputKeys: finalOutputKeys,
      shaderParameters: shaderParameters,
      workgroupSize: workgroupSize,
    );

    _operations.add(shaderOp);

    // Update current output key for chaining (use first output)
    _currentOutputKey = finalOutputKeys.first;

    return this;
  }

  /// Execute a simple shader function
  StageBuilder executeShaderFunction(
    String functionBody, {
    String functionName = 'process',
    Map<String, dynamic> functionParameters = const {},
    List<int>? workgroupSize,
    String? inputKey,
    String? outputKey,
  }) {
    // Prefer chaining from the previous op's output; then ports; then 'input'
    final actualInputKey =
        inputKey ??
        _currentOutputKey ??
        (_inputPorts.isNotEmpty ? _inputPorts.first.name : 'input');

    final actualOutputKey =
        outputKey ??
        (_outputPorts.isNotEmpty
            ? _outputPorts.first.name
            : _getNextTensorKey());

    final shaderFunctionOp = ShaderFunctionOperation(
      operationId: '${_stageId}_function_${_operations.length}',
      functionBody: functionBody,
      functionName: functionName,
      functionParameters: functionParameters,
      workgroupSize: workgroupSize,
      inputKeys: [actualInputKey],
      outputKeys: [actualOutputKey],
    );

    _operations.add(shaderFunctionOp);
    _currentOutputKey = actualOutputKey;

    return this;
  }

  /// Add merge operation
  StageBuilder merge({
    required List<String> inputKeys,
    String mergeStrategy = 'add',
    String? outputKey,
    Map<String, dynamic> mergeParameters = const {},
  }) {
    // Generate function body for merge operation (uses input0, input1, ...)
    final functionBody = _generateMergeFunctionBody(
      strategy: mergeStrategy,
      inputCount: inputKeys.length,
    );

    final finalOutputKey = outputKey ?? _getNextTensorKey();

    // Build a WGSL shader that declares all inputs and a single output,
    // defines fn shader_function(index, input0, input1, ...),
    // and calls it from main.
    final shaderCode = _buildMergeWGSL(
      inputCount: inputKeys.length,
      functionBody: functionBody,
      workgroupSizeX: 64,
    );

    final shaderOp = ShaderStageOperation(
      operationId: '${_stageId}_merge_${_operations.length}',
      shaderTemplate: shaderCode,
      inputKeys: inputKeys,
      outputKeys: [finalOutputKey],
      outputPorts: _outputPorts.asMap().map(
        (index, port) => MapEntry(port.name, port),
      ),
      // No special ports needed; we bind by input_/output_ tags
    );

    _operations.add(shaderOp);
    _currentOutputKey = finalOutputKey;

    return this;
  }

  /// Add split operation
  StageBuilder split({
    required String inputKey,
    required List<String> outputKeys,
    String splitStrategy = 'copy',
    Map<String, dynamic> splitParameters = const {},
  }) {
    // Generate function body for split operation
    final functionBody = _generateSplitFunctionBody(
      strategy: splitStrategy,
      outputCount: outputKeys.length,
    );

    // Use ShaderFunctionOperation so the function kernel generator can handle multiple outputs
    final shaderOp = ShaderFunctionOperation(
      operationId: '${_stageId}_split_${_operations.length}',
      functionBody: functionBody,
      inputKeys: [inputKey],
      outputKeys: outputKeys,
      functionParameters: splitParameters,
    );

    _operations.add(shaderOp);
    _currentOutputKey = outputKeys.first;

    return this;
  }

  // ============ BUILD METHODS ============

  /// Build the enhanced DynamicStage
  DynamicStage build() {
    if (_operations.isEmpty) {
      throw StateError('Stage must have at least one operation');
    }

    // Create parameter definitions map
    final parameterDefinitions = <String, Parameter>{};
    for (final param in _parameters) {
      parameterDefinitions[param.name] = param;
    }

    // Create the DynamicStage
    final stage = DynamicStage(
      stageId: _stageId,
      operations: List.unmodifiable(_operations),
      parameterDefinitions: parameterDefinitions,
    );

    // Configure I/O ports using proper collections
    final inputCollection = InputPortCollection();
    for (final port in _inputPorts) {
      inputCollection.add(port);
    }

    final outputCollection = OutputPortCollection();
    for (final port in _outputPorts) {
      outputCollection.add(port);
    }

    // Set port collections on stage
    stage.setInputPorts(inputCollection);
    stage.setOutputPorts(outputCollection);

    // Configure assets
    for (final asset in _assets) {
      stage.addAsset(asset);
    }

    // Configure resources
    if (_resourceRequirements != null) {
      stage.setResourceRequirements(_resourceRequirements!);
    }
    if (_memoryStrategy != null) {
      stage.setMemoryStrategy(_memoryStrategy!);
    }
    if (_realTimeConstraints != null) {
      stage.setRealTimeConstraints(_realTimeConstraints!);
    }

    // Initialize parameters
    for (final param in _parameters) {
      stage.addParameter(param);
      stage.setPropertyValue(param.name, param.defaultValue);
    }

    return stage;
  }

  /// Add multiple parameters at once
  StageBuilder withParameters(Map<String, Parameter> parameters) {
    _parameters.addAll(parameters.values);
    return this;
  }

  // ============ PRIVATE METHODS ============

  String _getNextTensorKey() {
    return 'temp_${_tempCounter++}';
  }

  /// Generate merge function body
  String _generateMergeFunctionBody({
    required String strategy,
    required int inputCount,
  }) {
    switch (strategy) {
      case 'add':
        final inputSum = List.generate(
          inputCount,
          (i) => 'input$i',
        ).join(' + ');
        return 'return $inputSum;';

      case 'multiply':
        final inputProduct = List.generate(
          inputCount,
          (i) => 'input$i',
        ).join(' * ');
        return 'return $inputProduct;';

      case 'average':
        final inputSum = List.generate(
          inputCount,
          (i) => 'input$i',
        ).join(' + ');
        return 'return ($inputSum) / ${inputCount}.0;';

      case 'max':
        if (inputCount == 1) return 'return input0;';
        String result = 'input0';
        for (int i = 1; i < inputCount; i++) {
          result = 'max($result, input$i)';
        }
        return 'return $result;';

      case 'min':
        if (inputCount == 1) return 'return input0;';
        String result = 'input0';
        for (int i = 1; i < inputCount; i++) {
          result = 'min($result, input$i)';
        }
        return 'return $result;';

      default:
        // Default to add
        final inputSum = List.generate(
          inputCount,
          (i) => 'input$i',
        ).join(' + ');
        return 'return $inputSum;';
    }
  }

  /// Generate split function body
  String _generateSplitFunctionBody({
    required String strategy,
    required int outputCount,
  }) {
    switch (strategy) {
      case 'copy':
        return 'return input_val;'; // Simple copy to all outputs

      case 'scale':
        return '''
    switch (outputIndex) {
        ${List.generate(outputCount, (i) => '''
        case ${i}u: { return input_val * ${(i + 1) * 0.5}; }''').join('\n        ')}
        default: { return input_val; }
    }
''';

      default:
        return 'return input_val;';
    }
  }

  // Build WGSL for merge: N inputs -> 1 output
  String _buildMergeWGSL({
    required int inputCount,
    required String functionBody,
    int workgroupSizeX = 64,
  }) {
    final b = StringBuffer();

    // Declarations: inputs first, then single output
    for (int i = 0; i < inputCount; i++) {
      b.writeln(
        '@group(0) @binding($i) var<storage, read_write> input_$i: array<f32>;',
      );
    }
    b.writeln(
      '@group(0) @binding($inputCount) var<storage, read_write> output_0: array<f32>;',
    );
    b.writeln();

    // Function with parameters named input0..inputN-1 so functionBody can use them
    final params = List.generate(inputCount, (i) => 'input$i: f32').join(', ');
    b.writeln('fn shader_function(index: u32, $params) -> f32 {');
    b.writeln('  $functionBody');
    b.writeln('}');
    b.writeln();

    // Main: guard, call function with input_k[i]
    b.writeln('@compute @workgroup_size($workgroupSizeX)');
    b.writeln('fn main(@builtin(global_invocation_id) gid: vec3<u32>) {');
    b.writeln('  let i: u32 = gid.x;');
    b.writeln('  if (i >= arrayLength(&input_0)) { return; }');
    final args = List.generate(inputCount, (i) => 'input_$i[i]').join(', ');
    b.writeln('  output_0[i] = shader_function(i, $args);');
    b.writeln('}');

    return b.toString();
  }

  // Getters for debugging
  int get operationCount => _operations.length;
  Set<String> get parameterNames => _parameters.map((p) => p.name).toSet();
  Set<String> get inputPortNames => _inputPorts.map((p) => p.name).toSet();
  Set<String> get outputPortNames => _outputPorts.map((p) => p.name).toSet();
  Set<String> get assetNames => _assets.map((a) => a.name).toSet();
}
