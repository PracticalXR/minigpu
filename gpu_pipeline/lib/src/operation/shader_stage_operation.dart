import 'dart:async';
import 'package:gpu_pipeline/src/pipeline_ports.dart';
import 'package:gpu_pipeline/src/stages/pipeline_resource_cache.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';
import 'stage_operation.dart';

/// Mixin for shader generation and binding logic
mixin ShaderGenerationMixin on StageOperation {
  String get shaderTemplate;
  Map<String, dynamic> get shaderParameters;
  List<int>? get workgroupSize;

  /// Generate complete shader code with dynamic bindings
  String generateCompleteShader(
    Map<String, Tensor> inputs,
    Map<String, Tensor> outputs,
    Map<String, dynamic> parameters,
  ) {
    // Start with parameter substitution
    String shader = _processParameterSubstitution(shaderTemplate, parameters);

    // Wrap in compute shader if needed
    shader = _wrapInComputeShaderIfNeeded(shader, inputs, outputs);

    // Add dynamic bindings
    shader = _addDynamicBindings(shader);

    // SANITIZE: force storage buffers to be read_write for our backend
    shader = _forceStorageReadWrite(shader);

    return shader;
  }

  String _getWgslType(dynamic value) {
    if (value is bool) return 'bool';
    if (value is double) return 'f32';
    if (value is int) return 'i32';
    if (value is String) {
      final doubleVal = double.tryParse(value);
      if (doubleVal != null) return 'f32';
      return 'f32'; // Default to f32 for unknown strings
    }
    return 'f32'; // Default fallback
  }

  /// Process all parameter substitutions
  String _processParameterSubstitution(
    String template,
    Map<String, dynamic> parameters,
  ) {
    String result = template;

    // Build resolved params:
    // - Start from shaderParameters
    // - If a value is 'param:NAME', pull NAME from runtime parameters
    // - If runtime provides the same key directly, override it
    final resolvedParams = <String, dynamic>{};

    for (final entry in shaderParameters.entries) {
      final key = entry.key;
      final val = entry.value;

      dynamic effective;
      if (val is String && val.startsWith('param:')) {
        final ref = val.substring(6);
        effective = parameters.containsKey(ref) ? parameters[ref] : null;
      } else {
        effective = parameters.containsKey(key) ? parameters[key] : val;
      }

      if (effective != null) {
        resolvedParams[key] = effective;
      }
    }

    // Also include any runtime-only params (placeholders in template not listed in shaderParameters)
    for (final entry in parameters.entries) {
      resolvedParams.putIfAbsent(entry.key, () => entry.value);
    }

    // Emit constants using resolved values
    final constants = StringBuffer();
    for (final entry in resolvedParams.entries) {
      final formattedValue = _formatShaderValue(entry.value);
      final wgslType = _getWgslType(entry.value);
      constants.writeln('const ${entry.key}: $wgslType = $formattedValue;');
    }

    // Insert constants at a sensible point
    if (constants.isNotEmpty) {
      final lines = result.split('\n');
      final constantsSection = constants.toString();

      var insertIndex = 0;
      for (var i = 0; i < lines.length; i++) {
        final line = lines[i].trim();
        if (line.startsWith('@group(') ||
            line.startsWith('@compute') ||
            line.startsWith('fn ')) {
          insertIndex = i;
          break;
        }
      }

      lines.insert(insertIndex, constantsSection);
      result = lines.join('\n');
    }

    // Replace ${param} placeholders with resolved values
    for (final entry in resolvedParams.entries) {
      final placeholder = '\${${entry.key}}';
      final value = _formatShaderValue(entry.value);
      result = result.replaceAll(placeholder, value);
    }

    // Process conditional compilation and loop unrolling with final params
    result = _processConditionals(result, resolvedParams);
    result = _processLoops(result, resolvedParams);

    return result;
  }

  /// Wrap shader in compute structure if it's just a function body
  String _wrapInComputeShaderIfNeeded(
    String shader,
    Map<String, Tensor> inputs,
    Map<String, Tensor> outputs,
  ) {
    // Already a complete shader
    if (shader.contains('@compute') || shader.contains('@binding(')) {
      return shader;
    }

    // Detect function body vs function definition
    final trimmed = shader.trim();
    final workgroupSizeValue = _getWorkgroupSize(inputs);

    if (_isFunctionBody(trimmed)) {
      return _wrapFunctionBody(trimmed, workgroupSizeValue);
    } else if (_isFunctionDefinition(trimmed)) {
      return _wrapFunctionDefinition(trimmed, workgroupSizeValue);
    } else {
      // Raw compute shader code
      return _wrapRawCode(trimmed, workgroupSizeValue);
    }
  }

  /// Check if shader is just a function body
  bool _isFunctionBody(String shader) {
    final lines = shader
        .split('\n')
        .map((l) => l.trim())
        .where((l) => l.isNotEmpty);

    if (lines.isEmpty) return false;

    // Starts with return
    if (lines.first.startsWith('return')) return true;

    // Has variables and return but no function definition
    if (shader.contains('return') && !shader.contains('fn ')) return true;

    // Switch statements (for split operations)
    if (shader.contains('switch') && !shader.contains('fn ')) return true;

    return false;
  }

  /// Check if shader contains function definitions
  bool _isFunctionDefinition(String shader) {
    return shader.contains('fn ') && !shader.contains('fn main');
  }

  /// Wrap a function body in appropriate compute shader
  String _wrapFunctionBody(String functionBody, int workgroupSizeValue) {
    if (inputKeys.length == 1 && outputKeys.length == 1) {
      // Simple 1:1 transform
      return '''
fn processElement(input_val: f32, index: u32) -> f32 {
    $functionBody
}

@compute @workgroup_size($workgroupSizeValue)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&input_0)) { return; }
    output_0[index] = processElement(input_0[index], index);
}
''';
    } else if (inputKeys.length > 1 && outputKeys.length == 1) {
      // Merge operation: N inputs -> 1 output
      final inputParams = List.generate(
        inputKeys.length,
        (i) => 'input$i: f32',
      ).join(', ');
      final inputReads = List.generate(
        inputKeys.length,
        (i) => 'input_$i[index]',
      ).join(', ');

      return '''
fn mergeElements($inputParams, index: u32) -> f32 {
    $functionBody
}

@compute @workgroup_size($workgroupSizeValue)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&input_0)) { return; }
    output_0[index] = mergeElements($inputReads, index);
}
''';
    } else if (inputKeys.length == 1 && outputKeys.length > 1) {
      // Split operation: 1 input -> N outputs
      final splitAssignments = List.generate(
        outputKeys.length,
        (i) => 'output_$i[index] = splitElement(input_0[index], index, ${i}u);',
      ).join('\n    ');

      return '''
fn splitElement(input_val: f32, index: u32, outputIndex: u32) -> f32 {
    $functionBody
}

@compute @workgroup_size($workgroupSizeValue)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&input_0)) { return; }
    $splitAssignments
}
''';
    } else {
      // Generic multi-input/output
      return '''
@compute @workgroup_size($workgroupSizeValue)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&input_0)) { return; }
    $functionBody
}
''';
    }
  }

  /// Wrap function definitions with main function
  String _wrapFunctionDefinition(String functionDef, int workgroupSizeValue) {
    return '''
$functionDef

@compute @workgroup_size($workgroupSizeValue)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&input_0)) { return; }
    // Call user-defined functions here
}
''';
  }

  /// Wrap raw compute code
  String _wrapRawCode(String code, int workgroupSizeValue) {
    return '''
@compute @workgroup_size($workgroupSizeValue)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&input_0)) { return; }
    $code
}
''';
  }

  /// Add dynamic binding declarations at the start of shader
  String _addDynamicBindings(String shader) {
    // If shader already declares explicit bindings, keep them as-is.
    if (shader.contains('@binding(')) {
      return shader;
    }

    final bindings = StringBuffer();
    var bindingIndex = 0;

    // Inputs first in binding space 0..N-1
    for (int i = 0; i < inputKeys.length; i++) {
      bindings.writeln(
        '@group(0) @binding($bindingIndex) var<storage, read_write> input_$i: array<f32>',
      );
      bindingIndex++;
    }

    // Outputs next in binding space N..N+M-1
    for (int i = 0; i < outputKeys.length; i++) {
      bindings.writeln(
        '@group(0) @binding($bindingIndex) var<storage, read_write> output_$i: array<f32>',
      );
      bindingIndex++;
    }

    return '${bindings.toString()}\n$shader';
  }

  /// Get workgroup size
  int _getWorkgroupSize(Map<String, Tensor> inputs) {
    if (workgroupSize != null && workgroupSize!.isNotEmpty) {
      return workgroupSize!.first;
    }

    // Calculate based on input size
    if (inputs.isNotEmpty) {
      final totalElements = inputs.values.first.shape.reduce((a, b) => a * b);
      if (totalElements >= 1024) return 256;
      if (totalElements >= 512) return 128;
      if (totalElements >= 256) return 64;
      if (totalElements >= 128) return 32;
      return 16;
    }

    return 64; // Default
  }

  /// Force storage buffers to be read_write (backend doesn’t support read-only)
  String _forceStorageReadWrite(String shader) {
    // Replace any var<storage, read> (with flexible whitespace) with read_write
    final rx = RegExp(r'var<\s*storage\s*,\s*read\s*>', multiLine: true);
    return shader.replaceAll(rx, 'var<storage, read_write>');
  }

  /// Format value for shader code
  String _formatShaderValue(dynamic value) {
    if (value is double) return '${value}';
    if (value is int) return '${value}';
    if (value is bool)
      return value.toString(); // Fixed: return "true" or "false", not numeric
    if (value is String) {
      final doubleVal = double.tryParse(value);
      if (doubleVal != null) return '${doubleVal}';
      return value;
    }
    return '${value}';
  }

  /// Process conditional compilation (#if, #elif, #else, #endif)
  String _processConditionals(String shader, Map<String, dynamic> params) {
    final lines = shader.split('\n');
    final result = <String>[];
    final ifStack = <bool>[];

    for (final line in lines) {
      final trimmed = line.trim();

      if (trimmed.startsWith('#if ')) {
        final condition = trimmed.substring(4).trim();
        final shouldInclude = _evaluateCondition(condition, params);
        ifStack.add(shouldInclude);
        continue;
      } else if (trimmed.startsWith('#elif ')) {
        if (ifStack.isEmpty) continue;
        final condition = trimmed.substring(6).trim();
        final shouldInclude =
            !ifStack.last && _evaluateCondition(condition, params);
        ifStack[ifStack.length - 1] = shouldInclude;
        continue;
      } else if (trimmed == '#else') {
        if (ifStack.isEmpty) continue;
        ifStack[ifStack.length - 1] = !ifStack.last;
        continue;
      } else if (trimmed == '#endif') {
        if (ifStack.isNotEmpty) ifStack.removeLast();
        continue;
      }

      // Include line if not inside a false conditional
      if (ifStack.isEmpty || ifStack.every((include) => include)) {
        result.add(line);
      }
    }

    return result.join('\n');
  }

  /// Process loop unrolling (#for var in range(count))
  String _processLoops(String shader, Map<String, dynamic> params) {
    final lines = shader.split('\n');
    final result = <String>[];

    for (var i = 0; i < lines.length; i++) {
      final line = lines[i];
      final trimmed = line.trim();

      if (trimmed.startsWith('#for ')) {
        final forMatch = RegExp(
          r'#for (\w+) in range\((\w+)\)',
        ).firstMatch(trimmed);
        if (forMatch != null) {
          final varName = forMatch.group(1)!;
          final countParam = forMatch.group(2)!;
          final count = params[countParam] as int? ?? 0;

          // Find matching #endfor
          var endForIndex = i + 1;
          var forDepth = 1;
          while (endForIndex < lines.length && forDepth > 0) {
            final endLine = lines[endForIndex].trim();
            if (endLine.startsWith('#for ')) forDepth++;
            if (endLine == '#endfor') forDepth--;
            endForIndex++;
          }

          // Unroll the loop
          final loopBody = lines.sublist(i + 1, endForIndex - 1);
          for (var j = 0; j < count; j++) {
            for (final bodyLine in loopBody) {
              result.add(bodyLine.replaceAll(varName, j.toString()));
            }
          }

          i = endForIndex - 1;
        }
      } else if (!trimmed.startsWith('#endfor')) {
        result.add(line);
      }
    }

    return result.join('\n');
  }

  /// Evaluate conditional expressions
  bool _evaluateCondition(String condition, Map<String, dynamic> params) {
    if (condition.contains('==')) {
      final parts = condition.split('==');
      if (parts.length == 2) {
        final leftValue = params[parts[0].trim()];
        final rightValue = _parseValue(parts[1].trim());
        return leftValue == rightValue;
      }
    } else if (condition.contains('>')) {
      final parts = condition.split('>');
      if (parts.length == 2) {
        final left = params[parts[0].trim()];
        final right = _parseValue(parts[1].trim());
        if (left is num && right is num) {
          return left > right;
        }
      }
    } else {
      final value = params[condition];
      if (value is bool) return value;
      if (value is num) return value != 0;
    }

    return false;
  }

  /// Parse string value to appropriate type
  dynamic _parseValue(String value) {
    if (value == 'true') return true;
    if (value == 'false') return false;
    if (value.contains('.')) return double.tryParse(value);
    return int.tryParse(value) ?? value;
  }
}

/// Mixin for shader execution and tensor binding
mixin ShaderExecutionMixin on StageOperation {
  // Safe wrapper for shader.setBuffer with context
  void _trySetBuffer(
    ComputeShader shader,
    String varName,
    Tensor t,
    String tag,
  ) {
    try {
      shader.setBuffer(varName, t.buffer);
    } catch (e) {
      throw Exception('Failed to bind buffer for $tag ($varName): $e');
    }
  }

  // Bind canonical input_i/output_i by slot order, name-based only.
  Future<void> bindTensors(
    ComputeShader shader,
    Map<String, Tensor> inputs,
    Map<String, Tensor> outputs,
  ) async {
    // Inputs first (bindings 0..N-1)
    for (int i = 0; i < inputKeys.length; i++) {
      final key = inputKeys[i];
      final t = inputs[key];
      if (t == null) throw Exception('Input tensor "$key" not found');
      _trySetBuffer(shader, 'input_$i', t, 'input[$i]');
    }
    // Outputs next (bindings N..N+M-1)
    for (int oi = 0; oi < outputKeys.length; oi++) {
      final key = outputKeys[oi];
      final t = outputs[key];
      if (t == null) throw Exception('Output tensor "$key" not found');
      _trySetBuffer(shader, 'output_$oi', t, 'output[$oi]');
    }
  }

  int _elementsOf(Tensor t) =>
      t.shape.isEmpty ? 0 : t.shape.fold<int>(1, (a, b) => a * b);

  // Compute dispatch X dimension from outputs, fallback to inputs
  // Uses provided workgroupSizeX (defaults to 64 if not set)
  ({int xGroups, int wgSizeX}) _computeDispatch(
    Map<String, Tensor> inputs,
    Map<String, Tensor> outputs, {
    int? workgroupSizeX,
  }) {
    final wg = (workgroupSizeX ?? 64).clamp(1, 1024);
    // Prefer first output tensor length; else first input
    Tensor? ref = outputs.isNotEmpty
        ? outputs.values.first
        : (inputs.isNotEmpty ? inputs.values.first : null);
    final elems = ref != null ? _elementsOf(ref) : 0;
    final groups = elems <= 0 ? 1 : ((elems + wg - 1) ~/ wg);
    return (xGroups: groups, wgSizeX: wg);
  }

  // Execute compute shader with sane defaults for dispatch
  Future<void> executeShader(
    ComputeShader shader,
    Map<String, Tensor> inputs, {
    required Map<String, Tensor> outputs,
    List<int>? workgroupSize,
  }) async {
    // Workgroup size X (we only care about X for 1D tensors)
    final wgX = (workgroupSize != null && workgroupSize.isNotEmpty)
        ? workgroupSize.first
        : 64;

    final disp = _computeDispatch(inputs, outputs, workgroupSizeX: wgX);

    // Bind happens in caller via bindTensors/_bindRespectingExplicitBindings

    // WebGPU caps workgroup count at 65535 PER DIMENSION. A 1D dispatch of
    // `xGroups` overflows above ~65535*wg elements (e.g. a 5120x1440 frame:
    // 7.4M px / 64 = 115200 groups) -> the whole CommandBuffer is invalidated.
    // Fold the overflow into Y: gx in [1,65535], gy covers the rest. This is
    // BACKWARD-COMPATIBLE -- when xGroups <= 65535, gy == 1 and every gid.y is
    // 0, so shaders that index by global_invocation_id.x are byte-identical.
    // Shaders that CAN exceed 65535 groups must reconstruct their flat index as
    // `gid.x + gid.y * (num_workgroups.x * wgX)` (see the mjpeg pipeline).
    final (gx, gy) = _fold2D(disp.xGroups);

    // Dispatch (backends that only support 1D see gy==1 here); ignore Y/Z.
    await shader.dispatch(gx, gy, 1);
  }

  /// Fold a 1D workgroup count into a 2D (x,y) grid so neither dimension
  /// exceeds WebGPU's 65535-per-dimension cap. `x` is clamped to 65535 and `y`
  /// covers the remainder; both are >=1. For counts <= 65535, returns (count,1)
  /// so the dispatch is unchanged.
  static (int, int) _fold2D(int groups) {
    if (groups <= 65535) return (groups <= 0 ? 1 : groups, 1);
    const maxX = 65535;
    final y = (groups + maxX - 1) ~/ maxX;
    return (maxX, y);
  }
}

/// Full shader operation with complete multi-input/output support
class ShaderStageOperation extends StageOperation
    with
        CacheAwareOperation,
        ArgumentResolvable,
        ShaderGenerationMixin,
        ShaderExecutionMixin {
  @override
  final String shaderTemplate;
  final Map<String, OutputPort> outputPorts;
  @override
  final Map<String, dynamic> shaderParameters;
  final Map<String, String> inputMappings;
  final Map<String, String> outputMappings;
  @override
  final List<int>? workgroupSize;

  int _cacheHits = 0;
  int _cacheMisses = 0;

  @override
  int get cacheHits => _cacheHits;

  @override
  int get cacheMisses => _cacheMisses;

  @override
  double get cacheHitRate {
    final total = cacheHits + cacheMisses;
    return total > 0 ? (cacheHits / total) * 100.0 : 0.0;
  }

  @override
  void resetCacheStats() {
    _cacheHits = 0;
    _cacheMisses = 0;
  }

  ShaderStageOperation({
    required super.operationId,
    required this.shaderTemplate,
    required this.outputPorts,
    super.inputKeys,
    super.outputKeys,
    this.shaderParameters = const {},
    this.inputMappings = const {},
    this.outputMappings = const {},
    this.workgroupSize,
    super.arguments,
    super.namedArguments,
  });

  /// Legacy execute method for backward compatibility
  @override
  Future<Map<String, Tensor>> execute(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache cache,
  ) async {
    return executeWithCache(inputs, parameters, cache);
  }

  int _flattenedLen(List<int> shape) => shape.fold(1, (a, b) => a * b);

  void _validateAlignedInputShapes(Map<String, Tensor> inputs) {
    if (inputKeys.length <= 1) return;

    int? baseLen;
    final detail = <String>[];
    for (final key in inputKeys) {
      final t = inputs[key];
      if (t == null) {
        throw Exception('Missing required input "$key"');
      }
      final len = _flattenedLen(t.shape);
      detail.add('$key=${t.shape.join("x")}($len)');
      baseLen ??= len;
      if (len != baseLen) {
        throw Exception(
          'Mismatched input shapes for element-wise shader: ${detail.join(", ")}',
        );
      }
    }
  }

  @override
  Future<Map<String, Tensor>> executeWithCache(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache cache,
  ) async {
    try {
      if (inputs.isEmpty) {
        throw Exception('No inputs provided to shader operation $operationId');
      }

      // Validate required inputs
      for (final inputKey in inputKeys) {
        if (!inputs.containsKey(inputKey)) {
          throw Exception(
            'Missing required input: $inputKey for operation $operationId. '
            'Available inputs: ${inputs.keys.join(", ")}',
          );
        }
      }

      // ensure all element-wise inputs have aligned flattened lengths
      _validateAlignedInputShapes(inputs);

      // Merge all parameters
      final allParameters = <String, dynamic>{
        ...shaderParameters,
        ...parameters,
      };

      // Create outputs first
      final outputs = await _createOutputTensors(inputs, allParameters, cache);

      // Generate complete shader with bindings
      final completeShader = generateCompleteShader(
        inputs,
        outputs,
        allParameters,
      );

      final shader = await _getOrCreateShader(
        completeShader,
        allParameters,
        cache,
      );

      // Ensure fresh tag order so binding indices match @binding order for this run
      shader.resetTagOrder();

      // Bind by slot order; the backend binds by index.
      await bindTensors(shader, inputs, outputs);

      // Dispatch using tensor lengths (key-agnostic)
      await executeShader(shader, inputs, outputs: outputs);

      return outputs;
    } catch (e, stackTrace) {
      throw Exception(
        'Shader execution failed for operation $operationId: $e\nStack trace: $stackTrace',
      );
    }
  }

  /// Get or create shader with caching
  Future<ComputeShader> _getOrCreateShader(
    String shaderCode,
    Map<String, dynamic> parameters,
    ResourceCache cache,
  ) async {
    // Create cache key from shader code and compile-time parameters
    final compileTimeParams = _getCompileTimeParameters(parameters);
    final cacheKey = cache.createShaderKey(shaderCode, compileTimeParams);

    // Try cache first
    final cachedShader = cache.getCachedShader(cacheKey);
    if (cachedShader != null) {
      _cacheHits++;
      return cachedShader;
    }

    // Cache miss - create new shader
    _cacheMisses++;
    try {
      final shader = DefaultMinigpu.instance.createComputeShader();
      shader.loadKernelString(shaderCode);
      cache.cacheShader(cacheKey, shader, shaderCode);

      return shader;
    } catch (e) {
      // Re-throw with more context for the test to catch
      throw Exception(
        'Failed to compile shader for operation $operationId: $e\n\nShader code:\n$shaderCode',
      );
    }
  }

  /// Create output tensors with appropriate shapes
  Future<Map<String, Tensor>> _createOutputTensors(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache cache,
  ) async {
    final outputs = <String, Tensor>{};

    for (final outputKey in outputKeys) {
      List<int> outputShape;
      outputShape = _calculateOutputShape(outputKey, inputs, parameters);

      final cacheKey = cache.createTensorKey(
        '${operationId}_$outputKey',
        outputShape,
        'f32',
      );

      // Check cache first — reuse existing GPU buffer to avoid per-frame VRAM growth.
      final cached = cache.getCachedTensor(cacheKey);
      if (cached != null) {
        _cacheHits++;
        outputs[outputKey] = cached;
      } else {
        _cacheMisses++;
        final newTensor = await Tensor.zeros(outputShape);
        outputs[outputKey] = newTensor;
        cache.cacheTensor(cacheKey, newTensor);
      }
    }

    return outputs;
  }

  /// Calculate output shape for a specific output (can be overridden).
  /// Throws a [StateError] when the output port cannot compute it.
  List<int> _calculateOutputShape(
    String outputKey,
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
  ) {
    final port = outputPorts[outputKey];
    if (port != null) {
      try {
        final shape = port.calculateOutputShape(inputs, parameters);
        if (shape.isNotEmpty) {
          return shape;
        }
      } catch (e) {
        throw StateError(
          'Operation $operationId cannot compute the shape of output '
          '"$outputKey": $e',
        );
      }
    }
    // Fallback: output shape matches first input shape
    return inputs.values.first.shape;
  }

  /// Get compile-time parameters that affect shader compilation
  Map<String, dynamic> _getCompileTimeParameters(
    Map<String, dynamic> parameters,
  ) {
    final compileTimeParams = <String, dynamic>{};

    for (final entry in parameters.entries) {
      if (_isCompileTimeParameter(entry.key, entry.value)) {
        compileTimeParams[entry.key] = entry.value;
      }
    }

    return compileTimeParams;
  }

  /// Check if parameter requires shader recompilation
  bool _isCompileTimeParameter(String key, dynamic value) {
    // Booleans always require recompilation
    if (value is bool) return true;

    // String constants (not parameter references)
    if (value is String && !value.startsWith('param:')) return true;

    // Parameter names that typically affect compilation
    final compileTimeHints = [
      'enable',
      'disable',
      'use',
      'size',
      'count',
      'length',
      'iterations',
      'unroll',
      'variant',
      'type',
      'mode',
    ];

    return compileTimeHints.any((hint) => key.toLowerCase().contains(hint));
  }

  @override
  Future<void> dispose() async {
    // Shader disposal handled by cache
  }

  @override
  Map<String, dynamic> toConfig() {
    return {
      'type': 'shader',
      'operationId': operationId,
      'shaderTemplate': shaderTemplate,
      'inputKeys': inputKeys,
      'outputKeys': outputKeys,
      'shaderParameters': shaderParameters,
      'inputMappings': inputMappings,
      'outputMappings': outputMappings,
      'workgroupSize': workgroupSize,
      'arguments': arguments,
      'namedArguments': namedArguments,
    };
  }

  static ShaderStageOperation fromConfig(Map<String, dynamic> config) {
    return ShaderStageOperation(
      operationId: config['operationId'] as String,
      shaderTemplate: config['shaderTemplate'] as String,
      outputPorts:
          (config['outputPorts'] as Map<String, dynamic>?)?.map(
            (k, v) => MapEntry(k, OutputPort.fromConfig(v)),
          ) ??
          {},
      inputKeys: (config['inputKeys'] as List?)?.cast<String>() ?? ['input'],
      outputKeys: (config['outputKeys'] as List?)?.cast<String>() ?? ['output'],
      shaderParameters:
          (config['shaderParameters'] as Map?)?.cast<String, dynamic>() ?? {},
      inputMappings:
          (config['inputMappings'] as Map?)?.cast<String, String>() ?? {},
      outputMappings:
          (config['outputMappings'] as Map?)?.cast<String, String>() ?? {},
      workgroupSize: (config['workgroupSize'] as List?)?.cast<int>(),
      arguments: (config['arguments'] as Map?)?.cast<String, dynamic>() ?? {},
      namedArguments:
          (config['namedArguments'] as Map?)?.cast<String, dynamic>() ?? {},
    );
  }
}

/// Simplified shader function operation for single expressions
class ShaderFunctionOperation extends StageOperation with CacheAwareOperation {
  final String functionBody;
  final String functionName;
  final Map<String, dynamic> functionParameters;
  final List<int>? workgroupSize;

  int _cacheHits = 0;
  int _cacheMisses = 0;

  @override
  int get cacheHits => _cacheHits;

  @override
  int get cacheMisses => _cacheMisses;

  @override
  double get cacheHitRate {
    final total = _cacheHits + _cacheMisses;
    return total > 0 ? (_cacheHits / total) * 100.0 : 0.0;
  }

  @override
  void resetCacheStats() {
    _cacheHits = 0;
    _cacheMisses = 0;
  }

  ShaderFunctionOperation({
    required super.operationId,
    required this.functionBody,
    this.functionName = 'shader_function',
    this.functionParameters = const {},
    this.workgroupSize,
    super.inputKeys,
    super.outputKeys,
    super.arguments,
    super.namedArguments,
  });

  @override
  Future<Map<String, Tensor>> execute(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache cache,
  ) async {
    return executeWithCache(inputs, parameters, cache);
  }

  int _flattenedLen(List<int> shape) => shape.fold(1, (a, b) => a * b);

  void _validateAlignedInputShapes(Map<String, Tensor> inputs) {
    if (inputKeys.length <= 1) return;

    int? baseLen;
    final detail = <String>[];
    for (final key in inputKeys) {
      final t = inputs[key];
      if (t == null) {
        throw Exception('Missing required input "$key"');
      }
      final len = _flattenedLen(t.shape);
      detail.add('$key=${t.shape.join("x")}($len)');
      baseLen ??= len;
      if (len != baseLen) {
        throw Exception(
          'Mismatched input shapes for element-wise shader: ${detail.join(", ")}',
        );
      }
    }
  }

  @override
  Future<Map<String, Tensor>> executeWithCache(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache cache,
  ) async {
    try {
      validateInputs(inputs);
      _validateAlignedInputShapes(inputs);

      // Get the correct input tensor by key, not just the first one
      final inputKey = inputKeys.first;
      final inputTensor = inputs[inputKey];

      if (inputTensor == null) {
        throw Exception(
          'Required input tensor "$inputKey" not found. Available: ${inputs.keys}',
        );
      }

      // Resolve parameters and generate shader
      final resolvedParams = _resolveParameters(functionParameters, parameters);
      final shaderCode = _generateFunctionShader(resolvedParams);

      // TODO: Hunt race condition with multiple inputs / outputs
      // probably memory allocation for the string on the native side
      //  (seems to happen only in tests with multiple rapid calls)
      await Future.delayed(const Duration(milliseconds: 1));

      // // TODO: DEBUGLOG print('🔧 Generated shader code for $operationId:\n$shaderCode');

      // Compile or get cached shader
      final cacheKey = cache.createShaderKey(shaderCode, resolvedParams);
      ComputeShader? shader = cache.getCachedShader(cacheKey);

      if (shader == null) {
        _cacheMisses++;
        shader = DefaultMinigpu.instance.createComputeShader();
        shader.loadKernelString(shaderCode);
        cache.cacheShader(cacheKey, shader, shaderCode);
      } else {
        _cacheHits++;
      }

      // Create output tensor with same shape as input — reuse from cache if available.
      final outputKey = outputKeys.first;
      Tensor outputTensor;
      if (inputs.containsKey(outputKey) &&
          inputs[outputKey]!.shape.first > inputTensor.shape.first) {
        // Use the larger pre-allocated output tensor (from TextureOutputPort)
        outputTensor = inputs[outputKey]!;
      } else {
        // Check cache before allocating to avoid per-frame VRAM growth.
        final outputCacheKey = cache.createTensorKey(
          '${operationId}_$outputKey',
          inputTensor.shape,
          'f32',
        );
        final cachedOutput = cache.getCachedTensor(outputCacheKey);
        if (cachedOutput != null) {
          _cacheHits++;
          outputTensor = cachedOutput;
        } else {
          _cacheMisses++;
          outputTensor = await Tensor.zeros(inputTensor.shape);
          cache.cacheTensor(outputCacheKey, outputTensor);
        }
      }

      // Bind buffers
      shader.setBuffer('input_0', inputTensor.buffer);
      shader.setBuffer('output_0', outputTensor.buffer);

      // Calculate dispatch size
      final first = inputs[inputKeys.first]!;
      final totalElems = _flattenedLen(first.shape);
      final workgroups = (totalElems + 255) ~/ 256;

      await shader.dispatch(workgroups, 1, 1);

      return {outputKeys.first: outputTensor};
    } catch (e) {
      throw Exception(
        'Shader function execution failed for operation $operationId: $e',
      );
    }
  }

  /// Resolve parameter references (param:name -> actual value)
  Map<String, dynamic> _resolveParameters(
    Map<String, dynamic> functionParams,
    Map<String, dynamic> contextParams,
  ) {
    final resolved = <String, dynamic>{};

    for (final entry in functionParams.entries) {
      final paramKey = entry.key;
      final paramValue = entry.value;

      if (paramValue is String && paramValue.startsWith('param:')) {
        // Extract parameter name and resolve from context
        final contextParamName = paramValue.substring(6);

        // Try exact match first
        dynamic resolvedValue = contextParams[contextParamName];

        // If not found, try case-insensitive match
        if (resolvedValue == null) {
          final lowerParamName = contextParamName.toLowerCase();
          for (final contextEntry in contextParams.entries) {
            if (contextEntry.key.toLowerCase() == lowerParamName) {
              resolvedValue = contextEntry.value;
              break;
            }
          }
        }

        if (resolvedValue != null) {
          resolved[paramKey] = resolvedValue;
        } else {
          // Use default value of 0.0 if parameter not found
          resolved[paramKey] = 0.0;
        }
      } else {
        // Check if there's a runtime override for this parameter key
        final runtimeOverride = contextParams[paramKey];
        if (runtimeOverride != null) {
          resolved[paramKey] = runtimeOverride;
        } else {
          // Use the original value
          resolved[paramKey] = paramValue;
          // TODO: DEBUGLOG print('🔧 Direct parameter $paramKey: $paramValue');
        }
      }
    }

    return resolved;
  }

  /// Generate complete shader function
  String _generateFunctionShader(Map<String, dynamic> resolvedParams) {
    // Generate constants from resolved parameters - avoid duplicates
    final constants = StringBuffer();
    final usedConstantNames = <String>{};

    for (final entry in resolvedParams.entries) {
      final constantName = entry.key
          .toLowerCase(); // Use original key name, lowercased

      // Skip duplicates
      if (usedConstantNames.contains(constantName)) {
        // TODO: DEBUGLOG print('🔧 Skipping duplicate constant: $constantName');
        continue;
      }

      final value = _formatShaderValue(entry.value);
      constants.writeln('const $constantName: f32 = $value;');
      usedConstantNames.add(constantName);
    }

    // Process function body with parameter substitution
    String processedBody = functionBody;

    // Replace parameter names with their lowercase constant names
    for (final entry in resolvedParams.entries) {
      final originalName = entry.key;
      final constantName = entry.key.toLowerCase();

      // Replace both original parameter name and any remaining placeholders
      processedBody = processedBody.replaceAll(originalName, constantName);
      processedBody = processedBody.replaceAll(
        '\${$originalName}',
        constantName,
      );
    }

    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

$constants

fn $functionName(index: u32, input_val: f32) -> f32 {
    $processedBody
}

@compute @workgroup_size(${workgroupSize?[0] ?? 64})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&input_0)) { return; }
    output_0[i] = $functionName(i, input_0[i]);
}
''';

    return shaderCode;
  }

  String _formatShaderValue(dynamic value) {
    if (value is double) return value.toString();
    if (value is int) return value.toDouble().toString();
    if (value is bool) return value ? '1.0' : '0.0';
    if (value is String) {
      final doubleVal = double.tryParse(value);
      if (doubleVal != null) return doubleVal.toString();
      return value;
    }
    return value.toString();
  }

  @override
  Future<void> dispose() async {
    // Handled by cache
  }

  @override
  Map<String, dynamic> toConfig() {
    return {
      'type': 'shader_function',
      'operationId': operationId,
      'functionBody': functionBody,
      'functionName': functionName,
      'functionParameters': functionParameters,
      'workgroupSize': workgroupSize,
      'inputKeys': inputKeys,
      'outputKeys': outputKeys,
      'arguments': arguments,
      'namedArguments': namedArguments,
    };
  }

  static ShaderFunctionOperation fromConfig(Map<String, dynamic> config) {
    return ShaderFunctionOperation(
      operationId: config['operationId'] as String,
      functionBody: config['functionBody'] as String,
      functionName: config['functionName'] as String? ?? 'shader_function',
      functionParameters:
          (config['functionParameters'] as Map?)?.cast<String, dynamic>() ?? {},
      workgroupSize: (config['workgroupSize'] as List?)?.cast<int>(),
      inputKeys: (config['inputKeys'] as List?)?.cast<String>() ?? ['input'],
      outputKeys: (config['outputKeys'] as List?)?.cast<String>() ?? ['output'],
      arguments: (config['arguments'] as Map?)?.cast<String, dynamic>() ?? {},
      namedArguments:
          (config['namedArguments'] as Map?)?.cast<String, dynamic>() ?? {},
    );
  }
}
