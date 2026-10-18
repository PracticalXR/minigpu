// lib/src/operation/cpu_stage_operation.dart
import 'dart:async';
import 'dart:typed_data';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'stage_operation.dart';
import 'stage_executor.dart';
import '../stages/pipeline_resource_cache.dart';

/// CPU processor function signature - gets raw data + rank info
typedef CPUProcessor =
    Future<Map<String, TypedData>> Function(
      Map<String, TypedData> inputs,
      Map<String, List<int>> ranks,
      Map<String, dynamic> parameters,
    );

// CPU operation that provides raw data and rank information to user processor
class CPUStageOperation extends StageOperation {
  final CPUProcessor processor;
  final bool isolateEnabled;

  /// Optional explicit execution backend. When omitted, the operation picks
  /// [InlineExecutor] (default) or [PersistentIsolateExecutor] based on
  /// [isolateEnabled]. The executor only ever receives raw host data, so
  /// stateful processors must keep the default inline backend.
  final StageExecutor? _explicitExecutor;

  late final StageExecutor _executor =
      _explicitExecutor ??
      (isolateEnabled
          ? PersistentIsolateExecutor(processor)
          : InlineExecutor(processor));

  /// The execution backend in use (resolved lazily on first run).
  StageExecutor get executor => _executor;

  // Per-input-key scratch buffers, reused across frames to avoid per-frame
  // TypedData allocation on the Dart heap. Populated lazily on the first
  // call per key and updated only when the tensor grows beyond the cached size.
  final Map<String, TypedData> _inputScratch = {};

  int _cacheHits = 0;
  int _cacheMisses = 0;

  CPUStageOperation({
    required super.operationId,
    required this.processor,
    this.isolateEnabled = false,
    StageExecutor? executor,
    super.inputKeys,
    super.outputKeys,
    super.arguments,
    super.namedArguments,
  }) : _explicitExecutor = executor;

  @override
  Future<Map<String, Tensor>> execute(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache? cache,
  ) async {
    // Follow the same pattern as other operations
    return await executeWithCache(
      inputs,
      parameters,
      cache ?? ResourceCache(cacheId: 'default'),
    );
  }

  @override
  Future<Map<String, Tensor>> executeWithCache(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache cache,
  ) async {
    validateInputs(inputs);

    // Step 1: Extract raw data and ranks from GPU tensors.
    // Reuse the per-key scratch buffer when possible so the hot path does not
    // allocate a new TypedData on every frame (avoids GC pressure for large
    // tensors — e.g. a 1080p YUV420 Float32List is ~3 MB per frame).
    final activeInputKeys = inputKeys.isEmpty ? inputs.keys : inputKeys;
    final rawInputs = <String, TypedData>{};
    final ranks = <String, List<int>>{};
    for (final entry in inputs.entries) {
      if (activeInputKeys.contains(entry.key)) {
        final key = entry.key;
        final tensor = entry.value;
        TypedData data;
        final cached = _inputScratch[key];
        if (cached != null) {
          // getData validates the buffer is large enough and performs the copy
          // into it; if the tensor has grown we catch the error, allocate
          // fresh, and update the cache.
          try {
            data = await tensor.getData(into: cached);
          } on ArgumentError {
            data = await tensor.getData();
            _inputScratch[key] = data;
          }
        } else {
          data = await tensor.getData();
          _inputScratch[key] = data;
        }
        rawInputs[key] = data;
        ranks[key] = tensor.shape;
      }
    }

    // Step 2: Execute CPU processing via the configured backend
    final rawOutputs = await _executor.run(rawInputs, ranks, parameters);

    // Step 3: Upload outputs (respect configured outputKeys or take all)
    final activeOutputKeys = outputKeys.isEmpty ? rawOutputs.keys : outputKeys;
    final finalOutputs = <String, Tensor>{};
    for (final entry in rawOutputs.entries) {
      // Skip shape sidecar keys
      if (_isShapeSidecarKey(entry.key)) continue;
      if (!activeOutputKeys.contains(entry.key)) continue;

      // Resolve output shape: explicit sidecar > infer from input ranks > 1D fallback
      final shape = _resolveOutputShape(
        entry.key,
        entry.value,
        ranks,
        rawOutputs,
      );

      // PROTECT: do not let CPU stage output overwrite canonical 'input' with different shape
      if (entry.key == 'input' &&
          inputs.containsKey('input') &&
          !_shapesMatch(inputs['input']!.shape, shape)) {
        final safeKey = 'cpu_${entry.key}_${shape.join("x")}';
        if (!rawOutputs.containsKey(safeKey)) {
          finalOutputs[safeKey] = await _uploadRawData(
            entry.value,
            shape,
            cache,
            outKey: safeKey,
          );
          continue;
        }
      }

      finalOutputs[entry.key] = await _uploadRawData(
        entry.value,
        shape,
        cache,
        outKey: entry.key,
      );
    }

    // Step 4: Pass through untouched required inputs
    for (final entry in inputs.entries) {
      if (!finalOutputs.containsKey(entry.key) &&
          !activeOutputKeys.contains(entry.key)) {
        finalOutputs[entry.key] = entry.value;
      }
    }

    return finalOutputs;
  }

  /// Calculate output shape based on data length (assumes 1D for now)
  List<int> _calculateOutputShape(TypedData data) {
    final length = _getTypedDataLength(data);
    return length > 0 ? [length] : [0];
  }

  // New: resolve output shape with precedence: sidecar > infer by rank elem count > 1D fallback
  List<int> _resolveOutputShape(
    String outKey,
    TypedData data,
    Map<String, List<int>> inputRanks,
    Map<String, TypedData> rawOutputs,
  ) {
    // 1) Explicit sidecar shape if provided by the CPU processor
    final sidecar1 = '${outKey}_shape';
    final sidecar2 = '${outKey}__shape';
    if (rawOutputs.containsKey(sidecar1)) {
      final ints = _typedDataToIntList(rawOutputs[sidecar1]!);
      if (ints.isNotEmpty) return ints;
    }
    if (rawOutputs.containsKey(sidecar2)) {
      final ints = _typedDataToIntList(rawOutputs[sidecar2]!);
      if (ints.isNotEmpty) return ints;
    }

    // 2) Prefer preserving 'input' rank if element counts match
    final len = _getTypedDataLength(data);
    final preferKeys = ['input', 'output'];
    for (final k in preferKeys) {
      final s = inputRanks[k];
      if (s != null && s.fold<int>(1, (a, b) => a * b) == len) {
        return List<int>.from(s);
      }
    }

    // 3) Otherwise, match any input rank with the same element count
    for (final s in inputRanks.values) {
      if (s.fold<int>(1, (a, b) => a * b) == len) {
        return List<int>.from(s);
      }
    }

    // 4) Fallback to 1D shape
    return _calculateOutputShape(data);
  }

  bool _isShapeSidecarKey(String key) =>
      key.endsWith('_shape') || key.endsWith('__shape');

  List<int> _typedDataToIntList(TypedData td) {
    if (td is Int32List) return td.toList();
    if (td is Uint32List) return td.toList();
    if (td is Int16List) return td.map((e) => e.toInt()).toList();
    if (td is Uint16List) return td.map((e) => e.toInt()).toList();
    if (td is Int8List) return td.map((e) => e.toInt()).toList();
    if (td is Uint8List) return td.map((e) => e.toInt()).toList();
    if (td is Float32List) return td.map((e) => e.toInt()).toList();
    if (td is Float64List) return td.map((e) => e.toInt()).toList();
    return const <int>[];
  }

  /// Upload raw data back to GPU tensor
  Future<Tensor> _uploadRawData(
    TypedData rawData,
    List<int> shape,
    ResourceCache cache, {
    required String outKey,
  }) async {
    final dtype = _dtypeTag(rawData);
    final cacheKey =
        'cpu_output_${operationId}_${outKey}_${dtype}_${shape.join('x')}';

    // Validate data length matches shape before creating/reusing tensor
    final expectedLength = shape.fold<int>(1, (a, b) => a * b);
    final actualLength = _getTypedDataLength(rawData);

    if (expectedLength != actualLength) {
      print(
        '⚠️ Shape/data mismatch: expected $expectedLength elements (shape $shape), got $actualLength',
      );
      // Correct the shape to match actual data
      shape = [actualLength];
    }

    final cached = cache.getCachedTensor(cacheKey);

    // --- Native dtype branches: preserve the exact CPU byte representation
    // when the processor returned a typed list other than Float32. This is
    // critical for codec stages that emit byte streams (JPEG, PNG, raw GPU
    // codecs) where Float32 round-tripping would corrupt the payload.
    if (rawData is Uint8List) {
      if (cached is Tensor<Uint8List> && _shapesMatch(cached.shape, shape)) {
        _cacheHits++;
        await cached.write(rawData);
        return cached;
      }
      _cacheMisses++;
      final Tensor<Uint8List> gpuTensor = await Tensor.zeros<Uint8List>(
        shape,
        dataType: BufferDataType.uint8,
      );
      await gpuTensor.write(rawData);
      cache.cacheTensor(cacheKey, gpuTensor);
      return gpuTensor;
    }

    if (rawData is Int32List) {
      if (cached is Tensor<Int32List> && _shapesMatch(cached.shape, shape)) {
        _cacheHits++;
        await cached.write(rawData);
        return cached;
      }

      _cacheMisses++;
      final Tensor<Int32List> gpuTensor = await Tensor.zeros<Int32List>(
        shape,
        dataType: BufferDataType.int32,
      );
      await gpuTensor.write(rawData);
      cache.cacheTensor(cacheKey, gpuTensor);
      return gpuTensor;
    }

    if (rawData is Uint32List) {
      if (cached is Tensor<Uint32List> && _shapesMatch(cached.shape, shape)) {
        _cacheHits++;
        await cached.write(rawData);
        return cached;
      }
      _cacheMisses++;
      final Tensor<Uint32List> gpuTensor = await Tensor.zeros<Uint32List>(
        shape,
        dataType: BufferDataType.uint32,
      );
      await gpuTensor.write(rawData);
      cache.cacheTensor(cacheKey, gpuTensor);
      return gpuTensor;
    }

    {
      // Float-family fallback: Float32List passes through; other numeric
      // lists are widened/narrowed to Float32 (legacy ML behavior).
      Float32List float32Data;
      if (rawData is Float32List) {
        float32Data = rawData;
      } else if (rawData is Float64List) {
        float32Data = Float32List.fromList(rawData.map((e) => e).toList());
      } else {
        // Fallback: interpret as bytes and convert to floats
        final bytes = rawData.buffer.asUint8List(
          rawData.offsetInBytes,
          rawData.lengthInBytes,
        );
        if (bytes.lengthInBytes % 4 == 0) {
          // Try to interpret as Float32
          float32Data = bytes.buffer.asFloat32List(
            rawData.offsetInBytes,
            bytes.lengthInBytes ~/ 4,
          );
        } else {
          // Convert bytes to individual float values
          float32Data = Float32List.fromList(
            bytes.map((e) => e.toDouble()).toList(),
          );
          shape = [bytes.length]; // Update shape to match byte array
        }
      }

      if (cached is Tensor<Float32List> && _shapesMatch(cached.shape, shape)) {
        _cacheHits++;
        await cached.write(float32Data);
        return cached;
      }

      _cacheMisses++;
      final Tensor<Float32List> gpuTensor = await Tensor.zeros<Float32List>(
        shape,
      );
      await gpuTensor.write(float32Data);
      cache.cacheTensor(cacheKey, gpuTensor);
      return gpuTensor;
    }
  }

  // Helper to get actual data length consistently
  int _getTypedDataLength(TypedData data) {
    if (data is Float32List) return data.length;
    if (data is Float64List) return data.length;
    if (data is Int32List) return data.length;
    if (data is Int16List) return data.length;
    if (data is Int8List) return data.length;
    if (data is Uint32List) return data.length;
    if (data is Uint16List) return data.length;
    if (data is Uint8List) return data.length;
    if (data is Int64List) return data.length;
    if (data is Uint64List) return data.length;

    // Fallback for ByteData or unknown types
    return data.lengthInBytes ~/ 4; // Assume 4-byte elements
  }

  String _dtypeTag(TypedData data) {
    if (data is Float32List) return 'f32';
    if (data is Float64List) return 'f64';
    if (data is Int32List) return 'i32';
    if (data is Uint32List) return 'u32';
    if (data is Int16List) return 'i16';
    if (data is Uint16List) return 'u16';
    if (data is Int8List) return 'i8';
    if (data is Uint8List) return 'u8';
    return 'td';
  }

  bool _shapesMatch(List<int> shape1, List<int> shape2) {
    if (shape1.length != shape2.length) return false;
    for (int i = 0; i < shape1.length; i++) {
      if (shape1[i] != shape2[i]) return false;
    }
    return true;
  }

  @override
  Future<void> dispose() async {
    _inputScratch.clear();
    await _executor.dispose();
    await super.dispose();
  }

  @override
  int get cacheHits => _cacheHits;

  @override
  int get cacheMisses => _cacheMisses;

  @override
  double get cacheHitRate => (_cacheHits + _cacheMisses) > 0
      ? (_cacheHits / (_cacheHits + _cacheMisses)) * 100.0
      : 0.0;

  @override
  void resetCacheStats() {
    _cacheHits = 0;
    _cacheMisses = 0;
  }
}

/// Dynamic CPU operation with runtime Dart code evaluation
class DynamicCPUOperation extends CPUStageOperation {
  final String dartCode;
  final Map<String, dynamic> initialParameters;

  DynamicCPUOperation({
    required super.operationId,
    required this.dartCode,
    this.initialParameters = const {},
    super.isolateEnabled,
    List<String>? inputKeys,
    List<String>? outputKeys,
  }) : super(
         processor: (rawInputs, ranks, runtimeParams) async {
           final merged = {...initialParameters, ...runtimeParams};
           return await _executeDynamicRaw(dartCode, rawInputs, ranks, merged);
         },
         inputKeys: inputKeys ?? const [],
         outputKeys: outputKeys ?? const [],
       ) {
    _validateUserCode();
  }

  void _validateUserCode() {
    if (!RegExp(r'Map<String,\s*TypedData>\s+main\s*\(').hasMatch(dartCode)) {
      throw Exception('Dynamic CPU code missing main signature');
    }
  }
}

// Standalone helper (no static shared state)
//
// Dart-source CPU stage execution previously used `dart_eval` to interpret
// user code at runtime; that dependency (and its vendored copy) was removed
// because its analyzer/dart:io import graph cannot compile for the web.
// The supported replacement is `ExpressionCPUOperation` (opKind
// 'expression_cpu'): per-element math expressions compiled once at stage
// build and applied vectorized — web + native, zero dependencies.
Future<Map<String, TypedData>> _executeDynamicRaw(
  String source,
  Map<String, TypedData> hostInputs,
  Map<String, List<int>> hostRanks,
  Map<String, dynamic> hostParams,
) async {
  throw UnimplementedError(
    'Dynamic Dart-source CPU stages are disabled (dart_eval removed — its '
    'analyzer dependency breaks web builds). Use ExpressionCPUOperation '
    "(opKind 'expression_cpu') for per-element transforms instead.",
  );
}

/// Prebuilt CPU operations with common utilities
class PrebuiltCPUOperation extends CPUStageOperation {
  final String operation;

  PrebuiltCPUOperation({
    required super.operationId,
    required this.operation,
    Map<String, dynamic> operationParameters = const {},
    super.inputKeys,
    super.outputKeys,
  }) : super(
         processor: (inputs, ranks, params) => CPUOperationRegistry.execute(
           operation,
           inputs,
           ranks,
           {...operationParameters, ...params},
         ),
       );
}

/// Registry for built-in CPU utilities (optional convenience functions)
class CPUOperationRegistry {
  static final Map<String, CPUProcessor> _processors = {};

  static void registerBuiltInProcessors() {
    _processors['passthrough'] = _passthroughProcessor;
    _processors['debug_info'] = _debugInfoProcessor;
    // Add more utility processors as needed
  }

  static Future<Map<String, TypedData>> execute(
    String operation,
    Map<String, TypedData> inputs,
    Map<String, List<int>> ranks,
    Map<String, dynamic> parameters,
  ) async {
    final processor = _processors[operation];
    if (processor == null) {
      throw Exception('Unknown CPU operation: $operation');
    }
    return await processor(inputs, ranks, parameters);
  }

  /// Simple passthrough processor (useful for debugging)
  static Future<Map<String, TypedData>> _passthroughProcessor(
    Map<String, TypedData> inputs,
    Map<String, List<int>> ranks,
    Map<String, dynamic> params,
  ) async {
    final firstKey = inputs.keys.first;
    final shape =
        ranks[firstKey] ?? <int>[inputs[firstKey]!.lengthInBytes ~/ 4];
    return {
      'output': inputs[firstKey]!,
      'output_shape': Int32List.fromList(shape),
    };
  }

  /// Debug info processor (prints tensor info)
  static Future<Map<String, TypedData>> _debugInfoProcessor(
    Map<String, TypedData> inputs,
    Map<String, List<int>> ranks,
    Map<String, dynamic> params,
  ) async {
    print('=== CPU Debug Info ===');
    for (final key in inputs.keys) {
      final data = inputs[key]!;
      final shape = ranks[key]!;
      print(
        '  $key: ${data.runtimeType} shape=$shape length=${data.lengthInBytes}',
      );
    }
    print('  Parameters: $params');
    print('======================');

    return {'output': inputs.values.first};
  }
}
