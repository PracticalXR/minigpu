import 'dart:async';
import 'package:gpu_pipeline/src/stages/pipeline_resource_cache.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'stage_operation.dart';

/// Tensor operation that uses the built-in tensor methods
class TensorOperation extends StageOperation
    with CacheAwareOperation, ArgumentResolvable {
  final String operation;

  int _cacheHits = 0;
  int _cacheMisses = 0;

  @override
  int get cacheHits => _cacheHits;

  @override
  int get cacheMisses => _cacheMisses;

  @override
  void resetCacheStats() {
    _cacheHits = 0;
    _cacheMisses = 0;
  }

  TensorOperation({
    required super.operationId,
    required this.operation,
    List<String>? inputKeys,
    List<String>? outputKeys,
    super.arguments,
    super.namedArguments,
  }) : super(
         inputKeys: inputKeys ?? ['input'],
         outputKeys: outputKeys ?? ['output'],
       ) {
    TensorOperationRegistry.ensureInitialized();
  }

  List<dynamic>? _resolvedPositional;
  Map<String, dynamic>? _resolvedNamed;

  @override
  void applyResolvedArguments(
    List<dynamic> positional,
    Map<String, dynamic> named,
  ) {
    _resolvedPositional = positional;
    _resolvedNamed = named;
  }

  @override
  Future<Map<String, Tensor>> executeWithCache(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache resourceCache,
  ) async {
    validateInputs(inputs);

    // Build positional args:
    // 1) use resolved values if present (from a resolver upstream),
    // 2) else use arguments['args'] if it's a List,
    // 3) else collect numeric keys '0','1',... from arguments and resolve "param:".
    List<dynamic> positional =
        _resolvedPositional ??
        (() {
          final fromArgsField = arguments['args'];
          if (fromArgsField is List) {
            return fromArgsField.cast<dynamic>();
          }
          final numericKeys =
              arguments.keys
                  .map((k) => int.tryParse(k))
                  .whereType<int>()
                  .toList()
                ..sort();
          if (numericKeys.isEmpty) return const <dynamic>[];
          return <dynamic>[
            for (final i in numericKeys)
              _resolveValue(arguments['$i'], parameters),
          ];
        })();

    // Build named args: prefer resolved, else apply "param:" substitution on namedArguments
    final Map<String, dynamic> named =
        _resolvedNamed ??
        {
          for (final entry in namedArguments.entries)
            entry.key: _resolveValue(entry.value, parameters),
        };

    // No memoization here; treat each execution as a cache miss
    _cacheMisses += inputKeys.length;

    final results = <String, Tensor>{};
    for (int i = 0; i < inputKeys.length; i++) {
      final inputKey = inputKeys[i];
      final outputKey = i < outputKeys.length ? outputKeys[i] : outputKeys.last;
      final inputTensor = inputs[inputKey]!;

      final result = await TensorOperationRegistry.executeOperation(
        operation,
        inputTensor,
        positional,
        named,
      );
      results[outputKey] = result;
    }
    return results;
  }

  // Resolve "param:foo" using parameters['foo'], otherwise return the value
  dynamic _resolveValue(dynamic value, Map<String, dynamic> parameters) {
    if (value is String && value.startsWith('param:')) {
      final key = value.substring('param:'.length);
      return parameters[key];
    }
    return value;
  }

  @override
  Future<Map<String, Tensor>> execute(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache resourceCache,
  ) async {
    return await executeWithCache(inputs, parameters, resourceCache);
  }

  @override
  Map<String, dynamic> toConfig() {
    return {
      'type': 'tensor',
      'operationId': operationId,
      'operation': operation,
      'inputKeys': inputKeys,
      'outputKeys': outputKeys,
      'arguments': arguments,
      'namedArguments': namedArguments,
    };
  }

  static TensorOperation fromConfig(Map<String, dynamic> config) {
    return TensorOperation(
      operationId: config['operationId'] as String,
      operation: config['operation'] as String,
      inputKeys: (config['inputKeys'] as List?)?.cast<String>() ?? ['input'],
      outputKeys: (config['outputKeys'] as List?)?.cast<String>() ?? ['output'],
      arguments: (config['arguments'] as Map?)?.cast<String, dynamic>() ?? {},
      namedArguments:
          (config['namedArguments'] as Map?)?.cast<String, dynamic>() ?? {},
    );
  }
}

/// Registry for tensor operations with caching support
class TensorOperationRegistry {
  static final Map<String, TensorOperationInfo> _operations = {};
  static bool _initialized = false;

  static void ensureInitialized() {
    if (!_initialized) {
      initialize();
    }
  }

  static void initialize() {
    if (_initialized) return;

    _registerBuiltinOperations();
    _initialized = true;
  }

  static void _registerBuiltinOperations() {
    // Scalar operations
    _register(
      'addScalar',
      (tensor, args, named) => tensor.addScalar(_getScalar(args, 0)),

      requiredArgs: 1,
      preservesShape: true,
    );

    _register(
      'multiplyScalar',
      (tensor, args, named) => tensor.multiplyScalar(_getScalar(args, 0)),
      requiredArgs: 1,
      preservesShape: true,
    );

    _register(
      'subtractScalar',
      (tensor, args, named) => tensor.subtractScalar(_getScalar(args, 0)),
      requiredArgs: 1,
      preservesShape: true,
    );

    _register(
      'divideScalar',
      (tensor, args, named) => tensor.divideScalar(_getScalar(args, 0)),
      requiredArgs: 1,
      preservesShape: true,
    );

    // Tensor-tensor operations
    _register(
      'add',
      (tensor, args, named) => tensor.add(_getTensor(args, 0)),
      requiredArgs: 1,
      preservesShape: true,
    );

    _register(
      'multiply',
      (tensor, args, named) => tensor.multiply(_getTensor(args, 0)),
      requiredArgs: 1,
      preservesShape: true,
    );

    _register(
      'subtract',
      (tensor, args, named) => tensor.subtract(_getTensor(args, 0)),
      requiredArgs: 1,
      preservesShape: true,
    );

    _register(
      'divide',
      (tensor, args, named) => tensor.divide(_getTensor(args, 0)),
      requiredArgs: 1,
      preservesShape: true,
    );

    _register(
      'max',
      (tensor, args, named) => tensor.max(_getTensor(args, 0)),
      requiredArgs: 1,
      preservesShape: true,
    );

    _register(
      'min',
      (tensor, args, named) => tensor.min(_getTensor(args, 0)),
      requiredArgs: 1,
      preservesShape: true,
    );

    // Unary operations
    _register(
      'abs',
      (tensor, args, named) => tensor.abs(),
      preservesShape: true,
    );

    // isRealInput matters: without it a flat real tensor of even length is
    // interpreted as interleaved complex pairs — a valid-looking but wrong
    // spectrum for audio samples.  Pass namedArguments: {'isRealInput': true}
    // for real-signal FFTs.
    _register(
      'fft',
      (tensor, args, named) =>
          tensor.fft(isRealInput: named['isRealInput'] == true),
      preservesShape: true,
    );

    _register(
      'sqrt',
      (tensor, args, named) => tensor.sqrt(),
      preservesShape: true,
    );

    _register(
      'exp',
      (tensor, args, named) => tensor.exp(),
      preservesShape: true,
    );

    _register(
      'log',
      (tensor, args, named) => tensor.log(),
      preservesShape: true,
    );

    _register(
      'sin',
      (tensor, args, named) => tensor.sin(),
      preservesShape: true,
    );

    _register(
      'cos',
      (tensor, args, named) => tensor.cos(),
      preservesShape: true,
    );

    _register(
      'tanh',
      (tensor, args, named) => tensor.tanh(),
      preservesShape: true,
    );

    // Reduction operations
    _register(
      'sum',
      (tensor, args, named) => tensor.sum(axis: named['axis'] as int? ?? -1),
      preservesShape: false,
    );

    _register(
      'mean',
      (tensor, args, named) => tensor.mean(axis: named['axis'] as int? ?? -1),
      preservesShape: false,
    );

    // Normalize operation
    _register('normalize', (tensor, args, named) async {
      final mean = await tensor.mean();
      final centered = await tensor.subtract(mean);
      final variance = await (await centered.multiply(centered)).mean();
      final std = await variance.sqrt();
      final normalized = await centered.divide(std);
      return normalized;
    }, preservesShape: true);
  }

  static void _register(
    String name,
    Future<Tensor> Function(Tensor, List<dynamic>, Map<String, dynamic>)
    executor, {
    int requiredArgs = 0,
    bool preservesShape = true,
  }) {
    _operations[name] = TensorOperationInfo(
      executor: executor,
      requiredArgs: requiredArgs,
      preservesShape: preservesShape,
    );
  }

  /// Register a custom operation
  static void registerCustomOperation(
    String name,
    Future<Tensor> Function(Tensor, List<dynamic>, Map<String, dynamic>)
    executor, {
    int requiredArgs = 0,
    bool preservesShape = true,
  }) {
    _operations[name] = TensorOperationInfo(
      executor: executor,
      requiredArgs: requiredArgs,
      preservesShape: preservesShape,
    );
  }

  /// Remove a registered operation
  static bool unregisterOperation(String name) {
    return _operations.remove(name) != null;
  }

  /// Clear all custom operations (keep built-in ones)
  static void clearCustomOperations() {
    // Clear all operations
    _operations.clear();

    // Re-register built-in operations
    _registerBuiltinOperations();
  }

  /// Get operation info (for debugging/inspection)
  static TensorOperationInfo? getOperationInfo(String name) {
    return _operations[name];
  }

  /// Check if an operation preserves shape
  static bool operationPreservesShape(String name) {
    return _operations[name]?.preservesShape ?? false;
  }

  /// Get required argument count for an operation
  static int getRequiredArgCount(String name) {
    return _operations[name]?.requiredArgs ?? 0;
  }

  /// Get operations by category
  static Set<String> getOperationsByCategory(String category) {
    switch (category.toLowerCase()) {
      case 'scalar':
        return {
          'addScalar',
          'multiplyScalar',
          'subtractScalar',
          'divideScalar',
        }.intersection(_operations.keys.toSet());

      case 'unary':
        return {
          'abs',
          'sqrt',
          'exp',
          'log',
          'sin',
          'cos',
          'tanh',
        }.intersection(_operations.keys.toSet());

      case 'binary':
        return {
          'add',
          'multiply',
          'subtract',
          'divide',
          'max',
          'min',
        }.intersection(_operations.keys.toSet());

      case 'reduction':
        return {'sum', 'mean'}.intersection(_operations.keys.toSet());

      case 'custom':
        final builtinOps = {
          'addScalar',
          'multiplyScalar',
          'subtractScalar',
          'divideScalar',
          'add',
          'multiply',
          'subtract',
          'divide',
          'max',
          'min',
          'abs',
          'sqrt',
          'fft',
          'exp',
          'log',
          'sin',
          'cos',
          'tanh',
          'sum',
          'mean',
          'normalize',
        };
        return _operations.keys.toSet().difference(builtinOps);

      default:
        return <String>{};
    }
  }

  static Future<Tensor> executeOperation(
    String name,
    Tensor input,
    List<dynamic> args,
    Map<String, dynamic> namedArgs,
  ) async {
    final opInfo = _operations[name];
    if (opInfo == null) {
      final available = _operations.keys.take(10).join(', ');
      final total = _operations.length;
      throw Exception(
        'Unknown tensor operation: "$name". '
        'Available operations (${total} total): $available${total > 10 ? ', ...' : ''}',
      );
    }

    if (args.length < opInfo.requiredArgs) {
      throw ArgumentError(
        'Operation $name requires at least ${opInfo.requiredArgs} arguments, got ${args.length}',
      );
    }

    try {
      return await opInfo.executor(input, args, namedArgs);
    } catch (e) {
      if (e is UnsupportedError) {
        rethrow;
      }
      throw Exception('Failed to execute tensor operation $name: $e');
    }
  }

  static bool hasOperation(String name) => _operations.containsKey(name);

  static Set<String> get availableOperations => _operations.keys.toSet();

  // Helper methods for argument extraction
  static double _getScalar(List<dynamic> args, int index) {
    if (index >= args.length) {
      throw ArgumentError('Missing scalar argument at index $index');
    }
    final value = args[index];
    if (value is num) return value.toDouble();
    throw ArgumentError(
      'Expected scalar value at index $index, got ${value.runtimeType}',
    );
  }

  static Tensor _getTensor(List<dynamic> args, int index) {
    if (index >= args.length) {
      throw ArgumentError('Missing tensor argument at index $index');
    }
    final value = args[index];
    if (value is Tensor) return value;
    throw ArgumentError(
      'Expected tensor value at index $index, got ${value.runtimeType}',
    );
  }
}

/// Internal class to hold tensor operation metadata
class TensorOperationInfo {
  final Future<Tensor> Function(Tensor, List<dynamic>, Map<String, dynamic>)
  executor;
  final int requiredArgs;
  final bool preservesShape;

  TensorOperationInfo({
    required this.executor,
    this.requiredArgs = 0,
    this.preservesShape = true,
  });
}
