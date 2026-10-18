import 'dart:async';
import 'package:gpu_pipeline/src/stages/pipeline_resource_cache.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:gpu_pipeline/src/pipeline_resources.dart';

/// Base class for all pipeline stage operations
abstract class StageOperation with CacheAwareOperation, ArgumentResolvable {
  final String operationId;
  final List<String> inputKeys;
  final List<String> outputKeys;
  final Map<String, dynamic> arguments;
  final Map<String, dynamic> namedArguments;

  // Resource requirements for this operation
  ResourceRequirements? _resourceRequirements;

  // Performance tracking
  Duration? _lastExecutionTime;
  int _executionCount = 0;

  StageOperation({
    required this.operationId,
    this.inputKeys = const ['input'],
    this.outputKeys = const ['output'],
    this.arguments = const {},
    this.namedArguments = const {},
    @Deprecated(unusedResourceHint) ResourceRequirements? resourceRequirements,
  }) : _resourceRequirements = resourceRequirements;

  /// Execute operation with multiple inputs and outputs
  Future<Map<String, Tensor>> execute(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache resourceCache,
  );

  /// Resource requirements for this operation
  @Deprecated(unusedResourceHint)
  ResourceRequirements? get resourceRequirements => _resourceRequirements;

  /// Set resource requirements
  @Deprecated(unusedResourceHint)
  void setResourceRequirements(ResourceRequirements requirements) {
    _resourceRequirements = requirements;
  }

  /// Performance metrics
  Duration? get lastExecutionTime => _lastExecutionTime;
  int get executionCount => _executionCount;

  /// Update performance tracking
  void _updatePerformanceMetrics(Duration executionTime) {
    _lastExecutionTime = executionTime;
    _executionCount++;
  }

  /// Estimate memory usage for this operation
  int estimateMemoryUsage(Map<String, Tensor> inputs) {
    int totalMemory = 0;

    // Input memory
    for (final tensor in inputs.values) {
      totalMemory += tensor.byteSize;
    }

    // Estimated output memory (override in subclasses for better estimates)
    for (final inputTensor in inputs.values) {
      totalMemory += inputTensor.byteSize; // Conservative estimate
    }

    return totalMemory;
  }

  /// Check if operation can run with given resource constraints
  @Deprecated(unusedResourceHint)
  bool canExecuteWithResources(ResourceAllocation allocation) {
    if (_resourceRequirements == null) return true;

    // Check memory
    if (_resourceRequirements!.maxMemoryBytes != null) {
      if (allocation.allocatedMemoryBytes <
          _resourceRequirements!.maxMemoryBytes!) {
        return false;
      }
    }

    // Check GPU availability - fixed to handle List<int>
    if (_resourceRequirements!.preferredGPUs != null &&
        _resourceRequirements!.preferredGPUs!.isNotEmpty) {
      final assignedGPUs = allocation.assignedGPUs;
      final hasMatchingGPU = _resourceRequirements!.preferredGPUs!.any(
        (preferredGPU) => assignedGPUs.contains(preferredGPU),
      );
      if (!hasMatchingGPU) {
        return false;
      }
    }

    return true;
  }

  /// Execute with performance tracking wrapper
  Future<Map<String, Tensor>> executeWithTracking(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache resourceCache,
  ) async {
    final stopwatch = Stopwatch()..start();

    try {
      final result = await execute(inputs, parameters, resourceCache);
      return result;
    } finally {
      stopwatch.stop();
      _updatePerformanceMetrics(stopwatch.elapsed);
    }
  }

  /// Optional cleanup method for GPU resources
  Future<void> dispose() async {}

  /// Serialize operation to configuration
  Map<String, dynamic> toConfig() {
    return {
      'type': runtimeType.toString().toLowerCase().replaceAll(
        'stageoperation',
        '',
      ),
      'operation_id': operationId,
      'input_keys': inputKeys,
      'output_keys': outputKeys,
      'arguments': arguments,
      'named_arguments': namedArguments,
      'resource_requirements': _resourceRequirements?.toString(),
    };
  }

  /// Validate that all required inputs are available
  void validateInputs(Map<String, Tensor> inputs) {
    for (final inputKey in inputKeys) {
      if (!inputs.containsKey(inputKey)) {
        throw Exception(
          'Missing required input: $inputKey for operation $operationId',
        );
      }
    }
  }

  /// Resolve parameter references in arguments
  List<dynamic> resolveArguments(
    List<dynamic> args,
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
  ) {
    return args.map((arg) {
      if (arg is String) {
        if (arg.startsWith('param:')) {
          final paramName = arg.substring(6);
          final value = parameters[paramName];
          if (value == null) {
            throw ParameterException(
              'Parameter "$paramName" not found for operation $operationId',
            );
          }
          return value;
        } else if (arg.startsWith('tensor:')) {
          final tensorKey = arg.substring(7);
          final tensor = inputs[tensorKey];
          if (tensor == null) {
            throw Exception(
              'Tensor "$tensorKey" not found for operation $operationId',
            );
          }
          return tensor;
        } else if (arg.startsWith('shape:')) {
          final tensorKey = arg.substring(6);
          final tensor = inputs[tensorKey];
          if (tensor == null) {
            throw Exception(
              'Tensor "$tensorKey" not found for shape reference in operation $operationId',
            );
          }
          return tensor.shape;
        } else if (arg.startsWith('size:')) {
          final tensorKey = arg.substring(5);
          final tensor = inputs[tensorKey];
          if (tensor == null) {
            throw Exception(
              'Tensor "$tensorKey" not found for size reference in operation $operationId',
            );
          }
          return tensor.byteSize;
        }
      }
      return arg;
    }).toList();
  }

  /// Resolve parameter references in named arguments
  Map<String, dynamic> resolveNamedArguments(
    Map<String, dynamic> namedArgs,
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
  ) {
    return namedArgs.map((key, value) {
      if (value is String) {
        if (value.startsWith('param:')) {
          final paramName = value.substring(6);
          final resolvedValue = parameters[paramName];
          if (resolvedValue == null) {
            throw ParameterException(
              'Parameter "$paramName" not found for operation $operationId',
            );
          }
          return MapEntry(key, resolvedValue);
        } else if (value.startsWith('tensor:')) {
          final tensorKey = value.substring(7);
          final tensor = inputs[tensorKey];
          if (tensor == null) {
            throw Exception(
              'Tensor "$tensorKey" not found for operation $operationId',
            );
          }
          return MapEntry(key, tensor);
        } else if (value.startsWith('shape:')) {
          final tensorKey = value.substring(6);
          final tensor = inputs[tensorKey];
          if (tensor == null) {
            throw Exception(
              'Tensor "$tensorKey" not found for shape reference in operation $operationId',
            );
          }
          return MapEntry(key, tensor.shape);
        }
      }
      return MapEntry(key, value);
    });
  }

  @override
  String toString() => '$runtimeType($operationId)';
}

/// Mixin for operations that can utilize pipeline resource caching
mixin CacheAwareOperation {
  /// Execute operation with access to pipeline resource cache
  Future<Map<String, Tensor>> executeWithCache(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
    ResourceCache resourceCache,
  );

  /// Number of cache hits for performance monitoring
  int get cacheHits;

  /// Cache miss count for performance monitoring
  int get cacheMisses;

  /// Cache hit rate as percentage (provided by mixin)
  double get cacheHitRate {
    final total = cacheHits + cacheMisses;
    return total > 0 ? (cacheHits / total) * 100.0 : 0.0;
  }

  /// Reset cache statistics
  void resetCacheStats();
}

// Add (or extend) a hook for resolving serialized arguments before execution.
// This is a no-op scaffold; integrate with ArgumentResolver when wiring ops.

mixin ArgumentResolvable {
  Map<String, dynamic> get arguments;
  Map<String, dynamic> get namedArguments;

  void applyResolvedArguments(
    List<dynamic> positional,
    Map<String, dynamic> named,
  ) {}
}

class ParameterException implements Exception {
  final String message;
  ParameterException(this.message);

  @override
  String toString() => 'ParameterException: $message';
}
