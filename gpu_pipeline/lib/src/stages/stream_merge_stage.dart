import 'dart:async';
import 'package:gpu_tensor/gpu_tensor.dart';
import '../pipeline_stage.dart';
import '../pipeline_properties.dart';

/// Strategies for merging multiple streams
enum MergeStrategy {
  average, // Average all tensors
  sum, // Sum all tensors
  maximum, // Element-wise maximum
  minimum, // Element-wise minimum
  first, // Use first available tensor
  last, // Use last received tensor
  weightedAverage, // Weighted average
}

/// Stage for merging multiple input streams with parameter support.
///
/// With `enable_shape_validation` (the default), streams whose shape
/// differs from the first stream's are left out of the merge.
/// [MergeStrategy.weightedAverage] falls back to the average when the
/// weights are missing, do not match the number of streams, or sum to zero.
/// When a merge fails, the pipeline reports a stage error and the stage
/// passes the first stream on unmerged.
class StreamMergeStage extends PipelineStage {
  final MergeStrategy _defaultStrategy;
  final Map<String, Tensor> _streamTensors = {};
  final Set<String> _expectedStreams = {};

  StreamMergeStage({
    required super.stageId,
    MergeStrategy strategy = MergeStrategy.average,
    List<double>? weights,
    super.isPersistent = false,
  }) : _defaultStrategy = strategy {
    // Add merge strategy parameter
    addParameter(
      Parameter.stageEnum<MergeStrategy>(
        'merge_strategy',
        defaultValue: strategy,
        allValues: MergeStrategy.values,
        description: 'Strategy for merging multiple streams',
      ),
    );

    // Always add weights parameter, even if null initially
    addParameter(
      Parameter.stageFloatList(
        'weights',
        defaultValue:
            weights ?? [], // Use empty list as default instead of null
        description: 'Weights for weighted average merging',
      ),
    );

    // Add merge threshold parameter
    addParameter(
      Parameter.stageFloat(
        'merge_threshold',
        defaultValue: 0.5,
        min: 0.0,
        max: 1.0,
        description: 'Threshold for merge quality',
      ),
    );

    // Add enable_shape_validation parameter
    addParameter(
      Parameter.stageBool(
        'enable_shape_validation',
        defaultValue: true,
        description: 'Enable strict shape validation for merging',
      ),
    );
  }

  /// Set which streams to expect for merging
  void expectStreams(Set<String> streamIds) {
    _expectedStreams.clear();
    _expectedStreams.addAll(streamIds);
  }

  @override
  Future<Map<String, Tensor>> process(
    Map<String, Tensor> inputs, [
    Map<String, Tensor>? outputs,
  ]) async {
    // Get current parameter values with safe defaults
    final strategy =
        getPropertyValue<MergeStrategy>('merge_strategy') ?? _defaultStrategy;

    // Safe weights retrieval
    List<double>? weights;
    try {
      final weightsList = getPropertyValue<List<double>>('weights');
      weights = (weightsList?.isNotEmpty == true) ? weightsList : null;
    } catch (e) {
      // Parameter doesn't exist or is wrong type, use null
      weights = null;
    }

    final enableShapeValidation =
        getPropertyValue<bool>('enable_shape_validation') ?? true;

    // Store all input tensors
    for (final entry in inputs.entries) {
      _streamTensors[entry.key] = entry.value;
    }

    // If we have expected streams, check if we have enough
    if (_expectedStreams.isNotEmpty) {
      final availableExpected = _expectedStreams
          .where((streamId) => _streamTensors.containsKey(streamId))
          .toList();

      if (availableExpected.length < _expectedStreams.length) {
        // Return the current input as passthrough while waiting
        if (inputs.containsKey('input')) {
          return {'input': inputs['input']!, 'output': inputs['input']!};
        }

        // Return empty result to indicate we're waiting
        return {};
      }
    }

    // Collect tensors to merge
    final tensorsToMerge = <Tensor>[];
    final tensorKeys = <String>[];

    // Priority order: stream_* tensors, then 'input', then any other tensors
    for (final entry in _streamTensors.entries) {
      if (entry.key.startsWith('stream_')) {
        tensorsToMerge.add(entry.value);
        tensorKeys.add(entry.key);
      }
    }

    // If no stream tensors found, use 'input'
    if (tensorsToMerge.isEmpty && _streamTensors.containsKey('input')) {
      tensorsToMerge.add(_streamTensors['input']!);
      tensorKeys.add('input');
    }

    // If still no tensors, use any available
    if (tensorsToMerge.isEmpty) {
      for (final entry in _streamTensors.entries) {
        tensorsToMerge.add(entry.value);
        tensorKeys.add(entry.key);
      }
    }

    if (tensorsToMerge.isEmpty) {
      // Final fallback - use direct input if available
      if (inputs.containsKey('input')) {
        return {'input': inputs['input']!, 'output': inputs['input']!};
      }

      throw Exception('No tensors available for merging');
    }

    // Merge the tensors
    final mergedTensor = await _mergeStreamTensors(
      tensorsToMerge,
      strategy,
      weights,
      enableShapeValidation,
    );

    // Clear stored tensors after successful merge
    _streamTensors.clear();

    // Return merged result
    return {
      'input': mergedTensor,
      'output': mergedTensor, // Provide both keys for compatibility
    };
  }

  /// Passes the first stored stream on, or else the `input` tensor, after a
  /// merge fails. The pipeline reports the failure first.
  @override
  Future<Map<String, Tensor>> handleError(
    Exception error,
    Map<String, Tensor> inputs,
  ) async {
    if (_streamTensors.isNotEmpty) {
      final fallback = _streamTensors.values.first;
      _streamTensors.clear();
      return {'input': fallback, 'output': fallback};
    }
    final input = inputs['input'];
    if (input != null) return {'input': input, 'output': input};
    throw error;
  }

  Future<Tensor> _mergeStreamTensors(
    List<Tensor> tensors,
    MergeStrategy strategy,
    List<double>? weights,
    bool enableShapeValidation,
  ) async {
    if (tensors.isEmpty) {
      throw Exception('No tensors to merge');
    }

    if (tensors.length == 1) {
      return tensors.first;
    }

    // Shape validation
    if (enableShapeValidation) {
      final referenceShape = tensors.first.shape;
      final compatibleTensors = <Tensor>[];

      for (int i = 0; i < tensors.length; i++) {
        final tensor = tensors[i];
        if (_shapesEqual(tensor.shape, referenceShape)) {
          compatibleTensors.add(tensor);
        }
      }

      if (compatibleTensors.isEmpty) {
        throw Exception('No compatible tensors found for merging');
      }

      if (compatibleTensors.length < tensors.length) {
        tensors = compatibleTensors;
      }
    }

    // Perform merge based on strategy
    switch (strategy) {
      case MergeStrategy.average:
        return await _averageTensors(tensors);
      case MergeStrategy.sum:
        return await _sumTensors(tensors);
      case MergeStrategy.maximum:
        return await _maximumTensors(tensors);
      case MergeStrategy.minimum:
        return await _minimumTensors(tensors);
      case MergeStrategy.first:
        return tensors.first;
      case MergeStrategy.last:
        return tensors.last;
      case MergeStrategy.weightedAverage:
        return await _weightedAverageTensors(tensors, weights);
    }
  }

  Future<Tensor> _averageTensors(List<Tensor> tensors) async {
    var result = tensors.first;
    for (int i = 1; i < tensors.length; i++) {
      result = await result.add(tensors[i]);
    }

    final scale = 1.0 / tensors.length;
    return await result.multiplyScalar(scale);
  }

  Future<Tensor> _sumTensors(List<Tensor> tensors) async {
    var result = tensors.first;
    for (int i = 1; i < tensors.length; i++) {
      result = await result.add(tensors[i]);
    }

    return result;
  }

  Future<Tensor> _maximumTensors(List<Tensor> tensors) async {
    var result = tensors.first;
    for (int i = 1; i < tensors.length; i++) {
      result = await result.max(tensors[i]);
    }

    return result;
  }

  Future<Tensor> _minimumTensors(List<Tensor> tensors) async {
    var result = tensors.first;
    for (int i = 1; i < tensors.length; i++) {
      result = await result.min(tensors[i]);
    }

    return result;
  }

  Future<Tensor> _weightedAverageTensors(
    List<Tensor> tensors,
    List<double>? weights,
  ) async {
    if (weights == null ||
        weights.isEmpty ||
        weights.length != tensors.length) {
      return await _averageTensors(tensors);
    }

    // Normalize weights
    final weightSum = weights.reduce((a, b) => a + b);
    if (weightSum == 0) {
      return await _averageTensors(tensors);
    }

    final normalizedWeights = weights.map((w) => w / weightSum).toList();

    var result = await tensors.first.multiplyScalar(normalizedWeights.first);

    for (int i = 1; i < tensors.length; i++) {
      final weightedTensor = await tensors[i].multiplyScalar(
        normalizedWeights[i],
      );
      result = await result.add(weightedTensor);
    }

    return result;
  }

  bool _shapesEqual(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  @override
  Future<void> dispose() async {
    _streamTensors.clear();
    _expectedStreams.clear();
    await super.dispose();
  }

  @override
  String toString() =>
      'StreamMergeStage($stageId, strategy: ${_defaultStrategy})';
}
