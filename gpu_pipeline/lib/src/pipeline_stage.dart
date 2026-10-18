import 'dart:async';
import 'dart:typed_data';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:gpu_pipeline/src/pipeline_ports.dart';
import 'package:gpu_pipeline/src/pipeline_properties.dart';
import 'package:gpu_pipeline/src/pipeline_assets.dart';
import 'package:gpu_pipeline/src/pipeline_resources.dart';
import 'pipeline_event.dart';

/// A scheduling hint the pipeline never read: [asyncSafe] was meant to let
/// a stateless stage overlap adjacent frames, but every stage runs the same
/// way whichever value it holds.
@Deprecated(unusedResourceHint)
enum StageConcurrency { ordered, asyncSafe }

/// Enhanced pipeline stage interface with I/O ports, parameters, and assets
abstract class PipelineStage with StageProperties {
  final String stageId;
  final bool isPersistent;

  /// Unused by the pipeline; see [StageConcurrency].
  @Deprecated(unusedResourceHint)
  StageConcurrency stageConcurrency = StageConcurrency.ordered;

  // I/O port collections
  final InputPortCollection _inputPorts = InputPortCollection();
  final OutputPortCollection _outputPorts = OutputPortCollection();

  // Asset requirements
  final List<StaticAsset> _requiredAssets = [];

  // The assets preloadAssets took a reference to
  final List<String> _heldAssets = [];

  // Resource requirements
  ResourceRequirements? _resourceRequirements;
  MemoryStrategy? _memoryStrategy;
  RealTimeConstraints? _realTimeConstraints;

  // Event and cache management
  PipelineEventNotifier? _eventNotifier;
  AssetManager? _assetManager;

  PipelineStage({required this.stageId, this.isPersistent = false});

  // I/O Port System
  List<InputPort> get inputPorts => _inputPorts.all;
  List<OutputPort> get outputPorts => _outputPorts.all;
  List<StaticAsset> get requiredAssets => List.unmodifiable(_requiredAssets);

  // Resource hints. The pipeline reads none of them: they are stored and
  // returned as set.
  @Deprecated(unusedResourceHint)
  ResourceRequirements? get resourceRequirements => _resourceRequirements;
  @Deprecated(unusedResourceHint)
  MemoryStrategy? get memoryStrategy => _memoryStrategy;
  @Deprecated(unusedResourceHint)
  RealTimeConstraints? get realTimeConstraints => _realTimeConstraints;

  // Legacy compatibility - deprecated
  @Deprecated('Use inputPorts instead')
  List<String> get inputFormats => ['tensor'];

  @Deprecated('Use outputPorts instead')
  String get outputFormat => 'tensor';

  /// Main processing method - override this in your stage implementations
  ///
  /// The pipeline framework will:
  /// 1. Validate inputs against your stage's input ports (if any)
  /// 2. Pre-allocate output tensors based on your output ports (if any)
  /// 3. Call your process method with validated inputs and pre-allocated outputs
  /// 4. Return the outputs you populate
  ///
  /// For stages without explicit I/O ports, this works exactly like before.
  Future<Map<String, Tensor>> process(Map<String, Tensor> inputs);

  // Asset Management

  /// Loads this stage's assets into [assetManager], which the asset getters
  /// then read, and holds each until [releaseAssets]. An asset whose
  /// [StaticAsset.isRequired] is false and that fails to load is skipped;
  /// the manager reports it with a `failed` event.
  Future<void> preloadAssets(AssetManager assetManager) async {
    _assetManager = assetManager;

    for (final asset in _requiredAssets) {
      try {
        await assetManager.preload(asset);
      } catch (_) {
        if (asset.isRequired) rethrow;
        continue;
      }
      _heldAssets.add(asset.name);
    }
  }

  /// Drops the references [preloadAssets] took. The pipeline calls it when
  /// it stops.
  Future<void> releaseAssets() async {
    final held = List.of(_heldAssets);
    _heldAssets.clear();
    final manager = _assetManager;
    if (manager == null) return;
    for (final name in held) {
      await manager.release(name);
    }
  }

  /// Gets a preloaded tensor asset
  Tensor? getAssetTensor(String name) {
    return _assetManager?.getPreloadedTensor(name);
  }

  /// Gets a preloaded config asset
  Map<String, dynamic>? getAssetConfig(String name) {
    return _assetManager?.getPreloadedConfig(name);
  }

  /// Gets preloaded binary asset
  Uint8List? getAssetBinary(String name) {
    return _assetManager?.getPreloadedBinary(name);
  }

  // Conditional Execution
  bool shouldExecute(Map<String, Tensor> inputs, Map<String, dynamic> metrics) {
    return true; // Default: always execute
  }

  // Error Recovery
  Future<Map<String, Tensor>> handleError(
    Exception error,
    Map<String, Tensor> inputs,
  ) async {
    // Default: rethrow error
    throw error;
  }

  @Deprecated(unusedResourceHint)
  void setResourceRequirements(ResourceRequirements requirements) {
    _resourceRequirements = requirements;
  }

  @Deprecated(unusedResourceHint)
  void setMemoryStrategy(MemoryStrategy strategy) {
    _memoryStrategy = strategy;
  }

  @Deprecated(unusedResourceHint)
  void setRealTimeConstraints(RealTimeConstraints constraints) {
    _realTimeConstraints = constraints;
  }

  // I/O Port Management (protected methods for builders)
  void addInputPort(InputPort port) {
    _inputPorts.add(port);
  }

  void addOutputPort(OutputPort port) {
    _outputPorts.add(port);
  }

  /// Set the complete input port collection (for builders)
  void setInputPorts(InputPortCollection inputPorts) {
    // Clear existing ports and add new ones
    _inputPorts.clear();
    for (final port in inputPorts.all) {
      _inputPorts.add(port);
    }
  }

  /// Set the complete output port collection (for builders)
  void setOutputPorts(OutputPortCollection outputPorts) {
    // Clear existing ports and add new ones
    _outputPorts.clear();
    for (final port in outputPorts.all) {
      _outputPorts.add(port);
    }
  }

  /// Get input port collection (for internal use)
  InputPortCollection get inputPortCollection => _inputPorts;

  /// Get output port collection (for internal use)
  OutputPortCollection get outputPortCollection => _outputPorts;

  void addAsset(StaticAsset asset) {
    _requiredAssets.add(asset);
  }

  // Event Management
  void setEventNotifier(PipelineEventNotifier eventNotifier) {
    _eventNotifier = eventNotifier;
  }

  void emitEvent(PipelineEvent event) {
    _eventNotifier?.emit(event);
  }

  // Lifecycle Management
  Future<void> initializeStage() async {
    // Default implementation - override if needed
  }

  Future<void> dispose() async {
    // Default implementation - override if needed
  }

  @override
  String toString() => 'PipelineStage($stageId)';
}
