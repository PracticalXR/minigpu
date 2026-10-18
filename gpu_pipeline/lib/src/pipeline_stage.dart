import 'dart:async';
import 'dart:typed_data';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:gpu_pipeline/src/pipeline_ports.dart';
import 'package:gpu_pipeline/src/pipeline_properties.dart';
import 'package:gpu_pipeline/src/pipeline_assets.dart';
import 'package:gpu_pipeline/src/pipeline_resources.dart';
import 'pipeline_event.dart';

/// Declares whether a stage's per-frame work must observe strict submission
/// order, or whether it may run concurrently with adjacent frames.
///
///   * [ordered]   — the default. Frame N must fully complete before frame
///     N+1 starts. Required for data-parallel GPU render/compute stages whose
///     output feeds the next frame, and for any **stateful** CPU stage (e.g. a
///     closed-loop video encoder mutating reference / entropy state).
///   * [asyncSafe] — the stage is a pure, per-frame function of its inputs and
///     holds no cross-frame state, so the scheduler may overlap it with other
///     frames (e.g. run it on a worker isolate while the GPU submits the next
///     frame). Marking a stateful stage [asyncSafe] is undefined behaviour.
enum StageConcurrency { ordered, asyncSafe }

/// Enhanced pipeline stage interface with I/O ports, parameters, and assets
abstract class PipelineStage with StageProperties {
  final String stageId;
  final bool isPersistent;

  /// How the scheduler may treat this stage relative to adjacent frames.
  /// Defaults to [StageConcurrency.ordered] (safe for every stage).
  StageConcurrency stageConcurrency = StageConcurrency.ordered;

  // I/O port collections
  final InputPortCollection _inputPorts = InputPortCollection();
  final OutputPortCollection _outputPorts = OutputPortCollection();

  // Asset requirements
  final List<StaticAsset> _requiredAssets = [];

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

  // Resource Management
  ResourceRequirements? get resourceRequirements => _resourceRequirements;
  MemoryStrategy? get memoryStrategy => _memoryStrategy;
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
  Future<void> preloadAssets(AssetManager assetManager) async {
    _assetManager = assetManager;

    for (final asset in _requiredAssets) {
      await assetManager.preload(asset);
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

  // Resource Configuration (protected methods for subclasses)
  void setResourceRequirements(ResourceRequirements requirements) {
    _resourceRequirements = requirements;
  }

  void setMemoryStrategy(MemoryStrategy strategy) {
    _memoryStrategy = strategy;
  }

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
