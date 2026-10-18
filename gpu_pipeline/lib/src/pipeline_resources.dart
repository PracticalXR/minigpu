import 'dart:async';

/// Compute capability levels for GPU hardware
enum ComputeCapability {
  basic, // Basic compute shaders
  advanced, // Advanced features (subgroups, etc.)
  ml, // ML/AI optimized
  raytracing, // Ray tracing support
}

/// Memory tiers for resource allocation
enum MemoryTier {
  gpuLocal, // GPU L1/L2 cache memory
  gpuGlobal, // GPU VRAM
  gpuShared, // Shared GPU memory
  systemRam, // System memory
  storage, // Disk/SSD storage
}

/// Memory pool allocation strategies
enum MemoryPoolStrategy {
  shared, // Shared memory pool across stages
  isolated, // Isolated memory per stage
  adaptive, // Dynamic allocation strategy
}

/// Resource allocation priority levels
enum PriorityLevel {
  background, // Low priority background tasks
  normal, // Standard processing priority
  high, // High priority tasks
  realTime, // Real-time critical tasks
  critical, // System critical tasks
}

/// GPU information structure
class GPUInfo {
  final int id;
  final String name;
  final String vendor;
  final int memoryBytes;
  final ComputeCapability computeCapability;
  final List<String> supportedExtensions;
  final bool isDiscrete;
  final double performanceRating;
  final Map<String, dynamic> metadata;

  const GPUInfo({
    required this.id,
    required this.name,
    required this.vendor,
    required this.memoryBytes,
    required this.computeCapability,
    this.supportedExtensions = const [],
    this.isDiscrete = false,
    this.performanceRating = 1.0,
    this.metadata = const {},
  });

  /// Gets memory in megabytes for easier reading
  double get memoryMB => memoryBytes / (1024 * 1024);

  /// Gets memory in gigabytes for easier reading
  double get memoryGB => memoryBytes / (1024 * 1024 * 1024);

  /// Checks if GPU supports a specific extension
  bool supportsExtension(String extension) {
    return supportedExtensions.contains(extension);
  }

  @override
  String toString() => 'GPUInfo($id: $name, ${memoryGB.toStringAsFixed(1)}GB)';
}

/// Resource requirements specification for stages
class ResourceRequirements {
  final List<int>? preferredGPUs;
  final int? maxMemoryBytes;
  final Duration? maxExecutionTime;
  final bool requiresHardwareCodec;
  final List<String> requiredExtensions;
  final ComputeCapability? minimumComputeCapability;
  final bool allowFallbackToCPU;
  final PriorityLevel priority;
  final Map<String, dynamic> customRequirements;

  const ResourceRequirements({
    this.preferredGPUs,
    this.maxMemoryBytes,
    this.maxExecutionTime,
    this.requiresHardwareCodec = false,
    this.requiredExtensions = const [],
    this.minimumComputeCapability,
    this.allowFallbackToCPU = true,
    this.priority = PriorityLevel.normal,
    this.customRequirements = const {},
  });

  /// Creates requirements for real-time processing
  factory ResourceRequirements.realTime({
    Duration? maxLatency,
    int? maxMemoryMB,
    List<int>? preferredGPUs,
  }) {
    return ResourceRequirements(
      maxExecutionTime: maxLatency ?? Duration(milliseconds: 10),
      maxMemoryBytes: maxMemoryMB != null ? maxMemoryMB * 1024 * 1024 : null,
      preferredGPUs: preferredGPUs,
      priority: PriorityLevel.realTime,
      allowFallbackToCPU: false,
    );
  }

  /// Creates requirements for ML inference
  factory ResourceRequirements.mlInference({
    int? minMemoryGB,
    ComputeCapability? minCapability,
  }) {
    return ResourceRequirements(
      maxMemoryBytes: minMemoryGB != null
          ? minMemoryGB * 1024 * 1024 * 1024
          : null,
      minimumComputeCapability: minCapability ?? ComputeCapability.ml,
      requiresHardwareCodec: true,
      priority: PriorityLevel.high,
      allowFallbackToCPU: false,
    );
  }

  ResourceRequirements copyWith({
    List<int>? preferredGPUs,
    int? maxMemoryBytes,
    Duration? maxExecutionTime,
    bool? requiresHardwareCodec,
    List<String>? requiredExtensions,
    ComputeCapability? minimumComputeCapability,
    bool? allowFallbackToCPU,
    PriorityLevel? priority,
    Map<String, dynamic>? customRequirements,
  }) {
    return ResourceRequirements(
      preferredGPUs: preferredGPUs ?? this.preferredGPUs,
      maxMemoryBytes: maxMemoryBytes ?? this.maxMemoryBytes,
      maxExecutionTime: maxExecutionTime ?? this.maxExecutionTime,
      requiresHardwareCodec:
          requiresHardwareCodec ?? this.requiresHardwareCodec,
      requiredExtensions: requiredExtensions ?? this.requiredExtensions,
      minimumComputeCapability:
          minimumComputeCapability ?? this.minimumComputeCapability,
      allowFallbackToCPU: allowFallbackToCPU ?? this.allowFallbackToCPU,
      priority: priority ?? this.priority,
      customRequirements: customRequirements ?? this.customRequirements,
    );
  }

  /// Creates basic requirements for simple processing
  factory ResourceRequirements.basic({int? maxMemoryMB}) {
    return ResourceRequirements(
      maxMemoryBytes: maxMemoryMB != null ? maxMemoryMB * 1024 * 1024 : null,
      priority: PriorityLevel.normal,
      allowFallbackToCPU: true,
    );
  }

  /// Checks if given GPU meets these requirements
  bool isCompatibleWith(GPUInfo gpu) {
    // Check memory requirements
    if (maxMemoryBytes != null && gpu.memoryBytes < maxMemoryBytes!) {
      return false;
    }

    // Check compute capability
    if (minimumComputeCapability != null) {
      final minIndex = ComputeCapability.values.indexOf(
        minimumComputeCapability!,
      );
      final gpuIndex = ComputeCapability.values.indexOf(gpu.computeCapability);
      if (gpuIndex < minIndex) return false;
    }

    // Check required extensions
    for (final extension in requiredExtensions) {
      if (!gpu.supportsExtension(extension)) return false;
    }

    // Check hardware codec requirement
    if (requiresHardwareCodec && !gpu.metadata.containsKey('hardware_codec')) {
      return false;
    }

    return true;
  }

  @override
  String toString() {
    final parts = <String>[];
    if (maxMemoryBytes != null) {
      parts.add(
        'Memory: ${(maxMemoryBytes! / (1024 * 1024)).toStringAsFixed(0)}MB',
      );
    }
    if (maxExecutionTime != null) {
      parts.add('MaxTime: ${maxExecutionTime!.inMilliseconds}ms');
    }
    parts.add('Priority: ${priority.name}');
    return 'ResourceRequirements(${parts.join(', ')})';
  }
}

/// Memory allocation strategy for stages
class MemoryStrategy {
  final MemoryTier inputTier;
  final MemoryTier workingTier;
  final MemoryTier outputTier;
  final int? memoryBudgetBytes;
  final MemoryPoolStrategy poolStrategy;
  final bool enableCompression;
  final double? compressionRatio;
  final bool allowSwapping;
  final Map<String, dynamic> customSettings;

  const MemoryStrategy({
    this.inputTier = MemoryTier.gpuGlobal,
    this.workingTier = MemoryTier.gpuGlobal,
    this.outputTier = MemoryTier.gpuGlobal,
    this.memoryBudgetBytes,
    this.poolStrategy = MemoryPoolStrategy.shared,
    this.enableCompression = false,
    this.compressionRatio,
    this.allowSwapping = true,
    this.customSettings = const {},
  });

  /// Creates strategy optimized for speed
  factory MemoryStrategy.speed() {
    return const MemoryStrategy(
      inputTier: MemoryTier.gpuLocal,
      workingTier: MemoryTier.gpuLocal,
      outputTier: MemoryTier.gpuLocal,
      poolStrategy: MemoryPoolStrategy.isolated,
      enableCompression: false,
      allowSwapping: false,
    );
  }

  /// Creates strategy optimized for memory efficiency
  factory MemoryStrategy.efficiency() {
    return const MemoryStrategy(
      inputTier: MemoryTier.gpuShared,
      workingTier: MemoryTier.gpuShared,
      outputTier: MemoryTier.gpuShared,
      poolStrategy: MemoryPoolStrategy.shared,
      enableCompression: true,
      compressionRatio: 0.5,
      allowSwapping: true,
    );
  }

  /// Creates strategy for large datasets
  factory MemoryStrategy.largescale({int? budgetGB}) {
    return MemoryStrategy(
      inputTier: MemoryTier.systemRam,
      workingTier: MemoryTier.gpuGlobal,
      outputTier: MemoryTier.systemRam,
      memoryBudgetBytes: budgetGB != null
          ? budgetGB * 1024 * 1024 * 1024
          : null,
      poolStrategy: MemoryPoolStrategy.adaptive,
      enableCompression: true,
      allowSwapping: true,
    );
  }

  /// Gets memory budget in megabytes
  double? get memoryBudgetMB {
    return memoryBudgetBytes != null
        ? memoryBudgetBytes! / (1024 * 1024)
        : null;
  }

  /// Gets memory budget in gigabytes
  double? get memoryBudgetGB {
    return memoryBudgetBytes != null
        ? memoryBudgetBytes! / (1024 * 1024 * 1024)
        : null;
  }

  @override
  String toString() {
    final parts = <String>[];
    parts.add('Pool: ${poolStrategy.name}');
    if (memoryBudgetBytes != null) {
      parts.add('Budget: ${memoryBudgetGB!.toStringAsFixed(1)}GB');
    }
    if (enableCompression) parts.add('Compressed');
    return 'MemoryStrategy(${parts.join(', ')})';
  }
}

/// Real-time processing constraints
class RealTimeConstraints {
  final Duration maxLatency;
  final PriorityLevel priority;
  final bool requiresDeterministicTiming;
  final double? targetFramerate;
  final Duration? maxJitter;
  final bool enableQualityAdaptation;
  final Map<String, dynamic> adaptationSettings;

  const RealTimeConstraints({
    required this.maxLatency,
    this.priority = PriorityLevel.realTime,
    this.requiresDeterministicTiming = false,
    this.targetFramerate,
    this.maxJitter,
    this.enableQualityAdaptation = true,
    this.adaptationSettings = const {},
  });

  /// Creates constraints for audio processing
  factory RealTimeConstraints.audio({
    Duration? bufferDuration,
    int? sampleRate,
  }) {
    final latency =
        bufferDuration ??
        Duration(microseconds: 2900); // ~128 samples at 44.1kHz
    return RealTimeConstraints(
      maxLatency: latency,
      priority: PriorityLevel.realTime,
      requiresDeterministicTiming: true,
      targetFramerate: sampleRate?.toDouble(),
      maxJitter: Duration(microseconds: latency.inMicroseconds ~/ 10),
      enableQualityAdaptation: false, // Audio quality shouldn't adapt
    );
  }

  /// Creates constraints for video processing
  factory RealTimeConstraints.video({
    double fps = 60.0,
    bool allowFrameSkip = true,
  }) {
    final frameTime = Duration(microseconds: (1000000 / fps).round());
    return RealTimeConstraints(
      maxLatency: frameTime,
      priority: PriorityLevel.high,
      targetFramerate: fps,
      maxJitter: Duration(microseconds: frameTime.inMicroseconds ~/ 4),
      enableQualityAdaptation: allowFrameSkip,
      adaptationSettings: {'allow_frame_skip': allowFrameSkip},
    );
  }

  /// Creates constraints for interactive applications
  factory RealTimeConstraints.interactive({Duration? maxResponseTime}) {
    return RealTimeConstraints(
      maxLatency: maxResponseTime ?? Duration(milliseconds: 16), // 60 FPS
      priority: PriorityLevel.high,
      enableQualityAdaptation: true,
      adaptationSettings: {
        'reduce_quality_on_pressure': true,
        'skip_expensive_effects': true,
      },
    );
  }

  /// Checks if timing requirements are met
  bool isSatisfiedBy(Duration actualLatency, {Duration? actualJitter}) {
    if (actualLatency > maxLatency) return false;
    if (maxJitter != null &&
        actualJitter != null &&
        actualJitter > maxJitter!) {
      return false;
    }
    return true;
  }

  @override
  String toString() {
    final parts = <String>[];
    parts.add('MaxLatency: ${maxLatency.inMilliseconds}ms');
    if (targetFramerate != null) {
      parts.add('FPS: ${targetFramerate!.toStringAsFixed(1)}');
    }
    parts.add('Priority: ${priority.name}');
    return 'RealTimeConstraints(${parts.join(', ')})';
  }
}

/// Resource allocation result
class ResourceAllocation {
  final String allocationId;
  final List<int> assignedGPUs;
  final int allocatedMemoryBytes;
  final PriorityLevel assignedPriority;
  final DateTime allocationTime;
  final Duration? estimatedDuration;
  final Map<String, dynamic> allocationDetails;

  const ResourceAllocation({
    required this.allocationId,
    required this.assignedGPUs,
    required this.allocatedMemoryBytes,
    required this.assignedPriority,
    required this.allocationTime,
    this.estimatedDuration,
    this.allocationDetails = const {},
  });

  /// Gets allocated memory in megabytes
  double get allocatedMemoryMB => allocatedMemoryBytes / (1024 * 1024);

  /// Gets allocated memory in gigabytes
  double get allocatedMemoryGB => allocatedMemoryBytes / (1024 * 1024 * 1024);

  /// Checks if allocation is still valid
  bool get isValid {
    if (estimatedDuration == null) return true;
    final elapsed = DateTime.now().difference(allocationTime);
    return elapsed < estimatedDuration!;
  }

  @override
  String toString() {
    return 'ResourceAllocation($allocationId: GPUs${assignedGPUs}, '
        '${allocatedMemoryMB.toStringAsFixed(0)}MB)';
  }
}

/// Resource usage metrics
class ResourceMetrics {
  final DateTime timestamp;
  final Map<int, double> gpuUtilization; // GPU ID -> utilization %
  final Map<int, int> gpuMemoryUsed; // GPU ID -> bytes used
  final int totalMemoryUsed;
  final int totalMemoryAvailable;
  final Duration averageExecutionTime;
  final Duration maxExecutionTime;
  final int activeAllocations;
  final Map<String, dynamic> customMetrics;

  const ResourceMetrics({
    required this.timestamp,
    this.gpuUtilization = const {},
    this.gpuMemoryUsed = const {},
    required this.totalMemoryUsed,
    required this.totalMemoryAvailable,
    required this.averageExecutionTime,
    required this.maxExecutionTime,
    required this.activeAllocations,
    this.customMetrics = const {},
  });

  /// Gets memory utilization percentage
  double get memoryUtilization {
    if (totalMemoryAvailable == 0) return 0.0;
    return (totalMemoryUsed / totalMemoryAvailable) * 100.0;
  }

  /// Gets average GPU utilization
  double get averageGpuUtilization {
    if (gpuUtilization.isEmpty) return 0.0;
    return gpuUtilization.values.reduce((a, b) => a + b) /
        gpuUtilization.length;
  }

  /// Checks if system is under resource pressure
  bool get isUnderPressure {
    return memoryUtilization > 80.0 || averageGpuUtilization > 90.0;
  }

  @override
  String toString() {
    return 'ResourceMetrics(Memory: ${memoryUtilization.toStringAsFixed(1)}%, '
        'GPU: ${averageGpuUtilization.toStringAsFixed(1)}%, '
        'Allocations: $activeAllocations)';
  }
}

/// Performance profiling result
class PerformanceProfile {
  final Duration profileDuration;
  final Map<String, Duration> stageExecutionTimes;
  final Map<String, int> stageMemoryUsage;
  final List<ResourceMetrics> resourceHistory;
  final Map<String, double> bottleneckAnalysis;
  final List<String> recommendations;
  final DateTime profileTime;

  const PerformanceProfile({
    required this.profileDuration,
    required this.stageExecutionTimes,
    required this.stageMemoryUsage,
    required this.resourceHistory,
    required this.bottleneckAnalysis,
    required this.recommendations,
    required this.profileTime,
  });

  /// Gets the slowest stage
  String? get bottleneckStage {
    if (stageExecutionTimes.isEmpty) return null;
    return stageExecutionTimes.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;
  }

  /// Gets total execution time across all stages
  Duration get totalExecutionTime {
    return stageExecutionTimes.values.fold(
      Duration.zero,
      (sum, duration) => sum + duration,
    );
  }

  /// Gets peak memory usage during profiling
  int get peakMemoryUsage {
    return resourceHistory
        .map((metrics) => metrics.totalMemoryUsed)
        .fold(0, (max, usage) => usage > max ? usage : max);
  }

  @override
  String toString() {
    return 'PerformanceProfile(Duration: ${profileDuration.inMilliseconds}ms, '
        'Stages: ${stageExecutionTimes.length}, '
        'Peak Memory: ${(peakMemoryUsage / (1024 * 1024)).toStringAsFixed(0)}MB)';
  }
}

/// Resource governor for managing GPU resources
class ResourceGovernor {
  final List<GPUInfo> _availableGPUs = [];
  final Map<String, ResourceAllocation> _activeAllocations = {};
  StreamController<ResourceMetrics>? _metricsController;
  Timer? _metricsTimer;

  int _maxMemoryBytes = 4 * 1024 * 1024 * 1024; // 4GB default
  bool _autoLoadBalancing = true;
  bool _disposed = false;

  /// Gets available GPUs
  List<GPUInfo> get availableGPUs => List.unmodifiable(_availableGPUs);

  /// Gets stream of resource metrics
  Stream<ResourceMetrics> get resourceMetrics => _metricsController!.stream;

  /// Gets current memory budget
  int get maxMemoryBytes => _maxMemoryBytes;

  /// Initializes the resource governor with detected GPUs
  Future<void> initialize() async {
    if (_disposed) throw StateError('ResourceGovernor has been disposed');

    await _detectGPUs();
    _startMetricsCollection();
  }

  /// Allocates resources based on requirements
  Future<ResourceAllocation> allocateResources(
    ResourceRequirements requirements,
  ) async {
    final compatibleGPUs = _findCompatibleGPUs(requirements);

    if (compatibleGPUs.isEmpty && !requirements.allowFallbackToCPU) {
      throw ResourceAllocationException(
        'No compatible GPUs found for requirements',
      );
    }

    final selectedGPUs = _selectOptimalGPUs(compatibleGPUs, requirements);
    final memoryToAllocate = _calculateMemoryAllocation(
      requirements,
      selectedGPUs,
    );

    final allocation = ResourceAllocation(
      allocationId: _generateAllocationId(),
      assignedGPUs: selectedGPUs.map((gpu) => gpu.id).toList(),
      allocatedMemoryBytes: memoryToAllocate,
      assignedPriority: requirements.priority,
      allocationTime: DateTime.now(),
      estimatedDuration: requirements.maxExecutionTime,
    );

    _activeAllocations[allocation.allocationId] = allocation;
    return allocation;
  }

  /// Deallocates resources
  Future<void> deallocateResources(ResourceAllocation allocation) async {
    _activeAllocations.remove(allocation.allocationId);
    // Implementation would free GPU memory and update tracking
  }

  /// Sets GPU affinity for current context
  Future<void> setGPUAffinity(List<int> gpuIds) async {
    // Implementation would set preferred GPUs for allocation
  }

  /// Sets memory budget limit
  Future<void> setMemoryBudget(int maxBytes) async {
    _maxMemoryBytes = maxBytes;
  }

  /// Gets current memory usage across all allocations
  int getCurrentMemoryUsage() {
    return _activeAllocations.values.fold(
      0,
      (sum, allocation) => sum + allocation.allocatedMemoryBytes,
    );
  }

  /// Enables or disables memory compression
  void enableMemoryCompression(bool enabled) {
    // Implementation would configure memory compression
  }

  /// Enables or disables automatic load balancing
  void enableAutoLoadBalancing(bool enabled) {
    _autoLoadBalancing = enabled;
  }

  /// Manually rebalances workload across GPUs
  Future<void> rebalanceWorkload() async {
    if (!_autoLoadBalancing) return;

    // Implementation would redistribute active allocations
    // based on current GPU utilization
  }

  /// Profiles performance for specified duration
  Future<PerformanceProfile> profileExecution(Duration duration) async {
    final startTime = DateTime.now();
    final initialMetrics = await _collectCurrentMetrics();
    final resourceHistory = <ResourceMetrics>[initialMetrics];

    final timer = Timer.periodic(Duration(milliseconds: 100), (timer) {
      resourceHistory.add(_createCurrentMetrics());
    });

    await Future.delayed(duration);
    timer.cancel();

    return PerformanceProfile(
      profileDuration: duration,
      stageExecutionTimes: {}, // Would be populated by actual profiling
      stageMemoryUsage: {}, // Would be populated by actual profiling
      resourceHistory: resourceHistory,
      bottleneckAnalysis: _analyzeBottlenecks(resourceHistory),
      recommendations: _generateRecommendations(resourceHistory),
      profileTime: startTime,
    );
  }

  /// Disposes the resource governor
  void dispose() {
    if (_disposed) return;

    _disposed = true;
    _metricsTimer?.cancel();
    _metricsTimer = null;

    if (_metricsController != null && !_metricsController!.isClosed) {
      _metricsController!.close();
    }
    _metricsController = null;
  }

  // Private implementation methods

  Future<void> _detectGPUs() async {
    // Mock GPU detection - real implementation would query webgpu but
    // needs minigpu extension
    _availableGPUs.addAll([
      GPUInfo(
        id: 0,
        name: 'Integrated GPU',
        vendor: 'Intel',
        memoryBytes: 2 * 1024 * 1024 * 1024, // 2GB
        computeCapability: ComputeCapability.basic,
        supportedExtensions: ['basic_compute'],
        isDiscrete: false,
        performanceRating: 0.3,
      ),
      GPUInfo(
        id: 1,
        name: 'Discrete GPU',
        vendor: 'NVIDIA',
        memoryBytes: 8 * 1024 * 1024 * 1024, // 8GB
        computeCapability: ComputeCapability.ml,
        supportedExtensions: ['basic_compute', 'ml_ops', 'raytracing'],
        isDiscrete: true,
        performanceRating: 1.0,
      ),
    ]);
  }

  List<GPUInfo> _findCompatibleGPUs(ResourceRequirements requirements) {
    return _availableGPUs
        .where((gpu) => requirements.isCompatibleWith(gpu))
        .toList();
  }

  List<GPUInfo> _selectOptimalGPUs(
    List<GPUInfo> compatibleGPUs,
    ResourceRequirements requirements,
  ) {
    if (compatibleGPUs.isEmpty) return [];

    // Sort by performance rating and select best
    compatibleGPUs.sort(
      (a, b) => b.performanceRating.compareTo(a.performanceRating),
    );

    // For now, just return the best GPU. Real implementation would consider
    // current load, memory availability, etc.
    return [compatibleGPUs.first];
  }

  int _calculateMemoryAllocation(
    ResourceRequirements requirements,
    List<GPUInfo> selectedGPUs,
  ) {
    if (requirements.maxMemoryBytes != null) {
      return requirements.maxMemoryBytes!;
    }

    // Default allocation based on GPU memory
    if (selectedGPUs.isNotEmpty) {
      return (selectedGPUs.first.memoryBytes * 0.8)
          .round(); // Use 80% of GPU memory
    }

    return 1024 * 1024 * 1024; // 1GB default
  }

  String _generateAllocationId() {
    return 'alloc_${DateTime.now().millisecondsSinceEpoch}';
  }

  void _startMetricsCollection() {
    if (_disposed) return;

    _metricsController = StreamController<ResourceMetrics>.broadcast();

    _metricsTimer?.cancel();
    _metricsTimer ??= Timer.periodic(Duration(seconds: 1), (timer) {
      if (_disposed) {
        timer.cancel();
        return;
      }

      if (_metricsController != null && !_metricsController!.isClosed) {
        final metrics = _createCurrentMetrics();
        _metricsController!.add(metrics);
      }
    });
  }

  Future<ResourceMetrics> _collectCurrentMetrics() async {
    return _createCurrentMetrics();
  }

  ResourceMetrics _createCurrentMetrics() {
    final memoryUsed = getCurrentMemoryUsage();

    return ResourceMetrics(
      timestamp: DateTime.now(),
      gpuUtilization: {
        for (final gpu in _availableGPUs)
          gpu.id: 50.0 + (gpu.id * 10.0), // Mock data
      },
      gpuMemoryUsed: {
        for (final gpu in _availableGPUs)
          gpu.id: memoryUsed ~/ _availableGPUs.length,
      },
      totalMemoryUsed: memoryUsed,
      totalMemoryAvailable: _maxMemoryBytes,
      averageExecutionTime: Duration(milliseconds: 5),
      maxExecutionTime: Duration(milliseconds: 20),
      activeAllocations: _activeAllocations.length,
    );
  }

  Map<String, double> _analyzeBottlenecks(List<ResourceMetrics> history) {
    // Analyze resource history to identify bottlenecks
    final avgMemoryUtil =
        history
            .map((m) => m.memoryUtilization)
            .fold(0.0, (sum, util) => sum + util) /
        history.length;

    final avgGpuUtil =
        history
            .map((m) => m.averageGpuUtilization)
            .fold(0.0, (sum, util) => sum + util) /
        history.length;

    return {
      'memory_pressure': avgMemoryUtil,
      'gpu_utilization': avgGpuUtil,
      'allocation_pressure': _activeAllocations.length.toDouble(),
    };
  }

  List<String> _generateRecommendations(List<ResourceMetrics> history) {
    final recommendations = <String>[];
    final analysis = _analyzeBottlenecks(history);

    if (analysis['memory_pressure']! > 80.0) {
      recommendations.add(
        'Consider increasing memory budget or enabling compression',
      );
    }

    if (analysis['gpu_utilization']! > 90.0) {
      recommendations.add(
        'GPU utilization high - consider load balancing or reducing quality',
      );
    }

    if (analysis['allocation_pressure']! > 10) {
      recommendations.add(
        'High number of concurrent allocations - consider resource pooling',
      );
    }

    return recommendations;
  }
}

/// Exception thrown when resource allocation fails
class ResourceAllocationException implements Exception {
  final String message;
  const ResourceAllocationException(this.message);

  @override
  String toString() => 'ResourceAllocationException: $message';
}

/// Distribution strategy for multi-GPU workloads
enum DistributionStrategy {
  replicated, // Replicate data across all GPUs
  partitioned, // Partition data across GPUs
  pipelined, // Pipeline stages across GPUs
  adaptive, // Adaptive distribution based on workload
}
