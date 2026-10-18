import 'dart:async';
import 'package:test/test.dart';
import 'package:gpu_pipeline/src/pipeline_resources.dart';

void main() {
  group('GPUInfo', () {
    test('creates GPU info with basic properties', () {
      final gpu = GPUInfo(
        id: 0,
        name: 'Test GPU',
        vendor: 'TestVendor',
        memoryBytes: 4 * 1024 * 1024 * 1024, // 4GB
        computeCapability: ComputeCapability.advanced,
      );

      expect(gpu.id, equals(0));
      expect(gpu.name, equals('Test GPU'));
      expect(gpu.vendor, equals('TestVendor'));
      expect(gpu.memoryBytes, equals(4 * 1024 * 1024 * 1024));
      expect(gpu.computeCapability, equals(ComputeCapability.advanced));
    });

    test('calculates memory sizes correctly', () {
      final gpu = GPUInfo(
        id: 0,
        name: 'Test GPU',
        vendor: 'TestVendor',
        memoryBytes: 8 * 1024 * 1024 * 1024, // 8GB
        computeCapability: ComputeCapability.basic,
      );

      expect(gpu.memoryMB, equals(8192.0));
      expect(gpu.memoryGB, equals(8.0));
    });

    test('checks extension support', () {
      final gpu = GPUInfo(
        id: 0,
        name: 'Test GPU',
        vendor: 'TestVendor',
        memoryBytes: 1024 * 1024 * 1024,
        computeCapability: ComputeCapability.ml,
        supportedExtensions: ['ml_ops', 'compute_shaders', 'raytracing'],
      );

      expect(gpu.supportsExtension('ml_ops'), isTrue);
      expect(gpu.supportsExtension('raytracing'), isTrue);
      expect(gpu.supportsExtension('vulkan'), isFalse);
    });

    test('toString provides readable format', () {
      final gpu = GPUInfo(
        id: 1,
        name: 'RTX 4090',
        vendor: 'NVIDIA',
        memoryBytes: 24 * 1024 * 1024 * 1024, // 24GB
        computeCapability: ComputeCapability.ml,
      );

      final str = gpu.toString();
      expect(str, contains('1'));
      expect(str, contains('RTX 4090'));
      expect(str, contains('24.0GB'));
    });
  });

  group('ResourceRequirements', () {
    test('creates basic requirements', () {
      final requirements = ResourceRequirements(
        maxMemoryBytes: 2 * 1024 * 1024 * 1024, // 2GB
        priority: PriorityLevel.high,
        allowFallbackToCPU: false,
      );

      expect(requirements.maxMemoryBytes, equals(2 * 1024 * 1024 * 1024));
      expect(requirements.priority, equals(PriorityLevel.high));
      expect(requirements.allowFallbackToCPU, isFalse);
    });

    test('creates real-time requirements', () {
      final requirements = ResourceRequirements.realTime(
        maxLatency: Duration(milliseconds: 5),
        maxMemoryMB: 512,
        preferredGPUs: [0, 1],
      );

      expect(requirements.maxExecutionTime, equals(Duration(milliseconds: 5)));
      expect(requirements.maxMemoryBytes, equals(512 * 1024 * 1024));
      expect(requirements.preferredGPUs, equals([0, 1]));
      expect(requirements.priority, equals(PriorityLevel.realTime));
      expect(requirements.allowFallbackToCPU, isFalse);
    });

    test('creates ML inference requirements', () {
      final requirements = ResourceRequirements.mlInference(
        minMemoryGB: 8,
        minCapability: ComputeCapability.ml,
      );

      expect(requirements.maxMemoryBytes, equals(8 * 1024 * 1024 * 1024));
      expect(
        requirements.minimumComputeCapability,
        equals(ComputeCapability.ml),
      );
      expect(requirements.requiresHardwareCodec, isTrue);
      expect(requirements.priority, equals(PriorityLevel.high));
    });

    test('creates basic requirements factory', () {
      final requirements = ResourceRequirements.basic(maxMemoryMB: 256);

      expect(requirements.maxMemoryBytes, equals(256 * 1024 * 1024));
      expect(requirements.priority, equals(PriorityLevel.normal));
      expect(requirements.allowFallbackToCPU, isTrue);
    });

    test('checks GPU compatibility - memory', () {
      final requirements = ResourceRequirements(
        maxMemoryBytes: 4 * 1024 * 1024 * 1024, // 4GB required
      );

      final smallGpu = GPUInfo(
        id: 0,
        name: 'Small GPU',
        vendor: 'Test',
        memoryBytes: 2 * 1024 * 1024 * 1024, // 2GB available
        computeCapability: ComputeCapability.basic,
      );

      final largeGpu = GPUInfo(
        id: 1,
        name: 'Large GPU',
        vendor: 'Test',
        memoryBytes: 8 * 1024 * 1024 * 1024, // 8GB available
        computeCapability: ComputeCapability.basic,
      );

      expect(requirements.isCompatibleWith(smallGpu), isFalse);
      expect(requirements.isCompatibleWith(largeGpu), isTrue);
    });

    test('checks GPU compatibility - compute capability', () {
      final requirements = ResourceRequirements(
        minimumComputeCapability: ComputeCapability.ml,
      );

      final basicGpu = GPUInfo(
        id: 0,
        name: 'Basic GPU',
        vendor: 'Test',
        memoryBytes: 1024 * 1024 * 1024,
        computeCapability: ComputeCapability.basic,
      );

      final mlGpu = GPUInfo(
        id: 1,
        name: 'ML GPU',
        vendor: 'Test',
        memoryBytes: 1024 * 1024 * 1024,
        computeCapability: ComputeCapability.ml,
      );

      final rayTracingGpu = GPUInfo(
        id: 2,
        name: 'RT GPU',
        vendor: 'Test',
        memoryBytes: 1024 * 1024 * 1024,
        computeCapability: ComputeCapability.raytracing,
      );

      expect(requirements.isCompatibleWith(basicGpu), isFalse);
      expect(requirements.isCompatibleWith(mlGpu), isTrue);
      expect(requirements.isCompatibleWith(rayTracingGpu), isTrue);
    });

    test('checks GPU compatibility - extensions', () {
      final requirements = ResourceRequirements(
        requiredExtensions: ['ml_ops', 'compute_shaders'],
      );

      final partialGpu = GPUInfo(
        id: 0,
        name: 'Partial GPU',
        vendor: 'Test',
        memoryBytes: 1024 * 1024 * 1024,
        computeCapability: ComputeCapability.basic,
        supportedExtensions: ['compute_shaders'], // Missing ml_ops
      );

      final fullGpu = GPUInfo(
        id: 1,
        name: 'Full GPU',
        vendor: 'Test',
        memoryBytes: 1024 * 1024 * 1024,
        computeCapability: ComputeCapability.basic,
        supportedExtensions: ['ml_ops', 'compute_shaders', 'raytracing'],
      );

      expect(requirements.isCompatibleWith(partialGpu), isFalse);
      expect(requirements.isCompatibleWith(fullGpu), isTrue);
    });
  });

  group('MemoryStrategy', () {
    test('creates default strategy', () {
      final strategy = MemoryStrategy();

      expect(strategy.inputTier, equals(MemoryTier.gpuGlobal));
      expect(strategy.workingTier, equals(MemoryTier.gpuGlobal));
      expect(strategy.outputTier, equals(MemoryTier.gpuGlobal));
      expect(strategy.poolStrategy, equals(MemoryPoolStrategy.shared));
      expect(strategy.enableCompression, isFalse);
      expect(strategy.allowSwapping, isTrue);
    });

    test('creates speed-optimized strategy', () {
      final strategy = MemoryStrategy.speed();

      expect(strategy.inputTier, equals(MemoryTier.gpuLocal));
      expect(strategy.workingTier, equals(MemoryTier.gpuLocal));
      expect(strategy.outputTier, equals(MemoryTier.gpuLocal));
      expect(strategy.poolStrategy, equals(MemoryPoolStrategy.isolated));
      expect(strategy.enableCompression, isFalse);
      expect(strategy.allowSwapping, isFalse);
    });

    test('creates efficiency-optimized strategy', () {
      final strategy = MemoryStrategy.efficiency();

      expect(strategy.inputTier, equals(MemoryTier.gpuShared));
      expect(strategy.workingTier, equals(MemoryTier.gpuShared));
      expect(strategy.outputTier, equals(MemoryTier.gpuShared));
      expect(strategy.poolStrategy, equals(MemoryPoolStrategy.shared));
      expect(strategy.enableCompression, isTrue);
      expect(strategy.compressionRatio, equals(0.5));
      expect(strategy.allowSwapping, isTrue);
    });

    test('creates large-scale strategy', () {
      final strategy = MemoryStrategy.largescale(budgetGB: 16);

      expect(strategy.inputTier, equals(MemoryTier.systemRam));
      expect(strategy.workingTier, equals(MemoryTier.gpuGlobal));
      expect(strategy.outputTier, equals(MemoryTier.systemRam));
      expect(strategy.memoryBudgetBytes, equals(16 * 1024 * 1024 * 1024));
      expect(strategy.poolStrategy, equals(MemoryPoolStrategy.adaptive));
      expect(strategy.enableCompression, isTrue);
    });

    test('calculates memory budget sizes', () {
      final strategy = MemoryStrategy(
        memoryBudgetBytes: 4 * 1024 * 1024 * 1024, // 4GB
      );

      expect(strategy.memoryBudgetMB, equals(4096.0));
      expect(strategy.memoryBudgetGB, equals(4.0));
    });

    test('handles null memory budget', () {
      final strategy = MemoryStrategy();

      expect(strategy.memoryBudgetMB, isNull);
      expect(strategy.memoryBudgetGB, isNull);
    });
  });

  group('RealTimeConstraints', () {
    test('creates basic constraints', () {
      final constraints = RealTimeConstraints(
        maxLatency: Duration(milliseconds: 10),
        priority: PriorityLevel.high,
        targetFramerate: 60.0,
      );

      expect(constraints.maxLatency, equals(Duration(milliseconds: 10)));
      expect(constraints.priority, equals(PriorityLevel.high));
      expect(constraints.targetFramerate, equals(60.0));
      expect(constraints.enableQualityAdaptation, isTrue);
    });

    test('creates audio constraints', () {
      final constraints = RealTimeConstraints.audio(
        bufferDuration: Duration(milliseconds: 5),
        sampleRate: 48000,
      );

      expect(constraints.maxLatency, equals(Duration(milliseconds: 5)));
      expect(constraints.priority, equals(PriorityLevel.realTime));
      expect(constraints.requiresDeterministicTiming, isTrue);
      expect(constraints.targetFramerate, equals(48000.0));
      expect(constraints.enableQualityAdaptation, isFalse);
    });

    test('creates video constraints', () {
      final constraints = RealTimeConstraints.video(
        fps: 30.0,
        allowFrameSkip: false,
      );

      expect(constraints.targetFramerate, equals(30.0));
      expect(constraints.priority, equals(PriorityLevel.high));
      expect(constraints.enableQualityAdaptation, isFalse);
      expect(constraints.adaptationSettings['allow_frame_skip'], isFalse);
    });

    test('creates interactive constraints', () {
      final constraints = RealTimeConstraints.interactive(
        maxResponseTime: Duration(milliseconds: 20),
      );

      expect(constraints.maxLatency, equals(Duration(milliseconds: 20)));
      expect(constraints.priority, equals(PriorityLevel.high));
      expect(constraints.enableQualityAdaptation, isTrue);
      expect(
        constraints.adaptationSettings['reduce_quality_on_pressure'],
        isTrue,
      );
    });

    test('validates timing satisfaction', () {
      final constraints = RealTimeConstraints(
        maxLatency: Duration(milliseconds: 10),
        maxJitter: Duration(milliseconds: 2),
      );

      expect(constraints.isSatisfiedBy(Duration(milliseconds: 8)), isTrue);
      expect(constraints.isSatisfiedBy(Duration(milliseconds: 12)), isFalse);

      expect(
        constraints.isSatisfiedBy(
          Duration(milliseconds: 8),
          actualJitter: Duration(milliseconds: 1),
        ),
        isTrue,
      );

      expect(
        constraints.isSatisfiedBy(
          Duration(milliseconds: 8),
          actualJitter: Duration(milliseconds: 3),
        ),
        isFalse,
      );
    });
  });

  group('ResourceAllocation', () {
    test('creates allocation with basic properties', () {
      final allocation = ResourceAllocation(
        allocationId: 'test_alloc_001',
        assignedGPUs: [0, 1],
        allocatedMemoryBytes: 2 * 1024 * 1024 * 1024, // 2GB
        assignedPriority: PriorityLevel.high,
        allocationTime: DateTime.now(),
      );

      expect(allocation.allocationId, equals('test_alloc_001'));
      expect(allocation.assignedGPUs, equals([0, 1]));
      expect(allocation.allocatedMemoryBytes, equals(2 * 1024 * 1024 * 1024));
      expect(allocation.assignedPriority, equals(PriorityLevel.high));
    });

    test('calculates memory sizes correctly', () {
      final allocation = ResourceAllocation(
        allocationId: 'test',
        assignedGPUs: [0],
        allocatedMemoryBytes: 4 * 1024 * 1024 * 1024, // 4GB
        assignedPriority: PriorityLevel.normal,
        allocationTime: DateTime.now(),
      );

      expect(allocation.allocatedMemoryMB, equals(4096.0));
      expect(allocation.allocatedMemoryGB, equals(4.0));
    });

    test('validates allocation validity with duration', () {
      final now = DateTime.now();

      final validAllocation = ResourceAllocation(
        allocationId: 'valid',
        assignedGPUs: [0],
        allocatedMemoryBytes: 1024 * 1024 * 1024,
        assignedPriority: PriorityLevel.normal,
        allocationTime: now,
        estimatedDuration: Duration(hours: 1),
      );

      final expiredAllocation = ResourceAllocation(
        allocationId: 'expired',
        assignedGPUs: [0],
        allocatedMemoryBytes: 1024 * 1024 * 1024,
        assignedPriority: PriorityLevel.normal,
        allocationTime: now.subtract(Duration(hours: 2)),
        estimatedDuration: Duration(hours: 1),
      );

      expect(validAllocation.isValid, isTrue);
      expect(expiredAllocation.isValid, isFalse);
    });

    test('validates allocation without duration', () {
      final allocation = ResourceAllocation(
        allocationId: 'no_duration',
        assignedGPUs: [0],
        allocatedMemoryBytes: 1024 * 1024 * 1024,
        assignedPriority: PriorityLevel.normal,
        allocationTime: DateTime.now(),
        // No estimatedDuration
      );

      expect(allocation.isValid, isTrue);
    });
  });

  group('ResourceMetrics', () {
    test('creates metrics with basic properties', () {
      final timestamp = DateTime.now();
      final metrics = ResourceMetrics(
        timestamp: timestamp,
        gpuUtilization: {0: 75.0, 1: 60.0},
        gpuMemoryUsed: {0: 2 * 1024 * 1024 * 1024, 1: 1 * 1024 * 1024 * 1024},
        totalMemoryUsed: 3 * 1024 * 1024 * 1024,
        totalMemoryAvailable: 8 * 1024 * 1024 * 1024,
        averageExecutionTime: Duration(milliseconds: 5),
        maxExecutionTime: Duration(milliseconds: 15),
        activeAllocations: 3,
      );

      expect(metrics.timestamp, equals(timestamp));
      expect(metrics.gpuUtilization[0], equals(75.0));
      expect(metrics.gpuUtilization[1], equals(60.0));
      expect(metrics.totalMemoryUsed, equals(3 * 1024 * 1024 * 1024));
      expect(metrics.activeAllocations, equals(3));
    });

    test('calculates memory utilization', () {
      final metrics = ResourceMetrics(
        timestamp: DateTime.now(),
        totalMemoryUsed: 6 * 1024 * 1024 * 1024, // 6GB used
        totalMemoryAvailable: 8 * 1024 * 1024 * 1024, // 8GB available
        averageExecutionTime: Duration(milliseconds: 1),
        maxExecutionTime: Duration(milliseconds: 1),
        activeAllocations: 1,
      );

      expect(metrics.memoryUtilization, equals(75.0)); // 6/8 * 100
    });

    test('calculates average GPU utilization', () {
      final metrics = ResourceMetrics(
        timestamp: DateTime.now(),
        gpuUtilization: {0: 80.0, 1: 60.0, 2: 90.0},
        totalMemoryUsed: 1024 * 1024 * 1024,
        totalMemoryAvailable: 8 * 1024 * 1024 * 1024,
        averageExecutionTime: Duration(milliseconds: 1),
        maxExecutionTime: Duration(milliseconds: 1),
        activeAllocations: 1,
      );

      expect(
        metrics.averageGpuUtilization,
        closeTo(76.67, 0.01),
      ); // (80+60+90)/3
    });

    test('detects resource pressure', () {
      final highMemoryMetrics = ResourceMetrics(
        timestamp: DateTime.now(),
        totalMemoryUsed: 85 * 1024 * 1024 * 1024, // 85GB
        totalMemoryAvailable:
            100 * 1024 * 1024 * 1024, // 100GB (85% utilization)
        averageExecutionTime: Duration(milliseconds: 1),
        maxExecutionTime: Duration(milliseconds: 1),
        activeAllocations: 1,
      );

      final highGpuMetrics = ResourceMetrics(
        timestamp: DateTime.now(),
        gpuUtilization: {0: 95.0}, // 95% GPU utilization
        totalMemoryUsed: 1024 * 1024 * 1024,
        totalMemoryAvailable: 8 * 1024 * 1024 * 1024,
        averageExecutionTime: Duration(milliseconds: 1),
        maxExecutionTime: Duration(milliseconds: 1),
        activeAllocations: 1,
      );

      final normalMetrics = ResourceMetrics(
        timestamp: DateTime.now(),
        gpuUtilization: {0: 50.0},
        totalMemoryUsed: 2 * 1024 * 1024 * 1024,
        totalMemoryAvailable: 8 * 1024 * 1024 * 1024, // 25% memory utilization
        averageExecutionTime: Duration(milliseconds: 1),
        maxExecutionTime: Duration(milliseconds: 1),
        activeAllocations: 1,
      );

      expect(highMemoryMetrics.isUnderPressure, isTrue);
      expect(highGpuMetrics.isUnderPressure, isTrue);
      expect(normalMetrics.isUnderPressure, isFalse);
    });

    test('handles empty GPU utilization', () {
      final metrics = ResourceMetrics(
        timestamp: DateTime.now(),
        gpuUtilization: {}, // Empty
        totalMemoryUsed: 1024 * 1024 * 1024,
        totalMemoryAvailable: 8 * 1024 * 1024 * 1024,
        averageExecutionTime: Duration(milliseconds: 1),
        maxExecutionTime: Duration(milliseconds: 1),
        activeAllocations: 1,
      );

      expect(metrics.averageGpuUtilization, equals(0.0));
    });

    test('handles zero available memory', () {
      final metrics = ResourceMetrics(
        timestamp: DateTime.now(),
        totalMemoryUsed: 1024 * 1024 * 1024,
        totalMemoryAvailable: 0, // Zero available
        averageExecutionTime: Duration(milliseconds: 1),
        maxExecutionTime: Duration(milliseconds: 1),
        activeAllocations: 1,
      );

      expect(metrics.memoryUtilization, equals(0.0));
    });
  });

  group('PerformanceProfile', () {
    test('creates profile with basic properties', () {
      final profileTime = DateTime.now();
      final profile = PerformanceProfile(
        profileDuration: Duration(seconds: 10),
        stageExecutionTimes: {
          'stage1': Duration(milliseconds: 5),
          'stage2': Duration(milliseconds: 3),
          'stage3': Duration(milliseconds: 8),
        },
        stageMemoryUsage: {
          'stage1': 1024 * 1024 * 512, // 512MB
          'stage2': 1024 * 1024 * 256, // 256MB
          'stage3': 1024 * 1024 * 1024, // 1GB
        },
        resourceHistory: [],
        bottleneckAnalysis: {},
        recommendations: [],
        profileTime: profileTime,
      );

      expect(profile.profileDuration, equals(Duration(seconds: 10)));
      expect(profile.stageExecutionTimes.length, equals(3));
      expect(profile.stageMemoryUsage.length, equals(3));
      expect(profile.profileTime, equals(profileTime));
    });

    test('identifies bottleneck stage', () {
      final profile = PerformanceProfile(
        profileDuration: Duration(seconds: 1),
        stageExecutionTimes: {
          'fast_stage': Duration(milliseconds: 2),
          'slow_stage': Duration(milliseconds: 15), // Bottleneck
          'medium_stage': Duration(milliseconds: 8),
        },
        stageMemoryUsage: {},
        resourceHistory: [],
        bottleneckAnalysis: {},
        recommendations: [],
        profileTime: DateTime.now(),
      );

      expect(profile.bottleneckStage, equals('slow_stage'));
    });

    test('handles empty stage execution times', () {
      final profile = PerformanceProfile(
        profileDuration: Duration(seconds: 1),
        stageExecutionTimes: {}, // Empty
        stageMemoryUsage: {},
        resourceHistory: [],
        bottleneckAnalysis: {},
        recommendations: [],
        profileTime: DateTime.now(),
      );

      expect(profile.bottleneckStage, isNull);
      expect(profile.totalExecutionTime, equals(Duration.zero));
    });

    test('calculates total execution time', () {
      final profile = PerformanceProfile(
        profileDuration: Duration(seconds: 1),
        stageExecutionTimes: {
          'stage1': Duration(milliseconds: 5),
          'stage2': Duration(milliseconds: 10),
          'stage3': Duration(milliseconds: 15),
        },
        stageMemoryUsage: {},
        resourceHistory: [],
        bottleneckAnalysis: {},
        recommendations: [],
        profileTime: DateTime.now(),
      );

      expect(profile.totalExecutionTime, equals(Duration(milliseconds: 30)));
    });

    test('calculates peak memory usage', () {
      final metrics1 = ResourceMetrics(
        timestamp: DateTime.now(),
        totalMemoryUsed: 2 * 1024 * 1024 * 1024, // 2GB
        totalMemoryAvailable: 8 * 1024 * 1024 * 1024,
        averageExecutionTime: Duration(milliseconds: 1),
        maxExecutionTime: Duration(milliseconds: 1),
        activeAllocations: 1,
      );

      final metrics2 = ResourceMetrics(
        timestamp: DateTime.now(),
        totalMemoryUsed: 5 * 1024 * 1024 * 1024, // 5GB - Peak
        totalMemoryAvailable: 8 * 1024 * 1024 * 1024,
        averageExecutionTime: Duration(milliseconds: 1),
        maxExecutionTime: Duration(milliseconds: 1),
        activeAllocations: 1,
      );

      final metrics3 = ResourceMetrics(
        timestamp: DateTime.now(),
        totalMemoryUsed: 3 * 1024 * 1024 * 1024, // 3GB
        totalMemoryAvailable: 8 * 1024 * 1024 * 1024,
        averageExecutionTime: Duration(milliseconds: 1),
        maxExecutionTime: Duration(milliseconds: 1),
        activeAllocations: 1,
      );

      final profile = PerformanceProfile(
        profileDuration: Duration(seconds: 1),
        stageExecutionTimes: {},
        stageMemoryUsage: {},
        resourceHistory: [metrics1, metrics2, metrics3],
        bottleneckAnalysis: {},
        recommendations: [],
        profileTime: DateTime.now(),
      );

      expect(profile.peakMemoryUsage, equals(5 * 1024 * 1024 * 1024));
    });

    test('handles empty resource history', () {
      final profile = PerformanceProfile(
        profileDuration: Duration(seconds: 1),
        stageExecutionTimes: {},
        stageMemoryUsage: {},
        resourceHistory: [], // Empty
        bottleneckAnalysis: {},
        recommendations: [],
        profileTime: DateTime.now(),
      );

      expect(profile.peakMemoryUsage, equals(0));
    });
  });

  group('ResourceGovernor', () {
    late ResourceGovernor governor;

    setUp(() {
      governor = ResourceGovernor();
    });

    tearDown(() {
      governor.dispose();
    });

    test('initializes with detected GPUs', () async {
      await governor.initialize();

      expect(governor.availableGPUs, isNotEmpty);
      expect(governor.availableGPUs.length, greaterThanOrEqualTo(1));
    });

    test('allocates resources successfully', () async {
      await governor.initialize();

      final requirements = ResourceRequirements.basic(maxMemoryMB: 512);
      final allocation = await governor.allocateResources(requirements);

      expect(allocation.allocationId, isNotEmpty);
      expect(allocation.assignedGPUs, isNotEmpty);
      expect(allocation.allocatedMemoryBytes, greaterThan(0));
      expect(allocation.assignedPriority, equals(PriorityLevel.normal));
    });

    test('throws exception for incompatible requirements', () async {
      await governor.initialize();

      // Requirements that no GPU can satisfy
      final impossibleRequirements = ResourceRequirements(
        maxMemoryBytes: 1000 * 1024 * 1024 * 1024, // 1000GB - impossible
        minimumComputeCapability: ComputeCapability.raytracing,
        allowFallbackToCPU: false,
      );

      expect(
        () async => await governor.allocateResources(impossibleRequirements),
        throwsA(isA<ResourceAllocationException>()),
      );
    });

    test('deallocates resources', () async {
      await governor.initialize();

      final requirements = ResourceRequirements.basic();
      final allocation = await governor.allocateResources(requirements);

      // Allocation should exist
      expect(allocation.allocationId, isNotEmpty);

      await governor.deallocateResources(allocation);
      // After deallocation, the governor should handle this gracefully
    });

    test('tracks memory usage', () async {
      await governor.initialize();

      final initialUsage = governor.getCurrentMemoryUsage();
      expect(initialUsage, equals(0));

      final requirements = ResourceRequirements.basic(maxMemoryMB: 256);
      final allocation = await governor.allocateResources(requirements);

      final usageAfterAllocation = governor.getCurrentMemoryUsage();
      expect(usageAfterAllocation, greaterThan(initialUsage));

      await governor.deallocateResources(allocation);
    });

    test('sets memory budget', () async {
      await governor.initialize();

      final newBudget = 16 * 1024 * 1024 * 1024; // 16GB
      await governor.setMemoryBudget(newBudget);

      expect(governor.maxMemoryBytes, equals(newBudget));
    });

    test('sets GPU affinity', () async {
      await governor.initialize();

      // Should not throw
      await governor.setGPUAffinity([0, 1]);
    });

    test('enables memory compression', () async {
      await governor.initialize();

      // Should not throw
      governor.enableMemoryCompression(true);
      governor.enableMemoryCompression(false);
    });

    test('enables auto load balancing', () async {
      await governor.initialize();

      governor.enableAutoLoadBalancing(true);
      governor.enableAutoLoadBalancing(false);

      // Should not throw
      await governor.rebalanceWorkload();
    });

    test('profiles execution', () async {
      await governor.initialize();

      final profileDuration = Duration(milliseconds: 100);
      final profile = await governor.profileExecution(profileDuration);

      expect(profile.profileDuration, equals(profileDuration));
      expect(profile.resourceHistory, isNotEmpty);
      expect(profile.profileTime, isNotNull);
    });

    test('provides resource metrics stream', () async {
      await governor.initialize();

      final metricsCompleter = Completer<ResourceMetrics>();
      late StreamSubscription subscription;

      subscription = governor.resourceMetrics.listen((metrics) {
        if (!metricsCompleter.isCompleted) {
          metricsCompleter.complete(metrics);
          subscription.cancel();
        }
      });

      final metrics = await metricsCompleter.future.timeout(
        Duration(seconds: 3),
      );

      expect(metrics.timestamp, isNotNull);
      expect(metrics.activeAllocations, isA<int>());
    });

    test('handles ML inference allocation', () async {
      await governor.initialize();

      final requirements = ResourceRequirements.mlInference(minMemoryGB: 2);

      // Should either allocate successfully or throw meaningful exception
      try {
        final allocation = await governor.allocateResources(requirements);
        expect(allocation.assignedPriority, equals(PriorityLevel.high));
        await governor.deallocateResources(allocation);
      } on ResourceAllocationException catch (e) {
        expect(e.message, contains('compatible'));
      }
    });

    test('handles real-time allocation', () async {
      await governor.initialize();

      final requirements = ResourceRequirements.realTime(
        maxLatency: Duration(milliseconds: 5),
        maxMemoryMB: 128,
      );

      final allocation = await governor.allocateResources(requirements);
      expect(allocation.assignedPriority, equals(PriorityLevel.realTime));

      await governor.deallocateResources(allocation);
    });
  });

  group('ResourceAllocationException', () {
    test('creates exception with message', () {
      final exception = ResourceAllocationException('Test error message');
      expect(exception.message, equals('Test error message'));
      expect(exception.toString(), contains('Test error message'));
    });
  });

  group('Enums', () {
    test('ComputeCapability enum values', () {
      expect(ComputeCapability.values, hasLength(4));
      expect(ComputeCapability.values, contains(ComputeCapability.basic));
      expect(ComputeCapability.values, contains(ComputeCapability.advanced));
      expect(ComputeCapability.values, contains(ComputeCapability.ml));
      expect(ComputeCapability.values, contains(ComputeCapability.raytracing));
    });

    test('MemoryTier enum values', () {
      expect(MemoryTier.values, hasLength(5));
      expect(MemoryTier.values, contains(MemoryTier.gpuLocal));
      expect(MemoryTier.values, contains(MemoryTier.gpuGlobal));
      expect(MemoryTier.values, contains(MemoryTier.gpuShared));
      expect(MemoryTier.values, contains(MemoryTier.systemRam));
      expect(MemoryTier.values, contains(MemoryTier.storage));
    });

    test('PriorityLevel enum values', () {
      expect(PriorityLevel.values, hasLength(5));
      expect(PriorityLevel.values, contains(PriorityLevel.background));
      expect(PriorityLevel.values, contains(PriorityLevel.normal));
      expect(PriorityLevel.values, contains(PriorityLevel.high));
      expect(PriorityLevel.values, contains(PriorityLevel.realTime));
      expect(PriorityLevel.values, contains(PriorityLevel.critical));
    });

    test('MemoryPoolStrategy enum values', () {
      expect(MemoryPoolStrategy.values, hasLength(3));
      expect(MemoryPoolStrategy.values, contains(MemoryPoolStrategy.shared));
      expect(MemoryPoolStrategy.values, contains(MemoryPoolStrategy.isolated));
      expect(MemoryPoolStrategy.values, contains(MemoryPoolStrategy.adaptive));
    });

    test('DistributionStrategy enum values', () {
      expect(DistributionStrategy.values, hasLength(4));
      expect(
        DistributionStrategy.values,
        contains(DistributionStrategy.replicated),
      );
      expect(
        DistributionStrategy.values,
        contains(DistributionStrategy.partitioned),
      );
      expect(
        DistributionStrategy.values,
        contains(DistributionStrategy.pipelined),
      );
      expect(
        DistributionStrategy.values,
        contains(DistributionStrategy.adaptive),
      );
    });
  });
}
