import 'dart:async';

class MainThreadMonitor {
  // Singleton implementation
  static MainThreadMonitor? _instance;
  static final Object _lock = Object();

  // Private constructor
  MainThreadMonitor._internal();

  // Singleton getter with thread-safe lazy initialization
  static MainThreadMonitor get instance {
    if (_instance == null) {
      synchronized(_lock, () {
        _instance ??= MainThreadMonitor._internal();
      });
    }
    return _instance!;
  }

  // Factory constructor
  factory MainThreadMonitor() => instance;

  // Instance variables
  bool _isMonitoring = false;
  double _currentLoad = 0.0;
  Timer? _monitorTimer;
  Timer? _microtaskTimer;
  Timer? _memoryTimer;
  Timer? _cpuTimer;

  // Event loop timing measurements
  final List<Duration> _eventLoopTimes = [];
  final List<Duration> _microtaskDelays = [];
  final List<Duration> _cpuSamples = [];
  DateTime? _lastMeasurement;
  bool _microtaskQueuePressured = false;
  bool _memoryPressure = false;
  bool _highCpuUsage = false;

  static const int _measurementWindow = 20;
  static const Duration _targetEventTime = Duration(milliseconds: 1);

  // Public getters
  bool get isMonitoring => _isMonitoring;
  bool get isOverloaded => _currentLoad > 0.8;
  bool get isHeavyLoad => _currentLoad > 0.6;
  double get currentLoad => _currentLoad;
  bool get hasMicrotaskPressure => _microtaskQueuePressured;
  bool get hasMemoryPressure => _memoryPressure;
  bool get hasCpuPressure => _highCpuUsage;

  /// Start monitoring (idempotent - safe to call multiple times)
  void startMonitoring() {
    if (_isMonitoring) {
      print('📊 Thread monitor already running');
      return;
    }

    _isMonitoring = true;
    _startEventLoopMonitoring();
    //_startMicrotaskMonitoring();
    //_startMemoryMonitoring();
    //_startCpuMonitoring();

    print('📊 MainThreadMonitor singleton started');
  }

  /// Stop monitoring (idempotent - safe to call multiple times)
  void stopMonitoring() {
    if (!_isMonitoring) {
      print('📊 Thread monitor not running');
      return;
    }

    _isMonitoring = false;

    _monitorTimer?.cancel();
    _microtaskTimer?.cancel();
    _memoryTimer?.cancel();
    _cpuTimer?.cancel();

    _monitorTimer = null;
    _microtaskTimer = null;
    _memoryTimer = null;
    _cpuTimer = null;

    _eventLoopTimes.clear();
    _microtaskDelays.clear();
    _cpuSamples.clear();

    _currentLoad = 0.0;
    _microtaskQueuePressured = false;
    _memoryPressure = false;
    _highCpuUsage = false;
    _lastMeasurement = null;

    print('📊 MainThreadMonitor singleton stopped');
  }

  void _startEventLoopMonitoring() {
    _monitorTimer = Timer.periodic(Duration(milliseconds: 5), (timer) {
      final start = DateTime.now();

      Timer(Duration.zero, () {
        final end = DateTime.now();
        final actualDelay = end.difference(start);

        _eventLoopTimes.add(actualDelay);

        if (_eventLoopTimes.length > _measurementWindow) {
          _eventLoopTimes.removeAt(0);
        }

        _calculateLoad();
        _lastMeasurement = end;
      });
    });
  }

  void _startMicrotaskMonitoring() {
    _microtaskTimer = Timer.periodic(Duration(milliseconds: 10), (timer) {
      final start = DateTime.now();

      scheduleMicrotask(() {
        final end = DateTime.now();
        final delay = end.difference(start);

        _microtaskDelays.add(delay);
        if (_microtaskDelays.length > 10) {
          _microtaskDelays.removeAt(0);
        }

        final avgDelay =
            _microtaskDelays
                .fold<Duration>(Duration.zero, (p, e) => p + e)
                .inMicroseconds /
            _microtaskDelays.length;

        _microtaskQueuePressured = avgDelay > 2000; // 2ms
      });
    });
  }

  void _startMemoryMonitoring() {
    _memoryTimer = Timer.periodic(Duration(seconds: 2), (timer) {
      final start = DateTime.now();

      // Small allocation test
      final temp = <String>[];
      for (int i = 0; i < 100; i++) {
        temp.add('test_$i');
      }
      temp.clear();

      final end = DateTime.now();
      final gcTime = end.difference(start);

      _memoryPressure = gcTime.inMilliseconds > 5;

      if (_memoryPressure) {
        print(
          '⚠️ Memory pressure detected (GC time: ${gcTime.inMilliseconds}ms)',
        );
      }
    });
  }

  void _startCpuMonitoring() {
    _cpuTimer = Timer.periodic(Duration(milliseconds: 100), (timer) {
      final start = DateTime.now();

      // CPU work sample
      var sum = 0;
      for (int i = 0; i < 10000; i++) {
        sum += i;
      }

      final end = DateTime.now();
      final cpuTime = end.difference(start);

      _cpuSamples.add(cpuTime);
      if (_cpuSamples.length > 10) {
        _cpuSamples.removeAt(0);
      }

      if (_cpuSamples.isNotEmpty) {
        final avgCpuTime =
            _cpuSamples
                .fold<Duration>(Duration.zero, (p, e) => p + e)
                .inMicroseconds /
            _cpuSamples.length;

        _highCpuUsage = avgCpuTime > 1000; // 1ms
      }
    });
  }

  void _calculateLoad() {
    if (_eventLoopTimes.isEmpty) return;

    final totalTime = _eventLoopTimes.fold<Duration>(
      Duration.zero,
      (previous, element) => previous + element,
    );

    final avgDelay = Duration(
      microseconds: totalTime.inMicroseconds ~/ _eventLoopTimes.length,
    );

    _currentLoad = (avgDelay.inMicroseconds / _targetEventTime.inMicroseconds)
        .clamp(0.0, 2.0);
  }

  bool shouldSkipFrame() {
    if (!_isMonitoring) return false;

    final now = DateTime.now();
    final staleData =
        _lastMeasurement == null ||
        now.difference(_lastMeasurement!) > Duration(milliseconds: 100);

    return isOverloaded ||
        staleData ||
        _microtaskQueuePressured ||
        _memoryPressure ||
        _highCpuUsage;
  }

  int getFrameSkipCount() {
    if (_currentLoad > 0.9) return 4;
    if (_currentLoad > 0.8) return 3;
    if (_currentLoad > 0.7) return 2;
    return 1;
  }

  void printStats() {
    print('📊 === MainThreadMonitor Stats ===');
    print('📊 Monitoring: $_isMonitoring');
    print('📊 Event Loop Load: ${(_currentLoad * 100).toStringAsFixed(1)}%');
    print('📊 Microtask Queue Pressured: $_microtaskQueuePressured');
    print('📊 Memory Pressure: $_memoryPressure');
    print('📊 High CPU Usage: $_highCpuUsage');
    print('📊 Should Skip Frame: ${shouldSkipFrame()}');

    if (_eventLoopTimes.isNotEmpty) {
      final avgMs =
          _eventLoopTimes
              .fold<Duration>(Duration.zero, (p, e) => p + e)
              .inMicroseconds /
          _eventLoopTimes.length /
          1000;
      print('📊 Avg Event Delay: ${avgMs.toStringAsFixed(2)}ms');
    }
    print('📊 === End Stats ===');
  }

  /// Reset all monitoring state (useful for testing)
  void reset() {
    stopMonitoring();
    // All state is cleared in stopMonitoring()
  }

  /// Dispose singleton
  static void disposeSingleton() {
    synchronized(_lock, () {
      _instance?.stopMonitoring();
      _instance = null;
    });
    print('📊 MainThreadMonitor singleton disposed');
  }
}

// Helper function for thread-safe operations
void synchronized(Object lock, void Function() operation) {
  operation();
}
