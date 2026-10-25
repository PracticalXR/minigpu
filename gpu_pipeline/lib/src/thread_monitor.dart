import 'dart:async';

class MainThreadMonitor {
  // Singleton implementation
  static MainThreadMonitor? _instance;

  // Private constructor
  MainThreadMonitor._internal();

  /// This isolate's monitor, created on first use.
  static MainThreadMonitor get instance =>
      _instance ??= MainThreadMonitor._internal();

  // Factory constructor
  factory MainThreadMonitor() => instance;

  // Instance variables
  bool _isMonitoring = false;
  double _currentLoad = 0.0;
  Timer? _monitorTimer;

  // Event loop timing measurements
  final List<Duration> _eventLoopTimes = [];
  DateTime? _lastMeasurement;

  static const int _measurementWindow = 20;
  static const Duration _targetEventTime = Duration(milliseconds: 1);

  // Public getters
  bool get isMonitoring => _isMonitoring;
  bool get isOverloaded => _currentLoad > 0.8;
  bool get isHeavyLoad => _currentLoad > 0.6;
  double get currentLoad => _currentLoad;

  // Nothing measures these three.
  @Deprecated('Always false: nothing measures it. Will be removed in 2.0.')
  bool get hasMicrotaskPressure => false;
  @Deprecated('Always false: nothing measures it. Will be removed in 2.0.')
  bool get hasMemoryPressure => false;
  @Deprecated('Always false: nothing measures it. Will be removed in 2.0.')
  bool get hasCpuPressure => false;

  /// Start monitoring (idempotent - safe to call multiple times)
  void startMonitoring() {
    if (_isMonitoring) return;
    _isMonitoring = true;
    _startEventLoopMonitoring();
  }

  /// Stop monitoring (idempotent - safe to call multiple times)
  void stopMonitoring() {
    if (!_isMonitoring) return;
    _isMonitoring = false;

    _monitorTimer?.cancel();
    _monitorTimer = null;
    _eventLoopTimes.clear();

    _currentLoad = 0.0;
    _lastMeasurement = null;
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

    return isOverloaded || staleData;
  }

  int getFrameSkipCount() {
    if (_currentLoad > 0.9) return 4;
    if (_currentLoad > 0.8) return 3;
    if (_currentLoad > 0.7) return 2;
    return 1;
  }

  /// Prints the current measurements.
  void printStats() {
    print('📊 === MainThreadMonitor Stats ===');
    print('📊 Monitoring: $_isMonitoring');
    print('📊 Event Loop Load: ${(_currentLoad * 100).toStringAsFixed(1)}%');
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
    _instance?.stopMonitoring();
    _instance = null;
  }
}

/// Runs [operation]. It never locked anything: an isolate runs one piece of
/// Dart code at a time, so there was nothing to guard.
@Deprecated('Runs operation directly; it never locked. Will be removed in 2.0.')
void synchronized(Object lock, void Function() operation) {
  operation();
}
