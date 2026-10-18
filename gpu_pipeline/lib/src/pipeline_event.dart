import 'dart:async';

/// Types of pipeline events
enum PipelineEventType {
  stageStart,
  stageComplete,
  stageError,
  streamStart,
  streamStop,
  streamError,
  tensorUpdate,
  pipelineStart,
  pipelineStop,
  pipelineError,
  pipelineRunComplete,
  benchmark,
  operationStart,
  operationComplete,
  operationError,
  parameterChange,
}

/// Moving average calculator
class MovingAverage {
  final int windowSize;
  final List<double> _values = [];
  double _sum = 0.0;

  MovingAverage({required this.windowSize});

  void add(double value) {
    _values.add(value);
    _sum += value;

    if (_values.length > windowSize) {
      _sum -= _values.removeAt(0);
    }
  }

  double get average => _values.isEmpty ? 0.0 : _sum / _values.length;

  double get current => _values.isEmpty ? 0.0 : _values.last;

  int get count => _values.length;

  bool get isFull => _values.length == windowSize;

  double get min =>
      _values.isEmpty ? 0.0 : _values.reduce((a, b) => a < b ? a : b);

  double get max =>
      _values.isEmpty ? 0.0 : _values.reduce((a, b) => a > b ? a : b);

  List<double> get values => List.unmodifiable(_values);

  void clear() {
    _values.clear();
    _sum = 0.0;
  }

  Map<String, dynamic> toMap() {
    return {
      'average': average,
      'current': current,
      'count': count,
      'min': min,
      'max': max,
      'window_size': windowSize,
      'is_full': isFull,
    };
  }
}

/// Base pipeline event
abstract class PipelineEvent {
  final PipelineEventType type;
  final String pipelineId;
  final DateTime timestamp;
  final Map<String, dynamic> metadata;

  PipelineEvent({
    required this.type,
    required this.pipelineId,
    Map<String, dynamic>? metadata,
  }) : timestamp = DateTime.now(),
       metadata = metadata ?? const {};
}

/// Stage-related events
class StageStartEvent extends PipelineEvent {
  final String stageId;
  final String inputFormat;
  final List<int> tensorShape;

  StageStartEvent({
    required this.stageId,
    required this.inputFormat,
    required this.tensorShape,
    required super.pipelineId,
    super.metadata,
  }) : super(type: PipelineEventType.stageStart);
}

class StageCompleteEvent extends PipelineEvent {
  final String stageId;
  final String outputFormat;
  final List<int> tensorShape;
  final Duration processingTime;

  StageCompleteEvent({
    required this.stageId,
    required this.outputFormat,
    required this.tensorShape,
    required this.processingTime,
    required super.pipelineId,
    super.metadata,
  }) : super(type: PipelineEventType.stageComplete);
}

class StageErrorEvent extends PipelineEvent {
  final String stageId;
  final String error;
  final StackTrace? stackTrace;

  StageErrorEvent({
    required this.stageId,
    required this.error,
    this.stackTrace,
    required super.pipelineId,
    super.metadata,
  }) : super(type: PipelineEventType.stageError);
}

/// Stream-related events
class StreamStartEvent extends PipelineEvent {
  final String streamId;
  final String streamType;

  StreamStartEvent({
    required this.streamId,
    required this.streamType,
    required super.pipelineId,
    super.metadata,
  }) : super(type: PipelineEventType.streamStart);
}

class StreamStopEvent extends PipelineEvent {
  final String streamId;
  final String streamType;

  StreamStopEvent({
    required this.streamId,
    required this.streamType,
    required super.pipelineId,
    super.metadata,
  }) : super(type: PipelineEventType.streamStop);
}

class StreamErrorEvent extends PipelineEvent {
  final String streamId;
  final String error;
  final StackTrace? stackTrace;

  StreamErrorEvent({
    required this.streamId,
    required this.error,
    this.stackTrace,
    required super.pipelineId,
    super.metadata,
  }) : super(type: PipelineEventType.streamError);
}

/// Tensor update event
class TensorUpdateEvent extends PipelineEvent {
  final String sourceId;
  final String format;
  final List<int> tensorShape;
  final int dataSize;

  TensorUpdateEvent({
    required this.sourceId,
    required this.format,
    required this.tensorShape,
    required this.dataSize,
    required super.pipelineId,
    super.metadata,
  }) : super(type: PipelineEventType.tensorUpdate);
}

/// Pipeline lifecycle events
class PipelineStartEvent extends PipelineEvent {
  final int stageCount;
  final int streamCount;

  PipelineStartEvent({
    required this.stageCount,
    required this.streamCount,
    required super.pipelineId,
    super.metadata,
  }) : super(type: PipelineEventType.pipelineStart);
}

class PipelineStopEvent extends PipelineEvent {
  final Duration totalRuntime;

  PipelineStopEvent({
    required this.totalRuntime,
    required super.pipelineId,
    super.metadata,
  }) : super(type: PipelineEventType.pipelineStop);
}

class PipelineErrorEvent extends PipelineEvent {
  final String error;
  final StackTrace? stackTrace;

  PipelineErrorEvent({
    required this.error,
    this.stackTrace,
    required super.pipelineId,
    super.metadata,
  }) : super(type: PipelineEventType.pipelineError);
}

/// Pipeline run complete event - fired when a full processing cycle completes
class PipelineRunCompleteEvent extends PipelineEvent {
  final Duration runDuration;
  final int stagesProcessed;
  final String finalFormat;
  final List<int> finalTensorShape;
  final Map<String, Duration> stageTimings;

  PipelineRunCompleteEvent({
    required this.runDuration,
    required this.stagesProcessed,
    required this.finalFormat,
    required this.finalTensorShape,
    required this.stageTimings,
    required super.pipelineId,
    super.metadata,
  }) : super(type: PipelineEventType.pipelineRunComplete);
}

/// Benchmark event for performance tracking
class PipelineBenchmarkEvent extends PipelineEvent {
  final String name;
  final String category;
  final Duration duration;
  final Map<String, num> metrics;

  PipelineBenchmarkEvent({
    required this.name,
    required this.category,
    required this.duration,
    this.metrics = const {},
    required super.pipelineId,
    super.metadata,
  }) : super(type: PipelineEventType.benchmark);
}

class OperationStartEvent extends PipelineEvent {
  final String stageId;
  final int operationIndex;
  final String operationType;

  OperationStartEvent({
    required super.pipelineId,
    required this.stageId,
    required this.operationIndex,
    required this.operationType,
    super.metadata,
  }) : super(type: PipelineEventType.operationStart);
}

class OperationCompleteEvent extends PipelineEvent {
  final String stageId;
  final int operationIndex;
  final String operationType;
  final Duration processingTime;

  OperationCompleteEvent({
    required super.pipelineId,
    required this.stageId,
    required this.operationIndex,
    required this.operationType,
    required this.processingTime,
    super.metadata,
  }) : super(type: PipelineEventType.operationComplete);
}

class OperationErrorEvent extends PipelineEvent {
  final String stageId;
  final int operationIndex;
  final String operationType;
  final String error;
  final StackTrace? stackTrace;

  OperationErrorEvent({
    required super.pipelineId,
    required this.stageId,
    required this.operationIndex,
    required this.operationType,
    required this.error,
    this.stackTrace,
    super.metadata,
  }) : super(type: PipelineEventType.operationError);
}

class ParameterChangeEvent extends PipelineEvent {
  final String stageId;
  final String parameterName;
  final dynamic oldValue;
  final dynamic newValue;
  final String parameterType;

  ParameterChangeEvent({
    required super.pipelineId,
    required this.stageId,
    required this.parameterName,
    required this.oldValue,
    required this.newValue,
    required this.parameterType,
    super.metadata,
  }) : super(type: PipelineEventType.parameterChange);
}

/// Manages and notifies pipeline events with thread-safe emission
class PipelineEventNotifier {
  final Map<PipelineEventType, List<void Function(PipelineEvent)>> _listeners =
      {};
  final List<PipelineEvent> _events = [];
  final StreamController<PipelineEvent> _eventStreamController =
      StreamController<PipelineEvent>.broadcast();
  final StreamController<void> _generalNotificationController =
      StreamController<void>.broadcast();

  // Configuration
  final int _maxEvents;
  final bool _enableBenchmarking;
  final bool _debugEnabled;

  // Benchmarking
  final Map<String, MovingAverage> _stagePerformance = {};
  final Map<String, MovingAverage> _operationPerformance = {};
  final Map<String, int> _errorCounts = {};

  // State management - no more microtasks!
  bool _isEmitting = false;
  final List<_PendingListenerOperation> _pendingOperations = [];

  PipelineEventNotifier({
    int maxEvents = 100,
    bool enableBenchmarking = false,
    bool debugEnabled = false,
  }) : _maxEvents = maxEvents,
       _enableBenchmarking = enableBenchmarking,
       _debugEnabled = debugEnabled;

  /// Add event listener for specific type
  void addEventListener(
    PipelineEventType type,
    void Function(PipelineEvent) listener,
  ) {
    if (_isEmitting) {
      // Queue the operation instead of using microtasks
      _pendingOperations.add(
        _PendingListenerOperation(
          type: _PendingOperationType.add,
          eventType: type,
          listener: listener,
        ),
      );
      if (_debugEnabled) {
        print('🔄 Queued listener addition for ${type.name}');
      }
      return;
    }

    _listeners.putIfAbsent(type, () => []).add(listener);
    if (_debugEnabled) {
      print(
        '➕ Added listener for ${type.name} (total: ${_listeners[type]!.length})',
      );
    }
  }

  /// Remove event listener
  void removeEventListener(
    PipelineEventType type,
    void Function(PipelineEvent) listener,
  ) {
    if (_isEmitting) {
      _pendingOperations.add(
        _PendingListenerOperation(
          type: _PendingOperationType.remove,
          eventType: type,
          listener: listener,
        ),
      );
      if (_debugEnabled) {
        print('🔄 Queued listener removal for ${type.name}');
      }
      return;
    }

    final removed = _listeners[type]?.remove(listener) ?? false;
    if (_debugEnabled) {
      print(
        '➖ ${removed ? 'Removed' : 'Failed to remove'} listener for ${type.name}',
      );
    }
  }

  /// Clear all listeners for a specific event type
  void clearEventListeners(PipelineEventType type) {
    if (_isEmitting) {
      _pendingOperations.add(
        _PendingListenerOperation(
          type: _PendingOperationType.clear,
          eventType: type,
        ),
      );
      return;
    }

    final count = _listeners[type]?.length ?? 0;
    _listeners[type]?.clear();
    if (_debugEnabled) {
      print('🗑️ Cleared $count listeners for ${type.name}');
    }
  }

  /// Clear all listeners
  void clearAllEventListeners() {
    if (_isEmitting) {
      _pendingOperations.add(
        _PendingListenerOperation(
          type: _PendingOperationType.clearAll,
          eventType: PipelineEventType.pipelineStart, // Dummy value
        ),
      );
      return;
    }

    final totalCount = _listeners.values.fold(
      0,
      (sum, list) => sum + list.length,
    );
    _listeners.clear();
    if (_debugEnabled) {
      print('🗑️ Cleared all $totalCount listeners');
    }
  }

  /// Emit an event to all registered listeners
  void emit(PipelineEvent event) {
    // Prevent recursive emissions - just drop the event
    if (_isEmitting) {
      if (_debugEnabled) {
        print('⚠️ Dropping recursive event: ${event.type.name}');
      }
      return;
    }

    _isEmitting = true;

    try {
      // Add to events list
      _events.add(event);
      if (_events.length > _maxEvents) {
        _events.removeAt(0);
      }

      if (_debugEnabled) {
        print('📡 Emitting ${event.type.name} event');
      }

      // Handle benchmarking
      if (_enableBenchmarking) {
        _handleBenchmarkEvent(event);
      }

      // Notify listeners with safe iteration
      final listenersForType = _listeners[event.type];
      if (listenersForType != null && listenersForType.isNotEmpty) {
        final listenerCount = listenersForType.length;

        // Iterate through current listeners safely
        for (int i = 0; i < listenerCount; i++) {
          // Double-check bounds in case list was modified somehow
          if (i < listenersForType.length) {
            try {
              listenersForType[i](event);
            } catch (e, stackTrace) {
              if (_debugEnabled) {
                print('⚠️ Error in event listener for ${event.type.name}: $e');
                print('Stack trace: $stackTrace');
              }
              // Don't remove the listener during emission - just log the error
            }
          }
        }

        if (_debugEnabled) {
          print('📡 Notified $listenerCount listeners for ${event.type.name}');
        }
      }

      // Emit to streams
      if (!_eventStreamController.isClosed) {
        _eventStreamController.add(event);
      }
      if (!_generalNotificationController.isClosed) {
        _generalNotificationController.add(null);
      }
    } catch (e, stackTrace) {
      if (_debugEnabled) {
        print('❌ Error during event emission: $e');
        print('Stack trace: $stackTrace');
      }
    } finally {
      _isEmitting = false;

      // Process any pending listener operations
      _processPendingOperations();
    }
  }

  /// Process queued listener operations
  void _processPendingOperations() {
    if (_pendingOperations.isEmpty) return;

    final operations = List<_PendingListenerOperation>.from(_pendingOperations);
    _pendingOperations.clear();

    if (_debugEnabled) {
      print('🔄 Processing ${operations.length} pending listener operations');
    }

    for (final op in operations) {
      try {
        switch (op.type) {
          case _PendingOperationType.add:
            if (op.listener != null) {
              _listeners.putIfAbsent(op.eventType, () => []).add(op.listener!);
            }
            break;
          case _PendingOperationType.remove:
            if (op.listener != null) {
              _listeners[op.eventType]?.remove(op.listener!);
            }
            break;
          case _PendingOperationType.clear:
            _listeners[op.eventType]?.clear();
            break;
          case _PendingOperationType.clearAll:
            _listeners.clear();
            break;
        }
      } catch (e) {
        if (_debugEnabled) {
          print('⚠️ Error processing pending operation: $e');
        }
      }
    }
  }

  /// Handle benchmarking-related events
  void _handleBenchmarkEvent(PipelineEvent event) {
    switch (event.type) {
      case PipelineEventType.stageComplete:
        if (event is StageCompleteEvent) {
          final avgTime = _stagePerformance.putIfAbsent(
            event.stageId,
            () => MovingAverage(windowSize: 10),
          );
          avgTime.add(event.processingTime.inMicroseconds.toDouble());
        }
        break;

      case PipelineEventType.operationComplete:
        if (event is OperationCompleteEvent) {
          final avgTime = _operationPerformance.putIfAbsent(
            event.operationIndex.toString(),
            () => MovingAverage(windowSize: 10),
          );
          avgTime.add(event.processingTime.inMicroseconds.toDouble());
        }
        break;

      case PipelineEventType.operationError:
        if (event is OperationErrorEvent) {
          final errorKey = '${event.operationIndex}_${event.error}';
          _errorCounts[errorKey] = (_errorCounts[errorKey] ?? 0) + 1;
        }
        break;

      default:
        // No benchmarking for other event types
        break;
    }
  }

  Map<String, dynamic> getBenchmarkStats() {
    return {
      'stagePerformance': stagePerformanceStats,
      'operationPerformance': operationPerformanceStats,
      'errorStats': errorStats,
      'totalEvents': _events.length,
      'totalListeners': totalListenerCount,
    };
  }

  /// Get recent events (read-only)
  List<PipelineEvent> get recentEvents => List.unmodifiable(_events);

  /// Get event stream for reactive programming
  Stream<PipelineEvent> get eventStream => _eventStreamController.stream;

  /// General notification stream (fires on any event)
  Stream<void> get notificationStream => _generalNotificationController.stream;

  /// Get performance statistics
  Map<String, double> get stagePerformanceStats {
    return _stagePerformance.map((key, avg) => MapEntry(key, avg.average));
  }

  /// Get operation performance statistics
  Map<String, double> get operationPerformanceStats {
    return _operationPerformance.map((key, avg) => MapEntry(key, avg.average));
  }

  /// Get error statistics
  Map<String, int> get errorStats => Map.unmodifiable(_errorCounts);

  /// Reset all statistics
  void resetStats() {
    _stagePerformance.clear();
    _operationPerformance.clear();
    _errorCounts.clear();
    if (_debugEnabled) {
      print('📊 Reset all performance statistics');
    }
  }

  /// Check if there are listeners for a specific event type
  bool hasListeners(PipelineEventType type) {
    return (_listeners[type]?.isNotEmpty ?? false);
  }

  /// Get listener count for debugging
  int getListenerCount(PipelineEventType type) {
    return _listeners[type]?.length ?? 0;
  }

  /// Get total listener count
  int get totalListenerCount {
    return _listeners.values.fold(0, (sum, list) => sum + list.length);
  }

  /// Dispose of resources
  void dispose() {
    if (_debugEnabled) {
      print('🗑️ Disposing PipelineEventNotifier');
    }

    clearAllEventListeners();
    _events.clear();
    _stagePerformance.clear();
    _operationPerformance.clear();
    _errorCounts.clear();
    _pendingOperations.clear();

    if (!_eventStreamController.isClosed) {
      _eventStreamController.close();
    }
    if (!_generalNotificationController.isClosed) {
      _generalNotificationController.close();
    }
  }
}

/// Internal helper for pending listener operations
enum _PendingOperationType { add, remove, clear, clearAll }

class _PendingListenerOperation {
  final _PendingOperationType type;
  final PipelineEventType eventType;
  final void Function(PipelineEvent)? listener;

  _PendingListenerOperation({
    required this.type,
    required this.eventType,
    this.listener,
  });
}
