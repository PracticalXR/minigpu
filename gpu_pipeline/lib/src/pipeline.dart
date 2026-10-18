import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_pipeline/src/pipeline_ports.dart' show ValidationResult;
import 'package:gpu_pipeline/src/serialization/connection_graph.dart'
    show defaultConnectionInput, defaultConnectionOutput;
import 'package:gpu_pipeline/src/stages/pipeline_resource_cache.dart';
import 'package:gpu_pipeline/src/pipeline_resources.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart' show ComputeShader;

/// Stream routing strategies
enum StreamRouting {
  perStream, // Process each stream individually
  merged, // Merge all streams before processing
  selective, // Process only selected streams
}

/// Configuration for individual stage stream processing
class StageStreamConfig {
  final StreamRouting routing;
  final Set<String>? includedStreams;
  final Set<String>? excludedStreams;

  const StageStreamConfig({
    this.routing = StreamRouting.perStream,
    this.includedStreams,
    this.excludedStreams,
  });

  bool shouldProcessStream(String streamId) {
    if (routing == StreamRouting.selective) {
      if (includedStreams != null) {
        return includedStreams!.contains(streamId);
      }
      if (excludedStreams != null) {
        return !excludedStreams!.contains(streamId);
      }
    }

    if (excludedStreams != null && excludedStreams!.contains(streamId)) {
      return false;
    }
    return true;
  }
}

/// Processing work item
class _ProcessingItem {
  final String streamId;
  final String format;
  final DateTime timestamp;

  // GPU path
  final Map<String, Tensor>? tensors;

  // Host path
  final Map<String, TypedData>? host;
  final Map<String, List<int>>? hostShapes;

  bool get isHost => host != null;

  _ProcessingItem({
    required this.streamId,
    required this.format,
    required this.timestamp,
    this.tensors,
    this.host,
    this.hostShapes,
  });
}

/// Queue processing strategies
enum QueueStrategy {
  /// Keep only latest item per stream (current behavior)
  latestOnly,

  /// Keep N latest items per stream
  latestN,

  /// Process all items but drop if processing is slower than input rate
  dropOnOverrun,

  /// Buffer items and process in batches
  batched,

  /// Process all items, may cause lag if processing is slow
  processAll,
}

/// Queue configuration
class QueueConfig {
  final QueueStrategy strategy;
  final int maxItemsPerStream;
  final int maxTotalItems;
  final int batchSize;
  final Duration maxAge;

  const QueueConfig({
    this.strategy = QueueStrategy.latestOnly,
    this.maxItemsPerStream = 1,
    this.maxTotalItems = 10,
    this.batchSize = 1,
    this.maxAge = const Duration(milliseconds: 100),
  });
}

/// Simplified Pipeline with queued processing and safe ingestion
class Pipeline {
  final String id;
  final List<PipelineStage> _stages = [];
  final List<MediaStream> _streams = [];
  final Map<String, Map<String, Tensor>> _stageOutputs = {};
  final Map<String, StageStreamConfig> _stageStreamConfigs = {};

  // Simple processing queue
  final List<_ProcessingItem> _processingQueue = [];
  final Set<String> _activeProcessingItems = <String>{};

  // Per-stream ingestion serializer (lock) chains
  final Map<String, Future<void>> _ingestSerializers = {};

  bool _isProcessing = false;

  // Enhanced subsystems
  final ResourceCache _resourceCache;
  final AssetManager _assetManager;
  ResourceGovernor? _resourceGovernor;
  final MainThreadMonitor _threadMonitor = MainThreadMonitor();

  final QueueConfig _queueConfig;

  // Pipeline state
  bool _isRunning = false;

  bool _isShuttingDown = false;

  bool debugEnabled = false;

  // Stream merging state
  final Map<String, Map<String, Tensor>> _lastStreamTensors = {};

  // Master-clock merging + per-stream snapshots.
  //
  //  1. Per-stream stages cache ONE output tensor per stage instance, shared
  //     by every stream — so "stream A's output" and "stream B's output"
  //     alias the SAME GPU buffer.  Merging latest-vs-latest requires a
  //     private snapshot copy per stream.
  //  2. Merge cadence follows ONE stream (the "master" — first to deliver),
  //     merging the latest snapshot of every other stream, fresh or not.
  //     The previous all-streams barrier tied the merge to the SLOWEST
  //     device and, worse, to pairing luck between independent device
  //     clocks: drifting mic/loopback clocks and silence-gated loopback
  //     sources (WASAPI delivers nothing while no audio renders) made the
  //     merged feed stall for up to the straggler timeout — the "tiny gaps"
  //     in multi-device spectrograms.  A master clock emits at exactly one
  //     device's hop cadence; other streams are at most one of their own
  //     frames stale, which is invisible in a visualisation.
  //  3. If the master itself stalls (device stopped / silence-gated), the
  //     freshest delivering stream is promoted after ~3× the master's
  //     observed frame interval, so output degrades to the surviving
  //     device's cadence instead of freezing.
  //  4. Snapshots older than ~4× their stream's own observed interval are
  //     excluded from the merge — a stopped device fades out instead of
  //     overlaying its last spectrum forever.
  final Map<String, Tensor> _mergeSnapshots = {};
  ComputeShader? _mergeCopyShader;
  int _mergeCopySize = 0;

  /// Per-stream time of the most recent snapshot.
  final Map<String, DateTime> _mergeSnapshotAt = {};

  /// Per-stream EMA of the interval between snapshots (frame cadence).
  final Map<String, Duration> _mergeCadence = {};

  /// Streams that have delivered a fresh frame since the last merge.
  final Set<String> _mergeFreshSince = <String>{};

  /// The stream whose frames drive the merge cadence.
  String? _mergeMasterId;

  /// Fallback stall/staleness bound used before a stream's cadence is known.
  static const Duration _mergeStragglerTimeout = Duration(milliseconds: 250);

  // Event system
  late final PipelineEventNotifier _eventNotifier;
  DateTime? _startTime;

  // Performance monitoring
  final Map<String, Duration> _stageTimings = {};
  final Map<String, int> _stageExecutionCounts = {};

  // Connections for serialized pipelines, and the ones into each stage
  List<ConnectionConfig> _connections = const [];
  Map<String, List<ConnectionConfig>> _incoming = const {};

  List<AssetConfig> _serializedAssets = const [];

  /// The network assets this pipeline holds, from [start] until [stop].
  final List<String> _heldAssets = [];

  // External inputs cache (for inter-pipeline links) and latest outputs
  final Map<String, Tensor> _externalInputs = <String, Tensor>{};
  Map<String, Tensor>? _latestFinalOutputs;

  // Reusable ingest tensors per stream
  final Map<String, List<Tensor>> _ingestPools = {};
  final Set<Tensor> _ingestInUse = <Tensor>{};
  final Map<Tensor, String> _ingestOwner = {};
  // Batched queue flush timer
  Timer? _batchFlushTimer;

  /// Creates a pipeline.
  ///
  /// [maxCachedShaders] and [maxCachedTensors] bound [resourceCache].
  /// [maxMemoryMB] is the [assetManager]'s cache limit. A manager the
  /// pipeline creates gets 4096 MB by default; a manager passed in may be
  /// shared with other pipelines, so it keeps its own limit unless
  /// [maxMemoryMB] is given.
  Pipeline({
    required this.id,
    QueueConfig? queueConfig,
    bool enableBenchmarking = false,
    int maxEvents = 1000,
    this.debugEnabled = false,
    int maxCachedShaders = 100,
    int maxCachedTensors = 200,
    int? maxMemoryMB,
    AssetManager? assetManager,
  }) : _resourceCache = ResourceCache.forPipeline(
         id,
         maxShaders: maxCachedShaders,
         maxTensors: maxCachedTensors,
       ),
       _assetManager = assetManager ?? AssetManager(),
       _queueConfig = queueConfig ?? const QueueConfig() {
    _eventNotifier = PipelineEventNotifier(
      enableBenchmarking: enableBenchmarking,
      maxEvents: maxEvents,
    );

    if (assetManager == null || maxMemoryMB != null) {
      _assetManager.setCacheLimit((maxMemoryMB ?? 4096) * 1024 * 1024);
    }
  }

  // Getters
  bool get isRunning => _isRunning;
  bool get isProcessing => _isProcessing;

  /// The process-wide event-loop monitor. The pipeline does not start it:
  /// call `startMonitoring()` to sample event-loop delay, and
  /// `stopMonitoring()` when done, since it runs a 5 ms timer while on.
  MainThreadMonitor get threadMonitor => _threadMonitor;
  PipelineEventNotifier get eventNotifier => _eventNotifier;
  ResourceCache get resourceCache => _resourceCache;
  AssetManager get assetManager => _assetManager;

  /// A standalone placeholder: the pipeline neither initializes nor
  /// consults it.
  @Deprecated(
    'The pipeline does not use it, and its device list and metrics are '
    'invented. Will be removed in 2.0.',
  )
  ResourceGovernor get resourceGovernor =>
      _resourceGovernor ??= ResourceGovernor();

  // Message is a CLOSURE: Dart evaluates argument expressions eagerly, so a
  // plain String parameter paid for every hot-path interpolation (shapes,
  // queue sizes) even with debug logging disabled.
  void _debugLog(String Function() message) {
    if (debugEnabled) {
      print('🐛 Pipeline $id: ${message()}');
    }
  }

  /// Event management
  void addEventListener(
    PipelineEventType type,
    void Function(PipelineEvent) listener,
  ) {
    _eventNotifier.addEventListener(type, listener);
  }

  void removeEventListener(
    PipelineEventType type,
    void Function(PipelineEvent) listener,
  ) {
    _eventNotifier.removeEventListener(type, listener);
  }

  /// Adds [stage], run for the streams [streamConfig] selects.
  ///
  /// Throws a [StateError] for a merge stage in a pipeline with serialized
  /// connections, since merged streams cannot be routed.
  void addStage(PipelineStage stage, [StageStreamConfig? streamConfig]) {
    if (streamConfig?.routing == StreamRouting.merged &&
        _connections.isNotEmpty) {
      throw StateError(_mergeWithConnections);
    }
    stage.setEventNotifier(_eventNotifier);

    if (stage is DynamicStage) {
      stage.setResourceCache(_resourceCache);
    }

    _stages.add(stage);
    if (streamConfig != null) {
      _stageStreamConfigs[stage.stageId] = streamConfig;
    }

    _debugLog(() => '➕ Added stage: ${stage.stageId} (${stage.runtimeType})');
  }

  /// Add stage with merge routing
  void addMergeStage(PipelineStage stage) {
    final config = StageStreamConfig(routing: StreamRouting.merged);
    addStage(stage, config);
  }

  /// Add stage with selective routing
  void addSelectiveStage(PipelineStage stage, {required Set<String> streams}) {
    final config = StageStreamConfig(
      routing: StreamRouting.selective,
      includedStreams: streams,
    );
    addStage(stage, config);
  }

  /// Add media stream with per-stream ingestion serialization and reuse
  void addStream(MediaStream stream) {
    _streams.add(stream);
    _debugLog(() => '➕ Added stream: ${stream.id} (${stream.type})');

    final format = _getStreamFormat(stream.type);

    bool unifiedAttached = false;
    // New unified callback (host or GPU) with ingestion serialization
    try {
      stream.onDataUpdate((frame) {
        if (!_isRunning || _isShuttingDown) return;
        if (!_streams.contains(stream)) return;

        // Chain ingestion work per-stream to prevent concurrent host uploads
        final prev = _ingestSerializers[stream.id] ?? Future.value();
        final next = prev
            .then((_) async {
              await _handleUnifiedFrame(stream.id, format, frame);
            })
            .catchError((e, st) {
              _debugLog(() => '❌ Error handling frame for ${stream.id}: $e');
            });
        _ingestSerializers[stream.id] = next.whenComplete(() {});
      });
      unifiedAttached = true;
    } catch (_) {
      unifiedAttached = false;
    }

    // Legacy fallback (GPU tensor only)
    if (!unifiedAttached) {
      stream.onTensorUpdate((tensor) {
        if (!_isRunning || _isShuttingDown) return;
        if (!_streams.contains(stream)) return;
        if (!_isValidTensor(tensor)) return;

        // Allocate the event only if someone is listening — these fire per
        // stage per frame.
        if (_eventNotifier.hasListeners(PipelineEventType.tensorUpdate)) {
          _eventNotifier.emit(
            TensorUpdateEvent(
              pipelineId: id,
              sourceId: stream.id,
              format: format,
              tensorShape: tensor.shape,
              dataSize: tensor.shape.fold(1, (a, b) => a * b),
            ),
          );
        }

        try {
          _enqueueProcessing({'input': tensor}, format, stream.id);
        } catch (e) {
          _debugLog(() => '❌ Error enqueueing tensor from ${stream.id}: $e');
        }
      });
    }
  }

  /// Unified frame handling for both GPU and host frames (serialized per stream)
  Future<void> _handleUnifiedFrame(
    String streamId,
    String format,
    dynamic frame,
  ) async {
    if (!_isRunning || _isShuttingDown) return;
    if (!_streams.any((s) => s.id == streamId)) return;

    try {
      if (frame.isGpu == true) {
        final tensor = frame.tensor as Tensor;
        if (!_isValidTensor(tensor)) return;

        // Allocate the event only if someone is listening — these fire per
        // stage per frame.
        if (_eventNotifier.hasListeners(PipelineEventType.tensorUpdate)) {
          _eventNotifier.emit(
            TensorUpdateEvent(
              pipelineId: id,
              sourceId: streamId,
              format: format,
              tensorShape: tensor.shape,
              dataSize: tensor.shape.fold(1, (a, b) => a * b),
            ),
          );
        }
        _enqueueProcessing({'input': tensor}, format, streamId);
      } else {
        // Host-first: enqueue host buffers without GPU upload
        final TypedData data = frame.hostData as TypedData;
        final List<int> shape = List<int>.from(frame.shape as List<int>);
        final BufferDataType dataType = frame.dataType as BufferDataType;

        final elementCount = _elementCountFor(dataType, data.lengthInBytes);
        final Float32List f32 = _convertToFloat32(data, dataType, elementCount);
        final List<int> fixedShape = _fixShape(shape, elementCount);

        // Allocate the event only if someone is listening — these fire per
        // stage per frame.
        if (_eventNotifier.hasListeners(PipelineEventType.tensorUpdate)) {
          _eventNotifier.emit(
            TensorUpdateEvent(
              pipelineId: id,
              sourceId: streamId,
              format: format,
              tensorShape: fixedShape,
              dataSize: elementCount,
            ),
          );
        }

        _enqueueHostProcessing(
          {'input': f32},
          {'input': fixedShape},
          format,
          streamId,
        );
      }
    } catch (e) {
      _debugLog(() => '❌ Error handling frame for $streamId: $e');
    }
  }

  /// Uploads [data], a host buffer of [streamId], into a pooled GPU tensor
  /// of the same element type. [shape] is flattened when its element count
  /// does not match the data's.
  Future<Tensor> _uploadHostData(
    String streamId,
    TypedData data,
    List<int> shape,
  ) async {
    // Bytes without an element type upload as uint8.
    final TypedData values = data is ByteData || data is Uint8ClampedList
        ? Uint8List.view(data.buffer, data.offsetInBytes, data.lengthInBytes)
        : data;
    final dataType = _hostDataType(values);
    final fixedShape = _fixShape(shape, _elementCountOf(values));

    final tensor = await _acquireIngestTensor(streamId, fixedShape, dataType);
    await tensor.write(values);

    _debugLog(
      () =>
          '[INGEST] uploaded stream=$streamId type=$dataType '
          'shape=${fixedShape.join("x")}',
    );
    return tensor;
  }

  /// The GPU element type for host buffer [data]: float32 for a list type
  /// without its own.
  static BufferDataType _hostDataType(TypedData data) => switch (data) {
    Float64List() => BufferDataType.float64,
    Int8List() => BufferDataType.int8,
    Uint8List() => BufferDataType.uint8,
    Int16List() => BufferDataType.int16,
    Uint16List() => BufferDataType.uint16,
    Int32List() => BufferDataType.int32,
    Uint32List() => BufferDataType.uint32,
    Int64List() => BufferDataType.int64,
    Uint64List() => BufferDataType.uint64,
    _ => BufferDataType.float32,
  };

  /// A zeroed tensor of [dataType] whose data reads back as that type's
  /// list.
  static Future<Tensor> _zeros(
    List<int> shape,
    BufferDataType dataType,
  ) => switch (dataType) {
    BufferDataType.float32 => Tensor.zeros<Float32List>(shape),
    BufferDataType.float64 => Tensor.zeros<Float64List>(
      shape,
      dataType: dataType,
    ),
    BufferDataType.int8 => Tensor.zeros<Int8List>(shape, dataType: dataType),
    BufferDataType.uint8 => Tensor.zeros<Uint8List>(shape, dataType: dataType),
    BufferDataType.int16 => Tensor.zeros<Int16List>(shape, dataType: dataType),
    BufferDataType.uint16 || BufferDataType.float16 => Tensor.zeros<Uint16List>(
      shape,
      dataType: dataType,
    ),
    BufferDataType.int32 => Tensor.zeros<Int32List>(shape, dataType: dataType),
    BufferDataType.uint32 => Tensor.zeros<Uint32List>(
      shape,
      dataType: dataType,
    ),
    BufferDataType.int64 => Tensor.zeros<Int64List>(shape, dataType: dataType),
    BufferDataType.uint64 => Tensor.zeros<Uint64List>(
      shape,
      dataType: dataType,
    ),
  };

  int _elementCountFor(BufferDataType dataType, int byteLength) {
    switch (dataType) {
      case BufferDataType.float64:
      case BufferDataType.int64:
      case BufferDataType.uint64:
        return byteLength ~/ 8;
      case BufferDataType.float32:
      case BufferDataType.int32:
      case BufferDataType.uint32:
        return byteLength ~/ 4;
      case BufferDataType.int16:
      case BufferDataType.uint16:
      case BufferDataType.float16:
        return byteLength ~/ 2;
      case BufferDataType.int8:
      case BufferDataType.uint8:
        return byteLength;
    }
  }

  Float32List _convertToFloat32(
    TypedData data,
    BufferDataType dataType,
    int elementCount,
  ) {
    if (dataType == BufferDataType.float32) {
      if (data is Float32List) return data;
      return Float32List.view(data.buffer, data.offsetInBytes, elementCount);
    }

    final f32 = Float32List(elementCount);
    if (dataType == BufferDataType.int16) {
      final view = data is Int16List
          ? data
          : Int16List.view(data.buffer, data.offsetInBytes, elementCount);
      for (var i = 0; i < elementCount; i++) {
        f32[i] = view[i] / 32768.0;
      }
    } else if (dataType == BufferDataType.uint16) {
      final view = data is Uint16List
          ? data
          : Uint16List.view(data.buffer, data.offsetInBytes, elementCount);
      for (var i = 0; i < elementCount; i++) {
        f32[i] = (view[i] - 32768) / 32768.0;
      }
    } else if (dataType == BufferDataType.int8) {
      final view = data is Int8List
          ? data
          : Int8List.view(data.buffer, data.offsetInBytes, elementCount);
      for (var i = 0; i < elementCount; i++) {
        f32[i] = view[i] / 128.0;
      }
    } else if (dataType == BufferDataType.uint8) {
      final view = data is Uint8List
          ? data
          : Uint8List.view(data.buffer, data.offsetInBytes, elementCount);
      for (var i = 0; i < elementCount; i++) {
        f32[i] = (view[i] - 128) / 128.0;
      }
    } else if (dataType == BufferDataType.int32) {
      final view = data is Int32List
          ? data
          : Int32List.view(data.buffer, data.offsetInBytes, elementCount);
      for (var i = 0; i < elementCount; i++) {
        f32[i] = view[i].toDouble();
      }
    } else if (dataType == BufferDataType.uint32) {
      final view = data is Uint32List
          ? data
          : Uint32List.view(data.buffer, data.offsetInBytes, elementCount);
      for (var i = 0; i < elementCount; i++) {
        f32[i] = view[i].toDouble();
      }
    } else if (dataType == BufferDataType.float64) {
      final view = data is Float64List
          ? data
          : Float64List.view(data.buffer, data.offsetInBytes, elementCount);
      for (var i = 0; i < elementCount; i++) {
        f32[i] = view[i];
      }
    } else if (dataType == BufferDataType.float16) {
      final view = data is Uint16List
          ? data
          : Uint16List.view(data.buffer, data.offsetInBytes, elementCount);
      for (var i = 0; i < elementCount; i++) {
        f32[i] = _halfToDouble(view[i]);
      }
    } else {
      // int64 and uint64, read as two 32-bit halves: the web has no 64-bit
      // integer lists.
      final bytes = ByteData.view(
        data.buffer,
        data.offsetInBytes,
        elementCount * 8,
      );
      final signed = dataType == BufferDataType.int64;
      final lowFirst = Endian.host == Endian.little;
      for (var i = 0; i < elementCount; i++) {
        final low = bytes.getUint32(i * 8 + (lowFirst ? 0 : 4), Endian.host);
        final highAt = i * 8 + (lowFirst ? 4 : 0);
        final high = signed
            ? bytes.getInt32(highAt, Endian.host)
            : bytes.getUint32(highAt, Endian.host);
        f32[i] = high * 4294967296.0 + low;
      }
    }
    return f32;
  }

  /// The value of the IEEE 754 half-precision number with these [bits].
  static double _halfToDouble(int bits) {
    final sign = (bits & 0x8000) != 0 ? -1.0 : 1.0;
    final exponent = (bits >> 10) & 0x1f;
    final fraction = bits & 0x3ff;
    if (exponent == 0x1f) {
      return fraction == 0 ? sign * double.infinity : double.nan;
    }
    if (exponent == 0) return sign * fraction / 16777216;
    return sign * (1024 + fraction) * math.pow(2, exponent - 25);
  }

  /// Start pipeline
  Future<void> start() async {
    if (_isRunning) return;

    _startTime = DateTime.now();
    _debugLog(() => '🚀 Starting pipeline...');

    _eventNotifier.emit(
      PipelineStartEvent(
        pipelineId: id,
        stageCount: _stages.length,
        streamCount: _streams.length,
      ),
    );

    try {
      await _loadNetworkAssets();

      // Preload all stage assets
      await _preloadAllAssets();

      // Initialize stages
      await _initializeStages();

      // Start streams
      await _startAllStreams();

      _isRunning = true;

      _debugLog(() => '✅ Pipeline started successfully');
    } catch (e, stackTrace) {
      _eventNotifier.emit(
        PipelineErrorEvent(
          pipelineId: id,
          error: e.toString(),
          stackTrace: stackTrace,
        ),
      );
      await _releaseAssets();
      rethrow;
    }
  }

  // allow RuntimeBuilder/Loader to pass serialized assets
  void setSerializedAssets(List<AssetConfig> assets) {
    _serializedAssets = List<AssetConfig>.unmodifiable(assets);
  }

  /// Loads the network assets whose strategy is `preload`, and holds each
  /// until [stop]. An optional one that fails to load is skipped; the asset
  /// manager reports it with a `failed` event.
  Future<void> _loadNetworkAssets() async {
    for (final ref in _serializedAssets) {
      if (ref.strategy != 'preload') continue;
      try {
        await _assetManager.load(ref);
      } catch (_) {
        if (ref.required) rethrow;
        continue;
      }
      _heldAssets.add(ref.name);
    }
  }

  /// Drops the references to network and stage assets that [start] took.
  Future<void> _releaseAssets() async {
    final held = List.of(_heldAssets);
    _heldAssets.clear();
    for (final name in held) {
      await _assetManager.release(name);
    }
    for (final stage in _stages) {
      await stage.releaseAssets();
    }
  }

  /// Preloads each stage's assets into the asset manager, once.
  Future<void> _preloadAllAssets() async {
    _debugLog(() => '📦 Preloading assets for ${_stages.length} stages...');

    for (final stage in _stages) {
      await stage.preloadAssets(_assetManager);
    }
  }

  /// Initialize stages
  Future<void> _initializeStages() async {
    _debugLog(() => '🔧 Initializing ${_stages.length} stages...');

    for (final stage in _stages) {
      try {
        _debugLog(() => '🔧 Initializing stage: ${stage.stageId}');
        await stage.initializeStage();
        _debugLog(() => '✅ Stage ${stage.stageId} initialized successfully');
      } catch (e, stackTrace) {
        _debugLog(() => '❌ Failed to initialize stage ${stage.stageId}: $e');
        _eventNotifier.emit(
          StageErrorEvent(
            pipelineId: id,
            stageId: stage.stageId,
            error: e.toString(),
            stackTrace: stackTrace,
          ),
        );
        rethrow;
      }
    }
  }

  /// Start all media streams
  Future<void> _startAllStreams() async {
    for (final stream in _streams) {
      try {
        _debugLog(() => '🎤 Starting stream: ${stream.id}');
        await stream.start();

        _eventNotifier.emit(
          StreamStartEvent(
            pipelineId: id,
            streamId: stream.id,
            streamType: stream.type.toString(),
          ),
        );

        _debugLog(() => '✅ Stream ${stream.id} started successfully');
      } catch (e, stackTrace) {
        _debugLog(() => '❌ Failed to start stream ${stream.id}: $e');
        _eventNotifier.emit(
          StreamErrorEvent(
            pipelineId: id,
            streamId: stream.id,
            error: e.toString(),
            stackTrace: stackTrace,
          ),
        );
        rethrow;
      }
    }
  }

  /// Enqueue a GPU-tensor item
  void _enqueueProcessing(
    Map<String, Tensor> tensors,
    String format,
    String streamId,
  ) {
    // Safety checks
    if (!_isRunning || _isShuttingDown) {
      _debugLog(
        () => '⚠️ Ignoring enqueue - pipeline not running or shutting down',
      );
      return;
    }

    final inputTensor = tensors['input'];
    if (inputTensor == null || !_isValidTensor(inputTensor)) {
      _debugLog(() => '❌ Invalid tensor from stream $streamId');
      return;
    }

    // Check if stream still exists
    if (!_streams.any((s) => s.id == streamId)) {
      _debugLog(() => '⚠️ Ignoring tensor from removed stream $streamId');
      return;
    }

    final item = _ProcessingItem(
      streamId: streamId,
      // Callers pass freshly built map literals — no defensive copy needed.
      tensors: tensors,
      format: format,
      timestamp: DateTime.now(),
    );

    _queueItem(item);
  }

  /// Enqueue a host item (TypedData on CPU)
  void _enqueueHostProcessing(
    Map<String, TypedData> host,
    Map<String, List<int>> shapes,
    String format,
    String streamId,
  ) {
    if (!_isRunning || _isShuttingDown) return;
    if (!_streams.any((s) => s.id == streamId)) return;

    final item = _ProcessingItem(
      streamId: streamId,
      // Callers pass freshly built map literals — no defensive copy needed.
      host: host,
      hostShapes: shapes,
      format: format,
      timestamp: DateTime.now(),
    );

    _queueItem(item);
  }

  void _queueItem(_ProcessingItem item) {
    // Add to active processing tracking
    _activeProcessingItems.add(item.streamId);

    switch (_queueConfig.strategy) {
      case QueueStrategy.latestOnly:
        _enqueueLatestOnly(item);
        break;
      case QueueStrategy.latestN:
        _enqueueLatestN(item);
        break;
      case QueueStrategy.dropOnOverrun:
        _enqueueDropOnOverrun(item);
        break;
      case QueueStrategy.batched:
        _enqueueBatched(item);
        break;
      case QueueStrategy.processAll:
        _enqueueProcessAll(item);
        break;
    }

    // Trigger processing immediately if not already processing
    _tryProcessNext();
  }

  /// Strategy: Keep only latest item per stream
  void _enqueueLatestOnly(_ProcessingItem item) {
    _processingQueue.removeWhere(
      (existing) => existing.streamId == item.streamId,
    );
    _processingQueue.add(item);
    _debugLog(
      () =>
          '📥 Queued latest for ${item.streamId} (queue: ${_processingQueue.length})',
    );
  }

  /// Strategy: Keep N latest items per stream
  void _enqueueLatestN(_ProcessingItem item) {
    final streamItems = _processingQueue
        .where((i) => i.streamId == item.streamId)
        .toList();

    // Remove excess items for this stream
    while (streamItems.length >= _queueConfig.maxItemsPerStream) {
      final oldest = streamItems.removeAt(0);
      _processingQueue.remove(oldest);
    }

    _processingQueue.add(item);
    _debugLog(
      () =>
          '📥 Queued (${streamItems.length + 1}/${_queueConfig.maxItemsPerStream}) for ${item.streamId}',
    );
  }

  /// Strategy: Drop if processing can't keep up
  void _enqueueDropOnOverrun(_ProcessingItem item) {
    if (_isProcessing &&
        _processingQueue.length >= _queueConfig.maxTotalItems) {
      _debugLog(
        () => '🗑️ Dropped item from ${item.streamId} - processing overrun',
      );
      return;
    }

    _processingQueue.add(item);
    _debugLog(
      () =>
          '📥 Queued for ${item.streamId} (queue: ${_processingQueue.length})',
    );
  }

  /// Strategy: Batch items for processing
  void _enqueueBatched(_ProcessingItem item) {
    _processingQueue.add(item);

    // Remove old items beyond max age
    final cutoff = DateTime.now().subtract(_queueConfig.maxAge);
    _processingQueue.removeWhere((it) => it.timestamp.isBefore(cutoff));

    // Schedule a flush after maxAge if we don't have enough items yet
    if (_processingQueue.length >= _queueConfig.batchSize) {
      // Enough items to process now; cancel any pending flush timer
      _batchFlushTimer?.cancel();
      _batchFlushTimer = null;
    } else {
      // Not enough items: ensure a flush will happen at maxAge
      _batchFlushTimer ??= Timer(_queueConfig.maxAge, () {
        _batchFlushTimer = null;
        if (_isRunning && !_isShuttingDown) {
          _tryProcessNext();
        }
      });
    }

    _debugLog(
      () =>
          '📥 Batched item from ${item.streamId} (queue: ${_processingQueue.length})',
    );
  }

  /// Strategy: Process all items
  void _enqueueProcessAll(_ProcessingItem item) {
    _processingQueue.add(item);

    // Safety limit to prevent memory issues
    while (_processingQueue.length > _queueConfig.maxTotalItems) {
      final removed = _processingQueue.removeAt(0);
      _debugLog(
        () => '🗑️ Dropped oldest item from ${removed.streamId} - safety limit',
      );
    }

    _debugLog(
      () =>
          '📥 Queued for ${item.streamId} (queue: ${_processingQueue.length})',
    );
  }

  /// Event-driven processing - try to process next item immediately
  void _tryProcessNext() {
    if (_isProcessing ||
        _processingQueue.isEmpty ||
        !_isRunning ||
        _isShuttingDown) {
      return;
    }

    // Use scheduleMicrotask for better async handling
    scheduleMicrotask(() async {
      if (!_isRunning ||
          _isProcessing ||
          _processingQueue.isEmpty ||
          _isShuttingDown) {
        return;
      }

      _isProcessing = true;

      try {
        final item = _getNextItem();
        if (item != null) {
          // Double-check stream still exists and is valid
          if (!_streams.any((s) => s.id == item.streamId)) {
            _debugLog(
              () =>
                  '⚠️ Skipping processing for removed stream ${item.streamId}',
            );
            return;
          }

          // Validate only if GPU tensors are provided
          if (!item.isHost && item.tensors != null) {
            for (final tensor in item.tensors!.values) {
              if (!_isValidTensor(tensor)) {
                _debugLog(
                  () => '❌ Invalid tensor in item from ${item.streamId}',
                );
                return;
              }
            }
          }

          await _processItem(item);
        }
      } catch (e, stackTrace) {
        _debugLog(() => '❌ Error processing queue item: $e');
        _eventNotifier.emit(
          PipelineErrorEvent(
            pipelineId: id,
            error: e.toString(),
            stackTrace: stackTrace,
          ),
        );
      } finally {
        _isProcessing = false;

        // Remove from active processing
        _activeProcessingItems.removeWhere(
          (sid) => _processingQueue.where((it) => it.streamId == sid).isEmpty,
        );

        // If queue drained, cancel any pending batch flush
        if (_processingQueue.isEmpty) {
          _batchFlushTimer?.cancel();
          _batchFlushTimer = null;
        }

        // Only continue if pipeline is still running
        if (_isRunning && !_isShuttingDown) {
          // Yield control before continuing
          await Future.delayed(Duration.zero);

          // Continue processing if more items available
          if (_processingQueue.isNotEmpty) {
            _tryProcessNext();
          }
        }
      }
    });
  }

  /// Get next item based on strategy
  _ProcessingItem? _getNextItem() {
    if (_processingQueue.isEmpty) return null;

    switch (_queueConfig.strategy) {
      case QueueStrategy.batched:
        return _getNextBatchItem();
      default:
        return _processingQueue.removeAt(0);
    }
  }

  /// Get next item for batch processing
  _ProcessingItem? _getNextBatchItem() {
    if (_processingQueue.length < _queueConfig.batchSize) {
      // Wait for more items unless we have old items
      final cutoff = DateTime.now().subtract(_queueConfig.maxAge);
      if (_processingQueue.isNotEmpty &&
          _processingQueue.first.timestamp.isBefore(cutoff)) {
        return _processingQueue.removeAt(0);
      }
      return null;
    }

    return _processingQueue.removeAt(0);
  }

  /// Process a single item from the queue
  Future<void> _processItem(_ProcessingItem item) async {
    _debugLog(() => '🔄 Processing stream ${item.streamId}');

    if (item.isHost) {
      // CPU-first path with CPU↔GPU toggling.
      // Host buffers are uploaded into tensors from the ingest pool —
      // release any tensors marked in-use for this stream after processing
      // so the pool can reuse them.
      try {
        if (_connections.isNotEmpty) {
          final shapes = item.hostShapes ?? const <String, List<int>>{};
          final frame = <String, Tensor>{
            for (final entry in item.host!.entries)
              entry.key: await _uploadHostData(
                item.streamId,
                entry.value,
                shapes[entry.key] ?? [_elementCountOf(entry.value)],
              ),
          };
          await _processStreamRouted(frame, item.format, item.streamId);
          return;
        }
        await _processStreamIndividuallyHost(
          item.host!,
          item.hostShapes ?? const {},
          item.format,
          item.streamId,
        );
        // Host-fed streams (the production audio path) must trigger merging
        // too — this early-return branch used to skip the merge check
        // entirely, so with a merge stage in the graph the post-merge stages
        // NEVER ran for real audio streams.
        if (_needsMerging() && _readyToMerge()) {
          await _processMergedStreams();
        }
      } finally {
        _releaseIngestTensorsForStream(item.streamId);
      }
      return;
    }

    // GPU path:

    if (_connections.isNotEmpty) {
      await _processStreamRouted(item.tensors!, item.format, item.streamId);
      _releaseIngestTensorsForItem(item.streamId, item.tensors!);
      return;
    }

    // Store for merging.  In merging mode the entry is replaced with a
    // SNAPSHOT after per-stream processing — raw ingest tensors are pooled
    // and released below, so they must never linger in the merge set.
    if (!_needsMerging()) {
      _lastStreamTensors[item.streamId] = item.tensors!;
    }

    // ALWAYS process individual streams first (for per-stream stages)
    await _processStreamIndividually(item.tensors!, item.format, item.streamId);

    // THEN merge — but only once every active stream has contributed a
    // fresh frame (or a straggler timed out).  See _readyToMerge().
    if (_needsMerging() && _readyToMerge()) {
      await _processMergedStreams();
    }

    // Release ingest tensors used by this item back to the pool
    _releaseIngestTensorsForItem(item.streamId, item.tensors!);
  }

  /// True when the merge cadence says a new merged frame is due: the master
  /// stream delivered a fresh frame, or the master has stalled past ~3× its
  /// observed frame interval and another stream is delivering (that stream
  /// is promoted to master so cadence stays regular).
  bool _readyToMerge() {
    if (_mergeFreshSince.isEmpty) return false;
    final master = _mergeMasterId;
    if (master == null || _mergeFreshSince.contains(master)) return true;

    final masterAt = _mergeSnapshotAt[master];
    if (masterAt == null) {
      _mergeMasterId = _freshestDeliveringStream();
      return true;
    }
    final cadence = _mergeCadence[master];
    Duration stallAfter = cadence == null
        ? _mergeStragglerTimeout
        : cadence * 3;
    if (stallAfter < const Duration(milliseconds: 50)) {
      stallAfter = const Duration(milliseconds: 50);
    }
    if (DateTime.now().difference(masterAt) > stallAfter) {
      final promoted = _freshestDeliveringStream();
      _debugLog(
        () =>
            '🔀 Merge master $master stalled (> $stallAfter) — promoting $promoted',
      );
      _mergeMasterId = promoted;
      return true;
    }
    return false;
  }

  /// The stream with the most recent fresh frame (merge-master candidate).
  String? _freshestDeliveringStream() {
    String? best;
    DateTime? bestAt;
    for (final id in _mergeFreshSince) {
      final at = _mergeSnapshotAt[id];
      if (at != null && (bestAt == null || at.isAfter(bestAt))) {
        best = id;
        bestAt = at;
      }
    }
    return best ?? _mergeMasterId;
  }

  /// The stream's meaningful payload from a stage-result map.
  Tensor? _primaryTensorOf(Map<String, Tensor> tensors) {
    return tensors['output'] ??
        tensors['input'] ??
        (tensors.isNotEmpty ? tensors.values.first : null);
  }

  /// Copies [src] into a per-stream persistent snapshot tensor and records
  /// it in the merge set.  Copying is REQUIRED: per-stream stages reuse one
  /// cached output buffer across streams (and ingest tensors are pooled), so
  /// without a private copy every stream's "latest" would alias whichever
  /// stream ran last.  The copy shader is cached and only recompiled when
  /// the element count changes (never per frame).
  Future<void> _storeMergeSnapshot(String streamId, Tensor src) async {
    var dst = _mergeSnapshots[streamId];
    if (dst == null || dst.size != src.size || !dst.buffer.isValid) {
      dst?.destroy();
      dst = await Tensor.create(
        src.shape,
        gpu: src.gpu,
        dataType: src.dataType,
      );
      _mergeSnapshots[streamId] = dst;
    }
    if (_mergeCopyShader == null || _mergeCopySize != src.size) {
      _mergeCopyShader?.destroy();
      _mergeCopyShader = src.gpu.createComputeShader()
        ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${src.size}u) { return; }
  dst[i] = src[i];
}
''');
      _mergeCopySize = src.size;
    }
    final shader = _mergeCopyShader!;
    shader.setBuffer('src', src.buffer);
    shader.setBuffer('dst', dst.buffer);
    await shader.dispatch((src.size + 255) ~/ 256, 1, 1);

    _lastStreamTensors[streamId] = {'output': dst};

    // Merge bookkeeping: freshness, per-stream cadence estimate (EMA over
    // snapshot intervals — drives master-stall detection and staleness
    // expiry), and master assignment (first deliverer).
    final now = DateTime.now();
    final prev = _mergeSnapshotAt[streamId];
    if (prev != null) {
      final interval = now.difference(prev);
      if (interval > Duration.zero && interval < const Duration(seconds: 2)) {
        final old = _mergeCadence[streamId];
        _mergeCadence[streamId] = old == null
            ? interval
            : (old * 3 + interval) ~/ 4;
      }
    }
    _mergeSnapshotAt[streamId] = now;
    _mergeFreshSince.add(streamId);
    _mergeMasterId ??= streamId;
  }

  /// Process stream individually through all stages UNTIL we hit a merge stage
  Future<void> _processStreamIndividually(
    Map<String, Tensor> tensors,
    String format,
    String streamId,
  ) async {
    String currentFormat = format;
    Map<String, Tensor> currentTensors = tensors;
    Map<String, PipelineTensor>? ptBuffer;
    // The PT fast path defers GPU materialization across consecutive
    // DynamicStages, so per-stage outputs can't be recorded as each stage
    // runs (that would force an upload per stage and defeat the deferral).
    // Track the LAST dynamic stage so the flush can record its output —
    // exact for single stages (the common case); interior stages of an
    // unbroken chain stay unrecorded by design.
    String? lastPtStageId;
    final stageTimings = <String, Duration>{};
    int stagesProcessed = 0;
    final runStart = DateTime.now();

    _debugLog(
      () => '🔄 Processing stream $streamId through stages until merge point',
    );

    for (int i = 0; i < _stages.length; i++) {
      final stage = _stages[i];
      if (!_isRunning) break;

      final stageConfig =
          _stageStreamConfigs[stage.stageId] ?? const StageStreamConfig();

      // STOP when we hit the first merge stage
      if (stageConfig.routing == StreamRouting.merged) {
        _debugLog(
          () =>
              '🔀 Hit merge stage ${stage.stageId} - stopping individual processing',
        );
        break;
      }

      // Check selective routing
      if (!stageConfig.shouldProcessStream(streamId)) {
        _debugLog(
          () => '⏭️ Skipping stage ${stage.stageId} for stream $streamId',
        );
        continue;
      }

      _debugLog(
        () => '⚙️ Executing stage ${stage.stageId} for stream $streamId',
      );

      if (stage is DynamicStage) {
        // Thread PipelineTensors across consecutive DynamicStages to defer
        // GPU uploads until a non-DynamicStage (or end of pipeline) needs them.
        ptBuffer ??= {
          for (final e in currentTensors.entries)
            e.key: PipelineTensor.gpu(e.value, e.value.shape),
        };
        final ptResult = await _executeDynamicStageWithPT(
          stage,
          ptBuffer,
          currentFormat,
          streamId,
          stageTimings,
        );
        if (ptResult != null) {
          ptBuffer = ptResult;
          currentFormat = stage.outputFormat;
          stagesProcessed++;
          lastPtStageId = stage.stageId;
        }
        continue;
      }

      // Non-DynamicStage: flush any deferred PipelineTensor buffer to GPU first.
      if (ptBuffer != null) {
        currentTensors = await _flushPipelineTensors(ptBuffer);
        ptBuffer = null;
        if (lastPtStageId != null) {
          _stageOutputs['${lastPtStageId}_$streamId'] = currentTensors;
          lastPtStageId = null;
        }
      }

      // Prepare inputs based on what the stage expects
      final stageInputs = _prepareStageInputs(stage, currentTensors);

      final result = await _executeStage(
        stage,
        stageInputs,
        currentFormat,
        streamId,
        stageTimings,
      );

      if (result != null) {
        currentTensors = result.$1;
        currentFormat = result.$2;
        stagesProcessed++;

        _stageOutputs['${stage.stageId}_$streamId'] = currentTensors;
      }
    }

    // Flush any remaining PipelineTensor buffer to GPU tensors.
    if (ptBuffer != null) {
      currentTensors = await _flushPipelineTensors(ptBuffer);
      ptBuffer = null;
      if (lastPtStageId != null) {
        _stageOutputs['${lastPtStageId}_$streamId'] = currentTensors;
        lastPtStageId = null;
      }
    }

    // Store the last individual output (before merging).  In merging mode
    // this MUST be a private snapshot — the tensors in currentTensors are
    // stage-cached buffers shared across streams.
    if (currentTensors.isNotEmpty) {
      if (_needsMerging()) {
        final primary = _primaryTensorOf(currentTensors);
        if (primary != null && _isValidTensor(primary)) {
          await _storeMergeSnapshot(streamId, primary);
        }
      } else if (stagesProcessed > 0) {
        _lastStreamTensors[streamId] = currentTensors;
      }
    }

    // Emit completion event
    final runDuration = DateTime.now().difference(runStart);

    // Make latest outputs available to the router for link forwarding
    _latestFinalOutputs = currentTensors.isNotEmpty
        ? Map<String, Tensor>.from(currentTensors)
        : null;

    // Record per-stream final outputs for test/introspection
    if (currentTensors.isNotEmpty) {
      _stageOutputs['final_$streamId'] = currentTensors;
    }

    _eventNotifier.emit(
      PipelineRunCompleteEvent(
        pipelineId: id,
        runDuration: runDuration,
        stagesProcessed: stagesProcessed,
        finalFormat: currentFormat,
        finalTensorShape: currentTensors.isNotEmpty
            ? currentTensors.values.first.shape
            : [],
        stageTimings: stageTimings,
        metadata: {'stream_id': streamId, 'processing_type': 'individual'},
      ),
    );

    _debugLog(
      () =>
          '🎯 Completed individual processing for stream $streamId through $stagesProcessed stages',
    );
  }

  /// Process a stream starting with host data through CPU-only stages,
  /// promote to GPU at the first non-CPU stage, allow demotion back to CPU-only
  /// stages, and re-promote again when a GPU stage is encountered.
  Future<void> _processStreamIndividuallyHost(
    Map<String, TypedData> hostData,
    Map<String, List<int>> hostShapes,
    String format,
    String streamId,
  ) async {
    String currentFormat = format;
    Map<String, TypedData> currentHost = Map.of(hostData);
    Map<String, List<int>> currentShapes = Map.of(hostShapes);
    Map<String, Tensor>? currentTensors; // non-null == GPU mode

    final stageTimings = <String, Duration>{};
    int stagesProcessed = 0;
    final runStart = DateTime.now();

    for (int i = 0; i < _stages.length; i++) {
      final stage = _stages[i];
      if (!_isRunning) break;

      final stageConfig =
          _stageStreamConfigs[stage.stageId] ?? const StageStreamConfig();
      if (stageConfig.routing == StreamRouting.merged) {
        _debugLog(
          () => '🔀 Hit merge stage ${stage.stageId} - stop host processing',
        );
        break;
      }
      if (!stageConfig.shouldProcessStream(streamId)) {
        _debugLog(
          () => '⏭️ Skipping stage ${stage.stageId} for stream $streamId',
        );
        continue;
      }

      final stageStart = DateTime.now();

      // GPU mode: allow demotion for CPU-only stage
      if (currentTensors != null && stage is DynamicStage && stage.isCpuOnly) {
        _debugLog(
          () =>
              '⬇️ Demoting GPU -> CPU for stage ${stage.stageId} (stream $streamId)',
        );

        // Download required inputs for this stage
        final demoted = await _demoteToHostForStage(stage, currentTensors);
        currentHost = demoted.host;
        currentShapes = demoted.shapes;
        currentTensors = null; // switch to CPU mode
      }

      if (currentTensors == null) {
        // CPU mode: run CPU-only stage or promote if GPU needed
        if (stage is DynamicStage && stage.isCpuOnly) {
          _debugLog(
            () => '⚙️ [CPU-only] ${stage.stageId} for stream $streamId',
          );
          try {
            var host = currentHost;
            var shapes = currentShapes;
            if (!stage.inputPortCollection.isEmpty) {
              final sources = _inputSources(stage, currentHost);
              host = _mapInputs(sources, currentHost);
              shapes = {
                for (final entry in host.entries)
                  entry.key:
                      currentShapes[sources[entry.key]] ??
                      [_elementCountOf(entry.value)],
              };
              _requireValidInputs(
                stage,
                stage.inputPortCollection.validateShapes(shapes),
              );
            }
            final hostOut = await stage.processCPUOnly(host, shapes);
            currentHost = hostOut.data;
            currentShapes = hostOut.shapes;
            stagesProcessed++;
            final dt = DateTime.now().difference(stageStart);
            stageTimings['${stage.stageId}_$streamId'] = dt;

            // Allocate the event only if someone is listening — these fire per
            // stage per frame.
            if (_eventNotifier.hasListeners(PipelineEventType.stageComplete)) {
              _eventNotifier.emit(
                StageCompleteEvent(
                  pipelineId: id,
                  stageId: stage.stageId,
                  outputFormat: stage.outputFormat,
                  tensorShape: currentShapes.values.isNotEmpty
                      ? currentShapes.values.first
                      : const <int>[],
                  processingTime: dt,
                  metadata: {'stream_id': streamId, 'location': 'cpu'},
                ),
              );
            }
            continue; // stay in CPU mode
          } catch (e, st) {
            _debugLog(() => '❌ CPU-only stage ${stage.stageId} failed: $e');
            _eventNotifier.emit(
              StageErrorEvent(
                pipelineId: id,
                stageId: stage.stageId,
                error: e.toString(),
                stackTrace: st,
                metadata: {'stream_id': streamId, 'location': 'cpu'},
              ),
            );
            return;
          }
        }

        // Promote: first non-CPU stage encountered
        _debugLog(
          () => '⬆️ Promoting host -> GPU before stage ${stage.stageId}',
        );
        currentTensors = <String, Tensor>{};

        // Only promote inputs required by this stage
        final requiredKeys = stage.inputPorts.isNotEmpty
            ? stage.inputPorts.map((p) => p.name).toList()
            : <String>['input'];

        final uploaded = <String, Tensor>{};
        for (final key in requiredKeys) {
          final source = currentHost.containsKey(key)
              ? key
              : currentHost.containsKey('output')
              ? 'output'
              : (currentHost.isNotEmpty ? currentHost.keys.first : null);
          if (source == null) continue;

          final td = currentHost[source]!;
          currentTensors[key] = uploaded[source] ??= await _uploadHostData(
            streamId,
            td,
            currentShapes[source] ?? [_elementCountOf(td)],
          );
        }
      }

      // GPU mode: execute stage normally
      _debugLog(() => '⚙️ [GPU] ${stage.stageId} for stream $streamId');
      final stageInputs = _prepareStageInputs(stage, currentTensors);

      final result = await _executeStage(
        stage,
        stageInputs,
        currentFormat,
        streamId,
        stageTimings,
      );

      if (result != null) {
        currentTensors = result.$1;
        currentFormat = result.$2;
        stagesProcessed++;
        _stageOutputs['${stage.stageId}_$streamId'] = currentTensors;
      }
    }

    // If we ended in GPU mode (tensors available), publish per-stream finals.
    // Merging mode stores a private snapshot (stage output buffers are
    // shared across streams — see _storeMergeSnapshot).
    if (currentTensors != null && currentTensors.isNotEmpty) {
      _stageOutputs['final_$streamId'] = currentTensors;
      if (_needsMerging()) {
        final primary = _primaryTensorOf(currentTensors);
        if (primary != null && _isValidTensor(primary)) {
          await _storeMergeSnapshot(streamId, primary);
        }
      } else {
        _lastStreamTensors[streamId] = currentTensors;
      }
    }

    final runDuration = DateTime.now().difference(runStart);
    _eventNotifier.emit(
      PipelineRunCompleteEvent(
        pipelineId: id,
        runDuration: runDuration,
        stagesProcessed: stagesProcessed,
        finalFormat: currentFormat,
        finalTensorShape: currentTensors != null && currentTensors.isNotEmpty
            ? currentTensors.values.first.shape
            : (currentShapes.values.isNotEmpty
                  ? currentShapes.values.first
                  : const <int>[]),
        stageTimings: stageTimings,
        metadata: {
          'stream_id': streamId,
          'processing_type': currentTensors == null
              ? 'cpu_only'
              : 'mixed_cpu_gpu',
        },
      ),
    );
  }

  /// Download required inputs for a CPU-only stage from current GPU tensors.
  Future<({Map<String, TypedData> host, Map<String, List<int>> shapes})>
  _demoteToHostForStage(
    PipelineStage stage,
    Map<String, Tensor> tensors,
  ) async {
    final host = <String, TypedData>{};
    final shapes = <String, List<int>>{};

    final requiredKeys = stage.inputPorts.isNotEmpty
        ? stage.inputPorts.map((p) => p.name).toList()
        : <String>['input'];

    for (final key in requiredKeys) {
      Tensor? t = tensors[key] ?? tensors['output'];
      t ??= tensors.isNotEmpty ? tensors.values.first : null;
      if (t == null) continue;

      host[key] = await t.getData();
      shapes[key] = List<int>.from(t.shape);
    }

    return (host: host, shapes: shapes);
  }

  /// Promote a map of [PipelineTensor] values to GPU tensors by uploading any
  /// host-dirty buffers. Called when a non-[DynamicStage] requires GPU input
  /// after a run of CPU-only [DynamicStage]s.
  Future<Map<String, Tensor>> _flushPipelineTensors(
    Map<String, PipelineTensor> pt,
  ) async {
    final result = <String, Tensor>{};
    for (final entry in pt.entries) {
      result[entry.key] = await entry.value.asGPU();
    }
    return result;
  }

  /// Execute a [DynamicStage] using the zero-copy [PipelineTensor] path,
  /// emitting the same events as [_executeStage] without converting to GPU.
  Future<Map<String, PipelineTensor>?> _executeDynamicStageWithPT(
    DynamicStage stage,
    Map<String, PipelineTensor> inputs,
    String format,
    String streamId,
    Map<String, Duration> timings,
  ) async {
    try {
      if (!stage.inputPortCollection.isEmpty) {
        inputs = _mapInputs(_inputSources(stage, inputs), inputs);
        _requireValidInputs(
          stage,
          stage.inputPortCollection.validateShapes({
            for (final entry in inputs.entries) entry.key: entry.value.shape,
          }),
        );
      }
      final firstShape = inputs.values.isNotEmpty
          ? inputs.values.first.shape
          : const <int>[];
      // Allocate the event only if someone is listening — these fire per
      // stage per frame.
      if (_eventNotifier.hasListeners(PipelineEventType.stageStart)) {
        _eventNotifier.emit(
          StageStartEvent(
            pipelineId: id,
            stageId: stage.stageId,
            inputFormat: format,
            tensorShape: firstShape,
            metadata: {'stream_id': streamId},
          ),
        );
      }

      final start = DateTime.now();
      final result = await stage.processWithPipelineTensors(inputs);
      final elapsed = DateTime.now().difference(start);

      timings['${stage.stageId}_$streamId'] = elapsed;
      _updateStagePerformance(stage.stageId, elapsed);

      final outShape = result.values.isNotEmpty
          ? result.values.first.shape
          : const <int>[];
      // Allocate the event only if someone is listening — these fire per
      // stage per frame.
      if (_eventNotifier.hasListeners(PipelineEventType.stageComplete)) {
        _eventNotifier.emit(
          StageCompleteEvent(
            pipelineId: id,
            stageId: stage.stageId,
            outputFormat: stage.outputFormat,
            tensorShape: outShape,
            processingTime: elapsed,
            metadata: {'stream_id': streamId},
          ),
        );
      }

      return result;
    } catch (e, stackTrace) {
      _debugLog(() => '❌ Stage ${stage.stageId} execution failed: $e');
      _eventNotifier.emit(
        StageErrorEvent(
          pipelineId: id,
          stageId: stage.stageId,
          error: e.toString(),
          stackTrace: stackTrace,
          metadata: {'stream_id': streamId},
        ),
      );
      return null;
    }
  }

  /// Runs every stage once on [initialInputs] and returns the final
  /// stage's outputs. Inputs set with [setExternalInput] are available too;
  /// [initialInputs] win over them.
  ///
  /// In a pipeline with serialized connections, each stage reads the
  /// outputs its connections name, a port no connection feeds reads the
  /// input of its own name, and the result is the outputs of the last
  /// stage. Without connections, each stage reads the latest tensors by
  /// name, and the result is the outputs of the last stage that produced
  /// any.
  ///
  /// Throws an [ArgumentError] when a stage's inputs do not satisfy its
  /// input ports, and a [StateError] when a connection names an output its
  /// source stage did not produce.
  Future<Map<String, Tensor>> runOnce(Map<String, Tensor> initialInputs) async {
    final stages = _stages;
    if (stages.isEmpty) return {};

    final runStart = DateTime.now();
    final routed = _connections.isNotEmpty;

    // External inputs first, then explicit inputs take precedence
    final runInputs = <String, Tensor>{..._externalInputs, ...initialInputs};
    final available = Map.of(runInputs);
    final stageOutputs = <String, Map<String, Tensor>>{};

    Map<String, Tensor> lastOutputs = {};
    int stagesProcessed = 0;
    String finalFormat = '';

    for (final stage in stages) {
      final inputs = routed
          ? _routeStageInputs(stage, stageOutputs, runInputs)
          : _prepareStageInputs(stage, available);
      _requireValidInputs(stage, stage.inputPortCollection.validateAll(inputs));
      final outputs = await stage.process(inputs);
      stageOutputs[stage.stageId] = outputs;
      if (routed) lastOutputs = outputs;
      if (outputs.isNotEmpty) {
        if (!routed) lastOutputs = outputs;
        available.addAll(outputs);
        stagesProcessed++;
        finalFormat = stage.outputFormat;

        // Record stage outputs for test inspection (no stream context -> merged)
        _stageOutputs['${stage.stageId}_merged'] = outputs;
      }
    }

    if (lastOutputs.isNotEmpty) {
      _stageOutputs['final_merged'] = lastOutputs;
    }

    // Store for router access
    _latestFinalOutputs = lastOutputs.isNotEmpty ? Map.of(lastOutputs) : null;

    // Emit a run-complete event so routers can react to runOnce as well.
    _eventNotifier.emit(
      PipelineRunCompleteEvent(
        pipelineId: id,
        runDuration: DateTime.now().difference(runStart),
        stagesProcessed: stagesProcessed,
        finalFormat: finalFormat,
        finalTensorShape: lastOutputs.isNotEmpty
            ? lastOutputs.values.first.shape
            : const <int>[],
        stageTimings: const <String, Duration>{},
        metadata: const {'processing_type': 'run_once'},
      ),
    );

    return lastOutputs;
  }

  // Allow router/tests to feed inputs externally
  void setExternalInput(String name, Tensor tensor) {
    _externalInputs[name] = tensor;
  }

  void clearExternalInputs() {
    _externalInputs.clear();
  }

  // Expose latest final outputs for router
  Map<String, Tensor>? getLatestFinalOutputs() => _latestFinalOutputs;

  /// Access merged (non-stream) outputs of a stage after runOnce / processing.
  Map<String, Tensor>? getStageOutputMerged(String stageId) {
    return _stageOutputs['${stageId}_merged'];
  }

  /// Connections of the serialized config this pipeline was built from;
  /// empty for a pipeline assembled in code.
  List<ConnectionConfig> get serializedConnections => _connections;

  /// Routes each stage's inputs along [connections], the connections of the
  /// serialized config this pipeline was built from. Each stage must be
  /// added after the stages it reads.
  ///
  /// Throws a [StateError] when the pipeline has a merge stage, since merged
  /// streams cannot be routed.
  void setSerializedConnections(List<ConnectionConfig> connections) {
    if (connections.isNotEmpty && _needsMerging()) {
      throw StateError(_mergeWithConnections);
    }
    final incoming = <String, List<ConnectionConfig>>{};
    for (final connection in connections) {
      (incoming[connection.toStage] ??= []).add(connection);
    }
    _connections = List.unmodifiable(connections);
    _incoming = incoming;
  }

  String get _mergeWithConnections =>
      'Pipeline $id routes its stages along serialized connections, so it '
      'cannot merge streams';

  /// The inputs [stage] reads in a pipeline with serialized connections:
  /// for each connection into it, the output the connection names of a
  /// stage in [stageOutputs], and for each port no connection feeds, the
  /// run input of its name in [runInputs]. A stage without ports or
  /// incoming connections gets all the run inputs.
  ///
  /// Throws a [StateError] when a connection's source stage has not run,
  /// or the output or input it means cannot be told.
  Map<String, Tensor> _routeStageInputs(
    PipelineStage stage,
    Map<String, Map<String, Tensor>> stageOutputs,
    Map<String, Tensor> runInputs,
  ) {
    final incoming = _incoming[stage.stageId] ?? const <ConnectionConfig>[];
    final ports = stage.inputPortCollection.names;
    if (ports.isEmpty && incoming.isEmpty) return Map.of(runInputs);

    final inputs = <String, Tensor>{};
    for (final connection in incoming) {
      final from = connection.fromStage;
      final outputs = stageOutputs[from];
      if (outputs == null) {
        throw StateError(
          'Stage ${stage.stageId} reads stage $from, which has not run '
          'before it',
        );
      }
      if (outputs.isEmpty) {
        throw StateError(
          'Stage ${stage.stageId} reads stage $from, which produced no '
          'outputs',
        );
      }
      final output =
          connection.fromOutput ?? defaultConnectionOutput(outputs.keys);
      if (output == null) {
        throw StateError(
          'The connection from $from to ${stage.stageId} needs a '
          'from_output: $from produced ${outputs.keys.toList()}',
        );
      }
      final tensor = outputs[output];
      if (tensor == null) {
        throw StateError(
          'Stage ${stage.stageId} reads output $output of stage $from, which '
          'produced ${outputs.keys.toList()}',
        );
      }
      final input = connection.toInput ?? defaultConnectionInput(ports);
      if (input == null) {
        throw StateError(
          'The connection from $from to ${stage.stageId} needs a to_input: '
          'the stage has ports $ports',
        );
      }
      inputs[input] = tensor;
    }
    for (final name in ports) {
      if (!inputs.containsKey(name)) {
        final tensor = runInputs[name];
        if (tensor != null) inputs[name] = tensor;
      }
    }
    return inputs;
  }

  /// Runs one frame of [streamId] through the stages along the serialized
  /// connections. A stage is skipped when a stage it reads produced nothing
  /// for this frame, because it was skipped for the stream, failed or
  /// returned no outputs. The frame's outputs are the last stage's.
  Future<void> _processStreamRouted(
    Map<String, Tensor> frame,
    String format,
    String streamId,
  ) async {
    final runInputs = <String, Tensor>{..._externalInputs, ...frame};
    final stageOutputs = <String, Map<String, Tensor>>{};
    final formats = <String, String>{};
    final stageTimings = <String, Duration>{};
    int stagesProcessed = 0;
    final runStart = DateTime.now();

    for (final stage in _stages) {
      if (!_isRunning) break;

      final stageConfig =
          _stageStreamConfigs[stage.stageId] ?? const StageStreamConfig();
      if (!stageConfig.shouldProcessStream(streamId)) continue;

      final incoming = _incoming[stage.stageId] ?? const <ConnectionConfig>[];
      if (incoming.any((c) => !stageOutputs.containsKey(c.fromStage))) {
        _debugLog(
          () =>
              '⏭️ Skipping stage ${stage.stageId} for stream $streamId: a '
              'stage it reads produced nothing',
        );
        continue;
      }

      final Map<String, Tensor> inputs;
      try {
        inputs = _routeStageInputs(stage, stageOutputs, runInputs);
      } catch (e, stackTrace) {
        _eventNotifier.emit(
          StageErrorEvent(
            pipelineId: id,
            stageId: stage.stageId,
            error: e.toString(),
            stackTrace: stackTrace,
            metadata: {'stream_id': streamId},
          ),
        );
        continue;
      }

      final result = await _executeStage(
        stage,
        inputs,
        incoming.isEmpty ? format : formats[incoming.first.fromStage]!,
        streamId,
        stageTimings,
      );
      // Null when the stage failed or returned no outputs.
      if (result == null) continue;

      stageOutputs[stage.stageId] = result.$1;
      formats[stage.stageId] = result.$2;
      stagesProcessed++;
      _stageOutputs['${stage.stageId}_$streamId'] = result.$1;
    }

    final last = _stages.isEmpty ? null : _stages.last.stageId;
    final finalOutputs = stageOutputs[last] ?? const <String, Tensor>{};
    if (finalOutputs.isNotEmpty) {
      _stageOutputs['final_$streamId'] = finalOutputs;
      _lastStreamTensors[streamId] = finalOutputs;
    }

    // Make latest outputs available to the router for link forwarding
    _latestFinalOutputs = finalOutputs.isNotEmpty
        ? Map<String, Tensor>.from(finalOutputs)
        : null;

    _eventNotifier.emit(
      PipelineRunCompleteEvent(
        pipelineId: id,
        runDuration: DateTime.now().difference(runStart),
        stagesProcessed: stagesProcessed,
        finalFormat: formats[last] ?? format,
        finalTensorShape: finalOutputs.isNotEmpty
            ? finalOutputs.values.first.shape
            : const <int>[],
        stageTimings: stageTimings,
        metadata: {'stream_id': streamId, 'processing_type': 'routed'},
      ),
    );
  }

  /// The inputs [stage] reads in a pipeline without connections: a stage
  /// without ports gets one tensor, as `input`; otherwise see
  /// [_inputSources].
  Map<String, Tensor> _prepareStageInputs(
    PipelineStage stage,
    Map<String, Tensor> availableTensors,
  ) {
    // DEBUG: show what we have before mapping
    if (debugEnabled) {
      _debugLog(
        () =>
            '🧩 _prepareStageInputs for ${stage.stageId} | wants: '
            '${stage.inputPorts.map((p) => p.name).join(",")} | available: '
            '${availableTensors.keys.join(",")}',
      );
    }

    // FAST PATH: no declared ports (legacy)
    if (stage.inputPortCollection.isEmpty) {
      final chosen =
          availableTensors['output'] ??
          availableTensors['input'] ??
          (availableTensors.isNotEmpty ? availableTensors.values.first : null);
      return {if (chosen != null) 'input': chosen};
    }

    final inputs = _mapInputs(
      _inputSources(stage, availableTensors),
      availableTensors,
    );

    if (debugEnabled) {
      _debugLog(
        () =>
            '🧩 Final mapped inputs for ${stage.stageId}: '
            '${inputs.entries.map((e) => "${e.key}:${e.value.shape}").join(", ")}',
      );
    }

    return inputs;
  }

  /// Where each input of [stage] comes from in a pipeline without
  /// connections, as input name to key of [available]: a port reads the
  /// key of its own name, else `output`, else `input`, else the first key,
  /// and every other key is passed on under its own name.
  Map<String, String> _inputSources(
    PipelineStage stage,
    Map<String, Object?> available,
  ) {
    if (available.isEmpty) return const {};
    final fallback = available.containsKey('output')
        ? 'output'
        : available.containsKey('input')
        ? 'input'
        : available.keys.first;
    final sources = <String, String>{
      for (final name in stage.inputPortCollection.names)
        name: available.containsKey(name) ? name : fallback,
    };
    for (final key in available.keys) {
      sources.putIfAbsent(key, () => key);
    }
    return sources;
  }

  /// [values] under the input names of [sources].
  static Map<String, T> _mapInputs<T>(
    Map<String, String> sources,
    Map<String, T> values,
  ) => {
    for (final entry in sources.entries)
      if (values[entry.value] case final value?) entry.key: value,
  };

  /// Throws an [ArgumentError] when [check] found inputs [stage] cannot
  /// run on.
  void _requireValidInputs(PipelineStage stage, ValidationResult check) {
    if (!check.isValid) {
      throw ArgumentError(
        'Stage ${stage.stageId} cannot run: ${check.errors.join('; ')}',
      );
    }
  }

  /// Runs [stage] on [tensors], first checking them against its input
  /// ports unless [checkInputs] is false. A failure is reported as a
  /// [StageErrorEvent]; the result is then the stage's recovery, or null.
  Future<(Map<String, Tensor>, String)?> _executeStage(
    PipelineStage stage,
    Map<String, Tensor> tensors,
    String format,
    String streamId,
    Map<String, Duration> timings, {
    bool checkInputs = true,
  }) async {
    try {
      if (checkInputs && !stage.inputPortCollection.isEmpty) {
        _requireValidInputs(
          stage,
          stage.inputPortCollection.validateAll(tensors),
        );
      }
      // Allocate the event only if someone is listening — these fire per
      // stage per frame.
      if (_eventNotifier.hasListeners(PipelineEventType.stageStart)) {
        _eventNotifier.emit(
          StageStartEvent(
            pipelineId: id,
            stageId: stage.stageId,
            inputFormat: format,
            tensorShape: tensors['input']?.shape ?? [],
            metadata: {'stream_id': streamId},
          ),
        );
      }

      final start = DateTime.now();
      final result = await stage.process(tensors);
      final elapsed = DateTime.now().difference(start);

      timings['${stage.stageId}_$streamId'] = elapsed;

      if (result.isEmpty) {
        _debugLog(() => '⚠️ Stage ${stage.stageId} returned empty result');
        return null;
      }

      // Update performance tracking
      _updateStagePerformance(stage.stageId, elapsed);

      // Allocate the event only if someone is listening — these fire per
      // stage per frame.
      if (_eventNotifier.hasListeners(PipelineEventType.stageComplete)) {
        _eventNotifier.emit(
          StageCompleteEvent(
            pipelineId: id,
            stageId: stage.stageId,
            outputFormat: stage.outputFormat,
            tensorShape: result.values.first.shape,
            processingTime: elapsed,
            metadata: {'stream_id': streamId},
          ),
        );
      }

      return (result, stage.outputFormat);
    } catch (e, stackTrace) {
      _debugLog(() => '❌ Stage ${stage.stageId} execution failed: $e');

      _eventNotifier.emit(
        StageErrorEvent(
          pipelineId: id,
          stageId: stage.stageId,
          error: e.toString(),
          stackTrace: stackTrace,
          metadata: {'stream_id': streamId},
        ),
      );

      // Try error recovery
      try {
        final recoveredResult = await stage.handleError(
          e is Exception ? e : Exception(e.toString()),
          tensors,
        );
        if (recoveredResult.isNotEmpty) {
          return (recoveredResult, stage.outputFormat);
        }
      } catch (recoveryError) {
        _debugLog(
          () => '❌ Stage ${stage.stageId} recovery failed: $recoveryError',
        );
      }

      return null;
    }
  }

  /// Update stage performance tracking
  void _updateStagePerformance(String stageId, Duration elapsed) {
    _stageTimings[stageId] = elapsed;
    _stageExecutionCounts[stageId] = (_stageExecutionCounts[stageId] ?? 0) + 1;

    _eventNotifier.emit(
      PipelineBenchmarkEvent(
        pipelineId: id,
        name: stageId,
        duration: elapsed,
        category: 'stage_execution',
        metadata: {'execution_count': _stageExecutionCounts[stageId]},
      ),
    );
  }

  /// Check if merging is needed
  bool _needsMerging() {
    return _stageStreamConfigs.values.any(
      (config) => config.routing == StreamRouting.merged,
    );
  }

  /// Process merged streams
  Future<void> _processMergedStreams() async {
    _debugLog(
      () =>
          '🔀 Processing merged streams: ${_lastStreamTensors.length} available',
    );

    try {
      final mergedTensors = await _createMergedTensors();
      await _processMergeStagesOnly(mergedTensors, 'merged_tensor', 'merged');

      // Snapshots stay — the next merge (on the master's next frame) reuses
      // the latest available frame of every stream, fresh or not.  Only the
      // freshness set resets.
      _mergeFreshSince.clear();
    } catch (e, stackTrace) {
      _debugLog(() => '❌ Stream merging failed: $e $stackTrace');

      // Fallback: process streams individually
      final streamsCopy = Map.of(_lastStreamTensors);
      _mergeFreshSince.clear();

      for (final entry in streamsCopy.entries) {
        try {
          await _processStreamIndividually(
            entry.value,
            'fallback_tensor',
            entry.key,
          );
        } catch (fallbackError) {
          _debugLog(
            () =>
                '❌ Fallback processing failed for stream ${entry.key}: $fallbackError',
          );
        }
      }
    }
  }

  /// Process only merge stages
  Future<void> _processMergeStagesOnly(
    Map<String, Tensor> tensors,
    String format,
    String streamId,
  ) async {
    String currentFormat = format;
    Map<String, Tensor> currentTensors = tensors;
    final stageTimings = <String, Duration>{};
    int stagesProcessed = 0;
    final runStart = DateTime.now();
    bool foundFirstMergeStage = false;

    _debugLog(() => '🔀 Processing from first merge stage onwards');

    for (final stage in _stages) {
      if (!_isRunning) break;

      final stageConfig =
          _stageStreamConfigs[stage.stageId] ?? const StageStreamConfig();

      // Skip stages until we find the first merge stage
      if (!foundFirstMergeStage) {
        if (stageConfig.routing == StreamRouting.merged) {
          foundFirstMergeStage = true;
          _debugLog(
            () => '🔀 Starting merged processing from stage ${stage.stageId}',
          );
        } else {
          continue; // Skip non-merge stages before the first merge stage
        }
      }

      // Once we've found the first merge stage, process ALL remaining stages (merge or not)
      _debugLog(() => '⚙️ Executing post-merge stage ${stage.stageId}');

      // Merged tensors are keyed by stream, not by port.
      final result = await _executeStage(
        stage,
        currentTensors,
        currentFormat,
        streamId,
        stageTimings,
        checkInputs: false,
      );

      if (result != null) {
        currentTensors = result.$1;
        currentFormat = result.$2;
        stagesProcessed++;

        _stageOutputs['${stage.stageId}_$streamId'] = currentTensors;
      }
    }

    // Store final merged output
    if (currentTensors.isNotEmpty) {
      _stageOutputs['final_$streamId'] = currentTensors;
      _debugLog(() => '💾 Stored final merged output');
    }

    // Publish latest outputs so router can forward merged results as well
    _latestFinalOutputs = currentTensors.isNotEmpty
        ? Map<String, Tensor>.from(currentTensors)
        : null;

    final runDuration = DateTime.now().difference(runStart);
    _eventNotifier.emit(
      PipelineRunCompleteEvent(
        pipelineId: id,
        runDuration: runDuration,
        stagesProcessed: stagesProcessed,
        finalFormat: currentFormat,
        finalTensorShape: currentTensors['input']?.shape ?? [],
        stageTimings: stageTimings,
        metadata: {'stream_id': streamId, 'processing_type': 'merged'},
      ),
    );

    _debugLog(
      () => '🎯 Completed merged processing through $stagesProcessed stages',
    );
  }

  /// Create merged tensors from all streams
  Future<Map<String, Tensor>> _createMergedTensors() async {
    final mergedTensors = <String, Tensor>{};

    _debugLog(
      () =>
          '🔗 Creating merged tensors from ${_lastStreamTensors.length} streams',
    );

    // Exactly ONE tensor per stream, under a 'stream_'-prefixed key (the
    // merge stage collects by that prefix).  Per-stream results carry their
    // payload under 'output' (builder stages) or 'input' (passthroughs) —
    // the old code only looked at 'input', so post-processing outputs never
    // made it into the merge set, and its extra '${id}_${key}' entries could
    // double-count streams whose ids already start with 'stream_'.
    //
    // Staleness: snapshots persist across merges (master-clock cadence), so
    // a stream that stopped delivering — device stopped, or a silence-gated
    // loopback source — must age out rather than overlay its last spectrum
    // forever.  A snapshot is dropped once it is older than ~4× the stream's
    // own observed frame interval.
    final now = DateTime.now();
    for (final entry in _lastStreamTensors.entries) {
      if (!_mergeFreshSince.contains(entry.key)) {
        final at = _mergeSnapshotAt[entry.key];
        final cadence = _mergeCadence[entry.key];
        Duration maxAge = cadence == null
            ? _mergeStragglerTimeout
            : cadence * 4;
        if (maxAge < _mergeStragglerTimeout) maxAge = _mergeStragglerTimeout;
        if (at != null && now.difference(at) > maxAge) {
          _debugLog(
            () => '   💤 Skipping stale stream_${entry.key} '
                '(${now.difference(at).inMilliseconds} ms old)',
          );
          continue;
        }
      }
      final primary = _primaryTensorOf(entry.value);
      if (primary != null && _isValidTensor(primary)) {
        mergedTensors['stream_${entry.key}'] = primary;
        _debugLog(() => '   📥 Added stream_${entry.key}: ${primary.shape}');
      }
    }

    // Set a primary input tensor (use first available)
    if (mergedTensors.isNotEmpty) {
      final firstTensor = mergedTensors.values.first;
      mergedTensors['input'] = firstTensor;
      _debugLog(() => '   🎯 Set primary input: ${firstTensor.shape}');
    }

    if (mergedTensors.isEmpty) {
      throw Exception('No tensors available for merging');
    }

    return mergedTensors;
  }

  /// Check if tensor is valid
  bool _isValidTensor(Tensor tensor) {
    return tensor.buffer.isValid &&
        tensor.shape.isNotEmpty &&
        tensor.shape.every((dim) => dim > 0);
  }

  /// Stop pipeline
  Future<void> stop() async {
    if (!_isRunning) return;
    _debugLog(() => '🛑 Stopping pipeline...');
    _isShuttingDown = true;

    // Cancel pending batch flush timer
    _batchFlushTimer?.cancel();
    _batchFlushTimer = null;

    try {
      // Stop all streams first
      final streamStopFutures = <Future<void>>[];
      for (final stream in List.of(_streams)) {
        // Create copy to avoid modification during iteration
        if (stream.isActive) {
          streamStopFutures.add(_stopStreamSafely(stream));
        }
      }

      // Wait for all streams to stop
      if (streamStopFutures.isNotEmpty) {
        await Future.wait(streamStopFutures, eagerError: false);
      }

      // Wait for current processing to complete with timeout
      await _waitForProcessingToComplete();

      // Wait for per-stream ingestion chains to complete
      await _waitForIngestToComplete();

      // Clear processing queue
      _processingQueue.clear();
      _activeProcessingItems.clear();

      // Destroy ingest pools
      for (final list in _ingestPools.values) {
        for (final t in list) {
          try {
            if (t.buffer.isValid) t.destroy();
          } catch (_) {}
        }
      }
      _ingestPools.clear();
      _ingestInUse.clear();
      _ingestOwner.clear();

      // Drop the asset references start() took
      await _releaseAssets();

      // Mark as stopped
      _isRunning = false;

      final runtime = _startTime != null
          ? DateTime.now().difference(_startTime!)
          : Duration.zero;

      _eventNotifier.emit(
        PipelineStopEvent(pipelineId: id, totalRuntime: runtime),
      );

      _debugLog(() => '✅ Pipeline stopped');
    } catch (e, stackTrace) {
      _debugLog(() => '❌ Error during pipeline stop: $e');
      _eventNotifier.emit(
        PipelineErrorEvent(
          pipelineId: id,
          error: e.toString(),
          stackTrace: stackTrace,
        ),
      );
    } finally {
      _isShuttingDown = false;
    }
  }

  /// Safely stop a single stream
  Future<void> _stopStreamSafely(MediaStream stream) async {
    try {
      await stream.stop();
      _eventNotifier.emit(
        StreamStopEvent(
          pipelineId: id,
          streamId: stream.id,
          streamType: stream.type.toString(),
        ),
      );
      _debugLog(() => '✅ Stream ${stream.id} stopped');
    } catch (e) {
      _debugLog(() => '❌ Error stopping stream ${stream.id}: $e');
      // Don't rethrow - continue with other streams
    }
  }

  /// Wait for processing to complete with timeout
  Future<void> _waitForProcessingToComplete() async {
    const maxWaitTime = Duration(seconds: 5);
    const checkInterval = Duration(milliseconds: 10);
    final stopwatch = Stopwatch()..start();

    while (_isProcessing && stopwatch.elapsed < maxWaitTime) {
      await Future.delayed(checkInterval);
    }

    if (_isProcessing) {
      _debugLog(
        () => '⚠️ Processing did not complete within timeout, forcing stop',
      );
      _isProcessing = false;
    }

    stopwatch.stop();
  }

  Future<Tensor> _acquireIngestTensor(
    String streamId,
    List<int> shape, [
    BufferDataType dataType = BufferDataType.float32,
  ]) async {
    final pool = _ingestPools.putIfAbsent(streamId, () => <Tensor>[]);
    // Find a free tensor with matching shape and element type
    for (final t in pool) {
      if (!_ingestInUse.contains(t) &&
          t.dataType == dataType &&
          _shapesMatch(t.shape, shape)) {
        _ingestInUse.add(t);
        _ingestOwner[t] = streamId;
        return t;
      }
    }
    // None free -> allocate new.
    //
    // A rapidly growing pool means acquired tensors are never being released
    // (see _releaseIngestTensorsForItem / _releaseIngestTensorsForStream).
    // Warn loudly so the bug is caught before it becomes a VRAM leak in
    // production. The threshold is generous — steady-state is 1-2 tensors per
    // stream; >10 is a strong signal of a missing release somewhere.
    const int _kIngestPoolLeakThreshold = 10;
    if (pool.length >= _kIngestPoolLeakThreshold) {
      assert(
        false,
        'Pipeline ingest pool for stream "$streamId" has ${pool.length} tensors '
        'in use and is about to allocate another. '
        'This is a strong sign that ingest tensors are not being released after '
        'processing. Check that every code path through _processItem calls '
        '_releaseIngestTensorsForItem or _releaseIngestTensorsForStream.',
      );
      // In release mode print a warning instead of crashing.
      _debugLog(
        () =>
            '⚠️ LEAK WARNING: ingest pool for "$streamId" has ${pool.length} '
            'tensors — possible missing _releaseIngestTensors call.',
      );
    }
    final Tensor t = await _zeros(shape, dataType);
    pool.add(t);
    _ingestInUse.add(t);
    _ingestOwner[t] = streamId;
    return t;
  }

  void _releaseIngestTensorsForItem(
    String streamId,
    Map<String, Tensor> tensors,
  ) {
    for (final t in tensors.values) {
      final owner = _ingestOwner[t];
      if (owner == streamId) {
        _ingestInUse.remove(t);
        // keep in pool for reuse
      }
    }
  }

  /// Release every in-use ingest tensor currently owned by [streamId].
  ///
  /// Used by the host path which uploads tensors via
  /// [_uploadHostData] (calling [_acquireIngestTensor]) but does not
  /// know which specific tensors were used by the stages it ran. Per-stream
  /// ingestion serialization (see [_ingestSerializers]) guarantees no other
  /// host frame for the same stream is active, so all in-use tensors with
  /// this owner must belong to the just-completed frame.
  void _releaseIngestTensorsForStream(String streamId) {
    final pool = _ingestPools[streamId];
    if (pool == null) return;
    for (final t in pool) {
      if (_ingestOwner[t] == streamId && _ingestInUse.contains(t)) {
        _ingestInUse.remove(t);
      }
    }
  }

  /// Wait for per-stream ingest chains to complete
  Future<void> _waitForIngestToComplete() async {
    if (_ingestSerializers.isEmpty) return;

    const maxWaitTime = Duration(seconds: 5);
    final tasks = _ingestSerializers.values.toList();
    try {
      await Future.any([
        Future.wait(tasks, eagerError: false),
        Future.delayed(maxWaitTime),
      ]);
    } catch (_) {
      // ignore errors during shutdown
    } finally {
      _ingestSerializers.clear();
    }
  }

  /// Get stage output
  Map<String, Tensor>? getStageOutput(String stageId, [String? streamId]) {
    if (stageId == 'final' && streamId != null) {
      return _stageOutputs['final_$streamId'];
    }

    if (streamId != null) {
      return _stageOutputs['${stageId}_$streamId'];
    }

    // Try merged output first
    final mergedOutput = _stageOutputs['${stageId}_merged'];
    if (mergedOutput != null) return mergedOutput;

    // Try single stream inference
    if (_streams.length == 1) {
      final singleStreamOutput =
          _stageOutputs['${stageId}_${_streams.first.id}'];
      if (singleStreamOutput != null) return singleStreamOutput;
    }

    return null;
  }

  /// Get all outputs for a stage across streams
  Map<String, Map<String, Tensor>> getStageOutputsForAllStreams(
    String stageId,
  ) {
    final outputs = <String, Map<String, Tensor>>{};
    for (final entry in _stageOutputs.entries) {
      if (entry.key.startsWith('${stageId}_') &&
          entry.key != '${stageId}_merged') {
        final streamId = entry.key.substring('${stageId}_'.length);
        outputs[streamId] = entry.value;
      }
    }
    return outputs;
  }

  /// Check if stage has output
  bool hasStageOutput(String stageId) {
    return _stageOutputs.keys.any((key) => key.startsWith('${stageId}_'));
  }

  /// Get stage by ID
  PipelineStage? getStage(String stageId) {
    for (final s in _stages) {
      if (s.stageId == stageId) return s;
    }
    return null;
  }

  /// Get benchmark statistics
  Map<String, dynamic> getBenchmarkStats() {
    final stats = _eventNotifier.getBenchmarkStats();

    if (!stats.containsKey('event_counts')) {
      stats['event_counts'] = <String, int>{};
    }

    stats['pipeline_stats'] = {
      'stages': _stages.length,
      'streams': _streams.length,
      'outputs': _stageOutputs.length,
      'is_running': _isRunning,
      'is_processing': _isProcessing,
      'queue_size': _processingQueue.length,
    };

    return stats;
  }

  /// Debug print all outputs
  Future<void> debugPrintAllOutputs({bool getOutputs = false}) async {
    print('🔍 Debugging all stage outputs:');
    for (final entry in _stageOutputs.entries) {
      final key = entry.key;
      final tensors = entry.value;

      print('   Key: $key');
      for (final tensorEntry in tensors.entries) {
        print('      ${tensorEntry.key}: ${tensorEntry.value.shape}');
        if (getOutputs) {
          final Object data = await tensorEntry.value.getData();
          if (data is List && data.isNotEmpty) {
            print('         Data: ${data.take(25).toString()}');
          } else {
            print('         Data: Invalid tensor buffer');
          }
        }
      }
    }
  }

  Future<void>? _disposing;

  /// Stops the pipeline and releases its stages and GPU resources.
  ///
  /// Calling it again returns the same future. The asset manager is not
  /// disposed.
  Future<void> dispose() => _disposing ??= _dispose();

  Future<void> _dispose() async {
    _debugLog(() => '🗑️ Disposing pipeline...');

    // Stop pipeline first (waits for processing and ingestion chains)
    await stop();

    // Wait a bit for any pending operations
    await Future.delayed(const Duration(milliseconds: 100));

    try {
      // Dispose stages in reverse order
      for (final stage in _stages.reversed) {
        try {
          await stage.dispose();
        } catch (e) {
          _debugLog(() => '❌ Error disposing stage ${stage.stageId}: $e');
        }
      }

      // Dispose subsystems
      try {
        _resourceCache.dispose();
      } catch (e) {
        _debugLog(() => '❌ Error disposing resource cache: $e');
      }

      // The asset manager is not disposed here: it may be one the caller
      // passed in, or one PatchExecutor reuses across a rebuild. It holds
      // only references, so when nothing else uses it, it is collected
      // with the pipeline.

      try {
        _resourceGovernor?.dispose();
      } catch (e) {
        _debugLog(() => '❌ Error disposing resource governor: $e');
      }

      try {
        _eventNotifier.dispose();
      } catch (e) {
        _debugLog(() => '❌ Error disposing event notifier: $e');
      }

      // Merge snapshot resources (pipeline-owned copies + copy shader)
      for (final snap in _mergeSnapshots.values) {
        try {
          snap.destroy();
        } catch (_) {}
      }
      _mergeSnapshots.clear();
      _mergeCopyShader?.destroy();
      _mergeCopyShader = null;
      _mergeSnapshotAt.clear();
      _mergeCadence.clear();
      _mergeFreshSince.clear();
      _mergeMasterId = null;

      // Clear all state
      _stageOutputs.clear();
      _stageStreamConfigs.clear();
      _lastStreamTensors.clear();
      _stageTimings.clear();
      _stageExecutionCounts.clear();
      _stages.clear();
      _streams.clear();
      _processingQueue.clear();
      _activeProcessingItems.clear();

      _debugLog(() => '✅ Pipeline disposed');
    } catch (e) {
      _debugLog(() => '❌ Error during disposal: $e');
      // Continue with disposal even if there are errors
    }
  }

  // Utilities

  String _getStreamFormat(MediaStreamType type) {
    switch (type) {
      case MediaStreamType.audio:
        return 'raw_audio';
      case MediaStreamType.video:
        return 'raw_video';
      case MediaStreamType.screen:
        return 'raw_screen';
    }
  }

  bool _shapesMatch(List<int> a, List<int> b) {
    if (identical(a, b)) return true;
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  static int _elementCountOf(TypedData data) =>
      data.lengthInBytes ~/ data.elementSizeInBytes;

  List<int> _fixShape(List<int> shape, int elementCount) {
    final expectedLen = shape.fold<int>(1, (a, b) => a * b);
    if (expectedLen != elementCount) {
      if (debugEnabled) {
        _debugLog(
          () => '⚠️ Adjusting host shape ${shape.join("x")} to [$elementCount]',
        );
      }
      return [elementCount];
    }
    return shape;
  }
}
