import 'dart:async';
import 'dart:typed_data';
import 'package:gpu_pipeline/src/operation/cpu_stage_operation.dart';
import 'package:gpu_pipeline/src/operation/operation.dart';
import 'package:gpu_pipeline/src/stages/pipeline_resource_cache.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:gpu_pipeline/src/pipeline_properties.dart';
import 'pipeline_event.dart';
import 'pipeline_stage.dart';
import 'serialization/argument_resolution.dart';

/// Unified tensor that can live on CPU and/or GPU.
/// Seamless CPU↔GPU switching: a copy is refreshed only when the next
/// operation needs that side and the other side was written since.
class PipelineTensor {
  Tensor? _gpu;
  TypedData? _host;
  List<int> shape;

  /// Element dtype of the host (and GPU) buffer. Defaults to float32.
  /// Tracked so that non-float byte streams (e.g. JPEG payloads emitted by
  /// codec CPU stages) round-trip without lossy conversion.
  BufferDataType _dtype;
  BufferDataType get dtype => _dtype;

  // At most one side is stale; the other holds the newest contents.
  bool _hostStale; // GPU written since the host copy: asCPU reads back
  bool _gpuStale; // host written since the GPU copy: asGPU uploads

  TensorLocation get tensorLocation =>
      _gpu != null && !_gpuStale ? TensorLocation.gpu : TensorLocation.cpu;

  PipelineTensor._(
    this._gpu,
    this._host,
    this.shape,
    this._dtype,
    this._hostStale,
    this._gpuStale,
  );

  factory PipelineTensor.gpu(Tensor t, List<int> shape) => PipelineTensor._(
    t,
    null,
    List<int>.from(shape),
    t.dataType,
    false,
    false,
  );

  factory PipelineTensor.cpu(TypedData data, List<int> shape) =>
      PipelineTensor._(
        null,
        data,
        List<int>.from(shape),
        _inferDtype(data),
        false,
        true,
      );

  bool get hasGpu => _gpu != null;
  bool get hasHost => _host != null;

  void setHostData(TypedData data, {List<int>? newShape}) {
    _host = data;
    _dtype = _inferDtype(data);
    if (newShape != null) {
      shape = List<int>.from(newShape);
    }
    _hostStale = false;
    _gpuStale = true;
  }

  /// Ensure GPU tensor is up to date and return it.
  Future<Tensor> asGPU() async {
    // If we don't have a GPU tensor or the shape/dtype changed, reallocate
    final needsAlloc =
        _gpu == null ||
        !_shapesEqual(_gpu!.shape, shape) ||
        _gpu!.dataType != _dtype;
    if (needsAlloc) {
      // Release old if needed
      try {
        if (_gpu != null && _gpu!.buffer.isValid) {
          _gpu!.destroy();
        }
      } catch (_) {}
      _gpu = await _allocZeros(shape, _dtype);
      _gpuStale = true;
    }

    // Upload from host if host is newer or we just (re)allocated
    if ((_host != null && _gpuStale) || needsAlloc) {
      await _writeHost(_gpu!, _host!);
      _gpuStale = false;
      _hostStale = false;
    }

    return _gpu!;
  }

  /// Ensure host data is up to date and return it as a [TypedData] view.
  /// The runtime type of the returned data corresponds to [dtype].
  Future<TypedData> asCPU() async {
    if (_host == null) {
      if (_gpu != null) {
        final td = await _gpu!.getData();
        _host = td;
        _dtype = _gpu!.dataType;
        // Preserve the GPU tensor's shape (don't flatten)
        shape = List<int>.from(_gpu!.shape);
        _gpuStale = false;
        _hostStale = false;
      } else {
        // Allocate fresh host buffer matching current shape & dtype
        final len = shape.fold<int>(1, (a, b) => a * b);
        _host = _allocHost(len, _dtype);
        _gpuStale = true;
      }
    } else if (_hostStale && _gpu != null) {
      final td = await _gpu!.getData();
      _host = td;
      _dtype = _gpu!.dataType;
      shape = List<int>.from(_gpu!.shape);
      _gpuStale = false;
      _hostStale = false;
    }
    return _host!;
  }

  /// Backwards-compatible: get host data as Float32List, converting if needed.
  /// Use [asCPU] to obtain the native dtype.
  Future<Float32List> asCPUFloat32() async {
    final td = await asCPU();
    if (td is Float32List) return td;
    return _toFloat32(td);
  }

  bool _shapesEqual(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  /// Adopt a GPU tensor (marks host stale).
  void setGpuTensor(Tensor tensor, {List<int>? newShape}) {
    _gpu = tensor;
    _dtype = tensor.dataType;
    shape =
        newShape ?? List<int>.from(tensor.shape); // Use tensor's actual shape
    _gpuStale = false;
    _hostStale = true; // host view (if any) stale until re-fetched
  }
}

/// Infer the [BufferDataType] tag for a [TypedData] runtime instance.
BufferDataType _inferDtype(TypedData d) {
  if (d is Float32List) return BufferDataType.float32;
  if (d is Float64List) return BufferDataType.float64;
  if (d is Int8List) return BufferDataType.int8;
  if (d is Uint8List) return BufferDataType.uint8;
  if (d is Int16List) return BufferDataType.int16;
  if (d is Uint16List) return BufferDataType.uint16;
  if (d is Int32List) return BufferDataType.int32;
  if (d is Uint32List) return BufferDataType.uint32;
  if (d is Int64List) return BufferDataType.int64;
  if (d is Uint64List) return BufferDataType.uint64;
  return BufferDataType.float32;
}

/// Allocate a host buffer of the right runtime type for [dtype].
TypedData _allocHost(int len, BufferDataType dtype) {
  switch (dtype) {
    case BufferDataType.float32:
      return Float32List(len);
    case BufferDataType.float64:
      return Float64List(len);
    case BufferDataType.int8:
      return Int8List(len);
    case BufferDataType.uint8:
      return Uint8List(len);
    case BufferDataType.int16:
      return Int16List(len);
    case BufferDataType.uint16:
      return Uint16List(len);
    case BufferDataType.int32:
      return Int32List(len);
    case BufferDataType.uint32:
      return Uint32List(len);
    case BufferDataType.int64:
      return Int64List(len);
    case BufferDataType.uint64:
      return Uint64List(len);
    case BufferDataType.float16:
      // No native Float16List in dart:typed_data; back with raw bytes.
      return Uint16List(len);
  }
}

Future<Tensor> _allocZeros(List<int> shape, BufferDataType dtype) async {
  switch (dtype) {
    case BufferDataType.float32:
      return await Tensor.zeros<Float32List>(shape, dataType: dtype);
    case BufferDataType.float64:
      return await Tensor.zeros<Float64List>(shape, dataType: dtype);
    case BufferDataType.int8:
      return await Tensor.zeros<Int8List>(shape, dataType: dtype);
    case BufferDataType.uint8:
      return await Tensor.zeros<Uint8List>(shape, dataType: dtype);
    case BufferDataType.int16:
      return await Tensor.zeros<Int16List>(shape, dataType: dtype);
    case BufferDataType.uint16:
      return await Tensor.zeros<Uint16List>(shape, dataType: dtype);
    case BufferDataType.int32:
      return await Tensor.zeros<Int32List>(shape, dataType: dtype);
    case BufferDataType.uint32:
      return await Tensor.zeros<Uint32List>(shape, dataType: dtype);
    case BufferDataType.int64:
      return await Tensor.zeros<Int64List>(shape, dataType: dtype);
    case BufferDataType.uint64:
      return await Tensor.zeros<Uint64List>(shape, dataType: dtype);
    case BufferDataType.float16:
      return await Tensor.zeros<Uint16List>(shape, dataType: dtype);
  }
}

Future<void> _writeHost(Tensor t, TypedData host) async {
  // Called through the base type: the tensor may be one this class allocated
  // (a Tensor<Float32List> from _allocZeros) or one it was handed, which can
  // be a Tensor<TypedData>. Casting to the dtype's list type rejected the
  // latter. asGPU reallocates on a dtype change, so host matches the buffer.
  await t.write(host);
}

Float32List _toFloat32(TypedData td) {
  if (td is Float32List) return td;
  final out = Float32List(_typedLen(td));
  if (td is Float64List) {
    for (var i = 0; i < out.length; i++) {
      out[i] = td[i];
    }
  } else if (td is Int32List) {
    for (var i = 0; i < out.length; i++) {
      out[i] = td[i].toDouble();
    }
  } else if (td is Uint32List) {
    for (var i = 0; i < out.length; i++) {
      out[i] = td[i].toDouble();
    }
  } else if (td is Int16List) {
    for (var i = 0; i < out.length; i++) {
      out[i] = td[i].toDouble();
    }
  } else if (td is Uint16List) {
    for (var i = 0; i < out.length; i++) {
      out[i] = td[i].toDouble();
    }
  } else if (td is Int8List) {
    for (var i = 0; i < out.length; i++) {
      out[i] = td[i].toDouble();
    }
  } else if (td is Uint8List) {
    for (var i = 0; i < out.length; i++) {
      out[i] = td[i].toDouble();
    }
  }
  return out;
}

int _typedLen(TypedData d) {
  if (d is Float32List) return d.length;
  if (d is Float64List) return d.length;
  if (d is Int8List) return d.length;
  if (d is Uint8List) return d.length;
  if (d is Int16List) return d.length;
  if (d is Uint16List) return d.length;
  if (d is Int32List) return d.length;
  if (d is Uint32List) return d.length;
  if (d is Int64List) return d.length;
  if (d is Uint64List) return d.length;
  return d.lengthInBytes;
}

enum TensorLocation { gpu, cpu }

/// Enhanced dynamic stage with I/O validation, parameters, and asset management
class DynamicStage extends PipelineStage {
  final List<StageOperation> operations;
  final _argResolver = const ArgumentResolver();

  ResourceCache? _resourceCache;

  DynamicStage({
    required super.stageId,
    required this.operations,
    required Map<String, Parameter> parameterDefinitions,
    super.isPersistent = false,
  }) {
    TensorOperationRegistry.ensureInitialized();
    CPUOperationRegistry.registerBuiltInProcessors();

    // Initialize parameters with default values
    for (final param in parameterDefinitions.values) {
      addParameter(param);
      // Set the default value in the current values map
      setPropertyValue(param.name, param.defaultValue);
    }
  }

  /// Set the unified resource cache for this stage
  void setResourceCache(ResourceCache cache) {
    _resourceCache = cache;
  }

  /// Legacy method for backward compatibility
  @Deprecated('Use setResourceCache instead')
  void setPipelineResourceCache(PipelineResourceCache cache) {
    _resourceCache = cache;
  }

  /// Current parameter name -> value snapshot
  Map<String, dynamic> currentParameterValues() {
    final map = <String, dynamic>{};
    for (final p in parameterDefinitions.values) {
      map[p.name] = getPropertyValue(p.name);
    }
    return map;
  }

  /// Obtain (or lazily create) a resource cache for an operation
  ResourceCache resourceCacheFor(StageOperation op) {
    _resourceCache ??= ResourceCache(cacheId: 'stage_$stageId');
    return _resourceCache!;
  }

  @override
  Future<Map<String, Tensor>> process(
    Map<String, Tensor> inputs, [
    Map<String, Tensor>? preAllocatedOutputs,
  ]) async {
    var currentData = <String, PipelineTensor>{};
    for (final entry in inputs.entries) {
      currentData[entry.key] = PipelineTensor.gpu(entry.value, entry.value.shape);
    }
    currentData = await _runOperations(currentData);
    return await _convertToGPUTensors(currentData);
  }

  /// Execute stage operations starting from pre-existing [PipelineTensor] inputs,
  /// returning ONLY the selected output tensors (same selection as
  /// [_convertToGPUTensors]) without forcing a GPU promotion at the end.
  ///
  /// Use this when chaining consecutive [DynamicStage]s so that CPU↔GPU transfers
  /// are deferred until a stage that actually needs GPU data is reached.
  ///
  /// IMPORTANT: only the declared output-port tensors are returned. Intermediate
  /// and versioned tensors (e.g. `temp_*`, `*_v<ts>`, `derived_input_*`) are
  /// dropped here just as the GPU-Tensor path drops them in
  /// [_convertToGPUTensors]. Returning the full working map would let those
  /// intermediates accumulate across threaded stages and be force-uploaded to
  /// GPU on flush, leaking VRAM frame-over-frame.
  Future<Map<String, PipelineTensor>> processWithPipelineTensors(
    Map<String, PipelineTensor> inputs,
  ) async {
    final worked = await _runOperations(
      Map<String, PipelineTensor>.from(inputs),
    );
    return _selectOutputPipelineTensors(worked);
  }

  /// Select only the declared output-port tensors from a working set, mirroring
  /// the key-selection logic of [_convertToGPUTensors] but WITHOUT promoting to
  /// GPU. Intermediate tensors are simply not referenced by the returned map so
  /// they become unreachable (matching pre-threading reachability semantics).
  Map<String, PipelineTensor> _selectOutputPipelineTensors(
    Map<String, PipelineTensor> pipelineTensors,
  ) {
    final selected = <String, PipelineTensor>{};

    if (outputPorts.isEmpty) {
      final finalTensor = _findFinalOutputTensor(pipelineTensors);
      if (finalTensor != null) {
        selected['output'] = finalTensor;
      }
    } else {
      for (final port in outputPorts) {
        final pt = _findBestOutputMatch(port.name, pipelineTensors);
        if (pt != null) {
          selected[port.name] = pt;
        }
      }
    }

    return selected;
  }

  /// Run all operations in order, threading [PipelineTensor] state.
  /// Does NOT promote results to GPU — callers handle the final conversion.
  Future<Map<String, PipelineTensor>> _runOperations(
    Map<String, PipelineTensor> currentData,
  ) async {
    for (final operation in operations) {
      // Resolve argument references (param:/tensor:/shape:/size:)
      final tensorsForRefs = <String, Tensor>{};
      for (final e in currentData.entries) {
        if (e.value.tensorLocation == TensorLocation.gpu &&
            e.value._gpu != null) {
          tensorsForRefs[e.key] = e.value._gpu!;
        }
      }
      final params = currentParameterValues();
      final resolved = _argResolver.resolve(
        operation.arguments,
        params: params,
        tensors: tensorsForRefs,
      );
      operation.applyResolvedArguments(resolved.positional, resolved.named);

      if (operation is CPUStageOperation) {
        currentData = await _processCPUOperation(operation, currentData);
      } else {
        currentData = await _processGPUOperation(operation, currentData);
      }
    }
    return currentData;
  }

  /// Process CPU operation without forcing a GPU round-trip:
  /// - Pull host data from PipelineTensor (or download once if only GPU exists)
  /// - Run the CPU processor through the operation's executor
  /// - Store outputs as host buffers (Float32List) in PipelineTensor
  /// - Only promote to GPU when a later GPU op needs it
  Future<Map<String, PipelineTensor>> _processCPUOperation(
    CPUStageOperation operation,
    Map<String, PipelineTensor> currentData,
  ) async {
    // Prepare host inputs for the CPU processor
    final hostInputs = <String, TypedData>{};
    final ranks = <String, List<int>>{};

    // When inputKeys is empty, fall back to all accumulated tensors (mirrors
    // the behaviour of CPUStageOperation.executeWithCache's activeInputKeys).
    final activeInputKeys = operation.inputKeys.isEmpty
        ? currentData.keys.toList()
        : operation.inputKeys;
    for (final key in activeInputKeys) {
      final pt = _findBestInputTensor(key, currentData);
      if (pt == null) {
        throw Exception('CPU operation input "$key" not found');
      }
      final host = await pt.asCPU();
      hostInputs[key] = host;
      // Preserve original shape from the pipeline tensor
      ranks[key] = List<int>.from(pt.shape);
    }

    // Gather stage parameters
    final parameters = currentParameterValues();

    // Run through the operation's executor: inline by default, or the one
    // the operation was built with (a worker isolate, say).
    final rawOutputs = await operation.executor.run(
      hostInputs,
      ranks,
      parameters,
    );

    // Integrate outputs into PipelineTensor as HOST data with preserved/inferred shape
    for (final entry in rawOutputs.entries) {
      final outKey = entry.key;

      // Skip shape sidecar keys like '<key>_shape' or '<key>__shape'
      if (_isShapeSidecarKey(outKey)) continue;

      final td = entry.value;
      // Preserve the original TypedData runtime type (e.g. Uint8List for codec
      // byte streams) so it round-trips through the GPU buffer with the right
      // BufferDataType instead of being lossily coerced to Float32.
      final outShape = _resolveCpuOutShape(outKey, td, ranks, rawOutputs);

      final existing = currentData[outKey];
      if (existing != null) {
        // If 'input' would change shape, redirect safely
        if (outKey == 'input') {
          final sameShape = _shapesEqual(existing.shape, outShape);
          if (!sameShape) {
            final redirectedKey = 'derived_input_${outShape.join("x")}';
            final redirected = currentData[redirectedKey];
            if (redirected != null) {
              redirected.setHostData(td, newShape: outShape);
            } else {
              currentData[redirectedKey] = PipelineTensor.cpu(td, outShape);
            }
            continue;
          }
        }
        existing.setHostData(td, newShape: outShape);
      } else {
        currentData[outKey] = PipelineTensor.cpu(td, outShape);
      }
    }

    return currentData;
  }

  // Prefer explicit sidecar shape; else preserve any input rank that matches element count; else 1D
  List<int> _resolveCpuOutShape(
    String outKey,
    TypedData data,
    Map<String, List<int>> inputRanks,
    Map<String, TypedData> rawOutputs,
  ) {
    final sidecar1 = '${outKey}_shape';
    final sidecar2 = '${outKey}__shape';
    if (rawOutputs.containsKey(sidecar1)) {
      final ints = _typedDataToIntList(rawOutputs[sidecar1]!);
      if (ints.isNotEmpty) return ints;
    }
    if (rawOutputs.containsKey(sidecar2)) {
      final ints = _typedDataToIntList(rawOutputs[sidecar2]!);
      if (ints.isNotEmpty) return ints;
    }

    final len = _typedLen(data);
    // Prefer 'input' then 'output' ranks if element count matches
    for (final k in const ['input', 'output']) {
      final s = inputRanks[k];
      if (s != null && _prod(s) == len) return List<int>.from(s);
    }
    // Any other input with matching element count
    for (final s in inputRanks.values) {
      if (_prod(s) == len) return List<int>.from(s);
    }
    return [len];
  }

  bool _isShapeSidecarKey(String key) =>
      key.endsWith('_shape') || key.endsWith('__shape');

  List<int> _typedDataToIntList(TypedData td) {
    if (td is Int32List) return td.toList();
    if (td is Uint32List) return td.toList();
    if (td is Int16List) return td.map((e) => e.toInt()).toList();
    if (td is Uint16List) return td.map((e) => e.toInt()).toList();
    if (td is Int8List) return td.map((e) => e.toInt()).toList();
    if (td is Uint8List) return td.map((e) => e.toInt()).toList();
    if (td is Float32List) return td.map((e) => e.toInt()).toList();
    if (td is Float64List) return td.map((e) => e.toInt()).toList();
    return const <int>[];
  }

  int _prod(List<int> s) => s.fold<int>(1, (a, b) => a * b);

  bool _shapesEqual(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  /// Process GPU operation with lazy promotion from CPU where needed.
  Future<Map<String, PipelineTensor>> _processGPUOperation(
    StageOperation operation,
    Map<String, PipelineTensor> currentData,
  ) async {
    final gpuInputs = <String, Tensor>{};
    for (final key in operation.inputKeys) {
      final located = _findBestInputTensor(key, currentData);
      if (located == null) {
        throw Exception('GPU operation input "$key" not found');
      }
      gpuInputs[key] = await located.asGPU();
    }

    final parameters = <String, dynamic>{};
    for (final param in parameterDefinitions.values) {
      parameters[param.name] = getPropertyValue(param.name);
    }

    final cache = resourceCacheFor(operation);
    final gpuOutputs = await operation.executeWithCache(
      gpuInputs,
      parameters,
      cache,
    );

    // SAFETY: never let an op overwrite the streaming base 'input' with a different shape.
    for (final entry in gpuOutputs.entries) {
      final outKey = entry.key;
      final tensor = entry.value;
      final newShape = tensor.shape;

      final existing = currentData[outKey];
      if (existing != null) {
        final oldShape = existing.shape;
        final sameShape =
            oldShape.length == newShape.length &&
            List.generate(
              oldShape.length,
              (i) => oldShape[i] == newShape[i],
            ).every((v) => v);

        if (outKey == 'input' && !sameShape) {
          // Redirect to a safer name to avoid corrupting future CPU stage expectations.
          final redirectedKey = 'derived_input_${newShape.join("x")}';
          currentData[redirectedKey] = PipelineTensor.gpu(tensor, newShape);
          continue;
        }

        // If shapes differ for any other key, keep old tensor under its name
        // and store new under a versioned temp key to avoid aliasing.
        if (!sameShape) {
          final altKey = '${outKey}_v${DateTime.now().microsecondsSinceEpoch}';
          currentData[altKey] = PipelineTensor.gpu(tensor, newShape);
          continue;
        }

        existing.setGpuTensor(tensor, newShape: newShape);
      } else {
        currentData[outKey] = PipelineTensor.gpu(tensor, newShape);
      }
    }

    return currentData;
  }

  /// Find the best input tensor for an operation input key
  PipelineTensor? _findBestInputTensor(
    String inputKey,
    Map<String, PipelineTensor> tensors,
  ) {
    // Strategy 1: Exact key match
    if (tensors.containsKey(inputKey)) {
      return tensors[inputKey];
    }

    // Strategy 2: Look for output from most recent operation
    // This handles the chaining case where we want the latest 'output'
    if (tensors.containsKey('output')) {
      return tensors['output'];
    }

    // Strategy 3: Look for highest temp_ index (most recent operation output)
    int maxTempIndex = -1;
    String? latestTensorKey;
    for (final key in tensors.keys) {
      if (key.startsWith('temp_')) {
        final indexStr = key.substring(5);
        final index = int.tryParse(indexStr);
        if (index != null && index > maxTempIndex) {
          maxTempIndex = index;
          latestTensorKey = key;
        }
      }
    }

    if (latestTensorKey != null) {
      return tensors[latestTensorKey];
    }

    // Strategy 4: Fallback to any available tensor
    return tensors.values.isNotEmpty ? tensors.values.last : null;
  }

  /// Convert PipelineTensors back to regular GPU Tensors for output
  Future<Map<String, Tensor>> _convertToGPUTensors(
    Map<String, PipelineTensor> pipelineTensors,
  ) async {
    final result = <String, Tensor>{};

    // If no explicit output ports, find the final tensor from operations
    if (outputPorts.isEmpty) {
      final finalTensor = _findFinalOutputTensor(pipelineTensors);
      if (finalTensor != null) {
        result['output'] = await finalTensor.asGPU();
      }
    } else {
      // Map to explicit output ports
      for (final port in outputPorts) {
        final pipelineTensor = _findBestOutputMatch(port.name, pipelineTensors);
        if (pipelineTensor != null) {
          result[port.name] = await pipelineTensor.asGPU();
        }
      }
    }

    return result;
  }

  /// Find the best output tensor match for a given port name
  PipelineTensor? _findBestOutputMatch(
    String portName,
    Map<String, PipelineTensor> tensors,
  ) {
    // Strategy 1: Look for exact name match
    if (tensors.containsKey(portName)) return tensors[portName];

    // Strategy 2: Look for output from last operation
    if (operations.isNotEmpty) {
      final lastOp = operations.last;
      for (final outputKey in lastOp.outputKeys) {
        if (tensors.containsKey(outputKey)) {
          return tensors[outputKey];
        }
      }
    }

    // Strategy 3: Fallback to any available tensor
    return tensors.values.isNotEmpty ? tensors.values.first : null;
  }

  /// Find the final output tensor from operation results
  PipelineTensor? _findFinalOutputTensor(Map<String, PipelineTensor> tensors) {
    // Strategy 1: Look for output from last operation
    if (operations.isNotEmpty) {
      final lastOp = operations.last;
      if (lastOp.outputKeys.isNotEmpty) {
        for (final outputKey in lastOp.outputKeys) {
          if (tensors.containsKey(outputKey)) {
            return tensors[outputKey];
          }
        }
      }
    }

    // Strategy 2: Look for 'output' key
    if (tensors.containsKey('output')) {
      return tensors['output'];
    }

    // Strategy 3: Look for highest temp_ index
    int maxTempIndex = -1;
    String? finalTensorKey;
    for (final key in tensors.keys) {
      if (key.startsWith('temp_')) {
        final indexStr = key.substring(5);
        final index = int.tryParse(indexStr);
        if (index != null && index > maxTempIndex) {
          maxTempIndex = index;
          finalTensorKey = key;
        }
      }
    }

    if (finalTensorKey != null) {
      return tensors[finalTensorKey];
    }

    // Strategy 4: Return first available tensor
    return tensors.values.isNotEmpty ? tensors.values.first : null;
  }

  @override
  Future<void> onParameterChanged(
    String name,
    dynamic oldValue,
    dynamic newValue,
  ) async {
    super.onParameterChanged(name, oldValue, newValue);

    // Emit parameter change event
    emitEvent(
      ParameterChangeEvent(
        pipelineId: 'unknown',
        stageId: stageId,
        parameterName: name,
        oldValue: oldValue,
        newValue: newValue,
        parameterType: newValue.runtimeType.toString(),
      ),
    );
  }

  @override
  Future<void> onParametersChanged(
    Map<String, dynamic> oldValues,
    Map<String, dynamic> newValues,
  ) async {
    super.onParametersChanged(oldValues, newValues);

    // Emit bulk parameter change event
    for (final entry in newValues.entries) {
      final oldValue = oldValues[entry.key];
      if (oldValue != entry.value) {
        emitEvent(
          ParameterChangeEvent(
            pipelineId: 'unknown',
            stageId: stageId,
            parameterName: entry.key,
            oldValue: oldValue,
            newValue: entry.value,
            parameterType: entry.value.runtimeType.toString(),
          ),
        );
      }
    }
  }

  @override
  Future<void> initializeStage() async {
    // Initialize any stage-specific resources
    await super.initializeStage();
  }

  @override
  Future<void> dispose() async {
    for (final operation in operations) {
      await operation.dispose();
    }

    // Clean up fallback cache if we created one
    if (_resourceCache != null &&
        _resourceCache!.cacheId.startsWith('operation_')) {
      _resourceCache!.dispose();
    }

    await super.dispose();
  }

  @override
  String toString() =>
      'DynamicStage($stageId, ops: ${operations.length}, '
      'params: ${parameterDefinitions.length}, '
      'inputs: ${inputPorts.length}, '
      'outputs: ${outputPorts.length})';

  /// True if all operations in this stage are CPU operations.
  bool get isCpuOnly => operations.every((op) => op is CPUStageOperation);

  /// Run this stage using only CPU operations. Only call when isCpuOnly == true.
  /// Returns host outputs and their shapes without promoting to GPU.
  Future<({Map<String, TypedData> data, Map<String, List<int>> shapes})>
  processCPUOnly(
    Map<String, TypedData> hostInputs,
    Map<String, List<int>> shapes,
  ) async {
    // Seed PipelineTensor set with HOST data
    var currentData = <String, PipelineTensor>{};
    hostInputs.forEach((k, v) {
      final s = shapes[k] ?? [v.lengthInBytes ~/ 4];
      currentData[k] = PipelineTensor.cpu(v, s);
    });

    // Resolve and run operations (must all be CPU ops)
    for (final operation in operations) {
      if (operation is! CPUStageOperation) {
        throw StateError('processCPUOnly called on a non-CPU stage "$stageId"');
      }

      // Resolve arguments (uses params; no tensors required for CPU ops)
      final params = currentParameterValues();
      final resolved = _argResolver.resolve(
        operation.arguments,
        params: params,
        tensors: const <String, Tensor>{},
      );
      operation.applyResolvedArguments(resolved.positional, resolved.named);

      currentData = await _processCPUOperation(operation, currentData);
    }

    // Convert to host outputs without GPU upload
    final host = await _convertToHostOutputs(currentData);
    return (data: host.$1, shapes: host.$2);
  }

  /// Convert PipelineTensors to host buffers (Float32List) and shapes.
  Future<(Map<String, TypedData>, Map<String, List<int>>)>
  _convertToHostOutputs(Map<String, PipelineTensor> pipelineTensors) async {
    final data = <String, TypedData>{};
    final shapes = <String, List<int>>{};

    if (outputPorts.isEmpty) {
      final finalTensor = _findFinalOutputTensor(pipelineTensors);
      if (finalTensor != null) {
        final host = await finalTensor.asCPU();
        data['output'] = host;
        shapes['output'] = List<int>.from(finalTensor.shape);
      }
      return (data, shapes);
    }

    for (final port in outputPorts) {
      final pt = _findBestOutputMatch(port.name, pipelineTensors);
      if (pt != null) {
        final host = await pt.asCPU();
        data[port.name] = host;
        shapes[port.name] = List<int>.from(pt.shape);
      }
    }

    return (data, shapes);
  }
}
