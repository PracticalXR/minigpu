import 'dart:typed_data';

import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

/// Returns the element (as num) at [index] for a TypedData, if supported.
num getTypedDataElement(TypedData data, int index) {
  if (data is Int8List) return data[index];
  if (data is Int16List) return data[index];
  if (data is Int32List) return data[index];
  if (data is Int64List) return data[index];
  if (data is Uint8List) return data[index];
  if (data is Uint16List) return data[index];
  if (data is Uint32List) return data[index];
  if (data is Float32List) return data[index];
  if (data is Float64List) return data[index];
  throw Exception("Unsupported TypedData type: ${data.runtimeType}");
}

/// Returns a sublist of [data] from [start] (inclusive) to [end] (exclusive),
/// preserving the underlying type.
T getTypedDataSublist<T extends TypedData>(T data, int start, int end) {
  if (data is Int8List) return (data.sublist(start, end)) as T;
  if (data is Int16List) return (data.sublist(start, end)) as T;
  if (data is Int32List) return (data.sublist(start, end)) as T;
  if (data is Int64List) return (data.sublist(start, end)) as T;
  if (data is Uint8List) return (data.sublist(start, end)) as T;
  if (data is Uint16List) return (data.sublist(start, end)) as T;
  if (data is Uint32List) return (data.sublist(start, end)) as T;
  if (data is Float32List) return (data.sublist(start, end)) as T;
  if (data is Float64List) return (data.sublist(start, end)) as T;
  throw Exception("Unsupported TypedData type: ${data.runtimeType}");
}

/// Process-wide cache of compiled compute shaders, keyed per-Minigpu
/// instance by the FULL WGSL source.
///
/// Every tensor op used to create + compile + destroy its shader PER CALL —
/// per-frame Tint compilation in disguise for anything invoking ops at frame
/// rate (e.g. gpu_pipeline's StreamMergeStage merging audio streams).  Op
/// sources embed sizes/dtypes, so steady-state workloads (the same shapes
/// every frame) hit the cache from the second call on.
///
/// Eviction is two-phase: on overflow the oldest half is RETIRED (dropped
/// from the cache) and destroyed only on the NEXT overflow — a shader that
/// was mid-dispatch when evicted must never be destroyed under it.
class TensorShaderCache {
  TensorShaderCache._();

  static const int _maxPerGpu = 512;
  static final Map<Minigpu, Map<String, ComputeShader>> _byGpu = Map.identity();
  static final Map<Minigpu, List<ComputeShader>> _retired = Map.identity();

  static ComputeShader acquire(Minigpu gpu, String source) {
    final cache = _byGpu.putIfAbsent(gpu, () => <String, ComputeShader>{});
    final existing = cache[source];
    if (existing != null) return existing;

    if (cache.length >= _maxPerGpu) {
      // Destroy the batch retired at the PREVIOUS overflow (their dispatches
      // finished long ago), then retire the oldest half of the current set.
      final retired = _retired.putIfAbsent(gpu, () => <ComputeShader>[]);
      for (final s in retired) {
        try {
          s.destroy();
        } catch (_) {}
      }
      retired.clear();
      for (final k in cache.keys.take(_maxPerGpu ~/ 2).toList()) {
        retired.add(cache.remove(k)!);
      }
    }

    final shader = gpu.createComputeShader()..loadKernelString(source);
    cache[source] = shader;
    return shader;
  }

  /// Number of cached shaders for [gpu] (tests/introspection).
  static int sizeFor(Minigpu gpu) => _byGpu[gpu]?.length ?? 0;

  /// Destroys all cached shaders (for [gpu], or every instance).  Only call
  /// when no tensor ops are in flight.
  static void clear([Minigpu? gpu]) {
    final targets = gpu != null ? [gpu] : _byGpu.keys.toList();
    for (final g in targets) {
      final cache = _byGpu.remove(g);
      if (cache != null) {
        for (final s in cache.values) {
          try {
            s.destroy();
          } catch (_) {}
        }
      }
      final retired = _retired.remove(g);
      if (retired != null) {
        for (final s in retired) {
          try {
            s.destroy();
          } catch (_) {}
        }
      }
    }
  }
}

extension CachedShaderAcquire on Minigpu {
  /// A compiled shader for [source], compiled at most once per source per
  /// Minigpu instance.  Callers MUST NOT destroy the returned shader, and
  /// must complete their setBuffer(...) calls + dispatch(...) without
  /// awaiting in between — the next caller re-binds the same shader object.
  ComputeShader cachedShader(String source) =>
      TensorShaderCache.acquire(this, source);
}

String prepareShader(
  String template,
  BufferDataType dataType,
  Map<String, dynamic> values,
) {
  String result = template;
  values.forEach((key, value) {
    String valueString;
    if (value is int) {
      valueString = '${value}'; // Format as WGSL unsigned integer literal
    } else if (value is double) {
      // Ensure float format for WGSL f32 literal
      valueString = value.toString().contains('.')
          ? value.toString()
          : '$value.0';
      // Optionally add 'f' suffix if needed: valueString = '${valueString}f';
    } else {
      valueString = value.toString(); // Default string conversion
    }
    // Replace placeholder like ${key}
    result = result.replaceAll('\${$key}', valueString);
  });
  // Templates are written against a canonical `array<f32>` element type,
  // rewritten here to the tensor's dtype.  `array<u32>`/`array<i32>` in a
  // template are intentional non-data bindings (params, packed byte views)
  // and must NOT be rewritten.
  final wgslType = getWGSLType(dataType);
  result = result.replaceAll('array<f32>', 'array<$wgslType>');
  return result;
}

/// Dispatches a 1D logical workload of [threads] invocations, folding into
/// (x, y) workgroup counts when the workgroup count exceeds WebGPU's 65535
/// per-dimension limit (tensors > ~16.7M elements at workgroup_size 256).
///
/// Kernels dispatched through this MUST compute their linear index as
/// `let i: u32 = gid.x + gid.y * (nwg.x * 256u);`
/// with `@builtin(num_workgroups) nwg` — see [linearIndexWGSL] — and bounds
/// check against the logical size (the fold overshoots by design).
extension LinearDispatch on ComputeShader {
  Future<void> dispatchLinear(int threads, {int workgroupSize = 256}) {
    final wg = (threads + workgroupSize - 1) ~/ workgroupSize;
    final x = wg <= 65535 ? wg : 65535;
    final y = (wg + x - 1) ~/ x;
    return dispatch(x == 0 ? 1 : x, y == 0 ? 1 : y, 1);
  }
}

/// Canonical main-signature + linear-index preamble for 1D kernels that are
/// dispatched via [LinearDispatch.dispatchLinear].  Uses num_workgroups so
/// the source stays independent of the dispatch size (cache friendly).
const String linearMainSignature =
    'fn main(@builtin(global_invocation_id) gid: vec3<u32>, '
    '@builtin(num_workgroups) nwg: vec3<u32>)';
const String linearIndexWGSL = 'let i: u32 = gid.x + gid.y * (nwg.x * 256u);';

// Helper functions for type-conscious shader generation
String getZeroValue(BufferDataType dataType) {
  switch (dataType) {
    case BufferDataType.float32:
    case BufferDataType.float64:
    case BufferDataType.float16:
      return '0.0';
    case BufferDataType.int8:
    case BufferDataType.int16:
    case BufferDataType.int32:
    case BufferDataType.int64:
      return '0i';
    case BufferDataType.uint8:
    case BufferDataType.uint16:
    case BufferDataType.uint32:
    case BufferDataType.uint64:
      return '0u';
  }
}

String getCastExpression(String value, BufferDataType dataType) {
  switch (dataType) {
    case BufferDataType.float32:
    case BufferDataType.float64:
    case BufferDataType.float16:
      return value; // Already float
    case BufferDataType.int8:
    case BufferDataType.int16:
    case BufferDataType.int32:
    case BufferDataType.int64:
      return 'i32($value)';
    case BufferDataType.uint8:
    case BufferDataType.uint16:
    case BufferDataType.uint32:
    case BufferDataType.uint64:
      return 'u32($value)';
  }
}

String getByteConversionCode(BufferDataType dataType) {
  switch (dataType) {
    case BufferDataType.float32:
      return '''
        let byte_index = i * 4u;
        let b0 = (input_bytes[byte_index / 4u] >> ((byte_index % 4u) * 8u)) & 0xFFu;
        let b1 = (input_bytes[(byte_index + 1u) / 4u] >> (((byte_index + 1u) % 4u) * 8u)) & 0xFFu;
        let b2 = (input_bytes[(byte_index + 2u) / 4u] >> (((byte_index + 2u) % 4u) * 8u)) & 0xFFu;
        let b3 = (input_bytes[(byte_index + 3u) / 4u] >> (((byte_index + 3u) % 4u) * 8u)) & 0xFFu;
        let bits = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
        output[i] = bitcast<f32>(bits);
      ''';
    case BufferDataType.int32:
      return '''
        let byte_index = i * 4u;
        let b0 = (input_bytes[byte_index / 4u] >> ((byte_index % 4u) * 8u)) & 0xFFu;
        let b1 = (input_bytes[(byte_index + 1u) / 4u] >> (((byte_index + 1u) % 4u) * 8u)) & 0xFFu;
        let b2 = (input_bytes[(byte_index + 2u) / 4u] >> (((byte_index + 2u) % 4u) * 8u)) & 0xFFu;
        let b3 = (input_bytes[(byte_index + 3u) / 4u] >> (((byte_index + 3u) % 4u) * 8u)) & 0xFFu;
        let packed = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
        output[i] = i32(packed);
      ''';
    case BufferDataType.uint8:
      return '''
        let byte_index = i;
        let packed = (input_bytes[byte_index / 4u] >> ((byte_index % 4u) * 8u)) & 0xFFu;
        output[i] = u32(packed);
      ''';
    default:
      return 'output[i] = input_bytes[i];';
  }
}

extension TesorHelper<T extends TypedData> on Tensor<T> {
  /// _shapesCompatible checks if two shapes are compatible for broadcasting.
  bool shapesCompatible(List<int> shape1, List<int> shape2) {
    if (shape1.length != shape2.length) return false;
    for (int i = 0; i < shape1.length; i++) {
      if (shape1[i] != shape2[i] && shape1[i] != 1 && shape2[i] != 1) {
        return false;
      }
    }
    return true;
  }
}
