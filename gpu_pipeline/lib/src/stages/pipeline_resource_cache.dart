import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

/// Unified resource cache for pipeline and operation management
class ResourceCache {
  final String _cacheId;
  final int _maxShaders;
  final int _maxTensors;

  final Map<String, ComputeShader> _shaderCache = {};
  final Map<String, Tensor> _tensorCache = {};
  final Map<String, String> _shaderSources = {};
  final Map<String, DateTime> _shaderLastUsed = {};
  final Map<String, DateTime> _tensorLastUsed = {};

  // When a ResourceCache goes out of scope without explicit disposal, the GC
  // finalizer destroys all cached ComputeShaders to free D3D12 pipeline objects.
  static final _shaderFinalizer = Finalizer<Map<String, ComputeShader>>((
    shaders,
  ) {
    for (final shader in shaders.values) {
      try {
        shader.destroy();
      } catch (_) {}
    }
  });

  // Cache performance tracking
  int _cacheHits = 0;
  int _cacheMisses = 0;

  ResourceCache({
    required String cacheId,
    int maxShaders = 50,
    int maxTensors = 100,
  }) : _cacheId = cacheId,
       _maxShaders = maxShaders,
       _maxTensors = maxTensors {
    _shaderFinalizer.attach(this, _shaderCache, detach: this);
  }

  // Factory constructors for different use cases
  factory ResourceCache.forPipeline(String pipelineId) {
    return ResourceCache(
      cacheId: 'pipeline_$pipelineId',
      maxShaders: 100,
      maxTensors: 200,
    );
  }

  factory ResourceCache.forOperation(String operationId) {
    return ResourceCache(
      cacheId: 'operation_$operationId',
      maxShaders: 20,
      maxTensors: 50,
    );
  }

  // Getters for public access
  String get cacheId => _cacheId;
  int get maxShaders => _maxShaders;
  int get maxTensors => _maxTensors;
  int get cacheHits => _cacheHits;
  int get cacheMisses => _cacheMisses;
  int get shaderCacheSize => _shaderCache.length;
  int get tensorCacheSize => _tensorCache.length;

  // Shader cache management
  ComputeShader? getCachedShader(String key) {
    final shader = _shaderCache[key];
    if (shader != null) {
      _shaderLastUsed[key] = DateTime.now();
      _cacheHits++;
    } else {
      _cacheMisses++;
    }
    return shader;
  }

  void cacheShader(String key, ComputeShader shader, String source) {
    // Check if we already have this shader to avoid duplicates
    if (_shaderCache.containsKey(key)) {
      _shaderLastUsed[key] = DateTime.now();
      return;
    }

    // Clean cache if needed BEFORE adding new shader
    if (_shaderCache.length >= _maxShaders) {
      _cleanShaderCache();
    }

    _shaderCache[key] = shader;
    _shaderSources[key] = source;
    _shaderLastUsed[key] = DateTime.now();
  }

  bool hasShader(String key) => _shaderCache.containsKey(key);

  // Tensor cache management
  Tensor? getCachedTensor(String key) {
    final tensor = _tensorCache[key];
    if (tensor != null) {
      _tensorLastUsed[key] = DateTime.now();
      _cacheHits++;
    } else {
      _cacheMisses++;
    }
    return tensor;
  }

  void cacheTensor(String key, Tensor tensor) {
    // Check if we already have this tensor to avoid duplicates
    if (_tensorCache.containsKey(key)) {
      _tensorLastUsed[key] = DateTime.now();
      return;
    }

    // Clean cache if needed BEFORE adding new tensor
    if (_tensorCache.length >= _maxTensors) {
      _cleanTensorCache();
    }

    _tensorCache[key] = tensor;
    _tensorLastUsed[key] = DateTime.now();
  }

  bool hasTensor(String key) => _tensorCache.containsKey(key);

  // Create cache keys with context
  String createShaderKey(
    String template,
    Map<String, dynamic> compileTimeParams,
  ) {
    final paramHash = compileTimeParams.entries
        .where((e) => _isCompileTimeParameter(e.key, e.value))
        .map((e) => '${e.key}:${e.value}')
        .join('|');
    return '${_cacheId}_${template.hashCode}_$paramHash';
  }

  String createTensorKey(String operationId, List<int> shape, String dtype) {
    return '${_cacheId}_${operationId}_${shape.join('x')}_$dtype';
  }

  static bool _isCompileTimeParameter(String key, dynamic value) {
    if (value is bool) return true;
    if (value is String && !value.startsWith('param:')) return true;

    if (key.contains('size') ||
        key.contains('count') ||
        key.contains('length')) {
      return true;
    }

    return false;
  }

  // Cache cleaning with LRU strategy - made safer
  void _cleanShaderCache() {
    if (_shaderCache.length < _maxShaders) return;

    try {
      final sortedEntries = _shaderLastUsed.entries.toList()
        ..sort((a, b) => a.value.compareTo(b.value));

      // Remove oldest 25% of shaders to avoid frequent cleanup
      final toRemoveCount = (_shaderCache.length * 0.25).ceil();
      final toRemove = sortedEntries.take(toRemoveCount);

      for (final entry in toRemove) {
        final key = entry.key;
        final shader = _shaderCache[key];

        if (shader != null) {
          try {
            // Only destroy if shader is not currently in use
            shader.destroy();
          } catch (e) {
            // Log but don't crash - shader might be in use
            print('⚠️ Warning: Could not destroy shader $key: $e');
          }
        }

        _shaderCache.remove(key);
        _shaderSources.remove(key);
        _shaderLastUsed.remove(key);
      }
    } catch (e) {
      print('⚠️ Warning: Shader cache cleanup failed: $e');
      // Don't crash - continue with full cache
    }
  }

  void _cleanTensorCache() {
    if (_tensorCache.length < _maxTensors) return;

    try {
      final sortedEntries = _tensorLastUsed.entries.toList()
        ..sort((a, b) => a.value.compareTo(b.value));

      // Remove oldest 25% of tensors to avoid frequent cleanup
      final toRemoveCount = (_tensorCache.length * 0.25).ceil();
      final toRemove = sortedEntries.take(toRemoveCount);

      for (final entry in toRemove) {
        final key = entry.key;
        final tensor = _tensorCache[key];

        if (tensor != null) {
          try {
            tensor.destroy();
          } catch (e) {
            // Log but don't crash - tensor might be in use
            print('⚠️ Warning: Could not destroy tensor $key: $e');
          }
        }

        _tensorCache.remove(key);
        _tensorLastUsed.remove(key);
      }
    } catch (e) {
      print('⚠️ Warning: Tensor cache cleanup failed: $e');
      // Don't crash - continue with full cache
    }
  }

  // Manual cleanup methods
  void clearShaderCache() {
    _shaderFinalizer.detach(
      this,
    ); // prevent finalizer from running on already-cleared cache
    final shaders = List<ComputeShader>.from(_shaderCache.values);
    _shaderCache.clear();
    _shaderSources.clear();
    _shaderLastUsed.clear();

    // Destroy shaders after clearing cache to avoid concurrent access
    for (final shader in shaders) {
      try {
        shader.destroy();
      } catch (e) {
        print('⚠️ Warning: Could not destroy shader during clear: $e');
      }
    }
  }

  void clearTensorCache() {
    final tensors = List<Tensor>.from(_tensorCache.values);
    _tensorCache.clear();
    _tensorLastUsed.clear();

    // Destroy tensors after clearing cache to avoid concurrent access
    for (final tensor in tensors) {
      try {
        tensor.destroy();
      } catch (e) {
        print('⚠️ Warning: Could not destroy tensor during clear: $e');
      }
    }
  }

  void clearAll() {
    clearShaderCache();
    clearTensorCache();
  }

  /// Get comprehensive cache statistics
  Map<String, dynamic> getCacheStats() {
    final stats = <String, dynamic>{
      'shader_count': _shaderCache.length,
      'tensor_count': _tensorCache.length,
      'max_shaders': _maxShaders,
      'max_tensors': _maxTensors,
      'cache_id': _cacheId,
      'cache_hits': _cacheHits,
      'cache_misses': _cacheMisses,
      'hit_rate': _cacheHits + _cacheMisses > 0
          ? _cacheHits / (_cacheHits + _cacheMisses)
          : 0.0,
    };

    // Calculate memory usage safely
    int memoryUsage = 0;

    // Add shader memory usage
    for (final shader in _shaderCache.values) {
      memoryUsage += shader.shaderCode!.length * 2; // Rough estimate for UTF-16
    }

    // Add tensor memory usage
    for (final tensor in _tensorCache.values) {
      try {
        memoryUsage += tensor.byteSize;
      } catch (e) {
        // Skip tensors that can't provide byte size
        continue;
      }
    }

    stats['memory_usage'] = memoryUsage;

    // Add size information for each cache type
    final shaderSizesBytes = <int>[];
    for (final shader in _shaderCache.values) {
      shaderSizesBytes.add(shader.shaderCode!.length * 2);
    }

    final tensorSizesBytes = <int>[];
    for (final tensor in _tensorCache.values) {
      try {
        tensorSizesBytes.add(tensor.byteSize);
      } catch (e) {
        // Skip tensors that can't provide byte size
        continue;
      }
    }

    // Safe statistics calculation with proper types
    if (shaderSizesBytes.isNotEmpty) {
      final avgShaderSize =
          shaderSizesBytes.reduce((a, b) => a + b) / shaderSizesBytes.length;
      stats['avg_shader_size'] = avgShaderSize.round(); // Convert to int
      stats['max_shader_size'] = shaderSizesBytes.reduce(
        (a, b) => a > b ? a : b,
      );
    } else {
      stats['avg_shader_size'] = 0;
      stats['max_shader_size'] = 0;
    }

    if (tensorSizesBytes.isNotEmpty) {
      final avgTensorSize =
          tensorSizesBytes.reduce((a, b) => a + b) / tensorSizesBytes.length;
      stats['avg_tensor_size'] = avgTensorSize.round(); // Convert to int
      stats['max_tensor_size'] = tensorSizesBytes.reduce(
        (a, b) => a > b ? a : b,
      );
    } else {
      stats['avg_tensor_size'] = 0;
      stats['max_tensor_size'] = 0;
    }

    return stats;
  }

  void debugListCachedShaders() {
    print('Cached Shaders ($_cacheId):');
    if (_shaderCache.isEmpty) {
      print('   No shaders cached.');
      return;
    }

    _shaderCache.forEach((key, shader) {
      final source = _shaderSources[key] ?? 'Unknown source';
      final lastUsed = _shaderLastUsed[key]?.toIso8601String() ?? 'Never';
      print('   $key: $source (Last used: $lastUsed)');
    });
  }

  void printStats() {
    print('🗄️  Resource Cache Stats ($_cacheId):');
    print('   📜 Shaders: ${_shaderCache.length}/$_maxShaders');
    print('   📦 Tensors: ${_tensorCache.length}/$_maxTensors');
    print(
      '   🎯 Hit rate: ${(_cacheHits + _cacheMisses > 0 ? (_cacheHits / (_cacheHits + _cacheMisses) * 100).toStringAsFixed(1) : "0.0")}%',
    );

    if (_shaderCache.isNotEmpty) {
      final avgShaderAge =
          _shaderLastUsed.values
              .map((d) => DateTime.now().difference(d).inSeconds)
              .reduce((a, b) => a + b) /
          _shaderLastUsed.length;
      print('   📜 Avg shader age: ${avgShaderAge.toStringAsFixed(1)}s');
    }

    if (_tensorCache.isNotEmpty) {
      final avgTensorAge =
          _tensorLastUsed.values
              .map((d) => DateTime.now().difference(d).inSeconds)
              .reduce((a, b) => a + b) /
          _tensorLastUsed.length;
      print('   📦 Avg tensor age: ${avgTensorAge.toStringAsFixed(1)}s');
    }
  }

  /// Reset cache statistics
  void resetStats() {
    _cacheHits = 0;
    _cacheMisses = 0;
  }

  void dispose() {
    try {
      clearAll();
    } catch (e) {
      print('⚠️ Warning: Cache disposal failed: $e');
    }
    resetStats();
  }
}

// Legacy type alias for backward compatibility
typedef PipelineResourceCache = ResourceCache;
typedef OperationResourceCache = ResourceCache;
