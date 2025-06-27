import 'package:minigpu/src/buffer.dart';
import 'package:minigpu/src/compute_shader.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'dart:convert';

/// Controls the initialization and destruction of the minigpu context.
final class Minigpu {
  Minigpu() {
    _finalizer.attach(this, _platform);
  }

  static final _finalizer = Finalizer<MinigpuPlatform>(
    (platform) => platform.destroyContext(),
  );
  static final _shaderFinalizer = Finalizer<PlatformComputeShader>(
    (shader) => shader.destroy(),
  );
  static final _bufferFinalizer = Finalizer<Buffer>(
    (buffer) => buffer.destroy(),
  );

  final _platform = MinigpuPlatform.instance;
  bool isInitialized = false;

  // Internal shader cache: code hash -> shader
  final Map<String, ComputeShader> _shaderCache = {};

  /// Initializes the minigpu context.
  Future<void> init() async {
    if (isInitialized) throw MinigpuAlreadyInitError();

    await _platform.initializeContext();
    isInitialized = true;
  }

  /// Destroys the minigpu context.
  Future<void> destroy() async {
    if (!isInitialized) throw MinigpuNotInitializedError();

    _clearShaderCache();
    await _platform.destroyContext();
    isInitialized = false;
  }

  /// Creates a compute shader.
  ComputeShader createComputeShader() {
    final platformShader = _platform.createComputeShader();
    final shader = CachedComputeShader(platformShader, this);
    return shader;
  }

  /// Internal method to get or create cached shader by code
  ComputeShader getOrCreateCachedShader(String shaderCode) {
    // Simple hash using built-in hashCode
    final codeHash = shaderCode.hashCode.toString();

    if (_shaderCache.containsKey(codeHash)) {
      return _shaderCache[codeHash]!;
    }

    final platformShader = _platform.createComputeShader();
    final shader = ComputeShader(platformShader);
    shader.loadKernelString(shaderCode);
    _shaderCache[codeHash] = shader;
    return shader;
  }

  /// Creates a buffer.
  Buffer createBuffer(int bufferSize, BufferDataType dataType) {
    if (!isInitialized) throw MinigpuNotInitializedError();

    final platformBuffer = _platform.createBuffer(bufferSize, dataType);
    final buff = Buffer(platformBuffer);
    return buff;
  }

  /// Clear shader cache (internal)
  void _clearShaderCache() {
    for (final shader in _shaderCache.values) {
      shader.destroy();
    }
    _shaderCache.clear();
  }
}
