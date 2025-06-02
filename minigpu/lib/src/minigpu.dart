import 'package:minigpu/src/buffer.dart';
import 'package:minigpu/src/compute_shader.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

/// Controls the initialization and destruction of the minigpu context.
final class Minigpu {
  Minigpu() {
    _finalizer.attach(this, _platform);
  }

  static final _finalizer = Finalizer<MinigpuPlatform>(
    (platform) => platform.destroyContext(),
  );
  static final _shaderFinalizer = Finalizer<ComputeShader>(
    (shader) => shader.destroy(),
  );
  static final _bufferFinalizer = Finalizer<Buffer>(
    (buffer) => buffer.destroy(),
  );

  final _platform = MinigpuPlatform.instance;
  bool isInitialized = false;

  /// Initializes the minigpu context.
  Future<void> init() async {
    if (isInitialized) throw MinigpuAlreadyInitError();

    await _platform.initializeContext();
    isInitialized = true;
  }

  /// Destroys the minigpu context.
  Future<void> destroy() async {
    if (!isInitialized) throw MinigpuNotInitializedError();

    await _platform.destroyContext();
    isInitialized = false;
  }

  /// Creates a compute shader.
  ComputeShader createComputeShader() {
    final platformShader = _platform.createComputeShader();
    final shader = ComputeShader(platformShader);
    _shaderFinalizer.attach(this, shader);
    return shader;
  }

  /// Creates a buffer.
  Buffer createBuffer(int bufferSize, BufferDataType dataType) {
    final platformBuffer = _platform.createBuffer(bufferSize, dataType);
    final buff = Buffer(platformBuffer);
    _bufferFinalizer.attach(this, buff);
    return buff;
  }
}
