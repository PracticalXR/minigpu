import 'package:minigpu/src/minigpu.dart';
import 'package:minigpu/src/buffer.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

/// A compute shader.
final class ComputeShader {
  ComputeShader(PlatformComputeShader shader) : _shader = shader {
    _finalizer.attach(this, shader, detach: this);
  }

  final PlatformComputeShader _shader;

  /// Expose the underlying platform shader for use by VideoTexture.setOnShader.
  PlatformComputeShader get platformShader => _shader;

  final Map<String, int> _kernelTags = {};
  String? shaderCode;
  static final Finalizer<PlatformComputeShader> _finalizer = Finalizer(
    (platformShader) => platformShader.destroy(),
  );

  /// Reset tag->binding index mapping so the next binding session starts at 0.
  void resetTagOrder() => _kernelTags.clear();

  void loadKernelString(String kernelString) {
    shaderCode = kernelString;
    _kernelTags.clear(); // fresh tag ordering for this kernel
    return _shader.loadKernelString(kernelString);
  }

  /// Checks if the shader has a kernel loaded.
  bool hasKernel() => _shader.hasKernel();

  /// Sets a buffer for the specified kernel and tag.
  void setBuffer(String tag, Buffer buffer) {
    try {
      if (!_kernelTags.containsKey(tag)) {
        _kernelTags[tag] = _kernelTags.length;
      } else {
        _kernelTags[tag] = _kernelTags[tag]!;
      }
      _shader.setBuffer(_kernelTags[tag]!, buffer.platformBuffer!);
    } catch (e, stackTrace) {
      print('Error setting buffer for tag $tag: $e\n$stackTrace');
      throw Exception('Failed to set buffer for tag $tag: $e');
    }
  }

  /// Sets a buffer at an explicit binding [slot] index.
  ///
  /// Use this when mixing texture bindings (set via [VideoTexture.setOnShader])
  /// with buffer bindings in the same shader, where slot numbers must be
  /// coordinated explicitly rather than derived from tag insertion order.
  void setBufferAtSlot(int slot, Buffer buffer) {
    _shader.setBuffer(slot, buffer.platformBuffer!);
  }

  /// Dispatches the specified kernel with the given work group counts.
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ) async =>
      _shader.dispatch(groupsX, groupsY, groupsZ);

  /// Destroys the compute shader.
  void destroy() {
    _finalizer.detach(this); // Use the same detach key
    _shader.destroy();
  }
}

/// Internal wrapper that provides caching behavior
final class CachedComputeShader extends ComputeShader {
  final Minigpu _gpu;

  CachedComputeShader(PlatformComputeShader shader, this._gpu) : super(shader);

  @override
  void destroy() {
    super.destroy();
    _gpu.onShaderDestroyed(this);
  }
}
