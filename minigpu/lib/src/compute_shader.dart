import 'package:minigpu/src/buffer.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

/// A compute shader.
final class ComputeShader {
  ComputeShader(PlatformComputeShader shader) : _shader = shader;

  final PlatformComputeShader _shader;
  final Map<String, int> _kernelTags = {};

  /// Loads a kernel string into the shader.
  void loadKernelString(String kernelString) =>
      _shader.loadKernelString(kernelString);

  /// Checks if the shader has a kernel loaded.
  bool hasKernel() => _shader.hasKernel();

  /// Sets a buffer for the specified kernel and tag.
  void setBuffer(String tag, Buffer buffer) {
    if (!_kernelTags.containsKey(tag)) {
      _kernelTags[tag] = _kernelTags.length;
    } else {
      _kernelTags[tag] = _kernelTags[tag]!;
    }
    _shader.setBuffer(_kernelTags[tag]!, buffer.platformBuffer!);
  }

  /// Dispatches the specified kernel with the given work group counts.
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ) async =>
      _shader.dispatch(groupsX, groupsY, groupsZ);

  /// Destroys the compute shader.
  void destroy() => _shader.destroy();
}
