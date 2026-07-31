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
  ///
  /// The bind joins the WebGPU-thread FIFO, so it is ordered against dispatches
  /// however they were issued — including [dispatchFire]. Each queued dispatch
  /// sees the binds queued before it, which makes bind → fire → rebind → fire
  /// correct on a single shader.
  void setBuffer(String tag, Buffer buffer) {
    try {
      if (!_kernelTags.containsKey(tag)) {
        _kernelTags[tag] = _kernelTags.length;
      } else {
        _kernelTags[tag] = _kernelTags[tag]!;
      }
      _shader.setBufferFire(_kernelTags[tag]!, buffer.platformBuffer!);
    } catch (e, stackTrace) {
      print('Error setting buffer for tag $tag: $e\n$stackTrace');
      throw Exception('Failed to set buffer for tag $tag: $e');
    }
  }

  /// Deprecated alias for [setBuffer], which is now always ordered.
  @Deprecated(
    'Binds are always ordered as of 1.5.9 — use setBuffer. '
    'This alias will be removed in a future release.',
  )
  void setBufferFire(String tag, Buffer buffer) => setBuffer(tag, buffer);

  /// Sets a buffer at an explicit binding [slot] index.
  ///
  /// Use this when mixing texture bindings (set via [VideoTexture.setOnShader])
  /// with buffer bindings in the same shader, where slot numbers must be
  /// coordinated explicitly rather than derived from tag insertion order.
  void setBufferAtSlot(int slot, Buffer buffer) {
    _shader.setBufferFire(slot, buffer.platformBuffer!);
  }

  /// WebGPU caps the workgroup count at 65535 PER DIMENSION. A dispatch that
  /// exceeds it invalidates the whole CommandBuffer — and because WebGPU
  /// validation errors are sticky, every SUBSEQUENT submit on the device then
  /// fails with "[Invalid CommandBuffer] is invalid due to a previous error",
  /// so one bad dispatch silently poisons unrelated work. Fail loudly and
  /// early instead, naming the offending dims, so callers get a catchable error
  /// (e.g. fall back to CPU) rather than a cascading device-wide failure.
  static const int maxWorkgroupsPerDim = 65535;
  static void _checkDispatch(int x, int y, int z) {
    if (x > maxWorkgroupsPerDim ||
        y > maxWorkgroupsPerDim ||
        z > maxWorkgroupsPerDim) {
      throw ArgumentError(
        'Compute dispatch ($x, $y, $z) exceeds WebGPU\'s limit of '
        '$maxWorkgroupsPerDim workgroups per dimension. Fold the overflow into '
        'another dimension: gx = min(n, 65535); gy = (n + gx - 1) ~/ gx; and '
        'reconstruct the flat index in the shader as '
        '`gid.x + gid.y * (num_workgroups.x * workgroup_size_x)`.',
      );
    }
  }

  /// Dispatches the specified kernel with the given work group counts.
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ) async {
    _checkDispatch(groupsX, groupsY, groupsZ);
    return _shader.dispatch(groupsX, groupsY, groupsZ);
  }

  /// Fire-and-forget dispatch: enqueues the compute pass and returns
  /// immediately (no per-dispatch completer round trip).  Dispatches, buffer
  /// writes and reads still execute in call order, so awaiting any later
  /// buffer read synchronizes every fired dispatch before it.
  ///
  /// Do NOT [setBuffer] on a shader that has a fired dispatch which hasn't
  /// been synchronized by a readback yet — bindings are snapshotted when the
  /// dispatch runs, not when it's fired. Use per-call-site shader instances
  /// with stable bindings on fire-and-forget hot paths.
  void dispatchFire(int groupsX, int groupsY, int groupsZ) {
    _checkDispatch(groupsX, groupsY, groupsZ);
    _shader.dispatchFire(groupsX, groupsY, groupsZ);
  }

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
