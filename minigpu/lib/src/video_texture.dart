import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

import 'buffer.dart';
import 'compute_shader.dart';
import 'shared_output_texture.dart';

/// An imported GPU video texture (zero-copy from a MiniAV frame or similar).
///
/// Obtained via [Minigpu.importVideoFrame]. Must be [destroy]ed when no longer
/// needed. The texture is valid only until the underlying GPU fence signals or
/// the next frame is captured, so it should be consumed in the same async turn.
final class VideoTexture {
  VideoTexture(PlatformVideoTexture platformTex, Object owner)
    : _platformTex = platformTex {
    _finalizer.attach(this, platformTex, detach: this);
  }

  static final _finalizer = Finalizer<PlatformVideoTexture>((t) => t.destroy());

  PlatformVideoTexture? _platformTex;
  bool _isValid = true;

  /// Returns true if the texture has not been destroyed.
  bool get isValid => _isValid && _platformTex != null;

  /// Number of GPU planes (1 for RGBA/BGRA, 2 for NV12, 3 for I420).
  int get numPlanes => _platformTex!.numPlanes;

  int get width => _platformTex!.width;
  int get height => _platformTex!.height;
  ExternalPixelFormat get pixelFormat => _platformTex!.pixelFormat;

  /// Bind plane [planeIndex] to compute shader binding [slot].
  void setOnShader(ComputeShader shader, int slot, {int planeIndex = 0}) {
    _platformTex!.setOnShader(shader.platformShader, slot, planeIndex);
  }

  /// Convert to an RGBA8 [Buffer] via an internal compute pass.
  /// The returned buffer has `width * height * 4` bytes (row-major, RGBA8).
  Buffer toRGBA() {
    final platformBuf = _platformTex!.toRGBA();
    return Buffer(platformBuf);
  }

  /// Zero-copy convert this BGRA video texture into the given cross-API
  /// shared RGBA output texture (Windows: D3D12<->D3D11 via NT shared
  /// handle). Returns true on success. The destination must have been
  /// created via [Minigpu.createSharedOutputTexture] with matching width
  /// and height.
  bool bgraToRgbaSharedOutput(SharedOutputTexture dst) {
    return _platformTex!.bgraToRgbaSharedOutput(dst.platformTexture);
  }

  /// Release the GPU texture.
  void destroy() {
    if (!_isValid) return;
    _isValid = false;
    final t = _platformTex;
    _platformTex = null;
    if (t != null) {
      _finalizer.detach(this);
      t.destroy();
    }
  }
}
