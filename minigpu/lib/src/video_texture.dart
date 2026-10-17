import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

import 'buffer.dart';
import 'compute_shader.dart';
import 'resource_retirement.dart';
import 'shared_output_texture.dart';

/// An imported GPU video texture (zero-copy from a MiniAV frame or similar).
///
/// Obtained via [Minigpu.importVideoFrame]. Must be [destroy]ed when no longer
/// needed. The texture is valid only until the underlying GPU fence signals or
/// the next frame is captured, so it should be consumed in the same async turn.
final class VideoTexture {
  /// [Minigpu.importVideoFrame] passes the [retirement] its context retires
  /// at destroy and device recovery, and [adoptBuffer], which registers each
  /// [toRGBA] result with that context the same way.
  VideoTexture(
    PlatformVideoTexture platformTex,
    Object owner, [
    GpuResourceRetirement? retirement,
    Buffer Function(PlatformBuffer)? adoptBuffer,
  ]) : _platformTex = platformTex,
       _retirement = retirement ?? GpuResourceRetirement(platformTex.destroy),
       _adoptBuffer = adoptBuffer {
    _finalizer.attach(this, _retirement, detach: this);
  }

  static final _finalizer = Finalizer<GpuResourceRetirement>(
    (retirement) => retirement.retire(),
  );

  PlatformVideoTexture? _platformTex;
  final GpuResourceRetirement _retirement;
  final Buffer Function(PlatformBuffer)? _adoptBuffer;
  bool _isValid = true;

  /// Returns true if the texture has not been destroyed.
  bool get isValid =>
      _isValid && _platformTex != null && !_retirement.isRetired;

  /// Number of GPU planes (1 for RGBA/BGRA, 2 for NV12, 3 for I420).
  int get numPlanes => _live.numPlanes;

  int get width => _live.width;
  int get height => _live.height;
  ExternalPixelFormat get pixelFormat => _live.pixelFormat;

  PlatformVideoTexture get platformTexture => _live;

  PlatformVideoTexture get _live {
    final texture = _platformTex;
    if (!isValid || texture == null) {
      throw StateError('Video texture is destroyed');
    }
    return texture;
  }

  /// Bind plane [planeIndex] to compute shader binding [slot].
  void setOnShader(ComputeShader shader, int slot, {int planeIndex = 0}) {
    shader.setVideoTexture(slot, this, planeIndex: planeIndex);
  }

  /// True when [landPackedRgba8] is the only way to consume this texture, so a
  /// failed land must NOT fall back to [setOnShader]. See
  /// [PlatformVideoTexture.requiresExternalLanding].
  bool get requiresExternalLanding => _live.requiresExternalLanding;

  /// Land this texture into [dst] as packed RGBA8 (`array<u32>`, row-major),
  /// returning false when this platform drives textures the ordinary way
  /// ([setOnShader]) instead. See [PlatformVideoTexture.landPackedRgba8].
  bool landPackedRgba8(
    Buffer dst, {
    required int width,
    required int height,
    int downscale = 1,
  }) {
    final pb = dst.platformBuffer;
    if (pb == null) return false;
    return _live.landPackedRgba8(
      pb,
      width: width,
      height: height,
      downscale: downscale,
    );
  }

  /// Convert to an RGBA8 [Buffer] via an internal compute pass.
  /// The returned buffer has `width * height * 4` bytes (row-major, RGBA8).
  ///
  /// GRAY8 becomes opaque grey; NV12 and both YUV 4:2:0 layouts convert with
  /// the [ExternalVideoBuffer.yuvMatrix] and [ExternalVideoBuffer.yuvFullRange]
  /// the frame was imported with (BT.709 full range unless it said otherwise).
  /// A frame of unknown format throws [UnsupportedError]; a frame missing a
  /// plane, or a lost device, throws [StateError]. Not supported for web video
  /// frames. Natively this BLOCKS until the GPU is done: on a per-frame path
  /// prefer [toRGBAAsync].
  Buffer toRGBA() => _adopt(_live.toRGBA());

  /// [toRGBA] without blocking the calling isolate while the GPU converts:
  /// completes with the buffer once the conversion has finished. Destroying
  /// the texture meanwhile is fine; it is released after the conversion. A
  /// device lost or retired before then throws [StateError].
  Future<Buffer> toRGBAAsync() async => _adopt(await _live.toRGBAAsync());

  Buffer _adopt(PlatformBuffer platformBuf) {
    final adopt = _adoptBuffer;
    return adopt != null ? adopt(platformBuf) : Buffer(platformBuf);
  }

  /// Zero-copy convert this BGRA video texture into the given cross-API
  /// shared RGBA output texture (Windows: D3D12<->D3D11 via NT shared
  /// handle). Returns true once the GPU has finished the copy, false on
  /// failure or when that wait timed out. The destination must have been
  /// created via [Minigpu.createSharedOutputTexture] with matching width
  /// and height. Natively this BLOCKS the calling isolate until the GPU is
  /// done: on the UI isolate prefer [bgraToRgbaSharedOutputAsync].
  bool bgraToRgbaSharedOutput(SharedOutputTexture dst) {
    return _live.bgraToRgbaSharedOutput(dst.platformTexture);
  }

  /// Async variant of [bgraToRgbaSharedOutput]: completes when the GPU work
  /// (including the cross-device present sync) has finished, without blocking
  /// the calling isolate on a busy-poll. Prefer this on a per-frame hot path.
  Future<bool> bgraToRgbaSharedOutputAsync(SharedOutputTexture dst) {
    return _live.bgraToRgbaSharedOutputAsync(dst.platformTexture);
  }

  /// Release the GPU texture.
  void destroy() {
    if (!_isValid) return;
    _isValid = false;
    _platformTex = null;
    _finalizer.detach(this);
    _retirement.retire();
  }
}
