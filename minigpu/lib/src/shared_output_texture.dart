import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

import 'buffer.dart';

/// A cross-API shared GPU output texture (RGBA8).
///
/// On Windows this is a D3D12 resource (allocated on Dawn's device) exported
/// as an NT shared handle that an external D3D11 device (e.g. FFmpeg's
/// hardware encoder) can open via `ID3D11Device1::OpenSharedResource1`.
///
/// Obtained via `Minigpu.createSharedOutputTexture`. Must be [destroy]ed
/// when no longer needed; the underlying NT handle is closed at that point,
/// so any external D3D11 view must be released first.
final class SharedOutputTexture {
  SharedOutputTexture(PlatformSharedOutputTexture platformTex, Object owner)
    : _platformTex = platformTex {
    _finalizer.attach(this, platformTex, detach: this);
  }

  static final _finalizer = Finalizer<PlatformSharedOutputTexture>(
    (t) => t.destroy(),
  );

  PlatformSharedOutputTexture? _platformTex;
  bool _isValid = true;

  bool get isValid => _isValid && _platformTex != null;

  int get width => _platformTex!.width;
  int get height => _platformTex!.height;

  /// Native NT HANDLE (as integer) for `OpenSharedResource1` on a D3D11
  /// device. Owned by this object; do NOT call `CloseHandle` on it.
  int get d3d11Handle => _platformTex!.d3d11Handle;

  /// Pointer (as integer) to the underlying `ID3D11Texture2D*`. Lives on
  /// the same device as [Minigpu.createD3D11DeviceOnDawnAdapter]. Use this
  /// to skip `OpenSharedResource1` when the consumer (e.g. FFmpeg) shares
  /// the same `ID3D11Device`.
  int get d3d11TexturePtr => _platformTex!.d3d11TexturePtr;

  /// Copy the contents of [src] (an RGBA8 GPU storage buffer, e.g. the output
  /// of a [GpuEffect] dispatch) into this shared texture entirely on the GPU.
  /// Returns true on success.
  bool copyFromBuffer(Buffer src) =>
      _platformTex!.copyFromBuffer(src.platformBuffer!);

  /// Variant of [copyFromBuffer] for buffers that hold 4 f32 components per
  /// pixel (R,G,B,A in [0,1]) instead of packed RGBA8 u32.  Used by
  /// visualizers (e.g. the spectrogram) that produce float colors directly
  /// into a tensor-backed buffer.
  bool copyFromBufferF32(Buffer src) =>
      _platformTex!.copyFromBufferF32(src.platformBuffer!);

  /// Debug-only: synchronously read the first pixel (BGRA8 packed u32) from
  /// the underlying D3D11 texture using the cached Dawn-adapter D3D11 device.
  /// Used to verify that Dawn writes are visible to the D3D11 consumer.
  /// Returns 0xDEAD000N codes on failure.
  int debugReadFirstPixel() => _platformTex!.debugReadFirstPixel();

  /// Debug-only: synchronously read the first pixel via Dawn's
  /// `CopyTextureToBuffer` + map. Compare against [debugReadFirstPixel] to
  /// determine whether Dawn sees its own writes (and the D3D11 side
  /// doesn't), or whether neither side does.
  int debugReadFirstPixelDawn() => _platformTex!.debugReadFirstPixelDawn();

  /// Internal: exposes the platform-layer texture for use by
  /// `VideoTexture.bgraToRgbaSharedOutput`.
  PlatformSharedOutputTexture get platformTexture => _platformTex!;

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
