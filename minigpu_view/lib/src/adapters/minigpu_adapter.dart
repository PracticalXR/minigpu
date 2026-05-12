/// Adapter: `SharedOutputTexture.asPreviewSource()`
library;

import 'dart:ui' show Size;

import 'package:minigpu/minigpu.dart';

import '../preview_source.dart';

/// Adapts a minigpu [SharedOutputTexture] into a [PreviewSource].
extension MinigpuPreviewSourceAdapter on SharedOutputTexture {
  /// Wrap as a [PreviewSource]. The widget borrows the underlying GPU
  /// texture; do **not** destroy the [SharedOutputTexture] until the
  /// preview controller has been disposed.
  PreviewSource asPreviewSource() => _SharedOutputPreviewSource(this);
}

class _SharedOutputPreviewSource extends PreviewSource {
  final SharedOutputTexture _tex;
  const _SharedOutputPreviewSource(this._tex);

  @override
  PreviewSourceKind get kind => PreviewSourceKind.nativeSharedTexture;

  @override
  Size get size => Size(_tex.width.toDouble(), _tex.height.toDouble());

  @override
  Map<String, Object?> toChannelMessage() => {
    // ID3D11Texture2D* (raw pointer) — used by same-device consumers like
    // the FFmpeg D3D11VA encoder (which lives on the same Dawn-adapter
    // D3D11 device).  Flutter cannot open this directly because ANGLE
    // runs on its own D3D11 device.
    'handle': _tex.d3d11TexturePtr,
    // Legacy DXGI shared HANDLE — produced via IDXGIResource::GetSharedHandle
    // on a texture created with D3D11_RESOURCE_MISC_SHARED.  Flutter's
    // ANGLE OpenSharedResource()'s this on its own device.
    'sharedHandle': _tex.d3d11Handle,
    'width': _tex.width,
    'height': _tex.height,
    // Pixel format is implicit RGBA8 from minigpu's SharedOutputTexture.
    'pixelFormat': 'rgba8',
  };
}
