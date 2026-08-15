/// Adapter: `SharedOutputTexture.asPreviewSource()`
library;

import 'dart:ui' show Size;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:minigpu/minigpu.dart';

import '../preview_source.dart';

/// Adapts a minigpu [SharedOutputTexture] into a [PreviewSource].
extension MinigpuPreviewSourceAdapter on SharedOutputTexture {
  /// Wrap as a [PreviewSource]. The widget borrows the underlying GPU
  /// texture; do **not** destroy the [SharedOutputTexture] until the
  /// preview controller has been disposed.
  PreviewSource asPreviewSource() {
    if (kIsWeb) {
      // On web: ALWAYS return a webGpuTexture kind source — the web plugin
      // does not handle nativeSharedTexture.  webGpuTextureJs may be null if
      // copyFromBufferF32 hasn't been called yet (first frame); the view
      // plugin skips the frame gracefully when 'buffer' is null.
      return _WebGpuBufferPreviewSource(
        platformTexture.webGpuTextureJs,
        Size(width.toDouble(), height.toDouble()),
      );
    }
    return _SharedOutputPreviewSource(this);
  }
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

/// Web-only: carries the Emscripten WGPUBuffer integer handle as a
/// [PreviewSourceKind.webGpuTexture] source.  Passing an integer (not a
/// JSObject) through the method channel avoids StandardMessageCodec crashes.
/// The view plugin resolves the JS GPUBuffer via WebGPU.getJsObject(handle).
/// [_bufferHandle] is 0 / null when no buffer has been written yet; the view
/// plugin skips the blit gracefully.
class _WebGpuBufferPreviewSource extends PreviewSource {
  final int? _bufferHandle; // Emscripten integer handle, 0/null = skip frame
  final Size _size;

  const _WebGpuBufferPreviewSource(Object? handle, this._size)
    : _bufferHandle = handle is int ? handle : null;

  @override
  PreviewSourceKind get kind => PreviewSourceKind.webGpuTexture;

  @override
  Size get size => _size;

  @override
  Map<String, Object?> toChannelMessage() => {
    // Integer handle — codec-safe.  0/null means skip this frame.
    'bufferHandle': _bufferHandle,
    'width': _size.width.toInt(),
    'height': _size.height.toInt(),
    'format': 'rgba32float',
  };
}

/// A present source built from RAW HANDLES, for producers that publish their
/// surface as integers rather than as a [SharedOutputTexture].
///
/// The extension above is the right entry point whenever the texture OBJECT is
/// in hand — it also picks the correct source kind on web, which this cannot.
/// But a producer that hands out `{sharedHandle, texturePtr, width, height}`
/// across an API boundary leaves the consumer with nothing to call it on, and
/// the consumer then hand-rolls a PreviewSource. This is that class, written
/// once.
///
/// [sharedHandle] is the DXGI shared HANDLE the compositor opens on its own
/// device; [texturePtr] is the raw `ID3D11Texture2D*`, used by same-device
/// consumers. Windows only — on other platforms present the texture object.
class RawSharedTexturePreviewSource extends PreviewSource {
  const RawSharedTexturePreviewSource({
    required this.sharedHandle,
    required this.texturePtr,
    required this.width,
    required this.height,
  });

  final int sharedHandle;
  final int texturePtr;
  final int width;
  final int height;

  @override
  PreviewSourceKind get kind => PreviewSourceKind.nativeSharedTexture;

  @override
  Size get size => Size(width.toDouble(), height.toDouble());

  @override
  Map<String, Object?> toChannelMessage() => {
        'handle': texturePtr,
        'sharedHandle': sharedHandle,
        'width': width,
        'height': height,
        'pixelFormat': 'rgba8',
      };
}

/// Web/storage-buffer flavour of [RawSharedTexturePreviewSource]: a WGPUBuffer
/// integer handle of packed RGBA8.
class RawGpuBufferPreviewSource extends PreviewSource {
  const RawGpuBufferPreviewSource({
    required this.bufferHandle,
    required this.width,
    required this.height,
  });

  final int bufferHandle;
  final int width;
  final int height;

  @override
  PreviewSourceKind get kind => PreviewSourceKind.webGpuTexture;

  @override
  Size get size => Size(width.toDouble(), height.toDouble());

  @override
  Map<String, Object?> toChannelMessage() => {
        'bufferHandle': bufferHandle,
        'width': width,
        'height': height,
        'format': 'rgba8',
      };
}
