/// Adapter: `MiniAVVideoBuffer.asPreviewSource()`
///
/// Provides a zero-copy [PreviewSource] for GPU-backed [MiniAVBuffer]s.
/// When the buffer's [MiniAVBufferContentType] is [gpuD3D11Handle],
/// the underlying `ID3D11Texture2D*` is forwarded directly to the
/// Windows plugin; no CPU readback occurs.
///
/// CPU buffers are NOT wrapped by this adapter — they return null.
/// Keep the existing `ui.decodeImageFromPixels` path for those.
library;

import 'dart:ui' show Size;

import 'package:miniav/miniav.dart'
    show MiniAVBuffer, MiniAVBufferContentType, MiniAVVideoBuffer;

import '../preview_source.dart';

/// Adapts a GPU-backed [MiniAVBuffer] into a [PreviewSource].
///
/// Returns null for CPU buffers or unsupported content types — the
/// caller should fall back to the legacy CPU rendering path.
extension MiniavBufferPreviewSourceAdapter on MiniAVBuffer {
  /// Wrap a GPU-backed video buffer as a [PreviewSource].
  ///
  /// Returns `null` if:
  /// - [contentType] is [MiniAVBufferContentType.cpu]
  /// - [data] is not a [MiniAVVideoBuffer]
  /// - The native handle is absent
  PreviewSource? asPreviewSource() {
    if (contentType == MiniAVBufferContentType.cpu) return null;
    final vb = data;
    if (vb is! MiniAVVideoBuffer) return null;
    return _MiniAVPreviewSource(contentType, vb);
  }
}

class _MiniAVPreviewSource extends PreviewSource {
  final MiniAVBufferContentType _contentType;
  final MiniAVVideoBuffer _vb;

  const _MiniAVPreviewSource(this._contentType, this._vb);

  @override
  PreviewSourceKind get kind => PreviewSourceKind.nativeSharedTexture;

  @override
  Size get size => Size(_vb.width.toDouble(), _vb.height.toDouble());

  @override
  Map<String, Object?> toChannelMessage() {
    final Object? handle;
    final String pixelFormat;

    switch (_contentType) {
      case MiniAVBufferContentType.gpuD3D11Handle:
        // nativeHandles[0] is ID3D11Texture2D* as int (Windows).
        handle = _vb.nativeHandles.isNotEmpty ? _vb.nativeHandles[0] : null;
        pixelFormat = _pixelFormatName();
      default:
        // gpuMetalTexture / gpuDmabufFd / gpuAHardwareBuffer —
        // stubs for those platforms return 'unsupported' anyway.
        handle = _vb.nativeHandles.isNotEmpty ? _vb.nativeHandles[0] : null;
        pixelFormat = _pixelFormatName();
    }

    return {
      'handle': handle,
      'width': _vb.width,
      'height': _vb.height,
      'pixelFormat': pixelFormat,
      'contentType': _contentType.name,
    };
  }

  String _pixelFormatName() {
    // Map MiniAVPixelFormat enum name to canonical lowercase string.
    return _vb.pixelFormat.name.toLowerCase();
  }
}
