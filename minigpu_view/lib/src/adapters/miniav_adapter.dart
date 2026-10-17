/// Adapter: `MiniAVVideoBuffer.asPreviewSource()`
///
/// Provides a zero-copy [PreviewSource] for GPU-backed [MiniAVBuffer]s whose
/// handle a host plugin can present as it stands.
///
/// WINDOWS BUFFERS ARE NOT ONE OF THEM, and return null. A
/// [MiniAVBufferContentType.gpuD3D11Handle] buffer carries a shared NT HANDLE,
/// not an `ID3D11Texture2D*`, and miniAV closes it when the buffer is
/// released, often handing out a new one per frame. A preview registered on it
/// would outlive it. Show those with `MiniAVBuffer.asGpuFrame()` and a
/// `GpuPane`, which copies the frame into a texture the pane owns before the
/// buffer is released.
///
/// CPU buffers are NOT wrapped by this adapter — they return null, because a
/// PreviewSource must reference a GPU resource and a host buffer is not one.
///
/// THAT IS NOT A REASON TO DECODE AN IMAGE. Use `MiniAVBuffer.asGpuFrame()`
/// with a `GpuPane`: host planes (NV12, I420, YUY2, BGRA) are uploaded and
/// unpacked by a compute shader, which is the same display path as everything
/// else. Converting them per-pixel on the isolate that builds the UI is
/// O(pixels) inside the capture callback and stalls the app before it looks
/// slow.
library;

import 'dart:ui' show Size;

import 'package:miniav/miniav.dart'
    show MiniAVBuffer, MiniAVBufferContentType, MiniAVVideoBuffer;

import '../preview_source.dart';

/// Adapts a GPU-backed [MiniAVBuffer] into a [PreviewSource].
///
/// Returns null for CPU buffers, Windows D3D11 buffers and unsupported content
/// types — show those with `asGpuFrame()` and a `GpuPane`.
extension MiniavBufferPreviewSourceAdapter on MiniAVBuffer {
  /// Wrap a GPU-backed video buffer as a [PreviewSource].
  ///
  /// Returns `null` if:
  /// - [contentType] is [MiniAVBufferContentType.cpu]
  /// - [contentType] is [MiniAVBufferContentType.gpuD3D11Handle] (see the
  ///   library comment: use `asGpuFrame()` with a `GpuPane`)
  /// - [data] is not a [MiniAVVideoBuffer]
  /// - The native handle is absent, zero, or not an integer
  PreviewSource? asPreviewSource() {
    if (contentType == MiniAVBufferContentType.cpu) return null;
    // The Windows plugin would take this NT HANDLE for a texture pointer and
    // AddRef a kernel handle value.
    if (contentType == MiniAVBufferContentType.gpuD3D11Handle) return null;
    final vb = data;
    if (vb is! MiniAVVideoBuffer) return null;
    final handle = vb.nativeHandles.isEmpty ? null : vb.nativeHandles[0];
    if (handle is! int || handle == 0) return null;
    return _MiniAVPreviewSource(contentType, vb, handle);
  }
}

class _MiniAVPreviewSource extends PreviewSource {
  final MiniAVBufferContentType _contentType;
  final MiniAVVideoBuffer _vb;
  final int _handle;

  const _MiniAVPreviewSource(this._contentType, this._vb, this._handle);

  @override
  PreviewSourceKind get kind => PreviewSourceKind.nativeSharedTexture;

  @override
  Size get size => Size(_vb.width.toDouble(), _vb.height.toDouble());

  @override
  Map<String, Object?> toChannelMessage() {
    // gpuMetalTexture / gpuAHardwareBuffer — the stubs for those platforms
    // return 'unsupported' anyway.
    return {
      'handle': _handle,
      'width': _vb.width,
      'height': _vb.height,
      'pixelFormat': _pixelFormatName(),
      'contentType': _contentType.name,
    };
  }

  String _pixelFormatName() {
    // Map MiniAVPixelFormat enum name to canonical lowercase string.
    return _vb.pixelFormat.name.toLowerCase();
  }
}
