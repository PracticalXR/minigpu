/// Abstract source of GPU frames that can be presented by [MiniavGpuPreview].
///
/// Concrete adapters live alongside their producer:
///   - [SharedOutputTexture.asPreviewSource] (minigpu)
///   - WebVideoFrame adapter (web only)
///   - MiniAVBuffer adapter (planned)
library;

import 'dart:ui' show Size;

import 'package:meta/meta.dart';

/// Discriminator for [PreviewSource] subtypes — used by the platform
/// plugin to dispatch to the correct host-side handler.
enum PreviewSourceKind {
  /// A native shared GPU texture (D3D11/Metal/Vulkan). Carries a raw
  /// platform handle (int for D3D11/AHardwareBuffer; opaque for others).
  nativeSharedTexture,

  /// A WebGPU `GPUTexture` JS object. Web only.
  webGpuTexture,

  /// A WebCodecs `VideoFrame` JS object. Web only.
  webVideoFrame,

  /// CPU fallback — RGBA bytes. Used when no zero-copy path is available.
  cpu,
}

/// A presentable frame source. Lifetime belongs to the producer; registration
/// borrows the GPU resource. Replacing a frame or disposing a view can retire
/// native presentation asynchronously. The producer must retain the resource
/// through the actual consumer retirement, not just the host's present reply.
@immutable
abstract class PreviewSource {
  const PreviewSource();

  /// What kind of underlying resource this source carries.
  PreviewSourceKind get kind;

  /// Natural size of the source in pixels.
  Size get size;

  /// Serialises into the wire format used by the method channel.
  ///
  /// For native handles this is `{handle: int, width: int, height: int}`.
  /// For web sources it returns the raw JS object under `'frame'`.
  /// Implementations are free to add backend-specific keys.
  Map<String, Object?> toChannelMessage();

  /// True if the producer guarantees this source remains valid until
  /// the next presentation. This declaration supplies no ownership transfer
  /// or GPU-read completion. [PreviewController.presentedAtUs] acknowledges
  /// host registration only; it does not authorize overwriting, reusing or
  /// destroying a borrowed native resource.
  bool get isOwned => false;
}
