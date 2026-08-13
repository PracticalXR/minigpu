/// A DISPLAYABLE PANE THE APP OWNS — destination texture, controller, and the
/// rule about when its pixels may change.
///
/// ── THE SHORT VERSION ───────────────────────────────────────────────────────
/// ```dart
/// final pane = GpuPane(gpu);
/// // ... whatever your producer gave you:
/// await pane.showSharedHandle(handle, width: w, height: h);
/// await pane.showBytes(rgba, width: w, height: h);
/// await pane.showBuffer(buf, width: w, height: h);
///
/// MiniavGpuPreview(controller: pane.controller)   // in the widget tree
/// ```
///
/// ── WHY OWN A TEXTURE AT ALL, RATHER THAN PRESENT THE PRODUCER'S ────────────
/// If the producer already has a shared texture, presenting it directly costs
/// nothing and is the right thing to do — use
/// `SharedOutputTexture.asPreviewSource()` and skip this class.
///
/// What that CANNOT give you is control over when the pane changes. A platform
/// `Texture` widget does not hold a snapshot: the compositor re-fetches the
/// descriptor on every composite that paints the layer, so a pane pointed at a
/// producer's texture shows whatever the producer last wrote, at the
/// producer's rate — regardless of how carefully presents are paced. Scrolling
/// the page is enough to resample it.
///
/// Owning the destination is what makes "this pane changes at most once per
/// [flush]" true rather than hopeful. Callers that need a bounded update rate
/// (a diagnostic view, a thumbnail wall, anything with an accessibility
/// constraint on motion) need this; callers that want the producer's rate do
/// not, and should not pay the blit.
///
/// ── THE CLOCK IS THE CALLER'S ───────────────────────────────────────────────
/// [markDirty] is cheap and safe to call from a capture callback; [flush] does
/// the work. This class never schedules its own presents, because the right
/// cadence is a policy only the caller knows — vsync for smooth motion, or a
/// fixed low rate for a view that must not track its source. A pane that
/// picked one would be wrong for the other.
library;

import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:minigpu/minigpu.dart';

import 'adapters/minigpu_adapter.dart';
import 'gpu_frame.dart';
import 'planar_convert.dart';
import 'preview_controller.dart';

/// One displayable pane: a destination texture, the controller that shows it,
/// and a pooled upload buffer for host-memory frames.
class GpuPane implements GpuPaneTarget {
  GpuPane(this._gpu, {this.debugLabel});

  final Minigpu _gpu;

  /// Names the pane in diagnostics; presentation failures are otherwise
  /// anonymous when several panes are live.
  final String? debugLabel;

  /// Hand to `MiniavGpuPreview`.
  MinigpuPreviewController get controller => _controller;
  MinigpuPreviewController _controller = MinigpuPreviewController();

  SharedOutputTexture? _texture;
  Buffer? _upload;
  Buffer? _scratch;
  Buffer? _config;
  int _uploadBytes = 0;
  int _scratchBytes = 0;
  int _configWords = 0;
  int _width = 0, _height = 0;

  GpuFrame? _pending;
  bool _busy = false;
  bool _disposed = false;
  bool _hasContent = false;

  /// True once a frame has landed. Until then a view should show a placeholder
  /// rather than an uninitialised texture.
  bool get hasContent => _hasContent;

  /// False when this platform has no cross-API shared texture, so nothing can
  /// be displayed through this class. Present the producer's own resource with
  /// `asPreviewSource()` instead.
  bool get isSupported => _unsupported == false;
  bool? _unsupported;

  @override
  SharedOutputTexture get texture => _texture!;

  @override
  Buffer uploadBuffer(int byteLength) {
    if (_upload == null || _uploadBytes < byteLength) {
      _upload?.destroy();
      _upload = _gpu.createBuffer(byteLength, BufferDataType.uint32);
      _uploadBytes = byteLength;
    }
    return _upload!;
  }

  @override
  Buffer scratchBuffer(int byteLength) {
    if (_scratch == null || _scratchBytes < byteLength) {
      _scratch?.destroy();
      _scratch = _gpu.createBuffer(byteLength, BufferDataType.uint32);
      _scratchBytes = byteLength;
    }
    return _scratch!;
  }

  @override
  Buffer configBuffer(int wordCount) {
    if (_config == null || _configWords < wordCount) {
      _config?.destroy();
      _config = _gpu.createBuffer(wordCount * 4, BufferDataType.uint32);
      _configWords = wordCount;
    }
    return _config!;
  }

  /// ONE pipeline for the process, not one per pane and not one per [Minigpu]
  /// WRAPPER.
  ///
  /// Compiling a shader costs seconds on backends that go through FXC, so a
  /// per-pane pipeline would make opening a second view stall the app.
  ///
  /// Keying a cache by the `Minigpu` object would have been worse than no
  /// cache: the native context is PROCESS-GLOBAL and the Dart object is just a
  /// handle to it, so an app that constructs a second wrapper — which is
  /// ordinary — would compile a second identical pipeline, and the map would
  /// pin every wrapper it ever saw. Both leak, and the compile is the
  /// expensive half.
  static ComputeShader? _converter;

  @override
  ComputeShader planarConverter() => _converter ??=
      (_gpu.createComputeShader()..loadKernelString(kPlanarToRgbaWgsl));

  /// Show planar or subsampled host bytes — NV12, I420, YUY2, BGRA — unpacked
  /// on the GPU. See [PlanarCpuGpuFrame] for the stride and colour-range
  /// caveats, both of which fail silently when wrong.
  Future<void> showPlanar({
    required PlanarFormat format,
    required List<PlanarPlane> planes,
    required int width,
    required int height,
    PlanarColorRange range = PlanarColorRange.limited,
  }) =>
      show(PlanarCpuGpuFrame(
        format: format,
        planes: planes,
        width: width,
        height: height,
        range: range,
      ));

  // ── Convenience: the common producer shapes, without naming a frame type ──
  //
  // The GpuFrame types stay public for custom producers, but the ordinary
  // cases should not require learning them — getting a frame from a source to
  // a view is meant to be one call.

  /// Show packed RGBA8 bytes from host memory. Uploaded once, then blitted.
  Future<void> showBytes(Uint8List rgba,
          {required int width, required int height}) =>
      show(CpuBytesGpuFrame(rgba: rgba, width: width, height: height));

  /// Show a minigpu buffer of packed RGBA8 (one u32 per pixel, row major).
  Future<void> showBuffer(Buffer buffer,
          {required int width, required int height}) =>
      show(BufferGpuFrame(buffer: buffer, width: width, height: height));

  /// Show a platform shared texture handle (Windows: a D3D11 shared NT
  /// handle). Valid only for the duration of this call — see
  /// [SharedHandleGpuFrame].
  Future<void> showSharedHandle(int sharedHandle,
          {required int width, required int height, bool bgra = true}) =>
      show(SharedHandleGpuFrame(
        gpu: _gpu,
        sharedHandle: sharedHandle,
        width: width,
        height: height,
        bgra: bgra,
      ));

  /// Mark and flush in one step — for callers with no pacing requirement.
  Future<void> show(GpuFrame frame) {
    markDirty(frame);
    return flush();
  }

  /// Records the newest frame without doing any GPU work. Cheap and
  /// synchronous, so it is safe from a capture callback.
  ///
  /// LATEST WINS: an unflushed frame is replaced rather than queued. A pane
  /// only shows the newest thing available at flush time, so a backlog would
  /// add latency and nothing else.
  void markDirty(GpuFrame frame) {
    if (_disposed) return;
    _pending = frame;
  }

  /// Copies the pending frame in and publishes it. Call from your own clock.
  ///
  /// Does nothing when no new frame is pending, and SKIPS while a previous
  /// copy is still running — the pane then keeps showing its last completed
  /// frame, which stops a slow GPU from letting presents pile up and land in a
  /// burst.
  Future<void> flush() async {
    if (_disposed || _busy) return;
    final frame = _pending;
    if (frame == null) return;
    _pending = null;
    if (!_ensure(frame.width, frame.height)) {
      _report('no destination texture at ${frame.width}x${frame.height} — '
          'the platform has none (isSupported=$isSupported) or creating it '
          'failed; a pane that never gets one stays empty forever');
      return;
    }
    _busy = true;
    try {
      if (!await frame.copyInto(this)) {
        _report('${frame.runtimeType} declined to copy — a stale/closed '
            'handle, a cross-adapter import, or a short buffer. The pane '
            'keeps its previous contents.');
        return;
      }
      if (_disposed) return;
      _hasContent = true;
      // The package's own adapter, which also handles the web case.
      await _controller.present(_texture!.asPreviewSource());
    } catch (e) {
      // A rejected present (not bindable on the compositor's adapter, or the
      // platform layer refusing the handle) leaves the pane on its previous
      // contents rather than blank.
      _report('present rejected: $e');
    } finally {
      _busy = false;
    }
  }

  final Set<String> _reported = <String>{};

  /// Reports a reason this pane is not showing what the caller expects.
  ///
  /// ONCE PER DISTINCT MESSAGE, because every one of these sits on a
  /// per-frame path and a repeating log is worse than none — it buries
  /// whatever else is being debugged.
  ///
  /// Silence here was a mistake worth naming: a pane that fails every frame
  /// and says nothing looks identical to a pane that was never fed, and the
  /// two have completely different causes.
  void _report(String message) {
    if (!_reported.add(message)) return;
    final tag = debugLabel == null ? 'GpuPane' : 'GpuPane $debugLabel';
    debugPrint('[$tag] $message');
  }

  bool _ensure(int width, int height) {
    if (_disposed || width <= 0 || height <= 0) return false;
    if (_texture != null && _width == width && _height == height) return true;
    _texture?.destroy();
    _texture = _gpu.createSharedOutputTexture(width, height);
    _unsupported = _texture == null;
    _width = width;
    _height = height;
    _hasContent = false;
    return _texture != null;
  }

  /// Drops the platform registration and frees GPU resources.
  ///
  /// ORDER MATTERS: the controller unregisters the Flutter texture — an async
  /// channel round trip — BEFORE the texture it points at is destroyed.
  /// Reversing it lets the compositor sample a freed resource.
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    _pending = null;
    await _controller.dispose();
    _texture?.destroy();
    _texture = null;
    _upload?.destroy();
    _upload = null;
    _scratch?.destroy();
    _scratch = null;
    _config?.destroy();
    _config = null;
  }

  /// Replaces the controller after something disposed it — a Flutter hot
  /// reload teardown, typically. `MinigpuPreviewController.dispose` is
  /// terminal: `present` throws afterwards, so a pane that outlives its
  /// controller needs a fresh one.
  void resetController() {
    if (_disposed) return;
    _controller = MinigpuPreviewController();
    _hasContent = false;
  }
}
