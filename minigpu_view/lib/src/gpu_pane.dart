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

import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';
import 'package:minigpu/minigpu.dart';

import 'adapters/minigpu_adapter.dart';
import 'gpu_frame.dart';
import 'planar_convert.dart';
import 'preview_controller.dart';

/// One displayable pane: a destination texture, the controller that shows it,
/// and a pooled upload buffer for host-memory frames.
class GpuPane implements GpuPaneTarget {
  GpuPane(this._gpu, {this.debugLabel, this.waitForReaders}) {
    _deviceOwner = _gpu.deviceCoordinator.register(
      label: 'presentation ${debugLabel ?? "pane"}',
      fence: () {
        clear().ignore();
        dispose().ignore();
      },
      drain: dispose,
    );
  }

  late final GpuDeviceParticipant _deviceOwner;

  /// Additional trusted readers of this destination (for example a scoped
  /// capture readback). Device-owner disposal must join them too, even when
  /// the parent widget's own disposal has not reached this pane yet.
  final Future<void> Function()? waitForReaders;

  final Minigpu _gpu;

  /// Names the pane in diagnostics; presentation failures are otherwise
  /// anonymous when several panes are live.
  final String? debugLabel;

  /// Hand to `MiniavGpuPreview`.
  MinigpuPreviewController get controller => _controller;
  MinigpuPreviewController _controller = MinigpuPreviewController();

  SharedOutputTexture? _texture;

  /// Textures a resize replaced, each with the controller that registered it.
  /// They stay registered and alive until nothing on screen can still show
  /// them — see [_retireReplaced].
  final _replaced =
      <({SharedOutputTexture texture, MinigpuPreviewController controller})>[];
  Future<void> _drops = Future<void>.value();

  Buffer? _upload;
  Buffer? _scratch;
  Buffer? _config;
  int _uploadBytes = 0;
  int _scratchBytes = 0;
  int _configWords = 0;
  int _width = 0, _height = 0;

  GpuFrame? _pending;
  bool _busy = false;
  Future<void>? _active, _disposing;
  bool _disposed = false;
  int _generation = 0;
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

  /// One pipeline per actual device and epoch, shared by its panes. The default
  /// wrapper is a singleton; independent adapters must never share its shader.
  static final _converters =
      Expando<({int epoch, ComputeShader shader})>();

  @override
  ComputeShader planarConverter() {
    final cached = _converters[_gpu];
    if (cached?.epoch == _gpu.deviceEpoch) return cached!.shader;
    final shader = _gpu.createComputeShader()
      ..loadKernelString(kPlanarToRgbaWgsl);
    _converters[_gpu] = (epoch: _gpu.deviceEpoch, shader: shader);
    return shader;
  }

  /// Show planar or subsampled host bytes — NV12, I420, YUY2, BGRA — unpacked
  /// on the GPU. See [PlanarCpuGpuFrame] for the stride and colour caveats,
  /// all of which fail silently when wrong.
  Future<void> showPlanar({
    required PlanarFormat format,
    required List<PlanarPlane> planes,
    required int width,
    required int height,
    PlanarColorRange range = PlanarColorRange.limited,
    YuvMatrix matrix = YuvMatrix.bt709,
  }) => show(
    PlanarCpuGpuFrame(
      format: format,
      planes: planes,
      width: width,
      height: height,
      range: range,
      matrix: matrix,
    ),
  );

  // ── Convenience: the common producer shapes, without naming a frame type ──
  //
  // The GpuFrame types stay public for custom producers, but the ordinary
  // cases should not require learning them — getting a frame from a source to
  // a view is meant to be one call.

  /// Show packed RGBA8 bytes from host memory. Uploaded once, then blitted.
  Future<void> showBytes(
    Uint8List rgba, {
    required int width,
    required int height,
  }) => show(CpuBytesGpuFrame(rgba: rgba, width: width, height: height));

  /// Show a minigpu buffer of packed RGBA8 (one u32 per pixel, row major).
  Future<void> showBuffer(
    Buffer buffer, {
    required int width,
    required int height,
  }) => show(BufferGpuFrame(buffer: buffer, width: width, height: height));

  /// Show a platform shared texture handle (Windows: a D3D11 shared NT
  /// handle). Valid only for the duration of this call — see
  /// [SharedHandleGpuFrame].
  Future<void> showSharedHandle(
    int sharedHandle, {
    required int width,
    required int height,
    bool bgra = true,
  }) => show(
    SharedHandleGpuFrame(
      gpu: _gpu,
      sharedHandle: sharedHandle,
      width: width,
      height: height,
      bgra: bgra,
    ),
  );

  /// Mark and flush in one step — for callers with no pacing requirement.
  Future<void> show(GpuFrame frame) async {
    await showChecked(frame);
  }

  /// Attempts this frame now, without retaining it in a deferred queue. Returns
  /// true only when its host presentation completed while still current.
  /// A borrowed source is never held by a skipped call.
  Future<bool> showChecked(GpuFrame frame, {bool Function()? isCurrent}) {
    if (_disposed || _busy) return Future.value(false);
    final retired = Completer<void>();
    final result = Completer<bool>();
    _active = retired.future;
    _busy = true;
    () async {
      try {
        result.complete(await _show(frame, isCurrent));
      } catch (error, stack) {
        result.completeError(error, stack);
      } finally {
        _busy = false;
        _active = null;
        retired.complete();
      }
    }();
    return result.future;
  }

  Future<bool> _show(GpuFrame frame, bool Function()? isCurrent) async {
    final generation = _generation;
    bool current() {
      final allowed = isCurrent?.call() ?? true;
      return !_disposed && generation == _generation && allowed;
    }

    try {
      if (!current()) return false;
      if (!await _ensure(frame.width, frame.height)) {
        _report('no destination texture at ${frame.width}x${frame.height}');
        return false;
      }
      if (!current()) return false;
      if (!await frame.copyInto(this)) {
        _report('frame declined to copy');
        return false;
      }
      if (!current()) {
        _hasContent = false;
        await _controller.clear();
        await _dropReplaced();
        return false;
      }
      await _controller.present(_texture!.asPreviewSource());
      if (!current()) {
        _hasContent = false;
        await _controller.clear();
        await _dropReplaced();
        return false;
      }
      _hasContent = true;
      _retireReplaced().ignore();
      return true;
    } catch (error) {
      _hasContent = false;
      await _controller.clear();
      await _dropReplaced();
      _report('present rejected: $error');
      return false;
    }
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
    await showChecked(frame);
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

  Future<bool> _ensure(int width, int height) async {
    if (_disposed || width <= 0 || height <= 0) return false;
    if (_texture != null && _width == width && _height == height) return true;
    final previous = _texture;
    if (previous != null) {
      // NOT hidden or destroyed here. The view keeps painting it until a
      // frame showing its successor has been drawn, so a resize never puts an
      // empty frame on screen; a blank per resize strobes while a window edge
      // is dragged. The next successful present retires it.
      _replaced.add((texture: previous, controller: _controller));
      _texture = null;
    }
    _texture = _gpu.createSharedOutputTexture(width, height);
    _unsupported = _texture == null;
    _width = width;
    _height = height;
    return _texture != null;
  }

  /// Retires the textures the frame just presented supersedes, once a frame
  /// showing it has been drawn: the host stops painting a retired texture at
  /// once, so doing it sooner would put an empty frame on screen.
  Future<void> _retireReplaced() async {
    // Taken before waiting: a texture replaced after this call stays on
    // screen until its own successor is presented.
    final superseded = List.of(_replaced);
    if (superseded.isEmpty) return;
    // Waits while frames are off (app hidden). Dispose does not wait for it:
    // disposing the controller drops every registration itself.
    await SchedulerBinding.instance.endOfFrame;
    await _dropReplaced(superseded);
  }

  /// Unregisters and frees replaced textures: [only] those, or every one. Only
  /// once nothing on screen can still show them: after a frame without them,
  /// or a controller clear. One run at a time, so overlapping resizes never
  /// retire a texture twice.
  Future<void> _dropReplaced([
    List<({SharedOutputTexture texture, MinigpuPreviewController controller})>?
        only,
  ]) =>
      _drops = _drops.then((_) async {
        // An earlier run may have freed some of [only] already.
        final batch = [
          for (final old in only ?? _replaced)
            if (_replaced.contains(old)) old,
        ];
        if (batch.isEmpty || _disposed) return;
        try {
          for (final old in batch) {
            await old.controller.retire(old.texture.asPreviewSource());
          }
          if (_disposed) return; // dispose frees what is left
          await _gpu.waitForSubmittedWork();
          if (_disposed) return;
          for (final old in batch) {
            _replaced.remove(old);
            old.texture.destroy();
          }
        } catch (error) {
          // Unregistration or GPU completion not proven: keep them alive.
          _report('replaced texture kept: $error');
        }
      });

  /// Hide the view and fence any in-flight frame from presenting it again.
  /// Keeps allocations for another frame of the same admitted view.
  Future<void> clear() {
    _generation++;
    _hasContent = false;
    _pending = null;
    return _controller.clear();
  }

  /// Drops the platform registration and frees GPU resources.
  ///
  /// ORDER MATTERS: the controller unregisters the Flutter textures BEFORE the
  /// textures they point at are destroyed, and the host answers only once the
  /// engine has really dropped them (it removes a texture on the raster
  /// thread, after the request). Reversing it lets the compositor sample a
  /// freed resource.
  Future<void> dispose() {
    final pending = _disposing;
    if (pending != null) return pending;
    _disposed = true;
    _hasContent = false;
    _pending = null;
    final done = Completer<void>();
    _disposing = done.future;
    _deviceOwner.observeRetirement(done.future);
    () async {
      try {
        await waitForReaders?.call();
        await _active;
        if (_texture != null ||
            _replaced.isNotEmpty ||
            _upload != null ||
            _scratch != null ||
            _config != null) {
          await _gpu.retireSubmittedWork(deviceEpoch: _deviceOwner.deviceEpoch);
        }
        await _controller.dispose();
        // A replaced texture registered under an earlier controller is gone
        // only once that controller's own disposal is.
        for (final old in List.of(_replaced)) {
          await old.controller.retire(old.texture.asPreviewSource());
        }
        _texture?.destroy();
        _texture = null;
        for (final old in _replaced) {
          old.texture.destroy();
        }
        _replaced.clear();
        _upload?.destroy();
        _upload = null;
        _scratch?.destroy();
        _scratch = null;
        _config?.destroy();
        _config = null;
        done.complete();
      } catch (error, stack) {
        // Neither GPU completion nor host unregistration may be guessed. Keep
        // allocations alive if either retirement boundary cannot be proved.
        done.completeError(error, stack);
      }
    }();
    return done.future;
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
