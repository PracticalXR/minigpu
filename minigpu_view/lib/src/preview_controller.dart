/// Imperative controller that pushes [PreviewSource] frames to the
/// platform plugin and exposes a Flutter [Texture] id.
library;

import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/services.dart';

import 'exceptions.dart';
import 'preview_source.dart';

/// Listenable controller backing a [MiniavGpuPreview] widget.
///
/// Call [present] each time the producer (minigpu/miniav) has produced a
/// new frame. The controller serialises the frame metadata across the
/// platform method channel; the host plugin sees the GPU handle and
/// hands it to Flutter's texture registry.
///
/// The controller is a [ChangeNotifier] — the widget rebuilds when
/// [textureId] becomes available or [size] changes.
class MinigpuPreviewController extends ChangeNotifier {
  static const MethodChannel _channel = MethodChannel('minigpu_view');

  final int _instanceId;
  static int _nextInstanceId = 1;

  int? _textureId;
  Size _size = Size.zero;
  bool _disposed = false;
  bool _registrationAttempted = false;
  bool _notificationPending = false;
  final _pending = <Future<void>>{};
  Future<void>? _clearing, _disposing;

  final StreamController<int> _presentedAtUs = StreamController.broadcast();

  MinigpuPreviewController() : _instanceId = _nextInstanceId++ {
    // Lazy: the host registers a texture only on first present().
  }

  /// Flutter texture id, or null if no frame has been presented yet
  /// (or if the host plugin couldn't allocate a registry slot).
  int? get textureId => _textureId;

  /// Natural size of the most recent frame.
  Size get size => _size;

  /// Stream of host-side acknowledgement timestamps (microseconds since
  /// Unix epoch). These can pace host submissions, but do not acknowledge
  /// raster/GPU-read completion or authorize resource reuse/destruction.
  Stream<int> get presentedAtUs => _presentedAtUs.stream;

  /// Push a new GPU frame.
  ///
  /// Returns when the host has registered the frame with Flutter's
  /// texture registry. May throw [UnsupportedPreviewException] on
  /// platforms whose surface API isn't yet wired, or
  /// [PreviewPresentException] when the host rejects the handle.
  Future<void> present(PreviewSource source) {
    if (_disposed || _clearing != null) {
      return Future.error(StateError('Preview controller is retiring'));
    }
    final done = Completer<void>();
    _pending.add(done.future);
    () async {
      try {
        await _present(source);
        done.complete();
      } catch (error, stack) {
        done.completeError(error, stack);
      } finally {
        _pending.remove(done.future);
      }
    }();
    return done.future;
  }

  Future<void> _present(PreviewSource source) async {
    final args = <String, Object?>{
      'instanceId': _instanceId,
      'kind': source.kind.name,
      ...source.toChannelMessage(),
    };
    try {
      _registrationAttempted = true;
      final result = await _channel.invokeMapMethod<String, Object?>(
        'present',
        args,
      );
      if (_disposed || _clearing != null) return;
      if (result == null) {
        throw const PreviewPresentException('host returned null');
      }
      final id = (result['textureId'] as num?)?.toInt();
      final w = (result['width'] as num?)?.toDouble() ?? source.size.width;
      final h = (result['height'] as num?)?.toDouble() ?? source.size.height;
      final ts = (result['presentedAtUs'] as num?)?.toInt();

      var changed = false;
      if (id != null && id != _textureId) {
        _textureId = id;
        changed = true;
      }
      final newSize = Size(w, h);
      if (newSize != _size) {
        _size = newSize;
        changed = true;
      }
      if (ts != null && !_presentedAtUs.isClosed) {
        _presentedAtUs.add(ts);
      }
      if (changed) _notifyViewChanged();
    } on MissingPluginException catch (e) {
      throw UnsupportedPreviewException(
        'minigpu_view plugin not registered for this platform: ${e.message}',
      );
    } on PlatformException catch (e) {
      if (e.code == 'unsupported') {
        throw UnsupportedPreviewException(e.message ?? 'unsupported source');
      }
      throw PreviewPresentException(e.message ?? 'host error', cause: e);
    }
  }

  /// Hide immediately, then retire the registered surface after any host
  /// requests already in flight. The same controller can present again later.
  Future<void> clear() {
    if (_disposed) return _disposing ?? Future<void>.value();
    final pending = _clearing;
    if (pending != null) return pending;
    final done = Completer<void>();
    _clearing = done.future;
    _textureId = null;
    _size = Size.zero;
    _notifyViewChanged();
    () async {
      try {
        await _releaseRegistration();
        done.complete();
      } catch (error, stack) {
        done.completeError(error, stack);
      } finally {
        _clearing = null;
      }
    }();
    return done.future;
  }

  /// Drops the host registration of [source], a frame this controller
  /// presented earlier and has since replaced with a different source (a
  /// resize, say). Completes once the engine has let go of it, so the
  /// producer texture behind [source] may then be destroyed.
  ///
  /// The host stops painting [source] at once, so call this only after a
  /// frame showing its replacement has been drawn, and never for the source
  /// presented last.
  Future<void> retire(PreviewSource source) async {
    // Clear and dispose drop every registration of this instance, [source]'s
    // included, and complete once the host has.
    if (_disposed) {
      await _disposing;
      return;
    }
    final clearing = _clearing;
    if (clearing != null) {
      await clearing;
      return;
    }
    if (!_registrationAttempted) return;
    try {
      await _channel.invokeMethod<void>('retire', {
        'instanceId': _instanceId,
        'kind': source.kind.name,
        ...source.toChannelMessage(),
      });
    } on MissingPluginException {
      // A host that keeps one registration per instance has none to drop.
    }
  }

  void _notifyViewChanged() {
    if (_disposed || _notificationPending) return;
    final scheduler = SchedulerBinding.instance;
    if (scheduler.schedulerPhase == SchedulerPhase.persistentCallbacks) {
      // Retirement can be triggered by a new view during build. State and
      // platform retirement change immediately; only widget notification waits.
      _notificationPending = true;
      scheduler.addPostFrameCallback((_) {
        _notificationPending = false;
        if (!_disposed) notifyListeners();
      });
    } else {
      notifyListeners();
    }
  }

  Future<void> _releaseRegistration() async {
    // A failed request can still have registered a host texture. Unregister by
    // instance even when no successful response provided a texture id.
    await Future.wait([
      for (final work in _pending.toList())
        work.then<void>((_) {}, onError: (Object _, StackTrace __) {}),
    ]);
    if (!_registrationAttempted) return;
    await _channel.invokeMethod<void>('dispose', {'instanceId': _instanceId});
    _registrationAttempted = false;
  }

  @override
  Future<void> dispose() {
    final pending = _disposing;
    if (pending != null) return pending;
    _disposed = true;
    _textureId = null;
    _size = Size.zero;
    super.dispose();
    // Paused listeners must not hold GPU registration teardown indefinitely.
    _presentedAtUs.close().ignore();
    final done = Completer<void>();
    _disposing = done.future;
    () async {
      try {
        await _clearing;
        await _releaseRegistration();
        done.complete();
      } catch (error, stack) {
        done.completeError(error, stack);
      }
    }();
    return done.future;
  }
}
