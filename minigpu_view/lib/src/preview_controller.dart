/// Imperative controller that pushes [PreviewSource] frames to the
/// platform plugin and exposes a Flutter [Texture] id.
library;

import 'dart:async';

import 'package:flutter/foundation.dart';
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
  /// Unix epoch). Producers can use this for backpressure.
  Stream<int> get presentedAtUs => _presentedAtUs.stream;

  /// Push a new GPU frame.
  ///
  /// Returns when the host has registered the frame with Flutter's
  /// texture registry. May throw [UnsupportedPreviewException] on
  /// platforms whose surface API isn't yet wired, or
  /// [PreviewPresentException] when the host rejects the handle.
  Future<void> present(PreviewSource source) async {
    if (_disposed) {
      throw StateError('MinigpuPreviewController used after dispose()');
    }
    final args = <String, Object?>{
      'instanceId': _instanceId,
      'kind': source.kind.name,
      ...source.toChannelMessage(),
    };
    try {
      final result = await _channel.invokeMapMethod<String, Object?>(
        'present',
        args,
      );
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
      if (changed) notifyListeners();
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

  @override
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    super.dispose();
    await _presentedAtUs.close();
    if (_textureId != null) {
      try {
        await _channel.invokeMethod<void>('dispose', {
          'instanceId': _instanceId,
        });
      } catch (_) {
        // Plugin may already be torn down — swallow.
      }
    }
  }
}
