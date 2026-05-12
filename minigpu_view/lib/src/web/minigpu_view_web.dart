/// Web plugin for minigpu_view.
///
/// Implements the `minigpu_view` method channel on the browser by:
///   1. Registering an `<canvas>` element per controller via
///      `ui_web.platformViewRegistry`.
///   2. On each `present` call, blitting the producer's `GPUTexture`
///      into the canvas's `GPUCanvasContext` on the **same** WebGPU
///      device — one command, zero PCIe traffic.
///
/// Producers (e.g. `minigpu_web`) MUST set the JS global
/// `window.gpuDevice` to the WebGPU device they're using before any
/// `present` call. The web plugin reads that same global so source
/// and sink share a queue.
library minigpu_view_web;

import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:ui_web' as ui_web;

import 'package:flutter/services.dart';
import 'package:flutter_web_plugins/flutter_web_plugins.dart';
import 'package:web/web.dart' as web;

/// Plugin entry point. Registered automatically by Flutter via the
/// `pluginClass` declaration in `pubspec.yaml`.
class MinigpuViewWebPlugin {
  static void registerWith(Registrar registrar) {
    final channel = MethodChannel(
      'minigpu_view',
      const StandardMethodCodec(),
      registrar,
    );
    final plugin = MinigpuViewWebPlugin._();
    channel.setMethodCallHandler(plugin._handle);
  }

  MinigpuViewWebPlugin._();

  // Per-instance canvas + its GPUCanvasContext.
  final Map<int, _CanvasSink> _sinks = {};

  // Synthesized texture-id space (does not map to a real Flutter
  // texture id; the widget uses HtmlElementView with the same id as
  // the registered view-type).
  int _nextTextureId = 1000;

  Future<dynamic> _handle(MethodCall call) async {
    final args = (call.arguments as Map).cast<String, Object?>();
    final instanceId = args['instanceId'] as int;

    switch (call.method) {
      case 'present':
        return _present(instanceId, args);
      case 'dispose':
        _sinks.remove(instanceId)?.dispose();
        return null;
      default:
        throw MissingPluginException('minigpu_view_web: ${call.method}');
    }
  }

  Map<String, Object?> _present(int instanceId, Map<String, Object?> args) {
    final kind = args['kind'] as String;
    final width = (args['width'] as num).toInt();
    final height = (args['height'] as num).toInt();

    var sink = _sinks[instanceId];
    sink ??= _registerSink(instanceId, width, height);

    if (kind == 'webGpuTexture') {
      final tex = args['texture'];
      if (tex is! JSObject) {
        throw PlatformException(
          code: 'invalid_handle',
          message: 'Expected JSObject GPUTexture under "texture" key',
        );
      }
      sink.copyFromGpuTexture(tex, width, height);
    } else if (kind == 'webVideoFrame') {
      final frame = args['frame'];
      if (frame is! JSObject) {
        throw PlatformException(
          code: 'invalid_handle',
          message: 'Expected JSObject VideoFrame under "frame" key',
        );
      }
      sink.copyFromVideoFrame(frame, width, height);
    } else {
      throw PlatformException(
        code: 'unsupported',
        message: 'Web plugin does not support kind=$kind',
      );
    }

    return {
      'textureId': sink.textureId,
      'width': width,
      'height': height,
      'presentedAtUs': DateTime.now().microsecondsSinceEpoch,
    };
  }

  _CanvasSink _registerSink(int instanceId, int width, int height) {
    final id = _nextTextureId++;
    final canvas = web.HTMLCanvasElement()
      ..width = width
      ..height = height
      ..style.width = '100%'
      ..style.height = '100%';

    final sink = _CanvasSink(textureId: id, canvas: canvas);

    // The view-type string MUST match what MiniavGpuPreview wraps. We
    // expose `minigpu_view::<id>` and the widget will switch on web
    // platforms to use HtmlElementView with the same string.
    ui_web.platformViewRegistry.registerViewFactory(
      'minigpu_view::$id',
      (int viewId) => canvas,
    );

    _sinks[instanceId] = sink;
    return sink;
  }
}

class _CanvasSink {
  final int textureId;
  final web.HTMLCanvasElement canvas;

  // Lazily configured on first frame, once we know the device.
  JSObject? _ctx;
  JSObject? _device;
  bool _disposed = false;

  _CanvasSink({required this.textureId, required this.canvas});

  /// Resolve the WebGPU device the producer is using. Producers set
  /// `window.gpuDevice = device` before any present() call.
  JSObject? _resolveDevice() {
    final globalThis = web.window as JSObject;
    final dev = globalThis['gpuDevice'];
    if (dev is JSObject) return dev;
    return null;
  }

  void _ensureContext(JSObject device) {
    if (_ctx != null && identical(_device, device)) return;
    final ctx = canvas.getContext('webgpu');
    if (ctx == null) {
      throw PlatformException(
        code: 'no_webgpu_canvas',
        message: 'canvas.getContext("webgpu") returned null',
      );
    }
    final ctxJs = ctx;

    // Pick the preferred format for the canvas.
    final navGpu = (web.window as JSObject)
        .getProperty<JSObject?>('navigator'.toJS)
        ?.getProperty<JSObject?>('gpu'.toJS);
    final format =
        navGpu?.callMethod<JSString>('getPreferredCanvasFormat'.toJS).toDart ??
        'bgra8unorm';

    ctxJs.callMethod<JSAny?>(
      'configure'.toJS,
      {'device': device, 'format': format, 'alphaMode': 'opaque'}.jsify(),
    );

    _ctx = ctxJs;
    _device = device;
  }

  /// Blit a producer GPUTexture into the canvas via
  /// `commandEncoder.copyTextureToTexture` on the same device.
  void copyFromGpuTexture(JSObject src, int width, int height) {
    if (_disposed) return;
    final device = _resolveDevice();
    if (device == null) {
      throw PlatformException(
        code: 'no_device',
        message: 'window.gpuDevice not set; producer must publish it',
      );
    }
    _ensureContext(device);

    if (canvas.width != width) canvas.width = width;
    if (canvas.height != height) canvas.height = height;

    final dst = _ctx!.callMethod<JSObject>('getCurrentTexture'.toJS);
    final encoder = device.callMethod<JSObject>('createCommandEncoder'.toJS);

    encoder.callMethod<JSAny?>(
      'copyTextureToTexture'.toJS,
      {'texture': src}.jsify(),
      {'texture': dst}.jsify(),
      {'width': width, 'height': height, 'depthOrArrayLayers': 1}.jsify(),
    );

    final cmd = encoder.callMethod<JSObject>('finish'.toJS);
    final queue = device['queue'] as JSObject;
    queue.callMethod<JSAny?>('submit'.toJS, [cmd].jsify());
  }

  /// Fast path for WebCodecs: draw a VideoFrame directly into the
  /// canvas. Browsers special-case this as a GPU blit on capable
  /// hardware.
  void copyFromVideoFrame(JSObject frame, int width, int height) {
    if (_disposed) return;
    if (canvas.width != width) canvas.width = width;
    if (canvas.height != height) canvas.height = height;

    // Use 2D context for VideoFrame fast path; the browser will keep
    // it on the GPU when possible.
    final ctx2d = canvas.getContext('2d');
    if (ctx2d == null) return;
    ctx2d.callMethod<JSAny?>('drawImage'.toJS, frame, 0.toJS, 0.toJS);
  }

  void dispose() {
    _disposed = true;
    canvas.remove();
  }
}
