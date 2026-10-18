/// Web plugin for minigpu_view.
///
/// Implements the `minigpu_view` method channel on the browser by:
///   1. Registering an `<canvas>` element per controller via
///      `ui_web.platformViewRegistry`.
///   2. On each `present` call, drawing the producer's GPU buffer
///      into the canvas's `GPUCanvasContext` on the **same** WebGPU
///      device — one render pass, zero PCIe traffic.
///
/// Producers (e.g. `minigpu_web`) MUST set the JS global
/// `window.gpuDevice` to the WebGPU device they're using before any
/// `present` call. The web plugin reads that same global so source
/// and sink share a queue.
library minigpu_view_web;

// 🔴 `.isA<JSObject>()`, NOT `is JSObject`.
//
// dart2wasm erases every interop type to one opaque JS value, so an `is` test
// against a JS interop type cannot mean anything there and the compiler refuses
// it (`invalid_runtime_check_with_js_interop_types`). dart2js could answer it
// because interop types are real JS objects there — which is why this compiled
// for years and only surfaced when a wasm build first pulled this library in.

import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';
import 'dart:ui_web' as ui_web;

import 'package:flutter/services.dart';
import 'package:flutter_web_plugins/flutter_web_plugins.dart';
import 'package:web/web.dart' as web;

// ---------------------------------------------------------------------------
// Emscripten WebGPU handle lookup — must NOT have @JS('Module') in scope.
// ---------------------------------------------------------------------------

/// Emscripten's global WebGPU helper object (window.WebGPU).
@JS('WebGPU')
extension type _EmscriptenWebGpu._(JSObject _) implements JSObject {
  external JSAny? getJsObject(JSNumber ptr);
}

@JS('WebGPU')
external _EmscriptenWebGpu? get _emscriptenWebGpu;

/// Resolve an Emscripten integer handle to the corresponding JS WebGPU object.
/// Returns null if [handle] is 0 or the WASM module is not yet loaded.
JSObject? _webGpuGetJsObject(int handle) {
  if (handle == 0) return null;
  try {
    final result = _emscriptenWebGpu?.getJsObject(handle.toJS);
    return (result != null && result.isA<JSObject>())
        ? result as JSObject
        : null;
  } catch (_) {
    return null;
  }
}

/// The wasm instance `minigpu_web` publishes (its bindings read only this
/// name), or null on a page without one.
@JS('MinigpuModule')
external JSObject? get _minigpuModule;

/// Submits the compute work minigpu still holds in its open batch, so a blit
/// submitted next on the shared queue runs after the dispatches that wrote its
/// source buffer. Nothing batched (or no minigpu) submits nothing.
void _flushMinigpuBatch() {
  final module = _minigpuModule;
  if (module == null || !module.has('_mgpuFlushBatch')) return;
  module.callMethod<JSAny?>('_mgpuFlushBatch'.toJS);
}

@JS('globalThis')
external JSObject get _globalThis;

/// Shared global registry of pending WebCodecs `VideoFrame`s, keyed by a
/// producer-assigned int handle. A raw JS `VideoFrame` cannot cross the
/// method-channel `StandardMessageCodec`, so producers (e.g. `miniav_player`)
/// stash the frame here and pass only the codec-safe handle — mirroring the
/// Emscripten `bufferHandle` path. Pops (reads + removes) the frame.
JSObject? _takeVideoFrame(int handle) {
  final reg = _globalThis.getProperty<JSAny?>('miniavVideoFrameRegistry'.toJS);
  if (reg == null || reg.isUndefined) return null;
  final regObj = reg as JSObject;
  final key = handle.toString().toJS;
  final frame = regObj.getProperty<JSAny?>(key);
  if (frame == null || frame.isUndefined) return null;
  regObj.delete(key);
  return frame as JSObject;
}

// ---------------------------------------------------------------------------

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
      case 'retire':
        // One canvas per instance, and no presented source is referenced
        // after its present: nothing to drop.
        return null;
      default:
        throw MissingPluginException('minigpu_view_web: ${call.method}');
    }
  }

  /// One-shot marker for the silent-blank failure mode below.
  static bool _saidBufferLookupMiss = false;

  Map<String, Object?> _present(int instanceId, Map<String, Object?> args) {
    final kind = args['kind'] as String;
    final width = (args['width'] as num).toInt();
    final height = (args['height'] as num).toInt();

    var sink = _sinks[instanceId];
    sink ??= _registerSink(instanceId, width, height);

    if (kind == 'webGpuTexture') {
      // 'bufferHandle' key: an Emscripten integer handle — we resolve the
      //   JS GPUBuffer locally via WebGPU.getJsObject() to stay codec-safe
      //   (a JS object cannot cross the method channel's codec).
      final bufferHandle = args['bufferHandle'];
      if (bufferHandle is int && bufferHandle != 0) {
        final jsBuffer = _webGpuGetJsObject(bufferHandle);
        if (jsBuffer != null) {
          final format = args['format'] as String? ?? 'rgba32float';
          sink.copyFromGpuBuffer(jsBuffer, width, height, format);
        } else if (!_saidBufferLookupMiss) {
          // One frame of "not ready yet" is normal; EVERY frame missing means
          // the handle table (`globalThis.WebGPU`) is absent or belongs to a
          // different module instance — the present "succeeds" but nothing is
          // ever blitted, which renders as a permanently blank surface with
          // zero errors. Say it once.
          _saidBufferLookupMiss = true;
          // ignore: avoid_print
          print(
            '[minigpu_view] WebGPU.getJsObject($bufferHandle) returned '
            'null — GPU-buffer presents are being skipped; if this frame is '
            'not the first, the handle table and the compute module do not '
            'match',
          );
        }
      }
      // else: handle is 0/null → first frame, skip blit silently.
    } else if (kind == 'webVideoFrame') {
      // Codec-safe path: an int handle into the shared global VideoFrame
      // registry (a raw JS VideoFrame can't cross StandardMessageCodec). The
      // legacy same-isolate 'frame' JSObject is still honored if present.
      final handle = args['videoFrameHandle'];
      JSObject? frame;
      if (handle is int) {
        frame = _takeVideoFrame(handle);
      } else if (args['frame'] case final JSAny f when f.isA<JSObject>()) {
        frame = f as JSObject;
      }
      // Registry miss (handle already consumed / frame not ready) → skip
      // silently, like the WebGPU-not-ready path above.
      if (frame != null) sink.copyFromVideoFrame(frame, width, height);
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
      ..style.height = '100%'
          // This surface only displays producer frames.  Flutter owns all input
          // above it (scene capture, mode tabs, and controls), so the DOM canvas
          // must never become the browser's pointer target.
          ..style
          .setProperty('pointer-events', 'none');

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
  String? _canvasFormat;
  bool _disposed = false;

  // Two bounded direct pipelines per device/canvas format. The producer
  // owns pixel storage; the sink owns only one reusable size uniform.
  final _bufferPipelines = <String, JSObject>{};
  JSObject? _bufferDevice;
  String? _bufferCanvasFormat;
  JSObject? _bufferParamBuf;

  _CanvasSink({required this.textureId, required this.canvas});

  /// Resolve the WebGPU device the producer is using. Producers set
  /// `window.gpuDevice = device` before any present() call.
  JSObject? _resolveDevice() {
    final globalThis = web.window as JSObject;
    final dev = globalThis['gpuDevice'];
    if (dev != null && dev.isA<JSObject>()) return dev as JSObject;
    return null;
  }

  void _ensureContext(JSObject device) {
    // JS identity: dart2wasm wraps each JS value read into Dart afresh, so
    // identical() never holds and the canvas would be reconfigured per frame.
    if (_ctx != null && _device.strictEquals(device).toDart) return;
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
    _canvasFormat = format;
  }

  /// Read tightly packed producer pixels directly from a storage buffer.
  /// Float output has no implicit row padding: aligning a copy layout without
  /// repacking the source breaks narrow surfaces. This single render pass
  /// also avoids a temporary texture and CPU pixel readback.
  void copyFromGpuBuffer(JSObject src, int width, int height, String format) {
    if (_disposed) return;
    if (format != 'rgba8' && format != 'rgba32float') {
      throw PlatformException(
        code: 'unsupported_buffer_format',
        message: 'Unsupported GPU buffer pixel format: $format',
      );
    }
    final bytesPerPixel = format == 'rgba8' ? 4 : 16;
    final sourceBytes = src.getProperty<JSNumber>('size'.toJS).toDartInt;
    if (width <= 0 ||
        height <= 0 ||
        width * height * bytesPerPixel > sourceBytes) {
      throw PlatformException(
        code: 'invalid_buffer_extent',
        message: 'GPU buffer does not contain the declared pixel extent',
      );
    }
    final device = _resolveDevice();
    if (device == null) return;
    _ensureContext(device);
    if (canvas.width != width) canvas.width = width;
    if (canvas.height != height) canvas.height = height;
    final canvasFormat = _canvasFormat ?? 'bgra8unorm';
    final pipeline = _ensureBufferPipeline(device, canvasFormat, format);
    final queue = device['queue'] as JSObject;
    final params = Uint32List.fromList([width, height, 0, 0]);
    queue.callMethodVarArgs<JSAny?>('writeBuffer'.toJS, [
      _bufferParamBuf!,
      0.toJS,
      params.buffer.toJS,
    ]);
    final bindGroup = device.callMethod<JSObject>(
      'createBindGroup'.toJS,
      {
        'layout': pipeline.callMethod<JSObject>(
          'getBindGroupLayout'.toJS,
          0.toJS,
        ),
        'entries': [
          <String, Object?>{
            'binding': 0,
            'resource': {'buffer': src},
          },
          <String, Object?>{
            'binding': 1,
            'resource': {'buffer': _bufferParamBuf},
          },
        ],
      }.jsify(),
    );
    final encoder = device.callMethod<JSObject>('createCommandEncoder'.toJS);
    final dstView = _ctx!
        .callMethod<JSObject>('getCurrentTexture'.toJS)
        .callMethod<JSObject>('createView'.toJS);
    final renderPass = encoder.callMethod<JSObject>(
      'beginRenderPass'.toJS,
      {
        'colorAttachments': [
          <String, Object?>{
            'view': dstView,
            'loadOp': 'clear',
            'storeOp': 'store',
            'clearValue': {'r': 0.0, 'g': 0.0, 'b': 0.0, 'a': 1.0},
          },
        ],
      }.jsify(),
    );
    renderPass.callMethod<JSAny?>('setPipeline'.toJS, pipeline);
    renderPass.callMethod<JSAny?>('setBindGroup'.toJS, 0.toJS, bindGroup);
    renderPass.callMethod<JSAny?>('draw'.toJS, 3.toJS);
    renderPass.callMethod<JSAny?>('end'.toJS);
    // The producer's dispatches into src may still wait in minigpu's batch;
    // submitted after this pass, they would leave the canvas a frame behind.
    _flushMinigpuBatch();
    queue.callMethod<JSAny?>(
      'submit'.toJS,
      [encoder.callMethod<JSObject>('finish'.toJS)].jsify(),
    );
  }

  // Only the two host-declared pixel formats enter this shader template.
  static String _bufferBlitShader(String format) {
    final texelType = format == 'rgba8' ? 'u32' : 'vec4f';
    final pixel = format == 'rgba8'
        ? 'unpack4x8unorm(src[y * p.w + x])'
        : 'src[y * p.w + x]';
    return '''
@group(0) @binding(0) var<storage, read> src: array<$texelType>;
struct Params { w: u32, h: u32 };
@group(0) @binding(1) var<uniform> p: Params;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0),
  );
  return vec4f(pos[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let x = u32(pos.x);
  let y = u32(pos.y);
  if (x >= p.w || y >= p.h) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  return $pixel;
}
''';
  }

  JSObject _ensureBufferPipeline(
    JSObject device,
    String canvasFormat,
    String format,
  ) {
    // strictEquals, not identical(): see _ensureContext. A miss here rebuilt
    // the pipelines and the uniform buffer on every present.
    if (!_bufferDevice.strictEquals(device).toDart) {
      _bufferParamBuf?.callMethod<JSAny?>('destroy'.toJS);
      _bufferParamBuf = null;
      _bufferPipelines.clear();
      _bufferDevice = device;
      _bufferCanvasFormat = canvasFormat;
    } else if (_bufferCanvasFormat != canvasFormat) {
      _bufferPipelines.clear();
      _bufferCanvasFormat = canvasFormat;
    }
    _bufferParamBuf ??= device.callMethod<JSObject>(
      'createBuffer'.toJS,
      {'size': 16, 'usage': 0x40 | 0x08}.jsify(),
    ); // UNIFORM | COPY_DST
    return _bufferPipelines.putIfAbsent(format, () {
      final shader = device.callMethod<JSObject>(
        'createShaderModule'.toJS,
        {'code': _bufferBlitShader(format)}.jsify(),
      );
      return device.callMethod<JSObject>(
        'createRenderPipeline'.toJS,
        {
          'layout': 'auto',
          'vertex': {'module': shader, 'entryPoint': 'vs'},
          'fragment': {
            'module': shader,
            'entryPoint': 'fs',
            'targets': [
              <String, Object?>{'format': canvasFormat},
            ],
          },
          'primitive': {'topology': 'triangle-list'},
        }.jsify(),
      );
    });
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
    _bufferParamBuf?.callMethod<JSAny?>('destroy'.toJS);
    _bufferParamBuf = null;
    _bufferPipelines.clear();
    _bufferDevice = null;
    canvas.remove();
  }
}
