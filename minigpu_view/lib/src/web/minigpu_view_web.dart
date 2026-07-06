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
    return result is JSObject ? result : null;
  } catch (_) {
    return null;
  }
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
      // 'texture' key: a JSObject GPUTexture passed directly (same-device path).
      // 'bufferHandle' key: an Emscripten integer handle — we resolve the
      //   JS GPUBuffer locally via WebGPU.getJsObject() to stay codec-safe.
      final texture = args['texture'];
      final bufferHandle = args['bufferHandle'];
      if (texture is JSObject) {
        sink.copyFromGpuTexture(texture, width, height);
      } else if (bufferHandle is int && bufferHandle != 0) {
        final jsBuffer = _webGpuGetJsObject(bufferHandle);
        if (jsBuffer != null) {
          final format = args['format'] as String? ?? 'rgba32float';
          sink.copyFromGpuBuffer(jsBuffer, width, height, format);
        }
        // jsBuffer == null → WASM not ready yet; skip frame silently.
      }
      // else: handle is 0/null → first frame, skip blit silently.
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
  String? _canvasFormat;
  bool _disposed = false;

  // Cached blit pipeline for rgba32float → canvas format conversion.
  JSObject? _blitPipeline;
  JSObject? _blitDevice;
  String? _blitCanvasFormat;

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
    _canvasFormat = format;
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

  // WGSL: fullscreen triangle that samples an rgba32float texture and
  // writes to the canvas render-attachment (any format, e.g. bgra8unorm).
  // Y-flip applied: WebGPU NDC Y+=up, texture V=0=top.
  // textureLoad is used instead of textureSample because rgba32float textures
  // have sample-type UnfilterableFloat — they cannot be used with a filtering
  // sampler or textureSample.  textureLoad reads the exact texel at the
  // fragment's screen-space pixel position, which is identical to the canvas
  // pixel, so no UV math or sampler is needed.
  static const _kBlitShader = '''
@group(0) @binding(0) var srcTex: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  return vec4f(pos[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return textureLoad(srcTex, vec2i(i32(pos.x), i32(pos.y)), 0);
}
''';

  /// Create (or recreate) the blit render pipeline when device or canvas
  /// format changes. Cached across frames.
  void _ensureBlitPipeline(JSObject device, String canvasFormat) {
    if (_blitPipeline != null &&
        identical(_blitDevice, device) &&
        _blitCanvasFormat == canvasFormat)
      return;

    final shader = device.callMethod<JSObject>(
      'createShaderModule'.toJS,
      {'code': _kBlitShader}.jsify(),
    );
    _blitPipeline = device.callMethod<JSObject>(
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
    _blitDevice = device;
    _blitCanvasFormat = canvasFormat;
  }

  /// Blit a producer GPUBuffer into the canvas via a two-step GPU path:
  ///   1. `copyBufferToTexture` → intermediate texture (COPY_DST | TEXTURE_BINDING)
  ///   2. Render pass with WGSL passthrough shader → canvas render-attachment
  ///      (handles any format mismatch, e.g. rgba32float → bgra8unorm).
  /// Both steps are encoded in a single command buffer.
  ///
  /// format == 'rgba8': the buffer holds tightly-packed RGBA8 pixels (one u32
  /// per pixel, R in the low byte) and takes the DIRECT storage-buffer path —
  /// a single render pass with unpack4x8unorm, no intermediate texture and no
  /// 256-byte bytesPerRow constraint (tightly-packed rows of arbitrary width).
  void copyFromGpuBuffer(JSObject src, int width, int height, String format) {
    if (_disposed) return;
    final device = _resolveDevice();
    if (device == null) return;
    _ensureContext(device);

    if (format == 'rgba8') {
      _blitRgba8Direct(device, src, width, height);
      return;
    }

    if (canvas.width != width) canvas.width = width;
    if (canvas.height != height) canvas.height = height;

    final canvasFormat = _canvasFormat ?? 'bgra8unorm';
    _ensureBlitPipeline(device, canvasFormat);

    final bpp = format == 'rgba32float' ? 16 : 4;
    final bytesPerRow = _alignBytesPerRow(width * bpp);

    // Intermediate texture: receives the buffer data, then sampled by the
    // render pass. Needs COPY_DST (0x02) + TEXTURE_BINDING (0x04).
    final intermediateTex = device.callMethod<JSObject>(
      'createTexture'.toJS,
      {
        'size': {'width': width, 'height': height},
        'format': format,
        'usage': 0x02 | 0x04, // COPY_DST | TEXTURE_BINDING
      }.jsify(),
    );

    try {
      final encoder = device.callMethod<JSObject>('createCommandEncoder'.toJS);

      // Step 1: buffer → intermediate rgba32float texture.
      encoder.callMethod<JSAny?>(
        'copyBufferToTexture'.toJS,
        {'buffer': src, 'bytesPerRow': bytesPerRow}.jsify(),
        {'texture': intermediateTex}.jsify(),
        {'width': width, 'height': height, 'depthOrArrayLayers': 1}.jsify(),
      );

      // Step 2: render pass — intermediate texture → canvas (format conversion).
      final pipeline = _blitPipeline!;
      final srcView = intermediateTex.callMethod<JSObject>('createView'.toJS);
      final bindGroup = device.callMethod<JSObject>(
        'createBindGroup'.toJS,
        {
          'layout': pipeline.callMethod<JSObject>(
            'getBindGroupLayout'.toJS,
            0.toJS,
          ),
          'entries': [
            <String, Object?>{'binding': 0, 'resource': srcView},
          ],
        }.jsify(),
      );

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

      final queue = device['queue'] as JSObject;
      queue.callMethod<JSAny?>(
        'submit'.toJS,
        [encoder.callMethod<JSObject>('finish'.toJS)].jsify(),
      );
    } finally {
      intermediateTex.callMethod<JSAny?>('destroy'.toJS);
    }
  }

  /// Returns [byteCount] rounded up to the nearest multiple of 256
  /// (WebGPU's required [bytesPerRow] alignment).
  static int _alignBytesPerRow(int byteCount) => (byteCount + 255) & ~255;

  // ── rgba8 DIRECT path: storage-buffer read + unpack4x8unorm ──
  // The zero-readback display route for packed-RGBA8 producers (e.g. the
  // gsplats420 codec framebuffer): the producer buffer is bound straight into
  // the fragment shader; unpack4x8unorm maps the little-endian u32 (R in the
  // low byte) to vec4f exactly as the bytes lie in memory.
  JSObject? _rgba8Pipeline;
  JSObject? _rgba8Device;
  String? _rgba8CanvasFormat;
  JSObject? _rgba8ParamBuf;

  static const _kRgba8BlitShader = '''
@group(0) @binding(0) var<storage, read> src : array<u32>;
struct Params { w: u32, h: u32 };
@group(0) @binding(1) var<uniform> p : Params;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  return vec4f(pos[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let x = u32(pos.x);
  let y = u32(pos.y);
  if (x >= p.w || y >= p.h) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  return unpack4x8unorm(src[y * p.w + x]);
}
''';

  void _ensureRgba8Pipeline(JSObject device, String canvasFormat) {
    if (_rgba8Pipeline != null &&
        identical(_rgba8Device, device) &&
        _rgba8CanvasFormat == canvasFormat) {
      return;
    }
    final shader = device.callMethod<JSObject>(
      'createShaderModule'.toJS,
      {'code': _kRgba8BlitShader}.jsify(),
    );
    _rgba8Pipeline = device.callMethod<JSObject>(
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
    // 16 bytes (uniform min alignment); holds w, h.
    _rgba8ParamBuf ??= device.callMethod<JSObject>(
      'createBuffer'.toJS,
      {'size': 16, 'usage': 0x40 | 0x08}.jsify(), // UNIFORM | COPY_DST
    );
    _rgba8Device = device;
    _rgba8CanvasFormat = canvasFormat;
  }

  void _blitRgba8Direct(JSObject device, JSObject src, int width, int height) {
    if (canvas.width != width) canvas.width = width;
    if (canvas.height != height) canvas.height = height;

    final canvasFormat = _canvasFormat ?? 'bgra8unorm';
    _ensureRgba8Pipeline(device, canvasFormat);

    final queue = device['queue'] as JSObject;
    final params = Uint32List.fromList([width, height, 0, 0]);
    queue.callMethodVarArgs<JSAny?>('writeBuffer'.toJS, [
      _rgba8ParamBuf!,
      0.toJS,
      params.buffer.toJS,
    ]);

    final pipeline = _rgba8Pipeline!;
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
            'resource': {'buffer': _rgba8ParamBuf},
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
    queue.callMethod<JSAny?>(
      'submit'.toJS,
      [encoder.callMethod<JSObject>('finish'.toJS)].jsify(),
    );
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
    _blitPipeline = null;
    canvas.remove();
  }
}
