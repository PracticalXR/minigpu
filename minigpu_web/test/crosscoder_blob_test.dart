/// Validates the MGPU_EXTRA_SOURCES single-blob build: when the livetensor
/// crosscoder is folded into minigpu_web.wasm (GSPLATS420 sec.9), the gs_*
/// codec must work in the browser — encode, decode, and (when WebGPU is
/// available) the GPU backend byte-exact against the CPU backend, all
/// in-module.
///
/// Run with:
///   flutter test -p chrome test/crosscoder_blob_test.dart \
///     --web-browser-flag="--enable-unsafe-webgpu" \
///     --web-browser-flag="--disable-dawn-features=disallow_unsafe_apis" \
///     --web-browser-flag="--enable-features=SharedArrayBuffer"
///
/// Skips (with a reason) when the blob asset isn't served or lacks the gs_*
/// exports (i.e. minigpu_web.wasm was built without MGPU_EXTRA_SOURCES).
@TestOn('browser')
@Timeout(Duration(minutes: 3))
library;

import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:web/web.dart' as web;

// ---------------------------------------------------------------------------
// Module bootstrap + ccall plumbing
// ---------------------------------------------------------------------------

JSObject? get _module {
  final m = web.window.getProperty('Module'.toJS);
  return m.isA<JSObject>() ? m as JSObject : null;
}

Future<bool> _loadBlob() async {
  if (_module != null) return true;
  const candidates = [
    // package:test serves the package root; flutter test serves assets/.
    '/lib/web/minigpu_web.js',
    'lib/web/minigpu_web.js',
    '../lib/web/minigpu_web.js',
    'assets/packages/minigpu_web/web/minigpu_web.js',
    'packages/minigpu_web/web/minigpu_web.js',
  ];
  for (final url in candidates) {
    final done = Completer<bool>();
    final s = web.HTMLScriptElement()..src = url;
    s.onload = (() => done.complete(true)).toJS;
    s.onerror = ((JSAny? _) => done.complete(false)).toJS;
    web.document.head!.append(s);
    if (!await done.future) continue;
    // ignore: avoid_print
    print('blob script loaded from: $url');
    // Wait for the emscripten runtime.
    for (var i = 0; i < 300; i++) {
      final m = _module;
      if (m != null && m.getProperty('calledRun'.toJS).dartify() == true) {
        return true;
      }
      await Future<void>.delayed(const Duration(milliseconds: 100));
    }
    // ignore: avoid_print
    print('script loaded but runtime never initialized');
  }
  return false;
}

/// ccall with async:true — safe for both plain and asyncify-suspending
/// exports (GPU probe/decode suspend on WebGPU promises).
Future<int> _call(String name, List<int> args) async {
  final m = _module!;
  final opts = JSObject()..setProperty('async'.toJS, true.toJS);
  try {
    final r = await (m.callMethodVarArgs('ccall'.toJS, [
      name.toJS,
      'number'.toJS,
      [for (final _ in args) 'number'.toJS].toJS,
      [for (final a in args) a.toJS].toJS,
      opts,
    ]) as JSPromise).toDart;
    return ((r as JSNumber?)?.toDartDouble ?? 0).toInt();
  } catch (e) {
    fail('ccall $name($args) threw: $e');
  }
}

bool _hasExport(String name) =>
    _module!.getProperty('_$name'.toJS).isA<JSFunction>();

Uint8List _heap() =>
    (_module!.getProperty('HEAPU8'.toJS) as JSUint8Array).toDart;

Future<int> _malloc(int n) => _call('malloc', [n]);
Future<void> _free(int p) async => _call('free', [p]);

String _cstr(int ptr) {
  final h = _heap();
  var end = ptr;
  while (h[end] != 0) end++;
  return String.fromCharCodes(h.sublist(ptr, end));
}

// ---------------------------------------------------------------------------
// A deterministic moving-gradient test clip (matches the harness style).
// ---------------------------------------------------------------------------
Uint8List _frame(int w, int h, int t) {
  final f = Uint8List(w * h * 4);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final i = (y * w + x) * 4;
      f[i] = (x * 2 + t * 3) & 0xFF;
      f[i + 1] = (y * 2 + t) & 0xFF;
      f[i + 2] = ((x ^ y) + t * 2) & 0xFF;
      f[i + 3] = 255;
    }
  }
  return f;
}

void main() {
  const w = 320, h = 240, frames = 8;

  test('single blob: gs_* codec works in-browser; GPU == CPU when available',
      () async {
    if (!await _loadBlob()) {
      markTestSkipped('minigpu_web.js not served (run under flutter test)');
      return;
    }
    if (!_hasExport('gs_codec_create_ex')) {
      markTestSkipped(
          'blob built without MGPU_EXTRA_SOURCES (no gs_* exports)');
      return;
    }

    final ver = _cstr(await _call('gs_version', []));
    expect(ver, contains('gsplats420'));

    final webgpu = await _call('gs_webgpu_available', []);
    // ignore: avoid_print
    print('blob: $ver, gs_webgpu_available=$webgpu');

    final enc = await _call('gs_encoder_create', [w, h, 0]);
    expect(enc, isNot(0));
    final cpu = await _call('gs_codec_create_ex', [w, h, 1]);
    final auto = await _call('gs_codec_create_ex', [w, h, 0]);
    expect(cpu, isNot(0));
    expect(auto, isNot(0));

    final autoBackend = await _call('gs_codec_backend', [auto]);
    // ignore: avoid_print
    print('auto codec backend: ${autoBackend == 2 ? 'GPU' : 'CPU'}');
    if (webgpu != 0) {
      expect(autoBackend, 2,
          reason: 'WebGPU available but the GPU backend was not selected');
    }

    final inPtr = await _malloc(w * h * 4);
    const outCap = 256 * 1024;
    final outPtr = await _malloc(outCap);

    var mismatches = 0;
    for (var t = 0; t < frames; t++) {
      _heap().setAll(inPtr, _frame(w, h, t));
      final budget = 24 * 1024;
      final pktLen =
          await _call('gs_encoder_encode', [enc, inPtr, budget, outPtr, outCap]);
      expect(pktLen, greaterThan(0), reason: 'encode failed at frame $t');

      expect(await _call('gs_codec_decode', [cpu, outPtr, pktLen]), 1);
      expect(await _call('gs_codec_decode', [auto, outPtr, pktLen]), 1);

      final fbC = await _call('gs_codec_framebuffer', [cpu]);
      final fbA = await _call('gs_codec_framebuffer', [auto]);
      final heap = _heap();
      for (var j = 0; j < w * h * 4; j++) {
        if (heap[fbC + j] != heap[fbA + j]) {
          mismatches++;
          if (mismatches < 4) {
            // ignore: avoid_print
            print('frame $t mismatch @ $j: '
                'cpu=${heap[fbC + j]} auto=${heap[fbA + j]}');
          }
        }
      }
    }
    expect(mismatches, 0,
        reason: 'auto backend (${autoBackend == 2 ? 'GPU' : 'CPU'}) must be '
            'byte-exact vs the CPU backend');

    await _call('gs_encoder_destroy', [enc]);
    await _call('gs_codec_destroy', [cpu]);
    await _call('gs_codec_destroy', [auto]);
    await _free(inPtr);
    await _free(outPtr);
  });

  test('web ZERO-READBACK present: decoded frames land in a JS-visible '
      'GPUBuffer, no wasm-heap readback', () async {
    if (!await _loadBlob()) {
      markTestSkipped('minigpu_web.js not served');
      return;
    }
    if (!_hasExport('gs_codec_present_handle')) {
      markTestSkipped('blob lacks gs_codec_present_handle');
      return;
    }
    if (await _call('gs_webgpu_available', []) == 0) {
      markTestSkipped('no WebGPU adapter in this browser');
      return;
    }

    final enc = await _call('gs_encoder_create', [w, h, 0]);
    final cpu = await _call('gs_codec_create_ex', [w, h, 1]);
    final auto = await _call('gs_codec_create_ex', [w, h, 0]);
    expect(await _call('gs_codec_backend', [auto]), 2);

    expect(await _call('gs_codec_present_begin', [auto]), 1,
        reason: 'web present mode should always engage on the GPU backend');
    final handle = await _call('gs_codec_present_handle', [auto]);
    expect(handle, isNot(0), reason: 'present handle must be live');

    final inPtr = await _malloc(w * h * 4);
    const outCap = 256 * 1024;
    final outPtr = await _malloc(outCap);
    for (var t = 0; t < frames; t++) {
      _heap().setAll(inPtr, _frame(w, h, t));
      final pktLen = await _call(
          'gs_encoder_encode', [enc, inPtr, 24 * 1024, outPtr, outCap]);
      expect(pktLen, greaterThan(0));
      // ZERO-READBACK path: decode only; never touch gs_codec_framebuffer
      // on the presenting codec.
      expect(await _call('gs_codec_decode', [auto, outPtr, pktLen]), 1);
      expect(await _call('gs_codec_decode', [cpu, outPtr, pktLen]), 1);
    }

    // Read the presented GPUBuffer from the JS side -- the same object the
    // display blit consumes -- and compare against the CPU reconstruction.
    final webgpu = web.window.getProperty('WebGPU'.toJS) as JSObject;
    final gpuBuf =
        webgpu.callMethod('getJsObject'.toJS, handle.toJS) as JSObject?;
    expect(gpuBuf, isNotNull, reason: 'handle must resolve to a GPUBuffer');

    final devHandle = await _call('mgpuGetWGPUDeviceHandle', []);
    final device =
        webgpu.callMethod('getJsObject'.toJS, devHandle.toJS) as JSObject;

    final n = w * h * 4;
    final desc = JSObject()
      ..setProperty('size'.toJS, n.toJS)
      ..setProperty('usage'.toJS, 9.toJS); // MAP_READ | COPY_DST
    final staging = device.callMethod('createBuffer'.toJS, desc) as JSObject;
    final encJs =
        device.callMethod('createCommandEncoder'.toJS) as JSObject;
    encJs.callMethodVarArgs('copyBufferToBuffer'.toJS,
        [gpuBuf!, 0.toJS, staging, 0.toJS, n.toJS]);
    final cmd = encJs.callMethod('finish'.toJS) as JSObject;
    final queue = device.getProperty('queue'.toJS) as JSObject;
    queue.callMethod('submit'.toJS, [cmd].toJS);
    await (staging.callMethod('mapAsync'.toJS, 1.toJS) as JSPromise).toDart;
    final range =
        staging.callMethod('getMappedRange'.toJS) as JSArrayBuffer;
    final gpuBytes = Uint8List.view(range.toDart);

    final fbC = await _call('gs_codec_framebuffer', [cpu]);
    final heap = _heap();
    var mismatches = 0;
    for (var j = 0; j < n; j++) {
      if (gpuBytes[j] != heap[fbC + j]) mismatches++;
    }
    staging.callMethod('unmap'.toJS);
    expect(mismatches, 0,
        reason: 'the JS-visible GPUBuffer (zero-readback present target) '
            'must hold the CPU-exact frame');

    // ── Display-shader validation: run the SAME storage-buffer +
    // unpack4x8unorm render the minigpu_view web plugin uses for
    // format:'rgba8', into an offscreen rgba8unorm target, and read it
    // back. This is the full display path minus the canvas. ──
    const blitWgsl = '''
@group(0) @binding(0) var<storage, read> src : array<u32>;
struct Params { w: u32, h: u32 };
@group(0) @binding(1) var<uniform> p : Params;
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(pos[vi], 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let x = u32(pos.x); let y = u32(pos.y);
  if (x >= p.w || y >= p.h) { return vec4f(0.0, 0.0, 0.0, 1.0); }
  return unpack4x8unorm(src[y * p.w + x]);
}
''';
    final shader = device.callMethod('createShaderModule'.toJS,
        (JSObject()..setProperty('code'.toJS, blitWgsl.toJS))) as JSObject;
    final pipeDesc = JSObject()
      ..setProperty('layout'.toJS, 'auto'.toJS)
      ..setProperty(
          'vertex'.toJS,
          JSObject()
            ..setProperty('module'.toJS, shader)
            ..setProperty('entryPoint'.toJS, 'vs'.toJS))
      ..setProperty(
          'fragment'.toJS,
          JSObject()
            ..setProperty('module'.toJS, shader)
            ..setProperty('entryPoint'.toJS, 'fs'.toJS)
            ..setProperty(
                'targets'.toJS,
                [JSObject()..setProperty('format'.toJS, 'rgba8unorm'.toJS)]
                    .toJS));
    final pipeline =
        device.callMethod('createRenderPipeline'.toJS, pipeDesc) as JSObject;

    final texDesc = JSObject()
      ..setProperty(
          'size'.toJS,
          JSObject()
            ..setProperty('width'.toJS, w.toJS)
            ..setProperty('height'.toJS, h.toJS))
      ..setProperty('format'.toJS, 'rgba8unorm'.toJS)
      ..setProperty('usage'.toJS, (0x10 | 0x01).toJS); // RENDER_ATTACHMENT|COPY_SRC
    final target = device.callMethod('createTexture'.toJS, texDesc) as JSObject;

    final paramData = Uint32List.fromList([w, h, 0, 0]);
    final paramDesc = JSObject()
      ..setProperty('size'.toJS, 16.toJS)
      ..setProperty('usage'.toJS, (0x40 | 0x08).toJS); // UNIFORM | COPY_DST
    final paramBuf = device.callMethod('createBuffer'.toJS, paramDesc) as JSObject;
    queue.callMethodVarArgs(
        'writeBuffer'.toJS, [paramBuf, 0.toJS, paramData.buffer.toJS]);

    final bgDesc = JSObject()
      ..setProperty('layout'.toJS,
          pipeline.callMethod('getBindGroupLayout'.toJS, 0.toJS) as JSObject)
      ..setProperty(
          'entries'.toJS,
          [
            JSObject()
              ..setProperty('binding'.toJS, 0.toJS)
              ..setProperty('resource'.toJS,
                  JSObject()..setProperty('buffer'.toJS, gpuBuf)),
            JSObject()
              ..setProperty('binding'.toJS, 1.toJS)
              ..setProperty('resource'.toJS,
                  JSObject()..setProperty('buffer'.toJS, paramBuf)),
          ].toJS);
    final bindGroup = device.callMethod('createBindGroup'.toJS, bgDesc) as JSObject;

    final enc2 = device.callMethod('createCommandEncoder'.toJS) as JSObject;
    final passDesc = JSObject()
      ..setProperty(
          'colorAttachments'.toJS,
          [
            JSObject()
              ..setProperty(
                  'view'.toJS, target.callMethod('createView'.toJS) as JSObject)
              ..setProperty('loadOp'.toJS, 'clear'.toJS)
              ..setProperty('storeOp'.toJS, 'store'.toJS),
          ].toJS);
    final pass = enc2.callMethod('beginRenderPass'.toJS, passDesc) as JSObject;
    pass.callMethod('setPipeline'.toJS, pipeline);
    pass.callMethodVarArgs('setBindGroup'.toJS, [0.toJS, bindGroup]);
    pass.callMethod('draw'.toJS, 3.toJS);
    pass.callMethod('end'.toJS);
    // texture -> staging buffer (w*4 = 1280 bytes/row, 256-aligned for w=320)
    final stage2Desc = JSObject()
      ..setProperty('size'.toJS, n.toJS)
      ..setProperty('usage'.toJS, 9.toJS); // MAP_READ | COPY_DST
    final stage2 = device.callMethod('createBuffer'.toJS, stage2Desc) as JSObject;
    final copySrc = JSObject()..setProperty('texture'.toJS, target);
    final copyDst = JSObject()
      ..setProperty('buffer'.toJS, stage2)
      ..setProperty('bytesPerRow'.toJS, (w * 4).toJS);
    final copySize = JSObject()
      ..setProperty('width'.toJS, w.toJS)
      ..setProperty('height'.toJS, h.toJS);
    enc2.callMethodVarArgs(
        'copyTextureToBuffer'.toJS, [copySrc, copyDst, copySize]);
    queue.callMethod('submit'.toJS, [enc2.callMethod('finish'.toJS)].toJS);
    await (stage2.callMethod('mapAsync'.toJS, 1.toJS) as JSPromise).toDart;
    final rendered = Uint8List.view(
        (stage2.callMethod('getMappedRange'.toJS) as JSArrayBuffer).toDart);
    var blitMismatches = 0;
    for (var j = 0; j < n; j++) {
      if (rendered[j] != heap[fbC + j]) blitMismatches++;
    }
    stage2.callMethod('unmap'.toJS);
    expect(blitMismatches, 0,
        reason: 'the rgba8 display blit (storage buffer + unpack4x8unorm) '
            'must render the CPU-exact frame');

    await _call('gs_codec_present_end', [auto]);
    await _call('gs_encoder_destroy', [enc]);
    await _call('gs_codec_destroy', [cpu]);
    await _call('gs_codec_destroy', [auto]);
    await _free(inPtr);
    await _free(outPtr);
  });
}
