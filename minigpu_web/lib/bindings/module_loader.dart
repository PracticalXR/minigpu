/// Lazy, self-contained loader for the minigpu Emscripten glue.
///
/// 🔴 WHY THIS EXISTS — the glue is a classic (non-MODULARIZE) Emscripten
/// build: its script declares a top-level `var Module` and keeps writing
/// through that binding for the LIFE of the page (heap growth re-assigns
/// `Module["HEAP8"]` etc.). Loaded as a plain `<script>`, that binding is
/// `globalThis.Module` — a name ANY other classic Emscripten build on the
/// page (miniav_web's audio wasm, for one) also claims. Two claimants and
/// whichever loads second corrupts the first; on a page that never injects
/// the glue at all, `Module` belongs to someone else entirely and every
/// `_mgpu*` lookup dies as "func is not a function".
///
/// So this loader runs the glue INSIDE A FUNCTION SCOPE — a fixed wrapper
/// in a blob-backed script, called with OUR private object —
/// which turns every top-level `var` in the glue (Module, HEAP views,
/// runtime state) into a function local. The glue's prelude
/// (`var Module = typeof Module != 'undefined' ? Module : {}`) sees the
/// parameter and adopts our object; exports land on it; the shared global
/// `Module` is untouched. Loading uses the browser's script loader rather
/// than `eval`/`Function`, so CSP needs `blob:` but not `unsafe-eval`.
/// The initialized instance is then published as
/// `globalThis.MinigpuModule`, which is the ONLY name the Dart bindings read.
///
/// Host pages that preload the glue the old way (`minigpu_web.loader.js` in
/// index.html — gui_bench, minigpu_av, the view example) still work: their
/// glue really does live at `globalThis.Module`, and [_load] ADOPTS it as
/// `MinigpuModule` instead of loading twice (awaiting the loader's in-flight
/// promise when that glue has not run yet).
library;

import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';

/// Where this package's web assets may be served from, most likely first:
/// a Flutter build serves `assets/packages/...`; `dart test`/`flutter test`
/// serve the pub layout `packages/<name>/` (mapped onto `lib/`). Whichever
/// candidate the glue fetch succeeds from is also the base the private-scope
/// `locateFile` resolves `minigpu_web.wasm` against — the blob script has no
/// usable asset directory, so without it the glue would fetch
/// the wasm from the page root and 404.
const List<String> _kAssetDirCandidates = [
  'assets/packages/minigpu_web/web/',
  'packages/minigpu_web/web/',
];

/// Appended to the packaged glue source, so it runs in the SAME function
/// scope and can reach the now-local `WebGPU` — the Emscripten Dawn handle
/// table that `webgpu_interop.dart` (deliberately un-scoped) reads off
/// `globalThis`. One page has at most one Dawn-WebGPU wasm, so the global is
/// this module's to publish.
const String _kScopeExports = '\n;globalThis.WebGPU = WebGPU;\n';

/// 🔴 **WebKit ships WebGPU without `GPUAdapter.info`, and the glue reads it
/// unguarded.** Dawn's Emscripten port implements `wgpuAdapterGetInfo` as
/// `fillAdapterInfoStruct(adapter.info, …)`, which dereferences the result
/// for `subgroupMinSize` — on an iPad that is `TypeError: undefined is not
/// an object (evaluating 'info.subgroupMinSize')`, thrown out of the FIRST
/// context init, so every encoder and decoder on the page is stillborn while
/// `navigator.gpu` itself works fine. Seen 2026-09-01 on iPadOS Safari.
///
/// The glue is generated (a wasm rebuild would drop any edit to it), so the
/// durable fix lives here, before the glue runs: give the adapter prototype an
/// `info` getter answering the spec's shape with blank strings and zero
/// subgroup sizes. Blank is honest — nothing here consumes the strings
/// except log lines — and zero is what the glue stores for a missing field
/// on browsers that DO have `info` but predate subgroups.
///
/// Installed on the global prototype (not the private scope), because that is
/// where the glue looks; idempotent, and a no-op wherever `info` exists.
@JS('Object.defineProperty')
external JSObject _defineProperty(
    JSObject object, JSString property, JSObject descriptor);

// Fixed-arity interop avoids the variable-argument bridge's runtime
// conversion of a missing argument list, which can require eval under CSP.
@JS('fetch')
external JSPromise<_GlueResponse> _fetchGlue(JSString url);

extension type _GlueResponse._(JSObject _) implements JSObject {
  external JSBoolean get ok;
  external JSNumber get status;
  external JSPromise<JSString> text();
}

@JS('Blob')
extension type _LoaderBlob._(JSObject _) implements JSObject {
  external factory _LoaderBlob(
      JSArray<JSAny?> parts, _LoaderBlobOptions options);
}

extension type _LoaderBlobOptions._(JSObject _) implements JSObject {
  external factory _LoaderBlobOptions({required String type});
}

@JS('URL.createObjectURL')
external JSString _createObjectURL(_LoaderBlob blob);

@JS('URL.revokeObjectURL')
external void _revokeObjectURL(JSString url);

@JS('importScripts')
external void _importScripts(JSString url);

extension type _LoaderDocument._(JSObject _) implements JSObject {
  external _LoaderScript createElement(JSString tag);
  external _LoaderParent? get head;
  external _LoaderParent? get documentElement;
}

extension type _LoaderParent._(JSObject _) implements JSObject {
  external JSAny? appendChild(_LoaderScript script);
}

extension type _LoaderScript._(JSObject _) implements JSObject {
  external set src(JSString url);
  external set onload(JSFunction? handler);
  external set onerror(JSFunction? handler);
  external void remove();
}

void installWebGpuAdapterInfoShim() {
  try {
    final g = globalContext;
    final ga = g.getProperty<JSObject?>('GPUAdapter'.toJS);
    if (ga == null || ga.isUndefinedOrNull) return;
    final proto = ga.getProperty<JSObject?>('prototype'.toJS);
    if (proto == null || proto.isUndefinedOrNull) return;
    if (proto.has('info')) return;
    final descriptor = JSObject();
    descriptor.setProperty('configurable'.toJS, true.toJS);
    descriptor.setProperty(
      'get'.toJS,
      (() {
        final info = JSObject();
        for (final field in const [
          'vendor',
          'architecture',
          'device',
          'description',
        ]) {
          info.setProperty(field.toJS, ''.toJS);
        }
        info.setProperty('subgroupMinSize'.toJS, 0.toJS);
        info.setProperty('subgroupMaxSize'.toJS, 0.toJS);
        info.setProperty('isFallbackAdapter'.toJS, false.toJS);
        return info;
      }).toJS,
    );
    _defineProperty(proto, 'info'.toJS, descriptor);
  } catch (_) {
    // A browser that refuses the property leaves the glue exactly as it was.
  }
}

/// The URL the app's assets are served under, in BOTH contexts.
///
/// On the main thread that is the document's base URL, which is what a
/// relative `fetch('assets/…')` resolves against anyway. In a WORKER there is
/// no document and `Uri.base` is the worker SCRIPT's URL —
/// `…/assets/packages/<pkg>/workers/build/x.dart.js` — so the same relative
/// fetch resolves under `workers/build/` and 404s. The 2026-09-01 Android
/// capture shows exactly that: worker-first engaged, and the encode worker
/// answered "minigpu glue fetch failed from every candidate (assets/…/ →
/// HTTP 404; packages/…/ → HTTP 404)" while the file sat where it always
/// sits. The app root is the prefix before the `assets/packages/` segment
/// (a Flutter build) or `packages/` (the pub layout tests serve).
String _appBase() {
  final g = globalContext;
  final document = g.getProperty<JSObject?>('document'.toJS);
  if (document != null && !document.isUndefinedOrNull) {
    final base = document.getProperty<JSString?>('baseURI'.toJS);
    if (base != null && !base.isUndefinedOrNull) return _dirOf(base.toDart);
  }
  final href = Uri.base.toString();
  for (final marker in const ['/assets/packages/', '/packages/']) {
    final i = href.indexOf(marker);
    if (i >= 0) return href.substring(0, i + 1);
  }
  return href;
}

/// The DIRECTORY of [url], with no query and no fragment.
///
/// 🔴 `document.baseURI` is only a directory when the page has a `<base>`
/// element (Flutter's index.html does). Without one it is the page's full
/// URL — and under `dart test -p chrome` that URL carries the test metadata
/// in its FRAGMENT, so `'$base' + 'assets/…'` resolved to the test page
/// itself: HTTP 200, HTML, and parsing the glue threw
/// `SyntaxError: Unexpected token '<'` out of every WebGPU browser test
/// (2026-09-01, the day after the worker-aware base landed). A page without
/// `<base>` and a hash route would do the same in production.
String _dirOf(String url) {
  try {
    final u = Uri.parse(url);
    final path = u.path.endsWith('/')
        ? u.path
        : u.path.substring(0, u.path.lastIndexOf('/') + 1);
    return Uri(
      scheme: u.scheme.isEmpty ? null : u.scheme,
      userInfo: u.userInfo.isEmpty ? null : u.userInfo,
      host: u.host.isEmpty ? null : u.host,
      port: u.hasPort ? u.port : null,
      path: path.isEmpty ? '/' : path,
    ).toString();
  } catch (_) {
    return url;
  }
}

Future<void>? _loading;

/// True once `globalThis.MinigpuModule` is available.
bool get isMinigpuModuleLoaded =>
    !globalContext.getProperty<JSAny?>('MinigpuModule'.toJS).isUndefinedOrNull;

/// Load (or adopt) the minigpu wasm module once. Safe to await repeatedly.
/// Throws on fetch/instantiation failure — callers surface that as "no GPU
/// on this platform" rather than a silent hang — and the next call tries again.
Future<void> ensureMinigpuModuleLoaded() {
  if (isMinigpuModuleLoaded) return Future<void>.value();
  return _loading ??= _loadOnce();
}

/// [_load], forgotten when it fails: a cached failure would answer every later
/// call without trying again (a fetch that failed once can succeed later).
Future<void> _loadOnce() async {
  try {
    await _load();
  } catch (_) {
    _loading = null;
    rethrow;
  }
}

/// The promise of a host page's `_minigpu.loader.load()` while it is in
/// flight (minigpu_web.loader.js keeps it until the load settles), else null.
JSPromise<JSAny?>? _hostLoad(JSObject g) {
  final minigpu = g.getProperty<JSAny?>('_minigpu'.toJS);
  if (minigpu.isUndefinedOrNull || !minigpu.isA<JSObject>()) return null;
  final loader = (minigpu as JSObject).getProperty<JSAny?>('loader'.toJS);
  if (loader.isUndefinedOrNull || !loader.isA<JSObject>()) return null;
  final loading = (loader as JSObject).getProperty<JSAny?>('loading'.toJS);
  if (loading.isUndefinedOrNull || !loading.isA<JSPromise>()) return null;
  return loading as JSPromise<JSAny?>;
}

Future<void> _load() async {
  final g = globalContext;
  // Before ANY path below — the legacy glue may already be initializing and
  // will read `adapter.info` the moment a context is created.
  installWebGpuAdapterInfoShim();

  // A host page's loader.js load in flight: share it. Its glue may not have
  // run yet (no `Module` to adopt), and a scoped load beside it would make a
  // second wasm whose handles and `WebGPU` table do not match the instance
  // the page publishes.
  final hostLoad = _hostLoad(g);
  if (hostLoad != null) {
    try {
      await hostLoad.toDart;
    } catch (_) {
      // Its one page-relative glue URL can miss; the scoped load below tries
      // every asset location.
    }
    if (isMinigpuModuleLoaded) return;
  }

  // Legacy adoption: a host page that injected the glue globally, runtime
  // ready. Any other `Module` (another wasm's, or ours after a failed host
  // load) is left alone: fall through to the scoped load.
  final legacy = g.getProperty<JSObject?>('Module'.toJS);
  if (legacy != null && !legacy.isUndefinedOrNull) {
    final ready =
        !legacy.getProperty<JSAny?>('_mgpuInitializeContext'.toJS).isUndefinedOrNull;
    if (ready) {
      g.setProperty('MinigpuModule'.toJS, legacy);
      return;
    }
  }

  // Fetch the glue source (same-origin asset; typed interop so this package
  // needs no extra dependency).
  String? src;
  final appBase = _appBase();
  String assetDir = '$appBase${_kAssetDirCandidates.first}';
  final failures = <String>[];
  for (final rel in _kAssetDirCandidates) {
    // ABSOLUTE, from the app root — see [_appBase] for the worker case.
    final dir = '$appBase$rel';
    try {
      final resp = await _fetchGlue('${dir}minigpu_web.js'.toJS).toDart;
      if (!resp.ok.toDart) {
        failures.add('$dir → HTTP ${resp.status.toDartInt}');
        continue;
      }
      src = (await resp.text().toDart).toDart;
      assetDir = dir;
      break;
    } catch (e) {
      failures.add('$dir → $e');
    }
  }
  if (src == null) {
    throw StateError(
        'minigpu glue fetch failed from every candidate (${failures.join('; ')}) '
        '— is minigpu_web.js being served?');
  }

  final module = JSObject();
  final ready = Completer<void>();
  module.setProperty(
    'locateFile'.toJS,
    ((JSString path, JSString _) => '$assetDir${path.toDart}'.toJS).toJS,
  );
  // Route the wasm's stdout/stderr through Dart's print instead of the
  // default console.log/error: the mgpu/Dawn narration then reaches anything
  // that captures Dart output — `dart test` reporters above all, where the
  // browser console is invisible and a C++ throw otherwise surfaces as a
  // bare exception-pointer number with no context. Verbosity is governed by
  // the native log level (Minigpu.setLogCallback), so a quiet level keeps
  // this silent in production.
  module.setProperty(
    'print'.toJS,
    ((JSString line) {
      // ignore: avoid_print
      print('[mgpu-wasm] ${line.toDart}');
    }).toJS,
  );
  module.setProperty(
    'printErr'.toJS,
    ((JSString line) {
      // ignore: avoid_print
      print('[mgpu-wasm!] ${line.toDart}');
    }).toJS,
  );
  module.setProperty(
    'onRuntimeInitialized'.toJS,
    (() {
      if (!ready.isCompleted) ready.complete();
    }).toJS,
  );
  module.setProperty(
    'onAbort'.toJS,
    ((JSAny? what) {
      if (!ready.isCompleted) {
        ready.completeError(StateError('minigpu wasm aborted: $what'));
      }
    }).toJS,
  );

  // Observe script execution and runtime readiness together. A script
  // failure must not wait forever for an uninitialized runtime.
  await Future.wait<void>([
    _runPackagedGlue(src, module),
    ready.future,
  ], eagerError: true);
  g.setProperty('MinigpuModule'.toJS, module);
}

/// Execute only our fetched package asset, inside the same private function
/// scope as the former Function-based loader. No caller-supplied code enters
/// this path. The package's browser workers are classic workers; module
/// workers would need an explicit module-loader entry point instead.
Future<void> _runPackagedGlue(String src, JSObject module) async {
  final g = globalContext;
  final options = _LoaderBlobOptions(type: 'text/javascript');
  // Object URLs are unique across independent Dart loads in the same realm.
  // This empty URL is only a collision-free handoff key, never a script.
  final token = _createObjectURL(_LoaderBlob(<JSAny?>[].toJS, options));
  _revokeObjectURL(token);
  final handoffKey = '__minigpuScopedLoader:${token.toDart}';
  final keyLiteral = jsonEncode(handoffKey);
  final executed = Completer<void>();
  final handoff = JSObject();
  handoff.setProperty('module'.toJS, module);
  handoff.setProperty(
    'loaded'.toJS,
    (() {
      if (!executed.isCompleted) executed.complete();
    }).toJS,
  );
  handoff.setProperty(
    'failed'.toJS,
    ((JSString error) {
      if (!executed.isCompleted) {
        executed.completeError(
            StateError('minigpu glue execution failed: ${error.toDart}'));
      }
    }).toJS,
  );

  JSString? scriptUrl;
  _LoaderScript? script;
  try {
    g.setProperty(handoffKey.toJS, handoff);
    final source = '''
(function (handoff) {
  delete globalThis[$keyLiteral];
  try {
    (function (Module) {
$src
$_kScopeExports
    }).call(null, handoff.module);
    handoff.loaded();
  } catch (error) {
    handoff.failed(String(error));
  }
})(globalThis[$keyLiteral]);
''';
    scriptUrl = _createObjectURL(_LoaderBlob([source.toJS].toJS, options));
    final document = g.getProperty<JSObject?>('document'.toJS);
    if (document != null && !document.isUndefinedOrNull) {
      final doc = _LoaderDocument._(document);
      final parent = doc.head ?? doc.documentElement;
      if (parent == null || parent.isUndefinedOrNull) {
        throw StateError('minigpu glue has no document script host');
      }
      final element = doc.createElement('script'.toJS);
      script = element;
      element.src = scriptUrl;
      element.onload = ((JSAny? _) {
        // A syntax error can still fire load, without reaching our wrapper.
        if (!executed.isCompleted) {
          executed.completeError(
              StateError('minigpu glue script loaded without executing'));
        }
      }).toJS;
      element.onerror = ((JSAny? _) {
        if (!executed.isCompleted) {
          executed.completeError(StateError('minigpu glue blob script failed'));
        }
      }).toJS;
      parent.appendChild(element);
    } else {
      if (!g.has('importScripts')) {
        throw UnsupportedError(
            'minigpu glue requires a document or classic worker importScripts');
      }
      _importScripts(scriptUrl);
      if (!executed.isCompleted) {
        throw StateError('minigpu worker glue loaded without executing');
      }
    }
    await executed.future;
  } finally {
    g.delete(handoffKey.toJS);
    if (script != null) {
      script.onload = null;
      script.onerror = null;
      script.remove();
    }
    if (scriptUrl != null) _revokeObjectURL(scriptUrl);
  }
}
