// Public Dart/Wasm raster lifecycle checks against the generated WebGPU
// backend. Serve the packaged backend assets beside the compiled entry and
// read bufferSmokeResult; rasterLifecycleResult carries the recorded numbers.
import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';

import 'raster_lifecycle_scenario.dart';

/// Stands between the backend and `GPUDevice.createRenderPipelineAsync`. The
/// browser starts compiling every pipeline at once, armed or not; only the
/// delivery of an armed completion waits for [release].
final class _BrowserPipelineHold implements RasterPipelineHold {
  _BrowserPipelineHold._(this._prototype, this._original);

  static const _method = 'createRenderPipelineAsync';
  final JSObject _prototype;
  final JSFunction _original;
  final _gates = <Completer<void>>[];
  int _armed = 0, _delivered = 0;

  static _BrowserPipelineHold install() {
    final device = globalContext.getProperty<JSObject?>('GPUDevice'.toJS);
    if (device == null || device.isUndefinedOrNull) {
      throw StateError('This browser exposes no GPUDevice');
    }
    final prototype = device.getProperty<JSObject>('prototype'.toJS);
    final hold = _BrowserPipelineHold._(
      prototype,
      prototype.getProperty<JSFunction>(_method.toJS),
    );
    prototype.setProperty(_method.toJS, hold._intercept.toJSCaptureThis);
    return hold;
  }

  void uninstall() => _prototype.setProperty(_method.toJS, _original);

  // The descriptor is optional so that any call arity reaches this function.
  JSAny? _intercept(JSObject device, [JSAny? descriptor]) {
    final real =
        _original.callAsFunction(device, descriptor) as JSPromise<JSAny?>;
    if (_armed == 0) return real;
    _armed--;
    final gate = Completer<void>();
    _gates.add(gate);
    // Observe the device's own outcome at once; hand it on after the gate.
    final settled = real.toDart.then<(JSAny?, Object?)>(
      (pipeline) => (pipeline, null),
      onError: (Object error) => (null, error),
    );
    return (() async {
      await gate.future;
      final (pipeline, error) = await settled;
      _delivered++;
      if (error != null) throw error;
      return pipeline;
    })().toJS;
  }

  @override
  void arm(int count) => _armed += count;

  @override
  int get held => _gates.where((gate) => !gate.isCompleted).length;

  @override
  int get delivered => _delivered;

  @override
  int release() {
    var released = 0;
    for (final gate in _gates) {
      if (gate.isCompleted) continue;
      gate.complete();
      released++;
    }
    return released;
  }
}

Future<void> main() async {
  // `?only=<check>` runs a single check, and the verdict then names it.
  final only = Uri.base.queryParameters['only'];
  bool runs(String check) => only == null || only == check;
  final receipt = <String, Object>{};
  var result = only == null ? 'passed' : 'passed: only $only';
  _BrowserPipelineHold? hold;
  try {
    if (runs('binding_cache')) {
      receipt['binding_cache'] = await runRasterBindingCacheScenario();
    }
    if (runs('pending_close')) {
      // One event-loop thread: no pipeline can complete before it is closed.
      receipt['pending_close'] = await runRasterPendingCloseScenario(
        requireEveryPending: true,
      );
    }
    if (runs('settle_signal')) {
      receipt['settle_signal'] = await runRasterSettleSignalScenario();
    }
    if (runs('preparation_timeout')) {
      hold = _BrowserPipelineHold.install();
      receipt['preparation_timeout'] =
          await runRasterPreparationTimeoutScenario(hold);
    }
    if (runs('foreign_owner')) {
      receipt['foreign_owner'] = await runRasterForeignOwnerScenario()
          ? 'refused'
          : 'not applicable: one context per page';
    }
    if (receipt.isEmpty) throw StateError('Unknown check: $only');
  } catch (error, stack) {
    result = 'failed: $error\n$stack';
  } finally {
    hold?.uninstall();
  }
  globalContext.setProperty(
    'rasterLifecycleResult'.toJS,
    jsonEncode(receipt).toJS,
  );
  globalContext.setProperty('bufferSmokeResult'.toJS, result.toJS);
}
