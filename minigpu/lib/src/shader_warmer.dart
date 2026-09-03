/// PRECOMPILE A SET OF KERNELS, IN THE BACKGROUND, AND SAY HOW IT IS GOING.
///
/// Why this exists. Turning WGSL into a backend shader is expensive, and
/// minigpu does it LAZILY — on a kernel's first `dispatch`, while holding the
/// GPU mutex. So the cost does not land where a caller can see it or plan for
/// it; it lands on the first frame, the first inference, the first anything,
/// and it looks like a hang rather than like work. Callers have been working
/// around it for a while by dispatching a throwaway 1x1x1 job at startup and
/// hoping.
///
/// This turns that idiom into an API with three properties the ad-hoc version
/// cannot have:
///
///  * **It is observable.** [ShaderWarmer.progress] streams N-of-M with the
///    label of the kernel being built, so a UI can show a real bar and a log
///    can say which kernel is slow — on the device where it is slow.
///  * **It names failures.** minigpu has had no compile-error channel to Dart
///    at all: `loadKernelString` returns void, a failed pipeline logs to the
///    console and returns false, and `dispatch()`'s future resolves either
///    way. A kernel that will never build has been indistinguishable from one
///    that has not built yet. Here it is a [WarmError] with a label and a
///    stage.
///  * **It is paced.** Kernels are built ONE AT A TIME by default, awaiting
///    each, which yields to the event loop between them. That is what makes it
///    "background" on the web, where there is no second thread to push it to.
///
/// 🔴 SERIAL IS THE DEFAULT AND IT IS NOT TIMIDITY. Compiling a batch of large
/// kernels back-to-back is how mobile GPU drivers stop answering: an iPad has
/// been observed dying inside one such build and taking the rest of init with
/// it. Concurrency here buys little (the driver serialises anyway) and risks
/// the one failure that cannot be recovered from.
///
/// This composes with the persistent shader cache rather than replacing it:
/// the cache makes the SECOND run cheap, this makes the FIRST run visible and
/// survivable. See `Minigpu.configureShaderCache`.
library;

import 'dart:async';

import 'package:minigpu/src/buffer.dart';
import 'package:minigpu/src/compute_shader.dart';
import 'package:minigpu/src/minigpu.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

/// One kernel to precompile.
class KernelWarmSpec {
  const KernelWarmSpec({
    required this.label,
    required this.source,
    required this.buffers,
    this.timeout = const Duration(seconds: 30),
  });

  /// Human name, used in progress and in every error about this kernel. Make
  /// it the name you would want to read in a device log.
  final String label;

  /// The WGSL. Byte-for-byte what you will really dispatch — a kernel warmed
  /// in a different form is a different pipeline, and warms nothing.
  final String source;

  /// Binding tag -> element count for a THROWAWAY buffer to bind while
  /// building.
  ///
  /// 🔴 THIS CANNOT BE OMITTED, and that is a property of the backend, not of
  /// this API. minigpu builds an EXPLICIT bind-group layout from the buffers
  /// actually bound, and refuses to build a pipeline when none are — so a
  /// "compile this source" call with no bindings cannot produce the pipeline
  /// the real dispatch will use. The counts only have to make the layout
  /// legal; a handful of elements each is enough, and the real buffers replace
  /// them later.
  ///
  /// Tags and their ORDER must match the real binding session, because binding
  /// slots are assigned in first-seen order.
  final Map<String, int> buffers;

  /// Give up on this kernel after this long. A timeout fails THIS spec only —
  /// the set carries on — because "this one kernel is too big for this driver"
  /// is both common and survivable.
  final Duration timeout;
}

/// Where a warm ended up.
enum WarmPhase {
  /// Not started.
  idle,

  /// Building.
  running,

  /// Every spec built.
  ready,

  /// At least one spec did not build. Inspect [WarmProgress.errors].
  failed,

  /// [ShaderWarmer.cancel] was called before the set finished.
  cancelled,
}

/// Which step of a build went wrong. The distinction matters: `pipeline` means
/// this DRIVER gave up on a shader it had already accepted as valid, which is
/// a statement about the device and not about the code.
enum WarmStage { create, dispatch, timeout, deviceLost }

/// One kernel's failure.
class WarmError {
  const WarmError(this.label, this.stage, this.message);

  final String label;
  final WarmStage stage;
  final String message;

  @override
  String toString() => '$label: ${stage.name} — $message';
}

/// A snapshot of a warm in flight. Immutable; a new one is emitted per step.
class WarmProgress {
  const WarmProgress({
    required this.phase,
    required this.completed,
    required this.total,
    required this.failed,
    required this.elapsed,
    this.label,
    this.errors = const <WarmError>[],
  });

  final WarmPhase phase;

  /// Specs finished, successfully or not — so this only ever goes up, which is
  /// what a progress bar needs.
  final int completed;
  final int total;
  final int failed;

  /// The kernel being built right now, or null when not running.
  final String? label;

  final Duration elapsed;
  final List<WarmError> errors;

  double get fraction => total == 0 ? 1.0 : completed / total;

  bool get isDone =>
      phase == WarmPhase.ready ||
      phase == WarmPhase.failed ||
      phase == WarmPhase.cancelled;

  @override
  String toString() => 'WarmProgress(${phase.name} $completed/$total'
      '${failed > 0 ? ", $failed failed" : ""}'
      '${label != null ? ", building $label" : ""})';
}

/// Handle on a running warm.
abstract class ShaderWarmer {
  /// Progress updates. Broadcast, and REPLAYS the latest value to a new
  /// listener — a diagnostics panel that subscribes after the warm started
  /// must not render an empty bar.
  Stream<WarmProgress> get progress;

  /// The latest snapshot, without subscribing.
  WarmProgress get value;

  /// Completes when the set finishes.
  ///
  /// 🔴 NEVER THROWS. A warm is typically kicked off unawaited at startup, and
  /// a future that rejects there is an unhandled error that either crashes the
  /// app or vanishes — both worse than the failure it is reporting. Read
  /// [WarmProgress.phase] and [WarmProgress.errors] instead.
  Future<WarmProgress> get done;

  /// Stop after the kernel currently building. Already-built pipelines stay
  /// built — they are process-global in the backend.
  void cancel();
}

/// Precompile [specs] on [gpu]. See the library doc for the rules.
///
/// Returns immediately; the work runs on the returned handle. Kernels are
/// built strictly one at a time — there is deliberately NO concurrency knob,
/// because the backend serialises GPU work behind one mutex anyway, so the
/// only thing parallelism could add here is the driver stampede that has
/// already been observed killing a mobile GPU mid-init.
/// [ensureReady] runs once before the first kernel — use it to bring the
/// device up. It is part of the warm rather than the caller's problem so that
/// this can stay SYNCHRONOUS: a caller kicking a warm off at launch wants a
/// handle to watch immediately, not a `Future<ShaderWarmer>` it has to await
/// before it can even subscribe to progress. If it throws, the set fails with
/// a named error like any other step.
ShaderWarmer warmShaders(
  Minigpu gpu,
  List<KernelWarmSpec> specs, {
  Future<void> Function()? ensureReady,
}) {
  final w = _Warmer(gpu, specs, ensureReady);
  unawaited(w._start());
  return w;
}

class _Warmer implements ShaderWarmer {
  _Warmer(this._gpu, this._specs, this._ensureReady) {
    _value = WarmProgress(
      phase: WarmPhase.idle,
      completed: 0,
      total: _specs.length,
      failed: 0,
      elapsed: Duration.zero,
    );
  }

  final Minigpu _gpu;
  final List<KernelWarmSpec> _specs;
  final Future<void> Function()? _ensureReady;

  final _ctrl = StreamController<WarmProgress>.broadcast();
  final _done = Completer<WarmProgress>();
  final _errors = <WarmError>[];
  final _stopwatch = Stopwatch();
  late WarmProgress _value;
  var _cancelled = false;

  /// Sources already built in this process. The backend caches pipelines
  /// globally, so re-warming the same text is pointless — and a caller warming
  /// per-stream would otherwise pay it per stream.
  static final Set<String> _built = <String>{};

  @override
  Stream<WarmProgress> get progress async* {
    yield _value;
    yield* _ctrl.stream;
  }

  @override
  WarmProgress get value => _value;

  @override
  Future<WarmProgress> get done => _done.future;

  @override
  void cancel() => _cancelled = true;

  void _emit(WarmProgress p) {
    _value = p;
    if (!_ctrl.isClosed) _ctrl.add(p);
  }

  void _finish(WarmPhase phase) {
    _emit(WarmProgress(
      phase: phase,
      completed: _value.completed,
      total: _specs.length,
      failed: _errors.length,
      elapsed: _stopwatch.elapsed,
      errors: List.unmodifiable(_errors),
    ));
    if (!_done.isCompleted) _done.complete(_value);
    unawaited(_ctrl.close());
  }

  Future<void> _start() async {
    _stopwatch.start();
    if (_specs.isEmpty) {
      _finish(WarmPhase.ready);
      return;
    }
    // Bring the device up first, if the caller gave us a way. A failure here
    // fails the SET — none of the kernels below could build without it, and
    // letting each discover that separately would cost one timeout apiece.
    final ready = _ensureReady;
    if (ready != null) {
      _emit(WarmProgress(
        phase: WarmPhase.running,
        completed: 0,
        total: _specs.length,
        failed: 0,
        elapsed: _stopwatch.elapsed,
        label: 'device',
      ));
      try {
        await ready();
      } catch (e) {
        _errors.add(WarmError('device', WarmStage.deviceLost, '$e'));
        _finish(WarmPhase.failed);
        return;
      }
    }
    var completed = 0;
    for (final spec in _specs) {
      if (_cancelled) {
        _finish(WarmPhase.cancelled);
        return;
      }
      _emit(WarmProgress(
        phase: WarmPhase.running,
        completed: completed,
        total: _specs.length,
        failed: _errors.length,
        elapsed: _stopwatch.elapsed,
        label: spec.label,
        errors: List.unmodifiable(_errors),
      ));
      final err = await _warmOne(spec);
      if (err != null) {
        _errors.add(err);
        // 🔴 A LOST DEVICE IS TERMINAL FOR THE SET. Everything after it would
        // fail the same way, each costing its own timeout — which is exactly
        // how a warm turns a bad device into a multi-minute hang.
        if (err.stage == WarmStage.deviceLost) {
          completed++;
          _emit(WarmProgress(
            phase: WarmPhase.running,
            completed: completed,
            total: _specs.length,
            failed: _errors.length,
            elapsed: _stopwatch.elapsed,
            errors: List.unmodifiable(_errors),
          ));
          _finish(WarmPhase.failed);
          return;
        }
      }
      completed++;
      _emit(WarmProgress(
        phase: WarmPhase.running,
        completed: completed,
        total: _specs.length,
        failed: _errors.length,
        elapsed: _stopwatch.elapsed,
        errors: List.unmodifiable(_errors),
      ));
    }
    _finish(_errors.isEmpty ? WarmPhase.ready : WarmPhase.failed);
  }

  /// Build one kernel. Returns null on success, or why it failed.
  Future<WarmError?> _warmOne(KernelWarmSpec spec) async {
    if (_built.contains(spec.source)) return null;
    final scratch = <Buffer>[];
    ComputeShader? shader;
    try {
      shader = _gpu.createComputeShader();
      shader.loadKernelString(spec.source);
      spec.buffers.forEach((tag, count) {
        final b = _gpu.createBuffer(count <= 0 ? 1 : count, BufferDataType.float32);
        scratch.add(b);
        shader!.setBuffer(tag, b);
      });
      // THE COMPILE IS THE DISPATCH. One workgroup, so the kernel runs on
      // whatever the scratch buffers contain and nobody reads the result —
      // building the pipeline is the entire point.
      await shader.dispatch(1, 1, 1).timeout(spec.timeout);
      _built.add(spec.source);
      return null;
    } on TimeoutException {
      return WarmError(spec.label, WarmStage.timeout,
          'did not finish within ${spec.timeout.inSeconds}s');
    } catch (e) {
      final msg = '$e';
      // "device lost" is the one failure that poisons everything after it.
      final lost = msg.toLowerCase().contains('device lost') ||
          msg.toLowerCase().contains('devicelost');
      return WarmError(spec.label,
          lost ? WarmStage.deviceLost : WarmStage.dispatch, msg);
    } finally {
      for (final b in scratch) {
        try {
          b.destroy();
        } catch (_) {}
      }
      try {
        shader?.destroy();
      } catch (_) {}
    }
  }
}
