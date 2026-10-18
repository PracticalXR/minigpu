/// Flutter companion for minigpu.
///
/// Re-exports the full `minigpu` public API so that `package:minigpu_flutter`
/// can be used as a drop-in replacement for `package:minigpu` in Flutter apps.
///
/// Additionally provides [MinigpuBinding], a thin root widget whose
/// [State.reassemble] fires all callbacks registered with
/// [MinigpuFlutterBinding.addDisposeCallback] on every Flutter hot RELOAD.
///
/// A hot reload keeps the Dart heap: every [Minigpu], buffer and shader the
/// app holds survives it, nothing is disposed and `initState` does not re-run.
/// So a callback may only drop what the app re-creates on its own when next
/// used (a cached kernel, so edited WGSL takes effect). Do NOT register
/// `gpu.destroySync`: nothing re-inits the context after the reload, so every
/// later dispatch fails until a full restart. A hot RESTART never runs
/// `reassemble`; keeping native work safe across one needs a native quiesce.
///
/// This is a teardown-ordering aid, NOT a crash guard: `reassemble` is
/// synchronous, so it can stop new work but cannot wait for work already
/// handed to the GPU worker thread. Completion safety lives in minigpu_ffi,
/// which delivers completions on a Dart native port.
///
/// ## Usage
///
/// ```dart
/// void main() {
///   runApp(const MinigpuBinding(child: MyApp()));
/// }
/// ```
///
/// Register a callback for GPU objects the app rebuilds when next used:
///
/// ```dart
/// MinigpuFlutterBinding.addDisposeCallback(effectCache.clear);
/// ```
library;

export 'package:minigpu/minigpu.dart';
export 'src/minigpu_flutter_binding.dart';
