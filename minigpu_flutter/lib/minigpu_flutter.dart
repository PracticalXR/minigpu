/// Flutter companion for minigpu.
///
/// Re-exports the full `minigpu` public API so that `package:minigpu_flutter`
/// can be used as a drop-in replacement for `package:minigpu` in Flutter apps.
///
/// Additionally provides [MinigpuBinding], a thin root widget whose
/// [State.reassemble] fires all callbacks registered with
/// [MinigpuFlutterBinding.addDisposeCallback] during Flutter hot reload —
/// the one moment a long-lived GPU resource has no other teardown hook
/// (nothing is disposed, `initState` does not re-run).
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
/// Register teardown callbacks for any long-lived GPU resources:
///
/// ```dart
/// final gpu = Minigpu();
/// await gpu.init();
/// MinigpuFlutterBinding.addDisposeCallback(gpu.destroySync);
/// ```
library;

export 'package:minigpu/minigpu.dart';
export 'src/minigpu_flutter_binding.dart';
