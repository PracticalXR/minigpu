/// Flutter companion for minigpu.
///
/// Re-exports the full `minigpu` public API so that `package:minigpu_flutter`
/// can be used as a drop-in replacement for `package:minigpu` in Flutter apps.
///
/// Additionally provides [MinigpuBinding], a thin root widget whose
/// [State.reassemble] fires all callbacks registered with
/// [MinigpuFlutterBinding.addDisposeCallback] during Flutter hot reload.
/// This allows GPU contexts to be torn down cleanly before the Dart isolate
/// is rebuilt, preventing stale [NativeCallable] invocations.
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
