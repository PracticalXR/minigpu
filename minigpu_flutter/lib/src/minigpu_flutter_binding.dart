import 'package:flutter/widgets.dart';

/// A thin widget that fires registered [MinigpuFlutterBinding] teardown
/// callbacks on every Flutter hot reload.
///
/// Wrap your root widget with [MinigpuBinding]:
///
/// ```dart
/// void main() {
///   runApp(const MinigpuBinding(child: MyApp()));
/// }
/// ```
///
/// A hot reload keeps every live GPU object and does not re-run `initState`,
/// so register callbacks only for what the app rebuilds when next used:
///
/// ```dart
/// MinigpuFlutterBinding.addDisposeCallback(effectCache.clear);
/// ```
///
/// Never `gpu.destroySync`: nothing re-inits the context after the reload.
/// A hot restart never reaches `reassemble`.
///
/// Unregister when the resource is torn down normally:
///
/// ```dart
/// MinigpuFlutterBinding.removeDisposeCallback(effectCache.clear);
/// ```
class MinigpuBinding extends StatefulWidget {
  const MinigpuBinding({super.key, required this.child});

  final Widget child;

  @override
  State<MinigpuBinding> createState() => _MinigpuBindingState();
}

class _MinigpuBindingState extends State<MinigpuBinding> {
  @override
  void reassemble() {
    super.reassemble();
    for (final cb in List.of(MinigpuFlutterBinding._callbacks)) {
      try {
        cb();
      } catch (_) {
        // Never let one bad callback block the others.
      }
    }
  }

  @override
  Widget build(BuildContext context) => widget.child;
}

/// Registry for synchronous teardown callbacks invoked during hot reload.
///
/// Callbacks are called in registration order inside [MinigpuBinding]'s
/// [State.reassemble] method.
abstract final class MinigpuFlutterBinding {
  static final List<void Function()> _callbacks = [];

  /// Adds [callback] to the list of functions called on hot reload.
  /// [callback] must be synchronous.
  static void addDisposeCallback(void Function() callback) {
    _callbacks.add(callback);
  }

  /// Removes a previously registered [callback].  A no-op if [callback] was
  /// not registered.
  static void removeDisposeCallback(void Function() callback) {
    _callbacks.remove(callback);
  }
}
