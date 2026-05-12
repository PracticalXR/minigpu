import 'package:flutter/widgets.dart';

/// A thin widget that fires registered [MinigpuFlutterBinding] teardown
/// callbacks during Flutter hot reload.
///
/// Wrap your root widget with [MinigpuBinding]:
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
/// MinigpuFlutterBinding.addDisposeCallback(gpu.destroySync);
/// ```
///
/// Unregister when the resource is torn down normally:
///
/// ```dart
/// MinigpuFlutterBinding.removeDisposeCallback(gpu.destroySync);
/// gpu.destroy(); // normal teardown path
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
