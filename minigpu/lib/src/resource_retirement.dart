/// Shared once-only cleanup for a public GPU wrapper and its finalizer.
final class GpuResourceRetirement {
  GpuResourceRetirement(this._destroy, [this._unregister]);

  final void Function() _destroy;
  final void Function()? _unregister;
  bool _retired = false;

  bool get isRetired => _retired;

  void retire() {
    if (_retired) return;
    _retired = true;
    try {
      _destroy();
    } finally {
      _unregister?.call();
    }
  }
}
