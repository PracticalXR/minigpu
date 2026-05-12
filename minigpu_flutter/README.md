# minigpu_flutter

Flutter companion for [minigpu](../minigpu). Re-exports the full `minigpu` API and adds a thin widget that fires registered teardown callbacks during Flutter hot reload — preventing stale `NativeCallable` invocations when the Dart isolate is rebuilt mid-dispatch.

## Why this package exists

`minigpu` creates short-lived `NativeCallable` handles for each GPU dispatch, read, or write operation. If a Flutter hot reload tears down the isolate while one of these operations is in-flight, the `finally` block that closes the handle never runs. The next time the C GPU callback fires it invokes a dead function pointer and the VM aborts unconditionally.

`MinigpuBinding` is a `StatefulWidget` whose `reassemble()` synchronously calls all callbacks registered via `MinigpuFlutterBinding.addDisposeCallback`, allowing you to destroy GPU contexts before the isolate is rebuilt.

## Installation

```yaml
dependencies:
  minigpu_flutter: ^1.4.1
```

Use `minigpu_flutter` **instead of** `minigpu` — it re-exports everything, so no other imports need to change.

## Usage

### 1 — Wrap your root widget

```dart
import 'package:minigpu_flutter/minigpu_flutter.dart';

void main() {
  runApp(const MinigpuBinding(child: MyApp()));
}
```

### 2 — Register a teardown callback for each long-lived `Minigpu` instance

```dart
final gpu = Minigpu();
await gpu.init();

// Called synchronously during hot reload.
MinigpuFlutterBinding.addDisposeCallback(gpu.destroySync);
```

### 3 — Unregister when the resource is torn down normally

```dart
MinigpuFlutterBinding.removeDisposeCallback(gpu.destroySync);
await gpu.destroy();
```

### Pure-Dart projects

If you are **not** using Flutter (e.g. a CLI tool or a Dart-only test), import `package:minigpu/minigpu.dart` directly and call `gpu.destroy()` / `gpu.destroySync()` in your own teardown logic.

## API surface

Everything exported by `package:minigpu/minigpu.dart`, plus:

| Symbol | Description |
|--------|-------------|
| `MinigpuBinding` | Root widget — `reassemble()` fires all registered dispose callbacks |
| `MinigpuFlutterBinding.addDisposeCallback(fn)` | Register a synchronous teardown callback |
| `MinigpuFlutterBinding.removeDisposeCallback(fn)` | Unregister a previously registered callback |
