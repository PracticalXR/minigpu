# minigpu_flutter

Flutter companion for [minigpu](../minigpu). Re-exports the full `minigpu` API and adds a thin widget that runs your GPU teardown at hot reload, before Flutter rebuilds the widget tree.

## Why this package exists

`MinigpuBinding` is a `StatefulWidget` whose `reassemble()` synchronously calls every callback registered via `MinigpuFlutterBinding.addDisposeCallback`. Hot reload is the one moment a long-lived GPU resource has no other teardown hook: nothing is disposed, `initState` does not re-run, and your app keeps whatever contexts, buffers and shaders it had. Register the ones that should not survive a reload and they get torn down deterministically instead of leaking or being rebuilt on top of themselves.

**It is not a crash guard.** It used to be described as one: minigpu built a short-lived `NativeCallable` per dispatch/read/write, and a completion that arrived after Dart closed the handle aborted the whole process (`Callback invoked after it has been deleted`). That was a real bug and this widget could not have fixed it — `reassemble()` is synchronous, so it can stop new work but cannot wait for work already handed to the GPU worker thread, and isolate teardown deletes the callbacks whether or not Dart closed them. The fix belonged in minigpu_ffi, where completions now arrive on a Dart native port that is silently inert once its isolate is gone. Upgrade minigpu to get it; use this widget for teardown ordering, not for safety.

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
