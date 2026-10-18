# minigpu_flutter

Flutter companion for [minigpu](../minigpu). Re-exports the full `minigpu` API and adds a thin widget that runs your registered callbacks at hot reload, before Flutter rebuilds the widget tree.

## Why this package exists

`MinigpuBinding` is a `StatefulWidget` whose `reassemble()` synchronously calls every callback registered via `MinigpuFlutterBinding.addDisposeCallback`. A hot reload keeps the Dart heap: nothing is disposed, `initState` does not re-run, and your app keeps every context, buffer and shader it had. Register callbacks that drop objects your app rebuilds the next time it uses them (a cache of effects, say), so the reloaded code takes effect. Never register `gpu.destroySync`: nothing runs `gpu.init()` after a reload, so every later dispatch would fail until a restart. A hot restart never runs `reassemble()`.

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

### 2 — Register callbacks for objects you rebuild on next use

```dart
// Called synchronously during hot reload. The cache rebuilds an effect from
// the reloaded code the next time it is asked for one.
MinigpuFlutterBinding.addDisposeCallback(effectCache.clear);
```

Keep the `Minigpu` instance itself out of this list: a reload does not run `gpu.init()` again.

### 3 — Unregister when the object is torn down normally

```dart
MinigpuFlutterBinding.removeDisposeCallback(effectCache.clear);
```

### Pure-Dart projects

If you are **not** using Flutter (e.g. a CLI tool or a Dart-only test), import `package:minigpu/minigpu.dart` directly and call `gpu.destroy()` / `gpu.destroySync()` in your own teardown logic.

## API surface

Everything exported by `package:minigpu/minigpu.dart`, plus:

| Symbol | Description |
|--------|-------------|
| `MinigpuBinding` | Root widget — `reassemble()` fires all registered dispose callbacks |
| `MinigpuFlutterBinding.addDisposeCallback(fn)` | Register a synchronous callback run at hot reload |
| `MinigpuFlutterBinding.removeDisposeCallback(fn)` | Unregister a previously registered callback |
