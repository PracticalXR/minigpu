# CPU↔GPU Boundary Performance & Architecture

> Added: May 2026 — iterations 1–3 of the generalized GPU/CPU/GPU stage interface.

## Motivation

Every CPU stage in a streaming pipeline requires a round-trip across the GPU↔host boundary:

```
GPU tensor  →  mapAsync  →  native scratch  →  Dart TypedData   (read)
Dart TypedData  →  native scratch  →  Buffer.write  →  GPU tensor  (write)
```

Before this work each round-trip allocated a native scratch buffer, constructed
a new `NativeCallable` listener, and allocated a fresh `TypedData` on the Dart
heap — on every frame, for every CPU stage. This document describes the three
layers that eliminate that overhead and generalise the CPU execution model.

---

## Layer 1 — FFI scratch pooling (`minigpu_ffi`)

**File:** `minigpu_ffi/lib/minigpu_ffi.dart` — `FfiBuffer`

`FfiBuffer` now owns a reusable native scratch buffer and a persistent
`NativeCallable` that are allocated once and kept alive across frames.

| Before | After |
|---|---|
| `malloc(scratchBytes)` per read/write | pooled `_readScratch` / `_writeScratch` reused |
| `NativeCallable.listener(...)` per read | pooled `_readCallable` reused |
| `malloc.free(...)` per read/write | freed only in `destroy()` |

**Re-entrancy:** if a read or write is already in flight (another `await` in
the same stage), a local one-shot buffer is used as a fallback — safe because
Dart is single-threaded; re-entry can only happen across an `await`.

`destroy()` is guarded by `_destroyed` to prevent double-free.

---

## Layer 2 — `getData({T? into})` buffer reuse (`gpu_tensor`)

**File:** `gpu_tensor/lib/src/gpu_tensor_base.dart` — `Tensor.getData`

`getData()` now accepts an optional `into` parameter. When supplied it validates
the buffer is large enough and writes directly into it, bypassing the per-frame
`TypedData` allocation entirely.

```dart
// Allocate once
final scratch = Float32List(width * height * 4);

// Reuse every frame
final data = await tensor.getData(into: scratch) as Float32List;
```

When `into` is omitted the existing allocation path is used unchanged.

---

## Layer 3 — Pluggable `StageExecutor` + `StageConcurrency` (`gpu_pipeline`)

### `StageExecutor` (`lib/src/operation/stage_executor.dart`)

`CPUStageOperation` no longer hard-codes _how_ its processor runs. It delegates
to a `StageExecutor`:

```
abstract class StageExecutor {
  String get backend;
  Future<Map<String, TypedData>> run(inputs, ranks, parameters);
  FutureOr<void> dispose();
}
```

Two implementations are provided:

| Class | Backend string | Notes |
|---|---|---|
| `InlineExecutor` | `'inline'` | Runs on the calling isolate. **Required for stateful processors** (closed-loop encoders, accumulators). Default. |
| `PersistentIsolateExecutor` | `'isolate'` | Long-lived worker isolate. Processor must be a top-level or static function (Dart isolate sendability requirement). |

`PersistentIsolateExecutor` spawns the worker lazily on the first call, uses a
`ReceivePort`/`SendPort` request/reply protocol, and queues pending completers
by sequence number so multiple in-flight requests (if ever issued) are matched
correctly. `dispose()` sends a `_Shutdown` message and kills the isolate.

Usage:

```dart
// Stateless heavy filter — offload to worker isolate
CPUStageOperation(
  operationId: 'denoise',
  processor: denoiseTopLevel,
  executor: PersistentIsolateExecutor(denoiseTopLevel),
)

// Stateful AV1 pack — must stay inline
CPUStageOperation(
  operationId: 'av1_pack',
  processor: _packClosure,  // captures encoder state
  // executor omitted → InlineExecutor
)
```

The `isolateEnabled` constructor flag remains supported; when `true` and no
explicit `executor` is provided, a `PersistentIsolateExecutor` is wired
automatically (matching prior semantics but now with a real implementation
instead of a stub).

### `StageConcurrency` (`lib/src/pipeline_stage.dart`)

```dart
enum StageConcurrency { ordered, asyncSafe }
```

Added as a mutable field on `PipelineStage` (default: `ordered`):

```dart
/// Render/compute stage: must observe frame order
myComputeStage.stageConcurrency = StageConcurrency.ordered;  // default

/// Stateless filter: frames may overlap on the worker isolate
myFilterStage.stageConcurrency = StageConcurrency.asyncSafe;
```

`ordered` is always safe. `asyncSafe` should only be set when the processor
holds no cross-frame state. Combining `asyncSafe` with
`PersistentIsolateExecutor` gives automatic GPU/CPU overlap: the worker
processes frame N−1 while the GPU-submit isolate queues frame N.

