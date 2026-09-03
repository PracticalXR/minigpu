# minigpu_ffi CHANGELOG

## 1.8.1

- Implement the new `PlatformBuffer.webBufferHandle` accessor (always 0 on native).

## 1.7.0

- **Persistent shader cache — Dawn's blob cache is now wired to disk, so
  compiled shaders survive process exit.** Dawn has always cached compiled
  shaders in memory, but it only writes through to storage when the embedder
  supplies `loadDataFunction` / `storeDataFunction` on a
  `DawnCacheDeviceDescriptor` chained onto the device descriptor. Nothing was
  chained, so every launch recompiled everything. On the D3D11 backend that
  means FXC, whose optimiser cost grows superlinearly with kernel size.
  Measured with a 3-kernel probe: total time in `wgpuDeviceCreateComputePipeline`
  **34 ms cold → 1 ms warm**, outputs byte-identical.

  New `src/src/shader_cache.cpp` implements the storage; `buffer.cpp` chains
  the descriptor at device creation and logs ONE summary line per device at
  INFO (per-entry hit/miss detail is DEBUG — this channel defaults to INFO and
  a line per blob would bury real warnings). `compute_shader.cpp` times
  pipeline creation into `pipelineCreateMs`.

  New exports: `mgpuShaderCacheSetEnabled`, `mgpuShaderCacheSetDirectory`,
  `mgpuShaderCacheSetCapBytes`, `mgpuShaderCacheSetExtraKey`,
  `mgpuShaderCacheSetProvider`, `mgpuShaderCacheClear`,
  `mgpuGetShaderCacheStats`, `mgpuGetShaderCacheDirectory`.

  Two environment overrides, following the `MGPU_BACKEND` / `MGPU_ADAPTER_NAME`
  / `MGPU_WAIT_PROF` convention: **`MGPU_SHADER_CACHE=0`** turns caching off and
  **`MGPU_SHADER_CACHE_DIR=<path>`** redirects it. Both OUTRANK the
  programmatic setters — an env var exists to change a binary you cannot edit,
  so a caller that hard-codes a setting must not be able to defeat it. This is
  the first thing to try when a shader misbehaves and you need to know whether
  the cache is involved.

  Implementation notes worth knowing before touching it:

  - **Entries store their full key and it is compared on read.** The filename
    is only a hash of the key. Dawn's own hash validation binds hash→value, not
    value→key, so on a filename collision it would accept a blob that is
    internally valid but belongs to a different pipeline. The stored-key
    compare is the only thing between a collision and a silently wrong
    pipeline; a mismatch is a miss.
  - **Nothing can fail device creation.** Unwritable directory, full disk,
    corrupt/truncated entry, lock, concurrent process — all become a miss.
  - Writes are atomic (temp file in the same directory, flushed, then renamed),
    and readers open with full sharing so eviction cannot fault an in-flight
    read.
  - Dawn's two-phase load (size, then fetch) is bridged by a single-entry memo,
    so the two calls cannot disagree and the file is read once per hit.
  - `mgpuShaderCacheSetProvider` is intentionally NOT exposed to Dart: Dawn's
    load callback is synchronous and may arrive on a Dawn-internal thread,
    while a Dart isolate can only be entered asynchronously from a foreign
    thread (`NativeCallable.listener`), which cannot return a blob to a blocked
    caller.
  - The file is inert under `__EMSCRIPTEN__` — the web build compiles the same
    source glob and has neither Dawn nor a filesystem.

- **`Buffer.writeRawBytes` no longer copies at all on the way to the GPU —
  `upload` measured 3.71 → 1.84 ms/frame at 4K (-50%).** The chunked path used
  to memcpy the payload into a native scratch first, purely so the FFI call had
  a stable address to pass; `wgpuQueueWriteBuffer` then copied it AGAIN into
  Dawn's staging ring. On a streaming path that is a whole-frame host memcpy per
  frame (33.2 MB at 4K) buying nothing. `TypedData.address` in a leaf call hands
  C the caller's own backing store, so the payload is copied exactly once, by
  the driver. The isolate-wide raw scratch is gone with it — one fewer 32 MiB
  pinned host allocation. Chunking (32 MiB) and byte ordering are unchanged; the
  chunked branch uses `Uint8List.sublistView`, which aliases rather than copies.
  Profiled confirmation that the removed time was pure waste: the time inside
  `wgpuQueueWriteBuffer` itself did not move (1.83 → 1.78 ms).

- **`mgpuWriteBufferAt` no longer lets a C++ exception escape into the caller.**
  `writeBytesAt` throws on an invalid context or an out-of-range write, and
  unwinding out of an FFI entry point is undefined behaviour for any caller — it
  never became a Dart exception, it corrupted the frame it unwound through.
  Now caught and logged. This became load-bearing with the leaf binding above,
  where the VM has not transitioned out of Dart state and an escaping exception
  is reliably fatal rather than merely undefined.

- **The read path's staging buffer grows on demand instead of being resized to
  fit every call.** It was destroyed and recreated whenever the next read asked
  for a different length — fine for a fixed-size tensor, pathological for the
  case this path actually serves, a compacted payload whose length changes every
  frame. Measured over 723 reads at 4K, the destroy/create cycle cost 33.05 ms;
  grow-on-demand costs 0.28 ms (-99%). Capacity only ever grows, to the next
  power of two, bounded by the source buffer's own size, and is released with
  the buffer — the same contract the batched-readback arena already shipped
  with. This is the sibling of the raw-upload scratch bug above: a per-frame
  allocate/free cycle wrapped around a transfer, costing more than the transfer.

- **New: `MGPU_WAIT_PROF=1` — per-site attribution for the blocking waits and
  per-call allocations on the transfer path**, with `MGPU_WAIT_PROF_MS=<n>` to
  log every individual event over n ms as it happens. It reports each site's
  count / total / mean / max AND the SPIN ITERATION COUNT of the waits, which is
  the one thing host-side stage timers cannot tell you: a wait that is long
  because the device is busy spins thousands of times, while a wait that is long
  because the OS descheduled the thread shows a long wall against a handful of
  iterations. Sites: the read path's lock / flush / staging-alloc / submit /
  map-wait / copy-out / trailing event pump, the write path's lock / flush /
  queue-write, the worker thread's enqueue-to-start latency, and the shared
  texture drain. Off by default (one cached bool check).

- **`Buffer.writeRawBytes` no longer `malloc`s its staging scratch per call —
  ~3 ms/frame and a much fatter tail off any streaming upload.** It is the
  whole-frame path (one call per frame, tens of times a second), and at 4K the
  allocator cost more than the copy it was wrapping: 33.2 MB x 240 iterations
  measured 4.55–4.67 ms median for allocate + copy + free against 1.31–1.69 ms
  for the same copy through a scratch allocated once, with p99 6.14–8.72 ms
  against 2.56–4.06 and a 12.8–15.6 ms worst case against 3.4–6.1. A per-frame
  33 MB commit / first-touch / decommit cycle is a frame-time spike source in
  its own right, which is what the tail numbers are showing. The scratch is now
  one ISOLATE-WIDE buffer, grown on demand and bounded by the existing 32 MiB
  chunk size no matter how many buffers stream through it — deliberately not a
  per-buffer field, because pinning a transfer-sized scratch on every live
  buffer is exactly what the 1 MiB `_maxPooledScratchBytes` cap exists to
  prevent (thousands of live weight buffers). No API or semantic change; byte
  ordering and chunking are unchanged.

- **Async completions are delivered on a Dart native port — fixes the VM abort
  `runtime_entry.cc: Callback invoked after it has been deleted` (whole-process
  death, surfaced to the user as `Lost connection to device.`).** Every async
  entry point built a `NativeCallable.listener` per call and `close()`d it in a
  `finally`. That is unsound: `close()` DELETES the trampoline, this library
  cannot be told to forget a pointer it has already been handed (there is no
  cancel), and the dart:ffi contract for invoking a closed callable is
  undefined behaviour that the VM implements as an unconditional `FATAL` —
  uncatchable, no stack, takes the app with it. Completions fire on the WebGPU
  worker thread, so a busy pipeline re-armed the race 5–10 times per frame.
  `FfiBuffer.destroy()` was the same bug without a race: it closed the pooled
  readback listener with no in-flight check at all. And `close()` was never the
  only deleter — ISOLATE TEARDOWN deletes every callback the isolate owns, so
  hot restart, a hard kill, and any worker isolate exiting with work in flight
  were fatal too, which no amount of Dart-side discipline could have covered.
  Posting to a Dart port that is closed, or whose isolate is gone, is a
  defined, silent no-op instead. New exports carry the destination with each
  call — no registration, so completions cannot be misrouted between isolates:
  `mgpuInitializeContextAsyncToPort`, `mgpuContextInitializeAsyncToPort`,
  `mgpuDispatchAsyncToPort`, `mgpuReadAsyncToPort`,
  `mgpuCopyBufferToSharedOutputTextureAsyncToPort` and
  `mgpuVideoTextureBGRAToRGBASharedOutputAsyncToPort`. Call
  `mgpuInitDartApi(NativeApi.initializeApiDLData)` once first (the same bridge
  the log port uses). WIRE FORMAT: one int64 per completion, `(token << 1) |
  ok`; tokens are allocated by Dart and never reused, so a late or duplicate
  completion is a lookup miss rather than somebody else's future resolving
  early — a silent-corruption window the callback form could not close. Every
  port entry point posts EXACTLY ONCE, including on argument-validation
  failures, so a bad call now fails instead of hanging forever.
  The `MGPUCallback` variants remain for embedders that own their function's
  lifetime, and back a pooled never-closed fallback used only where the Dart
  API is unavailable (the Emscripten build). Delivery is the same mechanism
  either way — `NativeCallable.listener` is itself a port post underneath — so
  there is no latency cost, and the per-call trampoline allocation is gone.
  `MGPU_UNSAFE_CALLBACK_COMPLETIONS=1` forces the old callback path; it exists
  only so the abort can be reproduced on demand (an isolate exiting with work
  in flight kills the process under it, passes without it) and must never be
  set in production.
  FULLY BACKWARD COMPATIBLE: no Dart or C signature changed, every previous
  export still exists and still behaves as before, and the Dart side PROBES for
  one of the new symbols before committing to the port path — the Dart API
  bridge shipped before these entry points did, so a binary older than the Dart
  code (or a stale one served from a build cache) falls back cleanly instead of
  throwing from the middle of a frame.
- **`mgpuDrainWorkQueue()`**, surfaced in Dart as
  `MinigpuPlatform.drainWorkQueue()` — blocks until everything already queued on
  the WebGPU worker thread has run. Teardown ordering: destroying a resource an
  enqueued task still references frees it under that task. Synchronous, because
  the caller that needs it most cannot await (Flutter's `reassemble` on hot
  reload). No-op when called from the worker thread itself, which cannot drain
  the queue it heads, and on a binary predating the export.
- **The build hook now declares `src/**` as build dependencies.** Only the Dawn
  DLL was declared, so editing the C/C++ never re-ran the hook and the runner
  served the previously built library — edits appeared to do nothing, silently,
  because the Dart side still compiled and the stale binary behaved exactly as
  it always had. Recognise it by a newly added export missing from the DLL
  while everything else works, and verify a NEW SYMBOL, never the mtime.
- Element-type codes for `mgpuReadAsyncToPort` are a dedicated `MGPUElementType`
  enum rather than `BufferDataType`: the Dart and C++ `BufferDataType` enums are
  ordered DIFFERENTLY, so an `.index` crossing the boundary is ambiguous.

## 1.6.1

- released 08/13/26 - MR

## 1.6.0

- **The process-global context is now serialized and reference-counted.**
  `mgpuInitializeContext` / `mgpuInitializeContextAsync` ATTACH,
  `mgpuDestroyContext` DETACHES, and teardown happens only when the last
  attached consumer leaves; new export `mgpuContextRefCount()` reports the
  outstanding attaches. Init is idempotent under concurrency: a caller that
  arrives while another thread is creating the device waits for it and shares
  the result. Lazy internal re-initialization (`getDevice` / `getQueue` /
  `isDeviceValid` / `ensureDeviceValid`) does NOT take a reference.
  Single-consumer behaviour is unchanged: attach → 1 → real init, detach → 0 →
  real teardown. TRAP this fixes: `initializeContext()` bailed out only on
  `ctx && ctx->initialized`, so while one thread sat in the multi-millisecond
  adapter/device request with `initialized` still false, another thread's lazy
  `getDevice()` fell straight through the guard and re-assigned `ctx` — freeing
  a live Context under a thread still holding raw pointers into it. Any host
  that puts several independent consumers in ONE process (Dart isolates being
  the obvious case) hit it as an access violation, and the surviving symptom
  when it did not crash was one consumer's teardown killing another's device.
  SECOND TRAP, for anyone touching this code: the lock must NOT be held across
  the device request — teardown drains the WebGPU thread with a synchronous
  enqueue, so a lock held across the slow work deadlocks. The state machine
  holds it only across transitions, and the async init now runs on a one-shot
  thread rather than the WebGPU thread for the same reason. Consequence worth
  knowing: a consumer that exits without detaching keeps the context alive for
  the life of the process — deliberate, and much cheaper than the alternative.
  `MGPU_UNSAFE_NO_CTX_LOCK=1` restores the old unguarded behaviour (it exists
  only to demonstrate the regression; it is the bug).
- **`minigpu_external`'s cached D3D11 device/immediate context are locked and
  dropped on teardown.** The lazy creation was an unlocked double-check —
  concurrent callers each created a device and the second assignment released
  the first out from under a pointer already handed out — and
  `ID3D11DeviceContext` is not free-threaded, so every immediate-context use in
  that file now takes the same lock (consumer-side debug device likewise; lock
  order is consumer → producer). TRAP: on the D3D11 backend the cached
  `ID3D11Device` IS Dawn's own device, so it used to survive a destroy/re-init
  pointing at the dead one and the next `createSharedOutputTexture` silently
  took the cross-device path and returned null.

- **`setLogCallback` now delivers over a Dart native port, not a
  `NativeCallable`.** New exports `mgpuInitDartApi(void*)` and
  `mgpuSetLogPort(int64)`; log lines arrive as `[int32 level, Uint8List utf8]`
  (bytes, because driver strings are not always valid UTF-8 — nothing to free,
  `mgpuFreeLogMessage` does not apply to this path). TRAP this fixes: the log
  registry is PROCESS-GLOBAL, so a `NativeCallable` registered by one isolate
  outlived it — after that isolate exited, the next line a Dawn worker thread
  logged ran a deleted trampoline and aborted the whole VM process
  (`Callback invoked after it has been deleted`). A whole-suite `dart test` run
  reproduced it every time and took down unrelated downstream suites. A closed
  port is inert. Semantics are unchanged otherwise: process-global, LAST WRITER
  WINS, and the registration is replaced natively *before* the old port is
  closed. `mgpuSetLogCallback` stays exported for non-Dart embedders. Web/wasm
  is unaffected (`MINIGPU_HAVE_DART_DL` is native-only; the Dart API is vendored
  under `src/third_party/dart_dl`).

## 1.5.9

- **`mgpuDestroyComputeShader` no longer deletes the shader inline** — it queues
  the delete on the WebGPU FIFO (new `ComputeShader::destroyQueued`), so it lands
  after any bind or dispatch already queued against that shader.
  Required by the ordered binds in minigpu 1.5.9: `setBufferQueued` captures the
  shader pointer and mutates its binding tables when the task runs, so an inline
  delete freed the object under a pending bind. This retires the caller-side
  constraint the fire-bind docs used to carry ("do not destroy the shader until a
  read has been awaited").
  Buffers are unaffected and still delete immediately — a queued bind captures
  only the raw `WGPUBuffer` handle by value, never the `Buffer` object. Both
  sites are commented with that asymmetry.
  Safe against re-entrancy: the destructor enqueues its own handle-release task,
  and the worker pops a task under `queueMutex` and runs it with the lock
  released, so enqueueing from inside a task cannot deadlock.

## 1.5.8

- **Fix: `MINIGPU_DAWN_DIR` was ignored when building through Flutter / dart
  pub.** The build hook always passes `-DDAWN_DIR=<platform central path>`, and
  because `dawn.cmake` only consults the env var when `DAWN_DIR` is undefined,
  that define outranked it — the env var changed only where the hook LOOKED for a
  prebuilt `webgpu_dawn` library, while cmake was still told
  `%SYSTEMDRIVE%\dawn` and cloned/built Dawn there. The hook now resolves
  `MINIGPU_DAWN_DIR` first for the define too.
- **Fix: a Windows Dawn root broke a from-source Dawn build with "Invalid
  character escape".** `DAWN_DIR` reached `FetchContent_Declare` with
  backslashes (`C:\dawn`), and FetchContent writes `SOURCE_DIR` / `BINARY_DIR`
  verbatim into the sub-build `CMakeLists.txt` it generates, where `\d` is an
  invalid escape — configure failed inside a generated file, pointing at
  `<dawn>/build_win_x86_64/tmp/CMakeLists.txt`. `dawn.cmake` now runs
  `file(TO_CMAKE_PATH …)` on `DAWN_DIR`, and the hook emits forward slashes.
  Only machines WITHOUT a prebuilt Dawn were affected: when `ENABLE_DAWN_FIND`
  finds an existing build, the FetchContent branch never runs.
- **Fix: `make build_weblib` (the Emscripten build) was broken by the
  `DAWN_COMMIT` bump.** `--use-port=` hardcoded a version-stamped
  `emdawnwebgpu-v<stamp>.remoteport.py` — a standalone port older Dawn shipped —
  so em++ failed with "not a valid port path", and only ~370 targets in, after
  Dawn itself had compiled. Current Dawn requires the port to come from its
  ASSEMBLED package: the port file checks for generated headers copied in beside
  it and otherwise errors "must sit in a built emdawnwebgpu_pkg". `--use-port`
  now points at `${CMAKE_BINARY_DIR}/emdawnwebgpu_pkg/emdawnwebgpu.port.py`, and
  `webgpu_web` takes a build dependency on Dawn's `emdawnwebgpu_pkg` target that
  produces it — without that edge ninja may compile before the package exists.
  The stamped layout is still accepted for older pins, the resolved path is
  logged as `emdawnwebgpu port -> …`, and a Dawn tree with neither layout is a
  configure-time `FATAL_ERROR` naming `DAWN_COMMIT` rather than a confusing
  failure deep in the build.
- **Fix: the Dawn event drain no longer costs a Windows timer quantum per GPU
  wait.** `drain_dawn_events_with_timeout` waited with a 1 ms timeout on a future
  nothing could notify early (the promise is set from a callback that runs later
  in the same loop), and a 1 ms wait on Windows rounds up to the ~15.6 ms system
  timer granularity — so every GPU wait cost a whole tick. It now probes with a
  zero timeout and yield-spins for a bounded budget before degrading to coarse
  sleeping. Measured p50 on an RTX 4090: shared-texture present 15.69 -> 2.57 ms
  at 1280x720, 15.69 -> 10.07 ms at 3840x2160. Affects every caller that waits
  through this drain (present, Dawn-side debug reads, video import).
  `MGPU_DRAIN_SPIN_MS=<0..1000>` tunes the budget (default 8); `0` restores the
  pre-fix behaviour, for A/B only. It is parsed strictly and warns on both a
  malformed value and an explicit `0` — it previously used `std::atoi`, so
  `on`/`true`/a typo silently selected the pathological 0.
- New `mgpuDrainSpinBudgetMs()` — the budget the loaded binary implements, so a
  caller can assert it is not running a stale artifact.
- New ADDITIVE batched-staging-upload scope: `mgpuBeginUploads` /
  `mgpuStageWrite` / `mgpuStageReserve` / `mgpuEndUploads` (+
  `mgpuUploadsSupported`, `mgpuUploadStats`, `MGPU_UPLOAD_PROF=1`, and
  `mgpuContextBeginUploads` / `mgpuContextEndUploads` for multi-GPU). N
  scattered host writes become ONE `wgpuQueueWriteBuffer` into a persistent
  staging buffer plus N `copyBufferToBuffer` recorded into the compute batch
  already being built — one mutex acquire, zero extra submits, unchanged
  ordering. A scope must not span a dispatch that READS what it stages. With no
  scope open (or an unaligned range) `mgpuStageWrite` writes inline.
- New ADDITIVE batched-readback scope: `mgpuBeginReadbacks` / `mgpuStageRead` /
  `mgpuReadbackReserve` / `mgpuEndReadbacks` (+ `mgpuReadbacksSupported`,
  `mgpuReadbackStats`, `MGPU_READBACK_PROF=1`, and `mgpuContextBeginReadbacks` /
  `mgpuContextEndReadbacks`). N per-buffer `mgpuReadSync*` calls — each taking
  the device mutex, flushing the batch with its own submit and blocking for GPU
  completion — become ONE submit, ONE fence and ONE map into a persistent
  readback buffer. Measured on an RTX 4090: 8 reads totalling 2.72 MB,
  1.098 -> 0.465 ms of API time. Three contract differences from the upload
  twin: a staged read is not filled until `mgpuEndReadbacks` (keep `dst` alive,
  do not inspect it earlier); every staged read observes its source as of End,
  so a scope must not span a dispatch that OVERWRITES what it stages; and
  `mgpuStageRead` is a RAW BYTE read (mirror of `mgpuWriteBufferAt`, not of
  `mgpuReadSyncUint8`). Unaligned or scopeless reads fall back inline.
- Both scopes resolve INLINE on the calling thread under the same device mutex
  the inline read/write paths take. Emitting on minigpu's WebGPU thread would
  race, because `mgpuReadSync*` / `mgpuWrite*` do not join that FIFO.
- New multi-adapter context handles: `mgpuCreateContextHandle(adapterFilter)`,
  `mgpuContextInitializeAsync`, `mgpuContextGetAdapterName`,
  `mgpuContextCreateBuffer`, `mgpuContextCreateComputeShader`,
  `mgpuDestroyContextHandle` — an independent device, queue and task FIFO per
  handle, exposed in Dart as `createSecondaryPlatform`. Resources from two
  contexts must never be mixed in one dispatch.
- New `mgpuEnumAdapters(namesOut, totalOut, usedOut, cap)` (DXGI) plus the Dart
  `listAdapters()` override returning `GpuAdapterInfo` — adapter name, total
  dedicated VRAM, current usage.
- New `mgpuSetBufferFire` plus Dart `setBufferFire` / `dispatchFire`:
  `dispatchFire` drops the per-dispatch completer round trip (the enqueue was
  already async) and `setBufferFire` makes the bind join that same FIFO, so
  bind -> fire -> rebind -> fire is ordered. Bindings are snapshotted when the
  dispatch RUNS, not when it is fired.
- New Dart `Buffer.writeRawBytes`: 4-byte-aligned raw upload streamed through a
  host scratch allocation of at most 32 MB, so a multi-GB write never pins its
  own size in host RAM or driver staging. `ArgumentError` on a misaligned
  offset or length.
- New `mgpuDebugConsumerChecksumSharedHandle(handle, w, h)` — FNV-1a over the
  shared output surface read through an independent ID3D11Device, i.e. the way a
  compositor sees it. Unlike `...DebugReadFirstPixel` (which reads through
  Dawn's own device) it can observe a half-written present, making it usable as
  a tearing/staleness oracle. Verification only.
- New `MGPU_PRESENT_NO_WAIT=1` — deliberately broken, test-only: drops the
  present's completion wait so the oracle above can be positive-controlled.
  Never set it in production.

## 1.5.7

- Release cut of the adapter-selection / Tier B–Tier C work documented under
  1.5.6; no additional API change.

## 1.5.6

- New pre-init hint `mgpuPreferDisplayAdapter(int enable)` (+ Dart
  `preferDisplayAdapter`): bind Dawn to the adapter driving the PRIMARY
  display so screen capture (Desktop Duplication / WGC), GPU processing and
  any D3D11 encoder created on Dawn's adapter share one GPU — same-adapter
  zero-copy — even on multi-output hybrid systems where the discrete GPU
  also drives a monitor (there the automatic "dGPU has no outputs" topology
  detection cannot see the capture/compute split, so capture used to fall to
  the Tier C CPU bridge). Returns whether the hint landed before context
  init; `MGPU_ADAPTER_NAME` still overrides. Also new:
  `mgpuGetSelectedAdapterName` to query which adapter Dawn actually bound.
- Fix hybrid-laptop backend auto-select probing the wrong adapter's
  cross-adapter capability. Tier B's cross-adapter texture is allocated on the
  *producer* (display / iGPU) device, so its viability is gated by the
  producer's `CrossAdapterRowMajorTextureSupported` — but startup detection
  tested the *compute* (dGPU) adapter instead. On machines where the dGPU
  advertises the capability but the iGPU does not, minigpu committed to the
  D3D12 / Tier-B backend and then degraded to the Tier C CPU bridge on every
  frame ("Tier B: adapter does not support CrossAdapterRowMajorTextureSupported.
  Falling back to Tier C."). Now the display (capture-producer) adapter's
  capability is what selects Tier B vs Tier A*; when it lacks the capability we
  take Tier A* (run Dawn on the iGPU) for zero-copy capture import and keep the
  FFmpeg HW encoder on the iGPU. Also made the Tier A* adapter binding robust:
  if the exact display-name match misses, auto-select now prefers the
  integrated GPU rather than falling back to the discrete one.

## 1.5.5

- Raise per-device GPU submission priority (`IDXGIDevice::SetGPUThreadPriority(+7)`)
  on the cached D3D11 device (both the Dawn-native-D3D11 fast path and the
  created-on-Dawn-adapter path), so minigpu's compute/copy submissions are less
  starved when another process saturates the GPU. Best-effort; failure is logged.
- Tier B (D3D12 cross-adapter bridge) probing now caches a sticky negative
  result per adapter: when the adapter lacks
  `CrossAdapterRowMajorTextureSupported` (or any Tier B init step fails), the
  probe used to re-run `D3D12CreateDevice` and re-log "Falling back to
  Tier C" on every call — frame-rate log spam plus wasted per-frame device
  creation. It now probes and warns once per adapter per process.

## 1.5.4

- fix release version pins

## 1.5.3

- Implement `copyFromBufferAsync` / `bgraToRgbaSharedOutputAsync`. New native
  exports `mgpuCopyBufferToSharedOutputTextureAsync` /
  `mgpuVideoTextureBGRAToRGBASharedOutputAsync` run the copy (including the
  `wgpuQueueOnSubmittedWorkDone` present sync) on the WebGPU worker thread and
  invoke a Dart `NativeCallable` completion callback, so the calling isolate is
  no longer blocked on the present-wait busy-poll.

## 1.5.2

- Add buffer copy

## 1.5.1

- fix frame wait and timeouts

## 1.5.0

- Fix handle issue, release 1.5.0

## 1.4.15

- fix fallback paths

## 1.4.14

- fix Tier C

## 1.4.12

- fix texture path

## 1.4.11

- fixing build hook

## 1.4.9

- tryfix central dawn location

## 1.4.8

- fixes texture path on windows, fixes texture view on web

## 1.4.7

- Fixes logger characters

## 1.4.6

- fix garbled adapter name in logs: WGPUStringView.data is not null-terminated, copy to std::string before passing to snprintf
- fix garbled adapter name in logs: WGPUStringView.data is not null-terminated, copy to std::string before passing to snprintf

## 1.4.5

- fix Dawn built inside pub-cache instead of system dir: pass DAWN_DIR from Dart hook as cmake -D define so cmake subprocess inherits the correct path regardless of env; fix FETCHCONTENT_BASE_DIR pointing to pub-cache (now uses cmake binary dir)
- add Minigpu.setLogCallback / setLogLevel: routes native Dawn/GPU log lines through a Dart callback (NativeCallable.listener); mgpuSetLogCallback + mgpuSetLogLevel exported from C layer; all stderr calls in minigpu_external.cpp replaced with structured LOG_ERROR/INFO/WARN/DEBUG macros

## 1.4.4

- fix FormatException on non-UTF-8 bytes in setLogCallback: use Utf8Decoder(allowMalformed: true) instead of toDartString()

## 1.4.3

- add dawn::native::Instance + EnumerateAdapters for explicit adapter type selection
- improve adapter selection to prefer discrete GPU using dawn native EnumerateAdapters; fixes incorrect adapter picked on Optimus laptops

## 1.4.2

- fixes dawn library not being found

## 1.4.1

- added bindings observer

## 1.4.0

- adds minigpu_view, gpu_pipeline libraries

## 1.3.0

- Adds Texture Sharing

## 1.2.4-WIP

- working on texture imports

## 1.2.3

- adds VRAM API
- fixes memleaks and broken tests

## 1.2.2

- fixes memleaks and broken tests

## 1.2.1

- Fix: fresh builds need dawn find off
- Change: 1.2.0 migrates minigpu to direct webgpu usage
- Breaking: Changed setData and references to .write
- Fix: broken compute shader and buffer finalizers

## 1.2.0

## 1.1.9

- fix pubspec version issue

## 1.1.8

- fixed concurrent buffer op crash

## 1.1.6

- fixed problem with audio input capture providing raw data

## 1.1.5

- Fixed memory leaks
