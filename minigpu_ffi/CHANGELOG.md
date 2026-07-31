# minigpu_ffi CHANGELOG

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
