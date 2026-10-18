# minigpu_ffi CHANGELOG

## 1.9.1

- The web (Emscripten) build in `src/` is an optimized release build unless a
  build type is given. Pass `-DCMAKE_BUILD_TYPE=Debug` for debug info. Native
  builds are unchanged.

## 1.9.0

- The shipped library is no longer tuned to the build machine's CPU
  (`-march=native`, which could crash on other CPUs) or built with fast-math
  (which removed the NaN and infinity checks).
- A typed write past the end of a buffer throws `RangeError` instead of
  aborting the process. A failed read throws `StateError` instead of handing
  back stale bytes. A `ByteData` read gets the raw bytes at its start. New
  `mgpuReadSyncChecked` returns whether a blocking read landed.
- The device is created with Dawn's own locking on, so minigpu's threads no
  longer race inside Dawn (for example a submitted-work wait against a write).
- A C++ exception no longer aborts the process from an exported call. Each
  export reports the failure through its usual result (null, 0, -1 or
  `false`), and a refused call that takes a callback still calls it.
- Destroying a buffer while a read of it is queued, or while an open upload
  or readback scope still uses it, waits for them instead of freeing memory
  under them.
- Reads and writes at offsets that are not a multiple of 4 bytes work; they
  used to return stale bytes or write nothing. 8- and 16-bit reads at an odd
  element offset return the right elements, and 8-, 16- and 64-bit reads and
  writes no longer convert element by element.
- `mgpuStageRead` and `mgpuStageWrite` return -1 when their direct fallback
  fails, instead of reporting success.
- Writes, uploads and the end of an upload or readback scope keep call order
  with queued dispatches and render passes, so a write after
  `await pass.submit()` no longer reaches the GPU first. At most 64 MB of such
  writes wait in the queue; past that, a write waits for the GPU thread.
- Waiting for a buffer map, for submitted work or for a GPU error check no
  longer spins a core, and a map wait no longer holds the device lock. The GPU
  thread runs other work while a submitted-work wait is pending.
- Native logging defaults to warnings and errors, and a call below the level
  costs almost nothing. Clearing the log callback in one isolate no longer
  disconnects another isolate's listener.
- A failed context init calls its callback and frees its Dawn instance. The
  Windows adapter list stops at the first failure instead of reading a null
  adapter.
- Calls that take a context handle refuse a destroyed or unknown context
  instead of reading freed memory.
- `createBuffer` returns null for a negative size or one over the device's
  limit, instead of an error buffer that looked valid. From Dart, a size that
  does not fit in 32 bits throws `RangeError` instead of being truncated.
- At exit, the GPU thread finishes before the state it uses is destroyed, and
  a task that throws no longer stops the thread.
- An awaited dispatch reports whether it was submitted. A dispatch over the
  device's workgroup limit, or a kernel or bind group that fails to build, is
  skipped and logged instead of breaking every other dispatch in the same
  batch. A binding slot out of range is refused instead of closing the shader.
- A kernel's pipeline compiles without holding the GPU lock, so other threads
  no longer wait behind a long compile. New `mgpuComputeShaderBuildPipeline`
  and `mgpuComputeShaderBuildPipelineAsyncToPort` build it without
  dispatching.
- A shader keeps its last 8 buffer bind groups, so going back to buffers it
  has used no longer builds a new bind group for each dispatch.
- A buffer bound without a binding kind takes the access its kernel
  declares: read-only for `var<storage, read>` or a bare `var<storage>`,
  read-write for `var<storage, read_write>`. A read-only input bound this
  way used to stop the kernel from building. The binding layout is rebuilt
  with the pipeline, so it follows a newly loaded kernel.
- Binding to a shader that a scoped use holds, or that is closing, returns
  busy (4) instead of reporting success and dropping the binding. From Dart it
  throws `StateError`; the call can be retried once the holder lets go.
- Work queued before a texture, view or sampler is destroyed still runs, and
  the texture's GPU memory is freed once it has, even while a cached bind
  group still names it. Shared output and video textures are destroyed behind
  the copies queued on them.
- The shader cache deletes only its own files, and evicts the least recently
  used entries first.
- `toRGBA` converts GRAY8 to grey and I420 to colour, refuses unknown
  formats, keeps one pipeline per input layout, and runs on the GPU thread.
  It follows the frame's colour matrix (BT.709, BT.601 or BT.2020) and range,
  set with the new `mgpuVideoTextureSetYuvColor`; the default stays BT.709
  full range. New `mgpuVideoTextureToRGBAAsyncToPort` converts without
  blocking.
- BGRA presents and the blocking shared-output copies return after the GPU
  copy has finished. New `mgpuCopyBufferF32ToSharedOutputTextureAsyncToPort`.
- A video frame that carries a producer fence is refused instead of read too
  early.
- Windows: D3D11/D3D12-specific calls first check which backend Dawn runs.
  Shared-texture access is checked and its state freed. The CPU bridge maps
  NV12 and more DXGI formats and reuses its staging texture. Imported
  same-adapter frames copy through a four-slot ring, and cross-adapter objects
  are released with the context. Cross-adapter frames use a three-slot ring
  and take the CPU path when every slot is busy, instead of waiting.
- Registered output pools: a cancelled or timed-out frame hands the keyed
  mutex back to the producer; a failed consumer signal is reported as device
  loss; the coordinator waits on fence events instead of polling; acquiring an
  acquired output returns `CONFLICT`; device loss no longer blocks release. A
  shader bound to a pool output keeps its texture view alive until rebound,
  so releasing the pool first no longer frees the view under it.
- A render pipeline's WGSL errors name the line and column.
- New `mgpuRenderPipelineNotifySettledToPort` posts once when a render
  pipeline finishes building; on the web the module calls
  `Module.mgpuRenderPipelineSettled(handle)` instead. Destroying a context
  closes its render pipelines, which settles any still building.
- Texture uploads from Dart no longer copy every frame into a temporary
  allocation. `importVideoFrame` zero-fills unused plane slots and checks the
  plane count.
- A failed `forAdapter` init frees its native context.
- The build hook takes Dawn's library from the paths CMake links instead of
  the first match of a recursive search.
- Windows: scoped capture sources need MSVC v143 (VS 2022) for x64. Another
  toolset still builds the library, with a CMake warning, and reports those
  sources as unsupported at run time.
- Web build settings: no fast-math, and a 1 MB Asyncify stack instead of
  64 MB.
- Removed the unused conversion kernels, `yuv_helpers.wgsl` and the
  `mgpuCreateKernel` declaration, which had no definition.

- Add exact-lease native D3D11 immutable output writers and bounded preview
  loans. Writers reuse their pipeline and intermediate image; each shared output
  is written once. Checked producer cleanup is separate from consumer removal,
  with retained charges for ambiguous ownership and no device/CPU fallback.
- Add checked final cleanup for dedicated native shaders and buffers. Admission
  failure preserves caller ownership without a callback; accepted cleanup retains
  the exact creation context through a native-reference-release acknowledgement.
  Uncertain cleanup retains its owner, separate from GPU/use retirement.
- Add strict creation-lease D3D11 BGRA snapshot import, opaque frame ownership,
  one-frame shader attachment and checked queue/access retirement. Uncertain
  cleanup returns a retained quarantine owner; no copy/device fallback occurs.
  Frame counts and logical retained bytes have fixed native bounds and diagnostics.
- Add bounded native exclusive shader scopes: exact creation-context retention,
  gated shader mutation, deferred deletion and completion-checked retirement.
  Callers retain their supporting resources; frame import/access retirement is
  separate. Emscripten refuses these native scopes without GPU work.
- Report failed async shader submission accurately to Dart ports and preserve
  callbacks on refusals. External-pool binds now enter the shader FIFO directly.
- Add a native callback form of the exact creation-lease submitted-work fence.
  It reports success after queue completion, separately from dispatch submission,
  and retains its own creation pin through acknowledgement. Dart keeps using
  the port form; Emscripten explicitly refuses the native callback form.
- Report submitted-work waiter allocation failure without throwing out of the
  GPU worker. Missing or failed completion never authorizes resource release.

## 1.8.1

- Implement the new `PlatformBuffer.webBufferHandle` accessor (always 0 on
  native).

## 1.7.0

- **Compiled shaders are cached on disk**: Dawn's blob cache now writes to
  storage, so shaders compile once per machine instead of on every launch. On
  D3D11 each compile runs FXC, whose cost grows steeply with kernel size.
  - New exports: `mgpuShaderCacheSetEnabled`, `mgpuShaderCacheSetDirectory`,
    `mgpuShaderCacheSetCapBytes`, `mgpuShaderCacheSetExtraKey`,
    `mgpuShaderCacheSetProvider`, `mgpuShaderCacheClear`,
    `mgpuGetShaderCacheStats` (including `pipelineCreateMs`, the total time
    spent creating compute pipelines) and `mgpuGetShaderCacheDirectory`.
  - `MGPU_SHADER_CACHE=0` turns caching off and
    `MGPU_SHADER_CACHE_DIR=<path>` moves it. Both override the setters, so you
    can rule the cache in or out on a build you cannot change.
  - A cache problem (unwritable directory, full disk, corrupt entry, lock,
    another process) is a cache miss and never fails device creation. Each
    entry stores its full key and is checked on read, so a hash collision
    recompiles instead of loading the wrong pipeline.
  - `mgpuShaderCacheSetProvider` is not available from Dart: Dawn calls a
    provider synchronously, possibly on its own thread, which a Dart callback
    cannot answer.
  - Logs one summary line per device at INFO; per-entry detail is at DEBUG.
    No effect on the web build.
- **`Buffer.writeRawBytes` no longer copies or allocates on the way to the
  GPU.** The payload used to be copied into a native scratch buffer before
  the driver copied it again; the driver now reads the caller's bytes
  directly, which halves upload time for a 4K frame and drops a pinned 32 MiB
  host allocation. Chunking (32 MiB) and byte order are unchanged.
- **`mgpuWriteBufferAt` no longer lets a C++ exception escape** on an invalid
  context or an out-of-range write; the error is caught and logged instead of
  crashing the caller.
- **The read staging buffer grows instead of being recreated** whenever a
  read asks for a different length. It grows to the next power of two, up to
  the source buffer's size, and is freed with the buffer. This matters for
  reads whose length changes every frame.
- **New `MGPU_WAIT_PROF=1`**: per-site count, total, mean, max and spin count
  for the blocking waits and per-call allocations on the transfer path;
  `MGPU_WAIT_PROF_MS=<n>` also logs each event longer than n ms. A long wait
  with many spins means the GPU was busy; with few spins, the thread was
  descheduled. Off by default.
- **Fix: async completions could abort the process** with
  `runtime_entry.cc: Callback invoked after it has been deleted` (seen as
  `Lost connection to device.`) on hot reload, hot restart, or when an isolate
  exited with GPU work in flight. Completions are now posted to a Dart native
  port, which is a safe no-op once the port or its isolate is gone.
  - New exports take the port with each call:
    `mgpuInitializeContextAsyncToPort`, `mgpuContextInitializeAsyncToPort`,
    `mgpuDispatchAsyncToPort`, `mgpuReadAsyncToPort`,
    `mgpuCopyBufferToSharedOutputTextureAsyncToPort` and
    `mgpuVideoTextureBGRAToRGBASharedOutputAsyncToPort`. Call
    `mgpuInitDartApi(NativeApi.initializeApiDLData)` once first.
  - Each completion is one int64, `(token << 1) | ok`. Tokens are never
    reused, so a late or duplicate completion cannot resolve the wrong future.
    Each entry point posts exactly once, even on invalid arguments, so a bad
    call fails instead of hanging.
  - `mgpuReadAsyncToPort` takes its element type as the new `MGPUElementType`
    enum, not `BufferDataType` (the Dart and C++ `BufferDataType` enums are
    ordered differently).
  - The `MGPUCallback` variants remain for embedders that manage their
    callback's lifetime; the web build uses them.
  - Backward compatible: no signature changed and every old export still
    works. The Dart side checks for the new exports and falls back on an older
    native library.
  - `MGPU_UNSAFE_CALLBACK_COMPLETIONS=1` forces the old callback path. It
    exists only to reproduce the crash; never set it in production.
- **New `mgpuDrainWorkQueue()`** (Dart: `MinigpuPlatform.drainWorkQueue()`):
  blocks until everything already queued on the WebGPU worker thread has run,
  so teardown cannot free a resource a queued task still uses. Synchronous,
  so it can run from Flutter's `reassemble` on hot reload. No-op when called
  on the worker thread itself, and with an older native library.
- **The build hook now rebuilds when anything under `src/` changes.** Before,
  edits to the C/C++ sources did not trigger a rebuild and the previously
  built library kept being used; a new export missing from the library was
  the sign.

## 1.6.1

- released 08/13/26 - MR

## 1.6.0

- **The process-global context is serialized and reference-counted.**
  `mgpuInitializeContext` / `mgpuInitializeContextAsync` attach,
  `mgpuDestroyContext` detaches, and the device is torn down only when the
  last consumer detaches; new `mgpuContextRefCount()` reports the count. A
  caller that arrives while another is creating the device waits and shares
  it. This fixes crashes, and one consumer's teardown killing another's
  device, when several consumers (such as Dart isolates) share a process.
  Single-consumer behaviour is unchanged. A consumer that exits without
  detaching keeps the context alive for the rest of the process.
  `MGPU_UNSAFE_NO_CTX_LOCK=1` restores the old unguarded behaviour; it exists
  only to demonstrate the bug.
- **The cached D3D11 device and immediate context are locked and released on
  teardown.** Concurrent first use could release a device another caller
  still held, and after a destroy and re-init the cached device still pointed
  at the old one, so `createSharedOutputTexture` silently returned null.
- **`setLogCallback` now delivers over a Dart native port.** This fixes a
  whole-process abort (`Callback invoked after it has been deleted`) when a
  log line arrived after the isolate that registered the callback had exited.
  New exports `mgpuInitDartApi(void*)` and `mgpuSetLogPort(int64)`; lines
  arrive as `[int32 level, Uint8List utf8]` (bytes, because driver strings
  are not always valid UTF-8; `mgpuFreeLogMessage` does not apply). Still
  process-wide, and the last registration wins. `mgpuSetLogCallback` remains
  for non-Dart embedders. Native only.

## 1.5.9

- `mgpuDestroyComputeShader` queues the delete behind any bind or dispatch
  already queued for that shader instead of deleting at once (needed by the
  ordered binds in minigpu 1.5.9). You no longer need to await a read before
  destroying a shader. Buffers are still deleted immediately.

## 1.5.8

- **Fix: `MINIGPU_DAWN_DIR` was ignored when building through Flutter or
  `dart pub`.** It only changed where a prebuilt library was looked for;
  Dawn was still cloned and built in `%SYSTEMDRIVE%\dawn`.
- **Fix: a Windows Dawn path broke a from-source Dawn build** with `Invalid
  character escape` in `<dawn>/build_win_x86_64/tmp/CMakeLists.txt`. Only
  machines without a prebuilt Dawn were affected.
- **Fix: the Emscripten build (`make build_weblib`) failed after the Dawn
  update** with "not a valid port path". It now uses the emdawnwebgpu port
  from Dawn's built `emdawnwebgpu_pkg` (older Dawn layouts still work), and
  fails at configure time, naming `DAWN_COMMIT`, when neither is found.
- **Fix: every GPU wait on Windows cost a ~15.6 ms timer tick.** Waits now
  spin briefly before sleeping; a shared-texture present at 1280x720 went
  from 15.7 ms to 2.6 ms (p50). This affects every wait through the event
  drain (present, debug reads, video import). `MGPU_DRAIN_SPIN_MS=<0..1000>`
  sets the spin budget (default 8; `0` restores the old behaviour, for
  comparison only); it warns on an invalid value and on an explicit `0`.
- New `mgpuDrainSpinBudgetMs()`: the budget the loaded library implements, so
  a caller can check it is not running a stale build.
- New batched upload scope: `mgpuBeginUploads` / `mgpuStageWrite` /
  `mgpuStageReserve` / `mgpuEndUploads` (+ `mgpuUploadsSupported`,
  `mgpuUploadStats`, `MGPU_UPLOAD_PROF=1`, and `mgpuContextBeginUploads` /
  `mgpuContextEndUploads` for secondary contexts). Many small host writes
  become one queue write plus copies recorded into the current compute
  batch. A scope must not span a dispatch that reads what it stages. Outside
  a scope, or for an unaligned range, `mgpuStageWrite` writes immediately.
- New batched readback scope: `mgpuBeginReadbacks` / `mgpuStageRead` /
  `mgpuReadbackReserve` / `mgpuEndReadbacks` (+ `mgpuReadbacksSupported`,
  `mgpuReadbackStats`, `MGPU_READBACK_PROF=1`, and
  `mgpuContextBeginReadbacks` / `mgpuContextEndReadbacks`). Many
  `mgpuReadSync*` calls become one submit, one fence and one map. Unlike
  uploads: a staged read is filled only at `mgpuEndReadbacks` (keep `dst`
  alive and do not read it earlier); every staged read sees its source as of
  End, so a scope must not span a dispatch that overwrites it; and
  `mgpuStageRead` reads raw bytes (like `mgpuWriteBufferAt`, not
  `mgpuReadSyncUint8`). Unaligned reads, and reads outside a scope, happen
  immediately.
- Both scopes run on the calling thread under the device mutex, like the
  plain read and write calls.
- New multi-adapter context handles: `mgpuCreateContextHandle(adapterFilter)`,
  `mgpuContextInitializeAsync`, `mgpuContextGetAdapterName`,
  `mgpuContextCreateBuffer`, `mgpuContextCreateComputeShader` and
  `mgpuDestroyContextHandle`. Each handle has its own device, queue and task
  queue (Dart: `createSecondaryPlatform`). Never mix resources from two
  contexts in one dispatch.
- New `mgpuEnumAdapters(namesOut, totalOut, usedOut, cap)` (DXGI) and Dart
  `listAdapters()` returning `GpuAdapterInfo`: adapter name, total dedicated
  VRAM and current usage.
- New `mgpuSetBufferFire` and Dart `setBufferFire` / `dispatchFire`:
  `dispatchFire` skips the per-dispatch completion round trip, and
  `setBufferFire` queues the bind in the same order, so bind → fire → rebind
  → fire is ordered. Bindings are read when the dispatch runs, not when it is
  fired.
- New Dart `Buffer.writeRawBytes`: 4-byte-aligned raw upload streamed through
  at most 32 MB of host scratch, so a very large write never holds its full
  size in host memory. `ArgumentError` on a misaligned offset or length.
- New `mgpuDebugConsumerChecksumSharedHandle(handle, w, h)`: an FNV-1a
  checksum of the shared output surface read through a separate D3D11 device,
  as a compositor sees it, for detecting torn or stale presents in tests.
  Verification only.
- New `MGPU_PRESENT_NO_WAIT=1`: test-only and deliberately broken (skips the
  present's completion wait) to check the checksum above. Never set it in
  production.

## 1.5.7

- Release of the adapter-selection and Tier B / Tier C work listed under
  1.5.6; no further API change.

## 1.5.6

- New pre-init hint `mgpuPreferDisplayAdapter(int enable)` (Dart:
  `preferDisplayAdapter`): use the GPU that drives the primary display, so
  screen capture, GPU processing and a D3D11 encoder on that adapter share
  frames without copies, even on hybrid systems where the discrete GPU also
  drives a monitor. Returns whether the hint arrived before context init;
  `MGPU_ADAPTER_NAME` still overrides. New `mgpuGetSelectedAdapterName`
  reports the adapter Dawn chose.
- Fix hybrid laptops falling back to the CPU bridge (Tier C) on every frame.
  The cross-adapter (Tier B) check tested the compute GPU instead of the
  display GPU that allocates the shared texture. When the display GPU lacks
  `CrossAdapterRowMajorTextureSupported`, Dawn now runs on the integrated GPU
  (Tier A*) for zero-copy capture, and the hardware encoder stays there too.
  If the exact display-name match fails, auto-select prefers the integrated
  GPU.

## 1.5.5

- Raise GPU submission priority (`IDXGIDevice::SetGPUThreadPriority(+7)`) on
  the cached D3D11 device, so minigpu's work is starved less when another
  process saturates the GPU. Best effort; a failure is logged.
- The Tier B (D3D12 cross-adapter) probe remembers a failure per adapter, so
  it no longer recreates a D3D12 device and logs "Falling back to Tier C" on
  every frame.

## 1.5.4

- fix release version pins

## 1.5.3

- New exports `mgpuCopyBufferToSharedOutputTextureAsync` /
  `mgpuVideoTextureBGRAToRGBASharedOutputAsync` (Dart: `copyFromBufferAsync`
  / `bgraToRgbaSharedOutputAsync`) run the copy and present sync on the
  WebGPU worker thread and call back when done, so the calling isolate no
  longer busy-waits on the present.

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

- Fix garbled adapter names in logs.

## 1.4.5

- Fix Dawn being built inside the pub cache instead of the shared Dawn
  directory.
- Add `Minigpu.setLogCallback` / `setLogLevel`: native Dawn/GPU log lines are
  routed to a Dart callback (C exports `mgpuSetLogCallback` /
  `mgpuSetLogLevel`).

## 1.4.4

- Fix a `FormatException` in `setLogCallback` when a log line is not valid
  UTF-8.

## 1.4.3

- Prefer the discrete GPU when selecting an adapter (using Dawn's adapter
  enumeration); fixes the wrong adapter being picked on Optimus laptops.

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

- Breaking: Changed setData and references to .write
- Change: 1.2.0 migrates minigpu to direct webgpu usage
- Fix: fresh builds need dawn find off
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
