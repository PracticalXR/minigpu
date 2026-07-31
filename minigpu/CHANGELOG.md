# minigpu

## 1.5.9

- **`ComputeShader.setBuffer` and `setBufferAtSlot` are now always ORDERED** —
  the bind joins the WebGPU-thread FIFO, so it is correct against `dispatchFire`
  as well as `dispatch`. This removes a silent-corruption trap rather than
  documenting it: the binds used to run inline on the caller's thread while
  `dispatchFire` enqueued, so rebinds raced ahead and **every fired dispatch saw
  the LAST binding**. Measured on a 3-iteration bind→fire→rebind→fire loop, the
  old inline binds produced `[[0,0], [0,0], [31,31]]` — two destinations never
  written, no error raised — where the ordered binds produce
  `[[11,11], [21,21], [31,31]]`. Nothing prevented the bad pairing but choosing
  the right one of four bind methods.
- `ComputeShader.setBufferFire` is **deprecated** — it is now an alias for
  `setBuffer`. `setBufferAtSlotFire` (added and never released during 1.5.9
  development) is removed; use `setBufferAtSlot`.
- **A compute shader is no longer destroyed inline.** `mgpuDestroyComputeShader`
  queues the delete on the same FIFO, so it lands after any bind or dispatch
  already queued against that shader. Ordered binds capture the shader pointer
  to mutate its binding tables when they run, so an inline delete would free it
  under a pending bind — the hazard `setBufferFire`'s docs previously pushed onto
  callers ("do not destroy the shader until a read has been awaited"). That
  constraint is now gone. (Buffers never had it: a queued bind captures only the
  raw WGPU handle by value, which is why `mgpuDestroyBuffer` can still delete
  immediately.)
- New test `test/minigpu_fire_bind_test.dart`: fire-then-read synchronization,
  rebind-between-fires ordering, equivalence with the fully awaited chain, and
  the 65535 cap on `dispatchFire`.

## 1.5.8

- **Breaking-ish fix: `Minigpu()` is now a per-isolate SINGLETON and the context
  is destroyed only by explicit `destroy()` / `destroySync()`.** Each
  construction used to attach a `Finalizer` calling `destroyContext()`, but there
  is only ONE process-global native context — so any temporary wrapper (e.g.
  `Minigpu().isInitialized` in a test `setUp`) destroyed the device, whenever the
  GC ran, out from under every live buffer and shader. Resources created before
  the loss were invalid on the auto-reinitialized device and their dispatches
  were silently dropped: zero outputs, no Dart-visible error.
- **`Minigpu.init()` is now idempotent and concurrency-safe** — returns
  immediately when already initialized, awaits the in-flight init when raced, and
  no longer throws `MinigpuAlreadyInitError`. Update any code relying on that
  throw.
- **`ComputeShader.dispatch` / `dispatchFire` now throw `ArgumentError` above
  `ComputeShader.maxWorkgroupsPerDim` (65535).** Exceeding WebGPU's per-dimension
  cap invalidated the whole CommandBuffer, and since validation errors are
  STICKY, every later submit on the device failed too — one oversized dispatch
  silently poisoned unrelated work. The error names the offending dims and gives
  the canonical fold (`gx = min(n, 65535); gy = (n + gx - 1) ~/ gx`, flat index
  rebuilt in the shader as
  `gid.x + gid.y * (num_workgroups.x * workgroup_size_x)`).
- New `ComputeShader.dispatchFire(x, y, z)` — fire-and-forget dispatch with no
  per-dispatch completer round trip. Call order is still honoured, so awaiting
  any later buffer read synchronizes every fired dispatch. **Bindings are
  snapshotted when the dispatch RUNS, not when it is fired**: do not `setBuffer`
  on a shader with an unsynchronized fired dispatch outstanding. Use
  `setBufferFire` (the bind joins the same FIFO) or a shader instance per call
  site.
- New `Minigpu.forAdapter(String adapterFilter)` — an INDEPENDENT context on the
  adapter whose name contains `adapterFilter` (case-insensitive substring), with
  its own device, queue and task FIFO. Not the singleton: `init()` before use,
  `destroy()` when done, and never mix two instances' resources in one dispatch.
  Throws `UnsupportedError` on web. Instance getter `adapterName` reports which
  adapter THIS context bound.
- New `Minigpu.listAdapters()` — hardware adapters with dedicated-VRAM total and
  usage (DXGI on Windows; empty elsewhere). Returns `GpuAdapterInfo` from
  `package:minigpu_platform_interface/minigpu_platform_interface.dart`.
- New `Buffer.writeRawBytes(bytes, {dstByteOffset = 0})` — raw 4-byte-aligned
  upload streamed in 32 MB chunks, so neither host scratch nor driver staging
  holds the whole payload. For LARGE transfers where a single `write` would spike
  or pin host RAM.
- New `Minigpu.drainSpinBudgetMs` — the event-drain spin budget the LOADED native
  binary implements; `null` on web, or on a binary predating the export, which
  for a native build means the drain fix below is NOT in it. Latency-sensitive
  callers should assert `> 0` at startup, since loading a stale native artifact
  is silent.
- Native (`minigpu_ffi` 1.5.8), reaching consumers of this package through the
  shared context — see `minigpu_ffi/CHANGELOG.md` for the contracts:
  - **The Dawn event drain no longer costs a Windows timer quantum (~15.6 ms)
    per GPU wait**: present p50 15.69 → 2.57 ms at 1280x720 and 15.69 →
    10.07 ms at 3840x2160 on an RTX 4090. Strictly better — waits can only
    return sooner. `MGPU_DRAIN_SPIN_MS=<0..1000>` tunes it (default 8).
  - Additive batched staging upload and readback scopes: N scattered host writes
    become one queue write plus N recorded copies, and N per-buffer
    `mgpuReadSync*` calls become one submit, one fence and one map (8 reads of
    2.72 MB: 1.098 → 0.465 ms of API time). **No Dart API on this package yet**
    — reachable through the `minigpu_ffi` bindings.
  - Multi-adapter context handles and `mgpuEnumAdapters`, backing
    `Minigpu.forAdapter` and `Minigpu.listAdapters` above.
  - Three build fixes worth knowing if you have ever fought the Dawn step:
    `MINIGPU_DAWN_DIR` is now honoured when building through Flutter / dart pub
    (it previously only moved the prebuilt-library search, not the Dawn root
    given to cmake); a Windows Dawn root no longer breaks a from-source Dawn
    build with `Invalid character escape '\d'`; and the Emscripten build's
    emdawnwebgpu port file is detected rather than hardcoded. See
    `minigpu_ffi/README.md` → Troubleshooting.

## 1.5.7

- Release cut of the adapter-selection and Tier B/Tier C work documented under
  1.5.6; no additional API change in this package.

## 1.5.6

- New statics `Minigpu.preferDisplayAdapter([enable])` and
  `Minigpu.selectedAdapterName`: pre-init hint that binds Dawn to the adapter
  driving the PRIMARY display (Windows) so screen capture, GPU processing and
  HW encode share one GPU (same-adapter zero-copy) on hybrid systems, and a
  query for which adapter Dawn actually selected. Call the hint before any
  minigpu use; `MGPU_ADAPTER_NAME` still overrides.

## 1.5.5

## 1.5.4

- fix release version pins

## 1.5.3

- Add async shared-output-texture copy: `SharedOutputTexture.copyFromBufferAsync`
  and `VideoTexture.bgraToRgbaSharedOutputAsync`. These run the GPU copy and the
  cross-device present sync on minigpu's WebGPU worker thread and complete a
  `Future` when finished, instead of busy-polling the present-wait on the calling
  isolate — removing the dominant per-frame blocking cost on the shared-output
  (zero-copy encode) path. The synchronous `copyFromBuffer` /
  `bgraToRgbaSharedOutput` are unchanged.

## 1.5.2

- Add `Minigpu.copyBuffer(src, dst, {required int elementCount})`: GPU-side
  buffer-to-buffer copy using a WGSL compute shader — no CPU round-trip. The
  copy shader (64-element workgroup, `array<u32>` storage bindings) is created
  once per `Minigpu` instance on first call and reused for all subsequent calls.
  Non-multiple-of-64 element counts are handled correctly via an `arrayLength`
  guard in the shader.
- `Minigpu.onShaderDestroyed` now nulls the cached copy-shader reference when
  that shader is destroyed, preventing a double-destroy if
  `destroyAllTrackedShaders()` is called before `destroy()`.
- `Minigpu.destroy()` explicitly tears down the cached copy shader before
  releasing the Dawn context.
- New test suite `test/minigpu_copy_buffer_test.dart`: 14 tests covering
  correctness (u32, f32, bit-pattern sentinel), partial copy, workgroup
  boundary alignment, shader reuse, and `liveShaderCount` / `liveBufferCount`
  stability.

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

- improve adapter selection to prefer discrete GPU using dawn native EnumerateAdapters; fixes incorrect adapter picked on Optimus laptops

## 1.4.2

- fixes dawn library not being found

## 1.4.1

- added bindings observer

## 1.4.0

- adds minigpu_view, gpu_pipeline libraries
- adds `Minigpu.destroySync()` for use in synchronous hot-reload teardown hooks (e.g. `MinigpuBinding` from `minigpu_flutter`)

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

- Refactored example
- Various fixes
- Updated native assets to code assets
- Memory problems fixed on web and ffi

## 1.1.3

- breaking: import package instead of buffer and shader separately
- fix: pubspec repository url
- adds: tensor package protoype
- fix: issue with reading buffer segments fixed

## 1.1.2

- fix: create dawn dir to prevent first run error.

## 1.1.1

- fix: split download command for quiet fail on remote add

## 1.1.0

- fix: dawn git not running properly

## 1.0.9

- fix: prevent using project root on ffi since pub wont see the file

## 1.0.8

- fix: pub.dev still missing project root file

## 1.0.7

- fix: project root file missing

## 1.0.6

- fix: minigpu_ffi must also use flutter in pubspec or pub.dev analysis fails
- fix: issue with project root finding as package

## 1.0.5

- fix: must have flutter in pubspec or pub.dev analysis fails

## 1.0.4

- fix: updates to readme

## 1.0.3

- fix: remove flutter from package pubspec.yaml
- fix: updates to readme

## 1.0.2

- new: explicity set supported platforms in pubspec.yaml for pub.dev

## 1.0.1

- breaking: Uses dart native assets
see updated readme.
- implements platform stub for native assets to coexist with flutter plugins.
- uses native_toolchain_cmake 0.0.4

## 1.0.0

- Initial version.
