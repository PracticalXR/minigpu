# minigpu

## 1.9.1

- Version bump for the minigpu 1.9.1 family. On the web, minigpu_web 1.9.1
  ships an optimized release build of its wasm module: about 0.56 MB instead
  of 13.1 MB.

## 1.9.0

- `copyBuffer` copies exactly the requested elements. It no longer writes
  past them in the destination or stops at 65,535 workgroups. A destroyed
  buffer throws `StateError`, a count past either buffer throws `RangeError`,
  and copying a buffer onto itself or copying 0 elements does nothing.
- On native, an awaited `dispatch` throws `StateError` when the work was not
  submitted: the kernel failed to build, a bound resource was destroyed, or
  the device is gone. It used to finish as if it had run. A negative group
  count throws `ArgumentError`; a count of 0 dispatches nothing.
- On native, `createBuffer` throws `RangeError` for a size that does not fit
  in 32 bits; it used to create a smaller buffer without saying so.
- `warmShaders` compiles each kernel without running it, binds each buffer
  the way the kernel declares it, and reports a kernel that fails to build.
  It keeps the shader it built, so the app's own copy of that kernel reuses
  the compiled pipeline.
- New `ComputeShader.buildPipeline()`: compiles the pipeline for the current
  bindings without running the kernel.
- Shaders and render pipelines dropped without `destroy()` are freed when
  they are garbage collected. `destroy()` and device recovery also release
  video textures, shared output textures and `toRGBA` buffers.
- A failed `destroy()` no longer blocks a later `init()`.
- `ownsBuffer` takes constant time, and minigpu's internal placeholder buffer
  no longer counts as a live buffer.
- `setBuffer` keeps the original error type. The `isInitialized` setter is
  deprecated.
- `setBuffer` and `setBufferAtSlot` bind a buffer with the access its
  kernel declares, so a kernel can take read-only inputs
  (`var<storage, read>`). Binding one used to stop the kernel from building.
- A render pipeline is seen as ready as soon as it compiles: preparing it
  waits for a signal from the backend instead of polling every 10 ms. A
  backend without the signal is polled, starting at 1 ms.
- On native, `toRGBA` follows the source's colour matrix and range: set
  `yuvMatrix` (`YuvMatrix.bt709`, `bt601` or `bt2020`) and `yuvFullRange` on
  `ExternalVideoBuffer`. The defaults keep the old BT.709 full-range output.
- New `VideoTexture.toRGBAAsync()`, which converts without blocking the
  calling thread on native. Destroying the texture during a conversion frees
  it once the conversion ends.
- On the web, a buffer read that fails throws `StateError` instead of
  returning stale data.
- New `ComputeShader.isValid` and `SharedOutputTexture.copyFromBufferF32Async`.
- `setLogCallback` defaults to warnings and errors.
- Docs: `GpuTexture.destroy` timing, `toRGBA` formats and errors, and the
  blocking shared-output copies.

## 1.8.1

- **Fix: a long-lived shader could keep reading a destroyed buffer** (zeros
  or unrelated data) when a new buffer reused the destroyed one's handle and
  `setBuffer` saw no change. Typical symptom: a later instance producing
  nothing or misreading a few rows. No API change.
- **New `warmShaders()`: compile kernels before first use, with progress and
  named failures.** A kernel is compiled on its first `dispatch`, so the cost
  lands on the first frame or first inference and looks like a hang.

  ```dart
  final warm = warmShaders(gpu, [
    KernelWarmSpec(label: 'blur', source: blurWgsl, buffers: {'src': 4, 'dst': 4}),
  ], ensureReady: gpu.init);
  warm.progress.listen((p) => print('${p.completed}/${p.total} ${p.label ?? ""}'));
  final result = await warm.done;   // never throws — inspect result.phase
  ```

  Kernels build one at a time and yield to the event loop in between, so it
  can run at app launch. There is deliberately no concurrency option: GPU work
  is serialised anyway, and compiling a batch back to back can make a mobile
  driver stop responding. `WarmProgress` reports `completed` / `total` /
  `fraction`, the kernel being built, and `WarmError`s naming the kernel and
  the stage (`create`, `dispatch`, `timeout`, `deviceLost`). A timeout fails
  only that kernel; a lost device ends the run. `progress` replays its latest
  value to late listeners. It complements the 1.7.0 disk cache, which makes
  later launches cheap.
- **The web build can now be loaded inside a Web Worker**, so GPU work on the
  web no longer has to run on the main thread. The device is published on
  `globalThis` instead of `window`.
- New `Buffer.webBufferHandle`: the underlying WebGPU buffer handle on web (0
  on native), for zero-copy presentation of buffer contents.

## 1.7.0

- **Compiled shaders are cached on disk**, so shader compilation is paid once
  per machine instead of on every launch. On by default. It matters most on
  Windows (D3D11), where a large compute kernel can take tens of seconds to
  compile.
  - Optional API: `Minigpu.configureShaderCache(enabled:, directory:,
    maxBytes:, extraKey:)`, `Minigpu.shaderCacheStats`,
    `Minigpu.shaderCacheDirectory`, `Minigpu.clearShaderCache()` and
    `ShaderCacheStats`. Configure it before init; settings are process-wide,
    like `preferDisplayAdapter`.
  - `MGPU_SHADER_CACHE=0` turns it off and `MGPU_SHADER_CACHE_DIR=<path>`
    moves it. Both override settings made in code.
  - Stored in the OS cache directory (`%LOCALAPPDATA%`, `~/Library/Caches`,
    `$XDG_CACHE_HOME`), capped at 256 MB, least recently used entries evicted
    first. **Android has no default**: caching stays off until you pass a
    directory to `configureShaderCache`. No effect on web.
  - A cache problem (missing or unwritable directory, full disk, corrupt
    entry, another process) only costs a recompile; it never fails device
    creation. A driver update or a different GPU recompiles rather than
    loading a stale shader.
- **`Buffer.writeRawBytes` no longer copies the data before handing it to the
  GPU**, on native or web, which halves upload time for a 4K frame. On web it
  also removes a whole-frame allocation per call. No API change.
- Reads whose length changes from call to call no longer recreate the GPU
  staging buffer each time; it only grows.
- **New `Minigpu.drainWorkQueue()`**: blocks until every GPU task already
  queued has run. Dispatches, reads and shared-texture copies run on a native
  worker thread, so destroying a buffer or texture with work still queued can
  free something that work uses. Use it in teardown (`stop producing →
  drainWorkQueue() → release`), never per frame. It is synchronous so it can
  be called from Flutter's `State.reassemble` on hot reload. No-op on web.
- **Fix: async GPU work could abort the whole process** with `Callback invoked
  after it has been deleted`, most often during Flutter hot reload, and also
  on hot restart or when a worker isolate exited with GPU work in flight.
  Completions now arrive on a Dart native port, which is ignored once its
  isolate is gone. No API change.

## 1.6.1

- released 08/13/26 - MR

## 1.6.0

- Picks up minigpu_ffi 1.6.0: several isolates initializing the GPU at the
  same time now share one device instead of destroying each other's. No API
  change in this package.

## 1.5.9

- **`ComputeShader.setBuffer` and `setBufferAtSlot` are now always ordered**
  with dispatches, including `dispatchFire`. Before, a bind ran at once while
  fired dispatches waited in a queue, so a bind → fire → rebind → fire loop
  silently ran every dispatch with the last binding.
- `ComputeShader.setBufferFire` is **deprecated**; it is now an alias for
  `setBuffer`.
- Destroying a compute shader now waits behind any bind or dispatch already
  queued for it, so you no longer need to await a read before destroying a
  shader.

## 1.5.8

- **Behaviour change: `Minigpu()` is a per-isolate singleton, and the GPU
  context is destroyed only by an explicit `destroy()` / `destroySync()`.**
  Before, garbage collection of any temporary `Minigpu()` (such as
  `Minigpu().isInitialized` in a test `setUp`) destroyed the shared device
  under live buffers and shaders, whose dispatches then silently produced
  zeros.
- **`Minigpu.init()` is idempotent and safe to call concurrently**: it
  returns at once when already initialized and waits for an init in
  progress. It no longer throws `MinigpuAlreadyInitError`; update code that
  relied on that.
- **`ComputeShader.dispatch` / `dispatchFire` throw `ArgumentError` above
  `ComputeShader.maxWorkgroupsPerDim` (65535) in any dimension.** An
  oversized dispatch used to make every later GPU submit on the device fail
  silently. To cover more work items, fold into two dimensions:
  `gx = min(n, 65535); gy = (n + gx - 1) ~/ gx`, and rebuild the flat index
  in the shader as `gid.x + gid.y * (num_workgroups.x * workgroup_size_x)`.
- New `ComputeShader.dispatchFire(x, y, z)`: dispatch without waiting for
  completion. Call order is kept, so awaiting any later buffer read waits for
  every fired dispatch. Bindings are read when the dispatch runs, not when it
  is fired; in this release, rebinding a shader with fired dispatches
  outstanding needed `setBufferFire` (from 1.5.9 every bind is ordered).
- New `Minigpu.forAdapter(String adapterFilter)`: a separate context on the
  adapter whose name contains `adapterFilter` (case-insensitive), with its own
  device and queue. It is not the singleton: call `init()` before use and
  `destroy()` when done, and never mix two contexts' resources in one
  dispatch. Throws `UnsupportedError` on web. `adapterName` reports the
  adapter a context uses.
- New `Minigpu.listAdapters()`: hardware adapters with total and used
  dedicated VRAM (Windows; empty elsewhere), as `GpuAdapterInfo` from
  `package:minigpu_platform_interface/minigpu_platform_interface.dart`.
- New `Buffer.writeRawBytes(bytes, {dstByteOffset = 0})`: uploads
  4-byte-aligned raw bytes in 32 MB chunks, for large transfers where a
  single `write` would spike host memory.
- New `Minigpu.drainSpinBudgetMs`: shows whether the loaded native library
  has the GPU-wait fix below (`null` on web or with an older library). A
  stale native library loads silently, so latency-sensitive apps can assert
  it is `> 0` at startup.
- From minigpu_ffi 1.5.8:
  - GPU waits on Windows no longer cost a ~15.6 ms timer tick each (a 720p
    shared-texture present went from 15.7 ms to 2.6 ms).
    `MGPU_DRAIN_SPIN_MS=<0..1000>` tunes it (default 8).
  - Batched upload and readback scopes that turn many small writes or reads
    into one submit. No Dart API in this package yet; use the `minigpu_ffi`
    bindings.
  - Build fixes: `MINIGPU_DAWN_DIR` is honoured when building through Flutter
    or `dart pub`; a Windows Dawn path no longer breaks a from-source Dawn
    build with `Invalid character escape '\d'`; the Emscripten build finds
    Dawn's emdawnwebgpu port. See `minigpu_ffi/README.md` → Troubleshooting.

## 1.5.7

- Release of the adapter-selection work listed under 1.5.6; no further API
  change in this package.

## 1.5.6

- New `Minigpu.preferDisplayAdapter([enable])` and
  `Minigpu.selectedAdapterName`. The hint, called before any minigpu use,
  makes Dawn use the GPU that drives the primary display (Windows), so on
  hybrid systems screen capture, GPU processing and hardware encode stay on
  one GPU and can share frames without copies. `selectedAdapterName` reports
  the adapter Dawn actually chose. `MGPU_ADAPTER_NAME` still overrides.

## 1.5.5

## 1.5.4

- fix release version pins

## 1.5.3

- New `SharedOutputTexture.copyFromBufferAsync` and
  `VideoTexture.bgraToRgbaSharedOutputAsync`: the copy and present sync run on
  minigpu's GPU worker thread and complete a `Future`, instead of
  busy-waiting on the calling isolate, which was the largest per-frame cost
  on the shared-output (zero-copy encode) path. The synchronous versions are
  unchanged.

## 1.5.2

- New `Minigpu.copyBuffer(src, dst, {required int elementCount})`: a GPU-side
  buffer-to-buffer copy with no CPU round trip. Any element count works; the
  copy shader is created on first use and reused.
- `destroy()` and `destroyAllTrackedShaders()` release that copy shader
  without destroying it twice.

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
  routed to a Dart callback.

## 1.4.4

- Fix a `FormatException` in `setLogCallback` when a log line is not valid
  UTF-8.

## 1.4.3

- Prefer the discrete GPU when selecting an adapter; fixes the wrong adapter
  being picked on Optimus laptops.

## 1.4.2

- fixes dawn library not being found

## 1.4.1

- added bindings observer

## 1.4.0

- adds minigpu_view, gpu_pipeline libraries
- adds `Minigpu.destroySync()` for synchronous hot-reload teardown hooks (e.g.
  `MinigpuBinding` from `minigpu_flutter`)

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

- Refactored example
- Various fixes
- Updated native assets to code assets
- Memory problems fixed on web and ffi

## 1.1.3

- breaking: import package instead of buffer and shader separately
- fix: pubspec repository url
- adds: tensor package prototype
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

- new: explicitly set supported platforms in pubspec.yaml for pub.dev

## 1.0.1

- breaking: Uses dart native assets; see updated readme.
- implements platform stub for native assets to coexist with flutter plugins.
- uses native_toolchain_cmake 0.0.4

## 1.0.0

- Initial version.
