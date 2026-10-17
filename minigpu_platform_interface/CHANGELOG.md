# minigpu_platform_interface CHANGELOG

## 1.9.0

- Add `PlatformSharedOutputTexture.copyFromBufferF32Async`. The default runs
  the blocking copy.
- `setLogCallback` defaults to warnings and errors (level 2), not info.
- Document that the blocking shared-output copies wait for the GPU, and how
  `toRGBA` fails.
- Add `YuvMatrix` and `ExternalVideoBuffer.yuvMatrix` / `yuvFullRange`
  (defaults: BT.709, full range).
- Add `PlatformVideoTexture.toRGBAAsync()`. The default calls `toRGBA()`.
- Add `PlatformRenderPipeline.settled`, which completes when the pipeline
  stops compiling. The default is null, and callers poll `status`.
- Add `PlatformComputeShader.buildPipeline()`, which compiles without running
  the kernel. The default dispatches one workgroup.

## 1.8.1

- Add `webBufferHandle` to `PlatformBuffer` (default 0), so implementations can
  expose the underlying WebGPU buffer handle on web.

## 1.7.0

- Add the shader cache API: `configureShaderCache(...)`, `shaderCacheStats`,
  `shaderCacheDirectory`, `clearShaderCache()` and the `ShaderCacheStats` value
  type. Every member has a default (no-op / null / 0), so existing
  implementations need no change; web keeps the defaults.
- Add `MinigpuPlatform.drainWorkQueue()`: blocks until every GPU task already
  queued on the native worker thread has run. Defaults to a no-op.

## 1.6.1

- released 08/13/26 - MR

## 1.6.0

- Version bump for the minigpu 1.6.0 family. No functional change in this
  package.

## 1.5.9

- No interface change. For backend implementors: since minigpu 1.5.9 the
  facade sends every buffer bind through `PlatformComputeShader.setBufferFire`,
  never `setBuffer`. A backend whose `setBufferFire` is not ordered with its
  dispatches will corrupt fire-and-forget callers. The default delegates to
  `setBuffer`, which is only correct for backends that run everything in
  order (such as web).

## 1.5.8

- **Every addition has a default, so a backend that `extends` these classes
  keeps compiling. A class that `implements` them must add the new members.**
- New `GpuAdapterInfo` (name, total and used dedicated VRAM, `freeVramBytes`)
  and `MinigpuPlatform.listAdapters()`, default `const []`.
- New `MinigpuPlatform.createSecondaryPlatform(String adapterFilter)`: an
  independent context on the matching adapter, with its own device and queue.
  Never mix two contexts' resources in one dispatch. Default `null` (no
  multi-adapter support).
- New `MinigpuPlatform.drainSpinBudgetMs`: the GPU-wait spin budget of the
  loaded native library, `null` where it cannot be asked. A library without
  that fix costs ~15.6 ms per GPU wait on Windows, so latency-sensitive
  callers can assert `> 0`.
- New `PlatformComputeShader.dispatchFire(x, y, z)`: dispatch without
  waiting. Order with reads and writes is kept, so a later awaited
  `PlatformBuffer.read` waits for every fired dispatch. Default delegates to
  `dispatch`. **Bindings are read when the dispatch runs, not when it is
  fired.**
- New `PlatformComputeShader.setBufferFire(tag, buffer)`: a bind ordered with
  dispatches, reads and writes, so bind → fire → rebind → fire is safe on one
  shader. Default delegates to `setBuffer`, which is correct for backends that
  run in order.
- New `PlatformBuffer.writeRawBytes(bytes, {dstByteOffset = 0})` for large
  transfers; implementations should stream in bounded chunks. Offset and
  length must be multiples of 4. The default uses `write` at offset 0 and
  throws `UnsupportedError` for any other offset.

## 1.5.7

- Version lockstep with minigpu 1.5.7; no interface change.

## 1.5.6

- New `MinigpuPlatform.preferDisplayAdapter([enable])` and
  `MinigpuPlatform.selectedAdapterName`, with defaults (`false` / `null`) so
  existing implementations keep compiling.

## 1.5.5

## 1.5.4

- fix release version pins

## 1.5.3

- Add `PlatformSharedOutputTexture.copyFromBufferAsync` and
  `PlatformVideoTexture.bgraToRgbaSharedOutputAsync` (the defaults call the
  synchronous versions), so backends can run the shared-output copy and
  present sync off the calling isolate.

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

- sync with other versions
