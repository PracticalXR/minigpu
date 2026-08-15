# minigpu_platform_interface CHANGELOG

## 1.7.0

- Add the persistent shader cache surface: `configureShaderCache(...)`,
  `shaderCacheStats`, `shaderCacheDirectory`, `clearShaderCache()`, and the
  `ShaderCacheStats` value type. Every method has a default implementation
  (no-op / null / 0), so existing platform implementations need no change —
  web inherits the defaults, which is correct: without Dawn there is no
  compiled shader to cache.

- Add `MinigpuPlatform.drainWorkQueue()` — blocks until every GPU task already
  queued on the native worker thread has run. Default implementation is a no-op,
  so existing platform implementations (web) need no change.

## 1.6.1

- released 08/13/26 - MR

## 1.6.0

- Version bump to keep the minigpu 1.6.0 family aligned. No functional change
  in this package.

## 1.5.9

- No interface change. Note for backend implementors: as of minigpu 1.5.9 the
  facade routes ALL buffer binds through `PlatformComputeShader.setBufferFire`,
  never `setBuffer`. A backend whose `setBufferFire` is not actually ordered
  against its dispatches will corrupt fire-and-forget callers. The default
  implementation delegates to `setBuffer`, which remains correct only for
  inherently in-order backends (e.g. single-threaded web).

## 1.5.8

All additions have concrete defaults, so a backend that `extends` these classes
keeps compiling unchanged. **A class that `implements` them must add the new
members** — Dart requires every member to be defined when implementing an
interface, whether or not the declaration has a body.

- New `GpuAdapterInfo` (name, total/used dedicated VRAM, `freeVramBytes`) and
  `MinigpuPlatform.listAdapters()` — defaults to `const []`.
- New `MinigpuPlatform.createSecondaryPlatform(String adapterFilter)`: an
  INDEPENDENT platform context on the matching adapter, with its own device,
  queue and task FIFO. Never mix two contexts' resources in one dispatch.
  Defaults to `null` (no multi-adapter support).
- New `MinigpuPlatform.drainSpinBudgetMs` — runtime provenance for the loaded
  native binary's event-drain spin budget; defaults to `null` where it cannot be
  asked. A build without the drain fix burns a ~15.6 ms Windows timer quantum per
  GPU wait, so a latency-path caller can assert `> 0` rather than trust that it
  linked a fresh artifact.
- New `PlatformComputeShader.dispatchFire(x, y, z)` — fire-and-forget dispatch;
  call order with reads/writes is preserved, so a later awaited
  `PlatformBuffer.read` synchronizes any number of fired dispatches. Default
  delegates to `dispatch`. **Bindings are snapshotted when the dispatch runs, not
  when it is fired.**
- New `PlatformComputeShader.setBufferFire(tag, buffer)` — ordered bind that
  joins the dispatch/read/write FIFO, making bind → fire → rebind → fire
  race-free on one shader. Default delegates to `setBuffer`, already correct for
  inherently in-order backends.
- New `PlatformBuffer.writeRawBytes(bytes, {dstByteOffset = 0})` for LARGE
  transfers: implementations should stream in bounded chunks so neither host
  scratch nor driver staging holds the full payload. Offset and length must be
  4-byte multiples; the default falls back to `write` at offset 0 and throws
  `UnsupportedError` otherwise.

## 1.5.7

- Version lockstep with minigpu 1.5.7; no interface change.

## 1.5.6

- New `MinigpuPlatform.preferDisplayAdapter([enable])` and
  `MinigpuPlatform.selectedAdapterName` with safe concrete defaults
  (`false` / `null`) so existing platform implementations keep compiling.

## 1.5.5

## 1.5.4

- fix release version pins

## 1.5.3

- Add `PlatformSharedOutputTexture.copyFromBufferAsync` and
  `PlatformVideoTexture.bgraToRgbaSharedOutputAsync` (default implementations
  delegate to the synchronous variants) so backends can run the shared-output
  GPU copy + present sync off the calling isolate.

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

- sync with other versions
