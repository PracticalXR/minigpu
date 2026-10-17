# minigpu_web CHANGELOG

## 1.9.0

- Reads put the data at the start of the output list; it used to land at the
  element offset.
- Reads and writes take any typed list as bytes, including 64-bit lists and
  `ByteData`, which used to be dropped or ignored.
- A count past the input, the output or the GPU buffer throws `RangeError`.
- Overlapping reads and context calls now run one at a time, and a destroy
  waits for them. Before, they could hang or corrupt memory.
- `writeRawBytes` uploads straight from the caller's bytes at every offset.
- A failed module load is retried, and a page loads the loader script once.
- `setLogCallback` defaults to warnings and errors, like native.
- A read that fails throws `StateError` instead of returning stale data.
- Render pipelines implement `settled`: the module reports when a pipeline
  finishes compiling, so preparing one no longer polls.
- Compute shaders implement `buildPipeline()`, which compiles without
  dispatching.
- `toRGBAAsync` throws `UnsupportedError`, like `toRGBA`.
- A buffer bound without a binding kind takes the access its kernel
  declares, as on native.
- The bundled wasm module is rebuilt, so it carries the native fixes listed
  for minigpu_ffi, with no fast-math and a 1 MB Asyncify stack instead of
  64 MB.

- Retire GPU-only resources after the exact creation device's observed WebGPU
  loss. Keep host-reader teardown separate and reject healthy, stale or merely
  unavailable device lookups.
- Add final module-retirement observation and root-owned browser recovery.
  Replacement preserves required features/limits on a fresh browser-selected
  device; it does not claim native physical-adapter identity. Rejected candidates
  must retire before explicit Retry. Outstanding readers still block replacement.

## 1.8.1

- Implement `webBufferHandle` on `WebBuffer` (via `mgpuGetWGPUBufferHandle`),
  exposing the WebGPU buffer handle for zero-copy presentation on the same
  device.

## 1.7.0

- **`writeRawBytes` no longer allocates and copies a second list per call.**
  It now uses a 4-byte-aligned view of the caller's bytes. At 4K the old copy
  was a 33 MB allocation per frame, and the garbage it left could cause
  frame-time spikes. A copy is still made when the caller's view does not start on a
  4-byte boundary.

## 1.6.1

- released 08/13/26 - MR

## 1.6.0

- Version bump for the minigpu 1.6.0 family. No functional change (that fix is
  native-only).

## 1.5.9

- Version lockstep with minigpu 1.5.9; no behaviour change. `setBufferFire`
  is an alias for `setBuffer` on web, which is correct because the browser
  runs GPU work in call order.

## 1.5.8

- Implements the new platform-interface members. `setBufferFire` aliases
  `setBuffer` and `dispatchFire` does not wait for the dispatch; on web both
  already run in call order. `writeRawBytes` works at offset 0 and throws
  `UnsupportedError` for any other offset.
- `listAdapters`, `createSecondaryPlatform` and `drainSpinBudgetMs` keep the
  interface's defaults: there are no multi-adapter contexts on web, and the
  native GPU-wait fix does not apply.

## 1.5.7

- Version lockstep with minigpu 1.5.7; no behaviour change.

## 1.5.6

- Version lockstep with minigpu 1.5.6. The new `preferDisplayAdapter` /
  `selectedAdapterName` members keep the interface's no-op defaults on web.

## 1.5.5

## 1.5.4

- fix release version pins

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

- Refactored web memory management
