# minigpu_web CHANGELOG

## 1.6.1

- released 08/13/26 - MR

## Unreleased

- **`writeRawBytes` no longer allocates and fills a whole second list per
  call.** It built a fresh `Uint32List` the size of the payload and memcpy'd
  into it on every call, only to reinterpret the same bytes as words. On a
  streaming path that is a whole-frame allocation plus a whole-frame copy per
  frame — 33 MB at 4K — and the allocation is per-frame garbage large enough to
  be a frame-time spike source in its own right, not just a copy. It now takes a
  4-byte-aligned VIEW over the caller's bytes, which aliases the same store and
  costs nothing. The copy remains only for the one case a view cannot describe
  (a caller-supplied view whose `offsetInBytes` is not 4-aligned). This is the
  web half of the same bug fixed natively in minigpu_ffi, and it matters more
  here: the browser has no equivalent of the native path's escape hatches.

## 1.6.0

- Version bump to keep the minigpu 1.6.0 family aligned. No functional change
  in this package (the fix is native-only and does not apply to the web build).

## 1.5.9

- Version lockstep with minigpu 1.5.9; no behavior change. `setBufferFire`
  aliases `setBuffer` on web, which stays correct now that the facade always
  binds through it — single-threaded wasm executes GPU tasks in call order.

## 1.5.8

- Implement the new platform-interface members on web:
  `setBufferFire` aliases `setBuffer` (single-threaded wasm runs GPU tasks in call
  order, so the plain bind already has FIFO semantics); `dispatchFire` drops the
  returned promise (`queue.submit` is synchronous in JS WebGPU — the promise only
  covers call plumbing, so submission order holds); `writeRawBytes` writes at
  offset 0 via the u32 path and throws `UnsupportedError` otherwise.
- `listAdapters` / `createSecondaryPlatform` / `drainSpinBudgetMs` inherit the
  interface's no-op defaults — no multi-adapter contexts on web, and the native
  drain fix does not apply.

## 1.5.7

- Version lockstep with minigpu 1.5.7; no behavior change.

## 1.5.6

- Version lockstep with minigpu 1.5.6 (new `preferDisplayAdapter` /
  `selectedAdapterName` platform members inherit the interface's no-op
  defaults on web — no behavior change).

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

- Refactored web memory management
