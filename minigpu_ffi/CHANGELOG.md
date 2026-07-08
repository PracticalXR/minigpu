# minigpu_ffi CHANGELOG

## 1.5.7

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
