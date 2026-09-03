# minigpu_view CHANGELOG

## 1.8.1

- Version alignment with the minigpu 1.8.1 family release.

## 1.7.0

- **Windows: one engine texture per source handle, not per controller —
  fixes resource exhaustion with multi-buffered producers.** A producer that
  double/triple-buffers its output presents a different shared handle each
  frame. The plugin used to re-point a single Flutter texture at each new
  handle, and the engine re-created its ANGLE offscreen swapchain (color +
  depth) for every content frame — thousands of allocations per minute that
  eventually failed with `E_OUTOFMEMORY` ("SwapChain11: Could not create
  depthstencil surface"). The plugin now keeps one registered texture per
  (controller, handle) — a ring of N handles costs N swapchains total,
  created once — and re-presenting a known handle is just
  `MarkTextureFrameAvailable`. Least-recently-presented textures are evicted
  above a small cap, so an ever-changing handle stream stays bounded too. The
  Dart `MinigpuPreviewController` already propagated `textureId` changes to
  its widget, so no API change.
- **Host-memory planar frames render on the GPU: NV12, I420/YV12, YUY2, BGRA.**
  `GpuPane.showPlanar` / `PlanarCpuGpuFrame` upload the planes as bytes and
  unpack them with a compute shader. These are what cameras and screen capture
  actually produce — RGBA usually only appears because someone asked a
  converter for it — and the previous guidance for them was to decode an image
  on the UI isolate. That is O(pixels) inside the capture callback: at 1080p60
  it exceeds a frame interval and freezes the app rather than slowing the
  preview. STRIDES ARE HONOURED, which matters because capture APIs pad rows
  and assuming `width * bpp` shears the picture progressively down the frame.
  BT.709 limited range by default (what MF/WGC deliver), overridable — guessing
  it wrong is silent and shows up as a quality loss that has nothing to do with
  the codec.
- **`MiniAVBuffer.asGpuFrame(gpu)`** — one call from a capture buffer to
  something displayable, whichever shape it arrived in (GPU handle or any of
  the host formats above). The format knowledge now lives in this package,
  which has to know both vocabularies, rather than being rewritten in every app
  that shows a camera. Returns null rather than guessing at an unknown layout:
  a wrong guess is skewed or miscoloured but not obviously broken, which is
  worse than an empty pane.

- **`GpuPane` — a pane that owns its destination texture and accepts frames in
  whatever shape a producer hands them over.** `showSharedHandle`,
  `showBuffer` and `showBytes` cover the common sources in one call each;
  `GpuFrame` stays public for custom producers. Consumers were writing a
  display path per producer, and typically an extra CPU one — read the frame
  back, decode a `dart:ui` image, paint it — which at 4K costs more than a
  frame interval on the isolate that also builds the UI and stalls the app
  rather than degrading. Host-memory frames now upload once (through a pooled
  buffer, so no per-frame VRAM allocation) and composite on the GPU like every
  other source.
  WHEN NOT TO USE IT: if you already have a `SharedOutputTexture`, present it
  with `asPreviewSource()` — that copies nothing. A pane costs one
  GPU-to-GPU blit and buys two things: it takes every source shape, and
  because the caller drives `flush()`, the pane's update rate is something the
  caller controls rather than something the producer decides. That matters
  more than it sounds: a platform `Texture` does not hold a snapshot, so a
  pane pointed at a producer's texture resamples it on any composite — pacing
  `present` alone cannot bound how often it changes.
- **`RawSharedTexturePreviewSource` / `RawGpuBufferPreviewSource`** — present
  sources built from raw integer handles, for producers that publish a surface
  as `{sharedHandle, texturePtr, width, height}` across an API boundary rather
  than as an object. Consumers had nothing to call `asPreviewSource()` on and
  were hand-rolling the wire format; getting it subtly wrong is silent.

## 1.6.1

- released 08/13/26 - MR

## 1.6.0

- Version bump to keep the minigpu 1.6.0 family aligned. No functional change
  in this package.

## 1.5.10

- Fixes the `miniav` constraint, which was still `^0.5.2` while miniav is at
  0.7.0 — this package was the only thing blocking any consumer that depends on
  both. Now `^0.7.0`.
- The published 1.5.9 shipped `minigpu`/`minigpu_platform_interface` at 1.5.8;
  those were bumped to 1.5.9 in the repo but never republished. This release
  carries them.
- TRAP: `pubspec_overrides.yaml` applies in-repo only, so a stale constraint here
  is invisible locally and fails only for consumers.

## 1.5.9

- Web: the `webVideoFrame` present path now works. A raw JS `VideoFrame` cannot
  cross the method-channel `StandardMessageCodec`, so the plugin resolves a
  codec-safe int `videoFrameHandle` from a shared global registry
  (`globalThis.miniavVideoFrameRegistry`) — mirroring the existing WebGPU
  `bufferHandle` path. (The prior branch expected the JSObject directly under
  `frame` and was never actually reachable.) This is what makes
  `miniav_player` web video playback present.

## 1.5.8

## 1.5.7

- Windows: `present()` now probes that a shared handle can actually be opened
  on the default (primary display) adapter before registering it, and fails
  with `cross_adapter` when the producer texture lives on a different GPU.
  Previously such handles registered fine and then failed every raster-time
  bind ("Binding D3D surface failed.") as a silently black texture with no
  error surfaced to Dart. Apps can catch the error and fall back to a CPU
  preview; producers should call `mgpuPreferDisplayAdapter` before GPU init
  to avoid the mismatch entirely.

## 1.5.6

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

## 0.1.0

- Initial release.
- Zero-copy Flutter texture rendering via DXGI shared handle (Windows) and platform-native equivalents.
- `SharedOutputTextureSource` adapter for `minigpu` `SharedOutputTexture`.
- `MiniAVBufferSource` adapter for `miniav` GPU buffers.
- `MiniavGpuPreview` widget with platform-channel-based texture registration.
- Web stub for multi-platform pub compatibility.
