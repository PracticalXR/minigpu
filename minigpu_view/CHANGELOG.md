# minigpu_view CHANGELOG

## 1.9.0

- YUV frames show the right colours. The conversion over-saturated chroma,
  by about 30% in limited-range frames and 14% in full-range ones.
- `PlanarCpuGpuFrame` and `GpuPane.showPlanar` take a `matrix`
  (`YuvMatrix.bt709`, `bt601` or `bt2020`; BT.709 by default), and
  `asGpuFrame` uses the matrix and range the capture reports.
  `kPlanarCfgWords` is now 11: the matrix is the last word.
- Requires minigpu 1.9.0 and miniav 0.8.0.
- `GpuPane` no longer goes blank while it resizes: the old picture stays up
  until the new one is shown. New `MinigpuPreviewController.retire(source)`
  drops one replaced source.
- Disposing or retiring a preview now answers only after the engine has let
  go of the texture.
- Windows: GPU frames shared through an NT handle now bind. The check retries
  with `OpenSharedResource1`, no longer reuses an earlier frame's answer, and
  recovers from a removed device.
- `MiniAVBuffer.asPreviewSource()` returns null for Windows GPU-handle frames
  instead of passing the handle where a texture was expected. Show them with
  `asGpuFrame(gpu)` and `GpuPane`. `asGpuFrame` takes the GPU route only for
  BGRA and RGBA handles; NV12 handles no longer go through the BGRA copy.
- CPU planar frames upload only the bytes the conversion reads, without a
  per-frame copy, and size odd-width chroma correctly. A stride shorter than
  a row is refused.
- Web: the canvas copy waits for dispatches issued before it, and under
  dart2wasm the view no longer reconfigures the canvas or rebuilds its
  pipelines every frame. `retire` is answered.

- Add fixed native platform cleanup for dormant preview parents and host-only
  partial setup. Bounded requests acknowledge actual platform close/window
  removal; missing pumping or failed work retains ownership. Child creation and
  closure share an admission gate. Caller retains the engine through cleanup.

- Add a bounded native same-engine registrar association and opaque binding API.
  Plugin teardown fences new work; closed generations retain their charges.
  Associated verification receives the core C identity, keeping the C++ wrapper
  private. Engine retention and independent renderer qualification are required;
  CPU/link checks do not qualify live capture.

- Add bounded trusted native platform dispatch for immutable previews, with
  irreversible source/deadline fences, callback-return ownership transfer and
  terminal shutdown quarantine. Headless CPU checks do not qualify a deployed
  engine or GPU capture; legacy channels remain separate.

- Add a separate trusted native Windows immutable-preview ownership API with
  bounded registration/descriptor loans, generation-safe callbacks and checked
  retirement receipts. Requires an independently qualified engine/thread profile;
  uncertain removal or cleanup retains charged owners. Legacy preview channels
  remain separate. CPU boundary checks do not qualify real engine/GPU execution.

## 1.8.2

- Increment downstream deps

## 1.8.1

- Version alignment with the minigpu 1.8.1 family release.

## 1.7.0

- **Windows: fix resource exhaustion with producers that rotate between
  several output textures.** The plugin re-pointed one Flutter texture at each
  new shared handle, so the engine rebuilt its swapchain on every frame until
  it failed with `E_OUTOFMEMORY` ("SwapChain11: Could not create depthstencil
  surface"). It now keeps one registered texture per controller and handle,
  evicting the least recently presented ones above a small cap. No API change.
- **Host-memory frames in NV12, I420/YV12, YUY2 and BGRA render on the GPU.**
  `GpuPane.showPlanar` / `PlanarCpuGpuFrame` upload the planes and convert
  them in a compute shader, instead of decoding an image on the UI isolate,
  which at 1080p60 can freeze the app. Row strides are honoured. Colour
  defaults to BT.709 limited range (what Media Foundation and Windows Graphics
  Capture deliver) and can be overridden.
- **New `MiniAVBuffer.asGpuFrame(gpu)`**: turns a miniav capture buffer into a
  displayable frame, whether it holds a GPU handle or one of the host formats
  above. Returns null for a layout it does not recognise rather than guessing.
- **New `GpuPane`**: a pane that owns its destination texture and accepts
  frames in any shape. `showSharedHandle`, `showBuffer` and `showBytes` cover
  the common sources, and `GpuFrame` stays public for custom producers.
  Host-memory frames are uploaded once through a pooled buffer and composited
  on the GPU, instead of a CPU decode-and-paint path that at 4K takes longer
  than a frame.
  When not to use it: if you already have a `SharedOutputTexture`, present it
  with `asPreviewSource()`, which copies nothing. A pane costs one GPU-to-GPU
  copy; in return it accepts any source and changes only when you call
  `flush()`, so you control its update rate. (A platform `Texture` shows
  whatever its source holds each time Flutter composites, so pacing `present`
  alone cannot limit how often it changes.)
- **New `RawSharedTexturePreviewSource` / `RawGpuBufferPreviewSource`**:
  preview sources built from raw integer handles, for producers that pass a
  surface as `{sharedHandle, texturePtr, width, height}` instead of as an
  object.

## 1.6.1

- released 08/13/26 - MR

## 1.6.0

- Version bump for the minigpu 1.6.0 family. No functional change in this
  package.

## 1.5.10

- Fix the `miniav` constraint: it was still `^0.5.2` while miniav is at 0.7.0,
  which blocked any app depending on both. Now `^0.7.0`.
- Depends on `minigpu` / `minigpu_platform_interface` 1.5.9; the published
  1.5.9 of this package still depended on 1.5.8.

## 1.5.9

- Web: the `webVideoFrame` present path now works. A JS `VideoFrame` cannot
  pass through the method channel, so the plugin looks it up by an int
  `videoFrameHandle` in a shared registry
  (`globalThis.miniavVideoFrameRegistry`), like the existing WebGPU
  `bufferHandle` path. This is what lets `miniav_player` show video on the
  web.

## 1.5.8

## 1.5.7

- Windows: `present()` now checks that a shared handle can be opened on the
  primary display's adapter before registering it, and fails with
  `cross_adapter` when the producer's texture is on another GPU. Before, such
  handles registered and then showed as a black texture with no error. Catch
  the error to fall back to a CPU preview; producers can call
  `mgpuPreferDisplayAdapter` before GPU init to avoid the mismatch.

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

## 0.1.0

- Initial release.
- Zero-copy Flutter texture rendering via DXGI shared handle (Windows) and
  platform-native equivalents.
- `SharedOutputTextureSource` adapter for `minigpu` `SharedOutputTexture`.
- `MiniAVBufferSource` adapter for `miniav` GPU buffers.
- `MiniavGpuPreview` widget with platform-channel-based texture registration.
- Web stub for multi-platform pub compatibility.
