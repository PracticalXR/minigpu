# minigpu_view

Zero-copy GPU-to-Flutter rendering for [minigpu](../minigpu) `SharedOutputTexture` and [miniav](../../miniav/miniav) GPU buffers.

GPU frames produced by a minigpu compute pipeline or a miniav capture device are handed to Flutter's texture registry as a raw GPU handle, no CPU readback, no pixel copy across the bus on every frame.

## How it works

```
┌─────────────────────────────────────────────────────────────┐
│  Your Dart code                                             │
│                                                             │
│  SharedOutputTexture ──.asPreviewSource()──► PreviewSource  │
│                                                    │        │
│  MiniavPreviewController.present(source) ◄─────────┘        │
│           │                                                  │
│           │  platform channel  'minigpu_view'                │
└───────────┼──────────────────────────────────────────────────┘
            ▼
┌─────────────────────────────────────────────────────────────┐
│  Native plugin (Windows)                                    │
│                                                             │
│  IDXGIResource::GetSharedHandle  ──►  ANGLE / EGL           │
│  FlutterTextureRegistrar::RegisterTexture                   │
│         │                                                   │
│         └──► textureId ──► Dart ──► Texture(textureId)      │
└─────────────────────────────────────────────────────────────┘
```

On **Windows** the plugin registers a `kFlutterDesktopGpuSurfaceTypeDxgiSharedHandle` texture. Flutter's ANGLE-backed rasterizer opens the legacy DXGI shared handle on its own D3D11 device — no device synchronisation is needed on the Dart side.

On **Web** the plugin registers a WebGPU canvas element as an `HtmlElementView`.

## Installation

```yaml
dependencies:
  minigpu_view:
    git:
      url: https://github.com/PracticalXR/minigpu.git
      path: minigpu_view
```

Or with a local path override:

```yaml
dependencies:
  minigpu_view: ^0.1.0

dependency_overrides:
  minigpu_view:
    path: ../minigpu_view
```

## Quick start

### 1 — Create a controller

```dart
final controller = MiniavPreviewController();
```

Hold this in your `State` and dispose it in `dispose()`.

### 2 — Produce GPU frames

Use a minigpu `SharedOutputTexture` as the frame source. Create one once and reuse it across frames:

```dart
final gpu = Minigpu();
await gpu.initialize();

final sharedTex = gpu.createSharedOutputTexture(width, height);
```

Write into it from a compute shader output buffer:

```dart
// outputBuffer holds 4 × f32 per pixel (RGBA in [0, 1]).
sharedTex.copyFromBufferF32(outputBuffer);

// Or from a packed RGBA8 u32 buffer (e.g. GpuEffect output):
sharedTex.copyFromBuffer(outputBuffer);
```

### 3 — Present the frame

```dart
await controller.present(sharedTex.asPreviewSource());
```

Call this once per frame — for example in a `Timer.periodic` loop or after each GPU dispatch.

### 4 — Display in your widget tree

```dart
MiniavGpuPreview(
  controller: controller,
  fit: BoxFit.contain,            // BoxFit.fill stretches to fill bounds
  placeholder: const Center(     // shown before the first frame arrives
    child: CircularProgressIndicator(),
  ),
)
```

`MiniavGpuPreview` is a plain `StatelessWidget` that listens to the controller via `AnimatedBuilder` — no `setState` needed in the parent.

## API reference

### `MiniavPreviewController`

| Member | Description |
|---|---|
| `present(PreviewSource)` | Pushes a frame to the platform. Returns once the texture registry slot is confirmed. |
| `textureId` | The Flutter texture id (`null` before the first `present`). |
| `size` | Natural pixel size of the most recent frame. |
| `presentedAtUs` | Broadcast stream of host-side ack timestamps (µs). Use for backpressure. |
| `dispose()` | Releases platform resources. Must be called when no longer needed. |

`MiniavPreviewController` is a `ChangeNotifier` — widgets rebuild automatically when `textureId` or `size` changes.

### `MiniavGpuPreview`

| Property | Default | Description |
|---|---|---|
| `controller` | — | Required. |
| `fit` | `BoxFit.contain` | How to fit the frame inside the available space. |
| `alignment` | `Alignment.center` | Alignment when `fit` leaves empty space. |
| `filterQuality` | `FilterQuality.medium` | Passed to Flutter's `Texture` widget. |
| `placeholder` | `SizedBox.shrink()` | Shown until the first frame arrives. |

### Adapters

| Extension | Source | Notes |
|---|---|---|
| `SharedOutputTexture.asPreviewSource()` | minigpu | Zero-copy on Windows via DXGI shared handle. |
| `MiniAVBuffer.asPreviewSource()` | miniav | Returns `null` for CPU buffers; use your existing CPU path in that case. |

### Exceptions

| Exception | When |
|---|---|
| `UnsupportedPreviewException` | Platform doesn't support the given `PreviewSourceKind` yet. Catch and fall back to a CPU path. |
| `PreviewPresentException` | Host plugin rejected the handle (invalid, device lost, format mismatch). |

## Example

See [`example/lib/main.dart`](example/lib/main.dart) for a self-contained runnable demo that drives a WGSL animated gradient through `SharedOutputTexture` and displays it via `MiniavGpuPreview`.

```
cd example
flutter run -d windows
```

## Platform support

| Platform | Status | Surface type |
|---|---|---|
| Windows | ✅ | `kFlutterDesktopGpuSurfaceTypeDxgiSharedHandle` (ANGLE cross-device) |
| Web | ✅ | `HtmlElementView` (WebGPU canvas) |
| macOS | 🚧 planned | Metal IOSurface |
| Linux | 🚧 planned | GBM / DMA-buf |
| Android | 🚧 planned | AHardwareBuffer |
| iOS | 🚧 planned | Metal IOSurface |

## Relationship to miniav_tools / FFmpeg encoder

`SharedOutputTexture` exposes **two** handles:

- `d3d11TexturePtr` — raw `ID3D11Texture2D*` on Dawn's D3D11 device. Used by the FFmpeg D3D11VA encoder (`miniav_tools`) because encoder and producer share the same device — no handle open needed.
- `d3d11Handle` — legacy DXGI shared `HANDLE` (from `IDXGIResource::GetSharedHandle`). Used by this plugin because Flutter's ANGLE has its own D3D11 device and cannot accept a foreign texture pointer directly.

Both handles remain valid simultaneously, so encoding and display can run concurrently.
