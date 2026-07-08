
# minigpu

A Flutter library for cross-platform GPU compute shaders integrating WGSL, GPU.CPP, and WebGPU via Dawn.

Try it: [https://minigpu.practicalxr.com/](https://minigpu.practicalxr.com/)

Use it: [https://pub.dev/packages/minigpu](https://pub.dev/packages/minigpu)

Tensor package: [https://pub.dev/packages/gpu_tensor](https://pub.dev/packages/gpu_tensor)

- [x] Windows
- [x] Linux
- [x] Mac
- [x] Web
- [x] Android
- [x] iOS

**Disclaimer: This library is experimental and supported at will, use in production at your own risk.**

## Three Things to Know

1. Dawn can take a while to build and our builds need a recent version of CMake. Run with -v to see errors and progress.
2. This package uses dart native assets, which are now supported out of the box in Dart >=3.11 and Flutter stable.
3. For flutter web, add

```html
  <script src="assets/packages/minigpu_web/web/minigpu_web.loader.js"></script>
  <script>
    _minigpu.loader.load();
  </script>
```

to your web/index.html file.

### Installation

Make sure you have:

- Dart SDK >=3.11 or Flutter stable channel.
- Emscripten 4.0.10 or greater for web support.
- CMake 3.22 or greater.

Add the following to your `pubspec.yaml`:

```yaml
dependencies:
  minigpu: ^1.5.2
```

Then run:

```shell
dart pub get
```

### Flutter apps: use `minigpu_flutter` instead

If you are building a Flutter app, add `minigpu_flutter` rather than `minigpu` directly.  It re-exports the full `minigpu` API and adds a thin widget that fires registered teardown callbacks during hot reload — preventing stale `NativeCallable` invocations when the Dart isolate is rebuilt mid-dispatch.

```yaml
dependencies:
  minigpu_flutter: ^1.5.2
```

Wrap your root widget with `MinigpuBinding` and register a `destroySync` callback for any long-lived `Minigpu` instance:

```dart
import 'package:minigpu_flutter/minigpu_flutter.dart';

void main() {
  runApp(const MinigpuBinding(child: MyApp()));
}

// After gpu.init():
MinigpuFlutterBinding.addDisposeCallback(gpu.destroySync);

// When the resource is torn down normally (not via hot reload):
MinigpuFlutterBinding.removeDisposeCallback(gpu.destroySync);
await gpu.destroy();
```

## Getting Started

```console
 git clone https://github.com/PracticalXR/minigpu.git
 dart test

 dart:
 cd minigpu
 dart bin/example.dart

 flutter:
 cd minigpu/example
 flutter run -d chrome/windows/linux/android/ios
```

## Example

 ```dart

Future<void> _runGELU() async {
  // Initialize the GPU.
  final gpu = Minigpu();
  await gpu.init();
  // Create the compute shader.
  final shader = gpu.createComputeShader();

// Load the compute kernel code as a string.
  shader.loadKernelString('''
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654; // sqrt(2.0 / PI)
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>
) {
  let i: u32 = GlobalInvocationID.x;
  if (i < arrayLength(&inp)) {
    let x: f32 = inp[i];
    out[i] = select(
      0.5 * x * (1.0 + tanh(
        GELU_SCALING_FACTOR * (x + .044715 * x * x * x)
      )),
      x,
      x > 10.0
    );
  }
}
''');

// Define the buffer size and generate input data.
  final bufferSize = 100;
  final inputData = Float32List.fromList(
    List<double>.generate(bufferSize, (i) => i / 10.0),
  );
  print('bufferSize: ${inputData.lengthInBytes}');
  final memSize = bufferSize * 4; // 4 bytes per float32

// Create GPU buffers for input and output data.
  final inputBuffer = gpu.createBuffer(bufferSize, memSize);
  final outputBuffer = gpu.createBuffer(bufferSize, memSize);

// Upload the input data.
 await inputBuffer.write(inputData, bufferSize);

// Bind the buffers to the shader.
  shader.setBuffer('inp', inputBuffer);
  shader.setBuffer('out', outputBuffer);

// Calculate the number of workgroups required.
  final workgroups = ((bufferSize + 255) / 256).floor();

// Dispatch the compute shader.
  await shader.dispatch(workgroups, 1, 1);

// Read the output data.
  final outputData = Float32List(bufferSize);
  await outputBuffer.read(outputData, bufferSize);

// Update the UI 
  setState(() {
    final result = outputData.sublist(0, 16).map((value) => value.toDouble()).toList();
    print('Result: $result');
  });
}
  ```

## GPU Buffer Copy

Copy a GPU buffer to another GPU buffer entirely on the GPU — no CPU round-trip, no PCIe transfer.

```dart
final gpu = Minigpu();
await gpu.init();

const n = 1024; // number of u32 elements
final src = gpu.createBuffer(n * 4, BufferDataType.uint32);
final dst = gpu.createBuffer(n * 4, BufferDataType.uint32);

// ... fill src with data via src.write() or a compute shader ...

// Copy src → dst on the GPU.  The WGSL copy shader is created once and
// reused across subsequent calls.
await gpu.copyBuffer(src, dst, elementCount: n);

// Read results back to verify (or pass dst to the next shader stage).
final result = Uint32List(n);
await dst.read(result, n, dataType: BufferDataType.uint32);
```

`elementCount` is the number of **elements** (not bytes) to copy. Both buffers
must be at least `elementCount × 4` bytes. Partial copies are supported:
elements beyond `elementCount` in `dst` are untouched.

The copy uses a cached WGSL compute shader bound via `setBufferAtSlot`; the
shader is created on the first `copyBuffer` call and reused for all subsequent
ones on the same `Minigpu` instance.

## VRAM Usage

Query dedicated GPU memory usage on supported platforms (Windows D3D12 via DXGI). Returns `-1` on unsupported platforms (Linux, Mac, Web, Android, iOS).

```dart
final gpu = Minigpu();
await gpu.init();

final vramBytes = gpu.queryVramBytes();
if (vramBytes >= 0) {
  final vramMB = vramBytes / (1024 * 1024);
  print('VRAM in use: ${vramMB.toStringAsFixed(1)} MB');
} else {
  print('VRAM query not supported on this platform.');
}
```

## GPU Video Interop

Import CPU or GPU video frames (e.g. from a camera or screen-capture library) directly into minigpu as GPU textures — no extra copy required on platforms with zero-copy support.

### Supported formats

| `ExternalPixelFormat` | Planes | Notes |
|-----------------------|--------|-------|
| `rgba32`              | 1      | Standard 8-bit RGBA, row-major |
| `bgra32`              | 1      | Standard 8-bit BGRA (typical of D3D11 / Windows screen capture) |
| `nv12`                | 2      | Luma (Y) plane + interleaved chroma (UV) plane |

### Import a CPU frame and read back RGBA

```dart
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:minigpu/minigpu.dart';

Future<void> importFrame(Uint8List rgbaPixels, int width, int height) async {
  final gpu = Minigpu();
  await gpu.init();

  // Copy pixel data onto the native heap so the pointer stays valid.
  final ptr = malloc<Uint8>(rgbaPixels.length);
  ptr.asTypedList(rgbaPixels.length).setAll(0, rgbaPixels);

  final tex = gpu.importVideoFrame(ExternalVideoBuffer(
    contentType: ExternalContentType.cpu,
    pixelFormat: ExternalPixelFormat.rgba32,
    width: width,
    height: height,
    planes: [
      ExternalPlane(
        dataPtr: ptr.address,
        width: width,
        height: height,
        strideBytes: width * 4,
      ),
    ],
  ));

  if (tex != null) {
    // Built-in NV12→RGBA or RGBA passthrough via an internal compute pass.
    final outBuf = tex.toRGBA();
    final bytes = Uint8List(width * height * 4);
    await outBuf.read(bytes, width * height * 4, dataType: BufferDataType.uint8);

    // Use bytes ...

    outBuf.destroy();
    tex.destroy();
  }

  malloc.free(ptr);
}
```

### Use an imported texture in a custom compute shader

Bind the imported texture to a specific slot with `setOnShader`, then set storage buffers for the remaining bindings using `setBufferAtSlot` (use explicit slot numbers to coordinate with the texture slot):

```dart
// WGSL: binding 0 = imported texture, binding 1 = output, binding 2 = params
const kShader = '''
@group(0) @binding(0) var in_tex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf : array<u32>;
struct Params { width: u32, height: u32 }
@group(0) @binding(2) var<storage, read_write> params : Params;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let px = textureLoad(in_tex, vec2<u32>(gid.x, gid.y), 0);
  // ... transform px, write to out_buf ...
  out_buf[gid.y * params.width + gid.x] =
      u32(px.r * 255.0) | (u32(px.g * 255.0) << 8u) |
      (u32(px.b * 255.0) << 16u) | (u32(px.a * 255.0) << 24u);
}
''';

final cs = gpu.createComputeShader();
cs.loadKernelString(kShader);

// Bind imported texture to slot 0.
tex.setOnShader(cs, 0);

// Use setBufferAtSlot for explicit slot numbers when mixing texture + buffer
// bindings — avoids the sequential tag-assignment of setBuffer().
cs.setBufferAtSlot(1, outBuf);
cs.setBufferAtSlot(2, paramsBuf);

await cs.dispatch((width + 7) ~/ 8, (height + 7) ~/ 8, 1);
```

> **Note**: Always use `var<storage, read_write>` (not `var<uniform>`) for parameter structs when setting buffers via `setBuffer` / `setBufferAtSlot` — the internal bind group layout maps all buffers to `WGPUBufferBindingType_Storage`.

### Import an NV12 frame

```dart
// NV12: Y plane is width×height bytes, UV plane is width×(height/2) bytes.
final yPtr  = malloc<Uint8>(width * height);
final uvPtr = malloc<Uint8>(width * (height ~/ 2));
// ... fill yPtr / uvPtr from capture callback ...

final tex = gpu.importVideoFrame(ExternalVideoBuffer(
  contentType: ExternalContentType.cpu,
  pixelFormat: ExternalPixelFormat.nv12,
  width: width,
  height: height,
  planes: [
    ExternalPlane(dataPtr: yPtr.address,  width: width,        height: height,        strideBytes: width),
    ExternalPlane(dataPtr: uvPtr.address, width: width ~/ 2,   height: height ~/ 2,   strideBytes: width),
  ],
));

// toRGBA() applies BT.709 full-range NV12→RGBA conversion on the GPU.
final outBuf = tex!.toRGBA();
```

### Resource lifecycle

- Call `tex.destroy()` and `outBuf.destroy()` when done with each frame.
- Use `gpu.liveBufferCount` in tests to assert no per-frame allocation leaks.
- Call `gpu.destroyAllTrackedShaders()` between test groups to free D3D12 pipeline resources.

### Cross-API shared output texture (zero-copy GPU hand-off)

Some pipelines need to hand a GPU texture to another native API on the same
device — for example, a hardware video encoder (NVENC/AMF/QuickSync via
FFmpeg's D3D11VA encoder) or an OS compositor — without ever copying through
system memory. `SharedOutputTexture` gives you an 8-bit BGRA texture
allocated on minigpu's underlying device and exported as an OS-level shared
handle.

#### Platform support

| Platform | Backing             | Status     |
|----------|---------------------|------------|
| Windows  | D3D11 NT shared handle (`DXGI_FORMAT_B8G8R8A8_UNORM`) | ✅ Working |
| macOS    | `IOSurface`         | ⏳ Planned |
| Linux    | `dmabuf`            | ⏳ Planned |
| Android  | `AHardwareBuffer`   | ⏳ Planned |
| Web      | (no equivalent yet) | ❌ Returns `null` |

`createSharedOutputTexture` returns `null` on every platform that is not
yet implemented; check for null and fall back to a copy path.

> **Note on the Windows implementation.** The shared texture is created
> *in D3D11* on the same DXGI adapter as Dawn (looked up via
> `EnumAdapterByLuid`), exported through `IDXGIResource1::CreateSharedHandle`,
> and re-imported into Dawn through
> `wgpuDeviceImportSharedTextureMemory(DXGISharedHandleDescriptor)`. We
> tried the inverse direction (D3D12 → D3D11 via `OpenSharedResource1`)
> first; it returns `E_INVALIDARG` on current NVIDIA drivers, so this
> direction is the working path. The format is **BGRA**, not RGBA, because
> NVENC / AMF / QSV / MediaFoundation D3D11VA hwframes pools require
> BGRA — `CopySubresourceRegion` silently fails between BGRA and RGBA
> texture formats (different DXGI type groups).

#### Quick start: shared texture as a GPU effect destination

The most common shape is "render an effect with a compute shader, then
hand the result to FFmpeg without a CPU round-trip":

```dart
import 'package:minigpu/minigpu.dart';

final gpu = Minigpu();
await gpu.init();

// 1. Allocate the shared output once for your stream resolution.
final shared = gpu.createSharedOutputTexture(1920, 1080);
if (shared == null) {
  // Not supported on this platform/backend yet — fall back to a CPU copy.
  return;
}

// 2. (Optional, Windows only) Get an `ID3D11Device*` on the SAME DXGI
//    adapter as Dawn so the encoder can read `shared.d3d11TexturePtr`
//    directly without `OpenSharedResource1`. This pointer's ownership
//    transfers to the caller — pass it to FFmpeg's hwdevice context.
final d3d11Device = gpu.createD3D11DeviceOnDawnAdapter(); // 0 on non-Windows.

// 3. Per frame:
//    a. Run your effect shader, producing an RGBA8-packed `Buffer`.
//    b. Copy that buffer into the shared texture (pure GPU compute).
//    c. Hand `shared.d3d11TexturePtr` (or `shared.d3d11Handle`) to
//       FFmpeg's D3D11VA encoder.
final rgbaBuf = await myEffect.apply(inputBuf, width, height);
final ok = shared.copyFromBuffer(rgbaBuf);
// ... encode ...

// 4. When the stream ends:
shared.destroy();
await gpu.destroy();
```

#### Two ways for the consumer to access the texture

`SharedOutputTexture` exposes both an NT handle and a raw texture pointer.
Pick whichever fits your consumer:

| Getter             | Type                  | When to use |
|--------------------|-----------------------|-------------|
| `d3d11Handle`      | NT `HANDLE` (int)     | Consumer is a *separate* D3D11 device. Pass to `ID3D11Device1::OpenSharedResource1`. |
| `d3d11TexturePtr`  | `ID3D11Texture2D*` (int) | Consumer shares the same `ID3D11Device` (e.g. you injected the device returned by `createD3D11DeviceOnDawnAdapter()` into FFmpeg). Skips `OpenSharedResource1` entirely. |

Both refer to the same underlying texture; do **not** `CloseHandle` the
NT handle — it is owned by `SharedOutputTexture` and closed in
`destroy()`.

#### Importing a captured frame and writing into the shared texture

If you start from a captured BGRA frame (e.g. screen capture via miniAV)
and want to deliver an RGBA result, the import + swizzle path looks like:

```dart
final tex = gpu.importVideoFrame(buf);
if (tex != null) {
  final ok = tex.bgraToRgbaSharedOutput(shared);
  tex.destroy();
  if (ok) {
    // The encoder can now read `shared` from its D3D11 view and submit it.
  }
}
```

Notes:

- `copyFromBuffer` and `bgraToRgbaSharedOutput` each run a tiny WGSL
  compute pass (`texture_storage_2d<bgra8unorm, write>`, 8×8 workgroup)
  that writes directly into the shared texture. Both block on the GPU
  queue before returning, so the texture is safe to consume on the
  D3D11 side as soon as the call returns — no extra fence required.
- Source and destination must have matching width and height.
- The shared texture is BGRA8. If your encoder needs NV12, let the
  encoder do RGBA/BGRA→NV12 on the GPU (`AV_PIX_FMT_D3D11` + scaler
  filter, or NVENC's built-in CSC).
- `SharedOutputTexture` uses a `Finalizer`, but you should still call
  `destroy()` deterministically when the stream ends to release the NT
  handle and the underlying D3D11 texture promptly. Release any external
  D3D11 view (e.g. `OpenSharedResource1` result) first.

For a complete end-to-end example (5K screen capture →
minigpu effect → FFmpeg HEVC NVENC) see
[`miniav_tools/examples/screenshare_mp4`](../../miniav_tools/examples/screenshare_mp4/).

## Funding

  If you are interested in funding further easy-to-port gpu development, please submit an inquiry on [https://practicalxr.com](https://practicalxr.com).
