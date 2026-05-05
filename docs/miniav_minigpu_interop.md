# MiniAV <---> Minigpu Interoperability Design

## 1. Introduction

This document outlines the design for enabling high-performance, zero-copy (where possible) interoperability between MiniAV video buffers and the Minigpu compute shader library. The primary goal is to allow raw video frames captured or provided by MiniAV to be efficiently used as input textures in Minigpu compute shaders, leveraging GPU-native handles when available.

## 2. Core Problem: Bridging Buffer Representations

MiniAV provides video frames in a `MiniAVBuffer` structure, which can contain either CPU-accessible pixel data or platform-specific GPU texture handles, using a wide variety of `MiniAVPixelFormat`s. Minigpu, built on Dawn (WebGPU), requires textures to be represented as `WGPUTexture` objects, often created from imported shared memory handles, and supports a specific set of `WGPUTextureFormat`s.

The key challenges are:

1. **Mapping Pixel Formats**: The extensive `MiniAVPixelFormat` enum needs to be mapped to the appropriate `WGPUTextureFormat`. Not all MiniAV formats will have a direct, efficient mapping.
2. **Format Negotiation**: Since MiniAV might offer formats that Minigpu cannot directly consume (or consume efficiently), a negotiation or conversion strategy is needed.
3. **Handling GPU Handles**: When MiniAV provides a GPU handle, Minigpu needs to import this handle correctly.
4. **Feature Capability Checks**: Minigpu must verify that the underlying WebGPU device supports the required texture formats and import mechanisms.
5. **Synchronization**: Ensuring that GPU operations are correctly synchronized.
6. **Library Independence**: Maintaining a clean separation between MiniAV and Minigpu.

## 3. Proposed Solution: External Buffer Abstraction with Negotiation

To maintain library independence, Minigpu will expose an "External Buffer" API. The application layer will be responsible for:

1. Querying supported formats from both MiniAV (for capture/source) and Minigpu (for consumption).
2. Negotiating a common, efficient format.
3. If necessary, requesting MiniAV to provide frames in the negotiated format or performing conversion in the application layer.
4. Mapping the (potentially converted) `MiniAVBuffer` to Minigpu's `MGPUExternalVideoBuffer` structure.

### 3.1. Minigpu External Buffer API

Minigpu will define C structures and enums that mirror the essential components of a video buffer.

**Key Structures in Minigpu (`minigpu_external.h`):**

* `MGPUExternalContentType`: Enum for buffer content (CPU, D3D11, Metal, DmaBuf, IOSurface, AHardwareBuffer).
* `MGPUExternalPixelFormat`: Enum for common pixel formats *that Minigpu can reasonably map to WGPUTextureFormat*. This list will be a subset of `MiniAVPixelFormat` or represent common interchange formats.
    *`MGPU_EXTERNAL_PIXEL_FORMAT_RGBA32` (maps to `Rgba8Unorm`)
    *`MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32` (maps to `Bgra8Unorm`)
    *`MGPU_EXTERNAL_PIXEL_FORMAT_NV12` (maps to `NV12` if feature supported, Y plane as `R8Unorm`, UV plane as `Rg8Unorm`)
    *`MGPU_EXTERNAL_PIXEL_FORMAT_GRAY8` (maps to `R8Unorm`)
    *`MGPU_EXTERNAL_PIXEL_FORMAT_RGBA64_HALF` (maps to `Rgba16Float`)
    *`MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_NV12_PLANES` (Indicates app will provide Y and interleaved UV planes suitable for `NV12` import)
    *`MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_RGB_PLANES` (Indicates app will provide Y, U, V planes to be treated as separate `R8Unorm` textures)
    **(Others as deemed necessary for common, efficient interop)*
* `MGPUExternalVideoInfo`, `MGPUExternalPlane`, `MGPUExternalGPUMetadata`, `MGPUExternalVideoBuffer`: As previously defined.

**Minigpu API Functions:**

* `MGPUSharedTextureMemory* mgpuCreateSharedTextureFromExternal(const MGPUExternalVideoBuffer* externalBuffer)`
* `int mgpuGetSharedTextureInfoFromExternal(const MGPUExternalVideoBuffer* externalBuffer, MGPUSharedTextureDescriptor* outDescriptor)`
* `MGPUHandleType mgpuMapExternalContentTypeToHandleType(MGPUExternalContentType contentType)`
* `uint32_t mgpuMapExternalPixelFormatToWGPU(MGPUExternalPixelFormat pixelFormat, uint32_t planeIndex)`: Now includes `planeIndex` for multi-planar formats.
* **New:** `int mgpuIsWGPUFormatSupported(uint32_t wgpuFormat, uint32_t usageFlags)`: Checks if a `WGPUTextureFormat` is supported for given usage.
* **New:** `int mgpuIsExternalPixelFormatSupported(MGPUExternalPixelFormat externalFormat)`: Checks if Minigpu has a reasonable mapping for this external format.

### 3.2. Feature Capability Checks in Minigpu

Minigpu must provide robust mechanisms for the application to query its capabilities regarding texture formats and import mechanisms.

**API Additions:**

* `int mgpuQuerySharedTextureSupport(MGPUHandleType handleType)`: Checks if importing a specific handle type is supported.
* `int mgpuQuerySharedFenceSupport(MGPUHandleType handleType)`: Checks if importing a specific fence handle type is supported.
* `int mgpuIsWGPUFormatSupported(uint32_t wgpuFormat, uint32_t usageFlags)`:
* Internally calls `wgpuDeviceHasFeature` for features tied to specific formats (e.g., `TEXTURE_FORMAT_NV12`, `TEXTURE_COMPRESSION_BC`, `TEXTURE_FORMAT_16BIT_NORM`, `DEPTH32FLOAT_STENCIL8`).
  * For formats not tied to a specific feature flag, it can attempt a "dry run" creation or rely on adapter limits if available, though direct support queries are preferred.
* `int mgpuIsExternalPixelFormatSupported(MGPUExternalPixelFormat externalFormat)`:
  * This function checks if `mgpuMapExternalPixelFormatToWGPU` can produce a valid `WGPUTextureFormat` for the given `externalFormat`.
  * It then calls `mgpuIsWGPUFormatSupported` for the target `WGPUTextureFormat(s)`. For multi-planar formats like `NV12`, it checks support for `R8Unorm` (for Y) and `Rg8Unorm` (for UV) and the `TEXTURE_FORMAT_NV12` feature itself.

**Internal Checks:**
When `mgpuCreateSharedTextureFromExternal` is called:

1. It first checks if the `MGPUHandleType` (derived from `MGPUExternalContentType`) is supported using `mgpuQuerySharedTextureSupport`.
2. It maps the `MGPUExternalPixelFormat` to one or more target `WGPUTextureFormat`s.
3. For each target `WGPUTextureFormat`, it calls `mgpuIsWGPUFormatSupported` with the intended usage (e.g., `WGPUTextureUsage_TextureBinding`).
4. If any check fails, the creation fails early with an informative error.

### 3.3. Format Negotiation Flow (Application Layer)

The application layer is responsible for negotiating a compatible format.

1. **Query MiniAV**: Application queries MiniAV for available capture formats for a given device/source (e.g., `MiniAV_Camera_GetSupportedFormats`). This returns a list of `MiniAVPixelFormat`s.
2. **Query Minigpu**: Application iterates through the formats MiniAV can provide:
    a.  For each `MiniAVPixelFormat`, the app determines a corresponding `MGPUExternalPixelFormat` it could map to.
    b.  Calls `mgpuIsExternalPixelFormatSupported(mappedExternalFormat)`.
    c.  If supported by Minigpu, this format is a candidate.
3. **Select Optimal Format**: From the list of candidate formats supported by both, the application selects the "best" one (e.g., preferring GPU native formats, formats requiring no conversion, or higher fidelity formats).
4. **Configure MiniAV**: Application configures MiniAV to deliver frames in the selected `MiniAVPixelFormat`.
5. **Mapping**: During frame processing, the application maps the `MiniAVBuffer` (now in the negotiated format) to `MGPUExternalVideoBuffer`, setting the corresponding `MGPUExternalPixelFormat`.

**Fallback Strategies:**

* **CPU Conversion**: If no direct GPU-compatible format can be agreed upon, the application might:
* Request MiniAV to provide frames in a common CPU format (e.g., `MINIAV_PIXEL_FORMAT_RGBA32`).
  * The application then maps this to `MGPU_EXTERNAL_CONTENT_TYPE_CPU`.
  * Minigpu will handle this by creating a standard `WGPUTexture` and the application will upload the data using `wgpuQueueWriteTexture`. This involves a copy but ensures compatibility.
* **Application-Side GPU Conversion**: If the application has its own GPU processing capabilities (e.g., a separate compute shader pass), it could take a widely available format from MiniAV, upload it to a GPU texture, perform a conversion shader, and then pass the converted texture handle to Minigpu. This is more complex.

### 3.4. Texture Format Mapping (Updated)

The mapping from `MGPUExternalPixelFormat` to `WGPUTextureFormat` needs to be robust.

| `MGPUExternalPixelFormat`                | Target `WGPUTextureFormat`(s)                                 | `WGPUFeatureName` Required                                          | Notes |
|------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------------|-------|
| `MGPU_EXTERNAL_PIXEL_FORMAT_RGBA32`      | `Rgba8Unorm`                                                  | None (universally supported)                                        | Best CPU-upload fallback |
| `MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32`      | `Bgra8Unorm`                                                  | None                                                                | Default Windows/macOS screen-capture format |
| `MGPU_EXTERNAL_PIXEL_FORMAT_NV12`        | plane 0 → `R8Unorm`, plane 1 → `Rg8Unorm`; multi-planar view `NV12` | `TextureFormatNV12` (Dawn extension)                         | Zero-copy camera on Windows/macOS/Linux |
| `MGPU_EXTERNAL_PIXEL_FORMAT_GRAY8`       | `R8Unorm`                                                     | None                                                                | Single-plane grayscale |
| `MGPU_EXTERNAL_PIXEL_FORMAT_RGBA64_HALF` | `Rgba16Float`                                                 | `Float32Filterable` (optional, for linear-filter sampling)          | HDR camera or post-process intermediate |
| `MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_NV12_PLANES` | plane 0 → `R8Unorm`, plane 1 → `Rg8Unorm`       | `TextureFormatNV12`                                                 | App interleaves U+V before handing to Minigpu |
| `MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_RGB_PLANES`  | planes 0,1,2 each → `R8Unorm`                   | None                                                                | I420/YV12; WGSL shader samples three `R8Unorm` textures |

---

## 4. Per-Platform GPU Handle Contracts

### 4.1 Windows — D3D11 NT Shared Handle

**MiniAV produces:** `MINIAV_BUFFER_CONTENT_TYPE_GPU_D3D11_HANDLE`  
`planes[0].data_ptr` is an `HANDLE` (NT shared handle, opened with `CreateSharedHandle` / `IDXGIResource1::CreateSharedHandle`).  
`planes[0].subresource_index` is the D3D11 subresource index (usually 0 unless from a texture array such as DXGI swap-chain back buffer).

**Dawn import path:**  
```cpp
WGPUSharedTextureMemoryDXGISharedHandleDescriptor desc{};
desc.chain.sType = WGPUSType_SharedTextureMemoryDXGISharedHandleDescriptor;
desc.handle = ntHandle;           // HANDLE from miniav
desc.useKeyedMutex = true;        // WGC / MediaFoundation always keyed-mutex
WGPUSharedTextureMemory mem = wgpuDeviceImportSharedTextureMemory(device, &desc.chain);
```

**Required Dawn feature:** `WGPUFeatureName_SharedTextureMemoryDXGISharedHandle`

**Sync:**
```cpp
// Begin access — acquires keyed-mutex key 0 with 0 ms timeout
WGPUSharedTextureMemoryBeginAccessDescriptor beginDesc{};
WGPUSharedTextureMemoryDXGIKeyedMutexAcquireReleaseDescriptor kmDesc{};
kmDesc.acquireCount = 1; kmDesc.acquireKeys = &key0; kmDesc.acquireTimeouts = &timeout0;
kmDesc.releaseCount = 1; kmDesc.releaseKeys = &key1;
beginDesc.chain = { .sType = WGPUSType_SharedTextureMemoryDXGIKeyedMutexAcquireReleaseDescriptor };
wgpuSharedTextureMemoryBeginAccess(mem, texture, &beginDesc);
// ... submit GPU work ...
wgpuSharedTextureMemoryEndAccess(mem, texture, &endDesc);
```

**Handle lifetime:** The NT handle must remain open until `wgpuSharedTextureMemoryRelease`. Call `CloseHandle` only after that. MiniAV's internal callback (`MiniAVNativeBufferInternalPayload`) owns the handle; the Minigpu interop layer must `DuplicateHandle` if it needs to outlive the MiniAV frame release.

---

### 4.2 macOS / iOS — IOSurface / MTLTexture

**MiniAV produces:** `MINIAV_BUFFER_CONTENT_TYPE_GPU_METAL_TEXTURE`  
`planes[i].data_ptr` is an `id<MTLTexture>` (ARC-retained, bridged to `void*` via `(__bridge void*)`).  
All planes of a CVPixelBuffer share the same underlying `IOSurface`; the canonical zero-copy path imports the `IOSurface`.

**Dawn import path:**
```cpp
// Retrieve IOSurface from CVPixelBuffer (or directly from MiniAV's internal payload)
IOSurfaceRef surface = CVPixelBufferGetIOSurface(cvPixelBuffer);
WGPUSharedTextureMemoryIOSurfaceDescriptor desc{};
desc.chain.sType = WGPUSType_SharedTextureMemoryIOSurfaceDescriptor;
desc.ioSurface = surface;
WGPUSharedTextureMemory mem = wgpuDeviceImportSharedTextureMemory(device, &desc.chain);
```

**Required Dawn feature:** `WGPUFeatureName_SharedTextureMemoryIOSurface`

**Sync:** Metal command-buffer completion signals are implicit across IOSurface consumers when using `MTLCommandBuffer.waitUntilCompleted` or via `MTLSharedEvent`. Dawn handles this internally through `WGPUSharedFenceMTLSharedEventDescriptor` on devices that expose `WGPUFeatureName_SharedFenceMTLSharedEvent`.

**Note:** MiniAV currently exposes the `id<MTLTexture>` pointer, not the IOSurface. Minigpu's import layer should call `[mtlTexture iosurface]` (ObjC) or the Swift equivalent to obtain the `IOSurface*`. This is always valid for textures backed by a `CVPixelBuffer`.

---

### 4.3 Linux — DMA-BUF

**MiniAV produces:** `MINIAV_BUFFER_CONTENT_TYPE_GPU_DMABUF_FD`  
`planes[i].data_ptr` holds the `int` fd cast to `void*`. `planes[i].offset_bytes` and `planes[i].stride_bytes` are valid.

**Missing today (will be added):** DRM format modifier (`uint64_t modifier`) is required by Dawn for tiled buffers. We will add `uint64_t drm_format_modifier` to `MiniAVVideoPlane` (or a companion `MiniAVDmaBufPlaneExtended` struct).

**Dawn import path:**
```cpp
WGPUSharedTextureMemoryDmaBufDescriptor desc{};
desc.chain.sType = WGPUSType_SharedTextureMemoryDmaBufDescriptor;
desc.planeCount = numPlanes;
// fill desc.planes[i].fd, .stride, .offset, .modifier per plane
WGPUSharedTextureMemory mem = wgpuDeviceImportSharedTextureMemory(device, &desc.chain);
```

**Required Dawn feature:** `WGPUFeatureName_SharedTextureMemoryDmaBuf`

**Sync:** `WGPUSharedFenceSyncFDDescriptor` carrying the implicit fence FD from PipeWire / Vulkan. If MiniAV doesn't yet expose the fence FD, a CPU-side `glFinish`-equivalent (`wgpuQueueOnSubmittedWorkDone`) is used as fallback.

---

### 4.4 Android — AHardwareBuffer

**MiniAV currently does NOT expose an AHardwareBuffer content type.** This will be added as part of this work.

**New MiniAV value (to be added):**
```c
MINIAV_BUFFER_CONTENT_TYPE_GPU_AHARDWAREBUFFER  // planes[0].data_ptr is AHardwareBuffer*
```

**Dawn import path:**
```cpp
WGPUSharedTextureMemoryAHardwareBufferDescriptor desc{};
desc.chain.sType = WGPUSType_SharedTextureMemoryAHardwareBufferDescriptor;
desc.handle = reinterpret_cast<AHardwareBuffer*>(planes[0].data_ptr);
WGPUSharedTextureMemory mem = wgpuDeviceImportSharedTextureMemory(device, &desc.chain);
```

**Required Dawn feature:** `WGPUFeatureName_SharedTextureMemoryAHardwareBuffer`

**Sync:** `WGPUSharedFenceSyncFDDescriptor` with the sync FD from `AHardwareBuffer_sendHandleToUnixSocket` / Camera2 API.

---

### 4.5 Web — GPUExternalTexture / WebCodecs

**MiniAV produces:** CPU `Uint8Array` (JS side). For getUserMedia / getDisplayMedia the browser provides a `VideoFrame` (WebCodecs).

**WebGPU path:**
```js
const externalTexture = device.importExternalTexture({ source: videoFrame });
// Valid for current task only — must re-import every frame.
```

**Dawn/emdawnwebgpu mapping:** The Emscripten build uses `wgpuDeviceImportExternalTexture` which maps directly to the browser's `GPUDevice.importExternalTexture`.

**Limitation:** `GPUExternalTexture` is read-only, cannot be used as a storage texture. For write-back, blit to a regular `WGPUTexture` via a render pass.

---

## 5. MiniAV API Extensions (Additive, ABI v2)

The following additions are backward-compatible (new enum values at end, new fields at end of structs).

### 5.1 New `MiniAVBufferContentType` value

```c
// miniav_buffer.h  — append to MiniAVBufferContentType enum
MINIAV_BUFFER_CONTENT_TYPE_GPU_AHARDWAREBUFFER, // planes[0].data_ptr is AHardwareBuffer*
```

### 5.2 DRM modifier on `MiniAVVideoPlane`

```c
// miniav_buffer.h  — append field to MiniAVVideoPlane
uint64_t drm_format_modifier;   // Linux DMA-BUF: DRM modifier (LINEAR=0, others per driver)
int      dmabuf_fd;             // Linux DMA-BUF: per-plane file descriptor (-1 if not DMA-BUF)
```

### 5.3 Native fence handle on `MiniAVBuffer`

```c
// miniav_buffer.h  — append to MiniAVBuffer struct
typedef struct {
    int      sync_fd;           // Linux/Android: sync_file fd (-1 if none)
    void    *d3d11_fence;       // Windows: ID3D11Fence* (NULL if none)
    void    *metal_shared_event;// macOS/iOS: id<MTLSharedEvent> (NULL if none)
} MiniAVNativeFence;

// Add to MiniAVBuffer:
MiniAVNativeFence native_fence; // Zero-initialized if no fence available
```

---

## 6. Minigpu External Texture API (`minigpu_external.h`)

```c
// minigpu_external.h

typedef enum {
    MGPU_EXTERNAL_CONTENT_TYPE_CPU = 0,
    MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE,
    MGPU_EXTERNAL_CONTENT_TYPE_METAL_IOSURFACE,
    MGPU_EXTERNAL_CONTENT_TYPE_DMABUF,
    MGPU_EXTERNAL_CONTENT_TYPE_AHARDWAREBUFFER,
    MGPU_EXTERNAL_CONTENT_TYPE_WEB_VIDEO_FRAME,   // JS-side only
} MGPUExternalContentType;

typedef enum {
    MGPU_EXTERNAL_PIXEL_FORMAT_UNKNOWN = 0,
    MGPU_EXTERNAL_PIXEL_FORMAT_RGBA32,
    MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32,
    MGPU_EXTERNAL_PIXEL_FORMAT_NV12,
    MGPU_EXTERNAL_PIXEL_FORMAT_GRAY8,
    MGPU_EXTERNAL_PIXEL_FORMAT_RGBA64_HALF,
    MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_NV12_PLANES,
    MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_RGB_PLANES,
} MGPUExternalPixelFormat;

typedef struct {
    void    *data_ptr;          // CPU: pixel data; GPU: handle/pointer (see content_type)
    uint32_t width;
    uint32_t height;
    uint32_t stride_bytes;
    uint32_t offset_bytes;
    uint32_t subresource_index;
    uint64_t drm_format_modifier; // Linux only
    int      dmabuf_fd;           // Linux only (-1 otherwise)
} MGPUExternalPlane;

typedef struct {
    int      sync_fd;
    void    *d3d11_fence;
    void    *metal_shared_event;
} MGPUExternalFence;

typedef struct {
    MGPUExternalContentType  content_type;
    MGPUExternalPixelFormat  pixel_format;
    uint32_t                 width;
    uint32_t                 height;
    uint32_t                 num_planes;
    MGPUExternalPlane        planes[4];
    MGPUExternalFence        fence;      // optional; zero = no explicit fence
    int64_t                  timestamp_us;
} MGPUExternalVideoBuffer;

typedef struct MGPUVideoTexture MGPUVideoTexture;

// --- Capability queries (call after mgpuInitializeContext) ---
EXPORT int  mgpuIsExternalPixelFormatSupported(MGPUExternalPixelFormat fmt);
EXPORT int  mgpuIsExternalContentTypeSupported(MGPUExternalContentType type);

// --- Import ---
// Returns an opaque texture wrapper. Caller must release with mgpuDestroyVideoTexture.
EXPORT MGPUVideoTexture* mgpuImportVideoFrame(const MGPUExternalVideoBuffer* buf);

// Bind plane [planeIndex] of the video texture to a compute shader binding slot.
EXPORT void mgpuSetVideoTexture(MGPUComputeShader* shader, int bindingSlot,
                                MGPUVideoTexture* tex, uint32_t planeIndex);

// Release. Ends GPU access and releases shared texture memory / external texture.
EXPORT void mgpuDestroyVideoTexture(MGPUVideoTexture* tex);

// --- YUV helpers ---
// Convert an NV12/I420 video texture to RGBA8 via an internal compute pass.
// Returns a new MGPUBuffer (RGBA8, row-major) that caller owns.
EXPORT MGPUBuffer* mgpuVideoTextureToRGBA(MGPUVideoTexture* tex);
```

---

## 7. NV12 / YUV Shader Sampling Strategy

### 7.1 Planar access (preferred for CV work)

The application binds plane textures individually and does the color conversion in their own WGSL shader:

```wgsl
@group(0) @binding(0) var y_plane  : texture_2d<f32>;  // R8Unorm
@group(0) @binding(1) var uv_plane : texture_2d<f32>;  // Rg8Unorm (NV12)
@group(0) @binding(2) var samp     : sampler;

fn nv12_to_rgb(uv: vec2f) -> vec3f {
    let y  = textureSample(y_plane, samp, uv).r;
    let cb = textureSample(uv_plane, samp, uv * 0.5).r - 0.5;
    let cr = textureSample(uv_plane, samp, uv * 0.5).g - 0.5;
    // BT.709 full-range
    let r = y + 1.5748 * cr;
    let g = y - 0.1873 * cb - 0.4681 * cr;
    let b = y + 1.8556 * cb;
    return clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0));
}
```

Minigpu will ship `yuv_helpers.wgsl` as a bundled asset that user shaders can `#include` (or the Dart layer can prepend at `loadKernel` time).

### 7.2 `mgpuVideoTextureToRGBA()` — internal compute pass

When the user calls this helper:
1. A `Rgba8Unorm` storage texture of `[width, height]` is allocated.
2. Minigpu dispatches an internal WGSL compute kernel (`nv12_to_rgba.wgsl` / `yuv420p_to_rgba.wgsl`) that reads the planar textures and writes RGBA8.
3. The result is wrapped in an `MGPUBuffer` (backed by a `WGPUBuffer`, `CopySrc|CopyDst`) and returned.
4. The intermediate storage texture is destroyed.

Color matrix is BT.709 by default; a future `mgpuVideoTextureToRGBAWithMatrix()` variant will accept a custom 3×4 matrix.

---

## 8. Object Lifecycle & Thread Safety

```
mgpuInitializeContext()
    └─ creates WGPUDevice / WGPUQueue on the WebGPUThread
mgpuImportVideoFrame(buf)
    └─ called from any thread; work is dispatched to WebGPUThread via enqueueSync
    └─ calls wgpuDeviceImportSharedTextureMemory (device-thread)
    └─ calls wgpuSharedTextureMemoryCreateTexture
    └─ calls wgpuSharedTextureMemoryBeginAccess  ← acquires fence / keyed-mutex
    └─ returns MGPUVideoTexture*
mgpuSetVideoTexture(shader, slot, tex, plane)
    └─ binds the WGPUTexture view to the shader's bind group
mgpuDispatch / mgpuDispatchAsync
    └─ GPU work reads the texture
mgpuDestroyVideoTexture(tex)
    └─ calls wgpuSharedTextureMemoryEndAccess  ← releases fence / keyed-mutex
    └─ calls wgpuTextureRelease
    └─ calls wgpuSharedTextureMemoryRelease
    └─ (Windows) CloseHandle(duplicate NT handle)
    └─ (Linux) close(dup dmabuf fd)
```

**Rule:** `mgpuDestroyVideoTexture` must be called **after** any dispatch that uses the texture has completed (i.e., after the callback from `mgpuDispatchAsync`, or after `mgpuDispatch` returns).

MiniAV's frame callback remains blocked (or the `MiniAVBuffer` is retained via `MiniAV_RetainBuffer`) until the caller releases the imported texture.

---

## 9. Implementation Checklist

| # | Location | Work item |
|---|----------|-----------|
| 1 | `miniav_c/include/miniav_buffer.h` | Add `MINIAV_BUFFER_CONTENT_TYPE_GPU_AHARDWAREBUFFER`, `drm_format_modifier`, `dmabuf_fd` on `MiniAVVideoPlane`, `MiniAVNativeFence` on `MiniAVBuffer` |
| 2 | `miniav_ffi/lib/miniav_ffi_types.dart` | Regenerate bindings; expose `nativeFence`, `drmModifier` in Dart types |
| 3 | `minigpu_ffi/src/include/minigpu_external.h` | New header — full C API from §6 |
| 4 | `minigpu_ffi/src/src/minigpu_external.cpp` | Platform-dispatched implementation: D3D11 (Win), IOSurface (Mac/iOS), DmaBuf (Linux), AHB (Android), CPU fallback (all), Web stub |
| 5 | `minigpu_ffi/src/include/minigpu.h` | `#include "minigpu_external.h"` |
| 6 | `minigpu_ffi/src/CMakeLists.txt` | Add `minigpu_external.cpp` to sources; link D3D11/DXGI on Win, IOSurface on Apple, no extra libs on Linux/Android |
| 7 | `minigpu_ffi/lib/minigpu_ffi.dart` | Dart FFI bindings for new symbols |
| 8 | `minigpu_ffi/src/src/` (WGSL assets) | `yuv_helpers.wgsl`, `nv12_to_rgba.wgsl`, `yuv420p_to_rgba.wgsl` |
| 9 | `minigpu_ffi/src/test/` | `external_texture_test.cpp` — gtest with synthetic D3D11 texture import (Windows-only CI gate) |
| 10 | `minigpu_av/test/` | Dart integration test: camera → `mgpuImportVideoFrame` → passthrough shader → `mgpuVideoTextureToRGBA` → compare first pixel |
