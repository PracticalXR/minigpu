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

| `MGPUExternalPixelFormat`                | Target `WGPUTextureFormat`(s)                                 | `WGPUFeatureName` Required (Examples)                               | Notes 