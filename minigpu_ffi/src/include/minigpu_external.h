#ifndef MINIGPU_EXTERNAL_H
#define MINIGPU_EXTERNAL_H

/**
 * minigpu_external.h
 *
 * Zero-copy / low-copy import of external video frames into Minigpu WebGPU
 * textures. Supports platform-native GPU handles (D3D11 shared handle, Metal
 * IOSurface, Linux DMA-BUF, Android AHardwareBuffer) as well as CPU fallback
 * for all platforms, and WebCodecs VideoFrame on the Web.
 *
 * Typical usage:
 *   1. Call mgpuIsExternalContentTypeSupported() / mgpuIsExternalPixelFormatSupported()
 *      to confirm the GPU supports the desired import path.
 *   2. Fill an MGPUExternalVideoBuffer from a MiniAVBuffer (application layer).
 *   3. Call mgpuImportVideoFrame() to obtain an MGPUVideoTexture*.
 *   4. Bind planes with mgpuSetVideoTexture() and dispatch your compute shader.
 *   5. Call mgpuDestroyVideoTexture() after the GPU work is complete.
 */

#include "export.h"
#include <stdint.h>
#include <stddef.h>

/* Forward declarations — avoids circular dependency with minigpu.h */
#ifdef __cplusplus
namespace mgpu { class ComputeShader; }
struct MGPUComputeShader;
struct MGPUBuffer;
#else
typedef struct MGPUComputeShader MGPUComputeShader;
typedef struct MGPUBuffer MGPUBuffer;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Enums
 * ---------------------------------------------------------------------- */

/** Describes the origin / handle type of the external buffer. */
typedef enum {
    MGPU_EXTERNAL_CONTENT_TYPE_CPU = 0,
    /**< CPU-accessible pixel data. Uploaded via wgpuQueueWriteTexture.
         All platforms. */
    MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE,
    /**< Windows: planes[0].data_ptr is a Windows NT HANDLE obtained from
         IDXGIResource1::CreateSharedHandle. Must be opened with
         GENERIC_ALL access. Minigpu will DuplicateHandle internally. */
    MGPU_EXTERNAL_CONTENT_TYPE_METAL_IOSURFACE,
    /**< macOS/iOS: planes[0].data_ptr is an IOSurfaceRef (retained).
         Minigpu imports via wgpuDeviceImportSharedTextureMemory with
         WGPUSType_SharedTextureMemoryIOSurfaceDescriptor. */
    MGPU_EXTERNAL_CONTENT_TYPE_DMABUF,
    /**< Linux: each plane has a dmabuf_fd, stride_bytes, offset_bytes, and
         drm_format_modifier. Minigpu imports via
         WGPUSType_SharedTextureMemoryDmaBufDescriptor. */
    MGPU_EXTERNAL_CONTENT_TYPE_AHARDWAREBUFFER,
    /**< Android: planes[0].data_ptr is an AHardwareBuffer* (reference held
         by caller until mgpuDestroyVideoTexture). */
    MGPU_EXTERNAL_CONTENT_TYPE_WEB_VIDEO_FRAME,
    /**< Web only (Emscripten build): planes[0].data_ptr is a JS VideoFrame
         object handle. Imported via device.importExternalTexture(). */
} MGPUExternalContentType;

/** Pixel format as seen by the external producer.
 *  Subset of MiniAVPixelFormat that Minigpu can consume efficiently. */
typedef enum {
    MGPU_EXTERNAL_PIXEL_FORMAT_UNKNOWN = 0,
    MGPU_EXTERNAL_PIXEL_FORMAT_RGBA32,          /**< Rgba8Unorm  */
    MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32,          /**< Bgra8Unorm  */
    MGPU_EXTERNAL_PIXEL_FORMAT_NV12,            /**< R8Unorm (Y) + Rg8Unorm (UV) */
    MGPU_EXTERNAL_PIXEL_FORMAT_GRAY8,           /**< R8Unorm     */
    MGPU_EXTERNAL_PIXEL_FORMAT_RGBA64_HALF,     /**< Rgba16Float */
    MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_NV12_PLANES,
    /**< I420 with app-interleaved UV → same import path as NV12 */
    MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_RGB_PLANES,
    /**< I420/YV12 as three separate R8Unorm textures (Y, U, V) */
} MGPUExternalPixelFormat;

/* -------------------------------------------------------------------------
 * Structures
 * ---------------------------------------------------------------------- */

/** Per-plane descriptor for an external video buffer. */
typedef struct {
    void    *data_ptr;          /**< CPU: pixel data ptr.
                                     D3D11: NT HANDLE (cast to void*).
                                     Metal: IOSurfaceRef or id<MTLTexture>.
                                     DMA-BUF: legacy cast of dmabuf_fd.
                                     AHB: AHardwareBuffer*.
                                     Web: unused (0). */
    uint32_t width;             /**< Plane width in pixels. */
    uint32_t height;            /**< Plane height in pixels. */
    uint32_t stride_bytes;      /**< Row stride in bytes. */
    uint32_t offset_bytes;      /**< Byte offset within the shared resource. */
    uint32_t subresource_index; /**< D3D11 subresource / Vulkan aspect index. */
    /* Linux DMA-BUF specific (ignored on other platforms) */
    int      dmabuf_fd;         /**< Per-plane DMA-BUF fd (-1 if not DMA-BUF). */
    uint64_t drm_format_modifier; /**< DRM modifier (0 = DRM_FORMAT_MOD_LINEAR). */
} MGPUExternalPlane;

/** Optional GPU synchronisation fence.
 *  Zero-initialise if the producer has already waited (e.g. CPU copy). */
typedef struct {
    int      sync_fd;             /**< Linux/Android: sync_file fd. -1 = none. */
    void    *d3d11_fence;         /**< Windows: ID3D11Fence*. NULL = none.     */
    void    *metal_shared_event;  /**< macOS/iOS: id<MTLSharedEvent> (bridged).
                                       NULL = none.                             */
    uint64_t metal_fence_value;   /**< Signal value for metal_shared_event.    */
} MGPUExternalFence;

/** Full descriptor of an external video frame to be imported into Minigpu. */
typedef struct {
    MGPUExternalContentType content_type;   /**< Handle / memory type.        */
    MGPUExternalPixelFormat pixel_format;   /**< Pixel format of the frame.   */
    uint32_t                width;          /**< Total frame width in pixels.  */
    uint32_t                height;         /**< Total frame height in pixels. */
    uint32_t                num_planes;     /**< Number of valid entries in planes[]. */
    MGPUExternalPlane       planes[4];      /**< Per-plane descriptors.        */
    MGPUExternalFence       fence;          /**< Optional GPU fence.           */
    int64_t                 timestamp_us;   /**< Frame timestamp (microseconds). */
} MGPUExternalVideoBuffer;

/* -------------------------------------------------------------------------
 * Opaque handle for an imported video texture
 * ---------------------------------------------------------------------- */
typedef struct MGPUVideoTexture MGPUVideoTexture;

/* -------------------------------------------------------------------------
 * Capability queries (call after mgpuInitializeContext)
 * ---------------------------------------------------------------------- */

/**
 * Returns 1 if the current WebGPU device can import the given content type
 * (i.e. the required Dawn SharedTextureMemory feature is present), 0 otherwise.
 */
EXPORT int mgpuIsExternalContentTypeSupported(MGPUExternalContentType type);

/**
 * Returns 1 if Minigpu has a valid WGPUTextureFormat mapping for the given
 * pixel format AND the necessary device features are available, 0 otherwise.
 */
EXPORT int mgpuIsExternalPixelFormatSupported(MGPUExternalPixelFormat fmt);

/* -------------------------------------------------------------------------
 * Import / export
 * ---------------------------------------------------------------------- */

/**
 * Import an external video frame as a Minigpu video texture.
 *
 * - For GPU content types: imports via Dawn SharedTextureMemory / importExternalTexture.
 *   Begins GPU access (acquires keyed-mutex / fence).
 * - For CPU: allocates a WGPUTexture and uploads with wgpuQueueWriteTexture.
 *
 * Returns NULL on failure (check log output for reason).
 * Caller must call mgpuDestroyVideoTexture() after GPU work is done.
 */
EXPORT MGPUVideoTexture* mgpuImportVideoFrame(const MGPUExternalVideoBuffer* buf);

/**
 * Bind plane [plane_index] of a video texture to a compute shader binding slot.
 *
 * For packed formats (RGBA32, BGRA32, GRAY8, RGBA64_HALF): use plane_index 0.
 * For NV12: plane_index 0 = Y (R8Unorm), plane_index 1 = UV (Rg8Unorm).
 * For YUV420P_AS_RGB_PLANES: plane_index 0/1/2 = Y/U/V (each R8Unorm).
 *
 * The binding corresponds to the @binding attribute in your WGSL shader.
 */
EXPORT void mgpuSetVideoTexture(MGPUComputeShader* shader, int binding_slot,
                                MGPUVideoTexture* tex, uint32_t plane_index);

/**
 * Release a video texture.
 *
 * For SharedTextureMemory imports: ends GPU access (releases keyed-mutex /
 * fence), releases the WGPUTexture and WGPUSharedTextureMemory.
 * For CPU uploads: releases the WGPUTexture.
 *
 * MUST be called AFTER any dispatch that reads the texture has completed.
 * (i.e. after mgpuDispatch returns, or inside the mgpuDispatchAsync callback.)
 */
EXPORT void mgpuDestroyVideoTexture(MGPUVideoTexture* tex);

/* -------------------------------------------------------------------------
 * YUV conversion helper
 * ---------------------------------------------------------------------- */

/**
 * Convert a video texture (NV12 or YUV420P variants) to an RGBA8 MGPUBuffer
 * via an internal compute pass (BT.709 full-range color matrix).
 *
 * For packed formats (RGBA32/BGRA32/GRAY8) this is a simple copy/expand.
 *
 * Returns a new MGPUBuffer* (caller owns; destroy with mgpuDestroyBuffer).
 * Returns NULL on failure.
 *
 * Note: this call is synchronous — it blocks until the GPU work is done.
 */
EXPORT MGPUBuffer* mgpuVideoTextureToRGBA(MGPUVideoTexture* tex);

/* -------------------------------------------------------------------------
 * Cross-API shared output texture (Windows D3D12 <-> D3D11 zero-copy)
 * ----------------------------------------------------------------------
 *
 * Allows minigpu to write the result of a compute pass directly into a
 * GPU texture that another D3D11 client (e.g. the FFmpeg D3D11VA hardware
 * encoder) can consume without a CPU round-trip.
 *
 * Implementation:
 *   - Allocates an ID3D12Resource (RGBA8, UAV-capable, SHARED) on Dawn's
 *     own D3D12 device (obtained via dawn::native::d3d12::GetD3D12Device).
 *   - Creates a Windows NT HANDLE for the resource via
 *     ID3D12Device::CreateSharedHandle (consumed by the D3D11 client via
 *     ID3D11Device1::OpenSharedResource1).
 *   - Imports the same D3D12 resource into Dawn as a WGPUSharedTextureMemory
 *     using the WGPUSType_SharedTextureMemoryD3D12ResourceDescriptor chained
 *     struct.  No NT-handle round-trip is needed on the Dawn side.
 *
 * Synchronisation:
 *   mgpuVideoTextureBGRAToRGBASharedOutput() submits the compute pass and
 *   blocks (CPU-side) until the GPU work has completed.  The D3D11 client
 *   may then access the texture freely.  Future versions will expose Dawn
 *   SharedFences for true GPU-side cross-API synchronisation.
 *
 * Platform: Windows only.  Returns NULL / 0 on other platforms.
 */

typedef struct MGPUSharedOutputTexture MGPUSharedOutputTexture;

/**
 * Create a shared RGBA8 output texture of the given dimensions.
 *
 * Returns NULL on non-Windows platforms or if the GPU does not support
 * the SharedTextureMemoryD3D12Resource feature, or on allocation failure.
 *
 * Caller must call mgpuDestroySharedOutputTexture().
 */
EXPORT MGPUSharedOutputTexture*
mgpuCreateSharedOutputTexture(uint32_t width, uint32_t height);

/**
 * Returns the Windows NT HANDLE that the D3D11 client should open via
 * ID3D11Device1::OpenSharedResource1(handle, IID_ID3D11Texture2D, ...).
 *
 * The handle remains owned by the MGPUSharedOutputTexture; the D3D11 client
 * must not CloseHandle() it.  (Open the resulting ID3D11Texture2D and let
 * the COM ref-count manage that view's lifetime.)
 *
 * Returns NULL on non-Windows platforms or for an invalid texture.
 */
EXPORT void* mgpuSharedOutputTextureGetD3D11Handle(MGPUSharedOutputTexture* tex);

/**
 * Returns the underlying ID3D11Texture2D* for the shared output texture.
 *
 * The texture lives on the same ID3D11Device returned by
 * mgpuCreateD3D11DeviceOnDawnAdapter(), so an external D3D11 client (e.g.
 * an FFmpeg encoder) can use this pointer directly without
 * OpenSharedResource1. Caller must NOT Release this pointer — its
 * lifetime is owned by the MGPUSharedOutputTexture.
 *
 * Returns NULL on non-Windows platforms or for an invalid texture.
 */
EXPORT void* mgpuSharedOutputTextureGetD3D11Texture(MGPUSharedOutputTexture* tex);

/**
 * Returns an AddRef'd ID3D11Device* on the SAME DXGI adapter as Dawn's
 * D3D12 device. The same cached device is used internally to back any
 * MGPUSharedOutputTexture we create. Caller is responsible for Release()
 * (or handing ownership to FFmpeg via av_hwdevice_ctx_init).
 *
 * Returns NULL on non-Windows platforms.
 */
EXPORT void* mgpuCreateD3D11DeviceOnDawnAdapter(void);

EXPORT uint32_t mgpuSharedOutputTextureGetWidth(MGPUSharedOutputTexture* tex);
EXPORT uint32_t mgpuSharedOutputTextureGetHeight(MGPUSharedOutputTexture* tex);

/**
 * Run a BGRA->RGBA swizzle (or pure RGBA passthrough) compute pass that
 * reads from [src] (typically an imported miniav D3D11 texture) and writes
 * the result into [dst].
 *
 * Blocks until the GPU work has completed; the D3D11 view of [dst] is then
 * safe to read.
 *
 * Returns 1 on success, 0 on failure.
 */
EXPORT int mgpuVideoTextureBGRAToRGBASharedOutput(MGPUVideoTexture* src,
                                                  MGPUSharedOutputTexture* dst);

/**
 * Copy an RGBA8 GPU storage buffer (the output of a GpuEffect / compute
 * dispatch) into the shared output texture entirely on the GPU.  The
 * buffer must hold exactly width*height u32 pixels packed as RGBA8.
 *
 * Returns 1 on success, 0 on failure.
 */
EXPORT int mgpuCopyBufferToSharedOutputTexture(MGPUBuffer* buf,
                                               MGPUSharedOutputTexture* dst);

/**
 * Like mgpuCopyBufferToSharedOutputTexture but reads the source as
 * `array<f32>` with 4 floats per pixel (R,G,B,A in [0,1]).  Used by
 * visualizers (e.g. the spectrogram) that produce float colors directly.
 *
 * Returns 1 on success, 0 on failure.
 */
EXPORT int mgpuCopyBufferF32ToSharedOutputTexture(MGPUBuffer* buf,
                                                  MGPUSharedOutputTexture* dst);

/**
 * Release all resources held by the shared output texture (WGPUTexture,
 * WGPUSharedTextureMemory, ID3D12Resource COM ref, NT HANDLE).
 *
 * Must NOT be called while the D3D11 client still holds an
 * ID3D11Texture2D view of the underlying resource.
 */
EXPORT void mgpuDestroySharedOutputTexture(MGPUSharedOutputTexture* tex);

/**
 * Debug-only: read the first pixel (BGRA8, packed as 0xAARRGGBB after Map)
 * of the shared output texture from the D3D11 side using the cached
 * Dawn-adapter D3D11 device. Useful to verify that Dawn's writes are
 * visible on the D3D11 consumer.
 *
 * Returns the raw 32-bit pixel value, or 0xDEAD000N on failure (N is the
 * stage that failed, see implementation).
 */
EXPORT uint32_t mgpuSharedOutputTextureDebugReadFirstPixel(
        MGPUSharedOutputTexture* tex);

/**
 * Debug-only: read the first pixel from the Dawn (D3D12) side via
 * CopyTextureToBuffer + map.  Used to compare with the D3D11-side
 * readback above to determine which side sees stale data.
 *
 * Returns the raw 32-bit pixel value, or 0xDEAD100N on failure (N is the
 * stage that failed, see implementation).
 */
EXPORT uint32_t mgpuSharedOutputTextureDebugReadFirstPixelDawn(
        MGPUSharedOutputTexture* tex);

#ifdef __cplusplus
}
#endif

#endif /* MINIGPU_EXTERNAL_H */
