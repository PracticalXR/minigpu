/**
 * minigpu_external.cpp
 *
 * Platform-dispatched implementation of the MGPUVideoTexture import API.
 * Handles D3D11 shared handle (Windows), IOSurface (macOS/iOS), DMA-BUF
 * (Linux), AHardwareBuffer (Android), CPU upload (all), and Web
 * importExternalTexture (Emscripten).
 */

#include "../include/minigpu_external.h"
#include "../include/buffer.h"
#include "../include/compute_shader.h"
#include "../include/log.h"

// webgpu.h is pulled in through buffer.h / minigpu.h
#include "webgpu.h"

#include <future>

// Dawn-specific shared texture / fence headers (not available in Emscripten)
#ifndef __EMSCRIPTEN__
#include "dawn/native/DawnNative.h"
#endif

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <d3d11.h>
#  include <d3d11_1.h>
#  include <d3d11_4.h>
#  include <d3d11on12.h>
#  include <d3d12.h>
#  include <dxgi1_4.h>
#  include <wrl/client.h>
#  include "dawn/native/D3D12Backend.h"
#  include "dawn/native/D3D11Backend.h"
#  pragma comment(lib, "d3d11.lib")
#  pragma comment(lib, "d3d12.lib")
#  pragma comment(lib, "dxgi.lib")
#endif

#ifdef __APPLE__
#  include <TargetConditionals.h>
#  include <IOSurface/IOSurfaceRef.h>
#endif

#ifdef __ANDROID__
#  include <android/hardware_buffer.h>
#endif

#include <cstring>
#include <cassert>
#include <cstdio>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <memory>

// ---- the global MGPU instance is defined in minigpu.cpp ----
// It is declared inside extern "C" { } in minigpu.cpp, so use C linkage here.
extern "C" mgpu::MGPU minigpu;

/* =========================================================================
 * Hang-tolerant Dawn event drain helpers
 *
 * The old implementation spun in a tight 0 ms-poll loop calling
 * wgpuInstanceProcessEvents until a std::promise was set.  Two problems:
 *
 *   1. It pegged a CPU core for the entire duration of every dispatch
 *      (~milliseconds × 30 Hz × multiple textures), starving the Dart
 *      isolate that owns the capture/audio callbacks and producing the
 *      stalls / crackly audio reported in the field.
 *   2. If the underlying GPU driver hung (e.g. NVIDIA TDR, Dawn device
 *      lost), the callback would never fire and the whole process would
 *      deadlock forever.
 *
 * drain_dawn_events_with_timeout() yields 1 ms between polls and bails
 * out with a logged warning after a configurable deadline so a hung GPU
 * surfaces as a recoverable encode error instead of a frozen app.
 * pump_dawn_events_nonblocking() is for the async encode path where we
 * just need to give Dawn a chance to fire pending callbacks without
 * actually waiting for anything.
 * ====================================================================== */

template <typename FutureT>
static bool drain_dawn_events_with_timeout(
        FutureT&             fut,
        const char*          op,
        int                  timeout_ms = 5000) {
    using namespace std::chrono;
    const auto deadline = steady_clock::now() + milliseconds(timeout_ms);
    while (fut.wait_for(milliseconds(1)) != std::future_status::ready) {
        wgpuInstanceProcessEvents(minigpu.getInstance());
        if (steady_clock::now() >= deadline) {
            LOG_ERROR("[minigpu_external] %s: GPU work did not complete "
                      "within %d ms; treating as driver hang and bailing out.",
                      op, timeout_ms);
            return false;
        }
    }
    // One final drain so callbacks chained after our promise also fire.
    wgpuInstanceProcessEvents(minigpu.getInstance());
    return true;
}

static inline void pump_dawn_events_nonblocking() {
    wgpuInstanceProcessEvents(minigpu.getInstance());
}

/* =========================================================================
 * Internal structure
 * ====================================================================== */

struct MGPUVideoTexture {
    // One WGPUTexture per plane (multi-planar formats have > 1).
    // For SharedTextureMemory imports, plane[i] comes from the same
    // WGPUSharedTextureMemory object; for CPU and Web only planes[0] is used.
    static constexpr int kMaxPlanes = 4;

    WGPUTexture             planes[kMaxPlanes]     = {};
    WGPUTextureView         views[kMaxPlanes]      = {};
    int                     num_planes             = 0;

    // SharedTextureMemory (NULL for CPU / Web paths; not available on Emscripten)
#ifndef __EMSCRIPTEN__
    WGPUSharedTextureMemory shared_mem             = nullptr;
#endif

    // Dimensions
    uint32_t width  = 0;
    uint32_t height = 0;

    MGPUExternalPixelFormat pixel_format = MGPU_EXTERNAL_PIXEL_FORMAT_UNKNOWN;
    MGPUExternalContentType content_type = MGPU_EXTERNAL_CONTENT_TYPE_CPU;
};

/* =========================================================================
 * Helpers
 * ====================================================================== */

static WGPUDevice get_device() {
    return minigpu.getDevice();
}

static WGPUQueue get_queue() {
    return minigpu.getQueue();
}

// Map a single-plane external pixel format to WGPUTextureFormat.
// For multi-planar formats pass planeIndex.
static WGPUTextureFormat map_format(MGPUExternalPixelFormat fmt, uint32_t plane) {
    switch (fmt) {
    case MGPU_EXTERNAL_PIXEL_FORMAT_RGBA32:         return WGPUTextureFormat_RGBA8Unorm;
    case MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32:         return WGPUTextureFormat_BGRA8Unorm;
    case MGPU_EXTERNAL_PIXEL_FORMAT_GRAY8:          return WGPUTextureFormat_R8Unorm;
    case MGPU_EXTERNAL_PIXEL_FORMAT_RGBA64_HALF:    return WGPUTextureFormat_RGBA16Float;
    case MGPU_EXTERNAL_PIXEL_FORMAT_NV12:
    case MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_NV12_PLANES:
        return (plane == 0) ? WGPUTextureFormat_R8Unorm : WGPUTextureFormat_RG8Unorm;
    case MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_RGB_PLANES:
        return WGPUTextureFormat_R8Unorm;
    default:
        return WGPUTextureFormat_Undefined;
    }
}

static int num_planes_for_format(MGPUExternalPixelFormat fmt) {
    switch (fmt) {
    case MGPU_EXTERNAL_PIXEL_FORMAT_NV12:
    case MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_NV12_PLANES:
        return 2;
    case MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_RGB_PLANES:
        return 3;
    default:
        return 1;
    }
}

/* =========================================================================
 * Capability queries
 * ====================================================================== */

extern "C" {

int mgpuIsExternalContentTypeSupported(MGPUExternalContentType type) {
#ifdef __EMSCRIPTEN__
    return (type == MGPU_EXTERNAL_CONTENT_TYPE_CPU ||
            type == MGPU_EXTERNAL_CONTENT_TYPE_WEB_VIDEO_FRAME) ? 1 : 0;
#else
    WGPUDevice device = get_device();
    if (!device) return 0;

    WGPUFeatureName feat = (WGPUFeatureName)0;
    switch (type) {
    case MGPU_EXTERNAL_CONTENT_TYPE_CPU:
        return 1; // Always supported
    case MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE:
#ifdef _WIN32
        feat = WGPUFeatureName_SharedTextureMemoryDXGISharedHandle;
#else
        return 0;
#endif
        break;
    case MGPU_EXTERNAL_CONTENT_TYPE_METAL_IOSURFACE:
#ifdef __APPLE__
        feat = WGPUFeatureName_SharedTextureMemoryIOSurface;
#else
        return 0;
#endif
        break;
    case MGPU_EXTERNAL_CONTENT_TYPE_DMABUF:
#if defined(__linux__) && !defined(__ANDROID__)
        feat = WGPUFeatureName_SharedTextureMemoryDmaBuf;
#else
        return 0;
#endif
        break;
    case MGPU_EXTERNAL_CONTENT_TYPE_AHARDWAREBUFFER:
#ifdef __ANDROID__
        feat = WGPUFeatureName_SharedTextureMemoryAHardwareBuffer;
#else
        return 0;
#endif
        break;
    default:
        return 0;
    }
    return wgpuDeviceHasFeature(device, feat) ? 1 : 0;
#endif // __EMSCRIPTEN__
}

int mgpuIsExternalPixelFormatSupported(MGPUExternalPixelFormat fmt) {
    // For every plane check that the corresponding WGPUTextureFormat is usable.
    int n = num_planes_for_format(fmt);
    WGPUDevice device = get_device();
    if (!device) return 0;

    for (int i = 0; i < n; ++i) {
        WGPUTextureFormat wfmt = map_format(fmt, (uint32_t)i);
        if (wfmt == WGPUTextureFormat_Undefined) return 0;
        // Validate by attempting to create a 1x1 texture and immediately
        // destroying it (cheapest probe available without an explicit
        // wgpuDeviceIsTextureFormatSupported() call in base WebGPU).
        WGPUTextureDescriptor td{};
        td.usage  = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        td.dimension = WGPUTextureDimension_2D;
        td.size   = {1, 1, 1};
        td.format = wfmt;
        td.mipLevelCount   = 1;
        td.sampleCount     = 1;
        WGPUTexture probe = wgpuDeviceCreateTexture(device, &td);
        if (!probe) return 0;
        wgpuTextureRelease(probe);
    }
    return 1;
}

/* =========================================================================
 * CPU upload path (all platforms)
 * ====================================================================== */

static MGPUVideoTexture* import_cpu(const MGPUExternalVideoBuffer* buf) {
    WGPUDevice device = get_device();
    WGPUQueue  queue  = get_queue();
    if (!device || !queue) return nullptr;

    int n = num_planes_for_format(buf->pixel_format);
    auto* tex = new MGPUVideoTexture();
    tex->width        = buf->width;
    tex->height       = buf->height;
    tex->num_planes   = n;
    tex->pixel_format = buf->pixel_format;
    tex->content_type = MGPU_EXTERNAL_CONTENT_TYPE_CPU;

    for (int i = 0; i < n; ++i) {
        const MGPUExternalPlane& p = buf->planes[i];
        WGPUTextureFormat wfmt = map_format(buf->pixel_format, (uint32_t)i);

        WGPUTextureDescriptor td{};
        td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        td.dimension       = WGPUTextureDimension_2D;
        td.size            = {p.width, p.height, 1};
        td.format          = wfmt;
        td.mipLevelCount   = 1;
        td.sampleCount     = 1;
        tex->planes[i] = wgpuDeviceCreateTexture(device, &td);
        if (!tex->planes[i]) {
            // Cleanup already created textures
            for (int j = 0; j < i; ++j) wgpuTextureRelease(tex->planes[j]);
            delete tex;
            return nullptr;
        }

        // Upload pixel data
        if (p.data_ptr && p.stride_bytes > 0) {
            WGPUTexelCopyTextureInfo dst{};
            dst.texture  = tex->planes[i];
            dst.mipLevel = 0;
            dst.aspect   = WGPUTextureAspect_All;

            WGPUTexelCopyBufferLayout layout{};
            layout.offset       = 0;
            layout.bytesPerRow  = p.stride_bytes;
            layout.rowsPerImage = p.height;

            WGPUExtent3D extent = {p.width, p.height, 1};
            size_t data_size = (size_t)p.stride_bytes * p.height;
            wgpuQueueWriteTexture(queue, &dst, p.data_ptr, data_size,
                                  &layout, &extent);
        }

        // Create a default view for convenience
        WGPUTextureViewDescriptor vd{};
        vd.format          = wfmt;
        vd.dimension       = WGPUTextureViewDimension_2D;
        vd.baseMipLevel    = 0;
        vd.mipLevelCount   = 1;
        vd.baseArrayLayer  = 0;
        vd.arrayLayerCount = 1;
        vd.aspect          = WGPUTextureAspect_All;
        tex->views[i] = wgpuTextureCreateView(tex->planes[i], &vd);
    }
    return tex;
}

/* =========================================================================
 * SharedTextureMemory import helper (native platforms)
 * ====================================================================== */

#ifndef __EMSCRIPTEN__
// Populate tex->num_planes + tex->views[] from a single imported Dawn
// `texture`. NV12 is multi-planar: a shader can't bind an all-aspect view of
// it, so we create one view per plane (Y = R8 via Plane0, UV = RG8 via Plane1)
// — matching what mgpuVideoTextureToRGBA / setOnShader expect (views[0]=Y,
// views[1]=UV). Every other format is a single all-aspect view of
// [fallbackFormat]. Shared across all import tiers (shared-handle, Tier A
// same-adapter, Tier B cross-adapter) so they agree on plane layout.
static void build_video_texture_views(MGPUVideoTexture* tex,
                                      WGPUTexture texture,
                                      MGPUExternalPixelFormat fmt,
                                      WGPUTextureFormat fallbackFormat) {
    const bool isNv12 =
        (fmt == MGPU_EXTERNAL_PIXEL_FORMAT_NV12 ||
         fmt == MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_NV12_PLANES);
    if (isNv12) {
        tex->num_planes = 2;
        WGPUTextureViewDescriptor yv{};
        yv.format          = WGPUTextureFormat_R8Unorm;
        yv.dimension       = WGPUTextureViewDimension_2D;
        yv.baseMipLevel    = 0; yv.mipLevelCount   = 1;
        yv.baseArrayLayer  = 0; yv.arrayLayerCount = 1;
        yv.aspect          = WGPUTextureAspect_Plane0Only;
        tex->views[0] = wgpuTextureCreateView(texture, &yv);

        WGPUTextureViewDescriptor uvv{};
        uvv.format          = WGPUTextureFormat_RG8Unorm;
        uvv.dimension       = WGPUTextureViewDimension_2D;
        uvv.baseMipLevel    = 0; uvv.mipLevelCount   = 1;
        uvv.baseArrayLayer  = 0; uvv.arrayLayerCount = 1;
        uvv.aspect          = WGPUTextureAspect_Plane1Only;
        tex->views[1] = wgpuTextureCreateView(texture, &uvv);
    } else {
        tex->num_planes = 1;
        WGPUTextureViewDescriptor vd{};
        vd.format          = fallbackFormat;
        vd.dimension       = WGPUTextureViewDimension_2D;
        vd.baseMipLevel    = 0; vd.mipLevelCount   = 1;
        vd.baseArrayLayer  = 0; vd.arrayLayerCount = 1;
        vd.aspect          = WGPUTextureAspect_All;
        tex->views[0] = wgpuTextureCreateView(texture, &vd);
    }
}

static MGPUVideoTexture* import_shared(WGPUSharedTextureMemory mem,
                                       const MGPUExternalVideoBuffer* buf) {
    if (!mem) return nullptr;

    // Query the descriptor so we know dimensions / format.
    WGPUSharedTextureMemoryProperties props{};
    props.nextInChain = nullptr;
    wgpuSharedTextureMemoryGetProperties(mem, &props);

    WGPUTextureDescriptor td{};
    td.usage = WGPUTextureUsage_TextureBinding;
    td.dimension     = WGPUTextureDimension_2D;
    td.size          = props.size;
    td.format        = props.format;
    td.mipLevelCount = 1;
    td.sampleCount   = 1;

    WGPUTexture texture = wgpuSharedTextureMemoryCreateTexture(mem, &td);
    if (!texture) {
        wgpuSharedTextureMemoryRelease(mem);
        return nullptr;
    }

    // Begin GPU access (acquire keyed-mutex on D3D11, no-op on others).
    WGPUSharedTextureMemoryBeginAccessDescriptor beginDesc{};
    beginDesc.initialized = true; // producer (CopyResource) already wrote
    beginDesc.fenceCount  = 0;
    wgpuSharedTextureMemoryBeginAccess(mem, texture, &beginDesc);

    auto* tex = new MGPUVideoTexture();
    tex->width        = buf->width;
    tex->height       = buf->height;
    tex->pixel_format = buf->pixel_format;
    tex->content_type = buf->content_type;
    tex->shared_mem   = mem;
    tex->planes[0]    = texture;
    build_video_texture_views(tex, texture, buf->pixel_format, props.format);
    return tex;
}
#endif // !__EMSCRIPTEN__

/* =========================================================================
 * Platform-specific import functions
 * ====================================================================== */

#ifdef _WIN32
static ID3D11Device* get_or_create_d3d11_device_on_dawn_adapter();

/* =========================================================================
 * Cross-adapter D3D11 bridge (for hybrid laptops: iGPU capture, dGPU compute)
 *
 * On hybrid laptops the integrated panel is wired to the iGPU, so DXGI
 * Desktop Duplication / Windows.Graphics.Capture produce shared NT handles
 * for textures that physically live on the iGPU. Dawn typically auto-
 * selects the discrete GPU. NT shared handles do NOT cross adapters in
 * D3D11 — `OpenSharedResource1` returns E_INVALIDARG (0x80070057).
 *
 * Resolution strategy, attempted in order per-frame (with caching):
 *
 *   Tier A (zero-copy, same-adapter):
 *     OpenSharedResource1 on Dawn's D3D11 device. If it works, we're on
 *     the same adapter — pure GPU CopyResource + import to Dawn.
 *
 *   Tier B (D3D12 cross-adapter committed resource):
 *     Create a D3D12 device on the producer adapter. Create a committed
 *     cross-adapter texture (D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER,
 *     D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ALLOW_CROSS_ADAPTER |
 *     ALLOW_SIMULTANEOUS_ACCESS). Wrap it via D3D11on12 so we can
 *     CopyResource from the D3D11 source texture into it. Share the
 *     D3D12 resource as an NT handle. On Dawn's D3D12 device, open
 *     the NT handle via OpenSharedHandle — Dawn imports it directly
 *     via SharedTextureMemoryDXGISharedHandleDescriptor. A shared
 *     D3D12 fence synchronises the producer copy with Dawn's consumer
 *     queue. No CPU memory involved. Requires Dawn D3D12 backend and
 *     adapter support for CrossAdapterRowMajorTextureSupported.
 *
 *   Tier C (CPU bridge):
 *     Cache a secondary ID3D11Device on the producer's adapter (looked up
 *     by the LUID extracted from the source texture). Open the NT handle
 *     there, CopyResource to a CPU-readable staging texture, Map, then
 *     wgpuQueueWriteTexture into Dawn. Always works. Cost: one PCIe
 *     round-trip per frame (~3-8 ms at 1080p).
 *
 * The chosen tier is sticky per source-handle adapter — once we know a
 * given producer adapter doesn't share with Dawn's adapter we go straight
 * to Tier C for that adapter.
 * ====================================================================== */

enum class BridgeTier : uint8_t {
    Unknown     = 0,
    SameAdapter = 1, // Tier A: zero-copy, same adapter
    CrossD3D12  = 2, // Tier B: D3D12 cross-adapter committed resource (GPU copy, no CPU)
    CpuBridge   = 3, // Tier C: CPU staging round-trip
    Failed      = 4, // Persistent failure even on CPU bridge
};

struct ProducerAdapterInfo {
    LUID                                        luid{};
    Microsoft::WRL::ComPtr<IDXGIAdapter1>       adapter;

    // Tier C: plain D3D11 device on producer adapter (for staging / CPU bridge).
    Microsoft::WRL::ComPtr<ID3D11Device>        device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;

    // ── Tier B: D3D12 cross-adapter bridge ───────────────────────────────
    // Only initialised when Dawn is running the D3D12 backend AND the
    // producer adapter reports CrossAdapterRowMajorTextureSupported.
    Microsoft::WRL::ComPtr<ID3D12Device>        b12Dev;       // D3D12 on producer adapter
    Microsoft::WRL::ComPtr<ID3D12CommandQueue>  b12Queue;     // direct command queue
    Microsoft::WRL::ComPtr<ID3D11Device>        b11on12Dev;   // D3D11on12 wrapping b12Dev
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> b11on12Ctx;
    Microsoft::WRL::ComPtr<ID3D12Fence>         b12Fence;     // cross-process signal fence
    HANDLE                                      b12FenceHandle = nullptr; // NT handle (owned)
    std::atomic<uint64_t>                       b12FenceValue{0};
    WGPUSharedFence                             dawnFence   = nullptr; // Dawn-imported fence

    // Cached cross-adapter texture (reallocated on resolution / format change).
    Microsoft::WRL::ComPtr<ID3D12Resource>      b12CrossTex;     // committed cross-adapter tex
    Microsoft::WRL::ComPtr<ID3D11Texture2D>     b11on12Wrapped;  // D3D11on12 view of b12CrossTex
    WGPUSharedTextureMemory                     dawnCrossMem = nullptr; // Dawn shared mem
    uint32_t                                    b12TexW = 0;
    uint32_t                                    b12TexH = 0;
    DXGI_FORMAT                                 b12TexFmt = DXGI_FORMAT_UNKNOWN;
    // ─────────────────────────────────────────────────────────────────────

    BridgeTier                                  tier = BridgeTier::Unknown;
    std::atomic<int>                            consecutiveFailures{0};
    std::atomic<int>                            successesSinceFailure{0};
    std::atomic<bool>                           loggedMismatch{false};

    // Sticky negative result for Tier B: once init_tier_b determines this
    // adapter cannot do the D3D12 cross-adapter bridge (no
    // CrossAdapterRowMajorTextureSupported, device/queue creation failure),
    // don't re-probe — the old behavior re-ran D3D12CreateDevice AND
    // re-logged the "Falling back to Tier C" warning on every call (log spam
    // at frame rate).
    bool                                        tierBUnavailable = false;
};

static std::mutex                                                       g_producerInfoMutex;
static std::unordered_map<uint64_t, std::unique_ptr<ProducerAdapterInfo>> g_producerInfoByLuid;

static inline uint64_t luid_key(const LUID& l) {
    return (uint64_t(uint32_t(l.HighPart)) << 32) | uint64_t(uint32_t(l.LowPart));
}

// Probe an opened ID3D11Texture2D for the LUID of its owning adapter.
// Returns {0,0} on failure.
static LUID get_texture_adapter_luid(ID3D11Texture2D* tex) {
    LUID zero{0, 0};
    if (!tex) return zero;
    Microsoft::WRL::ComPtr<ID3D11Device> dev;
    tex->GetDevice(&dev);
    if (!dev) return zero;
    Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDev;
    if (FAILED(dev.As(&dxgiDev)) || !dxgiDev) return zero;
    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    if (FAILED(dxgiDev->GetAdapter(&adapter)) || !adapter) return zero;
    DXGI_ADAPTER_DESC d{};
    if (FAILED(adapter->GetDesc(&d))) return zero;
    return d.AdapterLuid;
}

static LUID get_dawn_adapter_luid() {
    LUID zero{0, 0};
    WGPUDevice device = get_device();
    if (!device) return zero;
    // Try D3D12 first, then D3D11 backend.
    if (auto d12 = dawn::native::d3d12::GetD3D12Device(device)) {
        return d12->GetAdapterLuid();
    }
    if (auto d11 = dawn::native::d3d11::GetD3D11Device(device)) {
        Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDev;
        if (SUCCEEDED(d11.As(&dxgiDev)) && dxgiDev) {
            Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
            if (SUCCEEDED(dxgiDev->GetAdapter(&adapter)) && adapter) {
                DXGI_ADAPTER_DESC d{};
                if (SUCCEEDED(adapter->GetDesc(&d))) return d.AdapterLuid;
            }
        }
    }
    return zero;
}

// Find or create a cached secondary D3D11 device on the producer adapter
// identified by `luid`. Used for Tier C (CPU bridge).
static ProducerAdapterInfo* get_or_create_producer_info(const LUID& luid) {
    if (luid.HighPart == 0 && luid.LowPart == 0) return nullptr;
    std::lock_guard<std::mutex> lk(g_producerInfoMutex);
    uint64_t key = luid_key(luid);
    auto it = g_producerInfoByLuid.find(key);
    if (it != g_producerInfoByLuid.end()) return it->second.get();

    auto info = std::make_unique<ProducerAdapterInfo>();
    info->luid = luid;

    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || !factory) {
        LOG_ERROR("[minigpu_external] producer_info: CreateDXGIFactory1 failed");
        return nullptr;
    }
    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    if (FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))) || !adapter) {
        LOG_ERROR("[minigpu_external] producer_info: EnumAdapterByLuid(%08lX:%08lX) failed",
            (unsigned long)luid.HighPart, (unsigned long)luid.LowPart);
        return nullptr;
    }
    if (FAILED(adapter.As(&info->adapter))) return nullptr;

    static const D3D_FEATURE_LEVEL kFLs[] = {
        D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
    };
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    D3D_FEATURE_LEVEL got{};
    HRESULT hr = D3D11CreateDevice(
        info->adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags,
        kFLs, (UINT)std::size(kFLs), D3D11_SDK_VERSION,
        &info->device, &got, &info->context);
    if (FAILED(hr) || !info->device) {
        LOG_ERROR("[minigpu_external] producer_info: D3D11CreateDevice on producer adapter failed: 0x%08lX",
            (unsigned long)hr);
        return nullptr;
    }
    {
        Microsoft::WRL::ComPtr<ID3D11Multithread> mt;
        if (SUCCEEDED(info->device.As(&mt)) && mt) {
            mt->SetMultithreadProtected(TRUE);
        }
    }

    DXGI_ADAPTER_DESC desc{};
    info->adapter->GetDesc(&desc);
    char nameUtf8[160] = {};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1,
                        nameUtf8, (int)sizeof(nameUtf8), nullptr, nullptr);
    LOG_INFO("[minigpu_external] producer_info: cached D3D11 device on adapter '%s' (LUID=%08lX:%08lX)",
        nameUtf8, (unsigned long)luid.HighPart, (unsigned long)luid.LowPart);

    auto* raw = info.get();
    g_producerInfoByLuid.emplace(key, std::move(info));
    return raw;
}

// ────────────────────────────────────────────────────────────────────────
// Tier B helpers
// ────────────────────────────────────────────────────────────────────────

// Returns true if Dawn is currently using the D3D12 backend.
static bool dawn_is_d3d12() {
    WGPUDevice dev = get_device();
    if (!dev) return false;
    return dawn::native::d3d12::GetD3D12Device(dev) != nullptr;
}

// Initialise the D3D12 + D3D11on12 state on `info`.
// Returns false (and logs) if the adapter doesn't support cross-adapter
// row-major textures or if any device creation step fails.
// Call through init_tier_b(), which caches a sticky negative result — this
// impl re-probes (D3D12CreateDevice etc.) and re-logs on every invocation.
static bool init_tier_b_impl(ProducerAdapterInfo* info) {
    if (info->b12Dev) return true; // already initialised

    // ── 1. Create D3D12 device on producer adapter ──
    HRESULT hr = D3D12CreateDevice(info->adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                   IID_PPV_ARGS(&info->b12Dev));
    if (FAILED(hr) || !info->b12Dev) {
        LOG_WARN("[minigpu_external] Tier B: D3D12CreateDevice on producer adapter failed: 0x%08lX. "
                 "Falling back to Tier C.", (unsigned long)hr);
        return false;
    }

    // ── 2. Check cross-adapter row-major texture support ──
    D3D12_FEATURE_DATA_D3D12_OPTIONS opts{};
    info->b12Dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts, sizeof(opts));
    if (!opts.CrossAdapterRowMajorTextureSupported) {
        LOG_WARN("[minigpu_external] Tier B: adapter does not support "
                 "CrossAdapterRowMajorTextureSupported. Falling back to Tier C.");
        info->b12Dev.Reset();
        return false;
    }

    // ── 3. Create direct D3D12 command queue ──
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type     = D3D12_COMMAND_LIST_TYPE_DIRECT;
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    hr = info->b12Dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&info->b12Queue));
    if (FAILED(hr) || !info->b12Queue) {
        LOG_WARN("[minigpu_external] Tier B: CreateCommandQueue failed: 0x%08lX.", (unsigned long)hr);
        info->b12Dev.Reset();
        return false;
    }

    // ── 4. Create D3D11on12 device wrapping the D3D12 device/queue ──
    IUnknown* queues[] = { info->b12Queue.Get() };
    Microsoft::WRL::ComPtr<ID3D11Device> raw11;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> raw11ctx;
    hr = D3D11On12CreateDevice(
        info->b12Dev.Get(),
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        nullptr, 0,
        queues, 1, 0,
        &raw11, &raw11ctx, nullptr);
    if (FAILED(hr) || !raw11) {
        LOG_WARN("[minigpu_external] Tier B: D3D11On12CreateDevice failed: 0x%08lX. "
                 "Falling back to Tier C.", (unsigned long)hr);
        info->b12Queue.Reset();
        info->b12Dev.Reset();
        return false;
    }
    info->b11on12Dev = raw11;
    info->b11on12Ctx = raw11ctx;
    {
        Microsoft::WRL::ComPtr<ID3D11Multithread> mt;
        if (SUCCEEDED(raw11.As(&mt)) && mt) mt->SetMultithreadProtected(TRUE);
    }

    // ── 5. Create cross-process D3D12 fence ──
    hr = info->b12Dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED | D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER,
                                   IID_PPV_ARGS(&info->b12Fence));
    if (FAILED(hr) || !info->b12Fence) {
        LOG_WARN("[minigpu_external] Tier B: CreateFence failed: 0x%08lX.", (unsigned long)hr);
        info->b11on12Ctx.Reset(); info->b11on12Dev.Reset();
        info->b12Queue.Reset();   info->b12Dev.Reset();
        return false;
    }

    hr = info->b12Dev->CreateSharedHandle(info->b12Fence.Get(), nullptr,
                                          GENERIC_ALL, nullptr, &info->b12FenceHandle);
    if (FAILED(hr) || !info->b12FenceHandle) {
        LOG_WARN("[minigpu_external] Tier B: CreateSharedHandle(fence) failed: 0x%08lX.", (unsigned long)hr);
        info->b12Fence.Reset();
        info->b11on12Ctx.Reset(); info->b11on12Dev.Reset();
        info->b12Queue.Reset();   info->b12Dev.Reset();
        return false;
    }

    // ── 6. Import the fence into Dawn ──
    WGPUDevice dawnDev = get_device();
    WGPUSharedFenceDXGISharedHandleDescriptor fenceDesc{};
    fenceDesc.chain.sType = WGPUSType_SharedFenceDXGISharedHandleDescriptor;
    fenceDesc.handle      = info->b12FenceHandle;
    WGPUSharedFenceDescriptor sfDesc{};
    sfDesc.nextInChain = &fenceDesc.chain;
    info->dawnFence = wgpuDeviceImportSharedFence(dawnDev, &sfDesc);
    // NT handle ownership transfers to Dawn; close our copy.
    CloseHandle(info->b12FenceHandle);
    info->b12FenceHandle = nullptr;
    if (!info->dawnFence) {
        LOG_WARN("[minigpu_external] Tier B: wgpuDeviceImportSharedFence failed.");
        info->b12Fence.Reset();
        info->b11on12Ctx.Reset(); info->b11on12Dev.Reset();
        info->b12Queue.Reset();   info->b12Dev.Reset();
        return false;
    }

    LOG_INFO("[minigpu_external] Tier B: initialised D3D12 cross-adapter bridge on producer adapter.");
    return true;
}

// Cached Tier B init. On failure the negative result sticks to the (per-LUID,
// process-lifetime) ProducerAdapterInfo, so an adapter that can't do Tier B is
// probed and logged ONCE instead of re-running D3D12CreateDevice and
// re-warning "Falling back to Tier C" on every frame.
static bool init_tier_b(ProducerAdapterInfo* info) {
    if (info->b12Dev) return true;            // already initialised
    if (info->tierBUnavailable) return false; // sticky negative — skip re-probe
    if (init_tier_b_impl(info)) return true;
    info->tierBUnavailable = true;
    return false;
}

// Ensure the cached cross-adapter D3D12 texture matches `w × h × fmt`.
// Creates (or recreates) the texture, its D3D11on12 wrapper, and the Dawn
// SharedTextureMemory. Returns false on any failure.
static bool ensure_cross_adapter_texture(ProducerAdapterInfo* info,
                                         uint32_t w, uint32_t h, DXGI_FORMAT fmt) {
    if (info->b12CrossTex && info->b12TexW == w && info->b12TexH == h && info->b12TexFmt == fmt)
        return true; // already valid

    // Release old resources.
    if (info->dawnCrossMem) { wgpuSharedTextureMemoryRelease(info->dawnCrossMem); info->dawnCrossMem = nullptr; }
    info->b11on12Wrapped.Reset();
    info->b12CrossTex.Reset();
    info->b12TexW = 0; info->b12TexH = 0; info->b12TexFmt = DXGI_FORMAT_UNKNOWN;

    // ── 1. Create committed cross-adapter texture on producer D3D12 device ──
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type                 = D3D12_HEAP_TYPE_DEFAULT;
    hp.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    hp.CreationNodeMask     = 1;
    hp.VisibleNodeMask      = 1;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Alignment        = 0;
    rd.Width            = w;
    rd.Height           = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = fmt;
    rd.SampleDesc       = { 1, 0 };
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER
                        | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

    Microsoft::WRL::ComPtr<ID3D12Resource> crossTex;
    HRESULT hr = info->b12Dev->CreateCommittedResource(
        &hp,
        D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER,
        &rd,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&crossTex));
    if (FAILED(hr) || !crossTex) {
        LOG_ERROR("[minigpu_external] Tier B: CreateCommittedResource(cross-adapter) failed: 0x%08lX", (unsigned long)hr);
        return false;
    }

    // ── 2. Wrap via D3D11on12 so CopyResource works from D3D11 source ──
    D3D11_RESOURCE_FLAGS rf{};
    rf.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    Microsoft::WRL::ComPtr<ID3D11On12Device> on12;
    if (FAILED(info->b11on12Dev.As(&on12)) || !on12) {
        LOG_ERROR("[minigpu_external] Tier B: QI for ID3D11On12Device failed.");
        return false;
    }
    Microsoft::WRL::ComPtr<ID3D11Texture2D> wrapped11tex;
    hr = on12->CreateWrappedResource(
        crossTex.Get(),
        &rf,
        D3D12_RESOURCE_STATE_COPY_DEST,  // D3D11 acquires for writing
        D3D12_RESOURCE_STATE_COMMON,     // restore to COMMON for cross-adapter read
        IID_PPV_ARGS(&wrapped11tex));
    if (FAILED(hr) || !wrapped11tex) {
        LOG_ERROR("[minigpu_external] Tier B: CreateWrappedResource failed: 0x%08lX", (unsigned long)hr);
        return false;
    }

    // ── 3. Share the D3D12 resource as NT handle → import into Dawn ──
    HANDLE resHandle = nullptr;
    hr = info->b12Dev->CreateSharedHandle(crossTex.Get(), nullptr, GENERIC_ALL, nullptr, &resHandle);
    if (FAILED(hr) || !resHandle) {
        LOG_ERROR("[minigpu_external] Tier B: CreateSharedHandle(texture) failed: 0x%08lX", (unsigned long)hr);
        return false;
    }

    WGPUDevice dawnDev = get_device();
    WGPUSharedTextureMemoryDXGISharedHandleDescriptor handleDesc{};
    handleDesc.chain.sType = WGPUSType_SharedTextureMemoryDXGISharedHandleDescriptor;
    handleDesc.handle      = resHandle;
    handleDesc.useKeyedMutex = false;
    WGPUSharedTextureMemoryDescriptor memDesc{};
    memDesc.nextInChain = &handleDesc.chain;
    WGPUSharedTextureMemory dawnMem = wgpuDeviceImportSharedTextureMemory(dawnDev, &memDesc);
    CloseHandle(resHandle); // Dawn opened it; close our copy.
    if (!dawnMem) {
        LOG_ERROR("[minigpu_external] Tier B: wgpuDeviceImportSharedTextureMemory failed for "
                  "cross-adapter D3D12 resource. Dawn may not be on D3D12 backend.");
        return false;
    }

    info->b12CrossTex    = crossTex;
    info->b11on12Wrapped = wrapped11tex;
    info->dawnCrossMem   = dawnMem;
    info->b12TexW        = w;
    info->b12TexH        = h;
    info->b12TexFmt      = fmt;
    LOG_INFO("[minigpu_external] Tier B: allocated cross-adapter texture %ux%u fmt=%d.", w, h, (int)fmt);
    return true;
}

// ── Tier B per-frame import ──────────────────────────────────────────────
// Opens `srcHandle` on the D3D11on12 device, copies to the cross-adapter
// D3D12 texture, signals the D3D12 fence, then returns an MGPUVideoTexture
// backed by the cached Dawn SharedTextureMemory with a fence-wait BeginAccess.
static MGPUVideoTexture* import_d3d11_d3d12_bridge(const MGPUExternalVideoBuffer* buf,
                                                   HANDLE srcHandle,
                                                   ProducerAdapterInfo** outInfo) {
    *outInfo = nullptr;

    // ── 1. Locate producer adapter info ──
    // We don't yet know the producer LUID from just the handle, so try the
    // fast path (cached producers) then fall back to adapter enumeration,
    // same approach as Tier C's slow path.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> srcOnD3D11on12;
    ProducerAdapterInfo* info = nullptr;

    // Fast path: try previously-cached producers with D3D11on12 initialised.
    {
        std::lock_guard<std::mutex> lk(g_producerInfoMutex);
        for (auto& kv : g_producerInfoByLuid) {
            auto& c = kv.second;
            if (!c->b11on12Dev) continue;
            Microsoft::WRL::ComPtr<ID3D11Device1> dev1;
            if (FAILED(c->b11on12Dev.As(&dev1)) || !dev1) continue;
            Microsoft::WRL::ComPtr<ID3D11Texture2D> tex;
            if (SUCCEEDED(dev1->OpenSharedResource1(srcHandle, IID_PPV_ARGS(&tex))) && tex) {
                srcOnD3D11on12 = tex;
                info = c.get();
                break;
            }
        }
    }

    // Slow path: enumerate adapters, init Tier B state.
    if (!srcOnD3D11on12) {
        Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || !factory) return nullptr;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, &adapter)); ++i, adapter.Reset()) {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            ProducerAdapterInfo* c = get_or_create_producer_info(desc.AdapterLuid);
            if (!c) continue;
            if (!c->b12Dev && !init_tier_b(c)) continue; // adapter doesn't support Tier B
            Microsoft::WRL::ComPtr<ID3D11Device1> dev1;
            if (FAILED(c->b11on12Dev.As(&dev1)) || !dev1) continue;
            Microsoft::WRL::ComPtr<ID3D11Texture2D> tex;
            if (SUCCEEDED(dev1->OpenSharedResource1(srcHandle, IID_PPV_ARGS(&tex))) && tex) {
                srcOnD3D11on12 = tex;
                info = c;
                break;
            }
        }
    }

    if (!srcOnD3D11on12 || !info) {
        LOG_DEBUG("[minigpu_external] Tier B: could not open source handle on any D3D11on12 device.");
        return nullptr;
    }
    *outInfo = info;

    // ── 2. Ensure cross-adapter texture is allocated ──
    D3D11_TEXTURE2D_DESC sd{};
    srcOnD3D11on12->GetDesc(&sd);
    if (!ensure_cross_adapter_texture(info, sd.Width, sd.Height, sd.Format))
        return nullptr;

    // ── 3. GPU copy: D3D11on12 CopyResource into the wrapped cross-adapter texture ──
    Microsoft::WRL::ComPtr<ID3D11On12Device> on12;
    if (FAILED(info->b11on12Dev.As(&on12))) return nullptr;
    ID3D11Resource* wrapped = info->b11on12Wrapped.Get();
    on12->AcquireWrappedResources(&wrapped, 1);
    info->b11on12Ctx->CopyResource(info->b11on12Wrapped.Get(), srcOnD3D11on12.Get());
    on12->ReleaseWrappedResources(&wrapped, 1);
    info->b11on12Ctx->Flush();

    // ── 4. Signal D3D12 fence on producer queue ──
    uint64_t signalVal = info->b12FenceValue.fetch_add(1) + 1;
    HRESULT hr = info->b12Queue->Signal(info->b12Fence.Get(), signalVal);
    if (FAILED(hr)) {
        LOG_ERROR("[minigpu_external] Tier B: queue Signal failed: 0x%08lX", (unsigned long)hr);
        return nullptr;
    }

    // ── 5. Create a Dawn texture from the cached shared memory ──
    WGPUSharedTextureMemoryProperties props{};
    wgpuSharedTextureMemoryGetProperties(info->dawnCrossMem, &props);
    WGPUTextureDescriptor td{};
    td.usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
    td.dimension     = WGPUTextureDimension_2D;
    td.size          = props.size;
    td.format        = props.format;
    td.mipLevelCount = 1;
    td.sampleCount   = 1;
    WGPUTexture wTex = wgpuSharedTextureMemoryCreateTexture(info->dawnCrossMem, &td);
    if (!wTex) {
        LOG_ERROR("[minigpu_external] Tier B: wgpuSharedTextureMemoryCreateTexture failed.");
        return nullptr;
    }

    // ── 6. BeginAccess with fence wait ──
    WGPUSharedTextureMemoryBeginAccessDescriptor baDesc{};
    baDesc.initialized   = true;
    baDesc.fenceCount    = 1;
    baDesc.fences        = &info->dawnFence;
    baDesc.signaledValues = &signalVal;
    wgpuSharedTextureMemoryBeginAccess(info->dawnCrossMem, wTex, &baDesc);

    // ── 7. Build the returned texture object ──
    // AddRef the shared memory so mgpuDestroyVideoTexture's Release doesn't
    // free our cached copy.
    wgpuSharedTextureMemoryAddRef(info->dawnCrossMem);

    auto* tex = new MGPUVideoTexture();
    tex->width        = buf->width;
    tex->height       = buf->height;
    tex->pixel_format = buf->pixel_format;
    tex->content_type = buf->content_type;
    tex->shared_mem   = info->dawnCrossMem; // caller (via mgpuDestroyVideoTexture) will EndAccess + Release
    tex->planes[0]    = wTex;
    build_video_texture_views(tex, wTex, buf->pixel_format, props.format);
    return tex;
}

// ────────────────────────────────────────────────────────────────────────
// Tier A: same-adapter zero-copy import.
// Returns a Dawn-imported texture, or nullptr on failure.
// `outNeedsCpuFallback` is set to true ONLY when failure is consistent with
// a cross-adapter mismatch (E_INVALIDARG / E_ACCESSDENIED on
// OpenSharedResource1) so the caller can fall through to Tier B / C.
// ────────────────────────────────────────────────────────────────────────
static MGPUVideoTexture* import_d3d11_same_adapter(const MGPUExternalVideoBuffer* buf,
                                                   HANDLE srcHandle,
                                                   bool* outNeedsCpuFallback) {
    *outNeedsCpuFallback = false;
    WGPUDevice device = get_device();
    if (!device) return nullptr;

    ID3D11Device* d11 = get_or_create_d3d11_device_on_dawn_adapter();
    if (!d11) return nullptr;
    Microsoft::WRL::ComPtr<ID3D11Device1> d11_1;
    if (FAILED(d11->QueryInterface(IID_PPV_ARGS(&d11_1))) || !d11_1) return nullptr;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> ctx;
    d11->GetImmediateContext(&ctx);
    if (!ctx) return nullptr;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> srcOnDawnDev;
    HRESULT hr = d11_1->OpenSharedResource1(srcHandle, IID_PPV_ARGS(&srcOnDawnDev));
    if (FAILED(hr) || !srcOnDawnDev) {
        // E_INVALIDARG / E_ACCESSDENIED ⇒ almost certainly cross-adapter.
        if (hr == E_INVALIDARG || hr == E_ACCESSDENIED) {
            *outNeedsCpuFallback = true;
        }
        return nullptr;
    }

    D3D11_TEXTURE2D_DESC sd{};
    srcOnDawnDev->GetDesc(&sd);
    {
        static int s_probed = 0;
        if (s_probed++ < 2) {
            LOG_DEBUG("[minigpu_external] miniav source MiscFlags=0x%X format=%d %ux%u",
                (unsigned)sd.MiscFlags, (int)sd.Format, sd.Width, sd.Height);
        }
    }

    // Create the Dawn-private destination on the SAME device. No sharing
    // flags. Bind for both shader read and unordered access (so Dawn can use
    // it as a storage binding too if desired). We add RENDER_TARGET to keep
    // PropertiesFromD3D11Texture happy and so we could also init-clear if
    // ever needed.
    D3D11_TEXTURE2D_DESC dd = sd;
    dd.Usage          = D3D11_USAGE_DEFAULT;
    dd.BindFlags      = D3D11_BIND_SHADER_RESOURCE
                      | D3D11_BIND_UNORDERED_ACCESS
                      | D3D11_BIND_RENDER_TARGET;
    dd.CPUAccessFlags = 0;
    dd.MiscFlags      = 0;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> dst;
    hr = d11->CreateTexture2D(&dd, nullptr, &dst);
    if (FAILED(hr) || !dst) {
        LOG_ERROR("[minigpu_external] import_d3d11: CreateTexture2D(private dst) failed: 0x%08lX", (unsigned long)hr);
        return nullptr;
    }
    ctx->CopyResource(dst.Get(), srcOnDawnDev.Get());

    // CPU wait for the copy to complete so the consumer (Dawn compute pass)
    // is guaranteed to see fully-flushed pixel data.
    {
        D3D11_QUERY_DESC qd{};
        qd.Query = D3D11_QUERY_EVENT;
        Microsoft::WRL::ComPtr<ID3D11Query> q;
        if (SUCCEEDED(d11->CreateQuery(&qd, &q)) && q) {
            ctx->End(q.Get());
            ctx->Flush();
            ULONGLONG t0 = GetTickCount64();
            while (ctx->GetData(q.Get(), nullptr, 0, 0) == S_FALSE) {
                if (GetTickCount64() - t0 > 50) break;
                YieldProcessor();
            }
        } else {
            ctx->Flush();
        }
    }

    // Hand the Dawn-private texture to Dawn via the D3D11 backend's direct
    // texture descriptor (no shared handle path needed — same device).
    dawn::native::d3d11::SharedTextureMemoryD3D11Texture2DDescriptor d11Desc;
    d11Desc.texture = dst;
    WGPUSharedTextureMemoryDescriptor desc{};
    desc.nextInChain =
        const_cast<WGPUChainedStruct*>(
            reinterpret_cast<const WGPUChainedStruct*>(
                static_cast<const wgpu::ChainedStruct*>(&d11Desc)));
    WGPUSharedTextureMemory mem = wgpuDeviceImportSharedTextureMemory(device, &desc);
    if (!mem) {
        LOG_ERROR("[minigpu_external] import_d3d11: importSharedTextureMemory(D3D11Texture2D) failed.");
        return nullptr;
    }
    {
        static int s_logged = 0;
        if (s_logged++ < 3) {
            LOG_DEBUG("[minigpu_external] import_d3d11: device=%p mem=%p (CopyResource path)",
                (void*)device, (void*)mem);
        }
    }

    return import_shared(mem, buf);
}

// ────────────────────────────────────────────────────────────────────────
// Tier C: CPU staging round-trip via secondary D3D11 device on the
// producer's adapter. Always works on hybrid laptops at the cost of one
// PCIe transfer per frame.
// ────────────────────────────────────────────────────────────────────────
static MGPUVideoTexture* import_d3d11_cpu_bridge(const MGPUExternalVideoBuffer* buf,
                                                 HANDLE srcHandle,
                                                 ProducerAdapterInfo** outInfo) {
    *outInfo = nullptr;
    WGPUDevice device = get_device();
    WGPUQueue  queue  = get_queue();
    if (!device || !queue) return nullptr;

    // We don't yet know the producer adapter — try opening the handle on a
    // freshly-enumerated set of adapters until one succeeds. We cache the
    // winning adapter for subsequent frames.
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || !factory) {
        LOG_ERROR("[minigpu_external] cpu_bridge: CreateDXGIFactory1 failed");
        return nullptr;
    }

    Microsoft::WRL::ComPtr<ID3D11Texture2D> srcOnProducer;
    ProducerAdapterInfo* info = nullptr;

    // Fast path: try previously-cached producer adapters first.
    {
        std::lock_guard<std::mutex> lk(g_producerInfoMutex);
        for (auto& kv : g_producerInfoByLuid) {
            auto& candidate = kv.second;
            if (!candidate->device) continue;
            Microsoft::WRL::ComPtr<ID3D11Device1> dev1;
            if (FAILED(candidate->device.As(&dev1)) || !dev1) continue;
            Microsoft::WRL::ComPtr<ID3D11Texture2D> tex;
            HRESULT hr = dev1->OpenSharedResource1(srcHandle, IID_PPV_ARGS(&tex));
            if (SUCCEEDED(hr) && tex) {
                srcOnProducer = tex;
                info = candidate.get();
                break;
            }
        }
    }

    // Slow path: enumerate adapters.
    if (!srcOnProducer) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, &adapter)); ++i, adapter.Reset()) {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            ProducerAdapterInfo* candidate = get_or_create_producer_info(desc.AdapterLuid);
            if (!candidate || !candidate->device) continue;
            Microsoft::WRL::ComPtr<ID3D11Device1> dev1;
            if (FAILED(candidate->device.As(&dev1)) || !dev1) continue;
            Microsoft::WRL::ComPtr<ID3D11Texture2D> tex;
            HRESULT hr = dev1->OpenSharedResource1(srcHandle, IID_PPV_ARGS(&tex));
            if (SUCCEEDED(hr) && tex) {
                srcOnProducer = tex;
                info = candidate;
                break;
            }
        }
    }

    if (!srcOnProducer || !info) {
        LOG_ERROR("[minigpu_external] cpu_bridge: could not open shared handle on any adapter");
        return nullptr;
    }
    *outInfo = info;

    D3D11_TEXTURE2D_DESC sd{};
    srcOnProducer->GetDesc(&sd);

    // Create a CPU-readable staging texture on the producer device.
    D3D11_TEXTURE2D_DESC stagingDesc = sd;
    stagingDesc.Usage          = D3D11_USAGE_STAGING;
    stagingDesc.BindFlags      = 0;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    stagingDesc.MiscFlags      = 0;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
    HRESULT hr = info->device->CreateTexture2D(&stagingDesc, nullptr, &staging);
    if (FAILED(hr) || !staging) {
        LOG_ERROR("[minigpu_external] cpu_bridge: CreateTexture2D(staging) failed: 0x%08lX",
            (unsigned long)hr);
        return nullptr;
    }
    info->context->CopyResource(staging.Get(), srcOnProducer.Get());

    // Wait for the producer-adapter copy.
    {
        D3D11_QUERY_DESC qd{}; qd.Query = D3D11_QUERY_EVENT;
        Microsoft::WRL::ComPtr<ID3D11Query> q;
        if (SUCCEEDED(info->device->CreateQuery(&qd, &q)) && q) {
            info->context->End(q.Get());
            info->context->Flush();
            ULONGLONG t0 = GetTickCount64();
            while (info->context->GetData(q.Get(), nullptr, 0, 0) == S_FALSE) {
                if (GetTickCount64() - t0 > 100) break;
                YieldProcessor();
            }
        } else {
            info->context->Flush();
        }
    }

    // Map and read the staging texture.
    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = info->context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr) || !mapped.pData) {
        LOG_ERROR("[minigpu_external] cpu_bridge: Map(staging) failed: 0x%08lX",
            (unsigned long)hr);
        return nullptr;
    }

    // Allocate a Dawn texture and upload via wgpuQueueWriteTexture.
    WGPUTextureFormat wfmt = (sd.Format == DXGI_FORMAT_R8G8B8A8_UNORM)
                           ? WGPUTextureFormat_RGBA8Unorm
                           : WGPUTextureFormat_BGRA8Unorm;
    WGPUTextureDescriptor td{};
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
             | WGPUTextureUsage_CopySrc;
    td.dimension = WGPUTextureDimension_2D;
    td.size = {sd.Width, sd.Height, 1};
    td.format = wfmt;
    td.mipLevelCount = 1;
    td.sampleCount = 1;
    WGPUTexture wTex = wgpuDeviceCreateTexture(device, &td);
    if (!wTex) {
        info->context->Unmap(staging.Get(), 0);
        LOG_ERROR("[minigpu_external] cpu_bridge: wgpuDeviceCreateTexture failed");
        return nullptr;
    }

    WGPUTexelCopyTextureInfo dstInfo{};
    dstInfo.texture  = wTex;
    dstInfo.mipLevel = 0;
    dstInfo.aspect   = WGPUTextureAspect_All;
    WGPUTexelCopyBufferLayout layout{};
    layout.offset       = 0;
    layout.bytesPerRow  = mapped.RowPitch;
    layout.rowsPerImage = sd.Height;
    WGPUExtent3D extent = {sd.Width, sd.Height, 1};
    size_t dataSize = (size_t)mapped.RowPitch * sd.Height;
    wgpuQueueWriteTexture(queue, &dstInfo, mapped.pData, dataSize, &layout, &extent);

    info->context->Unmap(staging.Get(), 0);

    auto* tex = new MGPUVideoTexture();
    tex->width        = buf->width;
    tex->height       = buf->height;
    tex->pixel_format = buf->pixel_format;
    tex->content_type = buf->content_type;
    tex->planes[0]    = wTex;
    build_video_texture_views(tex, wTex, buf->pixel_format, wfmt);
    return tex;
}

// ────────────────────────────────────────────────────────────────────────
// Tiered import dispatch entry point.
// ────────────────────────────────────────────────────────────────────────
static MGPUVideoTexture* import_d3d11(const MGPUExternalVideoBuffer* buf) {
    WGPUDevice device = get_device();
    if (!device) return nullptr;

    HANDLE srcHandle = reinterpret_cast<HANDLE>(buf->planes[0].data_ptr);
    if (!srcHandle || srcHandle == INVALID_HANDLE_VALUE) return nullptr;

    // Tier A: same-adapter zero-copy.
    bool needsCpuFallback = false;
    if (auto* result = import_d3d11_same_adapter(buf, srcHandle, &needsCpuFallback)) {
        return result;
    }

    // Tier A failed. Regardless of the failure mode (cross-adapter, transient,
    // backend mismatch, format quirk), fall through to Tier B / Tier C — Tier
    // C is the universal CPU bridge and always works. Returning null here
    // would leave the upstream encoder with no frame this tick and produce
    // empty / no-frame output files.
    if (!needsCpuFallback) {
        static std::atomic<int> s_otherFailures{0};
        int n = s_otherFailures.fetch_add(1);
        if (n < 3 || (n % 256) == 0) {
            LOG_WARN("[minigpu_external] import_d3d11: Tier A failed for unknown reason "
                     "(count=%d) — falling through to Tier B/C.", n + 1);
        }
    }

    // Tier B (D3D12 cross-adapter committed resource) — only viable when
    // Dawn is on the D3D12 backend AND the producer adapter supports
    // CrossAdapterRowMajorTextureSupported.
    if (dawn_is_d3d12()) {
        ProducerAdapterInfo* bInfo = nullptr;
        if (MGPUVideoTexture* result = import_d3d11_d3d12_bridge(buf, srcHandle, &bInfo)) {
            if (bInfo) {
                bInfo->consecutiveFailures.store(0);
                bInfo->successesSinceFailure.fetch_add(1);
                bInfo->tier = BridgeTier::CrossD3D12;
                if (!bInfo->loggedMismatch.exchange(true)) {
                    LUID dawnLuid = get_dawn_adapter_luid();
                    LOG_INFO("[minigpu_external] Tier B active: GPU-only cross-adapter bridge "
                             "(producer LUID=%08lX:%08lX → Dawn LUID=%08lX:%08lX).",
                             (unsigned long)bInfo->luid.HighPart, (unsigned long)bInfo->luid.LowPart,
                             (unsigned long)dawnLuid.HighPart,    (unsigned long)dawnLuid.LowPart);
                }
            }
            return result;
        }
        // Tier B failed — fall through to Tier C.
    }

    // Tier C: CPU bridge.
    ProducerAdapterInfo* info = nullptr;
    MGPUVideoTexture* result = import_d3d11_cpu_bridge(buf, srcHandle, &info);

    if (info && !info->loggedMismatch.exchange(true)) {
        LUID dawnLuid = get_dawn_adapter_luid();
        LOG_WARN("[minigpu_external] Cross-adapter capture detected. Producer LUID=%08lX:%08lX, "
                 "Dawn LUID=%08lX:%08lX. Falling back to CPU staging bridge "
                 "(~one PCIe round-trip per frame). Set MGPU_ADAPTER_NAME to a "
                 "substring of the producer adapter's name (e.g. 'Intel') to force "
                 "Dawn onto the producer's adapter for zero-copy at the cost of "
                 "compute speed.",
                 (unsigned long)info->luid.HighPart, (unsigned long)info->luid.LowPart,
                 (unsigned long)dawnLuid.HighPart,   (unsigned long)dawnLuid.LowPart);
    }

    if (!result) {
        if (info) {
            int fails = info->consecutiveFailures.fetch_add(1) + 1;
            info->successesSinceFailure.store(0);
            // Tolerate up to 5 transient failures before declaring persistent.
            if (fails == 5) {
                LOG_ERROR("[minigpu_external] CPU bridge persistent failure (5 consecutive "
                          "frames). Captures will continue to be dropped until import recovers.");
            }
        }
        return nullptr;
    }

    if (info) {
        info->consecutiveFailures.store(0);
        info->successesSinceFailure.fetch_add(1);
        info->tier = BridgeTier::CpuBridge;
    }
    return result;
}
#endif // _WIN32

#if defined(__APPLE__)
static MGPUVideoTexture* import_iosurface(const MGPUExternalVideoBuffer* buf) {
    WGPUDevice device = get_device();
    if (!device) return nullptr;

    IOSurfaceRef surface = reinterpret_cast<IOSurfaceRef>(buf->planes[0].data_ptr);
    if (!surface) return nullptr;

    WGPUSharedTextureMemoryIOSurfaceDescriptor chainDesc{};
    chainDesc.chain.sType = WGPUSType_SharedTextureMemoryIOSurfaceDescriptor;
    chainDesc.ioSurface   = surface;

    WGPUSharedTextureMemoryDescriptor desc{};
    desc.nextInChain = &chainDesc.chain;
    WGPUSharedTextureMemory mem = wgpuDeviceImportSharedTextureMemory(device, &desc);
    return import_shared(mem, buf);
}
#endif // __APPLE__

#if defined(__linux__) && !defined(__ANDROID__)
static MGPUVideoTexture* import_dmabuf(const MGPUExternalVideoBuffer* buf) {
    WGPUDevice device = get_device();
    if (!device) return nullptr;

    // Build per-plane descriptors.
    // Dawn's DmaBuf descriptor accepts up to 4 planes.
    uint32_t nPlanes = buf->num_planes;
    if (nPlanes == 0 || nPlanes > 4) return nullptr;

    // WGPUSharedTextureMemoryDmaBufPlane is a Dawn extension struct.
    WGPUSharedTextureMemoryDmaBufPlane dawnPlanes[4]{};
    for (uint32_t i = 0; i < nPlanes; ++i) {
        const MGPUExternalPlane& p = buf->planes[i];
        int fd = (p.dmabuf_fd >= 0) ? p.dmabuf_fd
                                    : static_cast<int>(reinterpret_cast<intptr_t>(p.data_ptr));
        dawnPlanes[i].fd     = fd;
        dawnPlanes[i].stride = p.stride_bytes;
        dawnPlanes[i].offset = p.offset_bytes;
    }

    // The DRM modifier is shared across all planes (same buffer).
    uint64_t modifier = buf->planes[0].drm_format_modifier;

    // Map the pixel format to a DRM fourcc (Dawn expects it).
    // Only the formats we advertise are handled here.
    uint32_t drmFourcc = 0;
    switch (buf->pixel_format) {
    case MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32: drmFourcc = 0x34325241 /* DRM_FORMAT_ARGB8888 */; break;
    case MGPU_EXTERNAL_PIXEL_FORMAT_RGBA32: drmFourcc = 0x34324241 /* DRM_FORMAT_ABGR8888 */; break;
    case MGPU_EXTERNAL_PIXEL_FORMAT_NV12:
    case MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_NV12_PLANES:
        drmFourcc = 0x3231564E /* DRM_FORMAT_NV12 */; break;
    default:
        return nullptr; // Unsupported on DmaBuf path; fall back to CPU
    }

    WGPUSharedTextureMemoryDmaBufDescriptor chainDesc{};
    chainDesc.chain.sType    = WGPUSType_SharedTextureMemoryDmaBufDescriptor;
    chainDesc.size           = {buf->width, buf->height, 1};
    chainDesc.drmFormat      = drmFourcc;
    chainDesc.drmModifier    = modifier;
    chainDesc.planeCount     = nPlanes;
    chainDesc.planes         = dawnPlanes;

    WGPUSharedTextureMemoryDescriptor desc{};
    desc.nextInChain = &chainDesc.chain;
    WGPUSharedTextureMemory mem = wgpuDeviceImportSharedTextureMemory(device, &desc);
    return import_shared(mem, buf);
}
#endif // linux non-android

#ifdef __ANDROID__
static MGPUVideoTexture* import_ahardwarebuffer(const MGPUExternalVideoBuffer* buf) {
    WGPUDevice device = get_device();
    if (!device) return nullptr;

    AHardwareBuffer* ahb = reinterpret_cast<AHardwareBuffer*>(buf->planes[0].data_ptr);
    if (!ahb) return nullptr;

    WGPUSharedTextureMemoryAHardwareBufferDescriptor chainDesc{};
    chainDesc.chain.sType = WGPUSType_SharedTextureMemoryAHardwareBufferDescriptor;
    chainDesc.handle      = ahb;

    WGPUSharedTextureMemoryDescriptor desc{};
    desc.nextInChain = &chainDesc.chain;
    WGPUSharedTextureMemory mem = wgpuDeviceImportSharedTextureMemory(device, &desc);
    return import_shared(mem, buf);
}
#endif // __ANDROID__

/* =========================================================================
 * Public API implementation
 * ====================================================================== */

MGPUVideoTexture* mgpuImportVideoFrame(const MGPUExternalVideoBuffer* buf) {
    if (!buf) return nullptr;

    switch (buf->content_type) {
    case MGPU_EXTERNAL_CONTENT_TYPE_CPU:
        return import_cpu(buf);

#ifdef _WIN32
    case MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE:
        return import_d3d11(buf);
#endif

#if defined(__APPLE__)
    case MGPU_EXTERNAL_CONTENT_TYPE_METAL_IOSURFACE:
        return import_iosurface(buf);
#endif

#if defined(__linux__) && !defined(__ANDROID__)
    case MGPU_EXTERNAL_CONTENT_TYPE_DMABUF:
        return import_dmabuf(buf);
#endif

#ifdef __ANDROID__
    case MGPU_EXTERNAL_CONTENT_TYPE_AHARDWAREBUFFER:
        return import_ahardwarebuffer(buf);
#endif

    default:
        // Unsupported on this platform; attempt CPU fallback if content is CPU.
        return nullptr;
    }
}

void mgpuSetVideoTexture(MGPUComputeShader* shader_c, int binding_slot,
                         MGPUVideoTexture* tex, uint32_t plane_index) {
    if (!shader_c || !tex) return;
    if ((int)plane_index >= tex->num_planes) return;
    if (!tex->views[plane_index]) return;

    auto* shader = reinterpret_cast<mgpu::ComputeShader*>(shader_c);
    shader->setTextureView(binding_slot, tex->views[plane_index]);
}

void mgpuDestroyVideoTexture(MGPUVideoTexture* tex) {
    if (!tex) return;

#ifndef __EMSCRIPTEN__
    if (tex->shared_mem) {
        // End access releases keyed-mutex / fence.
        WGPUSharedTextureMemoryEndAccessState endState{};
        wgpuSharedTextureMemoryEndAccess(tex->shared_mem, tex->planes[0], &endState);
    }
#endif

    for (int i = 0; i < tex->num_planes; ++i) {
        if (tex->views[i])   wgpuTextureViewRelease(tex->views[i]);
        if (tex->planes[i])  wgpuTextureRelease(tex->planes[i]);
    }
#ifndef __EMSCRIPTEN__
    if (tex->shared_mem) wgpuSharedTextureMemoryRelease(tex->shared_mem);
#endif

    delete tex;
}

/* =========================================================================
 * YUV -> RGBA conversion helper
 * ====================================================================== */

// Minimal WGSL compute shader for NV12 -> RGBA8 conversion (BT.709 full-range)
static const char* kNV12ToRGBAShader = R"(
@group(0) @binding(0) var y_tex  : texture_2d<f32>;
@group(0) @binding(1) var uv_tex : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out_buf : array<u32>;

struct Uniforms { width: u32, height: u32 }
@group(0) @binding(3) var<uniform> uni : Uniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uni.width || gid.y >= uni.height) { return; }
    let uv = vec2<f32>(f32(gid.x) / f32(uni.width),
                       f32(gid.y) / f32(uni.height));
    let y  = textureLoad(y_tex,  vec2<u32>(gid.x, gid.y), 0).r;
    let cb = textureLoad(uv_tex, vec2<u32>(gid.x / 2u, gid.y / 2u), 0).r - 0.5;
    let cr = textureLoad(uv_tex, vec2<u32>(gid.x / 2u, gid.y / 2u), 0).g - 0.5;
    let r  = clamp(y + 1.5748 * cr,            0.0, 1.0);
    let g  = clamp(y - 0.1873 * cb - 0.4681 * cr, 0.0, 1.0);
    let b  = clamp(y + 1.8556 * cb,            0.0, 1.0);
    let ri = u32(r * 255.0);
    let gi = u32(g * 255.0);
    let bi = u32(b * 255.0);
    out_buf[gid.y * uni.width + gid.x] = ri | (gi << 8u) | (bi << 16u) | (255u << 24u);
}
)";

static const char* kRGBAPassthroughShader = R"(
@group(0) @binding(0) var in_tex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf : array<u32>;

struct Uniforms { width: u32, height: u32 }
@group(0) @binding(2) var<uniform> uni : Uniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uni.width || gid.y >= uni.height) { return; }
    let px = textureLoad(in_tex, vec2<i32>(gid.xy), 0);
    let ri = u32(clamp(px.r, 0.0, 1.0) * 255.0);
    let gi = u32(clamp(px.g, 0.0, 1.0) * 255.0);
    let bi = u32(clamp(px.b, 0.0, 1.0) * 255.0);
    let ai = u32(clamp(px.a, 0.0, 1.0) * 255.0);
    out_buf[gid.y * uni.width + gid.x] = ri | (gi << 8u) | (bi << 16u) | (ai << 24u);
}
)";

MGPUBuffer* mgpuVideoTextureToRGBA(MGPUVideoTexture* tex) {
    if (!tex || tex->num_planes == 0) return nullptr;

    WGPUDevice device = get_device();
    WGPUQueue  queue  = get_queue();
    if (!device || !queue) return nullptr;
    {
        static int s_logged = 0;
        if (s_logged++ < 3) {
            LOG_DEBUG("[minigpu_external] toRGBA: device=%p tex.plane0=%p tex.view0=%p",
                (void*)device, (void*)tex->planes[0], (void*)tex->views[0]);
        }
    }

    uint32_t W = tex->width;
    uint32_t H = tex->height;
    size_t pixelCount = (size_t)W * H;
    size_t outBytes   = pixelCount * 4; // RGBA8

    // Output buffer
    auto* outBuf = new mgpu::Buffer(minigpu);
    outBuf->createBuffer(outBytes, mgpu::kUInt8);

    // Uniform buffer: width, height
    uint32_t uniforms[2] = {W, H};
    WGPUBufferDescriptor ubd{};
    ubd.size  = sizeof(uniforms);
    ubd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    WGPUBuffer ubuf = wgpuDeviceCreateBuffer(device, &ubd);
    wgpuQueueWriteBuffer(queue, ubuf, 0, uniforms, sizeof(uniforms));

    bool isNV12 = (tex->pixel_format == MGPU_EXTERNAL_PIXEL_FORMAT_NV12 ||
                   tex->pixel_format == MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_NV12_PLANES);

    // We reuse mgpu::ComputeShader for the dispatch
    mgpu::ComputeShader cs(minigpu);
    cs.loadKernelString(isNV12 ? kNV12ToRGBAShader : kRGBAPassthroughShader);

    // Bind textures as storage via setTextureView helper
    if (isNV12) {
        cs.setTextureView(0, tex->views[0]); // Y
        cs.setTextureView(1, tex->views[1]); // UV
        cs.setStorageBuffer(2, outBuf->bufferData.buffer, outBytes, 0);
        cs.setUniformBuffer(3, ubuf, sizeof(uniforms));
    } else {
        cs.setTextureView(0, tex->views[0]); // packed
        cs.setStorageBuffer(1, outBuf->bufferData.buffer, outBytes, 0);
        cs.setUniformBuffer(2, ubuf, sizeof(uniforms));
    }

    uint32_t gx = (W + 7) / 8;
    uint32_t gy = (H + 7) / 8;

    // Use dispatchAsync with a wait so the GPU submit is ordered before any
    // subsequent readback. dispatch() is fire-and-forget (async); the caller
    // may call readDirect immediately after we return, which would race against
    // the dispatch submission and copy zeros instead of computed data.
    //
    // Hang tolerance: bail out after 5 s if dispatchAsync's callback never
    // fires (driver TDR, lost device, etc.) so we don't lock the calling
    // thread forever.  The output buffer is still released by the caller
    // path's normal cleanup; we just propagate the failure as nullptr.
    std::promise<void> dispatchDone;
    auto dispatchFuture = dispatchDone.get_future();
    cs.dispatchAsync((int)gx, (int)gy, 1, [&dispatchDone]() {
        dispatchDone.set_value();
    });
    if (!drain_dawn_events_with_timeout(
            dispatchFuture, "NV12->RGBA8 dispatch")) {
        wgpuBufferRelease(ubuf);
        // outBuf was created on this code path; release it so we don't leak
        // when reporting failure.
        delete outBuf;
        return nullptr;
    }

    wgpuBufferRelease(ubuf);
    return reinterpret_cast<MGPUBuffer*>(outBuf);
}

} // extern "C"

/* =========================================================================
 * Cross-API shared output texture (Windows D3D12 <-> D3D11)
 * ====================================================================== */

#ifdef _WIN32

struct MGPUSharedOutputTexture {
    // We now create the underlying texture in D3D11 (not D3D12). D3D11 NT-
    // handle sharing is universally supported across D3D11 devices on the
    // same adapter, whereas D3D12->D3D11 ID3D11Device1::OpenSharedResource1
    // returns E_INVALIDARG on many NVIDIA driver versions even when LUIDs
    // match. By keeping the D3D11 texture and handing the encoder the same
    // device that created it, we never need OpenSharedResource1 at all.
    Microsoft::WRL::ComPtr<ID3D11Texture2D> d3d11_texture;
    HANDLE                                 nt_handle  = nullptr; // owned
    WGPUSharedTextureMemory                shared_mem = nullptr;
    WGPUTexture                            texture    = nullptr;
    WGPUTextureView                        view       = nullptr;

    // Dawn-internal intermediate texture used as the destination of the
    // buffer->texture compute pass. The shared imported texture cannot be
    // bound as a storage texture (Dawn's D3D11on12 wrap of a shared D3D11
    // resource fails when StorageBinding is requested). Instead, we write
    // into this Dawn-private texture and then CopyTextureToTexture into the
    // shared one (which only needs CopyDst).
    WGPUTexture                            intermediate_texture = nullptr;
    WGPUTextureView                        intermediate_view    = nullptr;

    uint32_t                               width      = 0;
    uint32_t                               height     = 0;

    // Cached compute pipeline for BGRA VideoTexture → shared output (built lazily).
    WGPUShaderModule       cs_module   = nullptr;
    WGPUBindGroupLayout    cs_bgl      = nullptr;
    WGPUPipelineLayout     cs_pl       = nullptr;
    WGPUComputePipeline    cs_pipeline = nullptr;

    // Cached compute pipeline for GPU Buffer (u32 array) → shared output (built lazily).
    WGPUShaderModule       copy_module   = nullptr;
    WGPUBindGroupLayout    copy_bgl      = nullptr;
    WGPUPipelineLayout     copy_pl       = nullptr;
    WGPUComputePipeline    copy_pipeline = nullptr;

    // Cached compute pipeline for GPU Buffer (f32 RGBA, 4 floats per pixel)
    // → shared output (built lazily).  Used by visualizers like the
    // spectrogram which write float colors into a tensor buffer.
    WGPUShaderModule       copyf32_module   = nullptr;
    WGPUBindGroupLayout    copyf32_bgl      = nullptr;
    WGPUPipelineLayout     copyf32_pl       = nullptr;
    WGPUComputePipeline    copyf32_pipeline = nullptr;
};

// WGSL: read texel from src (auto-swizzled to RGBA semantics by WebGPU
// regardless of source memory layout) and store into a bgra8unorm storage
// texture.  The destination is the private intermediate texture created
// with StorageBinding usage (BGRA8Unorm, matching the shared output
// texture's format); a CopyTextureToTexture in the dispatcher copies the
// result into the actual shared D3D11 texture before EndAccess.
static const char* kBGRASharedOutputShader = R"(
@group(0) @binding(0) var in_tex  : texture_2d<f32>;
@group(0) @binding(1) var out_tex : texture_storage_2d<bgra8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(out_tex);
    if (gid.x >= dim.x || gid.y >= dim.y) { return; }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let px = textureLoad(in_tex, coord, 0);
    textureStore(out_tex, coord, px);
}
)";

// Build cached pipeline objects on the texture (no-op if already built).
static bool ensure_shared_output_pipeline(MGPUSharedOutputTexture* tex,
                                          WGPUDevice device) {
    if (tex->cs_pipeline) return true;

    WGPUShaderSourceWGSL wgsl{};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code.data   = kBGRASharedOutputShader;
    wgsl.code.length = std::strlen(kBGRASharedOutputShader);

    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgsl.chain;
    tex->cs_module = wgpuDeviceCreateShaderModule(device, &smDesc);
    if (!tex->cs_module) return false;

    WGPUBindGroupLayoutEntry entries[2]{};
    entries[0].binding    = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].texture.sampleType    = WGPUTextureSampleType_Float;
    entries[0].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[0].texture.multisampled  = false;

    entries[1].binding    = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].storageTexture.access        = WGPUStorageTextureAccess_WriteOnly;
    entries[1].storageTexture.format        = WGPUTextureFormat_BGRA8Unorm;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 2;
    bglDesc.entries    = entries;
    tex->cs_bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);
    if (!tex->cs_bgl) return false;

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts     = &tex->cs_bgl;
    tex->cs_pl = wgpuDeviceCreatePipelineLayout(device, &plDesc);
    if (!tex->cs_pl) return false;

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout                    = tex->cs_pl;
    cpDesc.compute.module            = tex->cs_module;
    cpDesc.compute.entryPoint.data   = "main";
    cpDesc.compute.entryPoint.length = 4;
    tex->cs_pipeline = wgpuDeviceCreateComputePipeline(device, &cpDesc);
    return tex->cs_pipeline != nullptr;
}

// WGSL: read packed RGBA u32 pixels from a storage buffer and write into a
// write-only bgra8unorm storage texture. Source pixel layout in the buffer
// is R=bits[7:0], G=bits[15:8], B=bits[23:16], A=bits[31:24] (same as
// GpuEffect output). The destination is a BGRA8Unorm storage texture; the
// shader writes vec4(r,g,b,a) as logical color components, and the format
// handles the byte-order swap on store.
static const char* kBufferToSharedOutputShader = R"(
@group(0) @binding(0) var<storage, read> pixels : array<u32>;
@group(0) @binding(1) var out_tex : texture_storage_2d<bgra8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(out_tex);
    if (gid.x >= dim.x || gid.y >= dim.y) { return; }
    let i = gid.y * dim.x + gid.x;
    let px = pixels[i];
    let r = f32((px >>  0u) & 0xffu) / 255.0;
    let g = f32((px >>  8u) & 0xffu) / 255.0;
    let b = f32((px >> 16u) & 0xffu) / 255.0;
    let a = f32((px >> 24u) & 0xffu) / 255.0;
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(r, g, b, a));
}
)";

// WGSL variant: read 4 f32 components per pixel from a storage buffer and
// write into a bgra8unorm storage texture.  Used when the source buffer
// holds RGBA float pixels (e.g. the spectrogram visualization stage).
static const char* kBufferF32ToSharedOutputShader = R"(
@group(0) @binding(0) var<storage, read> pixels : array<f32>;
@group(0) @binding(1) var out_tex : texture_storage_2d<bgra8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(out_tex);
    if (gid.x >= dim.x || gid.y >= dim.y) { return; }
    let base = (gid.y * dim.x + gid.x) * 4u;
    let r = clamp(pixels[base + 0u], 0.0, 1.0);
    let g = clamp(pixels[base + 1u], 0.0, 1.0);
    let b = clamp(pixels[base + 2u], 0.0, 1.0);
    let a = clamp(pixels[base + 3u], 0.0, 1.0);
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(r, g, b, a));
}
)";

// Build (or reuse) the cached buffer-to-texture copy pipeline.
static bool ensure_copy_pipeline(MGPUSharedOutputTexture* tex,
                                 WGPUDevice device) {
    if (tex->copy_pipeline) return true;

    WGPUShaderSourceWGSL wgsl{};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code.data   = kBufferToSharedOutputShader;
    wgsl.code.length = std::strlen(kBufferToSharedOutputShader);

    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgsl.chain;
    tex->copy_module = wgpuDeviceCreateShaderModule(device, &smDesc);
    if (!tex->copy_module) return false;

    WGPUBindGroupLayoutEntry entries[2]{};
    entries[0].binding    = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

    entries[1].binding    = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].storageTexture.access        = WGPUStorageTextureAccess_WriteOnly;
    entries[1].storageTexture.format        = WGPUTextureFormat_BGRA8Unorm;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 2;
    bglDesc.entries    = entries;
    tex->copy_bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);
    if (!tex->copy_bgl) return false;

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts     = &tex->copy_bgl;
    tex->copy_pl = wgpuDeviceCreatePipelineLayout(device, &plDesc);
    if (!tex->copy_pl) return false;

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout                    = tex->copy_pl;
    cpDesc.compute.module            = tex->copy_module;
    cpDesc.compute.entryPoint.data   = "main";
    cpDesc.compute.entryPoint.length = 4;
    tex->copy_pipeline = wgpuDeviceCreateComputePipeline(device, &cpDesc);
    return tex->copy_pipeline != nullptr;
}

// Build (or reuse) the cached buffer(f32 RGBA) -> texture pipeline.
static bool ensure_copy_pipeline_f32(MGPUSharedOutputTexture* tex,
                                     WGPUDevice device) {
    if (tex->copyf32_pipeline) return true;

    WGPUShaderSourceWGSL wgsl{};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code.data   = kBufferF32ToSharedOutputShader;
    wgsl.code.length = std::strlen(kBufferF32ToSharedOutputShader);

    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgsl.chain;
    tex->copyf32_module = wgpuDeviceCreateShaderModule(device, &smDesc);
    if (!tex->copyf32_module) return false;

    WGPUBindGroupLayoutEntry entries[2]{};
    entries[0].binding    = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

    entries[1].binding    = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].storageTexture.access        = WGPUStorageTextureAccess_WriteOnly;
    entries[1].storageTexture.format        = WGPUTextureFormat_BGRA8Unorm;
    entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor bglDesc{};
    bglDesc.entryCount = 2;
    bglDesc.entries    = entries;
    tex->copyf32_bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);
    if (!tex->copyf32_bgl) return false;

    WGPUPipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts     = &tex->copyf32_bgl;
    tex->copyf32_pl = wgpuDeviceCreatePipelineLayout(device, &plDesc);
    if (!tex->copyf32_pl) return false;

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.layout                    = tex->copyf32_pl;
    cpDesc.compute.module            = tex->copyf32_module;
    cpDesc.compute.entryPoint.data   = "main";
    cpDesc.compute.entryPoint.length = 4;
    tex->copyf32_pipeline = wgpuDeviceCreateComputePipeline(device, &cpDesc);
    return tex->copyf32_pipeline != nullptr;
}

// Cached D3D11 device created on the same DXGI adapter as Dawn's D3D12
// device. Used both for the shared-output texture and as the encoder's
// AVHWDeviceContext device. Caching is required so that the texture and
// the encoder live on the *same* ID3D11Device — that's what lets us skip
// OpenSharedResource1 entirely (no cross-device sharing).
static Microsoft::WRL::ComPtr<ID3D11Device>        g_d3d11_device;
static Microsoft::WRL::ComPtr<ID3D11DeviceContext> g_d3d11_context;

// Best-effort per-device GPU submission-priority boost. When another process
// (e.g. a game) saturates the GPU, minigpu's compute/copy submissions queue
// behind its work and the recorder's per-frame GPU stage balloons; +7 raises
// this device's scheduling priority so those submissions preempt. (The
// recorder additionally raises the process-wide D3DKMT scheduling priority
// when screen capture starts, which covers this device too.) Non-fatal on
// failure.
static void boost_d3d11_device_gpu_priority(ID3D11Device* dev) {
    if (!dev) return;
    Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDev;
    if (SUCCEEDED(dev->QueryInterface(IID_PPV_ARGS(&dxgiDev))) && dxgiDev) {
        HRESULT hr = dxgiDev->SetGPUThreadPriority(7);
        LOG_INFO("[minigpu_external] SetGPUThreadPriority(+7) on D3D11 "
                 "device: 0x%08lX", (unsigned long)hr);
    }
}

static ID3D11Device* get_or_create_d3d11_device_on_dawn_adapter() {
    if (g_d3d11_device) return g_d3d11_device.Get();

    WGPUDevice device = get_device();
    if (!device) {
        LOG_ERROR("[minigpu_external] get_or_create_d3d11_device: no WGPUDevice");
        return nullptr;
    }

    // FAST PATH: if Dawn itself is on the D3D11 backend, just reuse Dawn's
    // own ID3D11Device — then there's only one device in the whole pipeline
    // (Dawn compute, SharedOutputTexture, FFmpeg D3D11VA encoder) and no
    // cross-API or cross-device sharing of any kind is required.
    {
        Microsoft::WRL::ComPtr<ID3D11Device> dawnD11 =
            dawn::native::d3d11::GetD3D11Device(device);
        if (dawnD11) {
            g_d3d11_device = dawnD11;
            g_d3d11_device->GetImmediateContext(&g_d3d11_context);
            Microsoft::WRL::ComPtr<ID3D11Multithread> mt;
            if (SUCCEEDED(g_d3d11_device.As(&mt)) && mt) {
                mt->SetMultithreadProtected(TRUE);
            }
            boost_d3d11_device_gpu_priority(g_d3d11_device.Get());
            LOG_INFO("[minigpu_external] Using Dawn's own ID3D11Device @%p",
                (void*)g_d3d11_device.Get());
            return g_d3d11_device.Get();
        }
    }

    Microsoft::WRL::ComPtr<ID3D12Device> d3d12Device =
        dawn::native::d3d12::GetD3D12Device(device);
    if (!d3d12Device) {
        LOG_ERROR("[minigpu_external] get_or_create_d3d11_device: GetD3D12Device failed");
        return nullptr;
    }

    LUID luid = d3d12Device->GetAdapterLuid();

    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        LOG_ERROR("[minigpu_external] CreateDXGIFactory1 failed: 0x%08lX", (unsigned long)hr);
        return nullptr;
    }

    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    hr = factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter));
    if (FAILED(hr)) {
        LOG_ERROR("[minigpu_external] EnumAdapterByLuid(LUID=%08lX:%08lX) failed: 0x%08lX",
            (unsigned long)luid.HighPart, (unsigned long)luid.LowPart, (unsigned long)hr);
        return nullptr;
    }

    static const D3D_FEATURE_LEVEL kFeatureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    };
    UINT createFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
    createFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL featureLevel = {};
    hr = D3D11CreateDevice(
        adapter.Get(),
        D3D_DRIVER_TYPE_UNKNOWN,
        nullptr,
        createFlags,
        kFeatureLevels,
        static_cast<UINT>(std::size(kFeatureLevels)),
        D3D11_SDK_VERSION,
        &g_d3d11_device,
        &featureLevel,
        &g_d3d11_context);
    if (FAILED(hr) || !g_d3d11_device) {
        LOG_ERROR("[minigpu_external] D3D11CreateDevice on Dawn adapter (LUID=%08lX:%08lX) failed: 0x%08lX",
            (unsigned long)luid.HighPart, (unsigned long)luid.LowPart, (unsigned long)hr);
        return nullptr;
    }

    // Multi-thread protection is required because FFmpeg's encoder issues
    // its own immediate-context calls (CopySubresourceRegion + fence) on a
    // potentially different thread from minigpu's compute submissions.
    {
        Microsoft::WRL::ComPtr<ID3D11Multithread> mt;
        if (SUCCEEDED(g_d3d11_device.As(&mt)) && mt) {
            mt->SetMultithreadProtected(TRUE);
        }
    }
    boost_d3d11_device_gpu_priority(g_d3d11_device.Get());

    LOG_INFO("[minigpu_external] Created cached D3D11 device on Dawn adapter (LUID=%08lX:%08lX, FL=0x%04X)",
        (unsigned long)luid.HighPart, (unsigned long)luid.LowPart,
        static_cast<unsigned>(featureLevel));
    return g_d3d11_device.Get();
}

static MGPUSharedOutputTexture* create_shared_output_texture(uint32_t w,
                                                             uint32_t h) {
    WGPUDevice device = get_device();
    if (!device || w == 0 || h == 0) return nullptr;

    ID3D11Device* d3d11Device = get_or_create_d3d11_device_on_dawn_adapter();
    if (!d3d11Device) {
        LOG_ERROR("[minigpu_external] create_shared_output_texture: no D3D11 device.");
        return nullptr;
    }

    // Detect whether Dawn is running on its own D3D11 backend; if so, use the
    // simple "single-device" path: create the texture on Dawn's d11 device and
    // hand the SAME ID3D11Texture2D to both Dawn (via
    // SharedTextureMemoryD3D11Texture2DDescriptor) and to the FFmpeg encoder.
    // No NT handle, no keyed mutex, no cross-API sync — all GPU work runs on
    // one device with normal in-device serialization.
    Microsoft::WRL::ComPtr<ID3D11Device> dawnD11 =
        dawn::native::d3d11::GetD3D11Device(device);
    const bool dawnOnD3D11 = dawnD11 && dawnD11.Get() == d3d11Device;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> d3d11Tex;
    HANDLE ntHandle = nullptr;
    WGPUSharedTextureMemory mem = nullptr;

    if (dawnOnD3D11) {
        D3D11_TEXTURE2D_DESC td{};
        td.Width            = w;
        td.Height           = h;
        td.MipLevels        = 1;
        td.ArraySize        = 1;
        td.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage            = D3D11_USAGE_DEFAULT;
        // RENDER_TARGET so we can D3D11 init-clear; SHADER_RESOURCE so Dawn
        // can sample it.  NOTE: D3D11_BIND_UNORDERED_ACCESS is intentionally
        // omitted — it is incompatible with D3D11_RESOURCE_MISC_SHARED.
        // Compute writes go through a private Dawn intermediate texture and
        // are blit-copied into this shared texture before EndAccess.
        td.BindFlags        = D3D11_BIND_RENDER_TARGET
                            | D3D11_BIND_SHADER_RESOURCE;
        // D3D11_RESOURCE_MISC_SHARED is required by Flutter's external texture
        // API (IDXGIResource::GetSharedHandle) to import the texture onto
        // Flutter's D3D device.  Without this flag CopyDescriptor fails at
        // external_texture_d3d.cc with "Binding D3D surface failed".
        td.MiscFlags        = D3D11_RESOURCE_MISC_SHARED;
        HRESULT hr = d3d11Device->CreateTexture2D(&td, nullptr, &d3d11Tex);
        if (FAILED(hr) || !d3d11Tex) {
            LOG_ERROR("[minigpu_external] CreateTexture2D(SharedOutput, single-device) failed: 0x%08lX", (unsigned long)hr);
            return nullptr;
        }

        // Import into Dawn directly via the D3D11 texture descriptor (the
        // C ABI doesn't declare this struct — use Dawn's C++ wrapper from
        // D3D11Backend.h, which is layout-compatible with WGPUChainedStruct).
        dawn::native::d3d11::SharedTextureMemoryD3D11Texture2DDescriptor d11Desc;
        d11Desc.texture = d3d11Tex;
        WGPUSharedTextureMemoryDescriptor smDesc{};
        smDesc.nextInChain =
            const_cast<WGPUChainedStruct*>(
                reinterpret_cast<const WGPUChainedStruct*>(
                    static_cast<const wgpu::ChainedStruct*>(&d11Desc)));
        mem = wgpuDeviceImportSharedTextureMemory(device, &smDesc);
        if (!mem) {
            LOG_ERROR("[minigpu_external] importSharedTextureMemory(D3D11Texture2D) failed.");
            return nullptr;
        }
        LOG_INFO("[minigpu_external] SharedOutputTexture (D3D11 single-device) d11=%p mem=%p",
            (void*)d3d11Tex.Get(), (void*)mem);

        // Get the legacy DXGI shared HANDLE so other D3D11 devices (e.g.
        // Flutter's ANGLE D3D11 device) can OpenSharedResource() it.  This
        // works because we created the texture with D3D11_RESOURCE_MISC_SHARED.
        // The handle is owned by the texture — do NOT CloseHandle() it.
        Microsoft::WRL::ComPtr<IDXGIResource> dxgiRes;
        if (SUCCEEDED(d3d11Tex.As(&dxgiRes)) && dxgiRes) {
            HANDLE shareH = nullptr;
            HRESULT shr = dxgiRes->GetSharedHandle(&shareH);
            if (SUCCEEDED(shr) && shareH) {
                ntHandle = shareH;
                LOG_DEBUG("[minigpu_external] SharedOutputTexture share handle=%p", (void*)shareH);
            } else {
                LOG_WARN("[minigpu_external] GetSharedHandle failed: 0x%08lX", (unsigned long)shr);
            }
        }
    } else {
        LOG_ERROR("[minigpu_external] create_shared_output_texture: Dawn is not on D3D11 backend; cross-API path not implemented in this build. Set MGPU_BACKEND=d3d11.");
        return nullptr;
    }
    WGPUTextureDescriptor td{};
    // The shared d3d11 texture only needs CopyDst/CopySrc/TextureBinding —
    // compute writes go through the private intermediate texture below.
    td.usage         = WGPUTextureUsage_CopyDst |
                       WGPUTextureUsage_CopySrc |
                       WGPUTextureUsage_TextureBinding;
    td.dimension     = WGPUTextureDimension_2D;
    td.size          = {w, h, 1};
    td.format        = WGPUTextureFormat_BGRA8Unorm;
    td.mipLevelCount = 1;
    td.sampleCount   = 1;

    WGPUTexture wgpuTex = wgpuSharedTextureMemoryCreateTexture(mem, &td);
    if (!wgpuTex) {
        LOG_ERROR("[minigpu_external] wgpuSharedTextureMemoryCreateTexture failed.");
        wgpuSharedTextureMemoryRelease(mem);
        CloseHandle(ntHandle);
        return nullptr;
    }

    WGPUTextureViewDescriptor vd{};
    vd.format          = WGPUTextureFormat_BGRA8Unorm;
    vd.dimension       = WGPUTextureViewDimension_2D;
    vd.baseMipLevel    = 0;
    vd.mipLevelCount   = 1;
    vd.baseArrayLayer  = 0;
    vd.arrayLayerCount = 1;
    vd.aspect          = WGPUTextureAspect_All;
    WGPUTextureView view = wgpuTextureCreateView(wgpuTex, &vd);

    // Create a private Dawn-only texture for UAV compute writes.
    // D3D11_RESOURCE_MISC_SHARED is incompatible with D3D11_BIND_UNORDERED_ACCESS,
    // so compute writes land here first; a CopyTextureToTexture in
    // mgpuCopyBufferToSharedOutputTexture then blits the result into the
    // shared texture before EndAccess.
    WGPUTextureDescriptor priv_td{};
    priv_td.usage         = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc;
    priv_td.dimension     = WGPUTextureDimension_2D;
    priv_td.size          = {w, h, 1};
    priv_td.format        = WGPUTextureFormat_BGRA8Unorm;
    priv_td.mipLevelCount = 1;
    priv_td.sampleCount   = 1;
    WGPUTexture priv_tex = wgpuDeviceCreateTexture(device, &priv_td);
    if (!priv_tex) {
        LOG_ERROR("[minigpu_external] wgpuDeviceCreateTexture(intermediate UAV) failed.");
        wgpuTextureViewRelease(view);
        wgpuTextureRelease(wgpuTex);
        wgpuSharedTextureMemoryRelease(mem);
        return nullptr;
    }
    WGPUTextureView priv_view = wgpuTextureCreateView(priv_tex, &vd);

    auto* out = new MGPUSharedOutputTexture();
    out->d3d11_texture       = d3d11Tex;
    out->nt_handle           = ntHandle;
    out->shared_mem          = mem;
    out->texture             = wgpuTex;
    out->view                = view;
    out->intermediate_texture= priv_tex;
    out->intermediate_view   = priv_view;
    out->width               = w;
    out->height              = h;

    return out;
}

#endif // _WIN32

extern "C" {

EXPORT MGPUSharedOutputTexture*
mgpuCreateSharedOutputTexture(uint32_t width, uint32_t height) {
#ifdef _WIN32
    return create_shared_output_texture(width, height);
#else
    (void)width; (void)height;
    return nullptr;
#endif
}

EXPORT void* mgpuSharedOutputTextureGetD3D11Handle(MGPUSharedOutputTexture* tex) {
#ifdef _WIN32
    return tex ? (void*)tex->nt_handle : nullptr;
#else
    (void)tex;
    return nullptr;
#endif
}

// Returns the underlying ID3D11Texture2D* for the shared output texture.
// The texture lives on the same ID3D11Device returned by
// mgpuCreateD3D11DeviceOnDawnAdapter(), so the encoder can use this pointer
// directly (no OpenSharedResource1 needed). Caller must NOT Release this
// pointer — its lifetime is owned by the MGPUSharedOutputTexture.
EXPORT void* mgpuSharedOutputTextureGetD3D11Texture(MGPUSharedOutputTexture* tex) {
#ifdef _WIN32
    return tex ? (void*)tex->d3d11_texture.Get() : nullptr;
#else
    (void)tex;
    return nullptr;
#endif
}

EXPORT uint32_t mgpuSharedOutputTextureGetWidth(MGPUSharedOutputTexture* tex) {
#ifdef _WIN32
    return tex ? tex->width : 0;
#else
    (void)tex;
    return 0;
#endif
}

EXPORT uint32_t mgpuSharedOutputTextureGetHeight(MGPUSharedOutputTexture* tex) {
#ifdef _WIN32
    return tex ? tex->height : 0;
#else
    (void)tex;
    return 0;
#endif
}

// Returns an AddRef'd ID3D11Device* on the SAME DXGI adapter as Dawn's
// D3D12 device. The same cached device is used internally to back any
// MGPUSharedOutputTexture we create, so passing this pointer to the
// FFmpeg encoder lets the encoder use shared output textures directly
// without OpenSharedResource1. Caller is responsible for Release()
// (or handing ownership to FFmpeg via av_hwdevice_ctx_init).
EXPORT void* mgpuCreateD3D11DeviceOnDawnAdapter() {
#ifdef _WIN32
    ID3D11Device* dev = get_or_create_d3d11_device_on_dawn_adapter();
    if (!dev) return nullptr;
    dev->AddRef();
    return static_cast<void*>(dev);
#else
    return nullptr;
#endif
}

EXPORT int mgpuVideoTextureBGRAToRGBASharedOutput(MGPUVideoTexture* src,
                                                  MGPUSharedOutputTexture* dst) {
#ifdef _WIN32
    if (!src || !dst) return 0;
    if (src->num_planes == 0 || !src->views[0]) return 0;
    if (!dst->shared_mem || !dst->texture || !dst->view) return 0;

    WGPUDevice device = get_device();
    WGPUQueue  queue  = get_queue();
    if (!device || !queue) return 0;

    if (!ensure_shared_output_pipeline(dst, device)) {
        LOG_ERROR("[minigpu_external] ensure_shared_output_pipeline failed.");
        return 0;
    }

    // Acquire GPU access on the shared destination.  D3D12 resource state
    // becomes UNORDERED_ACCESS for the duration of the dispatch.
    WGPUSharedTextureMemoryBeginAccessDescriptor beginDesc{};
    beginDesc.initialized = true; // assume previous content (or zero) is valid
    beginDesc.fenceCount  = 0;
    wgpuSharedTextureMemoryBeginAccess(dst->shared_mem, dst->texture, &beginDesc);

    // Build per-dispatch bind group (textures may differ each frame).
    WGPUBindGroupEntry bgEntries[2]{};
    bgEntries[0].binding     = 0;
    bgEntries[0].textureView = src->views[0];
    bgEntries[1].binding     = 1;
    // Compute writes into the private intermediate texture (StorageBinding);
    // a CopyTextureToTexture below copies the result into the shared texture
    // so the D3D11 client sees it after EndAccess.  The shared texture
    // itself is created without StorageBinding (D3D11_RESOURCE_MISC_SHARED
    // is incompatible with D3D11_BIND_UNORDERED_ACCESS), so it cannot be
    // bound as a write storage texture directly.
    bgEntries[1].textureView = dst->intermediate_view
                               ? dst->intermediate_view
                               : dst->view;

    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout     = dst->cs_bgl;
    bgDesc.entryCount = 2;
    bgDesc.entries    = bgEntries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, nullptr);
    WGPUComputePassEncoder pass =
        wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, dst->cs_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    uint32_t gx = (dst->width  + 7) / 8;
    uint32_t gy = (dst->height + 7) / 8;
    wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    // Copy from the private UAV intermediate texture into the shared texture
    // so the D3D11 client (Flutter / encoder) sees the result after
    // EndAccess.  Skip only if intermediate_texture is unavailable (legacy
    // path); in that case the dispatch above would have failed validation
    // already because dst->view lacks StorageBinding usage.
    if (dst->intermediate_texture) {
        WGPUTexelCopyTextureInfo copy_src{};
        copy_src.texture  = dst->intermediate_texture;
        copy_src.mipLevel = 0;
        copy_src.aspect   = WGPUTextureAspect_All;
        WGPUTexelCopyTextureInfo copy_dst{};
        copy_dst.texture  = dst->texture;
        copy_dst.mipLevel = 0;
        copy_dst.aspect   = WGPUTextureAspect_All;
        WGPUExtent3D copy_size{dst->width, dst->height, 1};
        wgpuCommandEncoderCopyTextureToTexture(enc, &copy_src, &copy_dst, &copy_size);
    }

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuCommandEncoderRelease(enc);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuBindGroupRelease(bg);

    // EndAccess returns the resource to COMMON state so the D3D11 client
    // (the FFmpeg encoder / Flutter compositor) may read it.  In the single-
    // device path Dawn IS the D3D11 device that the encoder also uses (the
    // device is ID3D11Multithread-protected), so GPU command ordering is
    // guaranteed by the device's immediate context: any subsequent encode
    // call serializes after the dispatch+copy we just submitted.  No CPU
    // wait is needed.
    WGPUSharedTextureMemoryEndAccessState endState{};
    wgpuSharedTextureMemoryEndAccess(dst->shared_mem, dst->texture, &endState);

    // Drain any pending Dawn callbacks (device-lost, validation errors from
    // prior submissions, etc.) without blocking.  This keeps Dawn's event
    // queue from growing unbounded while letting the caller return
    // immediately and the GPU run the work asynchronously.
    pump_dawn_events_nonblocking();
    return 1;
#else
    (void)src; (void)dst;
    return 0;
#endif
}

/// Copy an RGBA8 GPU buffer (output from a GpuEffect dispatch) into the
/// shared output texture.  The buffer must hold exactly width*height u32
/// pixels packed as RGBA8 (same layout the GpuEffect WGSL produces).
/// Returns 1 on success, 0 on failure.
EXPORT int mgpuCopyBufferToSharedOutputTexture(
        MGPUBuffer*                buf,
        MGPUSharedOutputTexture*   dst) {
#ifdef _WIN32
    if (!buf || !dst) return 0;
    if (!dst->shared_mem || !dst->texture || !dst->view) return 0;

    WGPUDevice device = get_device();
    WGPUQueue  queue  = get_queue();
    if (!device || !queue) return 0;

    // Extract the underlying WGPUBuffer from the opaque MGPUBuffer handle.
    mgpu::Buffer* srcBuf = reinterpret_cast<mgpu::Buffer*>(buf);
    WGPUBuffer wgpuBuf = srcBuf->bufferData.buffer;
    size_t     bufSize = srcBuf->bufferData.size;
    if (!wgpuBuf || bufSize == 0) return 0;

    if (!ensure_copy_pipeline(dst, device)) {
        LOG_ERROR("[minigpu_external] ensure_copy_pipeline failed.");
        return 0;
    }

    // Acquire GPU access on the destination shared texture.
    WGPUSharedTextureMemoryBeginAccessDescriptor beginDesc{};
    beginDesc.initialized = true;
    beginDesc.fenceCount  = 0;
    wgpuSharedTextureMemoryBeginAccess(dst->shared_mem, dst->texture, &beginDesc);

    WGPUBindGroupEntry bgEntries[2]{};
    bgEntries[0].binding = 0;
    bgEntries[0].buffer  = wgpuBuf;
    bgEntries[0].offset  = 0;
    bgEntries[0].size    = bufSize;
    bgEntries[1].binding     = 1;
    // Compute writes into the private intermediate texture (StorageBinding);
    // a CopyTextureToTexture below copies the result into the shared texture.
    bgEntries[1].textureView = dst->intermediate_view
                               ? dst->intermediate_view
                               : dst->view;

    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout     = dst->copy_bgl;
    bgDesc.entryCount = 2;
    bgDesc.entries    = bgEntries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

    WGPUCommandEncoder enc  = wgpuDeviceCreateCommandEncoder(device, nullptr);
    WGPUComputePassEncoder pass =
        wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, dst->copy_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    uint32_t gx = (dst->width  + 7) / 8;
    uint32_t gy = (dst->height + 7) / 8;
    wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    // Copy from the private UAV intermediate texture into the shared texture
    // so the D3D11 client (Flutter) sees the result after EndAccess.
    if (dst->intermediate_texture) {
        WGPUTexelCopyTextureInfo copy_src{};
        copy_src.texture  = dst->intermediate_texture;
        copy_src.mipLevel = 0;
        copy_src.aspect   = WGPUTextureAspect_All;
        WGPUTexelCopyTextureInfo copy_dst{};
        copy_dst.texture  = dst->texture;
        copy_dst.mipLevel = 0;
        copy_dst.aspect   = WGPUTextureAspect_All;
        WGPUExtent3D copy_size{dst->width, dst->height, 1};
        wgpuCommandEncoderCopyTextureToTexture(enc, &copy_src, &copy_dst, &copy_size);
    }

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuCommandEncoderRelease(enc);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuBindGroupRelease(bg);

    // EndAccess returns the resource to COMMON so the D3D11 client can read it.
    WGPUSharedTextureMemoryEndAccessState endState{};
    wgpuSharedTextureMemoryEndAccess(dst->shared_mem, dst->texture, &endState);

    // CROSS-DEVICE PRESENT SYNC (ghost fix): the shared texture is legacy
    // D3D11_RESOURCE_MISC_SHARED (no keyed mutex), and the consumer (Flutter)
    // samples it on its OWN GPU device the moment MarkTextureFrameAvailable
    // fires. With no fence exported, a consumer read could race this producer
    // copy: under motion the compose+deblock+copy take longer, so Flutter read
    // a HALF-WRITTEN texture = torn/blended frames ("GPU ghosting at higher
    // bandwidth"; on a still image producer+consumer settle and it looks
    // clean). BLOCK until the copy actually completes on the producer GPU
    // before returning, so by the time the host hands the handle to Flutter the
    // pixels are final. Timeout-protected (a hung driver errors, never freezes
    // the caller). Cost: a sub-millisecond-to-few-ms producer stall per frame;
    // a keyed-mutex / double-buffer would remove even that, but this is the
    // certain fix that does not depend on the consumer honoring a fence.
    {
        std::promise<void> presentDone;
        auto presentFut = presentDone.get_future();
        WGPUQueueWorkDoneCallbackInfo cbInfo{};
        cbInfo.mode      = WGPUCallbackMode_AllowProcessEvents;
        cbInfo.callback  = [](WGPUQueueWorkDoneStatus, WGPUStringView,
                              void* ud1, void*) {
            static_cast<std::promise<void>*>(ud1)->set_value();
        };
        cbInfo.userdata1 = &presentDone;
        wgpuQueueOnSubmittedWorkDone(queue, cbInfo);
        drain_dawn_events_with_timeout(presentFut,
                                       "copyBufToShTex: present copy");
    }
    return 1;
#else
    (void)buf; (void)dst;
    return 0;
#endif
}

/// Async variant of mgpuCopyBufferToSharedOutputTexture. Runs the whole copy
/// (compute pass + cross-device present sync) on the WebGPU worker thread — the
/// same thread the dispatch path already uses — and invokes [callback] with the
/// result (1 = success, 0 = failure) once the GPU work has actually completed.
/// This moves the present-wait busy-poll OFF the calling (Dart) thread: the
/// caller awaits a Completer instead of blocking the isolate for the copy's
/// duration. The present-sync ordering (and thus the ghost/tearing fix) is
/// preserved because the work — including the OnSubmittedWorkDone wait — still
/// completes before the callback fires.
EXPORT void mgpuCopyBufferToSharedOutputTextureAsync(
        MGPUBuffer*                buf,
        MGPUSharedOutputTexture*   dst,
        void                     (*callback)(int)) {
#ifdef _WIN32
    minigpu.getWebGPUThread().enqueueAsync([buf, dst, callback]() {
        int r = mgpuCopyBufferToSharedOutputTexture(buf, dst);
        if (callback) callback(r);
    });
#else
    (void)buf; (void)dst;
    if (callback) callback(0);
#endif
}

/// Async variant of mgpuVideoTextureBGRAToRGBASharedOutput — same worker-thread
/// + Completer treatment as mgpuCopyBufferToSharedOutputTextureAsync above.
EXPORT void mgpuVideoTextureBGRAToRGBASharedOutputAsync(
        MGPUVideoTexture*          src,
        MGPUSharedOutputTexture*   dst,
        void                     (*callback)(int)) {
#ifdef _WIN32
    minigpu.getWebGPUThread().enqueueAsync([src, dst, callback]() {
        int r = mgpuVideoTextureBGRAToRGBASharedOutput(src, dst);
        if (callback) callback(r);
    });
#else
    (void)src; (void)dst;
    if (callback) callback(0);
#endif
}

/// Copy a GPU buffer of f32 RGBA pixels (4 floats per pixel, R,G,B,A in
/// [0,1]) into the shared output texture.  Used by visualizers like the
/// spectrogram which produce float colors directly into a tensor buffer.
/// Returns 1 on success, 0 on failure.
EXPORT int mgpuCopyBufferF32ToSharedOutputTexture(
        MGPUBuffer*                buf,
        MGPUSharedOutputTexture*   dst) {
#ifdef _WIN32
    if (!buf || !dst) return 0;
    if (!dst->shared_mem || !dst->texture || !dst->view) return 0;

    WGPUDevice device = get_device();
    WGPUQueue  queue  = get_queue();
    if (!device || !queue) return 0;

    mgpu::Buffer* srcBuf = reinterpret_cast<mgpu::Buffer*>(buf);
    WGPUBuffer wgpuBuf = srcBuf->bufferData.buffer;
    size_t     bufSize = srcBuf->bufferData.size;
    if (!wgpuBuf || bufSize == 0) return 0;

    if (!ensure_copy_pipeline_f32(dst, device)) {
        LOG_ERROR("[minigpu_external] ensure_copy_pipeline_f32 failed.");
        return 0;
    }

    WGPUSharedTextureMemoryBeginAccessDescriptor beginDesc{};
    beginDesc.initialized = true;
    beginDesc.fenceCount  = 0;
    wgpuSharedTextureMemoryBeginAccess(dst->shared_mem, dst->texture, &beginDesc);

    WGPUBindGroupEntry bgEntries[2]{};
    bgEntries[0].binding = 0;
    bgEntries[0].buffer  = wgpuBuf;
    bgEntries[0].offset  = 0;
    bgEntries[0].size    = bufSize;
    bgEntries[1].binding     = 1;
    bgEntries[1].textureView = dst->intermediate_view
                               ? dst->intermediate_view
                               : dst->view;

    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout     = dst->copyf32_bgl;
    bgDesc.entryCount = 2;
    bgDesc.entries    = bgEntries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bgDesc);

    WGPUCommandEncoder enc  = wgpuDeviceCreateCommandEncoder(device, nullptr);
    WGPUComputePassEncoder pass =
        wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, dst->copyf32_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
    uint32_t gx = (dst->width  + 7) / 8;
    uint32_t gy = (dst->height + 7) / 8;
    wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    if (dst->intermediate_texture) {
        WGPUTexelCopyTextureInfo copy_src{};
        copy_src.texture  = dst->intermediate_texture;
        copy_src.mipLevel = 0;
        copy_src.aspect   = WGPUTextureAspect_All;
        WGPUTexelCopyTextureInfo copy_dst{};
        copy_dst.texture  = dst->texture;
        copy_dst.mipLevel = 0;
        copy_dst.aspect   = WGPUTextureAspect_All;
        WGPUExtent3D copy_size{dst->width, dst->height, 1};
        wgpuCommandEncoderCopyTextureToTexture(enc, &copy_src, &copy_dst, &copy_size);
    }

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuCommandEncoderRelease(enc);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuBindGroupRelease(bg);

    WGPUSharedTextureMemoryEndAccessState endState{};
    wgpuSharedTextureMemoryEndAccess(dst->shared_mem, dst->texture, &endState);

    pump_dawn_events_nonblocking();
    return 1;
#else
    (void)buf; (void)dst;
    return 0;
#endif
}

// Debug helper: read first pixel of the SharedOutputTexture from the D3D11
// side using the cached Dawn-adapter D3D11 device. Returns BGRA8 packed u32.
EXPORT uint32_t mgpuSharedOutputTextureDebugReadFirstPixel(
        MGPUSharedOutputTexture* tex) {
#ifdef _WIN32
    if (!tex || !tex->d3d11_texture) return 0xDEAD0001u;
    ID3D11Device* dev = get_or_create_d3d11_device_on_dawn_adapter();
    if (!dev || !g_d3d11_context) return 0xDEAD0002u;

    Microsoft::WRL::ComPtr<IDXGIKeyedMutex> km;
    if (SUCCEEDED(tex->d3d11_texture.As(&km)) && km) {
        HRESULT ahr = km->AcquireSync(0, 200);
        if (FAILED(ahr)) {
            LOG_ERROR("[minigpu_external] debug AcquireSync failed: 0x%08lX", (unsigned long)ahr);
            return 0xDEAD0003u;
        }
    }

    D3D11_TEXTURE2D_DESC stagingDesc{};
    stagingDesc.Width = 1; stagingDesc.Height = 1;
    stagingDesc.MipLevels = 1; stagingDesc.ArraySize = 1;
    stagingDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    stagingDesc.SampleDesc.Count = 1;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.BindFlags = 0;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    stagingDesc.MiscFlags = 0;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
    HRESULT hr = dev->CreateTexture2D(&stagingDesc, nullptr, &staging);
    if (FAILED(hr) || !staging) {
        if (km) km->ReleaseSync(0);
        return 0xDEAD0004u;
    }

    D3D11_BOX box{};
    box.left = 0; box.right = 1;
    box.top = 0; box.bottom = 1;
    box.front = 0; box.back = 1;
    g_d3d11_context->CopySubresourceRegion(
        staging.Get(), 0, 0, 0, 0,
        tex->d3d11_texture.Get(), 0, &box);
    g_d3d11_context->Flush();

    if (km) km->ReleaseSync(0);

    D3D11_MAPPED_SUBRESOURCE m{};
    hr = g_d3d11_context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &m);
    if (FAILED(hr) || !m.pData) {
        return 0xDEAD0005u;
    }
    uint32_t pix = *static_cast<uint32_t*>(m.pData);
    g_d3d11_context->Unmap(staging.Get(), 0);
    return pix;
#else
    (void)tex;
    return 0xDEAD0006u;
#endif
}

// Debug helper: read first pixel of the SharedOutputTexture from the Dawn
// (D3D12) side via CopyTextureToBuffer + map. Used to compare with the
// D3D11-side readback above to determine which side sees stale data.
EXPORT uint32_t mgpuSharedOutputTextureDebugReadFirstPixelDawn(
        MGPUSharedOutputTexture* tex) {
#ifdef _WIN32
    if (!tex || !tex->shared_mem || !tex->texture) return 0xDEAD1001u;
    WGPUDevice device = get_device();
    WGPUQueue  queue  = get_queue();
    if (!device || !queue) return 0xDEAD1002u;

    // 256-byte aligned bytesPerRow required by WebGPU CopyTextureToBuffer.
    const uint32_t kBytesPerRow = 256;
    WGPUBufferDescriptor bd{};
    bd.usage            = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    bd.size             = kBytesPerRow;
    bd.mappedAtCreation = false;
    WGPUBuffer staging = wgpuDeviceCreateBuffer(device, &bd);
    if (!staging) return 0xDEAD1003u;

    WGPUSharedTextureMemoryBeginAccessDescriptor beginDesc{};
    beginDesc.initialized = true;
    beginDesc.fenceCount  = 0;
    wgpuSharedTextureMemoryBeginAccess(tex->shared_mem, tex->texture, &beginDesc);

    WGPUTexelCopyTextureInfo srcInfo{};
    srcInfo.texture  = tex->texture;
    srcInfo.mipLevel = 0;
    srcInfo.aspect   = WGPUTextureAspect_All;
    srcInfo.origin   = {0, 0, 0};

    WGPUTexelCopyBufferInfo dstInfo{};
    dstInfo.buffer            = staging;
    dstInfo.layout.offset     = 0;
    dstInfo.layout.bytesPerRow = kBytesPerRow;
    dstInfo.layout.rowsPerImage = 1;

    WGPUExtent3D ext{1, 1, 1};

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, nullptr);
    wgpuCommandEncoderCopyTextureToBuffer(enc, &srcInfo, &dstInfo, &ext);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuCommandEncoderRelease(enc);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);

    WGPUSharedTextureMemoryEndAccessState endState{};
    wgpuSharedTextureMemoryEndAccess(tex->shared_mem, tex->texture, &endState);

    // Wait for queue (timeout-protected: a hung driver returns an error
    // instead of freezing the caller forever).
    std::promise<void> done;
    auto fut = done.get_future();
    WGPUQueueWorkDoneCallbackInfo cbInfo{};
    cbInfo.mode      = WGPUCallbackMode_AllowProcessEvents;
    cbInfo.callback  = [](WGPUQueueWorkDoneStatus, WGPUStringView,
                          void* userdata1, void*) {
        static_cast<std::promise<void>*>(userdata1)->set_value();
    };
    cbInfo.userdata1 = &done;
    wgpuQueueOnSubmittedWorkDone(queue, cbInfo);
    if (!drain_dawn_events_with_timeout(
            fut, "DebugReadFirstPixelDawn: queue submit")) {
        wgpuBufferRelease(staging);
        return 0xDEAD1005u;
    }

    // Map.
    std::promise<WGPUMapAsyncStatus> mapDone;
    auto mapFut = mapDone.get_future();
    WGPUBufferMapCallbackInfo mci{};
    mci.mode = WGPUCallbackMode_AllowProcessEvents;
    mci.callback = [](WGPUMapAsyncStatus s, WGPUStringView,
                      void* ud1, void*) {
        static_cast<std::promise<WGPUMapAsyncStatus>*>(ud1)->set_value(s);
    };
    mci.userdata1 = &mapDone;
    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, kBytesPerRow, mci);
    if (!drain_dawn_events_with_timeout(
            mapFut, "DebugReadFirstPixelDawn: buffer map")) {
        wgpuBufferRelease(staging);
        return 0xDEAD1006u;
    }
    if (mapFut.get() != WGPUMapAsyncStatus_Success) {
        wgpuBufferRelease(staging);
        return 0xDEAD1004u;
    }
    const uint32_t* data = static_cast<const uint32_t*>(
        wgpuBufferGetConstMappedRange(staging, 0, 4));
    uint32_t pix = data ? *data : 0xDEAD1005u;
    wgpuBufferUnmap(staging);
    wgpuBufferRelease(staging);
    return pix;
#else
    (void)tex;
    return 0xDEAD1006u;
#endif
}

EXPORT void mgpuDestroySharedOutputTexture(MGPUSharedOutputTexture* tex) {
#ifdef _WIN32
    if (!tex) return;
    if (tex->copy_pipeline) wgpuComputePipelineRelease(tex->copy_pipeline);
    if (tex->copy_pl)       wgpuPipelineLayoutRelease(tex->copy_pl);
    if (tex->copy_bgl)      wgpuBindGroupLayoutRelease(tex->copy_bgl);
    if (tex->copy_module)   wgpuShaderModuleRelease(tex->copy_module);
    if (tex->cs_pipeline) wgpuComputePipelineRelease(tex->cs_pipeline);
    if (tex->cs_pl)       wgpuPipelineLayoutRelease(tex->cs_pl);
    if (tex->cs_bgl)      wgpuBindGroupLayoutRelease(tex->cs_bgl);
    if (tex->cs_module)   wgpuShaderModuleRelease(tex->cs_module);
    if (tex->intermediate_view)    wgpuTextureViewRelease(tex->intermediate_view);
    if (tex->intermediate_texture) wgpuTextureRelease(tex->intermediate_texture);
    if (tex->copyf32_pipeline) wgpuComputePipelineRelease(tex->copyf32_pipeline);
    if (tex->copyf32_pl)       wgpuPipelineLayoutRelease(tex->copyf32_pl);
    if (tex->copyf32_bgl)      wgpuBindGroupLayoutRelease(tex->copyf32_bgl);
    if (tex->copyf32_module)   wgpuShaderModuleRelease(tex->copyf32_module);
    if (tex->view)        wgpuTextureViewRelease(tex->view);
    if (tex->texture)     wgpuTextureRelease(tex->texture);
    if (tex->shared_mem)  wgpuSharedTextureMemoryRelease(tex->shared_mem);
    // Note: nt_handle here is a legacy DXGI share handle obtained via
    // IDXGIResource::GetSharedHandle().  Per MSDN, legacy share handles are
    // OWNED BY THE TEXTURE and must NOT be CloseHandle()'d.
    delete tex;
#else
    (void)tex;
#endif
}

/* =========================================================================
 * Web-only: expose Emscripten WebGPU integer handles so Dart can look up
 * the underlying JS GPUDevice / GPUBuffer objects via WebGPU.getJsObject().
 * ====================================================================== */
#ifdef __EMSCRIPTEN__

/// Returns the WGPUDevice integer handle (index into WebGPU.Internals.jsObjects).
/// In Dart: WebGPU.getJsObject(handle) → JS GPUDevice.
EXPORT uint32_t mgpuGetWGPUDeviceHandle() {
    WGPUDevice device = get_device();
    return device ? static_cast<uint32_t>(reinterpret_cast<uintptr_t>(device)) : 0u;
}

/// Returns the WGPUBuffer integer handle for a given MGPUBuffer pointer.
/// In Dart: WebGPU.getJsObject(handle) → JS GPUBuffer.
EXPORT uint32_t mgpuGetWGPUBufferHandle(MGPUBuffer* buf_c) {
    if (!buf_c) return 0u;
    auto* buf = reinterpret_cast<mgpu::Buffer*>(buf_c);
    return buf->bufferData.buffer
        ? static_cast<uint32_t>(reinterpret_cast<uintptr_t>(buf->bufferData.buffer))
        : 0u;
}

#endif // __EMSCRIPTEN__

} // extern "C"
