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
#include "../include/dart_port.h"
#include "../include/log.h"
#include "../include/wait_prof.h"

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
#include <cstdlib>
#include <thread>
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
 * drain_dawn_events_with_timeout() waits for a Dawn callback and bails
 * out with a logged warning after a configurable deadline so a hung GPU
 * surfaces as a recoverable encode error instead of a frozen app.
 * pump_dawn_events_nonblocking() is for the async encode path where we
 * just need to give Dawn a chance to fire pending callbacks without
 * actually waiting for anything.
 *
 * ── "1 ms" WAS REALLY ~15.6 ms, AND IT WAS THE WHOLE PRESENT COST ─────────
 * The previous body was `while (fut.wait_for(milliseconds(1)) != ready)`.
 * That looks like a 1 ms poll. It is not. NOTHING ELSE notifies the future:
 * the promise is set from a Dawn callback that only runs inside the
 * wgpuInstanceProcessEvents() call *further down this same loop body*, so
 * wait_for can never be woken early — it always burns its FULL timeout, and
 * on Windows a 1 ms condition-variable timeout is rounded up to the system
 * timer quantum, ~15.6 ms unless some process in the session has raised the
 * resolution. Every call therefore cost one whole tick.
 *
 * Measured (RTX 4090, a GPU video decode presented through a shared texture):
 * GPU decode + shared-texture
 * present p50 = 15.69 ms at 1280x720 AND 15.69 ms at 3840x2160 — bit-for-bit
 * the same number at 9x the pixels, while the same decode WITHOUT the present
 * costs 3.14 / 12.61 ms. A cost that is invariant in the work is not the work;
 * it is a clock tick. Buffer::readDirect already learned this exact lesson
 * (see the hot-poll comment in buffer.cpp) — the present path never got it.
 *
 * Fix: check the future with a ZERO timeout (never sleeps) and hand the wait
 * a bounded yield-spin. Real GPU completions land in microseconds-to-a-few-ms,
 * so they now return as soon as the callback fires. Only a wait that outlives
 * the spin budget (a cold shader compile, a wedged driver) degrades to the old
 * coarse sleep — which is exactly the case the 2-problem note above wanted to
 * stop pegging a core for. Set MGPU_DRAIN_SPIN_MS=0 to restore the pre-fix
 * behaviour exactly (A/B escape); the default budget is 8 ms.
 * ====================================================================== */

// Bounded yield-spin budget, in milliseconds, before falling back to coarse
// sleeping. Env-overridable so the fix can be priced (0 == old behaviour).
//
// The override is parsed STRICTLY, and that is the whole point: this used to be
// `std::atoi`, which returns 0 for any non-numeric string. `MGPU_DRAIN_SPIN_MS`
// set to "on", "true", "default" or a typo therefore selected budget 0 — the
// pathological pre-fix path, a ~15.6 ms timer quantum on EVERY GPU wait — with
// no diagnostic whatsoever. A knob whose failure mode is "silently 6x slower"
// is a trap; only a fully-consumed, in-range, non-negative integer is honoured
// now, anything else warns and keeps the default, and an explicit 0 says so out
// loud because it is a measurement mode, not a setting.
static int drain_spin_budget_ms() {
    static int cached = -1;
    if (cached < 0) {
        constexpr int kDefaultBudgetMs = 8;
        constexpr long kMaxBudgetMs    = 1000;  // beyond this it is a hang, not a spin
        cached = kDefaultBudgetMs;
        const char* s = std::getenv("MGPU_DRAIN_SPIN_MS");
        if (s && s[0]) {
            char* end = nullptr;
            const long v = std::strtol(s, &end, 10);
            if (end && *end == '\0' && v >= 0 && v <= kMaxBudgetMs) {
                cached = static_cast<int>(v);
                if (cached == 0) {
                    LOG_ERROR("[minigpu_external] MGPU_DRAIN_SPIN_MS=0 — the event "
                              "drain is reverted to its PRE-FIX behaviour and every "
                              "GPU wait will cost a ~15.6 ms timer quantum. This is "
                              "an A/B measurement mode; never ship with it set.");
                }
            } else {
                LOG_ERROR("[minigpu_external] MGPU_DRAIN_SPIN_MS=\"%s\" is not an "
                          "integer in [0, %ld] — ignoring it and using the default "
                          "%d ms. (It previously parsed as 0, silently restoring the "
                          "pre-fix timer-quantum stall.)",
                          s, kMaxBudgetMs, kDefaultBudgetMs);
            }
        }
    }
    return cached;
}

static inline void drain_yield() {
#ifdef _WIN32
    // Sleep(0) yields the remainder of the timeslice to any ready thread of
    // equal priority and returns immediately otherwise — no timer quantum
    // involved. Same primitive Buffer::readDirect's hot-poll settled on.
    Sleep(0);
#else
    std::this_thread::yield();
#endif
}

template <typename FutureT>
static bool drain_dawn_events_with_timeout(
        FutureT&             fut,
        const char*          op,
        int                  timeout_ms = 5000) {
    using namespace std::chrono;
    mgpu::WaitScope wpDrain(mgpu::WP_DRAIN);
    const auto start    = steady_clock::now();
    const auto deadline = start + milliseconds(timeout_ms);
    const auto spinEnd  = start + milliseconds(drain_spin_budget_ms());
    for (;;) {
        ++wpDrain.iters;
        // Fire any Dawn callbacks that are ready; this is what sets the promise.
        wgpuInstanceProcessEvents(minigpu.getInstance());
        // Zero-timeout probe: never sleeps, so a callback that just fired is
        // observed immediately instead of after a full timer tick.
        if (fut.wait_for(milliseconds(0)) == std::future_status::ready) break;
        const auto now = steady_clock::now();
        if (now >= deadline) {
            LOG_ERROR("[minigpu_external] %s: GPU work did not complete "
                      "within %d ms; treating as driver hang and bailing out.",
                      op, timeout_ms);
            return false;
        }
        if (now < spinEnd) {
            drain_yield();          // hot path: microseconds of latency
        } else {
            // Long wait (cold pipeline compile / driver trouble): stop burning
            // the core. This sleep is coarse and that is now deliberate.
            fut.wait_for(milliseconds(1));
        }
    }
    // One final drain so callbacks chained after our promise also fire.
    wgpuInstanceProcessEvents(minigpu.getInstance());
    return true;
}

static inline void pump_dawn_events_nonblocking() {
    wgpuInstanceProcessEvents(minigpu.getInstance());
}

// Introspection (also the freshness marker for a rebuilt artifact): which
// drain policy is compiled/configured into THIS binary.
EXPORT int mgpuDrainSpinBudgetMs(void) { return drain_spin_budget_ms(); }

// TEST-ONLY ESCAPE. MGPU_PRESENT_NO_WAIT=1 removes the shared-texture
// present's completion wait, reproducing the pre-ghost-fix producer/consumer
// race on purpose. See the call site for why it exists.
static bool present_wait_disabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* s = std::getenv("MGPU_PRESENT_NO_WAIT");
        cached = (s && s[0] && s[0] != '0') ? 1 : 0;
        if (cached) {
            LOG_WARN("[minigpu_external] MGPU_PRESENT_NO_WAIT=1: shared-texture "
                     "present will NOT wait for its copy to complete. This is a "
                     "test-only broken mode (torn/stale frames are expected).");
        }
    }
    return cached != 0;
}

static bool on_webgpu_worker() {
#ifdef __EMSCRIPTEN__
    return true; // enqueueAsync runs inline on the one thread there is
#else
    return std::this_thread::get_id() == minigpu.getWebGPUThread().threadId();
#endif
}

// Runs [fn] on the WebGPU worker so every Dawn use here is serialized behind
// the queued dispatches (FIFO). Inline when already on the worker (the async
// entry points): enqueueSync from the worker would deadlock. A throwing
// device lookup (lost / quarantined context) is a failed call, never a crash.
template <typename Fn>
static int run_on_webgpu_worker(Fn&& fn) {
    try {
        if (!on_webgpu_worker()) {
            return minigpu.getWebGPUThread().enqueueSync<int>(
                std::function<int()>(std::forward<Fn>(fn)));
        }
        return fn();
    } catch (...) {
        return 0;
    }
}

// Queues [task] on the WebGPU worker. False when it was not queued (the worker
// has exited at process teardown, or out of memory): the caller reports the
// failure itself, since no exception may leave an export.
template <typename Task>
static bool post_to_webgpu_worker(Task&& task) {
    try {
        minigpu.getWebGPUThread().enqueueAsync(std::forward<Task>(task));
        return true;
    } catch (...) {
        return false;
    }
}

// Blocks until everything submitted to [queue] so far has finished on the
// GPU (timeout-guarded). The completion state is heap-owned by the callback,
// so a wait that times out leaves a late callback nothing dangling to touch.
static bool wait_submitted_work(WGPUQueue queue, const char* op) {
    using Done = std::shared_ptr<std::promise<WGPUQueueWorkDoneStatus>>;
    auto done = std::make_shared<std::promise<WGPUQueueWorkDoneStatus>>();
    auto fut  = done->get_future();
    WGPUQueueWorkDoneCallbackInfo cbInfo{};
    cbInfo.mode      = WGPUCallbackMode_AllowProcessEvents;
    cbInfo.callback  = [](WGPUQueueWorkDoneStatus status, WGPUStringView,
                          void* ud1, void*) {
        std::unique_ptr<Done> owner(static_cast<Done*>(ud1));
        (*owner)->set_value(status);
    };
    cbInfo.userdata1 = new Done(done);
    wgpuQueueOnSubmittedWorkDone(queue, cbInfo);
    if (!drain_dawn_events_with_timeout(fut, op)) return false;
    const WGPUQueueWorkDoneStatus status = fut.get();
    if (status != WGPUQueueWorkDoneStatus_Success) {
        LOG_ERROR("[minigpu_external] %s: queue work-done status %d.",
                  op, (int)status);
        return false;
    }
    return true;
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
#ifdef _WIN32
    // A Tier B frame's ring slot: set free again once this frame is destroyed
    // and Dawn's queue has finished everything submitted up to then.
    std::shared_ptr<std::atomic<bool>> tier_b_free;
#endif

    // Dimensions
    uint32_t width  = 0;
    uint32_t height = 0;

    MGPUExternalPixelFormat pixel_format = MGPU_EXTERNAL_PIXEL_FORMAT_UNKNOWN;
    MGPUExternalContentType content_type = MGPU_EXTERNAL_CONTENT_TYPE_CPU;

    // How toRGBA reads the samples: an MGPUYuvMatrix, plus kYuvFullRange when
    // they are full range. One atomic, so a conversion snapshots both halves
    // of one mgpuVideoTextureSetYuvColor call.
    static constexpr int kYuvFullRange = 0x100;
    std::atomic<int> yuv_color{MGPU_YUV_MATRIX_BT709 | kYuvFullRange};
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
    // Every plane maps to a core WebGPU format (R8, RG8, RGBA8, BGRA8,
    // RGBA16F) that any device can sample and copy into, so the mapping is
    // the whole answer; a probe texture could not fail anyway (creation
    // returns an error object, never null).
    int n = num_planes_for_format(fmt);
    WGPUDevice device = get_device();
    if (!device) return 0;

    for (int i = 0; i < n; ++i) {
        if (map_format(fmt, (uint32_t)i) == WGPUTextureFormat_Undefined) return 0;
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
    if (wgpuSharedTextureMemoryGetProperties(mem, &props) != WGPUStatus_Success) {
        wgpuSharedTextureMemoryRelease(mem);
        return nullptr;
    }

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
    if (wgpuSharedTextureMemoryBeginAccess(mem, texture, &beginDesc) != WGPUStatus_Success) {
        wgpuTextureRelease(texture);
        wgpuSharedTextureMemoryRelease(mem);
        return nullptr;
    }

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

// Closes an access opened by a SUCCESSFUL BeginAccess. The end state owns the
// exported fences and must be freed even when nobody consumes them.
static bool end_shared_access(WGPUSharedTextureMemory mem, WGPUTexture texture) {
    WGPUSharedTextureMemoryEndAccessState endState{};
    const WGPUStatus status =
        wgpuSharedTextureMemoryEndAccess(mem, texture, &endState);
    wgpuSharedTextureMemoryEndAccessStateFreeMembers(endState);
    return status == WGPUStatus_Success;
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

// One cross-adapter texture of a producer's Tier B ring. An allocated slot
// has [free]; the frame copied into it holds the same flag (tier_b_free).
struct TierBSlot {
    Microsoft::WRL::ComPtr<ID3D12Resource>  b12CrossTex;     // committed cross-adapter tex
    Microsoft::WRL::ComPtr<ID3D11Texture2D> b11on12Wrapped;  // D3D11on12 view of b12CrossTex
    WGPUSharedTextureMemory                 dawnCrossMem = nullptr; // Dawn shared mem
    std::shared_ptr<std::atomic<bool>>      free;
};
static constexpr int kTierBSlots = 3;

struct ProducerAdapterInfo {
    LUID                                        luid{};
    Microsoft::WRL::ComPtr<IDXGIAdapter1>       adapter;

    // Tier C: plain D3D11 device on producer adapter (for staging / CPU bridge).
    Microsoft::WRL::ComPtr<ID3D11Device>        device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
    // Staging copy reused while the source description matches; the mutex
    // keeps one copy -> Map -> upload in flight per producer.
    std::mutex                                  cpuBridgeMutex;
    Microsoft::WRL::ComPtr<ID3D11Texture2D>     cpuStaging;

    // ── Tier B: D3D12 cross-adapter bridge ───────────────────────────────
    // Only initialised when Dawn is running the D3D12 backend AND the
    // producer adapter reports CrossAdapterRowMajorTextureSupported.
    Microsoft::WRL::ComPtr<ID3D12Device>        b12Dev;       // D3D12 on producer adapter
    Microsoft::WRL::ComPtr<ID3D12CommandQueue>  b12Queue;     // direct command queue
    Microsoft::WRL::ComPtr<ID3D11Device>        b11on12Dev;   // D3D11on12 wrapping b12Dev
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> b11on12Ctx;
    Microsoft::WRL::ComPtr<ID3D12Fence>         b12Fence;     // cross-process signal fence
    std::atomic<uint64_t>                       b12FenceValue{0};
    WGPUSharedFence                             dawnFence   = nullptr; // Dawn-imported fence

    // Ring of cross-adapter textures, all b12TexW x b12TexH x b12TexFmt
    // (reallocated on resolution / format change). A frame copies into a free
    // slot, so a copy never lands in a texture that a live frame, or GPU work
    // still reading one, uses.
    std::mutex                                  tierBMutex; // the ring, the Dawn fence, copies
    TierBSlot                                   b12Ring[kTierBSlots];
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

// Dawn's GetD3D11Device/GetD3D12Device are unchecked casts: each returns a
// non-null pointer of the wrong type on the other backend. Every call site
// asks the adapter's backendType first.
static WGPUBackendType dawn_backend_type(WGPUDevice device) {
    if (!device) return WGPUBackendType_Undefined;
    WGPUAdapterInfo info{};
    const auto status = wgpuDeviceGetAdapterInfo(device, &info);
    const WGPUBackendType backend = status == WGPUStatus_Success
                                        ? info.backendType
                                        : WGPUBackendType_Undefined;
    wgpuAdapterInfoFreeMembers(info);
    return backend;
}

static LUID get_dawn_adapter_luid() {
    LUID zero{0, 0};
    WGPUDevice device = get_device();
    if (!device) return zero;
    const WGPUBackendType backend = dawn_backend_type(device);
    if (backend == WGPUBackendType_D3D12) {
        if (auto d12 = dawn::native::d3d12::GetD3D12Device(device))
            return d12->GetAdapterLuid();
        return zero;
    }
    if (backend != WGPUBackendType_D3D11) return zero;
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
    return dawn_backend_type(get_device()) == WGPUBackendType_D3D12;
}

// Dawn's backend-specific GetD3D11Device helper casts without a backend check.
// Inspect the public adapter metadata before using it on an import fast path.
static bool dawn_is_d3d11(WGPUDevice device) {
    return dawn_backend_type(device) == WGPUBackendType_D3D11;
}

// Imports b12Fence into the CURRENT Dawn device. Apart from the D3D12 init
// because a context destroy drops it (it belongs to the old device) while the
// producer-side D3D12 state stays valid; the next bridge import re-imports.
static bool import_tier_b_fence(ProducerAdapterInfo* info, WGPUDevice device) {
    if (info->dawnFence) return true;
    if (!device) return false;
    HANDLE fenceHandle = nullptr;
    HRESULT hr = info->b12Dev->CreateSharedHandle(info->b12Fence.Get(), nullptr,
                                                  GENERIC_ALL, nullptr, &fenceHandle);
    WGPUSharedFence fence = nullptr;
    if (SUCCEEDED(hr) && fenceHandle) {
        WGPUSharedFenceDXGISharedHandleDescriptor fenceDesc{};
        fenceDesc.chain.sType = WGPUSType_SharedFenceDXGISharedHandleDescriptor;
        fenceDesc.handle      = fenceHandle;
        WGPUSharedFenceDescriptor sfDesc{};
        sfDesc.nextInChain = &fenceDesc.chain;
        fence = wgpuDeviceImportSharedFence(device, &sfDesc);
        CloseHandle(fenceHandle); // Dawn duplicated it; close our copy.
    }
    // Dawn returns an error object, never null: a refused import exports no type.
    WGPUSharedFenceExportInfo fenceInfo{};
    if (fence) wgpuSharedFenceExportInfo(fence, &fenceInfo);
    if (fenceInfo.type != WGPUSharedFenceType_DXGISharedHandle) {
        if (fence) wgpuSharedFenceRelease(fence);
        static std::atomic<int> s_failures{0};
        const int n = s_failures.fetch_add(1);
        if (n < 3 || (n % 256) == 0) {
            LOG_WARN("[minigpu_external] Tier B: fence import into Dawn failed "
                     "(hr=0x%08lX, count=%d).", (unsigned long)hr, n + 1);
        }
        return false;
    }
    info->dawnFence = fence;
    return true;
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

    // ── 6. Import the fence into Dawn ──
    if (!import_tier_b_fence(info, get_device())) {
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

// Allocates [slot]: a `w × h × fmt` committed cross-adapter texture on the
// producer, its D3D11on12 wrapper, and its import into [dawnDev]. The new
// slot comes back claimed. On failure (logged) [slot] is left unallocated.
static bool create_tier_b_slot(ProducerAdapterInfo* info, WGPUDevice dawnDev,
                               uint32_t w, uint32_t h, DXGI_FORMAT fmt,
                               TierBSlot& slot) {
    auto claimed = std::make_shared<std::atomic<bool>>(false);

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

    WGPUSharedTextureMemoryDXGISharedHandleDescriptor handleDesc{};
    handleDesc.chain.sType = WGPUSType_SharedTextureMemoryDXGISharedHandleDescriptor;
    handleDesc.handle      = resHandle;
    handleDesc.useKeyedMutex = false;
    WGPUSharedTextureMemoryDescriptor memDesc{};
    memDesc.nextInChain = &handleDesc.chain;
    WGPUSharedTextureMemory dawnMem = wgpuDeviceImportSharedTextureMemory(dawnDev, &memDesc);
    CloseHandle(resHandle); // Dawn opened it; close our copy.
    // Dawn returns an error object, never null: a refused import reports an
    // Undefined format.
    WGPUSharedTextureMemoryProperties memProps{};
    if (dawnMem) wgpuSharedTextureMemoryGetProperties(dawnMem, &memProps);
    if (memProps.format == WGPUTextureFormat_Undefined) {
        if (dawnMem) wgpuSharedTextureMemoryRelease(dawnMem);
        LOG_ERROR("[minigpu_external] Tier B: wgpuDeviceImportSharedTextureMemory failed for "
                  "cross-adapter D3D12 resource. Dawn may not be on D3D12 backend.");
        return false;
    }

    slot.b12CrossTex    = crossTex;
    slot.b11on12Wrapped = wrapped11tex;
    slot.dawnCrossMem   = dawnMem;
    slot.free           = std::move(claimed);
    LOG_INFO("[minigpu_external] Tier B: allocated cross-adapter ring slot %d (%ux%u fmt=%d).",
             (int)(&slot - info->b12Ring), w, h, (int)fmt);
    return true;
}

// Drops every ring slot. A live frame keeps its own reference to its slot's
// memory, and the flag it later sets is no longer read. Under tierBMutex.
static void release_tier_b_ring(ProducerAdapterInfo* info) {
    for (TierBSlot& slot : info->b12Ring) {
        if (slot.dawnCrossMem) wgpuSharedTextureMemoryRelease(slot.dawnCrossMem);
        slot = TierBSlot{};
    }
    info->b12TexW = 0; info->b12TexH = 0; info->b12TexFmt = DXGI_FORMAT_UNKNOWN;
}

// Claims a ring slot for a `w × h × fmt` frame: a free one, else a new one
// while the ring has room. -1 when every slot still holds a live frame or its
// unfinished GPU reads; the caller then takes Tier C rather than wait. Under
// tierBMutex.
static int claim_tier_b_slot(ProducerAdapterInfo* info, WGPUDevice dawnDev,
                             uint32_t w, uint32_t h, DXGI_FORMAT fmt) {
    if (info->b12TexW != w || info->b12TexH != h || info->b12TexFmt != fmt) {
        release_tier_b_ring(info);
        info->b12TexW = w; info->b12TexH = h; info->b12TexFmt = fmt;
    }
    for (int i = 0; i < kTierBSlots; ++i) {
        const auto& free = info->b12Ring[i].free;
        if (free && free->exchange(false, std::memory_order_acquire)) return i;
    }
    for (int i = 0; i < kTierBSlots; ++i) {
        if (info->b12Ring[i].free) continue;
        return create_tier_b_slot(info, dawnDev, w, h, fmt, info->b12Ring[i]) ? i : -1;
    }
    return -1;
}

// From a Tier B frame's destroy, after its EndAccess: frees its ring slot
// once Dawn's queue has finished everything submitted so far. That covers
// every read of the frame, all queued on the worker ahead of the destroy.
// (Dawn's EndAccess fence cannot gate the producer's next copy instead: it
// is not created SHARED_CROSS_ADAPTER, so the producer adapter cannot open
// it.) A cancelled wait (device lost, instance gone) keeps the slot taken;
// the teardown that follows rebuilds the ring.
static void release_tier_b_slot_when_idle(std::shared_ptr<std::atomic<bool>> free) {
    WGPUDevice   device   = nullptr;
    WGPUQueue    queue    = nullptr;
    WGPUInstance instance = nullptr;
    uint64_t     generation = 0;
    // No ready context: its teardown dropped the ring this slot belonged to.
    if (!minigpu.retainReadyContextSnapshot(&device, &queue, &instance, &generation))
        return;
    using Flag = std::shared_ptr<std::atomic<bool>>;
    try {
        WGPUQueueWorkDoneCallbackInfo info{};
        info.mode      = WGPUCallbackMode_AllowSpontaneous;
        info.callback  = [](WGPUQueueWorkDoneStatus status, WGPUStringView,
                            void* ud1, void*) {
            std::unique_ptr<Flag> flag(static_cast<Flag*>(ud1));
            if (status == WGPUQueueWorkDoneStatus_Success)
                (*flag)->store(true, std::memory_order_release);
        };
        info.userdata1 = new Flag(std::move(free));
        wgpuQueueOnSubmittedWorkDone(queue, info);
    } catch (...) {
        // Out of memory: the slot stays taken; frames use the others or Tier C.
    }
    wgpuQueueRelease(queue);
    wgpuInstanceRelease(instance);
    wgpuDeviceRelease(device);
}

// Steps 2-7 of the bridge below, under tierBMutex with [dawnDev] current:
// copies [src] into a claimed ring slot and opens that slot in Dawn behind
// the copy's fence. Null (Tier C takes the frame) when no slot is free or a
// step fails; a slot Dawn did not open goes back to the ring.
static MGPUVideoTexture* import_tier_b_frame(ProducerAdapterInfo* info,
                                             const MGPUExternalVideoBuffer* buf,
                                             ID3D11Texture2D* src,
                                             const D3D11_TEXTURE2D_DESC& sd,
                                             WGPUDevice dawnDev) {
    // ── 2. Claim a ring slot ──
    // A context destroy dropped the Dawn-side fence; re-import it here.
    if (!import_tier_b_fence(info, dawnDev)) return nullptr;
    const int k = claim_tier_b_slot(info, dawnDev, sd.Width, sd.Height, sd.Format);
    if (k < 0) {
        if (info->b12Ring[kTierBSlots - 1].free) { // all allocated, none free
            static std::atomic<int> s_busy{0};
            const int n = s_busy.fetch_add(1);
            if (n < 3 || (n % 256) == 0) {
                LOG_WARN("[minigpu_external] Tier B: all %d ring textures still hold live "
                         "or in-flight frames; this frame takes Tier C (count=%d).",
                         kTierBSlots, n + 1);
            }
        }
        return nullptr;
    }
    TierBSlot& slot = info->b12Ring[k];
    auto unclaim = [&slot] { slot.free->store(true, std::memory_order_release); };

    // ── 3. GPU copy: D3D11on12 CopyResource into the slot's wrapped texture ──
    Microsoft::WRL::ComPtr<ID3D11On12Device> on12;
    if (FAILED(info->b11on12Dev.As(&on12))) {
        unclaim();
        return nullptr;
    }
    ID3D11Resource* wrapped = slot.b11on12Wrapped.Get();
    on12->AcquireWrappedResources(&wrapped, 1);
    info->b11on12Ctx->CopyResource(wrapped, src);
    on12->ReleaseWrappedResources(&wrapped, 1);
    info->b11on12Ctx->Flush();

    // ── 4. Signal D3D12 fence on producer queue ──
    uint64_t signalVal = info->b12FenceValue.fetch_add(1) + 1;
    HRESULT hr = info->b12Queue->Signal(info->b12Fence.Get(), signalVal);
    if (FAILED(hr)) {
        LOG_ERROR("[minigpu_external] Tier B: queue Signal failed: 0x%08lX", (unsigned long)hr);
        unclaim();
        return nullptr;
    }

    // ── 5. Create a Dawn texture from the slot's shared memory ──
    WGPUSharedTextureMemoryProperties props{};
    if (wgpuSharedTextureMemoryGetProperties(slot.dawnCrossMem, &props) != WGPUStatus_Success ||
        props.format == WGPUTextureFormat_Undefined) {
        LOG_ERROR("[minigpu_external] Tier B: shared memory properties unavailable.");
        unclaim();
        return nullptr; // caller falls through to Tier C
    }
    WGPUTextureDescriptor td{};
    td.usage         = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
    td.dimension     = WGPUTextureDimension_2D;
    td.size          = props.size;
    td.format        = props.format;
    td.mipLevelCount = 1;
    td.sampleCount   = 1;
    WGPUTexture wTex = wgpuSharedTextureMemoryCreateTexture(slot.dawnCrossMem, &td);

    // ── 6. BeginAccess with fence wait ──
    // Dawn hands back an error texture rather than null; BeginAccess refuses it.
    WGPUSharedTextureMemoryBeginAccessDescriptor baDesc{};
    baDesc.initialized   = true;
    baDesc.fenceCount    = 1;
    baDesc.fences        = &info->dawnFence;
    baDesc.signaledValues = &signalVal;
    if (!wTex || wgpuSharedTextureMemoryBeginAccess(slot.dawnCrossMem, wTex, &baDesc) !=
                     WGPUStatus_Success) {
        LOG_ERROR("[minigpu_external] Tier B: texture creation or BeginAccess failed.");
        if (wTex) wgpuTextureRelease(wTex);
        unclaim();
        return nullptr; // caller falls through to Tier C
    }

    // ── 7. Build the returned texture object ──
    // AddRef the shared memory so mgpuDestroyVideoTexture's Release doesn't
    // free the ring's copy.
    wgpuSharedTextureMemoryAddRef(slot.dawnCrossMem);

    auto* tex = new MGPUVideoTexture();
    tex->width        = buf->width;
    tex->height       = buf->height;
    tex->pixel_format = buf->pixel_format;
    tex->content_type = buf->content_type;
    tex->shared_mem   = slot.dawnCrossMem; // caller (via mgpuDestroyVideoTexture) will EndAccess + Release
    tex->tier_b_free  = slot.free;         // ... and then hand the slot back
    tex->planes[0]    = wTex;
    build_video_texture_views(tex, wTex, buf->pixel_format, props.format);
    return tex;
}

// ── Tier B per-frame import ──────────────────────────────────────────────
// Opens `srcHandle` on the D3D11on12 device, copies to a free cross-adapter
// D3D12 texture of the ring, signals the D3D12 fence, then returns an
// MGPUVideoTexture backed by that slot's Dawn SharedTextureMemory with a
// fence-wait BeginAccess.
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

    D3D11_TEXTURE2D_DESC sd{};
    srcOnD3D11on12->GetDesc(&sd);
    // Retained before the ring lock, never under it: a context teardown takes
    // that lock to drop the ring, so nothing under it may wait on the context.
    uint64_t generation = 0;
    WGPUDevice dawnDev = minigpu.retainReadyDeviceGeneration(&generation);
    if (!dawnDev) return nullptr;
    MGPUVideoTexture* tex = nullptr;
    {
        std::lock_guard<std::mutex> ring(info->tierBMutex);
        // A teardown since the retain has already dropped this device's ring
        // and fence; filling them again would leave the next device stale ones.
        if (minigpu.matchesReadyDevice(dawnDev, generation))
            tex = import_tier_b_frame(info, buf, srcOnD3D11on12.Get(), sd, dawnDev);
    }
    wgpuDeviceRelease(dawnDev);
    return tex;
}

// Tier A's private copies, reused instead of allocated per frame. A slot is
// free once the ring holds its only reference: Dawn has released the video
// texture and every command buffer or binding that used it, and any read it
// recorded precedes the next copy on the same immediate context. All slots
// busy: the frame gets an unpooled texture. The fence replaces a busy-spin.
struct PrivateCopyRing {
    Microsoft::WRL::ComPtr<ID3D11Device>    device; // owns slots and fence
    Microsoft::WRL::ComPtr<ID3D11Texture2D> slots[4];
    Microsoft::WRL::ComPtr<ID3D11Fence>     fence;
    uint64_t                                fenceValue = 0;
};
static std::mutex      g_private_copy_mutex;
static PrivateCopyRing g_private_copy; // g_private_copy_mutex

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
    // Only a producer-certified, completed immutable snapshot may be shared
    // directly. In particular, a WGC-acquired surface can be recycled when
    // its frame is released and MUST retain the private-copy path below.
    // The D3D11 texture descriptor requires Dawn's own D3D11 device; D3D12
    // and every unrecognised/untagged input keep their existing path.
    const bool immutableReady =
        buf->content_type == MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE &&
        buf->num_planes == 1 &&
        buf->pixel_format == MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32 &&
        buf->planes[0].drm_format_modifier == MGPU_D3D11_IMMUTABLE_READY_TAG &&
        buf->planes[0].subresource_index == 0 &&
        sd.Format == DXGI_FORMAT_B8G8R8A8_UNORM &&
        sd.Width == buf->width && sd.Height == buf->height &&
        sd.ArraySize == 1 && sd.MipLevels == 1 && sd.SampleDesc.Count == 1 &&
        (sd.BindFlags & D3D11_BIND_SHADER_RESOURCE) != 0 &&
        dawn_is_d3d11(device) &&
        dawn::native::d3d11::GetD3D11Device(device).Get() == d11;
    if (immutableReady) {
        dawn::native::d3d11::SharedTextureMemoryD3D11Texture2DDescriptor direct;
        direct.texture = srcOnDawnDev;
        WGPUSharedTextureMemoryDescriptor desc{};
        desc.nextInChain = const_cast<WGPUChainedStruct*>(
            reinterpret_cast<const WGPUChainedStruct*>(
                static_cast<const wgpu::ChainedStruct*>(&direct)));
        auto mem = wgpuDeviceImportSharedTextureMemory(device, &desc);
        // SharedTextureMemory and its Texture retain the D3D11 resource.
        // The producer's handle and COM reference may be released as soon
        // as this import returns; queued GPU reads retain their own refs.
        if (auto* imported = import_shared(mem, buf)) {
            LOG_DEBUG("[minigpu_external] immutable D3D11 snapshot imported directly");
            return imported;
        }
    }
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
    Microsoft::WRL::ComPtr<ID3D11DeviceContext4> ctx4;
    if (FAILED(ctx.As(&ctx4)) || !ctx4) return nullptr;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> dst;
    Microsoft::WRL::ComPtr<ID3D11Fence> copyFence;
    uint64_t copyValue = 0;
    {
        std::lock_guard<std::mutex> lk(g_private_copy_mutex);
        PrivateCopyRing& ring = g_private_copy;
        if (ring.device.Get() != d11) ring = PrivateCopyRing{};
        if (!ring.fence) {
            Microsoft::WRL::ComPtr<ID3D11Device5> d11_5;
            hr = d11->QueryInterface(IID_PPV_ARGS(&d11_5));
            if (SUCCEEDED(hr))
                hr = d11_5->CreateFence(0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&ring.fence));
            if (FAILED(hr) || !ring.fence) {
                LOG_ERROR("[minigpu_external] import_d3d11: CreateFence(private copy) failed: 0x%08lX", (unsigned long)hr);
                return nullptr;
            }
            ring.device = d11;
        }
        Microsoft::WRL::ComPtr<ID3D11Texture2D>* spare = nullptr;
        for (auto& slot : ring.slots) {
            if (slot) {
                slot->AddRef();
                if (slot->Release() != 1) continue; // Dawn still holds it
                D3D11_TEXTURE2D_DESC cd{};
                slot->GetDesc(&cd);
                if (std::memcmp(&cd, &dd, sizeof dd) == 0) {
                    if (!dst) dst = slot;
                    continue;
                }
                slot.Reset(); // free, but shaped for an earlier source
            }
            if (!spare) spare = &slot;
        }
        if (!dst) {
            hr = d11->CreateTexture2D(&dd, nullptr, &dst);
            if (FAILED(hr) || !dst) {
                LOG_ERROR("[minigpu_external] import_d3d11: CreateTexture2D(private dst) failed: 0x%08lX", (unsigned long)hr);
                return nullptr;
            }
            if (spare) *spare = dst;
            LOG_DEBUG("[minigpu_external] import_d3d11: private copy allocated (%ux%u, %s)",
                dd.Width, dd.Height, spare ? "pooled" : "unpooled");
        }
        ctx->CopyResource(dst.Get(), srcOnDawnDev.Get());
        // Signalled under the lock so fence values complete in order.
        copyValue = ++ring.fenceValue;
        hr = ctx4->Signal(ring.fence.Get(), copyValue);
        ctx->Flush();
        copyFence = ring.fence;
    }

    // Bounded CPU wait for the copy: the producer may recycle its source as
    // soon as this returns, and the consumer must see the whole frame.
    if (SUCCEEDED(hr)) {
        HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (done) {
            if (SUCCEEDED(copyFence->SetEventOnCompletion(copyValue, done)))
                WaitForSingleObject(done, 50);
            CloseHandle(done);
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
// Dawn format for a single-plane producer texture; Undefined = unsupported.
static WGPUTextureFormat cpu_bridge_format(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:      return WGPUTextureFormat_RGBA8Unorm;
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return WGPUTextureFormat_RGBA8UnormSrgb;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:      return WGPUTextureFormat_BGRA8Unorm;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return WGPUTextureFormat_BGRA8UnormSrgb;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:  return WGPUTextureFormat_RGBA16Float;
    case DXGI_FORMAT_R10G10B10A2_UNORM:   return WGPUTextureFormat_RGB10A2Unorm;
    case DXGI_FORMAT_R8_UNORM:            return WGPUTextureFormat_R8Unorm;
    default:                              return WGPUTextureFormat_Undefined;
    }
}

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

    // Slow path: enumerate adapters (the factory is only needed here).
    if (!srcOnProducer) {
        Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || !factory) {
            LOG_ERROR("[minigpu_external] cpu_bridge: CreateDXGIFactory1 failed");
            return nullptr;
        }
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

    // NV12 travels as two planes (R8 Y + RG8 UV); any other format needs a
    // Dawn equivalent. The declared format must agree on the plane count,
    // since consumers bind views[] by it.
    const bool isNv12 = sd.Format == DXGI_FORMAT_NV12;
    const WGPUTextureFormat wfmt = cpu_bridge_format(sd.Format);
    if (num_planes_for_format(buf->pixel_format) != (isNv12 ? 2 : 1) ||
        (!isNv12 && wfmt == WGPUTextureFormat_Undefined)) {
        static std::atomic<int> s_rejects{0};
        const int n = s_rejects.fetch_add(1);
        if (n < 3 || (n % 256) == 0) {
            LOG_ERROR("[minigpu_external] cpu_bridge: unsupported source (DXGI format %u, "
                      "declared pixel format %d, count=%d).",
                      (unsigned)sd.Format, (int)buf->pixel_format, n + 1);
        }
        return nullptr;
    }

    // One copy in flight per producer: its frames share the staging texture,
    // recreated only when the source description changes.
    std::lock_guard<std::mutex> bridgeLock(info->cpuBridgeMutex);
    D3D11_TEXTURE2D_DESC stagingDesc = sd;
    stagingDesc.Usage          = D3D11_USAGE_STAGING;
    stagingDesc.BindFlags      = 0;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    stagingDesc.MiscFlags      = 0;
    if (info->cpuStaging) {
        D3D11_TEXTURE2D_DESC cached{};
        info->cpuStaging->GetDesc(&cached);
        if (cached.Width != sd.Width || cached.Height != sd.Height ||
            cached.MipLevels != sd.MipLevels || cached.ArraySize != sd.ArraySize ||
            cached.Format != sd.Format || cached.SampleDesc.Count != sd.SampleDesc.Count ||
            cached.SampleDesc.Quality != sd.SampleDesc.Quality) {
            info->cpuStaging.Reset();
        }
    }
    if (!info->cpuStaging) {
        HRESULT hr = info->device->CreateTexture2D(&stagingDesc, nullptr, &info->cpuStaging);
        if (FAILED(hr) || !info->cpuStaging) {
            info->cpuStaging.Reset();
            LOG_ERROR("[minigpu_external] cpu_bridge: CreateTexture2D(staging) failed: 0x%08lX",
                (unsigned long)hr);
            return nullptr;
        }
    }
    ID3D11Texture2D* staging = info->cpuStaging.Get();
    info->context->CopyResource(staging, srcOnProducer.Get());

    // Map(READ) without DO_NOT_WAIT blocks until the copy above has finished.
    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = info->context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr) || !mapped.pData) {
        LOG_ERROR("[minigpu_external] cpu_bridge: Map(staging) failed: 0x%08lX",
            (unsigned long)hr);
        return nullptr;
    }

    // A new Dawn texture per frame: the caller owns each returned frame until
    // mgpuDestroyVideoTexture, so two live frames must not share storage.
    auto upload = [&](WGPUTextureFormat format, uint32_t w, uint32_t h,
                      const uint8_t* data, WGPUTexture* outTex,
                      WGPUTextureView* outView) {
        WGPUTextureDescriptor td{};
        td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst
                 | WGPUTextureUsage_CopySrc;
        td.dimension = WGPUTextureDimension_2D;
        td.size = {w, h, 1};
        td.format = format;
        td.mipLevelCount = 1;
        td.sampleCount = 1;
        *outTex = wgpuDeviceCreateTexture(device, &td);

        WGPUTexelCopyTextureInfo dstInfo{};
        dstInfo.texture  = *outTex;
        dstInfo.mipLevel = 0;
        dstInfo.aspect   = WGPUTextureAspect_All;
        WGPUTexelCopyBufferLayout layout{};
        layout.offset       = 0;
        layout.bytesPerRow  = mapped.RowPitch;
        layout.rowsPerImage = h;
        WGPUExtent3D extent = {w, h, 1};
        wgpuQueueWriteTexture(queue, &dstInfo, data, (size_t)mapped.RowPitch * h,
                              &layout, &extent);

        WGPUTextureViewDescriptor vd{};
        vd.format          = format;
        vd.dimension       = WGPUTextureViewDimension_2D;
        vd.baseMipLevel    = 0; vd.mipLevelCount   = 1;
        vd.baseArrayLayer  = 0; vd.arrayLayerCount = 1;
        vd.aspect          = WGPUTextureAspect_All;
        *outView = wgpuTextureCreateView(*outTex, &vd);
    };

    auto* tex = new MGPUVideoTexture();
    tex->width        = buf->width;
    tex->height       = buf->height;
    tex->pixel_format = buf->pixel_format;
    tex->content_type = buf->content_type;
    const uint8_t* base = static_cast<const uint8_t*>(mapped.pData);
    if (isNv12) {
        // D3D11 maps NV12 as one block: the UV rows follow the Y rows at
        // RowPitch * Height, with the same pitch.
        tex->num_planes = 2;
        upload(WGPUTextureFormat_R8Unorm, sd.Width, sd.Height, base,
               &tex->planes[0], &tex->views[0]);
        upload(WGPUTextureFormat_RG8Unorm, sd.Width / 2, sd.Height / 2,
               base + (size_t)mapped.RowPitch * sd.Height,
               &tex->planes[1], &tex->views[1]);
    } else {
        tex->num_planes = 1;
        upload(wfmt, sd.Width, sd.Height, base, &tex->planes[0], &tex->views[0]);
    }
    info->context->Unmap(staging, 0);
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
    // Producer fences are not imported: sampling before their signal would
    // read a half-written frame, so a frame carrying one is refused. As for
    // imported frames, sync_fd 0 and -1 both mean none; it is never closed.
    const MGPUExternalFence& fence = buf->fence;
    if ((fence.sync_fd != 0 && fence.sync_fd != -1) || fence.d3d11_fence ||
        fence.metal_shared_event || fence.metal_fence_value) {
        LOG_ERROR("[minigpu_external] mgpuImportVideoFrame: producer fences are not "
                  "supported; wait on the producer side and pass an empty fence.");
        return nullptr;
    }

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

static void destroy_video_texture_on_worker(MGPUVideoTexture* tex) {
#ifndef __EMSCRIPTEN__
    if (tex->shared_mem) {
        // End access releases keyed-mutex / fence.
        end_shared_access(tex->shared_mem, tex->planes[0]);
    }
#endif
#ifdef _WIN32
    if (tex->tier_b_free) release_tier_b_slot_when_idle(std::move(tex->tier_b_free));
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

void mgpuDestroyVideoTexture(MGPUVideoTexture* tex) {
    if (!tex) return;
    // On the worker, behind every queued blit that still reads the texture.
    // Synchronous: the next import may reopen the same shared memory (a Tier
    // B ring slot handed back), and its BeginAccess needs this access ended.
    run_on_webgpu_worker([tex] {
        destroy_video_texture_on_worker(tex);
        return 1;
    });
}

/* =========================================================================
 * YUV -> RGBA conversion helper
 * ====================================================================== */

// Minimal WGSL compute shader for NV12 -> RGBA8 conversion; the matrix and
// range come from the uniform (RgbaUniforms).
static const char* kNV12ToRGBAShader = R"(
@group(0) @binding(0) var y_tex  : texture_2d<f32>;
@group(0) @binding(1) var uv_tex : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out_buf : array<u32>;

struct Uniforms {
    width: u32, height: u32,
    y_offset: f32, y_scale: f32, c_offset: f32, c_scale: f32,
    r_cr: f32, g_cb: f32, g_cr: f32, b_cb: f32,
    pad0: u32, pad1: u32,
}
@group(0) @binding(3) var<uniform> uni : Uniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uni.width || gid.y >= uni.height) { return; }
    let y  = (textureLoad(y_tex,  vec2<u32>(gid.x, gid.y), 0).r - uni.y_offset) * uni.y_scale;
    let cb = (textureLoad(uv_tex, vec2<u32>(gid.x / 2u, gid.y / 2u), 0).r - uni.c_offset) * uni.c_scale;
    let cr = (textureLoad(uv_tex, vec2<u32>(gid.x / 2u, gid.y / 2u), 0).g - uni.c_offset) * uni.c_scale;
    let r  = clamp(y + uni.r_cr * cr,                 0.0, 1.0);
    let g  = clamp(y - uni.g_cb * cb - uni.g_cr * cr, 0.0, 1.0);
    let b  = clamp(y + uni.b_cb * cb,                 0.0, 1.0);
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
    let ri = u32(clamp(px.r, 0.0, 1.0) * 255.0 + 0.5);
    let gi = u32(clamp(px.g, 0.0, 1.0) * 255.0 + 0.5);
    let bi = u32(clamp(px.b, 0.0, 1.0) * 255.0 + 0.5);
    let ai = u32(clamp(px.a, 0.0, 1.0) * 255.0 + 0.5);
    out_buf[gid.y * uni.width + gid.x] = ri | (gi << 8u) | (bi << 16u) | (ai << 24u);
}
)";

// GRAY8: one R8 plane splatted to grey, opaque; the luma range applies.
static const char* kGray8ToRGBAShader = R"(
@group(0) @binding(0) var in_tex : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf : array<u32>;

struct Uniforms { width: u32, height: u32, y_offset: f32, y_scale: f32 }
@group(0) @binding(2) var<uniform> uni : Uniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uni.width || gid.y >= uni.height) { return; }
    let s = (textureLoad(in_tex, vec2<i32>(gid.xy), 0).r - uni.y_offset) * uni.y_scale;
    let v = u32(clamp(s, 0.0, 1.0) * 255.0 + 0.5);
    out_buf[gid.y * uni.width + gid.x] = v | (v << 8u) | (v << 16u) | (255u << 24u);
}
)";

// YUV420P_AS_RGB_PLANES: Y, U, V as three R8 planes; same matrix as NV12.
static const char* kI420ToRGBAShader = R"(
@group(0) @binding(0) var y_tex : texture_2d<f32>;
@group(0) @binding(1) var u_tex : texture_2d<f32>;
@group(0) @binding(2) var v_tex : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> out_buf : array<u32>;

struct Uniforms {
    width: u32, height: u32,
    y_offset: f32, y_scale: f32, c_offset: f32, c_scale: f32,
    r_cr: f32, g_cb: f32, g_cr: f32, b_cb: f32,
    pad0: u32, pad1: u32,
}
@group(0) @binding(4) var<uniform> uni : Uniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uni.width || gid.y >= uni.height) { return; }
    let c  = min(gid.xy / 2u, textureDimensions(u_tex) - vec2<u32>(1u, 1u));
    let y  = (textureLoad(y_tex, gid.xy, 0).r - uni.y_offset) * uni.y_scale;
    let cb = (textureLoad(u_tex, c, 0).r - uni.c_offset) * uni.c_scale;
    let cr = (textureLoad(v_tex, c, 0).r - uni.c_offset) * uni.c_scale;
    let r  = clamp(y + uni.r_cr * cr,                 0.0, 1.0);
    let g  = clamp(y - uni.g_cb * cb - uni.g_cr * cr, 0.0, 1.0);
    let b  = clamp(y + uni.b_cb * cb,                 0.0, 1.0);
    let ri = u32(r * 255.0);
    let gi = u32(g * 255.0);
    let bi = u32(b * 255.0);
    out_buf[gid.y * uni.width + gid.x] = ri | (gi << 8u) | (bi << 16u) | (255u << 24u);
}
)";

// toRGBA input layouts; each has one cached pipeline (auto layout), so a
// call only builds a bind group. Textures bind first, in plane order, then
// the output buffer, then the RgbaUniforms uniform (size, colour terms).
enum RgbaConverter { kRgbaPacked, kRgbaGray, kRgbaNv12, kRgbaI420, kRgbaConverterCount };

static std::mutex          g_rgba_mutex;              // guards the two below
static WGPUDevice          g_rgba_device = nullptr;   // identity; the pipelines hold it
static WGPUComputePipeline g_rgba_pipelines[kRgbaConverterCount] = {};

static void release_rgba_pipelines_locked() {
    for (WGPUComputePipeline& p : g_rgba_pipelines) {
        if (p) wgpuComputePipelineRelease(p);
        p = nullptr;
    }
    g_rgba_device = nullptr;
}

static void release_rgba_pipelines() {
    std::lock_guard<std::mutex> lk(g_rgba_mutex);
    release_rgba_pipelines_locked();
}

// Returns a new reference to the converter for [kind] on [device].
static WGPUComputePipeline rgba_pipeline(WGPUDevice device, RgbaConverter kind) {
    static const char* const kSources[kRgbaConverterCount] = {
        kRGBAPassthroughShader, kGray8ToRGBAShader, kNV12ToRGBAShader, kI420ToRGBAShader,
    };
    std::lock_guard<std::mutex> lk(g_rgba_mutex);
    if (g_rgba_device != device) {
        release_rgba_pipelines_locked();
        g_rgba_device = device;
    }
    if (!g_rgba_pipelines[kind]) {
        WGPUShaderSourceWGSL wgsl{};
        wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgsl.code.data   = kSources[kind];
        wgsl.code.length = std::strlen(kSources[kind]);
        WGPUShaderModuleDescriptor smDesc{};
        smDesc.nextInChain = &wgsl.chain;
        WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &smDesc);

        WGPUComputePipelineDescriptor cpDesc{};
        cpDesc.compute.module            = module;
        cpDesc.compute.entryPoint.data   = "main";
        cpDesc.compute.entryPoint.length = 4;
        g_rgba_pipelines[kind] = wgpuDeviceCreateComputePipeline(device, &cpDesc);
        wgpuShaderModuleRelease(module);
    }
    wgpuComputePipelineAddRef(g_rgba_pipelines[kind]);
    return g_rgba_pipelines[kind];
}

// The per-call toRGBA uniform: the `Uniforms` struct of the YUV shaders (the
// grey and packed converters read a prefix of it). A sample maps to luma
// (y - y_offset) * y_scale and chroma (c - c_offset) * c_scale, then to RGB.
struct RgbaUniforms {
    uint32_t width, height;
    float    y_offset, y_scale;
    float    c_offset, c_scale;
    float    r_cr, g_cb, g_cr, b_cb;
    uint32_t pad[2];
};
static_assert(sizeof(RgbaUniforms) == 48, "matches the WGSL Uniforms struct");

// Colour terms for one MGPUVideoTexture::yuv_color value. BT.709 full range
// keeps the constants and the 0.5 chroma bias of the original fixed-matrix
// shaders, so its output is unchanged.
static void rgba_color_terms(int color, RgbaUniforms& u) {
    // 2(1-Kr), 2Kb(1-Kb)/Kg, 2Kr(1-Kr)/Kg, 2(1-Kb) from each standard's Kr/Kb.
    static const float kMatrices[3][4] = {
        {1.5748f, 0.1873f,   0.4681f,   1.8556f},  // BT.709
        {1.402f,  0.344136f, 0.714136f, 1.772f},   // BT.601
        {1.4746f, 0.164553f, 0.571353f, 1.8814f},  // BT.2020 (non-constant luminance)
    };
    const float* m = kMatrices[color & 0xff];
    u.r_cr = m[0]; u.g_cb = m[1]; u.g_cr = m[2]; u.b_cb = m[3];
    if (color & MGPUVideoTexture::kYuvFullRange) {
        u.y_offset = 0.0f;           u.y_scale = 1.0f;
        u.c_offset = 0.5f;           u.c_scale = 1.0f;
    } else {
        // Video range: luma 16-235, chroma 16-240 around 128.
        u.y_offset = 16.0f / 255.0f;  u.y_scale = 255.0f / 219.0f;
        u.c_offset = 128.0f / 255.0f; u.c_scale = 255.0f / 224.0f;
    }
}

// One toRGBA conversion, checked on the caller's thread: its input layout
// and the colour setting at the call. It is recorded on the worker.
struct RgbaJob {
    RgbaConverter kind   = kRgbaPacked;
    int           inputs = 0;
    RgbaUniforms  uniforms{};
};

static uint64_t rgba_output_bytes(const RgbaJob& job) {
    return (uint64_t)job.uniforms.width * job.uniforms.height * 4; // RGBA8
}

// False when [tex] cannot be converted; nothing is recorded then.
static bool rgba_job_for(MGPUVideoTexture* tex, RgbaJob& job) {
    if (!tex || tex->num_planes == 0) return false;

    // The declared format picks the input layout; any other format is
    // refused rather than read through the wrong one.
    switch (tex->pixel_format) {
    case MGPU_EXTERNAL_PIXEL_FORMAT_RGBA32:
    case MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32:
    case MGPU_EXTERNAL_PIXEL_FORMAT_RGBA64_HALF:
        job.kind = kRgbaPacked; job.inputs = 1; break;
    case MGPU_EXTERNAL_PIXEL_FORMAT_GRAY8:
        job.kind = kRgbaGray; job.inputs = 1; break;
    case MGPU_EXTERNAL_PIXEL_FORMAT_NV12:
    case MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_NV12_PLANES:
        job.kind = kRgbaNv12; job.inputs = 2; break;
    case MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_RGB_PLANES:
        job.kind = kRgbaI420; job.inputs = 3; break;
    default:
        LOG_ERROR("[minigpu_external] toRGBA: unsupported pixel format %d.",
                  (int)tex->pixel_format);
        return false;
    }
    if (tex->num_planes < job.inputs) return false;
    for (int i = 0; i < job.inputs; ++i) {
        if (!tex->views[i]) return false;
    }
    if (tex->width == 0 || tex->height == 0) return false;
    job.uniforms.width  = tex->width;
    job.uniforms.height = tex->height;
    rgba_color_terms(tex->yuv_color.load(std::memory_order_relaxed), job.uniforms);
    return true;
}

// Records [job] for [tex] into [out] and submits it. On the worker; the
// sync and async conversions share it.
static void submit_rgba_job(MGPUVideoTexture* tex, const RgbaJob& job,
                            WGPUBuffer out, WGPUDevice device, WGPUQueue queue) {
    const uint32_t W = job.uniforms.width;
    const uint32_t H = job.uniforms.height;
    WGPUBufferDescriptor ubd{};
    ubd.size  = sizeof(RgbaUniforms);
    ubd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    WGPUBuffer ubuf = wgpuDeviceCreateBuffer(device, &ubd);

    WGPUBindGroupEntry entries[5]{};
    uint32_t n = 0;
    for (int i = 0; i < job.inputs; ++i, ++n) {
        entries[n].binding     = n;
        entries[n].textureView = tex->views[i];
    }
    entries[n].binding = n;
    entries[n].buffer  = out;
    entries[n].size    = rgba_output_bytes(job);
    ++n;
    entries[n].binding = n;
    entries[n].buffer  = ubuf;
    entries[n].size    = sizeof(RgbaUniforms);
    ++n;

    WGPUComputePipeline pipeline = rgba_pipeline(device, job.kind);
    WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout     = bgl;
    bgDesc.entryCount = n;
    bgDesc.entries    = entries;
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);
    wgpuBindGroupLayoutRelease(bgl);

    {
        mgpu::lock_guard<mgpu::mutex> lock(minigpu.getGpuMutex());
        // Dispatches batched before this call reach the queue first.
        minigpu.flushBatchLocked();
        wgpuQueueWriteBuffer(queue, ubuf, 0, &job.uniforms, sizeof(RgbaUniforms));
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, nullptr);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
        wgpuComputePassEncoderSetPipeline(pass, pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, (W + 7) / 8, (H + 7) / 8, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuCommandEncoderRelease(enc);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
    }
    wgpuBindGroupRelease(bindGroup);
    wgpuComputePipelineRelease(pipeline);
    wgpuBufferRelease(ubuf);
}

MGPUBuffer* mgpuVideoTextureToRGBA(MGPUVideoTexture* tex) {
    RgbaJob job;
    if (!rgba_job_for(tex, job)) return nullptr;

    // On the worker, behind queued uploads and dispatches; the GPU wait sits
    // there too, so callbacks never fire on the caller's thread. Documented
    // as synchronous: the caller may destroy the frame as soon as it returns.
    mgpu::Buffer* result = nullptr;
    run_on_webgpu_worker([&]() -> int {
        WGPUDevice device = get_device();
        WGPUQueue  queue  = get_queue();
        if (!device || !queue) return 0;

        std::unique_ptr<mgpu::Buffer> outBuf(new mgpu::Buffer(minigpu));
        outBuf->createBuffer((size_t)rgba_output_bytes(job), mgpu::kUInt8);
        if (!outBuf->bufferData.buffer) return 0;

        submit_rgba_job(tex, job, outBuf->bufferData.buffer, device, queue);
        if (!wait_submitted_work(queue, "toRGBA")) return 0;
        result = outBuf.release();
        return 1;
    });
    return reinterpret_cast<MGPUBuffer*>(result);
}

// Completion of one async toRGBA. Dawn's work-done callback (any thread)
// records the status. While the wait is parked the worker keeps the instance
// pumped (nothing else is guaranteed to) and reports from there, where the
// device-generation check takes no lock inside a Dawn callback. Referenced
// by the callback and by the parked node.
struct RgbaAsyncWait {
    std::function<void(bool)>    done;
    WGPUDevice                   device = nullptr; // identity only, never used
    uint64_t                     generation = 0;
    std::atomic<int>             status{0};
    std::atomic<bool>            fired{false};
    std::atomic<bool>            parked{false};
    mgpu::task_queue::ParkedWait node;

    void finish(bool ok) noexcept {
        try { done(ok); } catch (...) {}
    }
    static void workDone(WGPUQueueWorkDoneStatus status, WGPUStringView,
                         void* ud1, void*) {
        std::unique_ptr<std::shared_ptr<RgbaAsyncWait>> self(
            static_cast<std::shared_ptr<RgbaAsyncWait>*>(ud1));
        RgbaAsyncWait& w = **self;
        w.status.store((int)status, std::memory_order_relaxed);
        // Unparked (the web): nothing will resume it, so report here.
        if (!w.parked.load(std::memory_order_relaxed))
            w.finish(status == WGPUQueueWorkDoneStatus_Success);
        w.fired.store(true, std::memory_order_release);
    }
    static void resume(mgpu::task_queue::ParkedWait& node) noexcept {
        std::unique_ptr<std::shared_ptr<RgbaAsyncWait>> self(
            static_cast<std::shared_ptr<RgbaAsyncWait>*>(node.owner));
        RgbaAsyncWait& w = **self;
        // A lost device reports its queue done as well; it converted nothing.
        w.finish(w.status.load(std::memory_order_relaxed) ==
                         (int)WGPUQueueWorkDoneStatus_Success &&
                 minigpu.matchesReadyDevice(w.device, w.generation));
    }
};

// The worker half of video_texture_to_rgba_async: records and submits, then
// has [done] run once the GPU is through. Calls [done] itself on failure.
static void submit_rgba_async(MGPUVideoTexture* tex, const RgbaJob& job,
                              WGPUBuffer out, std::function<void(bool)>& done) {
    std::shared_ptr<RgbaAsyncWait> wait;
    std::unique_ptr<std::shared_ptr<RgbaAsyncWait>> callbackRef, parkedRef;
    try {
        wait = std::make_shared<RgbaAsyncWait>();
        callbackRef.reset(new std::shared_ptr<RgbaAsyncWait>(wait));
        parkedRef.reset(new std::shared_ptr<RgbaAsyncWait>(wait));
    } catch (...) {
        done(false);
        return;
    }
    WGPUDevice   device   = nullptr;
    WGPUQueue    queue    = nullptr;
    WGPUInstance instance = nullptr;
    uint64_t     generation = 0;
    if (!minigpu.retainReadyContextSnapshot(&device, &queue, &instance, &generation)) {
        done(false);
        return;
    }
    bool submitted = false;
    try {
        submit_rgba_job(tex, job, out, device, queue);
        submitted = true;
    } catch (...) {
    }
    if (submitted) {
        wait->done       = std::move(done);
        wait->device     = device;
        wait->generation = generation;
        wait->node.ready  = &wait->fired;
        wait->node.resume = &RgbaAsyncWait::resume;
        wait->node.owner  = parkedRef.get();
        // Decided before the callback exists, so it knows who reports.
        if (minigpu.getWebGPUThread().park(wait->node, instance)) {
            parkedRef.release();
            wait->parked.store(true, std::memory_order_relaxed);
        }
        WGPUQueueWorkDoneCallbackInfo info{};
        info.mode      = WGPUCallbackMode_AllowSpontaneous;
        info.callback  = &RgbaAsyncWait::workDone;
        info.userdata1 = callbackRef.release();
        wgpuQueueOnSubmittedWorkDone(queue, info);
    }
    wgpuQueueRelease(queue);
    wgpuInstanceRelease(instance);
    wgpuDeviceRelease(device);
    if (!submitted) done(false);
}

// The work behind mgpuVideoTextureToRGBAAsyncToPort, with a native completion.
// Returns the output buffer at once, or null when [tex] cannot be converted
// (then [done] never runs). Otherwise [done] runs exactly once, on any thread:
// true after the GPU finished the conversion, false when it cannot.
static mgpu::Buffer* video_texture_to_rgba_async(MGPUVideoTexture* tex,
                                                 std::function<void(bool)> done) {
    RgbaJob job;
    if (!rgba_job_for(tex, job)) return nullptr;
    std::unique_ptr<mgpu::Buffer> outBuf;
    try {
        outBuf.reset(new mgpu::Buffer(minigpu));
        outBuf->createBuffer((size_t)rgba_output_bytes(job), mgpu::kUInt8);
    } catch (...) {
        return nullptr;
    }
    WGPUBuffer out = outBuf->bufferData.buffer;
    if (!out) return nullptr;
    // The task's own reference, in case the caller drops the buffer early.
    wgpuBufferAddRef(out);
    const bool queued = post_to_webgpu_worker([tex, job, out, done]() mutable {
        submit_rgba_async(tex, job, out, done);
        wgpuBufferRelease(out);
    });
    if (!queued) {
        wgpuBufferRelease(out);
        done(false);
    }
    return outBuf.release();
}

MGPUBuffer* mgpuVideoTextureToRGBAAsyncToPort(MGPUVideoTexture* tex,
                                              int64_t port, int64_t token) {
    try {
        return reinterpret_cast<MGPUBuffer*>(video_texture_to_rgba_async(
            tex, [port, token](bool ok) { mgpu::completionPost(port, token, ok); }));
    } catch (...) {
        return nullptr; // thrown before any work was handed off
    }
}

int mgpuVideoTextureSetYuvColor(MGPUVideoTexture* tex, int matrix, int full_range) {
    if (!tex) return -1;
    switch (matrix) {
    case MGPU_YUV_MATRIX_BT709:
    case MGPU_YUV_MATRIX_BT601:
    case MGPU_YUV_MATRIX_BT2020:
        break;
    default:
        return -1;
    }
    tex->yuv_color.store(matrix | (full_range ? MGPUVideoTexture::kYuvFullRange : 0),
                         std::memory_order_relaxed);
    return 0;
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

    // D3D12 BACKEND ONLY: the resource Dawn actually renders into. The
    // ID3D11Texture2D above is then a VIEW of this same allocation opened on
    // the sibling D3D11 device, which is what Flutter samples.
    Microsoft::WRL::ComPtr<ID3D12Resource>  d3d12_texture;

    HANDLE                                 nt_handle  = nullptr;

    // 🔴 THE TWO BACKENDS PRODUCE DIFFERENT KINDS OF HANDLE, and only one of
    // them may be closed. The D3D11 path uses IDXGIResource::GetSharedHandle(),
    // a LEGACY share handle that MSDN says is owned by the texture - closing it
    // is a double free. The D3D12 path uses ID3D12Device::CreateSharedHandle(),
    // a real NT handle that leaks unless we close it. Same field, opposite
    // rules, so the destructor is told which kind it is holding.
    bool                                   nt_handle_owned = false;
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
// THREADING: ID3D11DeviceContext is NOT free-threaded, and the lazy creation
// below was a bare double-check with no lock — two isolates racing it each
// created a device and the second ComPtr assignment RELEASED the first while
// the first caller was still holding the raw pointer it had just been handed.
// g_d3d11_mutex serializes both the creation and every immediate-context use
// in this file. (SetMultithreadProtected on the device covers D3D's own
// internal state, not the ComPtr slots themselves.)
static std::mutex                                  g_d3d11_mutex;
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

// Caller MUST hold g_d3d11_mutex.
static ID3D11Device* get_or_create_d3d11_device_on_dawn_adapter_locked() {
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
    const WGPUBackendType backend = dawn_backend_type(device);
    if (backend == WGPUBackendType_D3D11) {
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

    if (backend != WGPUBackendType_D3D12) {
        LOG_ERROR("[minigpu_external] get_or_create_d3d11_device: Dawn backend %d "
                  "is neither D3D11 nor D3D12", static_cast<int>(backend));
        return nullptr;
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

static ID3D11Device* get_or_create_d3d11_device_on_dawn_adapter() {
    std::lock_guard<std::mutex> lk(g_d3d11_mutex);
    return get_or_create_d3d11_device_on_dawn_adapter_locked();
}

static MGPUSharedOutputTexture* create_shared_output_texture(uint32_t w,
                                                             uint32_t h) {
    WGPUDevice device = get_device();
    // 🔴 THE LAST SILENT RETURN IN THIS FUNCTION. Every other failure path logs,
    // so a NULL with no log at all meant "we cannot tell you anything" - which
    // cost a debugging round on its own, because the Dart side then GUESSED at
    // a cause it had never observed. Both possible faults are named here.
    if (!device) {
        LOG_ERROR("[minigpu_external] create_shared_output_texture: get_device() "
                  "returned NULL - minigpu has no WGPUDevice on this thread.");
        return nullptr;
    }
    if (w == 0 || h == 0) {
        LOG_ERROR("[minigpu_external] create_shared_output_texture: degenerate size "
                  "%ux%u.", w, h);
        return nullptr;
    }
    // Entry marker: proves the call HAPPENED. A missing "SharedOutputTexture ..."
    // line is otherwise indistinguishable from the function never being called,
    // which is exactly the ambiguity that stalled this investigation.
    //
    // 🔴 WARN, NOT INFO, AND ONLY ONCE. Hosts run minigpu at warn+ because INFO
    // narrates several buffer ops per encoded frame - so an INFO breadcrumb here
    // is invisible in exactly the situation it exists to diagnose. Once is
    // enough to distinguish "never called" from "called and failed".
    static std::atomic<bool> saidEntered{false};
    if (!saidEntered.exchange(true)) {
        LOG_WARN("[minigpu_external] create_shared_output_texture(%ux%u) entered "
                 "(first call; logged once).", w, h);
    }

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
    const WGPUBackendType backend = dawn_backend_type(device);
    Microsoft::WRL::ComPtr<ID3D11Device> dawnD11;
    if (backend == WGPUBackendType_D3D11)
        dawnD11 = dawn::native::d3d11::GetD3D11Device(device);
    const bool dawnOnD3D11 = dawnD11 && dawnD11.Get() == d3d11Device;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> d3d11Tex;
    Microsoft::WRL::ComPtr<ID3D12Resource>  d3d12Texture;
    bool ownsNtHandle = false;
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
        // ── Dawn is on the D3D12 backend ────────────────────────────────
        //
        // Allocate the presentation texture on DAWN'S OWN D3D12 DEVICE, export
        // it as an NT handle, and open that handle on the sibling D3D11 device.
        // Both devices sit on the same adapter (guaranteed by
        // get_or_create_d3d11_device_on_dawn_adapter), so the two APIs are
        // views of ONE allocation - a cross-API view, NOT a cross-adapter copy.
        // Nothing is transferred.
        //
        // This used to be a flat "not implemented" bail, which left the D3D12
        // backend unable to present at all: every decoded frame fell back to a
        // full readback (measured 84-99 ms/frame). It is also why Tier B's
        // cross-adapter ingest was unusable in practice - Tier B requires the
        // D3D12 backend, so selecting it silently cost the whole present path.
        Microsoft::WRL::ComPtr<ID3D12Device> dawn12;
        if (backend == WGPUBackendType_D3D12)
            dawn12 = dawn::native::d3d12::GetD3D12Device(device);
        if (!dawn12) {
            LOG_ERROR("[minigpu_external] create_shared_output_texture: Dawn is on "
                      "neither the D3D11 nor the D3D12 backend; no cross-API "
                      "present path exists.");
            return nullptr;
        }

        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC rd{};
        rd.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width            = w;
        rd.Height           = h;
        rd.DepthOrArraySize = 1;
        rd.MipLevels        = 1;
        rd.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
        rd.SampleDesc.Count = 1;
        rd.Layout           = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        // 🔴 ALLOW_SIMULTANEOUS_ACCESS is REQUIRED, not defensive: Flutter's
        // compositor samples this texture on its own D3D11 device while Dawn
        // may still hold it, and without the flag the runtime forbids two
        // devices touching one resource concurrently. ALLOW_RENDER_TARGET
        // mirrors the D3D11 branch, which sets D3D11_BIND_RENDER_TARGET.
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                   D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

        Microsoft::WRL::ComPtr<ID3D12Resource> d3d12Tex;
        HRESULT hr12 = dawn12->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_SHARED, &rd,
            D3D12_RESOURCE_STATE_COMMON, /*pOptimizedClearValue=*/nullptr,
            IID_PPV_ARGS(&d3d12Tex));
        if (FAILED(hr12) || !d3d12Tex) {
            LOG_ERROR("[minigpu_external] CreateCommittedResource(SharedOutput, "
                      "D3D12) failed: 0x%08lX", (unsigned long)hr12);
            return nullptr;
        }

        // Export as an NT handle. It serves BOTH consumers: Dawn imports the
        // handle below, then the D3D11 device opens the same handle. We own it.
        HANDLE shareH = nullptr;
        hr12 = dawn12->CreateSharedHandle(d3d12Tex.Get(), /*pAttributes=*/nullptr,
                                          GENERIC_ALL, /*Name=*/nullptr, &shareH);
        if (FAILED(hr12) || !shareH) {
            LOG_ERROR("[minigpu_external] CreateSharedHandle(SharedOutput) failed: 0x%08lX",
                      (unsigned long)hr12);
            return nullptr;
        }

        // Import into Dawn THROUGH THE HANDLE rather than through the resource.
        //
        // 🔴 NOT `SharedTextureMemoryD3D12ResourceDescriptor` - that struct does
        // not exist in every Dawn this repo builds against. It is present in the
        // 2026-05 tree under external/dawn but ABSENT from the 2025-08 tree at
        // C:\dawn that actually gets linked, so using it compiles against one
        // checkout and fails against the other. The DXGI-shared-handle
        // descriptor has existed far longer and is what Tier B already relies
        // on, so this path works on both. Cost is one extra open at allocation,
        // never per frame.
        //
        // `useKeyedMutex=false` matches Tier B: synchronisation is by fence via
        // SharedTextureMemory Begin/EndAccess. A keyed mutex would additionally
        // require Flutter's compositor to acquire it, which it does not do.
        WGPUSharedTextureMemoryDXGISharedHandleDescriptor handleDesc{};
        handleDesc.chain.sType = WGPUSType_SharedTextureMemoryDXGISharedHandleDescriptor;
        handleDesc.handle        = shareH;
        handleDesc.useKeyedMutex = false;
        WGPUSharedTextureMemoryDescriptor smDesc12{};
        smDesc12.nextInChain = &handleDesc.chain;
        mem = wgpuDeviceImportSharedTextureMemory(device, &smDesc12);
        if (!mem) {
            LOG_ERROR("[minigpu_external] importSharedTextureMemory(DXGISharedHandle, "
                      "D3D12 output) failed.");
            CloseHandle(shareH);
            return nullptr;
        }

        // Open it on the sibling D3D11 device for the ID3D11Texture2D Flutter
        // needs. 🔴 THIS IS THE CALL a comment on MGPUSharedOutputTexture claims
        // returns E_INVALIDARG on many NVIDIA drivers "even when LUIDs match".
        // That was recorded during CROSS-ADAPTER work; here both devices are on
        // the same adapter by construction. If it does fail the HRESULT is
        // reported rather than swallowed, and the caller falls back to the CPU
        // present path instead of showing nothing.
        Microsoft::WRL::ComPtr<ID3D11Device1> d3d11_1;
        hr12 = d3d11Device->QueryInterface(IID_PPV_ARGS(&d3d11_1));
        if (SUCCEEDED(hr12) && d3d11_1) {
            hr12 = d3d11_1->OpenSharedResource1(shareH, IID_PPV_ARGS(&d3d11Tex));
        }
        if (FAILED(hr12) || !d3d11Tex) {
            LOG_ERROR("[minigpu_external] OpenSharedResource1(D3D12 shared output) "
                      "failed: 0x%08lX. Dawn is on D3D12 and this driver will not "
                      "open its resources on a D3D11 device - run with "
                      "MGPU_BACKEND=d3d11.", (unsigned long)hr12);
            CloseHandle(shareH);
            wgpuSharedTextureMemoryRelease(mem);
            return nullptr;
        }

        ntHandle     = shareH;
        ownsNtHandle = true;
        d3d12Texture = d3d12Tex;
        LOG_INFO("[minigpu_external] SharedOutputTexture (D3D12 cross-API) "
                 "d12=%p d11=%p mem=%p",
                 (void*)d3d12Tex.Get(), (void*)d3d11Tex.Get(), (void*)mem);
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

    // Dawn returns error objects, never null: an import it refused reports
    // Undefined properties. Refuse here so the caller falls back instead of
    // presenting through a texture every blit would reject.
    WGPUSharedTextureMemoryProperties memProps{};
    if (wgpuSharedTextureMemoryGetProperties(mem, &memProps) != WGPUStatus_Success ||
        memProps.format == WGPUTextureFormat_Undefined ||
        (memProps.usage & td.usage) != td.usage) {
        LOG_ERROR("[minigpu_external] SharedOutputTexture import was refused by Dawn "
                  "(format %d, usage 0x%llx).", static_cast<int>(memProps.format),
                  static_cast<unsigned long long>(memProps.usage));
        wgpuSharedTextureMemoryRelease(mem);
        // The D3D11 branch's legacy handle belongs to the texture.
        if (ownsNtHandle) CloseHandle(ntHandle);
        return nullptr;
    }
    WGPUTexture wgpuTex = wgpuSharedTextureMemoryCreateTexture(mem, &td);

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
    out->d3d12_texture       = d3d12Texture;   // null on the D3D11 backend
    out->nt_handle           = ntHandle;
    out->nt_handle_owned     = ownsNtHandle;
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
    if (!on_webgpu_worker()) {
        return run_on_webgpu_worker([src, dst] {
            return mgpuVideoTextureBGRAToRGBASharedOutput(src, dst);
        });
    }
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
    // becomes UNORDERED_ACCESS for the duration of the dispatch.  A refused
    // access opens nothing, so there is nothing to end.
    WGPUSharedTextureMemoryBeginAccessDescriptor beginDesc{};
    beginDesc.initialized = true; // assume previous content (or zero) is valid
    beginDesc.fenceCount  = 0;
    if (wgpuSharedTextureMemoryBeginAccess(dst->shared_mem, dst->texture,
                                           &beginDesc) != WGPUStatus_Success) {
        LOG_ERROR("[minigpu_external] BGRA->RGBA blit: BeginAccess failed.");
        return 0;
    }

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
    // The flush and this submit must not interleave with another thread's
    // batch recording; EndAccess closes the window the submit used.
    mgpu::unique_lock<mgpu::mutex> gpuLock(minigpu.getGpuMutex());
    minigpu.flushBatchLocked(); // submit pending batched compute before this blit
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuBindGroupRelease(bg);

    // EndAccess returns the resource to COMMON state so the D3D11 client
    // (the FFmpeg encoder / Flutter compositor) may read it.
    const bool ended = end_shared_access(dst->shared_mem, dst->texture);
    gpuLock.unlock();
    if (!ended) {
        LOG_ERROR("[minigpu_external] BGRA->RGBA blit: EndAccess failed.");
        return 0;
    }

    // Only the encoder shares Dawn's device; a compositor samples on its own
    // device, which nothing orders after our queue. So 1 means the dispatch +
    // copy have FINISHED on the GPU (waited here, on the worker). The test-only
    // MGPU_PRESENT_NO_WAIT skips it, as for the buffer copy below.
    if (present_wait_disabled()) {
        pump_dawn_events_nonblocking();
        return 1;
    }
    return wait_submitted_work(queue, "BGRA->RGBA blit: present copy") ? 1 : 0;
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
    if (!on_webgpu_worker()) {
        // On the worker the buffer is read behind its queued writers and
        // ahead of any queued destroy.
        return run_on_webgpu_worker([buf, dst] {
            return mgpuCopyBufferToSharedOutputTexture(buf, dst);
        });
    }
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
    if (wgpuSharedTextureMemoryBeginAccess(dst->shared_mem, dst->texture,
                                           &beginDesc) != WGPUStatus_Success) {
        LOG_ERROR("[minigpu_external] copyBufToShTex: BeginAccess failed.");
        return 0;
    }

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
    // The flush and this submit must not interleave with another thread's
    // batch recording; EndAccess closes the window the submit used.
    mgpu::unique_lock<mgpu::mutex> gpuLock(minigpu.getGpuMutex());
    minigpu.flushBatchLocked(); // submit pending batched compute before this blit
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuBindGroupRelease(bg);

    // EndAccess returns the resource to COMMON so the D3D11 client can read it.
    const bool ended = end_shared_access(dst->shared_mem, dst->texture);
    gpuLock.unlock();
    if (!ended) {
        LOG_ERROR("[minigpu_external] copyBufToShTex: EndAccess failed.");
        return 0;
    }

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
    //
    // MGPU_PRESENT_NO_WAIT=1 SKIPS this wait. It is a DELIBERATELY BROKEN mode
    // and exists only as the positive control for the present-correctness gate
    // (an external present gate): an oracle that
    // has never been shown to fail is not an oracle. Under it the consumer read
    // must start reporting stale/torn surfaces; if it does not, the gate is
    // measuring nothing. Never set it in production.
    // A timed-out or failed wait is a failed present: 1 promises final pixels.
    if (present_wait_disabled()) return 1;
    return wait_submitted_work(queue, "copyBufToShTex: present copy") ? 1 : 0;
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
    const bool queued = post_to_webgpu_worker([buf, dst, callback]() {
        // Through the helper: a throw must not escape the worker's task loop.
        int r = run_on_webgpu_worker([buf, dst] {
            return mgpuCopyBufferToSharedOutputTexture(buf, dst);
        });
        if (callback) callback(r);
    });
    if (!queued && callback) callback(0);
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
    const bool queued = post_to_webgpu_worker([src, dst, callback]() {
        int r = run_on_webgpu_worker([src, dst] {
            return mgpuVideoTextureBGRAToRGBASharedOutput(src, dst);
        });
        if (callback) callback(r);
    });
    if (!queued && callback) callback(0);
#else
    (void)src; (void)dst;
    if (callback) callback(0);
#endif
}

/// Port-delivered completions for the two blits above. Identical work; the
/// result travels as an int64 on a Dart port instead of through a function
/// pointer the caller cannot safely retire. See dart_port.h.
EXPORT void mgpuCopyBufferToSharedOutputTextureAsyncToPort(
        MGPUBuffer*                buf,
        MGPUSharedOutputTexture*   dst,
        int64_t                    port,
        int64_t                    token) {
#ifdef _WIN32
    const bool queued = post_to_webgpu_worker([buf, dst, port, token]() {
        const int r = run_on_webgpu_worker([buf, dst] {
            return mgpuCopyBufferToSharedOutputTexture(buf, dst);
        });
        mgpu::completionPost(port, token, r != 0);
    });
    if (!queued) mgpu::completionPost(port, token, false);
#else
    (void)buf; (void)dst;
    mgpu::completionPost(port, token, false);
#endif
}

EXPORT void mgpuVideoTextureBGRAToRGBASharedOutputAsyncToPort(
        MGPUVideoTexture*          src,
        MGPUSharedOutputTexture*   dst,
        int64_t                    port,
        int64_t                    token) {
#ifdef _WIN32
    const bool queued = post_to_webgpu_worker([src, dst, port, token]() {
        const int r = run_on_webgpu_worker([src, dst] {
            return mgpuVideoTextureBGRAToRGBASharedOutput(src, dst);
        });
        mgpu::completionPost(port, token, r != 0);
    });
    if (!queued) mgpu::completionPost(port, token, false);
#else
    (void)src; (void)dst;
    mgpu::completionPost(port, token, false);
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
    if (!on_webgpu_worker()) {
        return run_on_webgpu_worker([buf, dst] {
            return mgpuCopyBufferF32ToSharedOutputTexture(buf, dst);
        });
    }
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
    if (wgpuSharedTextureMemoryBeginAccess(dst->shared_mem, dst->texture,
                                           &beginDesc) != WGPUStatus_Success) {
        LOG_ERROR("[minigpu_external] copyBufF32ToShTex: BeginAccess failed.");
        return 0;
    }

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
    // The flush and this submit must not interleave with another thread's
    // batch recording; EndAccess closes the window the submit used.
    mgpu::unique_lock<mgpu::mutex> gpuLock(minigpu.getGpuMutex());
    minigpu.flushBatchLocked(); // submit pending batched compute before this blit
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuBindGroupRelease(bg);

    const bool ended = end_shared_access(dst->shared_mem, dst->texture);
    gpuLock.unlock();
    if (!ended) {
        LOG_ERROR("[minigpu_external] copyBufF32ToShTex: EndAccess failed.");
        return 0;
    }

    // Same present contract as the u32 copy: 1 means the copy has finished.
    if (present_wait_disabled()) {
        pump_dawn_events_nonblocking();
        return 1;
    }
    return wait_submitted_work(queue, "copyBufF32ToShTex: present copy") ? 1 : 0;
#else
    (void)buf; (void)dst;
    return 0;
#endif
}

/// Port-delivered async variant of mgpuCopyBufferF32ToSharedOutputTexture:
/// the copy and its completion wait run on the WebGPU worker, so the caller's
/// thread never blocks; the result is posted once the copy has finished.
EXPORT void mgpuCopyBufferF32ToSharedOutputTextureAsyncToPort(
        MGPUBuffer*                buf,
        MGPUSharedOutputTexture*   dst,
        int64_t                    port,
        int64_t                    token) {
#ifdef _WIN32
    const bool queued = post_to_webgpu_worker([buf, dst, port, token]() {
        const int r = run_on_webgpu_worker([buf, dst] {
            return mgpuCopyBufferF32ToSharedOutputTexture(buf, dst);
        });
        mgpu::completionPost(port, token, r != 0);
    });
    if (!queued) mgpu::completionPost(port, token, false);
#else
    (void)buf; (void)dst;
    mgpu::completionPost(port, token, false);
#endif
}

// Debug helper: read first pixel of the SharedOutputTexture from the D3D11
// side using the cached Dawn-adapter D3D11 device. Returns BGRA8 packed u32.
EXPORT uint32_t mgpuSharedOutputTextureDebugReadFirstPixel(
        MGPUSharedOutputTexture* tex) {
#ifdef _WIN32
    if (!tex || !tex->d3d11_texture) return 0xDEAD0001u;
    // Held across the whole read: the immediate context is not free-threaded,
    // and CopySubresourceRegion/Map/Unmap here must not interleave with
    // another thread's use of the same context.
    std::lock_guard<std::mutex> lk(g_d3d11_mutex);
    ID3D11Device* dev = get_or_create_d3d11_device_on_dawn_adapter_locked();
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
    if (!on_webgpu_worker()) {
        // Behind the queued presents, like every other Dawn use of [tex].
        return (uint32_t)run_on_webgpu_worker([tex] {
            return (int)mgpuSharedOutputTextureDebugReadFirstPixelDawn(tex);
        });
    }
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
    if (wgpuSharedTextureMemoryBeginAccess(tex->shared_mem, tex->texture,
                                           &beginDesc) != WGPUStatus_Success) {
        wgpuBufferRelease(staging);
        return 0xDEAD1007u;
    }

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
    {
        mgpu::lock_guard<mgpu::mutex> gpuLock(minigpu.getGpuMutex());
        minigpu.flushBatchLocked(); // submit pending batched compute before this blit
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        end_shared_access(tex->shared_mem, tex->texture);
    }

    // Wait for queue (timeout-protected: a hung driver returns an error
    // instead of freezing the caller forever).
    if (!wait_submitted_work(queue, "DebugReadFirstPixelDawn: queue submit")) {
        wgpuBufferRelease(staging);
        return 0xDEAD1005u;
    }

    // Map. The callback owns the completion state (a timed-out wait returns).
    using MapDone = std::shared_ptr<std::promise<WGPUMapAsyncStatus>>;
    auto mapDone = std::make_shared<std::promise<WGPUMapAsyncStatus>>();
    auto mapFut = mapDone->get_future();
    WGPUBufferMapCallbackInfo mci{};
    mci.mode = WGPUCallbackMode_AllowProcessEvents;
    mci.callback = [](WGPUMapAsyncStatus s, WGPUStringView,
                      void* ud1, void*) {
        std::unique_ptr<MapDone> owner(static_cast<MapDone*>(ud1));
        (*owner)->set_value(s);
    };
    mci.userdata1 = new MapDone(mapDone);
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

/* =========================================================================
 * CONSUMER-SIDE VERIFICATION READ
 *
 * mgpuSharedOutputTextureDebugReadFirstPixel() reads through
 * g_d3d11_device, which on this build IS Dawn's own ID3D11Device — the
 * PRODUCER. Its immediate context serialises against Dawn's submissions, so
 * it can never observe a half-written present. That makes it useless as a
 * tearing oracle: it is structurally blind to the exact bug the present's
 * OnSubmittedWorkDone wait exists to prevent.
 *
 * This function instead opens the texture by its legacy DXGI shared handle on
 * an INDEPENDENT ID3D11Device (same adapter, separate device) and reads the
 * whole surface — precisely how Flutter's compositor sees it. Nothing
 * synchronises that device against the producer except whatever the present
 * path itself guarantees, so:
 *   - if the present returned before its copy completed, this read lands on a
 *     surface that is partly frame N and partly frame N-1 and the checksum
 *     matches NEITHER frame's recon  => TORN (the race is caught),
 *   - if the copy had not started, it matches frame N-1's recon => STALE,
 *   - if it matches frame N's recon, the pixels the consumer can see are the
 *     pixels the decoder produced, at the instant the host would publish them.
 * Call it IMMEDIATELY after the decode returns: any wait inserted before the
 * read hands the producer free time and hides the race.
 *
 * FNV-1a is computed over RGBA-ordered bytes (the surface is BGRA8) so the
 * result is directly comparable to a checksum of the decoder's RGBA recon.
 * Returns 0 on failure.
 * ====================================================================== */
#ifdef _WIN32
// Same threading story as g_d3d11_*: unlocked lazy creation plus an immediate
// context that is not free-threaded. Lock order is ALWAYS
// g_consumer_mutex -> g_d3d11_mutex (never the reverse).
static std::mutex                                  g_consumer_mutex;
static Microsoft::WRL::ComPtr<ID3D11Device>        g_consumer_device;
static Microsoft::WRL::ComPtr<ID3D11DeviceContext> g_consumer_context;

// Caller MUST hold g_consumer_mutex.
static ID3D11Device* get_or_create_independent_consumer_device_locked() {
    if (g_consumer_device) return g_consumer_device.Get();
    ID3D11Device* producer = get_or_create_d3d11_device_on_dawn_adapter();
    if (!producer) return nullptr;

    // Same adapter as the producer, but a brand-new device.
    Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDev;
    if (FAILED(producer->QueryInterface(IID_PPV_ARGS(&dxgiDev))) || !dxgiDev)
        return nullptr;
    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    if (FAILED(dxgiDev->GetAdapter(&adapter)) || !adapter) return nullptr;

    D3D_FEATURE_LEVEL got{};
    HRESULT hr = D3D11CreateDevice(
        adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0,
        D3D11_SDK_VERSION, &g_consumer_device, &got, &g_consumer_context);
    if (FAILED(hr) || !g_consumer_device) {
        LOG_ERROR("[minigpu_external] consumer D3D11CreateDevice failed: 0x%08lX",
                  (unsigned long)hr);
        g_consumer_device.Reset();
        g_consumer_context.Reset();
        return nullptr;
    }
    return g_consumer_device.Get();
}

struct ConsumerSurface {
    Microsoft::WRL::ComPtr<ID3D11Texture2D> opened;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
    uint32_t w = 0, h = 0;
};
static std::unordered_map<void*, ConsumerSurface> g_consumer_surfaces; // g_consumer_mutex
#endif

// Drops the cached D3D11 devices when the context that produced them is torn
// down. On the D3D11 backend g_d3d11_device IS Dawn's own device, so keeping
// it across a destroy/re-init leaves a cache aliasing a dead device: the next
// create_shared_output_texture() compares it against the NEW Dawn device,
// concludes "not the same device", takes the cross-device NT-handle path on a
// device that no longer exists, and returns null. That is the second symptom
// of the process-global-context problem (seen in a serialized whole-package
// run, where the first suite's teardown poisoned every later suite).
// extern "C++": this sits inside the file's big extern "C" block, but it is a
// C++ function declared in buffer.h.
extern "C++" {
namespace mgpu {
void renderOnContextDestroyed(const void* context); // minigpu_render.cpp
void externalOnContextDestroyed(const void* context) {
    // Every context's render pipelines close with it.
    renderOnContextDestroyed(context);
    // Only the process-global context builds these caches; a secondary
    // Minigpu.forAdapter context must not clear them.
    if (context != static_cast<const void*>(&::minigpu)) return;
    // The toRGBA converters keep the destroyed device alive.
    release_rgba_pipelines();
#ifdef _WIN32
    {
        std::lock_guard<std::mutex> lk(g_consumer_mutex);
        g_consumer_surfaces.clear();
        g_consumer_context.Reset();
        g_consumer_device.Reset();
    }
    {
        std::lock_guard<std::mutex> lk(g_d3d11_mutex);
        g_d3d11_context.Reset();
        g_d3d11_device.Reset();
    }
    {
        // Tier A's private copies keep the old D3D11 device alive.
        std::lock_guard<std::mutex> lk(g_private_copy_mutex);
        g_private_copy = PrivateCopyRing{};
    }
    {
        // Tier B's Dawn objects belong to the device just destroyed (and keep
        // it alive); the producer-side D3D12 state does not. The next bridge
        // import refills the ring and re-imports the fence.
        std::lock_guard<std::mutex> lk(g_producerInfoMutex);
        for (auto& kv : g_producerInfoByLuid) {
            ProducerAdapterInfo& info = *kv.second;
            std::lock_guard<std::mutex> ring(info.tierBMutex);
            if (info.dawnFence) { wgpuSharedFenceRelease(info.dawnFence); info.dawnFence = nullptr; }
            release_tier_b_ring(&info);
        }
    }
#endif
}
} // namespace mgpu
} // extern "C++"

EXPORT uint32_t mgpuDebugConsumerChecksumSharedHandle(void*    sharedHandle,
                                                      uint32_t width,
                                                      uint32_t height) {
#ifdef _WIN32
    if (!sharedHandle || width == 0 || height == 0) return 0;
    // Held across the whole read: g_consumer_context is an immediate context
    // (not free-threaded) and g_consumer_surfaces is a plain unordered_map.
    std::lock_guard<std::mutex> lk(g_consumer_mutex);
    ID3D11Device* dev = get_or_create_independent_consumer_device_locked();
    if (!dev || !g_consumer_context) return 0;

    ConsumerSurface& cs = g_consumer_surfaces[sharedHandle];
    if (!cs.opened || cs.w != width || cs.h != height) {
        cs = ConsumerSurface{};
        // Legacy D3D11_RESOURCE_MISC_SHARED handle => OpenSharedResource
        // (NOT OpenSharedResource1, which is for NT handles).
        HRESULT hr = dev->OpenSharedResource(
            reinterpret_cast<HANDLE>(sharedHandle),
            __uuidof(ID3D11Texture2D),
            reinterpret_cast<void**>(cs.opened.GetAddressOf()));
        if (FAILED(hr) || !cs.opened) {
            LOG_ERROR("[minigpu_external] consumer OpenSharedResource failed: "
                      "0x%08lX", (unsigned long)hr);
            g_consumer_surfaces.erase(sharedHandle);
            return 0;
        }
        D3D11_TEXTURE2D_DESC sd{};
        sd.Width          = width;
        sd.Height         = height;
        sd.MipLevels      = 1;
        sd.ArraySize      = 1;
        sd.Format         = DXGI_FORMAT_B8G8R8A8_UNORM;
        sd.SampleDesc.Count = 1;
        sd.Usage          = D3D11_USAGE_STAGING;
        sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        hr = dev->CreateTexture2D(&sd, nullptr, &cs.staging);
        if (FAILED(hr) || !cs.staging) {
            g_consumer_surfaces.erase(sharedHandle);
            return 0;
        }
        cs.w = width;
        cs.h = height;
    }

    // No AcquireSync: the surface is legacy MISC_SHARED and carries no keyed
    // mutex. That is the point — this read is deliberately unsynchronised
    // against the producer.
    g_consumer_context->CopyResource(cs.staging.Get(), cs.opened.Get());

    D3D11_MAPPED_SUBRESOURCE m{};
    HRESULT hr = g_consumer_context->Map(cs.staging.Get(), 0,
                                         D3D11_MAP_READ, 0, &m);
    if (FAILED(hr) || !m.pData) return 0;

    uint32_t hash = 0x811c9dc5u;
    const uint8_t* base = static_cast<const uint8_t*>(m.pData);
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t* row = base + (size_t)y * m.RowPitch;
        for (uint32_t x = 0; x < width; ++x) {
            const uint8_t* p = row + (size_t)x * 4; // B,G,R,A
            const uint8_t rgba[4] = {p[2], p[1], p[0], p[3]};
            for (int i = 0; i < 4; ++i) {
                hash = (hash ^ rgba[i]) * 0x01000193u;
            }
        }
    }
    g_consumer_context->Unmap(cs.staging.Get(), 0);
    return hash;
#else
    (void)sharedHandle; (void)width; (void)height;
    return 0;
#endif
}

#ifdef _WIN32
static void destroy_shared_output_texture_on_worker(MGPUSharedOutputTexture* tex) {
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
    // 🔴 CLOSE THE HANDLE ONLY IF WE MADE IT. On the D3D11 backend nt_handle is
    // a LEGACY DXGI share handle from IDXGIResource::GetSharedHandle(), which
    // MSDN says is owned by the texture - closing it is a double free, and this
    // function deliberately did not. The D3D12 backend instead exports a real
    // NT handle via ID3D12Device::CreateSharedHandle(), which leaks unless it
    // IS closed. nt_handle_owned records which one this is.
    if (tex->nt_handle_owned && tex->nt_handle) {
        CloseHandle(tex->nt_handle);
    }
    delete tex;
}
#endif

EXPORT void mgpuDestroySharedOutputTexture(MGPUSharedOutputTexture* tex) {
#ifdef _WIN32
    if (!tex) return;
    // FIFO behind every queued blit into [tex]: those tasks hold only the raw
    // pointer, so freeing it inline here was a use-after-free.
    if (!post_to_webgpu_worker([tex]() {
            destroy_shared_output_texture_on_worker(tex);
        })) {
        // A refused queue does not prove the queued blits have run: keep the
        // texture rather than free it under them.
        LOG_ERROR("shared output texture retained: GPU enqueue failed");
    }
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
