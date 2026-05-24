// external_texture_test.cpp
//
// Tests for mgpuImportVideoFrame and related external texture API.
// Built as part of the minigpu_test target (see CMakeLists.txt).
//
// On Windows: tests D3D11 shared-handle import (requires D3D11 runtime).
// On all platforms: tests CPU-fallback import with a synthetic RGBA buffer.
//
// Run with:
//   cmake -B build -S src
//   cmake --build build
//   ./build/minigpu_test

#include "../include/minigpu.h"
#include "../include/minigpu_external.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void expectTrue(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << std::endl;
        std::abort();
    }
    std::cout << "[PASS] " << msg << std::endl;
}

// ---------------------------------------------------------------------------
// CPU fallback import: 4x4 RGBA8 synthetic frame
// ---------------------------------------------------------------------------

void testCpuImport() {
    std::cout << "\n-- testCpuImport --" << std::endl;

    const uint32_t W = 4, H = 4;
    const uint32_t stride = W * 4; // RGBA8 = 4 bytes/pixel

    // Fill a 4x4 magenta frame: R=255,G=0,B=255,A=255
    std::vector<uint8_t> pixels(stride * H);
    for (uint32_t y = 0; y < H; y++) {
        for (uint32_t x = 0; x < W; x++) {
            uint8_t* p = pixels.data() + y * stride + x * 4;
            p[0] = 0xFF; p[1] = 0x00; p[2] = 0xFF; p[3] = 0xFF;
        }
    }

    MGPUExternalVideoBuffer buf{};
    buf.content_type  = MGPU_EXTERNAL_CONTENT_TYPE_CPU;
    buf.pixel_format  = MGPU_EXTERNAL_PIXEL_FORMAT_RGBA32;
    buf.width         = W;
    buf.height        = H;
    buf.num_planes    = 1;
    buf.planes[0].data_ptr      = pixels.data();
    buf.planes[0].width         = W;
    buf.planes[0].height        = H;
    buf.planes[0].stride_bytes  = stride;
    buf.planes[0].dmabuf_fd     = -1;

    MGPUVideoTexture* tex = mgpuImportVideoFrame(&buf);
    expectTrue(tex != nullptr, "CPU import returns non-null texture");

    // Convert to RGBA buffer and read back
    MGPUBuffer* outBuf = mgpuVideoTextureToRGBA(tex);
    expectTrue(outBuf != nullptr, "videoTextureToRGBA returns non-null buffer");

    std::vector<uint8_t> readback(W * H * 4, 0);
    mgpuReadSyncUint8(outBuf, readback.data(),
                      static_cast<int>(readback.size()), 0);

    // First pixel should be magenta (255, 0, 255, 255)
    expectTrue(readback[0] == 0xFF, "RGBA[0].R == 0xFF");
    expectTrue(readback[1] == 0x00, "RGBA[0].G == 0x00");
    expectTrue(readback[2] == 0xFF, "RGBA[0].B == 0xFF");
    expectTrue(readback[3] == 0xFF, "RGBA[0].A == 0xFF");

    mgpuDestroyBuffer(outBuf);
    mgpuDestroyVideoTexture(tex);
    std::cout << "testCpuImport PASSED" << std::endl;
}

// ---------------------------------------------------------------------------
// CPU NV12 import: 4x4 NV12 frame (Y=0x80, UV=0x80 → grey)
// ---------------------------------------------------------------------------

void testCpuNv12Import() {
    std::cout << "\n-- testCpuNv12Import --" << std::endl;

    const uint32_t W = 4, H = 4;

    // Plane 0: Y (4x4, stride=4)
    std::vector<uint8_t> yPlane(W * H, 0x80);   // mid-grey luma
    // Plane 1: UV interleaved (2x2, stride=4)
    std::vector<uint8_t> uvPlane(W * (H / 2), 0x80); // neutral chroma

    MGPUExternalVideoBuffer buf{};
    buf.content_type = MGPU_EXTERNAL_CONTENT_TYPE_CPU;
    buf.pixel_format = MGPU_EXTERNAL_PIXEL_FORMAT_NV12;
    buf.width        = W;
    buf.height       = H;
    buf.num_planes   = 2;

    buf.planes[0].data_ptr     = yPlane.data();
    buf.planes[0].width        = W;
    buf.planes[0].height       = H;
    buf.planes[0].stride_bytes = W;
    buf.planes[0].dmabuf_fd    = -1;

    buf.planes[1].data_ptr     = uvPlane.data();
    buf.planes[1].width        = W / 2;
    buf.planes[1].height       = H / 2;
    buf.planes[1].stride_bytes = W; // interleaved U+V
    buf.planes[1].dmabuf_fd    = -1;

    MGPUVideoTexture* tex = mgpuImportVideoFrame(&buf);
    expectTrue(tex != nullptr, "NV12 CPU import returns non-null texture");

    MGPUBuffer* outBuf = mgpuVideoTextureToRGBA(tex);
    expectTrue(outBuf != nullptr, "NV12 videoTextureToRGBA returns non-null buffer");

    std::vector<uint8_t> readback(W * H * 4, 0);
    mgpuReadSyncUint8(outBuf, readback.data(),
                      static_cast<int>(readback.size()), 0);

    // Y=0x80, UV=0x80,0x80 → BT.709 full-range → approximately (128,128,128)
    // Allow ±8 tolerance for GPU rounding
    auto near = [](uint8_t v, uint8_t expected) {
        return std::abs((int)v - (int)expected) <= 8;
    };
    expectTrue(near(readback[0], 128), "NV12 RGBA[0].R ≈ 128");
    expectTrue(near(readback[1], 128), "NV12 RGBA[0].G ≈ 128");
    expectTrue(near(readback[2], 128), "NV12 RGBA[0].B ≈ 128");

    mgpuDestroyBuffer(outBuf);
    mgpuDestroyVideoTexture(tex);
    std::cout << "testCpuNv12Import PASSED" << std::endl;
}

// ---------------------------------------------------------------------------
// Capability query
// ---------------------------------------------------------------------------

void testCapabilityQueries() {
    std::cout << "\n-- testCapabilityQueries --" << std::endl;

    // CPU must always be supported
    expectTrue(
        mgpuIsExternalContentTypeSupported(MGPU_EXTERNAL_CONTENT_TYPE_CPU) != 0,
        "CPU content type is supported");
    expectTrue(
        mgpuIsExternalPixelFormatSupported(MGPU_EXTERNAL_PIXEL_FORMAT_RGBA32) != 0,
        "RGBA32 pixel format is supported");
    expectTrue(
        mgpuIsExternalPixelFormatSupported(MGPU_EXTERNAL_PIXEL_FORMAT_NV12) != 0,
        "NV12 pixel format is supported");

    // Platform-specific: just print, don't fail
    const char* d3d11 = mgpuIsExternalContentTypeSupported(
        MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE) ? "YES" : "NO";
    const char* dmabuf = mgpuIsExternalContentTypeSupported(
        MGPU_EXTERNAL_CONTENT_TYPE_DMABUF) ? "YES" : "NO";
    const char* iosurface = mgpuIsExternalContentTypeSupported(
        MGPU_EXTERNAL_CONTENT_TYPE_METAL_IOSURFACE) ? "YES" : "NO";
    std::cout << "  D3D11 shared handle: " << d3d11 << std::endl;
    std::cout << "  DMA-BUF:             " << dmabuf << std::endl;
    std::cout << "  IOSurface:           " << iosurface << std::endl;

    std::cout << "testCapabilityQueries PASSED" << std::endl;
}

// ---------------------------------------------------------------------------
// Windows D3D11 shared-handle import
// ---------------------------------------------------------------------------

#ifdef _WIN32
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

void testD3D11SharedHandleImport() {
    std::cout << "\n-- testTierA_SameAdapter (D3D11 same-adapter zero-copy) --" << std::endl;

    // Tier A does NOT require WGPUFeatureName_SharedTextureMemoryDXGISharedHandle.
    // It opens the NT handle with OpenSharedResource1 on Dawn's own D3D11 device,
    // copies to a private texture, then imports via SharedTextureMemoryD3D11Texture2DDescriptor.
    // We therefore skip the feature-capability check and just try it.
    //
    // To guarantee we land on Tier A (not Tier C), we create the texture on
    // the SAME physical adapter that Dawn selected — the dGPU (highest VRAM).

    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        std::cout << "  Skipped (CreateDXGIFactory1 failed)." << std::endl;
        return;
    }

    // Find the discrete / highest-VRAM adapter — mirrors Dawn's auto-select logic.
    ComPtr<IDXGIAdapter1> dawnAdapter;
    {
        SIZE_T bestVram = 0;
        bool   bestIsDiscrete = false;
        ComPtr<IDXGIAdapter1> a;
        for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, &a)); ++i, a.Reset()) {
            DXGI_ADAPTER_DESC1 d{}; a->GetDesc1(&d);
            if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            bool isDiscrete = d.DedicatedVideoMemory > (256ull * 1024 * 1024)
                           && d.DedicatedVideoMemory >= d.SharedSystemMemory / 2;
            bool better = !dawnAdapter
                || (isDiscrete && !bestIsDiscrete)
                || (isDiscrete == bestIsDiscrete && d.DedicatedVideoMemory > bestVram);
            if (better) { dawnAdapter = a; bestVram = d.DedicatedVideoMemory; bestIsDiscrete = isDiscrete; }
        }
    }
    if (!dawnAdapter) {
        std::cout << "  Skipped (no real hardware adapter found)." << std::endl;
        return;
    }
    {
        DXGI_ADAPTER_DESC1 d{}; dawnAdapter->GetDesc1(&d);
        char buf[128]{};
        WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, buf, sizeof(buf), nullptr, nullptr);
        std::cout << "  Producer (same as Dawn) adapter: " << buf << std::endl;
    }

    // Create a D3D11 device on the same adapter Dawn uses.
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(
        dawnAdapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
        D3D11_SDK_VERSION, &device, &featureLevel, &context);
    if (FAILED(hr)) {
        std::cout << "  Skipped (D3D11CreateDevice failed, hr=0x"
                  << std::hex << hr << std::dec << ")." << std::endl;
        return;
    }

    // Create a shared texture (BGRA8, 64x64)
    const UINT W = 64, H = 64;
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width            = W;
    desc.Height           = H;
    desc.MipLevels        = 1;
    desc.ArraySize        = 1;
    desc.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage            = D3D11_USAGE_DEFAULT;
    desc.BindFlags        = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.MiscFlags        = D3D11_RESOURCE_MISC_SHARED_NTHANDLE
                          | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    ComPtr<ID3D11Texture2D> tex2d;
    hr = device->CreateTexture2D(&desc, nullptr, &tex2d);
    if (FAILED(hr)) {
        std::cout << "  Skipped (CreateTexture2D failed, hr=0x"
                  << std::hex << hr << std::dec << ")." << std::endl;
        return;
    }

    // Get the NT shared handle
    ComPtr<IDXGIResource1> dxgiRes;
    hr = tex2d.As(&dxgiRes);
    expectTrue(SUCCEEDED(hr), "D3D11: tex2d.As<IDXGIResource1> succeeded");

    HANDLE sharedHandle = nullptr;
    hr = dxgiRes->CreateSharedHandle(
        nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
        nullptr, &sharedHandle);
    expectTrue(SUCCEEDED(hr), "D3D11: CreateSharedHandle succeeded");

    // Build MGPUExternalVideoBuffer
    MGPUExternalVideoBuffer buf{};
    buf.content_type = MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE;
    buf.pixel_format = MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32;
    buf.width        = W;
    buf.height       = H;
    buf.num_planes   = 1;
    buf.planes[0].data_ptr        = sharedHandle;
    buf.planes[0].width           = W;
    buf.planes[0].height          = H;
    buf.planes[0].stride_bytes    = W * 4;
    buf.planes[0].subresource_index = 0;
    buf.planes[0].dmabuf_fd       = -1;

    MGPUVideoTexture* gpuTex = mgpuImportVideoFrame(&buf);
    CloseHandle(sharedHandle);

    expectTrue(gpuTex != nullptr,
               "TierA: same-adapter import returns non-null texture");

    mgpuDestroyVideoTexture(gpuTex);
    std::cout << "testTierA_SameAdapter PASSED" << std::endl;
}
#endif // _WIN32

// ---------------------------------------------------------------------------
// Tier C test: cross-adapter CPU bridge
//
// Creates a D3D11 shared-NT-handle texture on the adapter that is NOT the
// one Dawn will select, then imports it.  Because OpenSharedResource1 returns
// E_INVALIDARG on the wrong adapter, Tier A falls through to Tier C (the CPU
// staging bridge) automatically.
//
// Skipped when fewer than two real hardware adapters are present.
// ---------------------------------------------------------------------------
#ifdef _WIN32
void testTierC_CpuBridge() {
    std::cout << "\n-- testTierC_CpuBridge (cross-adapter CPU bridge) --" << std::endl;

    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        std::cout << "  Skipped (CreateDXGIFactory1 failed)." << std::endl;
        return;
    }

    // Collect all real (non-software) adapters.
    std::vector<ComPtr<IDXGIAdapter1>> adapters;
    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, &a)); ++i, a.Reset()) {
        DXGI_ADAPTER_DESC1 d{};
        a->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        adapters.push_back(a);
    }

    if (adapters.size() < 2) {
        std::cout << "  Skipped (need >=2 real adapters; found "
                  << adapters.size() << ")." << std::endl;
        return;
    }

    // Pick the adapter that Dawn would NOT use (second in the list).
    // Dawn's auto-select picks the dGPU (highest VRAM / discrete), which is
    // usually adapters[0] from DXGI.  Using adapters[1] guarantees mismatch.
    ComPtr<IDXGIAdapter1>& otherAdapter = adapters[1];
    {
        DXGI_ADAPTER_DESC1 d{};
        otherAdapter->GetDesc1(&d);
        char nameBuf[128]{};
        WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, nameBuf, sizeof(nameBuf), nullptr, nullptr);
        std::cout << "  Producer adapter: " << nameBuf << std::endl;
    }

    // Create a D3D11 device on that other adapter.
    ComPtr<ID3D11Device>        prodDevice;
    ComPtr<ID3D11DeviceContext> prodCtx;
    D3D_FEATURE_LEVEL fl{};
    HRESULT hr = D3D11CreateDevice(
        otherAdapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
        D3D11_SDK_VERSION, &prodDevice, &fl, &prodCtx);
    if (FAILED(hr)) {
        std::cout << "  Skipped (D3D11CreateDevice on other adapter failed: 0x"
                  << std::hex << hr << std::dec << ")." << std::endl;
        return;
    }

    // Create a shared-NT texture on the OTHER adapter.
    const UINT W = 64, H = 64;
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width            = W;  desc.Height       = H;
    desc.MipLevels        = 1;  desc.ArraySize    = 1;
    desc.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage            = D3D11_USAGE_DEFAULT;
    desc.BindFlags        = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.MiscFlags        = D3D11_RESOURCE_MISC_SHARED_NTHANDLE
                          | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    ComPtr<ID3D11Texture2D> srcTex;
    hr = prodDevice->CreateTexture2D(&desc, nullptr, &srcTex);
    if (FAILED(hr)) {
        std::cout << "  Skipped (CreateTexture2D on other adapter failed: 0x"
                  << std::hex << hr << std::dec << ")." << std::endl;
        return;
    }

    ComPtr<IDXGIResource1> dxgiRes;
    hr = srcTex.As(&dxgiRes);
    expectTrue(SUCCEEDED(hr), "TierC: srcTex.As<IDXGIResource1> succeeded");

    HANDLE sharedHandle = nullptr;
    hr = dxgiRes->CreateSharedHandle(
        nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
        nullptr, &sharedHandle);
    expectTrue(SUCCEEDED(hr), "TierC: CreateSharedHandle succeeded");

    // Import — will fail Tier A (wrong adapter) and route to Tier C.
    MGPUExternalVideoBuffer buf{};
    buf.content_type          = MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE;
    buf.pixel_format          = MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32;
    buf.width                 = W;
    buf.height                = H;
    buf.num_planes            = 1;
    buf.planes[0].data_ptr    = sharedHandle;
    buf.planes[0].width       = W;
    buf.planes[0].height      = H;
    buf.planes[0].stride_bytes = W * 4;
    buf.planes[0].dmabuf_fd   = -1;

    MGPUVideoTexture* gpuTex = mgpuImportVideoFrame(&buf);
    CloseHandle(sharedHandle);

    expectTrue(gpuTex != nullptr,
        "TierC: cross-adapter CPU bridge returns non-null texture");

    mgpuDestroyVideoTexture(gpuTex);
    std::cout << "testTierC_CpuBridge PASSED" << std::endl;
}
#endif // _WIN32

// ---------------------------------------------------------------------------
// Tier B test: D3D12 cross-adapter GPU bridge
//
// Requires:
//   - MGPU_BACKEND=d3d12  (or Dawn already initialised with D3D12)
//   - >= 2 real hardware adapters
//   - Producer adapter supports CrossAdapterRowMajorTextureSupported
//
// Creates a D3D12 cross-adapter committed resource on the non-Dawn adapter,
// wraps it in a D3D11on12 device, exports an NT handle, and imports it via
// the Tier B code path in minigpu_external.cpp.
// ---------------------------------------------------------------------------
#ifdef _WIN32
#include <d3d12.h>

void testTierB_D3D12Bridge() {
    std::cout << "\n-- testTierB_D3D12Bridge (D3D12 cross-adapter GPU bridge) --" << std::endl;

    // Tier B requires Dawn on the D3D12 backend.  We switch to it for this
    // test by destroying the current context, setting MGPU_BACKEND=d3d12,
    // reinitialising, running the test, then restoring D3D11 so subsequent
    // tests run on the correct backend.
    const char* origBe = std::getenv("MGPU_BACKEND");
    const std::string origBeStr = origBe ? origBe : "";
    bool wasAlreadyD3D12 = (origBeStr == "d3d12" || origBeStr == "D3D12");

    if (!wasAlreadyD3D12) {
        std::cout << "  Reinitialising context with D3D12 backend for Tier B test..." << std::endl;
        mgpuDestroyContext();
        _putenv_s("MGPU_BACKEND", "d3d12");
        mgpuInitializeContext();
    }

    // Helper lambda: restore D3D11 backend if we changed it, then return.
    // Used by all early-exit paths below.
    auto skipAndRestore = [&](const char* reason) {
        std::cout << "  Skipped (" << reason << ")." << std::endl;
        if (!wasAlreadyD3D12) {
            mgpuDestroyContext();
            _putenv_s("MGPU_BACKEND", "d3d11");
            mgpuInitializeContext();
        }
    };

    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        skipAndRestore("CreateDXGIFactory1 failed"); return;
    }

    std::vector<ComPtr<IDXGIAdapter1>> adapters;
    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, &a)); ++i, a.Reset()) {
        DXGI_ADAPTER_DESC1 d{}; a->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        adapters.push_back(a);
    }
    if (adapters.size() < 2) {
        char msg[64]; snprintf(msg, sizeof(msg), "need >=2 real adapters; found %zu", adapters.size());
        skipAndRestore(msg); return;
    }

    // Create a D3D12 device on the non-Dawn adapter (adapters[1]).
    ComPtr<ID3D12Device> prod12;
    if (FAILED(D3D12CreateDevice(adapters[1].Get(), D3D_FEATURE_LEVEL_11_0,
                                  IID_PPV_ARGS(&prod12))) || !prod12) {
        skipAndRestore("D3D12CreateDevice on other adapter failed"); return;
    }

    // Check cross-adapter row-major texture support.
    D3D12_FEATURE_DATA_D3D12_OPTIONS opts{};
    prod12->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts, sizeof(opts));
    if (!opts.CrossAdapterRowMajorTextureSupported) {
        skipAndRestore("producer adapter lacks CrossAdapterRowMajorTextureSupported"); return;
    }

    const UINT W = 64, H = 64;

    // Allocate a cross-adapter committed resource on the producer adapter.
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type                 = D3D12_HEAP_TYPE_DEFAULT;
    hp.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension          = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width              = W;
    rd.Height             = H;
    rd.DepthOrArraySize   = 1;
    rd.MipLevels          = 1;
    rd.Format             = DXGI_FORMAT_B8G8R8A8_UNORM;
    rd.SampleDesc.Count   = 1;
    rd.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags              = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER
                          | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

    ComPtr<ID3D12Resource> crossTex;
    HRESULT hr = prod12->CreateCommittedResource(
        &hp,
        D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER,
        &rd,
        D3D12_RESOURCE_STATE_COMMON,
        nullptr,
        IID_PPV_ARGS(&crossTex));
    if (FAILED(hr) || !crossTex) {
        char msg[64]; snprintf(msg, sizeof(msg), "CreateCommittedResource failed: 0x%08lX", (unsigned long)hr);
        skipAndRestore(msg); return;
    }

    // Export an NT handle for the cross-adapter resource.
    HANDLE sharedHandle = nullptr;
    hr = prod12->CreateSharedHandle(crossTex.Get(), nullptr,
                                     GENERIC_ALL, nullptr, &sharedHandle);
    expectTrue(SUCCEEDED(hr), "TierB: D3D12 CreateSharedHandle succeeded");

    // Import — Tier A fails (wrong adapter + D3D12 backend can't open D3D11
    // handle), Tier B should succeed because the handle IS a D3D12 cross-
    // adapter resource.
    MGPUExternalVideoBuffer buf{};
    buf.content_type          = MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE;
    buf.pixel_format          = MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32;
    buf.width                 = W;
    buf.height                = H;
    buf.num_planes            = 1;
    buf.planes[0].data_ptr    = sharedHandle;
    buf.planes[0].width       = W;
    buf.planes[0].height      = H;
    buf.planes[0].stride_bytes = W * 4;
    buf.planes[0].dmabuf_fd   = -1;

    MGPUVideoTexture* gpuTex = mgpuImportVideoFrame(&buf);
    CloseHandle(sharedHandle);

    expectTrue(gpuTex != nullptr,
        "TierB: D3D12 cross-adapter GPU bridge returns non-null texture");

    mgpuDestroyVideoTexture(gpuTex);

    // Restore D3D11 backend so remaining tests use the correct context.
    if (!wasAlreadyD3D12) {
        std::cout << "  Restoring D3D11 backend..." << std::endl;
        mgpuDestroyContext();
        _putenv_s("MGPU_BACKEND", "d3d11");
        mgpuInitializeContext();
    }

    std::cout << "testTierB_D3D12Bridge PASSED" << std::endl;
}
#endif // _WIN32

// ---------------------------------------------------------------------------
// External texture test entry point (called from minigpu_test.cpp main)
// ---------------------------------------------------------------------------

void runExternalTextureTests() {
    std::cout << "\n===== External Texture Tests =====" << std::endl;
    testCapabilityQueries();
    testCpuImport();
    testCpuNv12Import();
#ifdef _WIN32
    testD3D11SharedHandleImport();
    testTierC_CpuBridge();
    testTierB_D3D12Bridge();
#endif
    std::cout << "===== All External Texture Tests Passed =====" << std::endl;
}
