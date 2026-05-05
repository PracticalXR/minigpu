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
    std::cout << "\n-- testD3D11SharedHandleImport --" << std::endl;

    // Skip if D3D11 shared handles not supported by this Dawn build
    if (!mgpuIsExternalContentTypeSupported(
            MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE)) {
        std::cout << "  Skipped (D3D11 shared handles not supported)." << std::endl;
        return;
    }

    // Create a D3D11 device
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
        D3D11_SDK_VERSION, &device, &featureLevel, &context);
    if (FAILED(hr)) {
        std::cout << "  Skipped (D3D11 device creation failed, hr=0x"
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
               "D3D11 shared handle import returns non-null texture");

    mgpuDestroyVideoTexture(gpuTex);
    std::cout << "testD3D11SharedHandleImport PASSED" << std::endl;
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
#endif
    std::cout << "===== All External Texture Tests Passed =====" << std::endl;
}
