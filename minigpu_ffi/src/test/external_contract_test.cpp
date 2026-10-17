// Native gate for the external-texture contracts (default D3D11 backend).
//
// Shared output (Windows): every blit into a MGPUSharedOutputTexture - u32
// buffer, f32 buffer and BGRA video texture, sync and async - reports success
// only once the GPU has finished, so an INDEPENDENT D3D11 device (the way a
// compositor samples it) sees the new frame the moment the call or its
// callback returns. A destroy issued while a blit is still queued runs after
// that blit instead of freeing what it reads.
//
// Each frame uses a new colour, so a stale or torn surface fails the checksum.
//
// Imports: toRGBA reads every declared format through its own layout (GRAY8,
// three-plane I420), a frame carrying a producer fence is refused, and the
// cross-adapter CPU bridge (Windows, producer = the adapter with the least
// dedicated memory) maps each DXGI format, uploads NV12 as two planes,
// refuses a declaration that disagrees with the surface, and keeps two live
// frames from one producer apart. The same-adapter private copy keeps more
// live frames apart than its ring holds, a read still queued when its frame
// is destroyed keeps that frame's pixels, and sequential frames reuse the
// ring instead of allocating.
#include "../include/minigpu.h"
#include "../include/minigpu_external.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <d3d11.h>
#  include <dxgi1_2.h>
#  include <wrl/client.h>
#  pragma comment(lib, "d3d11.lib")
#  pragma comment(lib, "dxgi.lib")
#endif

extern "C" {
// Exported function-pointer variants (not in the public header).
void mgpuCopyBufferToSharedOutputTextureAsync(MGPUBuffer*, MGPUSharedOutputTexture*,
                                              void (*)(int));
void mgpuVideoTextureBGRAToRGBASharedOutputAsync(MGPUVideoTexture*,
                                                 MGPUSharedOutputTexture*,
                                                 void (*)(int));
}

static void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

static std::atomic<int> privateCopies{0};

static void recordLog(int level, const char* message) {
    if (std::strstr(message, "private copy allocated")) ++privateCopies;
    if (level >= 3) std::cerr << message << '\n';
    mgpuFreeLogMessage(message);
}

struct Rgba { uint8_t r, g, b, a; };

// Reads a toRGBA result, compares every pixel within [tolerance], frees it.
static void expectRgba(MGPUBuffer* out, uint32_t w, uint32_t h,
                       const std::function<Rgba(uint32_t, uint32_t)>& want,
                       int tolerance, const char* what) {
    require(out != nullptr, what);
    std::vector<uint8_t> got(size_t(w) * h * 4);
    mgpuReadSyncUint8(out, got.data(), int(got.size()), 0);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            const Rgba c = want(x, y);
            const uint8_t expected[4] = {c.r, c.g, c.b, c.a};
            const uint8_t* seen = &got[(size_t(y) * w + x) * 4];
            for (int k = 0; k < 4; ++k) {
                if (std::abs(int(seen[k]) - int(expected[k])) > tolerance) {
                    std::cerr << "FAIL: " << what << ": pixel (" << x << ',' << y
                              << ") channel " << k << " = " << int(seen[k])
                              << ", want " << int(expected[k]) << '\n';
                    std::exit(1);
                }
            }
        }
    }
    mgpuDestroyBuffer(out);
}

// The documented BT.709 full-range conversion, truncated like the shader.
static uint8_t unorm8(float v) { return uint8_t((v < 0.f ? 0.f : v > 1.f ? 1.f : v) * 255.0f); }
static Rgba bt709(uint8_t yb, uint8_t ub, uint8_t vb) {
    const float y = yb / 255.0f, cb = ub / 255.0f - 0.5f, cr = vb / 255.0f - 0.5f;
    return {unorm8(y + 1.5748f * cr), unorm8(y - 0.1873f * cb - 0.4681f * cr),
            unorm8(y + 1.8556f * cb), 255};
}

static MGPUExternalVideoBuffer cpuFrame(MGPUExternalPixelFormat format, uint32_t w, uint32_t h) {
    MGPUExternalVideoBuffer input{};
    input.content_type = MGPU_EXTERNAL_CONTENT_TYPE_CPU;
    input.pixel_format = format;
    input.width = w;
    input.height = h;
    input.fence.sync_fd = -1;
    return input;
}

static void setPlane(MGPUExternalVideoBuffer& input, int i, std::vector<uint8_t>& bytes,
                     uint32_t w, uint32_t h, uint32_t stride) {
    input.planes[i].data_ptr = bytes.data();
    input.planes[i].width = w;
    input.planes[i].height = h;
    input.planes[i].stride_bytes = stride;
    input.planes[i].dmabuf_fd = -1;
    input.num_planes = uint32_t(i + 1);
}

static void testCpuFormatsAndFences() {
    const uint32_t w = 6, h = 4;
    std::vector<uint8_t> gray(size_t(w) * h);
    for (size_t i = 0; i < gray.size(); ++i) gray[i] = uint8_t(i * 11 + 1);
    MGPUExternalVideoBuffer input = cpuFrame(MGPU_EXTERNAL_PIXEL_FORMAT_GRAY8, w, h);
    setPlane(input, 0, gray, w, h, w);
    MGPUVideoTexture* tex = mgpuImportVideoFrame(&input);
    require(tex != nullptr, "GRAY8 CPU import");
    expectRgba(mgpuVideoTextureToRGBA(tex), w, h, [&](uint32_t x, uint32_t y) {
        const uint8_t v = gray[size_t(y) * w + x];
        return Rgba{v, v, v, 255};
    }, 0, "GRAY8 toRGBA is opaque grey");
    mgpuDestroyVideoTexture(tex);

    // Chroma differs per 2x2 block and U != V, so a wrong plane order or
    // chroma mapping shows.
    const uint32_t cw = w / 2, ch = h / 2;
    std::vector<uint8_t> yp(size_t(w) * h), up(size_t(cw) * ch), vp(size_t(cw) * ch);
    for (size_t i = 0; i < yp.size(); ++i) yp[i] = uint8_t(40 + i * 7);
    for (size_t i = 0; i < up.size(); ++i) {
        up[i] = uint8_t(60 + i * 40);
        vp[i] = uint8_t(200 - i * 30);
    }
    input = cpuFrame(MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_RGB_PLANES, w, h);
    setPlane(input, 0, yp, w, h, w);
    setPlane(input, 1, up, cw, ch, cw);
    setPlane(input, 2, vp, cw, ch, cw);
    tex = mgpuImportVideoFrame(&input);
    require(tex != nullptr, "I420 CPU import");
    expectRgba(mgpuVideoTextureToRGBA(tex), w, h, [&](uint32_t x, uint32_t y) {
        const size_t c = size_t(y / 2) * cw + x / 2;
        return bt709(yp[size_t(y) * w + x], up[c], vp[c]);
    }, 2, "I420 toRGBA uses all three planes");
    mgpuDestroyVideoTexture(tex);

    // Fences are not imported, so a frame carrying one is refused.
    std::vector<uint8_t> rgba(4 * 4 * 4, 0x7F);
    input = cpuFrame(MGPU_EXTERNAL_PIXEL_FORMAT_RGBA32, 4, 4);
    setPlane(input, 0, rgba, 4, 4, 16);
    input.fence.d3d11_fence = &input;
    require(mgpuImportVideoFrame(&input) == nullptr, "frame with a D3D11 fence refused");
    input.fence.d3d11_fence = nullptr;
    input.fence.sync_fd = 7;
    require(mgpuImportVideoFrame(&input) == nullptr, "frame with a sync_fd refused");
    input.fence.sync_fd = 0; // 0 and -1 both mean none
    tex = mgpuImportVideoFrame(&input);
    require(tex != nullptr, "frame with an empty fence imported");
    mgpuDestroyVideoTexture(tex);
    std::cout << "PASS CPU formats (GRAY8, I420) and fence refusal" << std::endl;
}

#ifdef _WIN32
// 1280x720 keeps each copy long enough that a present returning before the
// GPU finished would be read half-written by the consumer.
static constexpr uint32_t kW = 1280, kH = 720;

static std::mutex blitMutex;
static std::condition_variable blitDone;
static int blitResult = -1;

static void onBlit(int result) {
    std::lock_guard<std::mutex> guard(blitMutex);
    blitResult = result;
    blitDone.notify_all();
}

static int waitBlit() {
    std::unique_lock<std::mutex> lock(blitMutex);
    require(blitDone.wait_for(lock, std::chrono::seconds(10),
                              [] { return blitResult >= 0; }),
            "async blit callback arrived");
    const int result = blitResult;
    blitResult = -1;
    return result;
}

static uint32_t frameChecksum(Rgba c) {
    uint32_t hash = 0x811c9dc5u; // FNV-1a over RGBA bytes, as the consumer read
    for (size_t i = 0; i < size_t(kW) * kH; ++i) {
        for (uint8_t byte : {c.r, c.g, c.b, c.a}) hash = (hash ^ byte) * 0x01000193u;
    }
    return hash;
}

// Consumer read FIRST: any wait before it would hand the producer free time.
static void expectConsumerFrame(MGPUSharedOutputTexture* out, Rgba c, const char* what) {
    const uint32_t seen = mgpuDebugConsumerChecksumSharedHandle(
        mgpuSharedOutputTextureGetD3D11Handle(out), kW, kH);
    if (seen != frameChecksum(c)) {
        std::cerr << "FAIL: " << what << ": consumer saw a stale or torn surface\n";
        std::exit(1);
    }
}

static void expectDawnPixel(MGPUSharedOutputTexture* out, Rgba c, const char* what) {
    // The surface is BGRA8; the reader returns its first four bytes.
    const uint32_t want = uint32_t(c.b) | uint32_t(c.g) << 8 |
                          uint32_t(c.r) << 16 | uint32_t(c.a) << 24;
    const uint32_t got = mgpuSharedOutputTextureDebugReadFirstPixelDawn(out);
    if (got != want) {
        std::cerr << "FAIL: " << what << ": Dawn read 0x" << std::hex << got
                  << ", want 0x" << want << std::dec << '\n';
        std::exit(1);
    }
}

static void fillU32(MGPUBuffer* buf, Rgba c) {
    const uint32_t px = uint32_t(c.r) | uint32_t(c.g) << 8 |
                        uint32_t(c.b) << 16 | uint32_t(c.a) << 24;
    std::vector<uint32_t> pixels(size_t(kW) * kH, px);
    mgpuWriteBufferAt(buf, pixels.data(), pixels.size() * 4, 0);
}

static void fillF32(MGPUBuffer* buf, Rgba c) {
    std::vector<float> pixels(size_t(kW) * kH * 4);
    for (size_t i = 0; i < pixels.size(); i += 4) {
        pixels[i + 0] = c.r / 255.0f;
        pixels[i + 1] = c.g / 255.0f;
        pixels[i + 2] = c.b / 255.0f;
        pixels[i + 3] = c.a / 255.0f;
    }
    mgpuWriteBufferAt(buf, pixels.data(), pixels.size() * 4, 0);
}

static MGPUVideoTexture* importBgra(Rgba c) {
    std::vector<uint8_t> bytes(size_t(kW) * kH * 4);
    for (size_t i = 0; i < bytes.size(); i += 4) {
        bytes[i + 0] = c.b;
        bytes[i + 1] = c.g;
        bytes[i + 2] = c.r;
        bytes[i + 3] = c.a;
    }
    MGPUExternalVideoBuffer input{};
    input.content_type = MGPU_EXTERNAL_CONTENT_TYPE_CPU;
    input.pixel_format = MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32;
    input.width = kW;
    input.height = kH;
    input.num_planes = 1;
    input.planes[0].data_ptr = bytes.data(); // uploaded before import returns
    input.planes[0].width = kW;
    input.planes[0].height = kH;
    input.planes[0].stride_bytes = kW * 4;
    input.planes[0].dmabuf_fd = -1;
    MGPUVideoTexture* tex = mgpuImportVideoFrame(&input);
    require(tex != nullptr, "import BGRA source frame");
    return tex;
}

static void testSharedOutputPresents() {
    MGPUSharedOutputTexture* out = mgpuCreateSharedOutputTexture(kW, kH);
    require(out != nullptr, "create shared output texture");
    MGPUBuffer* u32 = mgpuCreateBuffer(int(kW * kH * 4), 7);
    MGPUBuffer* f32 = mgpuCreateBuffer(int(kW * kH * 16), 1);
    require(u32 && f32, "create source buffers");

    const Rgba a{0x10, 0x80, 0xF0, 0xFF};
    fillU32(u32, a);
    require(mgpuCopyBufferToSharedOutputTexture(u32, out) == 1, "u32 copy");
    expectConsumerFrame(out, a, "u32 copy");
    expectDawnPixel(out, a, "u32 copy");

    const Rgba b{0xF0, 0x20, 0x40, 0xFF};
    fillF32(f32, b);
    require(mgpuCopyBufferF32ToSharedOutputTexture(f32, out) == 1, "f32 copy");
    expectConsumerFrame(out, b, "f32 copy");

    const Rgba c{0x33, 0x66, 0x99, 0xFF};
    MGPUVideoTexture* src = importBgra(c);
    require(mgpuVideoTextureBGRAToRGBASharedOutput(src, out) == 1, "BGRA blit");
    expectConsumerFrame(out, c, "BGRA blit");
    mgpuDestroyVideoTexture(src);

    const Rgba d{0x99, 0x11, 0x55, 0xFF};
    fillU32(u32, d);
    mgpuCopyBufferToSharedOutputTextureAsync(u32, out, onBlit);
    require(waitBlit() == 1, "async u32 copy result");
    expectConsumerFrame(out, d, "async u32 copy");

    const Rgba e{0x05, 0xFA, 0x7F, 0xFF};
    src = importBgra(e);
    mgpuVideoTextureBGRAToRGBASharedOutputAsync(src, out, onBlit);
    require(waitBlit() == 1, "async BGRA blit result");
    expectConsumerFrame(out, e, "async BGRA blit");
    mgpuDestroyVideoTexture(src);

    // No Dart port here: the result is dropped, but the worker runs it in
    // FIFO order, so the Dawn read (also on the worker) sees its frame.
    const Rgba f{0xC0, 0xC1, 0x02, 0xFF};
    fillF32(f32, f);
    mgpuCopyBufferF32ToSharedOutputTextureAsyncToPort(f32, out, 0, 0);
    expectDawnPixel(out, f, "async f32 copy (port)");
    expectConsumerFrame(out, f, "async f32 copy (port)");

    // Destroys issued while a blit is still queued must run after it.
    const Rgba g{0x21, 0x43, 0x65, 0xFF};
    src = importBgra(g);
    mgpuVideoTextureBGRAToRGBASharedOutputAsync(src, out, onBlit);
    mgpuDestroyVideoTexture(src);
    require(waitBlit() == 1, "BGRA blit queued before its source's destroy");
    expectConsumerFrame(out, g, "BGRA blit queued before its source's destroy");

    MGPUSharedOutputTexture* doomed = mgpuCreateSharedOutputTexture(kW, kH);
    require(doomed != nullptr, "create second shared output texture");
    mgpuCopyBufferToSharedOutputTextureAsync(u32, doomed, onBlit);
    mgpuDestroySharedOutputTexture(doomed);
    require(waitBlit() == 1, "u32 copy queued before its target's destroy");

    mgpuDestroyBuffer(u32);
    mgpuDestroyBuffer(f32);
    mgpuDestroySharedOutputTexture(out);
    std::cout << "PASS shared output presents (sync, async, queued destroy)" << std::endl;
}

using Microsoft::WRL::ComPtr;

// The CPU bridge's producer: the hardware adapter with less dedicated memory
// than Dawn's (an iGPU beside the dGPU, the case the bridge exists for).
static ComPtr<ID3D11Device> bridgeProducer() {
    ComPtr<ID3D11Device> dawnDevice;
    dawnDevice.Attach(static_cast<ID3D11Device*>(mgpuCreateD3D11DeviceOnDawnAdapter()));
    ComPtr<IDXGIDevice> dxgiDevice;
    ComPtr<IDXGIAdapter> dawnAdapter;
    DXGI_ADAPTER_DESC dawnDesc{};
    if (!dawnDevice || FAILED(dawnDevice.As(&dxgiDevice)) ||
        FAILED(dxgiDevice->GetAdapter(&dawnAdapter)) || FAILED(dawnAdapter->GetDesc(&dawnDesc))) {
        return nullptr;
    }
    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return nullptr;
    ComPtr<IDXGIAdapter1> best, adapter;
    SIZE_T bestMemory = dawnDesc.DedicatedVideoMemory;
    for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, &adapter)); ++i, adapter.Reset()) {
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (desc.DedicatedVideoMemory < bestMemory) {
            best = adapter;
            bestMemory = desc.DedicatedVideoMemory;
        }
    }
    ComPtr<ID3D11Device> device;
    if (!best || FAILED(D3D11CreateDevice(best.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                                          D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
                                          D3D11_SDK_VERSION, &device, nullptr, nullptr))) {
        return nullptr;
    }
    return device;
}

// Fences are refused, so the producer's writes must be done before import.
static void producerFinish(ID3D11Device* device) {
    ComPtr<ID3D11DeviceContext> context;
    device->GetImmediateContext(&context);
    D3D11_QUERY_DESC queryDesc{D3D11_QUERY_EVENT, 0};
    ComPtr<ID3D11Query> query;
    require(SUCCEEDED(device->CreateQuery(&queryDesc, &query)), "producer event query");
    context->End(query.Get());
    BOOL done = FALSE;
    while (context->GetData(query.Get(), &done, sizeof(done), 0) == S_FALSE) Sleep(1);
}

// A shared NT-handle texture holding [data] (uninitialised when null); null
// when the producer cannot share [format].
static ComPtr<ID3D11Texture2D> producerTexture(ID3D11Device* device, DXGI_FORMAT format,
                                               UINT w, UINT h, const void* data, UINT pitch) {
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = w;
    desc.Height = h;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    const D3D11_SUBRESOURCE_DATA init{data, pitch, 0};
    ComPtr<ID3D11Texture2D> texture;
    if (FAILED(device->CreateTexture2D(&desc, data ? &init : nullptr, &texture))) return nullptr;
    producerFinish(device);
    return texture;
}

static MGPUVideoTexture* importShared(ID3D11Texture2D* texture, MGPUExternalPixelFormat format,
                                      UINT w, UINT h) {
    ComPtr<IDXGIResource1> resource;
    HANDLE handle = nullptr;
    require(SUCCEEDED(texture->QueryInterface(IID_PPV_ARGS(&resource))) &&
                SUCCEEDED(resource->CreateSharedHandle(
                    nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                    nullptr, &handle)),
            "producer shared handle");
    MGPUExternalVideoBuffer input{};
    input.content_type = MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE;
    input.pixel_format = format;
    input.width = w;
    input.height = h;
    input.num_planes = 1;
    input.planes[0].data_ptr = handle;
    input.planes[0].width = w;
    input.planes[0].height = h;
    input.planes[0].dmabuf_fd = -1;
    input.fence.sync_fd = -1;
    MGPUVideoTexture* imported = mgpuImportVideoFrame(&input);
    CloseHandle(handle);
    return imported;
}

static std::vector<uint8_t> bgraPattern(UINT w, UINT h, int seed) {
    std::vector<uint8_t> bytes(size_t(w) * h * 4);
    for (UINT y = 0; y < h; ++y) {
        for (UINT x = 0; x < w; ++x) {
            uint8_t* px = &bytes[(size_t(y) * w + x) * 4];
            px[0] = uint8_t(x * 7 + y * 3 + seed);       // B
            px[1] = uint8_t(255 - x * 5 - seed);         // G
            px[2] = uint8_t(y * 11 + 9 + seed * 3);      // R
            px[3] = uint8_t(128 + (x + y + seed) % 120); // A
        }
    }
    return bytes;
}

static std::function<Rgba(uint32_t, uint32_t)> fromBgra(const std::vector<uint8_t>& bytes,
                                                        UINT w) {
    return [&bytes, w](uint32_t x, uint32_t y) {
        const uint8_t* px = &bytes[(size_t(y) * w + x) * 4];
        return Rgba{px[2], px[1], px[0], px[3]};
    };
}

static void testCpuBridge() {
    ComPtr<ID3D11Device> producer = bridgeProducer();
    if (!producer) {
        std::cout << "SKIP: CPU bridge (no hardware adapter smaller than Dawn's)" << std::endl;
        return;
    }
    ComPtr<ID3D11DeviceContext> context;
    producer->GetImmediateContext(&context);

    // BGRA8, exact. The producer then rewrites the same texture while the
    // first frame is alive: each frame must keep its own pixels.
    const UINT w = 32, h = 16;
    const std::vector<uint8_t> first = bgraPattern(w, h, 0);
    const std::vector<uint8_t> second = bgraPattern(w, h, 50);
    ComPtr<ID3D11Texture2D> bgra =
        producerTexture(producer.Get(), DXGI_FORMAT_B8G8R8A8_UNORM, w, h, first.data(), w * 4);
    require(bgra != nullptr, "producer BGRA8 texture");
    MGPUVideoTexture* frame1 = importShared(bgra.Get(), MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32, w, h);
    require(frame1 != nullptr, "CPU bridge BGRA8 import");
    context->UpdateSubresource(bgra.Get(), 0, nullptr, second.data(), w * 4, 0);
    producerFinish(producer.Get());
    MGPUVideoTexture* frame2 = importShared(bgra.Get(), MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32, w, h);
    require(frame2 != nullptr, "CPU bridge BGRA8 re-import");
    expectRgba(mgpuVideoTextureToRGBA(frame1), w, h, fromBgra(first, w), 0,
               "CPU bridge: first frame keeps its pixels");
    expectRgba(mgpuVideoTextureToRGBA(frame2), w, h, fromBgra(second, w), 0,
               "CPU bridge: second frame of the same texture");
    mgpuDestroyVideoTexture(frame1);
    mgpuDestroyVideoTexture(frame2);

    // A new size replaces the cached staging copy.
    const UINT bw = 48, bh = 24;
    const std::vector<uint8_t> bigger = bgraPattern(bw, bh, 90);
    ComPtr<ID3D11Texture2D> resized =
        producerTexture(producer.Get(), DXGI_FORMAT_B8G8R8A8_UNORM, bw, bh, bigger.data(), bw * 4);
    require(resized != nullptr, "producer resized BGRA8 texture");
    MGPUVideoTexture* frame = importShared(resized.Get(), MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32, bw, bh);
    expectRgba(mgpuVideoTextureToRGBA(frame), bw, bh, fromBgra(bigger, bw), 0,
               "CPU bridge: resized source");
    mgpuDestroyVideoTexture(frame);

    // A declaration that disagrees with the surface's plane count is refused.
    require(importShared(bgra.Get(), MGPU_EXTERNAL_PIXEL_FORMAT_NV12, w, h) == nullptr,
            "CPU bridge refuses NV12 declared on a BGRA8 surface");

    // RGBA16F keeps its format: out-of-range values clamp, others round.
    struct Half { uint16_t bits; float value; };
    static const Half kHalves[] = {{0x0000, 0.0f}, {0x3400, 0.25f}, {0x3800, 0.5f},
                                   {0x3A00, 0.75f}, {0x3C00, 1.0f}, {0x4000, 2.0f},
                                   {0xBC00, -1.0f}, {0x2E66, 0.0999755859375f}};
    const UINT fw = 8, fh = 4;
    std::vector<uint16_t> halves(size_t(fw) * fh * 4);
    for (size_t i = 0; i < halves.size(); ++i) halves[i] = kHalves[(i * 3 + i / 4) % 8].bits;
    ComPtr<ID3D11Texture2D> half = producerTexture(
        producer.Get(), DXGI_FORMAT_R16G16B16A16_FLOAT, fw, fh, halves.data(), fw * 8);
    if (half) {
        frame = importShared(half.Get(), MGPU_EXTERNAL_PIXEL_FORMAT_RGBA64_HALF, fw, fh);
        auto channel = [&](size_t i) {
            const float v = kHalves[(i * 3 + i / 4) % 8].value;
            return uint8_t((v < 0.f ? 0.f : v > 1.f ? 1.f : v) * 255.0f + 0.5f);
        };
        expectRgba(mgpuVideoTextureToRGBA(frame), fw, fh, [&](uint32_t x, uint32_t y) {
            const size_t i = (size_t(y) * fw + x) * 4;
            return Rgba{channel(i), channel(i + 1), channel(i + 2), channel(i + 3)};
        }, 1, "CPU bridge: RGBA16F source");
        mgpuDestroyVideoTexture(frame);
    } else {
        std::cout << "SKIP: CPU bridge RGBA16F (producer cannot share it)" << std::endl;
    }

    // R8 as GRAY8: an exact grey splat.
    std::vector<uint8_t> gray(size_t(fw) * fh);
    for (size_t i = 0; i < gray.size(); ++i) gray[i] = uint8_t(i * 9 + 3);
    ComPtr<ID3D11Texture2D> r8 =
        producerTexture(producer.Get(), DXGI_FORMAT_R8_UNORM, fw, fh, gray.data(), fw);
    if (r8) {
        frame = importShared(r8.Get(), MGPU_EXTERNAL_PIXEL_FORMAT_GRAY8, fw, fh);
        expectRgba(mgpuVideoTextureToRGBA(frame), fw, fh, [&](uint32_t x, uint32_t y) {
            const uint8_t v = gray[size_t(y) * fw + x];
            return Rgba{v, v, v, 255};
        }, 0, "CPU bridge: R8 source as GRAY8");
        mgpuDestroyVideoTexture(frame);
    } else {
        std::cout << "SKIP: CPU bridge R8 (producer cannot share it)" << std::endl;
    }

    // NV12 crosses as two planes: Y rows, then interleaved UV rows at the
    // same pitch.
    const UINT nw = 16, nh = 8;
    std::vector<uint8_t> nv12(size_t(nw) * nh * 3 / 2);
    for (UINT y = 0; y < nh; ++y) {
        for (UINT x = 0; x < nw; ++x) nv12[size_t(y) * nw + x] = uint8_t(16 + (x * 13 + y * 29) % 220);
    }
    for (UINT y = 0; y < nh / 2; ++y) {
        for (UINT x = 0; x < nw / 2; ++x) {
            uint8_t* uv = &nv12[size_t(nh + y) * nw + x * 2];
            uv[0] = uint8_t(40 + (x * 37 + y * 53) % 180);
            uv[1] = uint8_t(220 - (x * 41 + y * 23) % 170);
        }
    }
    ComPtr<ID3D11Texture2D> nv =
        producerTexture(producer.Get(), DXGI_FORMAT_NV12, nw, nh, nv12.data(), nw);
    if (nv) {
        frame = importShared(nv.Get(), MGPU_EXTERNAL_PIXEL_FORMAT_NV12, nw, nh);
        expectRgba(mgpuVideoTextureToRGBA(frame), nw, nh, [&](uint32_t x, uint32_t y) {
            const uint8_t* uv = &nv12[size_t(nh + y / 2) * nw + (x / 2) * 2];
            return bt709(nv12[size_t(y) * nw + x], uv[0], uv[1]);
        }, 2, "CPU bridge: NV12 source");
        mgpuDestroyVideoTexture(frame);
        require(importShared(nv.Get(), MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32, nw, nh) == nullptr,
                "CPU bridge refuses BGRA32 declared on an NV12 surface");
    } else {
        std::cout << "SKIP: CPU bridge NV12 (producer cannot share it)" << std::endl;
    }
    std::cout << "PASS CPU bridge (formats, live frames, resize, refusals)" << std::endl;
}

// Same-adapter producer (a separate device, as screen capture uses): an
// untagged frame takes the private copy.
static void testPrivateCopyRing() {
    ComPtr<ID3D11Device> dawnDevice;
    dawnDevice.Attach(static_cast<ID3D11Device*>(mgpuCreateD3D11DeviceOnDawnAdapter()));
    ComPtr<IDXGIDevice> dxgiDevice;
    ComPtr<IDXGIAdapter> adapter;
    require(dawnDevice && SUCCEEDED(dawnDevice.As(&dxgiDevice)) &&
                SUCCEEDED(dxgiDevice->GetAdapter(&adapter)),
            "Dawn adapter");
    ComPtr<ID3D11Device> producer;
    ComPtr<ID3D11DeviceContext> context;
    require(SUCCEEDED(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                                        D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
                                        D3D11_SDK_VERSION, &producer, nullptr, &context)),
            "same-adapter producer device");
    mgpuSetLogLevel(0); // counts "private copy allocated"

    // Six live frames, more than the ring's four slots: a busy slot is never
    // handed out again.
    // Every frame's pixels arrive through the producer's context: a shared
    // texture's initial data reads as zeros on the other device here.
    const UINT w = 32, h = 16;
    std::vector<std::vector<uint8_t>> patterns;
    ComPtr<ID3D11Texture2D> source =
        producerTexture(producer.Get(), DXGI_FORMAT_B8G8R8A8_UNORM, w, h, nullptr, 0);
    require(source != nullptr, "same-adapter producer texture");
    const int firstCopies = privateCopies.load();
    std::vector<MGPUVideoTexture*> live;
    for (int i = 0; i < 6; ++i) {
        patterns.push_back(bgraPattern(w, h, i * 20));
        context->UpdateSubresource(source.Get(), 0, nullptr, patterns[i].data(), w * 4, 0);
        producerFinish(producer.Get());
        live.push_back(importShared(source.Get(), MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32, w, h));
        require(live.back() != nullptr, "private copy import");
    }
    require(privateCopies.load() - firstCopies == 6, "each live frame has its own private copy");
    for (size_t i = 0; i < live.size(); ++i) {
        expectRgba(mgpuVideoTextureToRGBA(live[i]), w, h, fromBgra(patterns[i], w), 0,
                   "private copy: live frames keep their own pixels");
        mgpuDestroyVideoTexture(live[i]);
    }

    // Each frame is destroyed while its read is still queued, and the next
    // import may copy into the same slot before that read is checked.
    const int before = privateCopies.load();
    std::vector<uint8_t> queuedPattern;
    MGPUBuffer* queued = nullptr;
    for (int i = 0; i < 8; ++i) {
        std::vector<uint8_t> pattern = bgraPattern(w, h, 200 + i * 7);
        context->UpdateSubresource(source.Get(), 0, nullptr, pattern.data(), w * 4, 0);
        producerFinish(producer.Get());
        MGPUVideoTexture* frame =
            importShared(source.Get(), MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32, w, h);
        require(frame != nullptr, "private copy import");
        if (queued) {
            expectRgba(queued, w, h, fromBgra(queuedPattern, w), 0,
                       "private copy: a queued read keeps its destroyed frame");
        }
        queued = mgpuVideoTextureToRGBA(frame);
        queuedPattern = pattern;
        mgpuDestroyVideoTexture(frame);
    }
    expectRgba(queued, w, h, fromBgra(queuedPattern, w), 0,
               "private copy: last queued read");
    const int allocated = privateCopies.load() - before;
    mgpuSetLogLevel(2);
    std::cout << "OBSERVED: " << allocated << " private copies allocated over 8 sequential frames"
              << std::endl;
    require(allocated <= 2, "sequential private copies reuse the ring");
    std::cout << "PASS same-adapter private copy (live frames, queued reads, reuse)" << std::endl;
}
#endif

int main() {
    mgpuSetLogCallback(recordLog);
    mgpuSetLogLevel(2);
    mgpuInitializeContext();
    testCpuFormatsAndFences();
#ifdef _WIN32
    testSharedOutputPresents();
    testCpuBridge();
    testPrivateCopyRing();
#else
    std::cout << "SKIP: shared output presents, CPU bridge and private copy (Windows only)"
              << std::endl;
#endif
    mgpuDestroyContext();
    mgpuSetLogCallback(nullptr);
    return 0;
}
