// Standalone Windows integration gate for immutable D3D11 frame imports.
// Optional args: width height frames-per-path (default 257 129 8).
// --benchmark [width height frames] defaults to 3840 2160 24, with 4 warmups.
// Benchmark timers exclude producer work and CPU readback; sampler time ends
// at queue submission and is a staging proxy, not GPU completion/GSZC encode.
// Source pixels are generated with a GPU clear; no CPU pixel upload is used.
#include "../include/minigpu.h"
#include "../include/minigpu_external.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <vector>

#ifdef _WIN32
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

static std::atomic<int> directImports{0};
static std::mutex submitMutex;
static std::condition_variable submitCondition;
static bool submitted = false;

static void onSubmitted() {
    std::lock_guard<std::mutex> guard(submitMutex);
    submitted = true;
    submitCondition.notify_one();
}

static constexpr const char* kLandPacked = R"WGSL(
@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> pixels: array<u32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(frame);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let rgba = vec4<u32>(clamp(textureLoad(frame, vec2<i32>(gid.xy), 0),
                            vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0 + 0.5);
  pixels[gid.y * dims.x + gid.x] = rgba.r | (rgba.g << 8u) |
                                   (rgba.b << 16u) | (rgba.a << 24u);
}
)WGSL";

static void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

static void recordLog(int level, const char* message) {
    if (std::strstr(message, "immutable D3D11 snapshot imported directly")) {
        ++directImports;
    }
    if (level >= 2) std::cerr << message << '\n';
    mgpuFreeLogMessage(message);
}

static void completeProducer(ID3D11Device* device, ID3D11DeviceContext* context) {
    D3D11_QUERY_DESC desc{};
    desc.Query = D3D11_QUERY_EVENT;
    ComPtr<ID3D11Query> done;
    require(SUCCEEDED(device->CreateQuery(&desc, &done)), "create producer query");
    context->End(done.Get());
    context->Flush();
    const auto start = std::chrono::steady_clock::now();
    HRESULT result;
    do {
        result = context->GetData(done.Get(), nullptr, 0, 0);
        require(std::chrono::steady_clock::now() - start < std::chrono::seconds(5),
                "producer query timed out");
        if (result == S_FALSE) SwitchToThread();
    } while (result == S_FALSE);
    require(result == S_OK, "producer query completed successfully");
}

static void runPath(ID3D11Device* device, ID3D11DeviceContext* context,
                    uint32_t width, uint32_t height, int frames, uint64_t tag,
                    bool benchmark = false) {
    const bool direct = tag == MGPU_D3D11_IMMUTABLE_READY_TAG;
    const int before = directImports.load();
    std::vector<double> importMs;
    std::vector<double> samplingSubmitMs;
    const int warmup = benchmark ? 4 : 0;
    MGPUComputeShader* land = nullptr;
    MGPUBuffer* landed = nullptr;
    if (benchmark) {
        // Match the persistent output/shader lifetime of GSZC's packed stage.
        // Only the imported texture binding changes on each frame.
        // mgpuCreateBuffer uses mapIntToBufferDataType's public code 7 for
        // uint8; the private C++ BufferDataType enum has different numbering.
        landed = mgpuCreateBuffer(int(size_t(width) * height * 4), 7);
        land = mgpuCreateComputeShader();
        require(landed && land, "create persistent benchmark landing resources");
        mgpuLoadKernel(land, kLandPacked);
        mgpuSetBuffer(land, 1, landed);
    }
    for (int frame = 0; frame < frames + warmup; ++frame) {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                         D3D11_RESOURCE_MISC_SHARED;
        ComPtr<ID3D11Texture2D> producer;
        require(SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &producer)),
                "create producer snapshot");
        ComPtr<ID3D11RenderTargetView> target;
        require(SUCCEEDED(device->CreateRenderTargetView(producer.Get(), nullptr, &target)),
                "create source render target");
        const float color[4] = {float(frame & 1), float((frame >> 1) & 1),
                                float((frame >> 2) & 1), 1.0f};
        context->ClearRenderTargetView(target.Get(), color);
        completeProducer(device, context);

        ComPtr<IDXGIResource1> resource;
        require(SUCCEEDED(producer.As(&resource)), "query source sharing interface");
        HANDLE handle = nullptr;
        require(SUCCEEDED(resource->CreateSharedHandle(nullptr,
                    DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                    nullptr, &handle)), "create NT handle");

        MGPUExternalVideoBuffer input{};
        input.content_type = MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE;
        input.pixel_format = MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32;
        input.width = width;
        input.height = height;
        input.num_planes = 1;
        input.planes[0].data_ptr = handle;
        input.planes[0].width = width;
        input.planes[0].height = height;
        input.planes[0].stride_bytes = width * 4; // SDK forwarding layout.
        input.planes[0].dmabuf_fd = -1;
        input.planes[0].drm_format_modifier = tag;
        const auto start = std::chrono::steady_clock::now();
        auto* imported = mgpuImportVideoFrame(&input);
        const auto afterImport = std::chrono::steady_clock::now();
        if (frame >= warmup) importMs.push_back(
            std::chrono::duration<double, std::milli>(afterImport - start).count());
        require(imported != nullptr, "import source snapshot");
        CloseHandle(handle);

        if (!direct && !benchmark) {
            // Unhinted and unknown-tag callers retain the private snapshot:
            // later producer writes must not alter the already-imported frame.
            const float changed[4] = {1.0f - color[0], 1.0f - color[1],
                                       1.0f - color[2], 1.0f};
            context->ClearRenderTargetView(target.Get(), changed);
            completeProducer(device, context);
        }
        // Drop EVERY producer texture reference before the consumer submits
        // its sampling shader. Direct import must own the underlying resource.
        target.Reset();
        resource.Reset();
        producer.Reset();
        MGPUBuffer* pixels;
        if (benchmark) {
            mgpuSetVideoTexture(land, 0, imported, 0);
            {
                std::lock_guard<std::mutex> guard(submitMutex);
                submitted = false;
            }
            mgpuDispatchAsync(land, int((width + 7) / 8), int((height + 7) / 8),
                              1, onSubmitted);
            std::unique_lock<std::mutex> lock(submitMutex);
            require(submitCondition.wait_for(lock, std::chrono::seconds(5),
                        [] { return submitted; }), "landing submit timed out");
            pixels = landed;
        } else {
            pixels = mgpuVideoTextureToRGBA(imported);
        }
        require(pixels != nullptr, "submit consumer sampling shader");
        mgpuDestroyVideoTexture(imported); // Submission retains GPU resources.
        if (benchmark && frame >= warmup) samplingSubmitMs.push_back(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
        // Readback drains the submission and checks every pixel, outside both
        // benchmark timers. It also prevents accumulating an artificial queue.
        std::vector<uint8_t> output(size_t(width) * height * 4);
        mgpuReadSyncUint8(pixels, output.data(), int(output.size()), 0);
        if (!benchmark) mgpuDestroyBuffer(pixels);
        for (size_t i = 0; i < output.size(); ++i) {
            require(output[i] == uint8_t(color[i & 3] * 255),
                    "consumer pixel differs after producer release");
        }
    }
    if (land) mgpuDestroyComputeShader(land);
    if (landed) mgpuDestroyBuffer(landed);
    require(directImports.load() - before == (direct ? frames + warmup : 0),
            "only the exact provenance tag selects direct import");
    std::sort(importMs.begin(), importMs.end());
    std::cout << (benchmark ? "BENCH " : "PASS ") << (direct ? "direct" : "copy")
              << " tag=" << tag << " " << width << "x" << height
              << " verified_frames=" << frames
              << " import_p50_ms=" << importMs[frames / 2];
    if (benchmark) {
        std::sort(samplingSubmitMs.begin(), samplingSubmitMs.end());
        std::cout << " import_sampler_submit_p50_ms=" << samplingSubmitMs[frames / 2]
                  << " warmup_frames=" << warmup;
    }
    std::cout << " (GPU-clear fixture; CPU readback excluded, not encode)" << std::endl;
}
#endif

int main(int argc, char** argv) {
#ifndef _WIN32
    std::cout << "SKIP: Windows D3D11 integration test" << std::endl;
    return 0;
#else
    const bool benchmark = argc > 1 && std::strcmp(argv[1], "--benchmark") == 0;
    const int firstArg = benchmark ? 2 : 1;
    const int width = argc > firstArg ? std::atoi(argv[firstArg]) : (benchmark ? 3840 : 257);
    const int height = argc > firstArg + 1 ? std::atoi(argv[firstArg + 1]) : (benchmark ? 2160 : 129);
    const int frames = argc > firstArg + 2 ? std::atoi(argv[firstArg + 2]) : (benchmark ? 24 : 8);
    require(width > 0 && width <= 4096 && height > 0 && height <= 4096 &&
            frames > 0 && frames <= 120, "bounded positive test geometry/count");
    _putenv_s("MGPU_BACKEND", "d3d11");
    mgpuSetLogCallback(recordLog);
    mgpuSetLogLevel(0);
    mgpuInitializeContext();
    ComPtr<ID3D11Device> dawnDevice;
    dawnDevice.Attach(static_cast<ID3D11Device*>(mgpuCreateD3D11DeviceOnDawnAdapter()));
    require(dawnDevice != nullptr, "get Dawn adapter device");
    ComPtr<IDXGIDevice> dxgiDevice;
    require(SUCCEEDED(dawnDevice.As(&dxgiDevice)), "get Dawn DXGI device");
    ComPtr<IDXGIAdapter> adapter;
    require(SUCCEEDED(dxgiDevice->GetAdapter(&adapter)), "get Dawn adapter");
    // WGC produces on a separate device. Use that same cross-device shape,
    // rather than allowing source writes to be ordered by Dawn's own queue.
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    require(SUCCEEDED(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN,
                nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
                D3D11_SDK_VERSION, &device, nullptr, &context)),
            "create independent producer device on Dawn adapter");
    require(device.Get() != dawnDevice.Get(), "producer and consumer devices differ");
    runPath(device.Get(), context.Get(), width, height, frames, 0, benchmark);
    runPath(device.Get(), context.Get(), width, height, frames,
             MGPU_D3D11_IMMUTABLE_READY_TAG, benchmark);
    runPath(device.Get(), context.Get(), width, height, frames,
             MGPU_D3D11_IMMUTABLE_READY_TAG ^ 1u, benchmark);
    context.Reset();
    device.Reset();
    adapter.Reset();
    dxgiDevice.Reset();
    dawnDevice.Reset();
    mgpuDestroyContext();
    mgpuSetLogCallback(nullptr);
    return 0;
#endif
}
