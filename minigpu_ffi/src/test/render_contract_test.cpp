// Native gate for the render and texture lifetime contracts (default D3D11
// backend).
//
// A WGSL error fails the pipeline with Tint's line and column. Work admitted
// before a destroy still lands: an upload to a temporary texture and a pass
// that samples it, both queued behind a large upload, survive destroying the
// texture, its view and the attachment's view right after submit. Destroying
// a texture frees its memory while a pipeline's cached bind group still holds
// its view (Windows: the process's local video memory on Dawn's adapter).
// None of it may raise a validation error.
#include "../include/minigpu.h"
#include "../include/minigpu_external.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#endif

namespace {
std::atomic<int> errors{0};
std::atomic<bool> done{false};

void recordLog(int level, const char *message) {
  if (level >= 3) {
    errors.fetch_add(1);
    std::cerr << message << '\n';
  }
  mgpuFreeLogMessage(message);
}
void callback() { done.store(true, std::memory_order_release); }
void require(bool value, const char *message) {
  if (!value) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}
int waitPipeline(MGPURenderPipeline *p) {
  auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  for (;;) {
    int s = mgpuRenderPipelineStatus(p);
    if (s != 0)
      return s;
    require(std::chrono::steady_clock::now() < end, "render pipeline timeout");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

// Copies the sampled texture into the attachment texel for texel.
const char *copyWgsl = R"WGSL(
@group(0) @binding(0) var source: texture_2d<f32>;
@vertex fn vsMain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fsMain(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
  return textureLoad(source, vec2<i32>(at.xy), 0);
}
)WGSL";
const char *readWgsl = R"WGSL(
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<storage,read_write> result: array<u32>;
@compute @workgroup_size(1) fn main(){let v=vec4<u32>(round(textureLoad(source,vec2<i32>(1,1),0)*255.0));result[0]=v.r|(v.g<<8u)|(v.b<<16u)|(v.a<<24u);}
)WGSL";

// RGBA8 target, no blend, no depth, no vertex buffers, one unfilterable 2D
// texture at fragment slot 0.
std::vector<uint32_t> copyDescriptor() {
  return {1, 1, 0, 0, 0, 0, 0, 0, 4, 0, 1, 0, 3, 2, 0, 0};
}
// Bind view 0 at slot 0, draw one full-screen triangle.
const uint32_t copyCommands[] = {4, 0, 0, 6, 3, 1, 0, 0};
const float clearColor[] = {0, 0, 0, 0, 1};

int submitCopy(MGPURenderPipeline *p, MGPUTextureView *target,
               MGPUTextureView *source) {
  MGPUTextureView *views[] = {source};
  return mgpuSubmitRender(p, target, nullptr, clearColor, 0, nullptr, 0, views,
                          1, nullptr, 0, copyCommands,
                          int(sizeof(copyCommands) / sizeof(copyCommands[0])));
}

// Texel (1,1) of one view, packed RGBA8.
struct Reader {
  MGPUBuffer *out = nullptr;
  MGPUComputeShader *cs = nullptr;
  explicit Reader(MGPUTextureView *view) {
    out = mgpuCreateBufferWithUsage(4, 4 | 8 | 128);
    cs = mgpuCreateComputeShader();
    require(out && cs, "readback allocation failed");
    mgpuLoadKernel(cs, readWgsl);
    require(mgpuSetOwnedTexture(cs, 0, view, 0) == 0, "readback texture bind");
    require(mgpuSetBufferRange(cs, 1, out, 0, 0, 4) == 0, "readback buffer bind");
  }
  ~Reader() {
    mgpuDestroyComputeShader(cs);
    mgpuDestroyBuffer(out);
  }
  uint32_t read() {
    done = false;
    mgpuDispatchAsync(cs, 1, 1, 1, callback);
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!done.load(std::memory_order_acquire)) {
      require(std::chrono::steady_clock::now() < end, "readback timeout");
      std::this_thread::yield();
    }
    uint32_t pixel = 0;
    mgpuReadSyncUint32(out, &pixel, 1, 0);
    return pixel;
  }
};

#ifdef _WIN32
using Microsoft::WRL::ComPtr;

ComPtr<IDXGIAdapter3> dawnAdapter() {
  ComPtr<ID3D11Device> device;
  device.Attach(
      static_cast<ID3D11Device *>(mgpuCreateD3D11DeviceOnDawnAdapter()));
  ComPtr<IDXGIDevice> dxgiDevice;
  ComPtr<IDXGIAdapter> adapter;
  ComPtr<IDXGIAdapter3> adapter3;
  if (!device || FAILED(device.As(&dxgiDevice)) ||
      FAILED(dxgiDevice->GetAdapter(&adapter)) || FAILED(adapter.As(&adapter3)))
    return nullptr;
  return adapter3;
}

int64_t localUsage(IDXGIAdapter3 *adapter) {
  DXGI_QUERY_VIDEO_MEMORY_INFO info{};
  if (FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                                           &info)))
    return -1;
  return int64_t(info.CurrentUsage);
}
#endif

void testCompileDiagnostic() {
  auto d = copyDescriptor();
  std::string typo = copyWgsl;
  typo.replace(typo.find("textureLoad(source"), 18, "textureLoad(sauce");
  auto *p = mgpuCreateRenderPipeline(typo.c_str(), "vsMain", "fsMain", d.data(),
                                     (int)d.size());
  require(p, "pipeline allocation failed");
  require(waitPipeline(p) == 2, "WGSL error did not fail the pipeline");
  char diagnostic[600] = {};
  mgpuRenderPipelineDiagnostic(p, diagnostic, sizeof(diagnostic));
  std::string text = diagnostic;
  if (text.rfind("line 8:", 0) != 0 || text.find("sauce") == std::string::npos) {
    std::cerr << "diagnostic: " << text << std::endl;
    require(false, "WGSL error lost its position");
  }
  mgpuDestroyRenderPipeline(p);
  mgpuDrainWorkQueue();
  std::cout << "PASS WGSL error diagnostic (" << text.substr(0, 40) << "...)"
            << std::endl;
}

void testDestroyAfterSubmit() {
  auto d = copyDescriptor();
  auto *p = mgpuCreateRenderPipeline(copyWgsl, "vsMain", "fsMain", d.data(),
                                     (int)d.size());
  require(p, "pipeline allocation failed");
  require(waitPipeline(p) == 1, "copy pipeline compilation failed");
  MGPUTexture *atlas = mgpuCreateTexture2D(4, 4, 1, 4 | 16, 1, 1);
  auto *rv = mgpuCreateTextureView(atlas, 0, 0, 1, 0, 1);
  require(atlas && rv, "atlas allocation failed");
  Reader reader(rv);

  // Keep the worker busy so everything below is admitted before it runs.
  const int side = 4096;
  MGPUTexture *ballast = mgpuCreateTexture2D(side, side, 1, 2 | 4, 1, 1);
  require(ballast, "ballast allocation failed");
  std::vector<uint8_t> ballastBytes(size_t(side) * side * 4, 7);
  require(mgpuWriteTexture2D(ballast, ballastBytes.data(),
                             (int)ballastBytes.size(), 0, 0) == 0,
          "ballast upload refused");

  MGPUTexture *temp = mgpuCreateTexture2D(4, 4, 1, 2 | 4, 1, 1);
  auto *tv = mgpuCreateTextureView(temp, 0, 0, 1, 0, 1);
  auto *av = mgpuCreateTextureView(atlas, 0, 0, 1, 0, 1);
  require(temp && tv && av, "temporary allocation failed");
  uint8_t texels[4 * 4 * 4];
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) {
      uint8_t *t = texels + (y * 4 + x) * 4;
      t[0] = uint8_t(10 + 40 * x);
      t[1] = uint8_t(20 + 40 * y);
      t[2] = 200;
      t[3] = 255;
    }
  require(mgpuWriteTexture2D(temp, texels, sizeof(texels), 0, 0) == 0,
          "temporary upload refused");
  require(submitCopy(p, av, tv) == 0, "copy pass refused");
  mgpuDestroyTextureView(tv);
  mgpuDestroyTexture(temp);
  mgpuDestroyTextureView(av);
  mgpuDrainWorkQueue();
  const uint32_t expected = 50u | 60u << 8 | 200u << 16 | 255u << 24;
  require(reader.read() == expected,
          "upload or pass admitted before a destroy was dropped");
  std::cout << "PASS work admitted before a destroy lands" << std::endl;

#ifdef _WIN32
  // Destroying a texture frees its memory although the pipeline's cached
  // bind group still holds its view.
  ComPtr<IDXGIAdapter3> adapter = dawnAdapter();
  require(adapter != nullptr, "Dawn adapter lookup failed");
  mgpuDrainWorkQueue();
  reader.read();
  const int64_t before = localUsage(adapter.Get());
  const int big = 8192;
  const int64_t bytes = int64_t(big) * big * 4;
  MGPUTexture *texture = mgpuCreateTexture2D(big, big, 1, 4, 1, 1);
  auto *view = mgpuCreateTextureView(texture, 0, 0, 1, 0, 1);
  auto *target = mgpuCreateTextureView(atlas, 0, 0, 1, 0, 1);
  require(texture && view && target, "large texture allocation failed");
  require(submitCopy(p, target, view) == 0, "large texture pass refused");
  mgpuDrainWorkQueue();
  // The lazily cleared texture overwrites the earlier texel, which proves the
  // pass ran.
  require(reader.read() == 0, "large texture pass did not run");
  require(mgpuRenderPipelineBindingStat(p, 3) >= 1,
          "pipeline cached no bind group");
  const int64_t used = localUsage(adapter.Get());
  if (before < 0 || used - before < bytes * 3 / 4) {
    std::cout << "SKIP texture memory release: the adapter counter moved "
              << (used - before) / (1 << 20) << " MiB for a "
              << bytes / (1 << 20) << " MiB texture" << std::endl;
    mgpuDestroyTextureView(view);
    mgpuDestroyTexture(texture);
  } else {
    mgpuDestroyTextureView(view);
    mgpuDestroyTexture(texture);
    mgpuDrainWorkQueue();
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    int64_t now = used;
    // Each readback submits and waits, which lets the driver retire the
    // freed allocation.
    while (used - (now = localUsage(adapter.Get())) < bytes * 3 / 4) {
      if (std::chrono::steady_clock::now() >= end) {
        std::cerr << "still held: " << (now - before) / (1 << 20) << " MiB"
                  << std::endl;
        require(false, "destroyed texture's memory is still held");
      }
      reader.read();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::cout << "PASS destroyed texture frees its memory (" << bytes / (1 << 20)
              << " MiB, " << (used - now) / (1 << 20) << " MiB released)"
              << std::endl;
  }
  mgpuDestroyTextureView(target);
#else
  std::cout << "SKIP texture memory release (Windows only)" << std::endl;
#endif

  mgpuDestroyTexture(ballast);
  mgpuDestroyTextureView(rv);
  mgpuDestroyTexture(atlas);
  mgpuDestroyRenderPipeline(p);
  mgpuDrainWorkQueue();
}
} // namespace

int main() {
  mgpuSetLogCallback(recordLog);
  mgpuInitializeContext();
  testCompileDiagnostic();
  // The WGSL error above is reported as a validation error; nothing below
  // may raise one.
  const int before = errors.load();
  testDestroyAfterSubmit();
  require(errors.load() == before, "validation error during the lifetime tests");
  mgpuDestroyContext();
  mgpuSetLogCallback(nullptr);
  std::cout << "Render contract test passed." << std::endl;
  return 0;
}
