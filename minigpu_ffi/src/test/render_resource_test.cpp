#include "../include/minigpu.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace {
std::atomic<bool> done{false};
void callback() { done.store(true, std::memory_order_release); }
void require(bool value, const char *message) {
  if (!value) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}
void waitCallback() {
  auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!done.load(std::memory_order_acquire)) {
    require(std::chrono::steady_clock::now() < end, "callback timeout");
    std::this_thread::yield();
  }
}
void dispatch(MGPUComputeShader *s) {
  done = false;
  mgpuDispatchAsync(s, 1, 1, 1, callback);
  waitCallback();
}
void waitPipeline(MGPURenderPipeline *p) {
  auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  for (;;) {
    int s = mgpuRenderPipelineStatus(p);
    if (s != 0) {
      require(s == 1, "render pipeline compilation failed");
      return;
    }
    require(std::chrono::steady_clock::now() < end, "render pipeline timeout");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

const char *renderWgsl = R"WGSL(
struct Params { tint: vec4<f32> };
@group(0) @binding(0) var<uniform> params: Params;
struct VSOut { @builtin(position) position: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vsMain(@location(0) p: vec2<f32>, @location(1) depth: f32,
                  @location(2) rgb: vec3<f32>) -> VSOut {
  var o: VSOut; o.position=vec4<f32>(p,depth,1.0); o.color=vec4<f32>(rgb,0.5); return o;
}
@fragment fn fsMain(i: VSOut) -> @location(0) vec4<f32> { return i.color * params.tint; }
)WGSL";
const char *readWgsl = R"WGSL(
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<storage,read_write> result: array<u32>;
@compute @workgroup_size(1) fn main(){let v=vec4<u32>(round(textureLoad(source,vec2<i32>(1,1),0)*255.0));result[0]=v.r|(v.g<<8u)|(v.b<<16u)|(v.a<<24u);}
)WGSL";

std::vector<uint32_t> descriptor() {
  return {1, 1,  1, 0, 0, 0, 1, 1, 4, 2, 1, 8, 0, 1, 0, 0,
          1, 16, 1, 2, 1, 0, 0, 2, 4, 2, 0, 0, 2, 0, 16};
}

void mainScenario() {
  auto d = descriptor();
  auto *p = mgpuCreateRenderPipeline(renderWgsl, "vsMain", "fsMain", d.data(),
                                     (int)d.size());
  require(p, "pipeline allocation failed");
  waitPipeline(p);
  MGPUTexture *color = mgpuCreateTexture2D(4, 4, 1, 4 | 16, 1, 1);
  MGPUTexture *depth = mgpuCreateTexture2D(4, 4, 5, 16, 1, 1);
  auto *cv = mgpuCreateTextureView(color, 0, 0, 1, 0, 1);
  auto *dv = mgpuCreateTextureView(depth, 0, 0, 1, 0, 1);
  require(cv && dv, "attachments failed");
  float positions[] = {-1, -1, 3, -1, -1, 3};
  float instances[] = {0.75f, 1, 0, 0, 0.25f, 0, 1, 0};
  uint16_t paddedIndices[] = {0, 1, 2, 0};
  auto *vb = mgpuCreateBufferWithUsage(sizeof(positions), 8 | 32);
  auto *ib = mgpuCreateBufferWithUsage(sizeof(paddedIndices), 8 | 16);
  auto *inst = mgpuCreateBufferWithUsage(sizeof(instances), 8 | 32);
  float tint[] = {1, 1, 1, 1};
  auto *params = mgpuCreateBufferWithUsage(sizeof(tint), 8 | 64);
  require(vb && ib && inst && params, "geometry buffers failed");
  mgpuWriteBufferAt(vb, positions, sizeof(positions), 0);
  mgpuWriteBufferAt(ib, paddedIndices, sizeof(paddedIndices), 0);
  mgpuWriteBufferAt(inst, instances, sizeof(instances), 0);
  mgpuWriteBufferAt(params, tint, sizeof(tint), 0);
  MGPUBuffer *buffers[] = {vb, inst, ib, params};
  uint32_t commands[] = {1,
                         0,
                         0,
                         0,
                         sizeof(positions),
                         1,
                         1,
                         1,
                         0,
                         sizeof(instances),
                         2,
                         2,
                         0,
                         sizeof(paddedIndices),
                         0,
                         3,
                         0,
                         3,
                         0,
                         sizeof(tint),
                         7,
                         3,
                         2,
                         0,
                         0,
                         0};
  float clear[] = {0, 0, 0, 0, 1};
  require(mgpuSubmitRender(p, cv, dv, clear, 0, buffers, 4, nullptr, 0, nullptr,
                           0, commands,
                           int(sizeof(commands) / sizeof(commands[0]))) == 0,
          "indexed instanced submit refused");
  mgpuDrainWorkQueue();
  require(mgpuSubmitRender(p, cv, dv, clear, 0, buffers, 4, nullptr, 0, nullptr,
                           0, commands,
                           int(sizeof(commands) / sizeof(commands[0]))) == 0,
          "repeated indexed instanced submit refused");
  mgpuDrainWorkQueue();
  require(mgpuRenderPipelineBindingStat(p, 0) == 1,
          "stable binding created more than one bind group");
  require(mgpuRenderPipelineBindingStat(p, 1) == 1,
          "stable binding did not reuse its bind group");
  require(mgpuRenderPipelineBindingStat(p, 2) == 0 &&
              mgpuRenderPipelineBindingStat(p, 3) == 1,
          "stable binding retired its bind group");
  auto *out = mgpuCreateBufferWithUsage(4, 4 | 8 | 128);
  auto *cs = mgpuCreateComputeShader();
  require(out && cs, "readback allocation failed");
  mgpuLoadKernel(cs, readWgsl);
  require(mgpuSetOwnedTexture(cs, 0, cv, 0) == 0,
          "render target sampling failed");
  require(mgpuSetBufferRange(cs, 1, out, 0, 0, 4) == 0, "readback bind failed");
  dispatch(cs);
  uint32_t pixel = 0;
  mgpuReadSyncUint32(out, &pixel, 1, 0);
  auto channel = [&](int shift) { return int((pixel >> shift) & 255); };
  require(std::abs(channel(0) - 64) <= 2 && std::abs(channel(8) - 128) <= 2 &&
              channel(16) <= 2 && std::abs(channel(24) - 191) <= 2,
          "depth/blend pixel mismatch");
  uint32_t malformed[] = {99};
  require(mgpuSubmitRender(p, cv, dv, clear, 0, buffers, 4, nullptr, 0, nullptr,
                           0, malformed, 1) == 1,
          "unknown opcode accepted");
  MGPUBuffer *bad = mgpuCreateBufferWithUsage(sizeof(positions), 8 | 128);
  MGPUBuffer *badBuffers[] = {bad, inst, ib, params};
  require(mgpuSubmitRender(p, cv, dv, clear, 0, badBuffers, 4, nullptr, 0,
                           nullptr, 0, commands,
                           int(sizeof(commands) / sizeof(commands[0]))) == 1,
          "missing vertex usage accepted");
  auto *staleTexture = mgpuCreateTexture2D(4, 4, 1, 16, 1, 1);
  auto *staleView = mgpuCreateTextureView(staleTexture, 0, 0, 1, 0, 1);
  mgpuDestroyTexture(staleTexture);
  require(mgpuSubmitRender(p, staleView, dv, clear, 0, buffers, 4, nullptr, 0,
                           nullptr, 0, commands,
                           int(sizeof(commands) / sizeof(commands[0]))) == 2,
          "destroyed attachment accepted");
  mgpuDestroyTextureView(staleView);
  mgpuDestroyBuffer(bad);
  mgpuDestroyComputeShader(cs);
  mgpuDestroyBuffer(out);
  mgpuDestroyBuffer(vb);
  mgpuDestroyBuffer(inst);
  mgpuDestroyBuffer(ib);
  mgpuDestroyBuffer(params);
  mgpuDestroyTextureView(cv);
  mgpuDestroyTextureView(dv);
  mgpuDestroyTexture(color);
  mgpuDestroyTexture(depth);
  mgpuDestroyRenderPipeline(p);
  mgpuDrainWorkQueue();
}

void invalidAndOwnership() {
  auto d = descriptor();
  auto bad = d;
  bad.push_back(0);
  require(!mgpuCreateRenderPipeline(renderWgsl, "vsMain", "fsMain", bad.data(),
                                    (int)bad.size()),
          "trailing descriptor accepted");
  auto *closing = mgpuCreateRenderPipeline(renderWgsl, "vsMain", "fsMain",
                                           d.data(), (int)d.size());
  require(closing, "preparing pipeline failed");
  mgpuDestroyRenderPipeline(closing);
  mgpuDrainWorkQueue();
  auto *ctx = mgpuCreateContextHandle(nullptr);
  done = false;
  mgpuContextInitializeAsync(ctx, callback);
  waitCallback();
  auto *local = mgpuContextCreateRenderPipeline(
      ctx, renderWgsl, "vsMain", "fsMain", d.data(), (int)d.size());
  require(local, "secondary pipeline failed");
  waitPipeline(local);
  auto *foreign = mgpuCreateTexture2D(2, 2, 1, 16, 1, 1);
  auto *fv = mgpuCreateTextureView(foreign, 0, 0, 1, 0, 1);
  auto *localDepth = mgpuContextCreateTexture2D(ctx, 2, 2, 5, 16, 1, 1);
  auto *ldv = mgpuCreateTextureView(localDepth, 0, 0, 1, 0, 1);
  float clear[] = {0, 0, 0, 0, 1};
  require(mgpuSubmitRender(local, fv, ldv, clear, 0, nullptr, 0, nullptr, 0,
                           nullptr, 0, nullptr, 0) == 1,
          "foreign attachment accepted");
  mgpuDestroyTextureView(fv);
  mgpuDestroyTexture(foreign);
  mgpuDestroyTextureView(ldv);
  mgpuDestroyTexture(localDepth);
  mgpuDestroyRenderPipeline(local);
  mgpuDestroyContextHandle(ctx);
}
} // namespace

int main() {
  auto d = descriptor();
  require(mgpuRasterResourceVersion() == 1, "raster ABI version");
  require(!mgpuCreateRenderPipeline(renderWgsl, "vsMain", "fsMain", d.data(),
                                    (int)d.size()),
          "pipeline lazily initialized device");
  mgpuInitializeContext();
  mainScenario();
  invalidAndOwnership();
  mgpuDestroyContext();
  std::cout << "Render resource C ABI test passed." << std::endl;
  return 0;
}
