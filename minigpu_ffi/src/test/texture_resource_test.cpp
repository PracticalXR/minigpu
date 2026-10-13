#include "../include/minigpu.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

namespace {

std::atomic<bool> callbackDone{false};

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

void complete() { callbackDone.store(true, std::memory_order_release); }

void waitForCallback(const char *message) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!callbackDone.load(std::memory_order_acquire)) {
    require(std::chrono::steady_clock::now() < deadline, message);
    std::this_thread::yield();
  }
}

void dispatch(MGPUComputeShader *shader) {
  callbackDone.store(false, std::memory_order_release);
  mgpuDispatchAsync(shader, 1, 1, 1, complete);
  waitForCallback("compute dispatch timed out");
}

constexpr const char *kSampleToStorage = R"WGSL(
@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;
@group(0) @binding(2) var destination:
    texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(2, 2)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < 2u && id.y < 2u) {
    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(2.0);
    textureStore(destination, vec2<i32>(id.xy),
                 textureSampleLevel(sourceTexture, sourceSampler, uv, 0.0));
  }
}
)WGSL";

constexpr const char *kTextureToBuffer = R"WGSL(
@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputValues: array<u32>;
@compute @workgroup_size(2, 2)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < 2u && id.y < 2u) {
    let value = vec4<u32>(round(clamp(
        textureLoad(sourceTexture, vec2<i32>(id.xy), 0),
        vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0));
    outputValues[id.y * 2u + id.x] = value.r | (value.g << 8u) |
        (value.b << 16u) | (value.a << 24u);
  }
}
)WGSL";

void testCreationValidation() {
  require(mgpuTextureResourceVersion() == 1,
          "texture resource ABI version must be 1");
  require(mgpuCreateTexture2D(0, 1, 1, 4, 1, 1) == nullptr,
          "zero width must be rejected");
  require(mgpuCreateTexture2D(4, 4, 99, 4, 1, 1) == nullptr,
          "unknown format must be rejected");
  require(mgpuCreateTexture2D(4, 4, 5, 2, 1, 1) == nullptr,
          "depth copy usage must be rejected");
  require(mgpuCreateTexture2D(4, 4, 2, 8, 1, 1) == nullptr,
          "unsupported storage format must be rejected");
  require(mgpuCreateTexture2D(4, 4, 1, 4, 1, 4) == nullptr,
          "excess mip count must be rejected");
  require(mgpuCreateSampler(0, 0, 0, 0, 0, 0, 4.0f, 2.0f, 0) == nullptr,
          "inverted sampler LOD range must be rejected");
}

void testViews() {
  MGPUTexture *arrayTexture = mgpuCreateTexture2D(4, 4, 1, 4, 6, 3);
  require(arrayTexture != nullptr, "array texture creation failed");
  MGPUTextureView *cube =
      mgpuCreateTextureView(arrayTexture, 2, 0, 1, 0, 6);
  require(cube != nullptr, "valid cube view creation failed");
  require(mgpuCreateTextureView(arrayTexture, 0, 0, 1, 0, 2) == nullptr,
          "2D view with multiple layers must be rejected");
  require(mgpuCreateTextureView(arrayTexture, 1, 3, 1, 0, 1) == nullptr,
          "mip base at source end must be rejected");
  require(mgpuCreateTextureView(arrayTexture, 1, 0, 1, 6, 1) == nullptr,
          "layer base at source end must be rejected");
  require(mgpuCreateTextureView(arrayTexture, 1, 2147483647, 2147483647,
                                0, 1) == nullptr,
          "overflowing view range must be rejected");
  mgpuDestroyTextureView(cube);
  mgpuDestroyTexture(arrayTexture);
}

void testGpuPath() {
  const uint8_t pixels[16] = {
      255, 0, 0, 255, 0, 255, 0, 255,
      0, 0, 255, 255, 255, 255, 255, 255,
  };
  MGPUTexture *source = mgpuCreateTexture2D(2, 2, 1, 2 | 4, 1, 1);
  MGPUTexture *storage = mgpuCreateTexture2D(2, 2, 1, 4 | 8, 1, 1);
  require(source && storage, "owned texture creation failed");
  MGPUTextureView *sourceView =
      mgpuCreateTextureView(source, 0, 0, 1, 0, 1);
  MGPUTextureView *storageView =
      mgpuCreateTextureView(storage, 0, 0, 1, 0, 1);
  MGPUSampler *nearest =
      mgpuCreateSampler(0, 0, 0, 0, 0, 0, 0.0f, 32.0f, 0);
  require(sourceView && storageView && nearest,
          "view or nearest sampler creation failed");
  require(mgpuWriteTexture2D(source, pixels, sizeof(pixels), 0, 0) == 0,
          "RGBA upload was not accepted");

  MGPUComputeShader *copyShader = mgpuCreateComputeShader();
  require(copyShader != nullptr, "copy shader creation failed");
  mgpuLoadKernel(copyShader, kSampleToStorage);
  require(mgpuSetOwnedTexture(copyShader, 0, sourceView, 0) == 0,
          "sampled texture bind failed");
  require(mgpuSetOwnedSampler(copyShader, 1, nearest, 0) == 0,
          "nearest sampler bind failed");
  require(mgpuSetOwnedTexture(copyShader, 2, storageView, 3) == 0,
          "storage texture bind failed");
  dispatch(copyShader);

  MGPUBuffer *readback = mgpuCreateBufferWithUsage(16, 4 | 8 | 128);
  MGPUComputeShader *readShader = mgpuCreateComputeShader();
  require(readback && readShader, "readback resources failed to allocate");
  mgpuLoadKernel(readShader, kTextureToBuffer);
  require(mgpuSetOwnedTexture(readShader, 0, storageView, 0) == 0,
          "storage result sampled bind failed");
  require(mgpuSetBufferRange(readShader, 1, readback, 0, 0, 16) == 0,
          "texture result buffer bind failed");
  dispatch(readShader);
  uint32_t actual[4] = {};
  mgpuReadSyncUint32(readback, actual, 4, 0);
  const uint32_t expected[4] = {0xff0000ffu, 0xff00ff00u, 0xffff0000u,
                                0xffffffffu};
  require(std::memcmp(actual, expected, sizeof(actual)) == 0,
          "sampled/storage texture round trip mismatch");

  MGPUTexture *wrongUsage = mgpuCreateTexture2D(2, 2, 1, 2, 1, 1);
  MGPUTextureView *wrongView =
      mgpuCreateTextureView(wrongUsage, 0, 0, 1, 0, 1);
  require(wrongUsage && wrongView, "wrong-usage fixture creation failed");
  require(mgpuSetOwnedTexture(readShader, 0, wrongView, 0) == 1,
          "sample bind without texture usage must be rejected");
  require(mgpuSetOwnedTexture(readShader, 0, sourceView, 2) == 1,
          "depth binding with color view must be rejected");

  MGPUTexture *staleTexture = mgpuCreateTexture2D(2, 2, 1, 4, 1, 1);
  MGPUTextureView *staleView =
      mgpuCreateTextureView(staleTexture, 0, 0, 1, 0, 1);
  require(staleTexture && staleView, "stale-view fixture creation failed");
  require(mgpuSetOwnedTexture(readShader, 0, staleView, 0) == 0,
          "live stale-view fixture bind failed");
  mgpuDrainWorkQueue();
  const uint32_t zeros[4] = {};
  mgpuWriteUint32(readback, zeros, sizeof(zeros));
  mgpuDestroyTexture(staleTexture);
  dispatch(readShader);
  uint32_t staleOutput[4] = {1, 1, 1, 1};
  mgpuReadSyncUint32(readback, staleOutput, 4, 0);
  require(std::memcmp(staleOutput, zeros, sizeof(zeros)) == 0,
          "dispatch used an explicitly destroyed texture");

  mgpuDestroyComputeShader(copyShader);
  mgpuDestroyComputeShader(readShader);
  mgpuDestroyBuffer(readback);
  mgpuDestroyTextureView(wrongView);
  mgpuDestroyTexture(wrongUsage);
  mgpuDestroyTextureView(staleView);
  mgpuDestroyTextureView(sourceView);
  mgpuDestroyTextureView(storageView);
  mgpuDestroySampler(nearest);
  mgpuDestroyTexture(source);
  mgpuDestroyTexture(storage);
  mgpuDrainWorkQueue();
}

void testRegionUploads() {
  require(mgpuTextureRegionUploadVersion() == 1,
          "texture region upload ABI version must be 1");
  MGPUTexture *texture = mgpuCreateTexture2D(4, 4, 1, 2 | 4, 2, 3);
  MGPUTextureView *selected = mgpuCreateTextureView(texture, 0, 1, 1, 1, 1);
  MGPUTextureView *otherLayer = mgpuCreateTextureView(texture, 0, 1, 1, 0, 1);
  MGPUTextureView *thinMip = mgpuCreateTextureView(texture, 0, 2, 1, 1, 1);
  MGPUBuffer *output = mgpuCreateBufferWithUsage(16, 4 | 8 | 128);
  MGPUComputeShader *read = mgpuCreateComputeShader();
  require(texture && selected && otherLayer && thinMip && output && read,
          "region upload fixtures failed to allocate");
  mgpuLoadKernel(read, kTextureToBuffer);
  require(mgpuSetBufferRange(read, 1, output, 0, 0, 16) == 0,
          "region readback buffer bind failed");
  const uint8_t zeros[16] = {};
  require(mgpuWriteTexture2D(texture, zeros, sizeof(zeros), 1, 1) == 0,
          "selected mip initialization failed");
  uint8_t column[8] = {255, 0, 0, 255, 0, 255, 0, 255};
  require(mgpuWriteTextureRegion2D(texture, column, sizeof(column),
                                  1, 0, 1, 2, 1, 1) == 0,
          "nonzero-origin narrow multi-row upload failed");
  // Accepted work owns the payload even before the worker consumes it.
  std::memset(column, 0, sizeof(column));
  const uint8_t blue[4] = {0, 0, 255, 255};
  require(mgpuWriteTextureRegion2D(texture, blue, sizeof(blue),
                                  0, 1, 1, 1, 1, 1) == 0,
          "second rectangular upload failed");
  require(mgpuSetOwnedTexture(read, 0, selected, 0) == 0,
          "selected region view bind failed");
  dispatch(read);
  uint32_t actual[4] = {};
  mgpuReadSyncUint32(output, actual, 4, 0);
  const uint32_t expected[4] = {0u, 0xff0000ffu, 0xffff0000u, 0xff00ff00u};
  require(std::memcmp(actual, expected, sizeof(actual)) == 0,
          "region upload overwrote untouched pixels or lost source bytes");
  require(mgpuSetOwnedTexture(read, 0, otherLayer, 0) == 0,
          "other layer view bind failed");
  dispatch(read);
  mgpuReadSyncUint32(output, actual, 4, 0);
  require(std::memcmp(actual, zeros, sizeof(actual)) == 0,
          "region upload changed another array layer");
  require(mgpuWriteTextureRegion2D(texture, blue, sizeof(blue),
                                  0, 0, 1, 1, 2, 1) == 0,
          "one-texel mip region upload failed");
  require(mgpuSetOwnedTexture(read, 0, thinMip, 0) == 0,
          "thin region mip bind failed");
  dispatch(read);
  mgpuReadSyncUint32(output, actual, 4, 0);
  require(actual[0] == 0xffff0000u, "thin region mip texel mismatch");

  require(mgpuWriteTextureRegion2D(texture, blue, 4, -1, 0, 1, 1, 1, 1) == 1,
          "negative origin was accepted");
  require(mgpuWriteTextureRegion2D(texture, blue, 4,
                                  2147483647, 0, 2147483647, 1, 1, 1) == 1,
          "overflowing region was accepted");
  require(mgpuWriteTextureRegion2D(texture, blue, 4, 2, 0, 1, 1, 1, 1) == 1 &&
          mgpuWriteTextureRegion2D(texture, blue, 4, 1, 0, 2, 1, 1, 1) == 1 &&
          mgpuWriteTextureRegion2D(texture, blue, 4, 0, 0, 0, 1, 1, 1) == 1 &&
          mgpuWriteTextureRegion2D(texture, blue, 3, 0, 0, 1, 1, 1, 1) == 1 &&
          mgpuWriteTextureRegion2D(texture, blue, 4, 0, 0, 1, 1, 3, 1) == 1 &&
          mgpuWriteTextureRegion2D(texture, blue, 4, 0, 0, 1, 1, 1, 2) == 1,
          "invalid region extent, length, mip or layer was accepted");
  MGPUTexture *noCopy = mgpuCreateTexture2D(2, 2, 1, 4, 1, 1);
  MGPUTexture *depth = mgpuCreateTexture2D(2, 2, 5, 4, 1, 1);
  require(noCopy && depth, "region refusal fixtures failed to allocate");
  require(mgpuWriteTextureRegion2D(noCopy, blue, 4, 0, 0, 1, 1, 0, 0) == 1 &&
          mgpuWriteTextureRegion2D(depth, blue, 4, 0, 0, 1, 1, 0, 0) == 1,
          "region upload accepted missing copy usage or a depth format");

  mgpuDestroyComputeShader(read);
  mgpuDestroyBuffer(output);
  mgpuDestroyTextureView(selected);
  mgpuDestroyTextureView(otherLayer);
  mgpuDestroyTextureView(thinMip);
  mgpuDestroyTexture(texture);
  mgpuDestroyTexture(noCopy);
  mgpuDestroyTexture(depth);
  mgpuDrainWorkQueue();
}

void testSecondaryOwnership() {
  MGPUContextHandle *context = mgpuCreateContextHandle(nullptr);
  require(context != nullptr, "secondary context creation failed");
  callbackDone.store(false, std::memory_order_release);
  mgpuContextInitializeAsync(context, complete);
  waitForCallback("secondary context initialization timed out");
  MGPUTexture *texture =
      mgpuContextCreateTexture2D(context, 2, 2, 1, 4, 1, 1);
  MGPUTextureView *view = mgpuCreateTextureView(texture, 0, 0, 1, 0, 1);
  MGPUSampler *sampler =
      mgpuContextCreateSampler(context, 0, 0, 0, 0, 0, 0, 0, 32, 0);
  MGPUComputeShader *globalShader = mgpuCreateComputeShader();
  MGPUComputeShader *localShader = mgpuContextCreateComputeShader(context);
  require(texture && view && sampler && globalShader && localShader,
          "secondary ownership fixtures failed");
  require(mgpuSetOwnedTexture(globalShader, 0, view, 0) == 1,
          "foreign texture view must be rejected");
  require(mgpuSetOwnedSampler(globalShader, 1, sampler, 0) == 1,
          "foreign sampler must be rejected");
  require(mgpuSetOwnedTexture(localShader, 0, view, 0) == 0,
          "same-context texture control binding must succeed");
  require(mgpuSetOwnedSampler(localShader, 1, sampler, 0) == 0,
          "same-context sampler control binding must succeed");
  mgpuDestroyComputeShader(localShader);
  mgpuDestroyComputeShader(globalShader);
  mgpuDestroyTextureView(view);
  mgpuDestroySampler(sampler);
  mgpuDestroyTexture(texture);
  // Context destruction drains its own FIFO; keep this public-ABI test free of
  // non-exported WebGPUThread linkage.
  mgpuDestroyContextHandle(context);
  mgpuDrainWorkQueue();
}

} // namespace

int main() {
  require(mgpuCreateTexture2D(2, 2, 1, 4, 1, 1) == nullptr,
          "texture creation must not lazily initialize the default context");
  require(mgpuCreateSampler(0, 0, 0, 0, 0, 0, 0, 32, 0) == nullptr,
          "sampler creation must not lazily initialize the default context");
  mgpuInitializeContext();
  testCreationValidation();
  testViews();
  testGpuPath();
  testRegionUploads();
  testSecondaryOwnership();
  mgpuDestroyContext();
  std::cout << "Texture resource C ABI test passed." << std::endl;
  return 0;
}
