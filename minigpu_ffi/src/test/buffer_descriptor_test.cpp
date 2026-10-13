#include "../include/minigpu.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

namespace {

constexpr uint32_t kCopySrc = 4;
constexpr uint32_t kCopyDst = 8;
constexpr uint32_t kIndex = 16;
constexpr uint32_t kVertex = 32;
constexpr uint32_t kUniform = 64;
constexpr uint32_t kStorage = 128;

std::atomic<bool> secondaryInitialized{false};
std::atomic<bool> dispatchFinished{false};

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

void onSecondaryInitialized() {
  secondaryInitialized.store(true, std::memory_order_release);
}

void onDispatchFinished() {
  dispatchFinished.store(true, std::memory_order_release);
}

void waitForFlag(std::atomic<bool> &flag, const char *timeoutMessage) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!flag.load(std::memory_order_acquire)) {
    require(std::chrono::steady_clock::now() < deadline, timeoutMessage);
    std::this_thread::yield();
  }
}

void waitForSecondaryInitialization() {
  waitForFlag(secondaryInitialized,
              "secondary context initialization timed out");
}

mgpu::Buffer *nativeBuffer(MGPUBuffer *buffer) {
  return reinterpret_cast<mgpu::Buffer *>(buffer);
}

void requireBuffer(MGPUBuffer *buffer, uint64_t size, WGPUBufferUsage usage,
                   const char *message) {
  require(buffer != nullptr, message);
  WGPUBuffer handle = nativeBuffer(buffer)->getWGPUBuffer();
  require(handle != nullptr, "native WebGPU buffer is null");
  require(wgpuBufferGetSize(handle) == size, "unexpected WebGPU buffer size");
  require(wgpuBufferGetUsage(handle) == usage,
          "unexpected WebGPU buffer usage");
}

void testValidation() {
  require(mgpuBufferDescriptorVersion() == 1,
          "buffer descriptor ABI version must be 1");
  require(mgpuCreateBufferWithUsage(0, kStorage) == nullptr,
          "zero size must be rejected");
  require(mgpuCreateBufferWithUsage(6, kStorage) == nullptr,
          "unaligned size must be rejected");
  require(mgpuCreateBufferWithUsage(-4, kStorage) == nullptr,
          "negative size must be rejected");
  require(mgpuCreateBufferWithUsage(4, 0) == nullptr,
          "zero usage must be rejected");
  require(mgpuCreateBufferWithUsage(4, 0x100) == nullptr,
          "unknown usage bits must be rejected");
  require(mgpuContextCreateBufferWithUsage(nullptr, 4, kStorage) == nullptr,
          "null context must be rejected");
}

void testGlobalBuffers() {
  const WGPUBufferUsage transferUniform =
      WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst |
      WGPUBufferUsage_Uniform;
  MGPUBuffer *uniform = mgpuCreateBufferWithUsage(
      256, kCopySrc | kCopyDst | kUniform);
  requireBuffer(uniform, 256, transferUniform,
                "uniform descriptor allocation failed");

  const uint32_t input[4] = {0x10203040u, 0x55667788u, 0x90abcdefu,
                             0xfedcba09u};
  uint32_t output[4] = {};
  mgpuWriteUint32(uniform, input, sizeof(input));
  mgpuReadSyncUint32(uniform, output, 4, 0);
  require(std::memcmp(input, output, sizeof(input)) == 0,
          "descriptor buffer upload/readback mismatch");

  uint32_t offsetOutput[2] = {};
  mgpuReadSyncUint32(uniform, offsetOutput, 2, 1);
  require(offsetOutput[0] == input[1] && offsetOutput[1] == input[2],
          "descriptor buffer element offset mismatch");
  mgpuDestroyBuffer(uniform);

  struct UsageCase {
    uint32_t abiUsage;
    WGPUBufferUsage nativeUsage;
    const char *failure;
  };
  const UsageCase cases[] = {
      {kVertex, WGPUBufferUsage_Vertex, "vertex allocation failed"},
      {kIndex, WGPUBufferUsage_Index, "index allocation failed"},
      {kStorage, WGPUBufferUsage_Storage, "storage allocation failed"},
  };
  for (const UsageCase &entry : cases) {
    MGPUBuffer *buffer = mgpuCreateBufferWithUsage(256, entry.abiUsage);
    requireBuffer(buffer, 256, entry.nativeUsage, entry.failure);
    mgpuDestroyBuffer(buffer);
  }

  const WGPUBufferUsage legacyUsage =
      WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst |
      WGPUBufferUsage_Storage;
  MGPUBuffer *legacy = mgpuCreateBuffer(256, 9);
  requireBuffer(legacy, 256, legacyUsage, "legacy uint32 allocation failed");
  mgpuDestroyBuffer(legacy);
  mgpuDrainWorkQueue();
}

void testSecondaryContext() {
  MGPUContextHandle *context = mgpuCreateContextHandle(nullptr);
  require(context != nullptr, "secondary context creation failed");
  secondaryInitialized.store(false, std::memory_order_release);
  mgpuContextInitializeAsync(context, onSecondaryInitialized);
  waitForSecondaryInitialization();

  const WGPUBufferUsage usage =
      WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex | WGPUBufferUsage_Storage;
  MGPUBuffer *buffer =
      mgpuContextCreateBufferWithUsage(context, 128, kCopyDst | kVertex | kStorage);
  requireBuffer(buffer, 128, usage,
                "secondary descriptor allocation failed");

  MGPUComputeShader *globalShader = mgpuCreateComputeShader();
  require(globalShader != nullptr, "global shader allocation failed");
  require(mgpuSetBufferRange(globalShader, 0, buffer, 0, 0, 128) == 1,
          "foreign-context binding must be rejected");
  MGPUComputeShader *localShader = mgpuContextCreateComputeShader(context);
  require(localShader != nullptr, "secondary shader allocation failed");
  require(mgpuSetBufferRange(localShader, 0, buffer, 0, 0, 128) == 0,
          "same-context control binding must succeed");
  mgpuDestroyComputeShader(localShader);
  mgpuDestroyComputeShader(globalShader);
  mgpuDestroyBuffer(buffer);

  // Context destruction drains its own FIFO; do not call non-exported C++
  // worker internals from a consumer of the public C ABI.
  mgpuDestroyContextHandle(context);
  mgpuDrainWorkQueue();
}

constexpr const char *kReadOnlyKernel = R"WGSL(
struct Params {
  addend: u32,
  multiplier: u32,
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputValues: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<u32>;
@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < 4u) {
    outputValues[id.x] =
        inputValues[id.x] * params.multiplier + params.addend;
  }
}
)WGSL";

constexpr const char *kLegacyKernel = R"WGSL(
struct Params {
  addend: u32,
  multiplier: u32,
  pad0: u32,
  pad1: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> inputValues: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputValues: array<u32>;
@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < 4u) {
    outputValues[id.x] =
        inputValues[id.x] * params.multiplier + params.addend;
  }
}
)WGSL";

void dispatchAndRequire(MGPUComputeShader *shader, const uint32_t expected[4],
                        MGPUBuffer *output, const char *message) {
  dispatchFinished.store(false, std::memory_order_release);
  mgpuDispatchAsync(shader, 1, 1, 1, onDispatchFinished);
  waitForFlag(dispatchFinished, "compute dispatch timed out");
  uint32_t actual[4] = {};
  mgpuReadSyncUint32(output, actual, 4, 0);
  require(std::memcmp(actual, expected, sizeof(actual)) == 0, message);
}

void testTypedBindings() {
  require(mgpuBufferBindingVersion() == 1,
          "buffer binding ABI version must be 1");

  MGPUComputeShader *shader = mgpuCreateComputeShader();
  require(shader != nullptr, "compute shader allocation failed");
  mgpuLoadKernel(shader, kReadOnlyKernel);

  MGPUBuffer *params =
      mgpuCreateBufferWithUsage(512, kUniform | kCopyDst);
  MGPUBuffer *input =
      mgpuCreateBufferWithUsage(768, kStorage | kCopyDst);
  MGPUBuffer *output =
      mgpuCreateBufferWithUsage(16, kStorage | kCopySrc);
  require(params && input && output, "typed binding buffers failed to allocate");

  const uint32_t paramValues[4] = {3, 2, 0, 0};
  const uint32_t firstInput[4] = {5, 7, 11, 13};
  const uint32_t secondInput[4] = {17, 19, 23, 29};
  const uint32_t legacyInput[4] = {31, 37, 41, 43};
  mgpuWriteBufferAt(params, paramValues, sizeof(paramValues), 256);
  mgpuWriteBufferAt(input, legacyInput, sizeof(legacyInput), 0);
  mgpuWriteBufferAt(input, firstInput, sizeof(firstInput), 256);
  mgpuWriteBufferAt(input, secondInput, sizeof(secondInput), 512);

  require(mgpuSetBufferRange(shader, 0, params, 2, 256, 16) == 0,
          "uniform range bind was not accepted");
  require(mgpuSetBufferRange(shader, 1, input, 1, 256, 16) == 0,
          "read-only storage range bind was not accepted");
  require(mgpuSetBufferRange(shader, 2, output, 0, 0, 16) == 0,
          "storage output bind was not accepted");

  require(mgpuSetBufferRange(shader, 0, params, 2, 4, 16) == 1,
          "misaligned uniform offset must be rejected");
  require(mgpuSetBufferRange(shader, 1, input, 1, 256, 6) == 1,
          "unaligned binding size must be rejected");
  require(mgpuSetBufferRange(shader, 1, input, 1, 768, 4) == 1,
          "out-of-range binding must be rejected");
  require(mgpuSetBufferRange(shader, 0, output, 2, 0, 16) == 1,
          "wrong buffer usage must be rejected");

  const uint32_t firstExpected[4] = {13, 17, 25, 29};
  dispatchAndRequire(shader, firstExpected, output,
                     "typed offset-256 dispatch mismatch");

  require(mgpuSetBufferRange(shader, 1, input, 1, 512, 16) == 0,
          "same-handle offset rebind was not accepted");
  const uint32_t secondExpected[4] = {37, 41, 49, 61};
  dispatchAndRequire(shader, secondExpected, output,
                     "same-handle offset rebind did not rebuild bind group");

  mgpuSetBuffer(shader, 1, input);
  mgpuLoadKernel(shader, kLegacyKernel);
  const uint32_t legacyExpected[4] = {65, 77, 85, 89};
  dispatchAndRequire(shader, legacyExpected, output,
                     "legacy bind did not restore full storage binding");

  mgpuDestroyComputeShader(shader);
  mgpuDestroyBuffer(params);
  mgpuDestroyBuffer(input);
  mgpuDestroyBuffer(output);
  mgpuDrainWorkQueue();
}

} // namespace

int main() {
  mgpuInitializeContext();
  testValidation();
  testGlobalBuffers();
  testTypedBindings();
  testSecondaryContext();
  mgpuDestroyContext();
  std::cout << "Buffer descriptor C ABI test passed." << std::endl;
  return 0;
}
