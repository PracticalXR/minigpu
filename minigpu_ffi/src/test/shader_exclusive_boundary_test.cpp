// Production C ABI cold paths only. No device initialization, capture or pixels.
#include "../include/minigpu.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>

static void require(bool value, const char *message) {
  if (!value) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}
struct Receipt { int calls = 0; int result = 99; };
static void complete(int result, void *userdata) {
  auto *receipt = static_cast<Receipt *>(userdata);
  ++receipt->calls;
  receipt->result = result;
}
static MGPUComputeShader *coldShader;
static std::promise<void> submitted;
static std::atomic<int> dispatchCalls{0};
static void afterRefusedDispatch() {
  // A callback reentering a mutator would deadlock if submission/GPU locks
  // remained held. Loading source text does not compile/initialize anything.
  require(!mgpuHasKernel(coldShader), "cold shader has no kernel");
  mgpuLoadKernel(coldShader, "// source retained without GPU compilation");
  require(mgpuHasKernel(coldShader), "callback can reenter shader mutation");
  dispatchCalls.fetch_add(1);
  submitted.set_value();
}
static int nullDispatchCalls = 0;
static void afterNullDispatch() { ++nullDispatchCalls; }

int main() {
  MGPUContextLifetimeStats before{}, after{};
  require(mgpuContextGetLifetimeStats(&before, sizeof(before)) == 0,
          "initial native ownership snapshot");
  const auto references = mgpuContextRefCount();
  auto *sentinel = reinterpret_cast<MGPUComputeShaderUse *>(uintptr_t(1));
  auto *use = sentinel;
  require(mgpuComputeShaderAcquireExclusiveUse(nullptr, nullptr, &use) ==
              MGPU_CONTEXT_LEASE_INVALID && !use,
          "missing original owners clear output and refuse");
  require(mgpuComputeShaderAcquireExclusiveUse(nullptr, nullptr, nullptr) ==
              MGPU_CONTEXT_LEASE_INVALID,
          "missing output refuses");
  require(mgpuComputeShaderUseDispatch(nullptr, 1, 1, 1) ==
              MGPU_CONTEXT_LEASE_INVALID &&
          mgpuComputeShaderUseRelease(nullptr) == MGPU_CONTEXT_LEASE_INVALID,
          "missing use never submits or releases an owner");
  Receipt receipt;
  mgpuComputeShaderUseRetireAsync(nullptr, complete, &receipt);
  require(receipt.calls == 1 && receipt.result == MGPU_CONTEXT_LEASE_INVALID,
          "missing token reports one refusal, not retirement");
  mgpuComputeShaderUseRetireAsync(nullptr, nullptr, &receipt);
  require(receipt.calls == 1, "null callback starts no retirement");

  coldShader = mgpuCreateComputeShader();
  require(coldShader != nullptr, "create native CPU shader owner");
  use = sentinel;
  require(mgpuComputeShaderAcquireExclusiveUse(nullptr, coldShader, &use) ==
              MGPU_CONTEXT_LEASE_INVALID && !use,
          "a raw shader cannot substitute for an original creation lease");
  auto future = submitted.get_future();
  mgpuDispatchAsync(coldShader, 1, 1, 1, afterRefusedDispatch);
  require(future.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
          "refused cold dispatch acknowledges outside native locks");
  mgpuDrainWorkQueue();
  require(dispatchCalls.load() == 1, "cold dispatch callback occurs once");
  mgpuDispatchAsync(nullptr, 1, 1, 1, afterNullDispatch);
  require(nullDispatchCalls == 1, "null shader preserves callback completion");
  mgpuDestroyComputeShader(coldShader);
  coldShader = nullptr;
  mgpuDrainWorkQueue();
  MGPUContextLease *lease = nullptr;
  MGPUContextHandle *context = nullptr;
  require(mgpuDefaultContextRetainLease(&lease, &context) ==
              MGPU_CONTEXT_LEASE_NOT_READY && !lease && !context,
          "cold shader/refusal/deletion paths did not initialize a GPU");
  require(mgpuContextGetLifetimeStats(&after, sizeof(after)) == 0 &&
              before.live_contexts == after.live_contexts &&
              before.live_leases == after.live_leases &&
              before.quarantined_contexts == after.quarantined_contexts &&
              references == mgpuContextRefCount(),
          "cold use refusals consume no context, lease or attachment");
  std::cout << "PASS shader scope C ABI refusals, reentrant cold dispatch "
               "acknowledgement and FIFO deletion; no GPU initialization\n";
}
