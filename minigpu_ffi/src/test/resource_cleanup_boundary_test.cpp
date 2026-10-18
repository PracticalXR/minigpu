// Actual exported C ABI, cold resource ownership only. Never initializes GPU.
#include "../include/minigpu.h"
#include <cstdlib>
#include <iostream>

extern "C" int resource_cleanup_c_abi(void);

static void require(bool value, const char *message) {
  if (!value) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}
static int callbacks = 0;
static void completed(int, void *) { ++callbacks; }

int main() {
  MGPUContextLifetimeStats before{}, after{};
  require(mgpuContextGetLifetimeStats(&before, sizeof(before)) == 0,
          "initial creation-owner accounting");
  const auto references = mgpuContextRefCount();
  require(resource_cleanup_c_abi(), "plain C declarations link and refuse");
  auto *leaseSentinel = reinterpret_cast<MGPUContextLease *>(uintptr_t(1));
  auto *bufferSentinel = reinterpret_cast<MGPUBuffer *>(uintptr_t(1));
  require(mgpuComputeShaderDestroyCheckedAsync(nullptr, nullptr, completed,
                                               nullptr) ==
              MGPU_CONTEXT_LEASE_INVALID &&
          mgpuBufferDestroyCheckedAsync(nullptr, nullptr, completed, nullptr) ==
              MGPU_CONTEXT_LEASE_INVALID,
          "missing resources refuse without a cleanup callback");
  require(mgpuBufferDestroyCheckedAsync(leaseSentinel, bufferSentinel, nullptr,
                                        nullptr) == MGPU_CONTEXT_LEASE_INVALID,
          "callback validation precedes touching an untrusted owner");
  require(mgpuBufferDestroyCheckedAsync(nullptr, bufferSentinel, completed,
                                        nullptr) == MGPU_CONTEXT_LEASE_INVALID &&
          mgpuBufferDestroyCheckedAsync(leaseSentinel, nullptr, completed,
                                        nullptr) == MGPU_CONTEXT_LEASE_INVALID,
          "absent original lease/resource never dereferences its counterpart");

  auto *shader = mgpuCreateComputeShader();
  require(shader != nullptr, "native CPU-only shader owner exists");
  require(mgpuComputeShaderDestroyCheckedAsync(nullptr, shader, completed,
                                               nullptr) ==
              MGPU_CONTEXT_LEASE_INVALID &&
          mgpuComputeShaderDestroyCheckedAsync(leaseSentinel, shader, nullptr,
                                               nullptr) ==
              MGPU_CONTEXT_LEASE_INVALID,
          "refused shader cleanup consumes no owner");
  mgpuLoadKernel(shader, "// cold shader remains caller owned after refusal");
  require(mgpuHasKernel(shader), "caller can still mutate its refused resource");
  require(callbacks == 0, "synchronous refusal never schedules a callback");
  mgpuDestroyComputeShader(shader);
  mgpuDrainWorkQueue();

  MGPUContextLease *lease = nullptr;
  MGPUContextHandle *context = nullptr;
  require(mgpuDefaultContextRetainLease(&lease, &context) ==
              MGPU_CONTEXT_LEASE_NOT_READY && !lease && !context,
          "cleanup refusal and caller-owned deletion initialized no GPU");
  require(mgpuContextGetLifetimeStats(&after, sizeof(after)) == 0 &&
              before.live_contexts == after.live_contexts &&
              before.live_leases == after.live_leases &&
              before.quarantined_contexts == after.quarantined_contexts &&
              references == mgpuContextRefCount() && callbacks == 0,
          "cold cleanup leaves native owners/attachments/callbacks unchanged");
  std::cout << "PASS checked resource cleanup C/C++ ABI refusals and retained "
               "caller ownership; no GPU initialization\n";
}
