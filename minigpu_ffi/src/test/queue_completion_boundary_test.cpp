// Actual native ABI refusals only: no initialization, shaders, windows or pixels.
#include "../include/minigpu.h"
#include "../include/minigpu_context_lifetime.h"
#include <cstdlib>
#include <iostream>

struct Receipt {
  int calls = 0;
  int result = 99;
};

static void complete(int result, void *userdata) {
  auto *receipt = static_cast<Receipt *>(userdata);
  ++receipt->calls;
  receipt->result = result;
}

static void require(bool value, const char *message) {
  if (!value) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

int main() {
  MGPUContextLifetimeStats before{}, after{};
  require(mgpuContextGetLifetimeStats(&before, sizeof(before)) == 0,
          "initial native ownership snapshot");
  const auto references = mgpuContextRefCount();
  Receipt receipt;
  mgpuContextLeaseWaitForSubmittedWorkAsync(nullptr, complete, &receipt);
  require(receipt.calls == 1 && receipt.result == MGPU_CONTEXT_LEASE_INVALID,
          "missing original lease reports one refusal, never completion");
  mgpuContextLeaseWaitForSubmittedWorkAsync(nullptr, nullptr, &receipt);
  require(receipt.calls == 1, "null callback cannot start or acknowledge a job");
  MGPUContextLease *lease = nullptr;
  MGPUContextHandle *context = nullptr;
  require(mgpuDefaultContextRetainLease(&lease, &context) ==
              MGPU_CONTEXT_LEASE_NOT_READY && !lease && !context,
          "wait refusal cannot initialize a display device");
  require(mgpuContextGetLifetimeStats(&after, sizeof(after)) == 0 &&
              before.live_contexts == after.live_contexts &&
              before.live_leases == after.live_leases &&
              before.quarantined_contexts == after.quarantined_contexts &&
              references == mgpuContextRefCount(),
          "refused waits consume no context, lease or attachment");
  std::cout << "PASS native queue callback refusals and cold ownership; "
               "no GPU initialization\n";
}
