#include "../include/minigpu.h"
#include "../include/minigpu_external_pool.h"
#include <atomic>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

namespace {
std::atomic<bool> initialized{false};
void ready() { initialized.store(true, std::memory_order_release); }
void require(bool good, const char *message) {
  if (!good) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}
MGPUContextLifetimeStats stats() {
  MGPUContextLifetimeStats value{};
  require(mgpuContextGetLifetimeStats(&value, sizeof(value)) == 0, "stats");
  return value;
}
template <typename Predicate>
void until(Predicate predicate, const char *message) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  while (!predicate() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  require(predicate(), message);
}

void testDefaultContext() {
  const auto baseline = stats();
  require(mgpuContextRefCount() == 0, "cold display attachment baseline");
  auto *lease = reinterpret_cast<MGPUContextLease *>(uintptr_t(1));
  auto *context = reinterpret_cast<MGPUContextHandle *>(uintptr_t(1));
  require(mgpuDefaultContextRetainLease(&lease, &context) ==
                  MGPU_CONTEXT_LEASE_NOT_READY &&
              !lease && !context && mgpuContextRefCount() == 0,
          "cold display retention clears outputs without initializing");
  initialized = false;
  mgpuInitializeContextAsync(ready);
  until([] { return initialized.load(std::memory_order_acquire); },
        "real display initialization");
  require(mgpuDefaultContextRetainLease(&lease, &context) == 0 && lease &&
              context && mgpuContextRefCount() == 2,
          "ready display retention owns one attachment");
  MGPUContextLeaseInfo before{}, after{};
  require(mgpuContextLeaseGetInfo(lease, &before, sizeof(before)) == 0 &&
              before.current && before.context_generation &&
              before.device_generation,
          "display creation state is current");
  MGPUExternalPoolCapabilities facilities{};
  require(mgpuExternalPoolQueryCapabilities(context, &facilities,
                                            sizeof(facilities)) == 0 &&
              facilities.context_generation == before.context_generation &&
              facilities.device_generation == before.device_generation,
          "native pool queries the same display creation device");
  MGPUContextLease *consumer = nullptr, *otherScope = nullptr;
  MGPUContextHandle *otherContext = nullptr;
  require(mgpuContextRetainLease(context, &consumer) == 0 && consumer &&
              mgpuDefaultContextRetainLease(&otherScope, &otherContext) == 0 &&
              otherContext == context && mgpuContextRefCount() == 2,
          "native consumer and second host scope share one attachment");
  require(mgpuContextLeaseGetInfo(otherScope, &after, sizeof(after)) == 0 &&
              after.current && after.context_generation == before.context_generation &&
              after.device_generation == before.device_generation,
          "concurrent host scopes preserve exact creation identity");
  mgpuContextReleaseLease(otherScope);
  mgpuDestroyContext(); // The renderer releases its ordinary attachment.
  mgpuContextReleaseLease(lease); // The synchronous host scope returns.
  require(mgpuContextRefCount() == 1 &&
              mgpuContextLeaseGetInfo(consumer, &after, sizeof(after)) == 0 &&
              after.current && after.context_generation == before.context_generation &&
              after.device_generation == before.device_generation &&
              mgpuExternalPoolQueryCapabilities(context, &facilities,
                                                sizeof(facilities)) == 0,
          "native consumer keeps the display device and worker live");
  mgpuContextLeaseWaitForSubmittedWorkAsyncToPort(consumer, 0, 1);
  mgpuContextReleaseLease(consumer);
  until([&] { return stats().live_contexts == baseline.live_contexts; },
        "last native completion drains the ready display attachment");
  require(mgpuContextRefCount() == 0 &&
              stats().live_leases == baseline.live_leases &&
              stats().quarantined_contexts == baseline.quarantined_contexts &&
              mgpuDefaultContextRetainLease(&lease, &context) ==
                  MGPU_CONTEXT_LEASE_NOT_READY && !lease && !context,
          "default teardown recovers quotas and cannot lazily recreate");
  std::cout << "PASS: actual display device shared by native consumers; cold "
               "refusal, renderer detach, exact completion and quota recovery\n";
}

void deviceRetired(int result, void *userdata) {
  static_cast<std::atomic<int> *>(userdata)->store(result,
                                                   std::memory_order_release);
}

void testDeviceRetirement() {
  const auto baseline = stats();
  std::atomic<int> result{99};
  mgpuContextLeaseRetireDeviceAsync(nullptr, deviceRetired, &result);
  require(result.load() == MGPU_CONTEXT_LEASE_INVALID,
          "missing creation pin cannot certify device retirement");
  mgpuContextLeaseObserveRetirementAsync(nullptr, deviceRetired, &result);
  require(result.load() == MGPU_CONTEXT_LEASE_INVALID,
          "missing creation pin cannot observe context retirement");

  initialized = false;
  mgpuInitializeContextAsync(ready);
  until([] { return initialized.load(); }, "display initialization");
  MGPUContextLease *lease = nullptr, *observer = nullptr;
  MGPUContextHandle *context = nullptr;
  require(mgpuDefaultContextRetainLease(&lease, &context) == 0 &&
              mgpuContextRetainLease(context, &observer) == 0,
          "retain original device before loss");
  MGPUContextLeaseInfo before{}, after{};
  require(mgpuContextLeaseGetInfo(observer, &before, sizeof(before)) == 0 &&
              before.current, "original generation");
  const auto observing = stats();
  std::array<std::atomic<int>, 64> contextResults;
  for (auto &completion : contextResults) {
    completion = 99;
    mgpuContextLeaseObserveRetirementAsync(lease, deviceRetired, &completion);
    require(completion.load() == 99, "observation does not retire the context");
  }
  result = 99;
  mgpuContextLeaseObserveRetirementAsync(lease, deviceRetired, &result);
  require(result.load() == MGPU_CONTEXT_LEASE_WOULD_BLOCK &&
              stats().live_leases == observing.live_leases &&
              stats().live_contexts == observing.live_contexts,
          "bounded observers retain no attachment or lease");
  char adapterBefore[256]{};
  mgpuGetSelectedAdapterName(adapterBefore, sizeof(adapterBefore));
  auto *buffer = mgpuCreateBufferWithUsage(16, 0x8c);
  require(buffer != nullptr, "real original-device buffer");

  result = 99;
  mgpuContextLeaseRetireDeviceAsync(lease, deviceRetired, &result);
  mgpuContextReleaseLease(lease); // Work retains its own exact creation pin.
  require(mgpuContextLeaseGetInfo(observer, &after, sizeof(after)) == 0 &&
              !after.current &&
              before.device_generation == after.device_generation,
          "retirement closes admission synchronously without replacement");
  until([&] { return result.load() != 99; }, "native device destruction receipt");
  require(result.load() == MGPU_CONTEXT_LEASE_OK,
          "actual native destruction completed");

  uint64_t size = 0, binding = 0, generation = 0;
  require(!mgpuCreateBufferWithUsage(16, 0x8c) &&
              !mgpuContextCreateBuffer(context, 16, 0) &&
              !mgpuGetCurrentDeviceBufferLimits(&size, &binding, &generation),
          "ordinary allocation cannot resurrect the lost device");
  const auto attachments = mgpuContextRefCount();
  mgpuInitializeContext(); // C ABI must swallow refusal and return its charge.
  initialized = false;
  mgpuInitializeContextAsync(ready);
  until([] { return initialized.load(); }, "refused init still completes");
  require(mgpuContextRefCount() == attachments &&
              !mgpuGetCurrentDeviceBufferLimits(&size, &binding, &generation) &&
              mgpuContextLeaseGetInfo(observer, &after, sizeof(after)) == 0 &&
              !after.current &&
              after.device_generation == before.device_generation,
          "explicit initialization also waits for consumer retirement");
  require(stats().live_contexts == baseline.live_contexts + 1 &&
              stats().live_leases == baseline.live_leases + 1,
          "device destruction is not consumer or export retirement");
  for (auto &completion : contextResults)
    require(completion.load() == 99, "GPU abort is not final context retirement");

  result = 99;
  mgpuContextLeaseRetireDeviceAsync(observer, deviceRetired, &result);
  until([&] { return result.load() != 99; }, "repeat exact-device retirement");
  require(result.load() == MGPU_CONTEXT_LEASE_OK,
          "repeat retirement preserves the creation device");
  mgpuDestroyBuffer(buffer);
  mgpuDestroyContext();
  require(contextResults[0].load() == 99, "held creation lease blocks final receipt");
  mgpuContextReleaseLease(observer);
  until([&] { return contextResults.back().load() != 99; },
        "non-owning observer completes after the last attachment");
  for (auto &completion : contextResults)
    require(completion.load() == MGPU_CONTEXT_LEASE_OK,
            "every admitted observer receives successful final retirement");
  until([&] { return stats().live_contexts == baseline.live_contexts; },
        "only final consumers release the context");
  require(mgpuContextRefCount() == 0 &&
              stats().live_leases == baseline.live_leases &&
              stats().quarantined_contexts == baseline.quarantined_contexts,
          "actual consumer retirement recovers its quota");

  initialized = false;
  mgpuInitializeContextAsync(ready);
  until([] { return initialized.load(); }, "fresh explicit initialization");
  char adapterAfter[256]{};
  mgpuGetSelectedAdapterName(adapterAfter, sizeof(adapterAfter));
  require(mgpuGetCurrentDeviceBufferLimits(&size, &binding, &generation) &&
              generation > before.device_generation &&
              std::string(adapterAfter) == adapterBefore,
          "fresh generation on the same available adapter");
  buffer = mgpuCreateBufferWithUsage(16, 0x8c);
  require(buffer != nullptr, "new device remains usable");
  mgpuDestroyBuffer(buffer);
  mgpuDestroyContext();
  std::cout << "PASS: native device retirement, retained consumers, no implicit "
               "repair and explicit same-adapter reinitialization\n";
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--retirement") {
    testDeviceRetirement();
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) == "--default") {
    testDefaultContext();
    return 0;
  }
  auto baseline = stats();
  auto *context = mgpuCreateContextHandle(nullptr);
  require(context != nullptr, "create owned context");
  auto *lease = reinterpret_cast<MGPUContextLease *>(uintptr_t(1));
  require(mgpuContextRetainLease(context, &lease) ==
                  MGPU_CONTEXT_LEASE_NOT_READY &&
              lease == nullptr,
          "uninitialized retention refuses without repair");
  MGPUExternalPoolCapabilities facilities{};
  require(mgpuExternalPoolQueryCapabilities(context, &facilities,
                                            sizeof(facilities)) ==
              MGPU_CONTEXT_LEASE_NOT_READY,
          "uninitialized capability query never initializes");
  mgpuContextInitializeAsync(context, ready);
  until([] { return initialized.load(std::memory_order_acquire); },
        "native initialization");
  require(mgpuContextRetainLease(context, &lease) == 0 && lease,
          "ready device lease");
  require(mgpuExternalPoolQueryCapabilities(context, &facilities,
                                            sizeof(facilities)) == 0 &&
              facilities.context_generation && facilities.device_generation &&
              facilities.max_texture_dimension_2d,
          "query actual receiving backend and limits");
  if (facilities.backend == MGPU_EXTERNAL_POOL_BACKEND_D3D11) {
    require(
        (facilities.adapter_luid_low || facilities.adapter_luid_high) &&
            (facilities.refusal == MGPU_EXTERNAL_POOL_FACILITIES_READY ||
             facilities.refusal == MGPU_EXTERNAL_POOL_NATIVE_FEATURE_MISSING),
        "D3D11 probes exact adapter and interfaces");
  } else {
    require(facilities.refusal == MGPU_EXTERNAL_POOL_BACKEND_UNQUALIFIED &&
                !facilities.facility_flags,
            "other backend never uses D3D11 helper");
  }
  MGPUContextLeaseInfo before{}, after{};
  require(mgpuContextLeaseGetInfo(lease, &before, sizeof(before)) == 0 &&
              before.current == 1 && before.context_generation &&
              before.device_generation,
          "exact generation reports current");
  std::vector<MGPUContextLease *> leases;
  for (uint32_t index = 1; index < baseline.max_leases; ++index) {
    MGPUContextLease *extra = nullptr;
    require(mgpuContextRetainLease(context, &extra) == 0,
            "bounded retention capacity");
    leases.push_back(extra);
  }
  MGPUContextLease *overflow =
      reinterpret_cast<MGPUContextLease *>(uintptr_t(1));
  require(mgpuContextRetainLease(context, &overflow) ==
                  MGPU_CONTEXT_LEASE_WOULD_BLOCK &&
              !overflow,
          "retention pressure clears output");
  for (auto *extra : leases)
    mgpuContextReleaseLease(extra);
  mgpuDestroyContextHandle(context);
  require(mgpuContextLeaseGetInfo(lease, &after, sizeof(after)) == 0 &&
              !after.current &&
              after.context_generation == before.context_generation &&
              after.device_generation == before.device_generation,
          "retirement invalidates admission, preserves creation identity");
  require(stats().live_contexts == baseline.live_contexts + 1 &&
              stats().live_leases == baseline.live_leases + 1,
          "retired creation device remains charged by held lease");
  overflow = reinterpret_cast<MGPUContextLease *>(uintptr_t(1));
  require(mgpuContextRetainLease(context, &overflow) ==
                  MGPU_CONTEXT_LEASE_RETIRED &&
              !overflow,
          "retired handle cannot be repaired by retaining it");
  // Completion draining is permitted after retirement and retains its exact
  // queue/instance even when the app drops its own lease immediately afterward.
  mgpuContextLeaseWaitForSubmittedWorkAsyncToPort(lease, 0, 1);
  mgpuContextReleaseLease(lease);
  until([&] { return stats().live_contexts == baseline.live_contexts; },
        "final lease drains owner");
  require(stats().live_leases == baseline.live_leases &&
              stats().quarantined_contexts == baseline.quarantined_contexts,
          "clean retirement");

  // Retirement immediately after scheduling init must retain its raw captures.
  initialized = false;
  context = mgpuCreateContextHandle(nullptr);
  require(context != nullptr, "create asynchronous retirement context");
  mgpuContextInitializeAsync(context, ready);
  mgpuDestroyContextHandle(context);
  until([] { return initialized.load(std::memory_order_acquire); },
        "retired init callback completes");
  until([&] { return stats().live_contexts == baseline.live_contexts; },
        "retired initializer joins safely");
  std::cout << "PASS: native creation-device retention, bounded pressure and "
               "async retirement\n";
}
