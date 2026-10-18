// Actual exported C/C++ refusal paths; never initializes GPU/renderer/capture.
#include "../../include/minigpu.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
extern "C" int completion_ticket_c_abi(void);
static void require(bool value, const char *why) {
  if (!value) {std::cerr << "FAIL " << why << '\n'; std::exit(1);}
}
static unsigned calls = 0;
static void callback(int result, void *userdata) {
  require(result == MGPU_CONTEXT_LEASE_INVALID && userdata == &calls, "actual inline refusal");
  MGPUCompletionStats stats{};
  require(mgpuCompletionTicketGetStats(&stats, sizeof(stats)) == 0 && stats.live_tickets == 1,
    "callback reenters native diagnostics outside locks");
  ++calls;
}
static void finish(MGPUCompletionTicket *ticket, uint32_t kind) {
  MGPUCompletionInfo i{};
  require(ticket && mgpuCompletionTicketGetInfo(ticket, &i, sizeof(i)) == 0 &&
    i.phase == MGPU_COMPLETION_COMPLETE && i.kind == kind && i.ticket_id && !i.flags, "real completed identity");
  MGPUCompletionReceipt r{};
  require(mgpuCompletionTicketCollect(ticket, &r, sizeof(r)) == 0 && r.kind == kind &&
    r.ticket_id == i.ticket_id && r.result == -1 && !r.flags && !r.reserved, "real return receipt");
  require(mgpuCompletionTicketRelease(ticket) == 0, "real exclusive release");
}
int main() {
  const auto references = mgpuContextRefCount();
  MGPUContextLifetimeStats before{}, after{};
  require(mgpuContextGetLifetimeStats(&before, sizeof(before)) == 0, "cold context counts");
  require(completion_ticket_c_abi(), "C layouts/all actual exports");
  MGPUCompletionTicket *ticket = nullptr;
  require(mgpuPrivateOutputWriteTrackedAsync(nullptr, nullptr, callback, &calls, &ticket) == 0, "tracked write refusal");
  finish(ticket, MGPU_COMPLETION_WRITE_OUTPUT);
  require(mgpuComputeShaderUseRetireTrackedAsync(nullptr, callback, &calls, &ticket) == 0, "tracked use refusal");
  finish(ticket, MGPU_COMPLETION_RETIRE_USE);
  require(mgpuImportedFrameRetireTrackedAsync(nullptr, callback, &calls, &ticket) == 0, "tracked frame refusal");
  finish(ticket, MGPU_COMPLETION_RETIRE_FRAME);
  require(mgpuPrivateOutputRetireTrackedAsync(nullptr, callback, &calls, &ticket) == 0, "tracked output refusal");
  finish(ticket, MGPU_COMPLETION_RETIRE_OUTPUT);
  require(mgpuPrivateOutputWriterRetireTrackedAsync(nullptr, callback, &calls, &ticket) == 0, "tracked writer refusal");
  finish(ticket, MGPU_COMPLETION_RETIRE_WRITER);
  require(calls == 5, "exactly five void-operation callbacks");
  ticket = reinterpret_cast<MGPUCompletionTicket *>(uintptr_t(1));
  require(mgpuComputeShaderDestroyTrackedAsync(nullptr, nullptr, callback, &calls, &ticket) == -1 && !ticket &&
    mgpuBufferDestroyTrackedAsync(nullptr, nullptr, callback, &calls, &ticket) == -1 && !ticket && calls == 5,
    "checked refusal consumes no pointer, callback or ticket");
  auto *leaseSentinel = reinterpret_cast<MGPUContextLease *>(uintptr_t(1));
  auto *bufferSentinel = reinterpret_cast<MGPUBuffer *>(uintptr_t(2));
  require(mgpuBufferDestroyTrackedAsync(leaseSentinel, nullptr, callback, &calls, &ticket) == -1 && !ticket &&
    mgpuBufferDestroyTrackedAsync(nullptr, bufferSentinel, callback, &calls, &ticket) == -1 && !ticket &&
    mgpuBufferDestroyTrackedAsync(leaseSentinel, bufferSentinel, nullptr, &calls, &ticket) == -1 && !ticket && calls == 5,
    "argument validation precedes lease/owner lookup and callback acquisition");
  MGPUCompletionReceipt r{};
  require(mgpuCompletionTicketCollect(nullptr, &r, sizeof(r)) == -1 && !r.ticket_id &&
    mgpuCompletionTicketRelease(nullptr) == -1, "missing ticket refuses");
  MGPUCompletionStats stats{};
  require(mgpuCompletionTicketGetStats(&stats, sizeof(stats)) == 0 && stats.live_tickets == 0 &&
    stats.max_tickets == 128 && !stats.quarantined_tickets && !stats.released_with_native_users, "no leaked cold tickets");
  require(mgpuContextGetLifetimeStats(&after, sizeof(after)) == 0 &&
    std::memcmp(&before, &after, sizeof(before)) == 0 && references == mgpuContextRefCount(), "no context/GPU initialization");
  std::cout << "PASS actual completion-ticket C/C++ ABI and cold refusal/return custody; no GPU/engine/capture\n";
}
