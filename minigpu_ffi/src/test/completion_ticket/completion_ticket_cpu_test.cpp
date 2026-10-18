#include "../capture_cpu_queue.h"
// Production completion owner with seven compile-time backend substitutes.
// No renderer/context, native image, window or capture is constructed.
#include "../../include/minigpu_completion_ticket.h"
#include "../../include/minigpu_capture_cleanup_adapter.h"
#include "../../include/minigpu_capture_work_owner.h"
#include "../../include/minigpu_capture_recipe_owner.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

extern "C" int completion_ticket_c_abi(void);
static void require(bool value, const char *why) {
  if (!value) {std::cerr << "FAIL " << why << '\n'; std::exit(1);}
}
namespace mgpu::source_scope {
std::array<std::atomic<unsigned>, metadataKinds> metadataAttempts{}, metadataFrees{};
std::atomic<int> failMetadataKind{-1};
std::function<void(MetadataKind)> beforeMetadataAllocate, beforeMetadataFree;
void testMetadataBeforeAllocate(MetadataKind kind, size_t, size_t) {
  ++metadataAttempts[static_cast<size_t>(kind)];
  if (beforeMetadataAllocate) beforeMetadataAllocate(kind);
  int expected = static_cast<int>(kind);
  if (failMetadataKind.compare_exchange_strong(expected, -1)) throw std::bad_alloc();
}
void testMetadataBeforeDeallocate(MetadataKind kind) noexcept {
  ++metadataFrees[static_cast<size_t>(kind)];
  if (beforeMetadataFree) beforeMetadataFree(kind);
}
}
struct Gate {
  std::mutex gate;
  std::condition_variable changed;
  bool entered = false, resumed = false;
  void hold() {
    std::unique_lock<std::mutex> lock(gate); entered = true; changed.notify_all();
    require(changed.wait_for(lock, std::chrono::seconds(5), [&] {return resumed;}), "held call timed out");
  }
  void observe() {
    std::unique_lock<std::mutex> lock(gate);
    require(changed.wait_for(lock, std::chrono::seconds(5), [&] {return entered;}), "call not entered");
  }
  void resume() {std::lock_guard<std::mutex> lock(gate); resumed = true; changed.notify_all();}
};
enum class Mode {Deferred, Inline, Refuse, RefuseRacing, Failure, Duplicate, Throw};
struct Pending {MGPUQueueCompletionCallback callback; void *userdata;};
static Mode mode = Mode::Deferred;
static std::vector<Pending> pending;
static std::atomic<unsigned> backendCalls{0};
static std::atomic<bool> pauseTail{false};
static std::atomic<bool> pauseAdmission{false};
static Gate *tailGate = nullptr, *returnGate = nullptr;
static Gate *admissionGate = nullptr;
static std::thread refusalWorker;
// Native lookup is substituted, but entrypoint argument validation remains real.
static auto *const fakeLease = reinterpret_cast<MGPUContextLease *>(uintptr_t(1));
static auto *const fakeBuffer = reinterpret_cast<MGPUBuffer *>(uintptr_t(2));
static auto *const fakeShader = reinterpret_cast<MGPUComputeShader *>(uintptr_t(3));
namespace mgpu::completion {
void testProxyTail() {if (pauseTail.exchange(false)) tailGate->hold();}
void testCallbackAdmitted() {if (pauseAdmission.exchange(false)) admissionGate->hold();}
}
static int backend(MGPUQueueCompletionCallback cb, void *ud) {
  ++backendCalls;
  if (mode == Mode::Refuse) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  if (mode == Mode::RefuseRacing) {
    refusalWorker = std::thread([=] {cb(MGPU_CONTEXT_LEASE_OK, ud);});
    admissionGate->observe();
    return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  }
  if (mode == Mode::Throw) throw std::runtime_error("unknown backend acceptance");
  if (mode == Mode::Deferred) pending.push_back({cb, ud});
  else {
    cb(mode == Mode::Failure ? MGPU_CONTEXT_LEASE_INTERNAL : MGPU_CONTEXT_LEASE_OK, ud);
    if (mode == Mode::Duplicate) cb(MGPU_CONTEXT_LEASE_OK, ud);
    if (returnGate) returnGate->hold();
  }
  return MGPU_CONTEXT_LEASE_OK;
}
extern "C" void mgpuPrivateOutputWriteAsync(MGPUPrivateOutputWriter *, MGPUPrivateOutput *, MGPUQueueCompletionCallback cb, void *ud) {backend(cb, ud);}
extern "C" void mgpuComputeShaderUseRetireAsync(MGPUComputeShaderUse *, MGPUQueueCompletionCallback cb, void *ud) {backend(cb, ud);}
extern "C" void mgpuImportedFrameRetireAsync(MGPUImportedFrame *, MGPUQueueCompletionCallback cb, void *ud) {backend(cb, ud);}
extern "C" void mgpuPrivateOutputRetireAsync(MGPUPrivateOutput *, MGPUQueueCompletionCallback cb, void *ud) {backend(cb, ud);}
extern "C" void mgpuPrivateOutputWriterRetireAsync(MGPUPrivateOutputWriter *, MGPUQueueCompletionCallback cb, void *ud) {backend(cb, ud);}
extern "C" int mgpuComputeShaderDestroyCheckedAsync(MGPUContextLease *, MGPUComputeShader *, MGPUResourceCleanupCallback cb, void *ud) {return backend(cb, ud);}
extern "C" int mgpuBufferDestroyCheckedAsync(MGPUContextLease *, MGPUBuffer *, MGPUResourceCleanupCallback cb, void *ud) {return backend(cb, ud);}

// The production prepared cleanup task, family and ticket owners are real.
// Only original-pin lookup, native release and worker queue are substituted.
static mgpu::source_scope::SourceDomain cleanupSource;
struct CpuQueue {
  bool reject = false, release = true, fault = false;
  std::vector<std::function<void()>> tasks;
};
static CpuQueue cleanupQueue;
static unsigned outputJobs = 0;
static bool reserveOutputJob() { ++outputJobs; return true; }
static bool refundOutputJob() noexcept {
  mgpu::source_scope::Snapshot s;
  if (!outputJobs || cleanupSource.snapshot(s) != mgpu::source_scope::Result::Ok ||
      s.operationFamilies != 1) return false;
  --outputJobs; return true;
}
static int enqueueCleanup(void *context, mgpu::task_queue::Prepared &task) noexcept {
  auto &queue = *static_cast<CpuQueue *>(context);
  return queue.reject ? MGPU_CONTEXT_LEASE_INTERNAL
      : mgpu::capture_cpu_test::enqueueWitness(queue.tasks, task);
}
struct PinProbe {
  mgpu::source_scope::SourceDomain source;
  ~PinProbe() {
    mgpu::source_scope::Snapshot s;
    require(source.snapshot(s) == mgpu::source_scope::Result::Ok &&
      s.operationFamilies == 1 && s.continuations[0] == 1,
      "pin/release members die before primary-task and family refund");
  }
};
namespace mgpu::capture_cleanup {
int prepareWriteFamily(MGPUPrivateOutputWriter *, MGPUPrivateOutput *, Family &out) noexcept {
  return nativeResult(Family::beginWork(cleanupSource, out));
}
int prepareWriterFamily(MGPUPrivateOutputWriter *, Family &out) noexcept {
  return nativeResult(Family::begin(cleanupSource, out));
}
int prepareOutputFamily(MGPUPrivateOutput *, Family &out) noexcept {
  return nativeResult(Family::begin(cleanupSource, out));
}
static void outputScoped(bool write, MGPUQueueCompletionCallback cb, void *ud, const Family &family) {
  using namespace capture_work;
  if (!family) { backend(cb, ud); return; }
  require(family.matches(cleanupSource) && family.cleanup() != write,
    "output operation inherits exact Work/Cleanup family");
  std::shared_ptr<GlobalTaskCustody> global;
  require(GlobalTaskCustody::create(family, reserveOutputJob, refundOutputJob, global) ==
    source_scope::Result::Ok, "one global output job before native work");
  std::shared_ptr<PreparedWait> wait;
  if (write) require(PreparedWait::create(family, wait) == 0 && wait->prepare(family, [](WaitSignal &signal) {
    signal.result = 0; signal.completed = true; return Outcome::done();
  }, [=](int result) { cb(result, ud); }, global) == 0, "write wait shares family/global job");
  PreparedWork job;
  require(job.reserve(family, source_scope::ContinuationKind::PrimaryTask, global) == 0 &&
    job.configure([wait, write] {
      if (!write) return Outcome::done();
      return wait->enqueue(enqueueCleanup, &cleanupQueue) == 0 ? Outcome::done() : Outcome::uncertain();
    }, [=](int result) { if (!write || result != 0) cb(result, ud); }) == 0,
    "prepared output primary task");
  const auto queued = job.enqueue(enqueueCleanup, &cleanupQueue);
  if (queued != 0) cb(queued, ud);
}
void writeOutput(MGPUPrivateOutputWriter *, MGPUPrivateOutput *, MGPUQueueCompletionCallback cb,
    void *ud, const Family &family) { outputScoped(true, cb, ud, family); }
void retireWriter(MGPUPrivateOutputWriter *, MGPUQueueCompletionCallback cb,
    void *ud, const Family &family) { outputScoped(false, cb, ud, family); }
void retireOutput(MGPUPrivateOutput *, MGPUQueueCompletionCallback cb,
    void *ud, const Family &family) { outputScoped(false, cb, ud, family); }
int prepareUseFamily(MGPUComputeShaderUse *, Family &out) noexcept {
  return nativeResult(Family::begin(cleanupSource, out));
}
int prepareFrameFamily(MGPUImportedFrame *, Family &out) noexcept {
  return nativeResult(Family::begin(cleanupSource, out));
}
static void retireScoped(bool primary, MGPUQueueCompletionCallback cb, void *ud, const Family &family) {
  using namespace capture_work;
  if (!family) { backend(cb, ud); return; }
  require(family.cleanup() && family.matches(cleanupSource), "retirement inherits exact Cleanup family");
  std::shared_ptr<PreparedWait> wait;
  require(PreparedWait::create(family, wait) == 0 && wait->prepare(family, [](WaitSignal &signal) {
    signal.result = 0; signal.completed = true; return Outcome::done();
  }, [=](int result) { cb(result, ud); }) == 0, "production fence/wait reserve in ticket family");
  if (!primary) { require(wait->enqueue(enqueueCleanup, &cleanupQueue) == 0, "standalone frame fence"); return; }
  PreparedWork job;
  require(job.reserve(family) == 0 && job.configure([wait] {
    return wait->enqueue(enqueueCleanup, &cleanupQueue) == 0 ? Outcome::done() : Outcome::uncertain();
  }, [=](int result) { if (result != 0) cb(result, ud); }) == 0 &&
    job.enqueue(enqueueCleanup, &cleanupQueue) == 0, "production use primary and nested fence in ONE family");
}
void retireUse(MGPUComputeShaderUse *, MGPUQueueCompletionCallback cb, void *ud, const Family &family) {
  retireScoped(true, cb, ud, family);
}
void retireFrame(MGPUImportedFrame *, MGPUQueueCompletionCallback cb, void *ud, const Family &family) {
  retireScoped(false, cb, ud, family);
}
int prepareFamily(MGPUContextLease *, Family &out) noexcept {
  return nativeResult(Family::begin(cleanupSource, out));
}
static int destroyScoped(MGPUResourceCleanupCallback callback, void *userdata, const Family &family) {
  if (!family.matches(cleanupSource)) return MGPU_CONTEXT_LEASE_INVALID;
  if (!family || cleanupQueue.fault) return backend(callback, userdata);
  PreparedTask job;
  auto pin = std::make_shared<PinProbe>(); pin->source = cleanupSource;
  const bool released = cleanupQueue.release;
  const auto ready = job.prepare(family, [pin, released] { return released; }, callback, userdata);
  pin.reset();
  if (ready != 0) return ready;
  return job.enqueue(enqueueCleanup, &cleanupQueue);
}
int destroyShader(MGPUContextLease *, MGPUComputeShader *, MGPUResourceCleanupCallback cb,
                  void *ud, const Family &family) { return destroyScoped(cb, ud, family); }
int destroyBuffer(MGPUContextLease *, MGPUBuffer *, MGPUResourceCleanupCallback cb,
                  void *ud, const Family &family) { return destroyScoped(cb, ud, family); }
} // namespace mgpu::capture_cleanup

struct Caller {
  std::atomic<unsigned> calls{0};
  MGPUCompletionTicket *ticket = nullptr;
  Gate *gate = nullptr;
  bool throws = false;
};
static void callback(int, void *ud) {
  auto &caller = *static_cast<Caller *>(ud);
  require(++caller.calls == 1, "one SDK callback only");
  if (caller.ticket) {
    MGPUCompletionReceipt r{};
    require(mgpuCompletionTicketCollect(caller.ticket, &r, sizeof(r)) == MGPU_CONTEXT_LEASE_WOULD_BLOCK && !r.ticket_id,
      "callback entry cannot collect its return");
    require(mgpuCompletionTicketRelease(caller.ticket) == MGPU_CONTEXT_LEASE_WOULD_BLOCK,
      "callback cannot release itself");
  }
  if (caller.gate) caller.gate->hold();
  if (caller.throws) throw std::runtime_error("violated caller callback contract");
}
static MGPUCompletionStats stats() {
  MGPUCompletionStats s{};
  require(mgpuCompletionTicketGetStats(&s, sizeof(s)) == 0 && s.max_tickets == 128, "bounded stats"); return s;
}
static MGPUCompletionReceipt collect(MGPUCompletionTicket *ticket, uint32_t kind, int result, uint32_t flags = 0) {
  MGPUCompletionInfo info{};
  require(mgpuCompletionTicketGetInfo(ticket, &info, sizeof(info)) == 0, "ticket info");
  MGPUCompletionReceipt r{};
  require(mgpuCompletionTicketCollect(ticket, &r, sizeof(r)) == 0 && r.struct_size == 32 && r.version == 1 &&
    r.ticket_id == info.ticket_id && r.kind == kind && r.result == result && r.flags == flags && !r.reserved,
    "exact returned-callback receipt");
  MGPUCompletionReceipt again{};
  require(mgpuCompletionTicketCollect(ticket, &again, sizeof(again)) == MGPU_CONTEXT_LEASE_RETIRED && !again.ticket_id,
    "no double transfer"); return r;
}
static void scopedCleanupChecks() {
  using namespace mgpu::source_scope;
  using mgpu::capture_cleanup::Family;
  auto makeRoot = [] {
    SourceRootOwner root;
    require(SourceRoot::create({10, 9, 1, 0}, root) == Result::Ok, "source root");
    return root;
  };
  auto counts = [](uint32_t primary, uint32_t tickets, uint32_t families) {
    Snapshot s;
    require(cleanupSource.snapshot(s) == Result::Ok &&
      s.continuations[0] == primary && s.continuations[1] == 0 && s.continuations[2] == 0 &&
      s.completionTickets == tickets && s.operationFamilies == families,
      "exact cleanup primary/ticket/family counts without invented fence/wait");
  };
  auto close = [](SourceRootOwner &root) {
    JoinReceipt receipt;
    require(root->tryJoin(receipt) == Result::Ok && root->release(receipt) == Result::Ok,
      "known final cleanup joins original source");
    cleanupSource = {}; root.reset();
  };

  // Ticket disappears first; a retained real worker-closure copy holds task
  // custody and its pin until the actual final copy and members are destroyed.
  auto root = makeRoot(); cleanupSource = root->domain();
  require(root->seal() == Result::Ok, "seal before tracked cleanup");
  cleanupQueue = {};
  Caller held;
  require(mgpuBufferDestroyTrackedAsync(fakeLease, fakeBuffer, callback, &held, &held.ticket) == 0,
    "tracked scoped cleanup after Seal");
  counts(1, 1, 1);
  Family nested;
  require(Family::begin(cleanupSource, nested) == Result::WouldBlock, "reentrant family refuses");
  auto retainedTask = cleanupQueue.tasks.front();
  Gate proxy; tailGate = &proxy; pauseTail = true;
  std::thread taskWorker([&] { cleanupQueue.tasks.front()(); });
  proxy.observe();
  collect(held.ticket, MGPU_COMPLETION_DESTROY_BUFFER, 0, MGPU_COMPLETION_RESOURCE_CONSUMED);
  require(mgpuCompletionTicketRelease(held.ticket) == 0 && stats().released_with_native_users == 1,
    "public ticket release retains counted native proxy tail");
  counts(1, 1, 1);
  proxy.resume(); taskWorker.join(); tailGate = nullptr;
  counts(1, 0, 1);
  cleanupQueue.tasks.clear(); counts(1, 0, 1);
  retainedTask = {}; counts(0, 0, 0);
  close(root);

  // Task disappears first; public ticket alone retains the family. Poison
  // permits necessary known cleanup, but never a fabricated joined receipt.
  root = makeRoot(); cleanupSource = root->domain(); cleanupSource.quarantine();
  cleanupQueue = {};
  Caller poisoned;
  require(mgpuComputeShaderDestroyTrackedAsync(fakeLease, fakeShader, callback, &poisoned, &poisoned.ticket) == 0,
    "tracked scoped cleanup after source Poison");
  cleanupQueue.tasks.front()(); cleanupQueue.tasks.clear(); counts(0, 1, 1);
  collect(poisoned.ticket, MGPU_COMPLETION_DESTROY_SHADER, 0, MGPU_COMPLETION_RESOURCE_CONSUMED);
  require(mgpuCompletionTicketRelease(poisoned.ticket) == 0, "known ticket cleanup on poisoned source");
  counts(0, 0, 0);
  JoinReceipt refused;
  require(root->tryJoin(refused) == Result::Poisoned && !refused, "Poison never turns into join");
  cleanupSource = {}; root.reset();

  // Both direct-style and tracked admission use the same cleanup family/task.
  root = makeRoot(); cleanupSource = root->domain(); root->seal(); cleanupQueue = {};
  cleanupQueue.reject = true;
  Caller rejected;
  MGPUCompletionTicket *ticket = reinterpret_cast<MGPUCompletionTicket *>(uintptr_t(1));
  require(mgpuBufferDestroyTrackedAsync(fakeLease, fakeBuffer, callback, &rejected, &ticket) == -5 &&
    !ticket && !rejected.calls && cleanupQueue.tasks.empty(), "tracked enqueue rejection consumes nothing");
  counts(0, 0, 0);
  {
    Family direct;
    require(Family::begin(cleanupSource, direct) == Result::Ok, "direct cleanup after Seal");
    require(mgpu::capture_cleanup::destroyBuffer(nullptr, nullptr, callback, &rejected, direct) == -5,
      "direct enqueue refusal is known and callback-free");
  }
  counts(0, 0, 0); close(root);

  // A typed family cannot authorize another original source's cleanup.
  root = makeRoot(); auto other = makeRoot(); cleanupSource = root->domain();
  {
    Family foreign;
    require(Family::begin(other->domain(), foreign) == Result::Ok, "foreign family");
    require(mgpu::capture_cleanup::destroyShader(nullptr, nullptr, callback, &rejected, foreign) == -1,
      "source-family substitution refuses before queueing");
  }
  counts(0, 0, 0); root->seal(); close(root);
  other->seal(); JoinReceipt joined;
  require(other->tryJoin(joined) == Result::Ok && other->release(joined) == Result::Ok, "foreign root clean");

  // Unknown native cleanup keeps the preallocated actual job/pin anchor.
  root = makeRoot(); cleanupSource = root->domain(); cleanupQueue = {}; cleanupQueue.release = false;
  Caller failed;
  require(mgpuBufferDestroyTrackedAsync(fakeLease, fakeBuffer, callback, &failed, &failed.ticket) == 0,
    "accepted scoped cleanup failure");
  cleanupQueue.tasks.front()(); cleanupQueue.tasks.clear(); counts(1, 1, 1);
  collect(failed.ticket, MGPU_COMPLETION_DESTROY_BUFFER, -5, MGPU_COMPLETION_RESOURCE_CONSUMED);
  require(mgpuCompletionTicketRelease(failed.ticket) == 0, "failure receipt releases only CPU ticket");
  counts(1, 0, 1);
  require(root->tryJoin(refused) == Result::Poisoned, "unknown native job remains poisoned/retained");
  cleanupSource = {}; root.reset();

  // A callback admitted during contradictory refusal poisons scoped ticket
  // custody as well as the existing completion registry and actual userdata.
  root = makeRoot(); cleanupSource = root->domain(); cleanupQueue = {}; cleanupQueue.fault = true;
  Gate admission; admissionGate = &admission; pauseAdmission = true; mode = Mode::RefuseRacing;
  auto *racing = new Caller;
  require(mgpuBufferDestroyTrackedAsync(fakeLease, fakeBuffer, callback, racing, &racing->ticket) == -7 && racing->ticket,
    "scoped contradictory refusal retains ticket and family");
  admission.resume(); refusalWorker.join(); admissionGate = nullptr;
  counts(0, 1, 1);
  require(mgpuCompletionTicketRelease(racing->ticket) == -7 && root->tryJoin(refused) == Result::Poisoned,
    "uncertain scoped ticket cannot refund original custody");
  cleanupSource = {}; root.reset(); cleanupQueue = {};
}
static void scopedRetirementChecks() {
  using namespace mgpu::source_scope;
  for (bool use : {false, true}) {
    SourceRootOwner root;
    require(SourceRoot::create({10, 9, 1, 0}, root) == Result::Ok, "retirement root");
    cleanupSource = root->domain(); root->seal(); cleanupQueue = {};
    Caller caller;
    auto primary = std::function<void()>{};
    require((use ? mgpuComputeShaderUseRetireTrackedAsync(nullptr, callback, &caller, &caller.ticket)
                 : mgpuImportedFrameRetireTrackedAsync(nullptr, callback, &caller, &caller.ticket)) == 0,
      "tracked source retirement after Seal");
    auto check = [&](uint32_t tasks, uint32_t fences, uint32_t waits, uint32_t tickets) {
      Snapshot s; require(cleanupSource.snapshot(s) == Result::Ok && s.operationFamilies == 1 &&
        s.continuations[0] == tasks && s.continuations[1] == fences && s.continuations[2] == waits &&
        s.completionTickets == tickets, "exact single-family tracked retirement counts");
    };
    check(use ? 1u : 0u, 1, 1, 1);
    if (use) {
      primary = std::move(cleanupQueue.tasks.front()); cleanupQueue.tasks.clear();
      primary(); // Nested enqueue uses its prepared family, no public recursion.
      check(1, 1, 1, 1);
    }
    auto fence = std::move(cleanupQueue.tasks.front()); cleanupQueue.tasks.clear();
    fence();
    collect(caller.ticket, use ? MGPU_COMPLETION_RETIRE_USE : MGPU_COMPLETION_RETIRE_FRAME, 0);
    if (use) {
      // Ticket finishes first, both real queued closure tails stay charged.
      require(mgpuCompletionTicketRelease(caller.ticket) == 0, "ticket before use closure tails");
      check(1, 1, 1, 0);
      fence = {}; check(1, 1, 1, 0); // Primary still retains prepared wait holder.
      primary = {};
    } else {
      // Job finishes first, only the actual public ticket retains its family.
      fence = {}; check(0, 0, 0, 1);
      require(mgpuCompletionTicketRelease(caller.ticket) == 0, "frame ticket after job tails");
    }
    Snapshot done; require(cleanupSource.snapshot(done) == Result::Ok && !done.operationFamilies &&
      !done.completionTickets && !done.continuations[0] && !done.continuations[1] && !done.continuations[2],
      "last native/tracked retirement tails disposed");
    JoinReceipt receipt;
    require(root->tryJoin(receipt) == Result::Ok && root->release(receipt) == Result::Ok, "retirement source joins");
    cleanupSource = {}; cleanupQueue = {};
  }
}
static void scopedOutputChecks() {
  using namespace mgpu::source_scope;
  for (unsigned operation = 0; operation != 3; ++operation) for (bool ticketFirst : {false, true}) {
    const bool write = operation == 0;
    const uint32_t kind = write ? MGPU_COMPLETION_WRITE_OUTPUT : operation == 1
      ? MGPU_COMPLETION_RETIRE_OUTPUT : MGPU_COMPLETION_RETIRE_WRITER;
    SourceRootOwner root;
    require(SourceRoot::create({10, 9, 1, 0}, root) == Result::Ok, "output source root");
    cleanupSource = root->domain(); cleanupQueue = {};
    if (!write) root->seal();
    Caller caller;
    const auto result = write
      ? mgpuPrivateOutputWriteTrackedAsync(nullptr, nullptr, callback, &caller, &caller.ticket)
      : operation == 1 ? mgpuPrivateOutputRetireTrackedAsync(nullptr, callback, &caller, &caller.ticket)
                       : mgpuPrivateOutputWriterRetireTrackedAsync(nullptr, callback, &caller, &caller.ticket);
    require(result == 0 && caller.ticket, "tracked output operation");
    if (write) {
      Snapshot peak;
      require(cleanupSource.snapshot(peak) == Result::Ok, "peak metadata snapshot");
      for (size_t i = 0; i < metadataKinds; ++i)
        require(peak.metadataAllocations[i] == (i < operationMetadataKinds ? 1u : 0u),
          "exact thirteen operation records before tracked write; no imported-frame allocation");
    }
    root->seal();
    auto check = [&](uint32_t primary, uint32_t fence, uint32_t wait, uint32_t tickets,
                     uint32_t families, unsigned jobs) {
      Snapshot s;
      require(cleanupSource.snapshot(s) == Result::Ok && s.continuations[0] == primary &&
        s.continuations[1] == fence && s.continuations[2] == wait && s.completionTickets == tickets &&
        s.operationFamilies == families && outputJobs == jobs, "exact output task/ticket/global tails");
    };
    check(1, write ? 1u : 0u, write ? 1u : 0u, 1, 1, 1);
    auto primary = std::move(cleanupQueue.tasks.front()); cleanupQueue.tasks.clear();
    std::function<void()> fence;
    if (write) {
      primary();
      fence = std::move(cleanupQueue.tasks.front()); cleanupQueue.tasks.clear();
    }
    // The actual completion proxy can outlive public ticket release. The
    // primary/fence/wait AND global output job remain until its tail returns.
    Gate proxy; tailGate = &proxy; pauseTail = true;
    std::thread worker([&] { if (write) fence(); else primary(); });
    proxy.observe();
    collect(caller.ticket, kind, 0);
    if (ticketFirst) require(mgpuCompletionTicketRelease(caller.ticket) == 0, "output ticket before task tails");
    check(1, write ? 1u : 0u, write ? 1u : 0u, 1, 1, 1);
    proxy.resume(); worker.join(); tailGate = nullptr;
    check(1, write ? 1u : 0u, write ? 1u : 0u, ticketFirst ? 0u : 1u, 1, 1);
    fence = {}; // Write's primary closure still retains its prepared wait.
    primary = {};
    check(0, 0, 0, ticketFirst ? 0u : 1u, ticketFirst ? 0u : 1u, 0);
    if (!ticketFirst) require(mgpuCompletionTicketRelease(caller.ticket) == 0, "output ticket after task tails");
    check(0, 0, 0, 0, 0, 0);
    JoinReceipt receipt;
    require(root->tryJoin(receipt) == Result::Ok && root->release(receipt) == Result::Ok, "output source joins");
    cleanupSource = {}; cleanupQueue = {};
  }
}
static void taskMetadataChecks() {
  using namespace mgpu::source_scope;
  using namespace mgpu::capture_cleanup;
  using namespace mgpu::capture_work;
  auto snapshot = [] {
    Snapshot value;
    require(cleanupSource.snapshot(value) == Result::Ok, "metadata source snapshot");
    return value;
  };
  auto drained = [&] {
    const auto value = snapshot();
    require(value.phase != Phase::Poisoned && !value.operationFamilies &&
      !value.completionTickets && !outputJobs, "known metadata rollback retained obligations");
    for (auto count : value.metadataAllocations) require(!count, "known rollback retained allocation");
    for (auto count : value.continuations) require(!count, "known rollback retained task");
  };
  auto close = [&](SourceRootOwner &root) {
    drained();
    JoinReceipt receipt;
    require(root->seal() == Result::Ok && root->tryJoin(receipt) == Result::Ok &&
      root->release(receipt) == Result::Ok, "known metadata disposition joins");
    root.reset(); cleanupSource = {};
  };
  // Hooks run at the real allocator boundary, outside the source gate. A
  // reentrant caller must see the original family and already-reserved slot.
  auto verifyAllocation = [&](MetadataKind kind) {
    const auto value = snapshot();
    Family refused;
    require(value.operationFamilies == 1 &&
      value.metadataAllocations[static_cast<size_t>(kind)] == 1 &&
      Family::begin(cleanupSource, refused) == Result::WouldBlock,
      "allocation/free boundary escaped source admission or held its gate");
  };
  beforeMetadataAllocate = verifyAllocation;
  beforeMetadataFree = verifyAllocation;
  for (size_t index = static_cast<size_t>(MetadataKind::GlobalTaskCustody);
       index < operationMetadataKinds; ++index) {
    SourceRootOwner root;
    require(SourceRoot::create({10, 9, 1, 0}, root) == Result::Ok, "failure source");
    cleanupSource = root->domain(); cleanupQueue = {};
    const auto kind = static_cast<MetadataKind>(index);
    failMetadataKind = static_cast<int>(kind);
    if (index >= static_cast<size_t>(MetadataKind::TicketCustody)) {
      // Ticket failures must occur before any original backend invocation.
      Caller caller;
      const auto originalCalls = backendCalls.load();
      require(mgpuPrivateOutputWriteTrackedAsync(nullptr, nullptr, callback, &caller,
        &caller.ticket) == MGPU_CONTEXT_LEASE_INTERNAL && !caller.ticket && !caller.calls &&
        backendCalls == originalCalls && cleanupQueue.tasks.empty(),
        "ticket metadata failure reached original work");
    } else {
      Family family;
      require(Family::beginWork(cleanupSource, family) == Result::Ok, "failure family");
      if (kind == MetadataKind::GlobalTaskCustody) {
        std::shared_ptr<GlobalTaskCustody> global;
        require(GlobalTaskCustody::create(family, reserveOutputJob, refundOutputJob, global) ==
          Result::Internal && !global && !outputJobs, "global allocation failure charged output job");
      } else if (kind == MetadataKind::PrimaryJob || kind == MetadataKind::PrimaryAnchor) {
        PreparedWork work;
        require(work.reserve(family) == MGPU_CONTEXT_LEASE_INTERNAL,
          "primary metadata allocation failure not refused");
      } else {
        std::shared_ptr<PreparedWait> wait;
        const auto created = PreparedWait::create(family, wait);
        if (kind == MetadataKind::PreparedWaitHolder)
          require(created == MGPU_CONTEXT_LEASE_INTERNAL && !wait, "wait-holder allocation failure");
        else require(created == 0 && wait->prepare(family,
          [](WaitSignal &) { return Outcome::done(); }, [](int) {}) == MGPU_CONTEXT_LEASE_INTERNAL,
          "fence/wait metadata allocation failure not refused");
      }
    }
    require(failMetadataKind == -1, "covered factory missed its fault-injected allocation");
    close(root);
  }
  // Weak control blocks keep their own exact slot AND the operation, after
  // payload destruction. Duplicate factories refuse before allocating again.
  for (bool holder : {false, true}) {
    SourceRootOwner root;
    require(SourceRoot::create({10, 9, 1, 0}, root) == Result::Ok, "weak source");
    cleanupSource = root->domain();
    Family family;
    require(Family::beginWork(cleanupSource, family) == Result::Ok, "weak family");
    const auto kind = holder ? MetadataKind::PreparedWaitHolder : MetadataKind::GlobalTaskCustody;
    std::weak_ptr<void> weak;
    if (holder) {
      std::shared_ptr<PreparedWait> wait, duplicate;
      require(PreparedWait::create(family, wait) == 0, "covered wait holder");
      weak = wait; wait.reset();
      const auto attempts = metadataAttempts[static_cast<size_t>(kind)].load();
      require(PreparedWait::create(family, duplicate) == MGPU_CONTEXT_LEASE_WOULD_BLOCK && !duplicate &&
        metadataAttempts[static_cast<size_t>(kind)] == attempts, "weak wait holder reused its live slot");
    } else {
      std::shared_ptr<GlobalTaskCustody> global, duplicate;
      require(GlobalTaskCustody::create(family, reserveOutputJob, refundOutputJob, global) == Result::Ok,
        "covered global custody");
      weak = global; global.reset();
      const auto attempts = metadataAttempts[static_cast<size_t>(kind)].load();
      require(GlobalTaskCustody::create(family, reserveOutputJob, refundOutputJob, duplicate) ==
        Result::WouldBlock && !duplicate && metadataAttempts[static_cast<size_t>(kind)] == attempts,
        "weak global reused its live slot");
    }
    require(root->seal() == Result::Ok, "seal weak metadata");
    family = {};
    JoinReceipt receipt;
    require(weak.expired() && snapshot().metadataAllocations[static_cast<size_t>(kind)] == 1 &&
      snapshot().operationFamilies == 1 && root->tryJoin(receipt) == Result::WouldBlock,
      "weak metadata tail released operation before free");
    weak.reset();
    close(root);
  }
  // The cleanup and work implementations share the same primary envelope.
  {
    SourceRootOwner root;
    require(SourceRoot::create({10, 9, 1, 0}, root) == Result::Ok, "shared primary source");
    cleanupSource = root->domain();
    {
      Family family;
      require(Family::begin(cleanupSource, family) == Result::Ok, "shared primary family");
      PreparedWork work;
      require(work.reserve(family) == 0, "shared primary work");
      const auto attempts = metadataAttempts[static_cast<size_t>(MetadataKind::PrimaryJob)].load();
      PreparedTask cleanup;
      require(cleanup.prepare(family, [] { return true; }, [](int, void *) {}, nullptr) ==
        MGPU_CONTEXT_LEASE_WOULD_BLOCK &&
        metadataAttempts[static_cast<size_t>(MetadataKind::PrimaryJob)] == attempts,
        "cleanup allocated a second primary job");
    }
    close(root);
  }
  // A private-resource Job's CPU member may throw after its lifetime base was
  // moved. Constructor unwind is known unpublished rollback, not uncertainty.
  struct ResourceJob : mgpu::capture_recipe::JobCustody {
    ResourceJob(MetadataAllocation allocation, mgpu::capture_recipe::JobCustody &&held, bool fail)
      : JobCustody(std::move(held), std::move(allocation)) {
      if (fail) throw std::runtime_error("pure CPU member construction failed");
    }
  };
  {
    SourceRootOwner root;
    require(SourceRoot::create({10, 9, 1, 0}, root) == Result::Ok, "private job source");
    cleanupSource = root->domain();
    for (bool fail : {true, false}) {
      {
        mgpu::capture_recipe::JobCustody held;
        require(held.prepareSourceJob(cleanupSource) == Result::Ok, "private job admission");
        MetadataReservations slots;
        require(held.reserveJobMetadata(slots) == Result::Ok, "private job metadata admission");
        bool threw = false;
        try {
          auto job = makeMetadataShared<ResourceJob>(slots, MetadataKind::PrimaryJob, std::move(held), fail);
          require(!fail && job->acceptSourceJob() == Result::Ok, "private job after known rollback");
          job->proveJobFinished();
        } catch (const std::runtime_error &) { threw = true; }
        require(threw == fail, "private job CPU constructor fault");
      }
      drained();
    }
    close(root);
  }
  beforeMetadataAllocate = {}; beforeMetadataFree = {};
  cleanupQueue = {};
}
static void frameMetadataChecks() {
  using namespace mgpu::source_scope;
  using mgpu::capture_recipe::ResourceCustody;
  struct Native : MetadataUniqueRecord {
    Native(MetadataUniqueRecord storage, bool fail) : MetadataUniqueRecord(std::move(storage)) {
      if (fail) throw std::runtime_error("native CPU wrapper constructor refused");
    }
  };
  struct Frame : ResourceCustody {
    Frame(MetadataAllocation allocation, ResourceCustody &&role, MetadataReservations &slots, bool fail)
      : ResourceCustody(std::move(role), std::move(allocation)),
        native(makeMetadataUnique<Native>(slots, MetadataKind::ImportedFrameNative, fail)) {}
    MetadataUnique<Native> native;
  };
  struct Box : MetadataUniqueRecord {
    explicit Box(MetadataUniqueRecord storage) noexcept : MetadataUniqueRecord(std::move(storage)) {}
    std::shared_ptr<Frame> state;
  };
  auto allocate = [](ResourceCustody &&role, MetadataReservations &slots, bool fail) {
    auto box = makeMetadataUnique<Box>(slots, MetadataKind::ImportedFrameBox);
    box->state = makeMetadataShared<Frame>(slots, MetadataKind::ImportedFrameState,
      std::move(role), slots, fail);
    return box;
  };
  auto snapshot = [] {
    Snapshot value;
    require(cleanupSource.snapshot(value) == Result::Ok, "frame metadata snapshot");
    return value;
  };
  auto create = [&] {
    SourceRootOwner root;
    require(SourceRoot::create({10, 9, 1, 0}, root) == Result::Ok, "frame metadata source");
    cleanupSource = root->domain();
    return root;
  };
  auto close = [&](SourceRootOwner &root) {
    auto value = snapshot();
    require(!value.operationFamilies && !value.resources[0], "frame known disposition retained role/operation");
    for (auto count : value.metadataAllocations) require(!count, "frame known disposition retained allocation");
    JoinReceipt receipt;
    require(root->seal() == Result::Ok && root->tryJoin(receipt) == Result::Ok &&
      root->release(receipt) == Result::Ok, "frame known disposition joins");
    root.reset(); cleanupSource = {};
  };
  auto verifyBoundary = [&](MetadataKind kind) {
    if (static_cast<size_t>(kind) < operationMetadataKinds) return;
    require(snapshot().metadataAllocations[static_cast<size_t>(kind)] == 1,
      "resource allocation/free escaped its source permit");
  };
  beforeMetadataAllocate = verifyBoundary;
  beforeMetadataFree = [&](MetadataKind kind) {
    verifyBoundary(kind);
    if (static_cast<size_t>(kind) < operationMetadataKinds) return;
    SourceClaim next;
    const auto status = cleanupSource.reserve(ResourceKind::ImportedFrame, next);
    if (status == Result::Ok) {
      MetadataReservations refused;
      require(next.reserveResourceMetadata(refused) == Result::WouldBlock && !refused.scoped() &&
        next.cancelUnaccepted() == Result::Ok, "resource free admitted replacement allocation");
    } else require(status == Result::WouldBlock || status == Result::Sealed,
      "unexpected resource disposition during free");
  };
  // Fixed claim authority cannot be supplied by an arbitrary source, resource,
  // operation claim, moved handle, accepted handle or reused reservation batch.
  {
    auto root = create();
    {
      SourceClaim empty, other;
      MetadataReservations slots;
      require(empty.reserveResourceMetadata(slots) == Result::Invalid &&
        cleanupSource.reserve(ResourceKind::ScopedPin, other) == Result::Ok &&
        other.reserveResourceMetadata(slots) == Result::Invalid &&
        other.cancelUnaccepted() == Result::Ok, "wrong resource allocation authority");
      SourceOperation operation;
      require(cleanupSource.beginOperation(OperationPurpose::Work, operation) == Result::Ok,
        "resource-kind refusal operation");
      auto view = operation.view();
      require(view.reserveContinuation(ContinuationKind::PrimaryTask, other) == Result::Ok &&
        other.reserveResourceMetadata(slots) == Result::Invalid &&
        view.reserveMetadata({MetadataKind::ImportedFrameState}, slots) == Result::Invalid &&
        other.cancelUnaccepted() == Result::Ok && operation.finish() == Result::Ok,
        "operation authority admitted a resource metadata kind");
      view = {};
      require(cleanupSource.reserve(ResourceKind::ImportedFrame, other) == Result::Ok,
        "exact resource claim");
      SourceClaim moved(std::move(other));
      require(other.reserveResourceMetadata(slots) == Result::Invalid &&
        moved.reserveResourceMetadata(slots) == Result::Ok && slots.scoped() && !slots.view() &&
        moved.reserveResourceMetadata(slots) == Result::Invalid && moved.cancelUnaccepted() == Result::Ok,
        "moved resource authority or batch identity");
      root->seal();
      JoinReceipt blocked;
      require(root->tryJoin(blocked) == Result::WouldBlock, "join ignored reserved resource metadata");
    } // Untaken reservations cancel independently of the finished claim.
    close(root);
  }
  const auto frameEnd = static_cast<size_t>(MetadataKind::PrivateResourceState);
  for (size_t index = operationMetadataKinds; index < frameEnd; ++index) {
    auto root = create();
    {
      ResourceCustody role;
      MetadataReservations slots;
      require(role.reserveSourceRole(cleanupSource, ResourceKind::ImportedFrame) == Result::Ok &&
        role.reserveResourceMetadata(slots) == Result::Ok && slots.scoped() && !slots.view(),
        "resource batch before heap construction");
      for (size_t i = operationMetadataKinds; i < frameEnd; ++i)
        require(snapshot().metadataAllocations[i] == 1, "complete frame batch not prepaid");
      failMetadataKind = static_cast<int>(index);
      bool threw = false;
      try { (void)allocate(std::move(role), slots, false); }
      catch (const std::bad_alloc &) { threw = true; }
      require(threw && failMetadataKind == -1, "resource allocation fault did not reach exact factory");
    }
    close(root);
  }
  {
    auto root = create();
    {
      ResourceCustody role;
      MetadataReservations slots;
      require(role.reserveSourceRole(cleanupSource, ResourceKind::ImportedFrame) == Result::Ok &&
        role.reserveResourceMetadata(slots) == Result::Ok, "throwing resource batch");
      bool threw = false;
      try { (void)allocate(std::move(role), slots, true); }
      catch (const std::runtime_error &) { threw = true; }
      require(threw && snapshot().phase == Phase::Open, "CPU constructor unwind poisoned resource source");
    }
    close(root);
  }
  {
    auto root = create();
    std::weak_ptr<Frame> weak;
    MetadataUnique<Box> box;
    {
      ResourceCustody role;
      MetadataReservations slots;
      require(role.reserveSourceRole(cleanupSource, ResourceKind::ImportedFrame) == Result::Ok &&
        role.reserveResourceMetadata(slots) == Result::Ok, "live resource batch");
      box = allocate(std::move(role), slots, false);
    }
    require(box->state->acceptSourceRole() == Result::Ok, "accept live frame resource");
    MetadataReservations late;
    require(box->state->reserveResourceMetadata(late) == Result::Invalid,
      "accepted resource requested a new allocation batch");
    mgpu::capture_cleanup::Family later;
    require(mgpu::capture_cleanup::Family::beginWork(cleanupSource, later) == Result::Ok,
      "live resource allocation monopolized later public operation");
    later = {};
    require(snapshot().operationFamilies == 0 && snapshot().resources[0] == 1,
      "resource allocator retained a completed operation");
    weak = box->state;
    box->state->proveNativeReleased();
    box.reset();
    auto value = snapshot();
    require(weak.expired() && !value.resources[0] && !value.operationFamilies &&
      value.metadataAllocations[operationMetadataKinds] == 1,
      "weak frame control block was refunded with its payload");
    SourceClaim next;
    MetadataReservations refused;
    require(cleanupSource.reserve(ResourceKind::ImportedFrame, next) == Result::Ok &&
      next.reserveResourceMetadata(refused) == Result::WouldBlock && !refused.scoped() &&
      next.cancelUnaccepted() == Result::Ok, "weak frame allocation reused the covered slot");
    require(root->seal() == Result::Ok, "seal weak frame");
    JoinReceipt blocked;
    require(root->tryJoin(blocked) == Result::WouldBlock, "join ignored weak frame allocation");
    weak.reset();
    close(root);
  }
  for (bool poison : {false, true}) {
    auto root = create();
    SourceClaim reserved;
    MetadataReservations refused;
    require(cleanupSource.reserve(ResourceKind::ImportedFrame, reserved) == Result::Ok,
      "closing source resource claim");
    if (poison) cleanupSource.quarantine();
    else require(root->seal() == Result::Ok, "seal before resource allocation");
    require(reserved.reserveResourceMetadata(refused) == (poison ? Result::Poisoned : Result::Sealed) &&
      !refused.scoped() && reserved.cancelUnaccepted() == Result::Ok,
      "sealed/poisoned source admitted resource allocation");
    if (poison) { root.reset(); cleanupSource = {}; }
    else close(root);
  }
  // Legacy unscoped resources keep the same uncharged path, despite accepting
  // the same storage-aware constructors. No scoped source is substituted.
  {
    ResourceCustody legacy;
    MetadataReservations slots;
    require(legacy.reserveSourceRole({}, ResourceKind::ImportedFrame) == Result::Ok &&
      legacy.reserveResourceMetadata(slots) == Result::Ok && !slots.scoped(), "legacy resource batch");
    (void)allocate(std::move(legacy), slots, false);
  }
  {
    auto root = create();
    ResourceCustody role;
    MetadataReservations slots;
    require(role.reserveSourceRole(cleanupSource, ResourceKind::ImportedFrame) == Result::Ok &&
      role.reserveResourceMetadata(slots) == Result::Ok, "uncertain frame resource batch");
    auto box = allocate(std::move(role), slots, false);
    require(box->state->acceptSourceRole() == Result::Ok, "uncertain accepted frame");
    box->state->quarantineSourceRole();
    SourceClaim refusedClaim;
    require(cleanupSource.reserve(ResourceKind::Writer, refusedClaim) == Result::Poisoned,
      "poisoned source admitted a resource claim");
    JoinReceipt receipt;
    const auto held = snapshot();
    require(held.phase == Phase::Poisoned && held.resources[0] == 1 &&
      held.metadataAllocations[operationMetadataKinds] == 1 && root->tryJoin(receipt) == Result::Poisoned,
      "unknown resource disposition manufactured refund");
    (void)box.release(); // Retain the actual heap owner, as native registry quarantine does.
    root.reset(); cleanupSource = {};
  }
  beforeMetadataAllocate = {}; beforeMetadataFree = {};
}
static void remainingResourceMetadataChecks() {
  using namespace mgpu::source_scope;
  using mgpu::capture_recipe::ResourceCustody;
  struct State : ResourceCustody {
    State(MetadataAllocation allocation, ResourceCustody &&role, bool fail)
        : ResourceCustody(std::move(role), std::move(allocation)) {
      if (fail) throw std::runtime_error("resource State CPU construction");
    }
  };
  struct alignas(64) Box : MetadataUniqueRecord {
    Box(MetadataUniqueRecord storage, std::shared_ptr<State> value, bool fail)
        : MetadataUniqueRecord(std::move(storage)), state(std::move(value)) {
      if (fail) throw std::runtime_error("resource wrapper CPU construction");
    }
    std::shared_ptr<State> state;
  };
  struct Shape { ResourceKind role; MetadataKind state, box, anchor; unsigned cap; };
  const Shape shapes[] = {
    {ResourceKind::RecipeShader, MetadataKind::PrivateResourceState, MetadataKind::ShaderCpu, MetadataKind::Count, 1},
    {ResourceKind::PackedBuffer, MetadataKind::PrivateResourceState, MetadataKind::BufferCpu, MetadataKind::Count, 1},
    {ResourceKind::Parameters, MetadataKind::PrivateResourceState, MetadataKind::BufferCpu, MetadataKind::Count, 1},
    {ResourceKind::ExclusiveUse, MetadataKind::ExclusiveUseState, MetadataKind::ExclusiveUseBox, MetadataKind::Count, 1},
    {ResourceKind::Writer, MetadataKind::WriterState, MetadataKind::WriterBox, MetadataKind::Count, 1},
    {ResourceKind::Output, MetadataKind::OutputState, MetadataKind::OutputBox, MetadataKind::Count, 2},
    {ResourceKind::PreviewLoan, MetadataKind::LoanState, MetadataKind::LoanBox, MetadataKind::LoanAnchor, 2},
    {ResourceKind::DescriptorBorrow, MetadataKind::BorrowState, MetadataKind::BorrowBox, MetadataKind::BorrowAnchor, 8},
  };
  auto snapshot = [] {
    Snapshot value;
    require(cleanupSource.snapshot(value) == Result::Ok, "resource allocation snapshot");
    return value;
  };
  auto create = [&] {
    SourceRootOwner root;
    require(SourceRoot::create({10, 9, 1, 0}, root) == Result::Ok, "resource allocation source");
    cleanupSource = root->domain(); return root;
  };
  auto close = [&](SourceRootOwner &root) {
    const auto value = snapshot();
    for (auto count : value.resources) require(!count, "resource role survived known CPU disposal");
    for (auto count : value.metadataAllocations) require(!count, "resource allocation survived actual free");
    JoinReceipt receipt;
    require(root->seal() == Result::Ok && root->tryJoin(receipt) == Result::Ok &&
      root->release(receipt) == Result::Ok, "all resource allocation tails join");
    root.reset(); cleanupSource = {};
  };
  struct Prepared {
    MetadataUnique<Box> box;
    MetadataUnique<MetadataAnchor<State>> anchor;
  };
  auto allocate = [&](const Shape &shape, int constructorFailure = 0) {
    ResourceCustody role;
    require(role.reserveSourceRole(cleanupSource, shape.role) == Result::Ok, "exact resource role before allocation");
    MetadataReservations slots;
    require(role.reserveResourceMetadata(slots) == Result::Ok && slots.scoped() && !slots.view(),
      "fixed resource batch without operation lifetime");
    MetadataReservations duplicate;
    require(role.reserveResourceMetadata(duplicate) == Result::Invalid && !duplicate.scoped(),
      "same unaccepted claim issued two batches using spare capacity");
    auto state = makeMetadataShared<State>(slots, shape.state, std::move(role), constructorFailure == 1);
    Prepared result;
    result.box = makeMetadataUnique<Box>(slots, shape.box, state, constructorFailure == 2);
    if (shape.anchor != MetadataKind::Count)
      result.anchor = makeMetadataUnique<MetadataAnchor<State>>(slots, shape.anchor, state);
    return result;
  };
  beforeMetadataAllocate = [&](MetadataKind kind) {
    require(snapshot().metadataAllocations[static_cast<size_t>(kind)] > 0,
      "resource heap call precedes its exact prepaid batch");
  };
  beforeMetadataFree = [&](MetadataKind kind) {
    require(snapshot().metadataAllocations[static_cast<size_t>(kind)] > 0,
      "resource deallocation refunded before actual free");
  };
  for (const auto &shape : shapes) {
    for (auto kind : {shape.state, shape.box, shape.anchor}) {
      if (kind == MetadataKind::Count) continue;
      auto root = create();
      failMetadataKind = static_cast<int>(kind);
      bool threw = false;
      try { (void)allocate(shape); } catch (const std::bad_alloc &) { threw = true; }
      require(threw && failMetadataKind == -1 && snapshot().phase == Phase::Open,
        "resource allocation failure did not roll back exactly");
      close(root);
    }
    for (int failure : {1, 2}) {
      auto root = create(); bool threw = false;
      try { (void)allocate(shape, failure); } catch (const std::runtime_error &) { threw = true; }
      require(threw && snapshot().phase == Phase::Open, "resource constructor failure poisoned source");
      close(root);
    }
  }
  auto root = create();
  std::vector<Prepared> live;
  std::vector<std::weak_ptr<State>> weak;
  for (const auto &shape : shapes) {
    for (unsigned i = 0; i < shape.cap; ++i) {
      live.push_back(allocate(shape));
      weak.push_back(live.back().box->state);
      require(live.back().box->state->acceptSourceRole() == Result::Ok, "resource native acceptance seam");
    }
  }
  auto value = snapshot();
  unsigned total = 0;
  for (size_t i = static_cast<size_t>(MetadataKind::PrivateResourceState); i < metadataKinds; ++i) {
    require(value.metadataAllocations[i] == metadataCapacity(static_cast<MetadataKind>(i)),
      "exact shared resource allocation capacity");
    total += value.metadataAllocations[i];
  }
  require(total == 44 && !value.operationFamilies, "remaining resource inventory or operation coupling");
  mgpu::capture_cleanup::Family later;
  require(mgpu::capture_cleanup::Family::beginWork(cleanupSource, later) == Result::Ok,
    "live resource CPU objects blocked unrelated operation");
  later = {};
  for (auto &item : live) {
    item.box->state->proveNativeReleased(); // Native action is substituted; this checks CPU custody only.
    item.anchor.reset();
    // The explicit deleter reads the live record before payload destruction.
    // Class-specific destroying delete is deliberately absent: failed legacy
    // construction must never invoke a completed-object destructor (CWG2623).
    MetadataDelete<Box>{}(item.box.release());
  }
  value = snapshot();
  for (auto count : value.resources) require(!count, "known native proof did not dispose role payloads");
  for (const auto &shape : shapes) {
    SourceClaim claim; MetadataReservations blocked;
    require(cleanupSource.reserve(shape.role, claim) == Result::Ok, "resource role reusable after payload destruction");
    const auto before = snapshot().metadataAllocations;
    require(claim.reserveResourceMetadata(blocked) == Result::WouldBlock && !blocked.scoped() &&
      snapshot().metadataAllocations == before, "weak resource batch was reused or partially charged");
    require(claim.cancelUnaccepted() == Result::Ok, "known refused resource claim rollback");
  }
  JoinReceipt blocked;
  require(root->seal() == Result::Ok && root->tryJoin(blocked) == Result::WouldBlock,
    "source join ignored resource weak control blocks");
  weak.clear(); close(root);
  beforeMetadataAllocate = {}; beforeMetadataFree = {};
  MetadataReservations legacy;
  auto legacyState = std::make_shared<State>(MetadataAllocation{}, ResourceCustody{}, false);
  auto legacyBox = makeMetadataUnique<Box>(legacy, MetadataKind::BufferCpu, legacyState, false);
  MetadataDelete<Box>{}(legacyBox.release());
  bool threw = false;
  try { (void)new Box(MetadataUniqueRecord{}, legacyState, true); }
  catch (const std::runtime_error &) { threw = true; }
  require(threw && legacyState.use_count() == 1,
    "legacy constructor unwind destroyed a completed-object payload twice");
}
int main() {
  pending.reserve(128);
  require(completion_ticket_c_abi() && stats().live_tickets == 0, "C ABI/empty process");
  taskMetadataChecks();
  remainingResourceMetadataChecks();
  frameMetadataChecks();
  Caller none;
  auto *noTicket = reinterpret_cast<MGPUCompletionTicket *>(uintptr_t(1));
  require(mgpuPrivateOutputWriteTrackedAsync(nullptr, nullptr, nullptr, &none, &noTicket) == -1 && !noTicket && !backendCalls,
    "missing callback accepts no operation");
  require(mgpuImportedFrameRetireTrackedAsync(nullptr, callback, &none, nullptr) == -1 && !backendCalls,
    "missing output accepts no operation");

  // Actual SDK callback return is independent of backend/native proxy epilogues.
  Caller held;
  Gate callerGate, proxyGate;
  held.gate = &callerGate; tailGate = &proxyGate; pauseTail = true;
  require(mgpuPrivateOutputWriteTrackedAsync(nullptr, nullptr, callback, &held, &held.ticket) == 0, "deferred write ticket");
  auto delivery = pending.back(); pending.clear();
  std::thread worker([&] {delivery.callback(0, delivery.userdata);});
  callerGate.observe();
  MGPUCompletionReceipt r{};
  require(mgpuCompletionTicketCollect(held.ticket, &r, sizeof(r)) == -4 && !r.ticket_id, "held callback cannot collect");
  callerGate.resume(); proxyGate.observe();
  collect(held.ticket, MGPU_COMPLETION_WRITE_OUTPUT, 0);
  require(mgpuCompletionTicketRelease(held.ticket) == 0 && stats().live_tickets == 1 &&
    stats().released_with_native_users == 1, "public release cannot refund a live native proxy");

  std::vector<MGPUCompletionTicket *> tickets(127);
  std::vector<Caller> callers(127);
  for (size_t i = 0; i < tickets.size(); ++i)
    require(mgpuPrivateOutputRetireTrackedAsync(nullptr, callback, &callers[i], &tickets[i]) == 0, "remaining quota");
  require(stats().live_tickets == 128, "native tail remains charged");
  auto calls = backendCalls.load();
  require(mgpuImportedFrameRetireTrackedAsync(nullptr, callback, &none, &noTicket) == -4 && !noTicket && backendCalls == calls,
    "quota refusal precedes backend work");
  proxyGate.resume(); worker.join();
  require(stats().live_tickets == 127, "proxy return refunds only its own slot");
  for (size_t i = 0; i < tickets.size(); ++i) {
    pending[i].callback(0, pending[i].userdata);
    collect(tickets[i], MGPU_COMPLETION_RETIRE_OUTPUT, 0);
    require(mgpuCompletionTicketRelease(tickets[i]) == 0, "retirement receipt release");
  }
  pending.clear(); require(stats().live_tickets == 0, "all known users drained");

  // The retired callback cookie cannot route to any live successor, even if
  // the allocator recycles its former State address.
  Caller successor;
  require(mgpuPrivateOutputWriteTrackedAsync(nullptr, nullptr, callback, &successor, &successor.ticket) == 0,
    "successor ticket");
  const auto successorDelivery = pending.back(); pending.clear();
  require(delivery.userdata != successorDelivery.userdata, "callback identities never recycle");
  delivery.callback(0, delivery.userdata);
  MGPUCompletionInfo successorInfo{};
  require(held.calls == 1 && !successor.calls &&
    mgpuCompletionTicketGetInfo(successor.ticket, &successorInfo, sizeof(successorInfo)) == 0 &&
    successorInfo.phase == MGPU_COMPLETION_PENDING, "stale callback cannot complete successor");
  successorDelivery.callback(0, successorDelivery.userdata);
  collect(successor.ticket, MGPU_COMPLETION_WRITE_OUTPUT, 0);
  require(mgpuCompletionTicketRelease(successor.ticket) == 0, "successor release");

  // Inline callback is not underlying-call return. No holder escapes early.
  mode = Mode::Inline;
  Gate originalReturn; returnGate = &originalReturn;
  Caller early;
  MGPUCompletionTicket *earlyTicket = nullptr;
  std::thread creating([&] {require(mgpuComputeShaderUseRetireTrackedAsync(nullptr, callback, &early, &earlyTicket) == 0,
    "inline callback before original API returns");});
  originalReturn.observe(); require(early.calls == 1 && stats().live_tickets == 1, "inline callback stays rooted");
  originalReturn.resume(); creating.join(); returnGate = nullptr;
  collect(earlyTicket, MGPU_COMPLETION_RETIRE_USE, 0); require(mgpuCompletionTicketRelease(earlyTicket) == 0, "early release after return");

  mode = Mode::Refuse;
  Caller refused;
  require(mgpuBufferDestroyTrackedAsync(fakeLease, fakeBuffer, callback, &refused, &noTicket) == -4 && !noTicket && !refused.calls &&
    stats().live_tickets == 0, "sync destroy refusal consumes no pointer/callback/ticket");
  mode = Mode::Failure;
  Caller failed;
  require(mgpuComputeShaderDestroyTrackedAsync(fakeLease, fakeShader, callback, &failed, &failed.ticket) == 0, "accepted checked destroy");
  collect(failed.ticket, MGPU_COMPLETION_DESTROY_SHADER, -5, MGPU_COMPLETION_RESOURCE_CONSUMED);
  require(mgpuCompletionTicketRelease(failed.ticket) == 0 && failed.calls == 1, "failure receipt is not resource reuse permission");

  mode = Mode::Inline;
  Caller frame, writer;
  require(mgpuImportedFrameRetireTrackedAsync(nullptr, callback, &frame, &frame.ticket) == 0 &&
    mgpuPrivateOutputWriterRetireTrackedAsync(nullptr, callback, &writer, &writer.ticket) == 0, "remaining fixed entrypoints");
  collect(frame.ticket, MGPU_COMPLETION_RETIRE_FRAME, 0); collect(writer.ticket, MGPU_COMPLETION_RETIRE_WRITER, 0);
  require(mgpuCompletionTicketRelease(frame.ticket) == 0 && mgpuCompletionTicketRelease(writer.ticket) == 0 && stats().live_tickets == 0,
    "fixed calls cleanly release");

  // A faulty backend returns refusal while its callback already holds a native
  // reference but has not reached State::gate. The caller remains quarantined;
  // the resumed proxy must not invoke callback storage cleared by refusal.
  Gate refusedAdmission; admissionGate = &refusedAdmission; pauseAdmission = true;
  mode = Mode::RefuseRacing;
  auto *racing = new Caller;
  require(mgpuBufferDestroyTrackedAsync(fakeLease, fakeBuffer, callback, racing, &racing->ticket) == -7 && racing->ticket &&
    !racing->calls, "admitted callback on checked refusal keeps quarantine");
  refusedAdmission.resume(); refusalWorker.join(); admissionGate = nullptr;
  require(!racing->calls && mgpuCompletionTicketCollect(racing->ticket, &r, sizeof(r)) == -7 && !r.ticket_id &&
    mgpuCompletionTicketRelease(racing->ticket) == -7, "refused callback never invokes caller storage");

  // Contract violations keep heap-backed caller userdata and ticket roots until
  // process exit. Nothing asserts these failures permit native/resource cleanup.
  for (auto fault : {Mode::Duplicate, Mode::Inline, Mode::Throw}) {
    mode = fault;
    auto *caller = new Caller; caller->throws = fault == Mode::Inline;
    MGPUCompletionTicket *ticket = nullptr;
    const auto accepted = mgpuPrivateOutputWriteTrackedAsync(nullptr, nullptr, callback, caller, &ticket);
    if (accepted != -7 || !ticket) std::cerr << "fault=" << static_cast<int>(fault) << " result=" << accepted << " ticket=" << !!ticket << '\n';
    require(accepted == -7 && ticket, "uncertain completion owns quarantine");
    require(mgpuCompletionTicketCollect(ticket, &r, sizeof(r)) == -7 && !r.ticket_id &&
      mgpuCompletionTicketRelease(ticket) == -7, "unknown failure cannot release its owner");
  }
  require(stats().live_tickets == 4 && stats().quarantined_tickets == 4, "quarantine stays charged");
  scopedCleanupChecks();
  scopedRetirementChecks();
  scopedOutputChecks();
  require(stats().live_tickets == 5 && stats().quarantined_tickets == 5, "scoped uncertainty adds one retained ticket");
  std::cout << "PASS completion tickets, scoped cleanup, frame/use and output retirement/write: thirteen operation/task/ticket plus forty-seven resource slots, allocation-fault rollbacks, fixed claim authority, later public work, weak/free tails and CPU constructor unwind; one family through primary/fence/wait/ticket and final global job, both ticket/job orders, Seal/Poison, real closure/proxy tails, enqueue refusal, exact source and retained uncertainty; legacy C ABI/callback/quota cases pass; CPU backends only\n";
}
