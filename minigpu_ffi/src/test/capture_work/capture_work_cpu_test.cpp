#include "../capture_cpu_queue.h"
// The real prepared production jobs/receipts and role custody. Only native
// actions, enqueue and event pumping are substituted; no GPU/context is created.
#include "../../include/minigpu_capture_work_owner.h"
#include "../../include/minigpu_inline_result.h"
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
using namespace mgpu::source_scope;
using namespace mgpu::capture_work;
using mgpu::capture_recipe::ResourceCustody;
namespace mgpu::capture_work {
static std::atomic<bool> failResultConstruction{false}, failResultWait{false};
static std::atomic<bool> resultWaitLocked{false};
void testInlineResultConstruct() {
  if (failResultConstruction.exchange(false)) throw std::runtime_error("CPU result initialization failed");
}
void testInlineResultWait() {
  if (failResultWait.exchange(false)) throw std::runtime_error("CPU result wait failed");
}
void testInlineResultLockedWait() { resultWaitLocked = true; }
}
static void require(bool good, const char *why) {
  if (!good) { std::cerr << "FAIL " << why << '\n'; std::exit(1); }
}
static SourceRootOwner root() {
  SourceRootOwner value;
  require(SourceRoot::create({10, 9, 1, 0}, value) == Result::Ok, "source creation");
  return value;
}
static Snapshot snapshot(const SourceDomain &source) {
  Snapshot s; require(source.snapshot(s) == Result::Ok, "snapshot"); return s;
}
static void counts(const SourceDomain &source, uint32_t primary, uint32_t fence,
                   uint32_t wait, uint32_t families) {
  const auto s = snapshot(source);
  require(s.continuations[0] == primary && s.continuations[1] == fence &&
      s.continuations[2] == wait && s.operationFamilies == families, "actual task/fence/wait/family tails");
}
static void close(SourceRoot &source) {
  source.seal(); JoinReceipt joined;
  require(source.tryJoin(joined) == Result::Ok && source.release(joined) == Result::Ok, "known source join");
}
struct Queue {
  bool reject = false;
  std::vector<std::function<void()>> tasks;
  static int enqueue(void *context, mgpu::task_queue::Prepared &task) noexcept {
    auto &queue = *static_cast<Queue *>(context);
    return queue.reject ? MGPU_CONTEXT_LEASE_INTERNAL
        : mgpu::capture_cpu_test::enqueueWitness(queue.tasks, task);
  }
};
struct Probe {
  std::function<void()> onDestruction;
  ~Probe() { if (onDestruction) onDestruction(); }
};
struct Owner : ResourceCustody {
  explicit Owner(ResourceCustody &&role) : ResourceCustody(std::move(role)) {}
  std::shared_ptr<Probe> native;
};
struct Gate {
  std::mutex mutex; std::condition_variable cv;
  bool entered = false, resumed = false;
  void hold() {
    std::unique_lock<std::mutex> lock(mutex); entered = true; cv.notify_all();
    require(cv.wait_for(lock, std::chrono::seconds(5), [&] { return resumed; }), "held native/callback tail timed out");
  }
  void observe() {
    std::unique_lock<std::mutex> lock(mutex);
    require(cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }), "native/callback tail not entered");
  }
  void resume() { std::lock_guard<std::mutex> lock(mutex); resumed = true; cv.notify_all(); }
};

static void synchronousResults() {
  InlineResult one;
  require(one.publish(17) && !one.publish(99) && one.wait() == 17,
    "one-shot result cannot be overwritten by duplicate publication");
  for (bool waitFirst : {false, true}) for (int status : {0, -5}) {
    auto source = root(); const auto domain = source->domain(); Queue queue;
    std::weak_ptr<Probe> native;
    {
      Family family;
      require(Family::beginWork(domain, family) == Result::Ok, "synchronous Work family");
      PreparedWork work;
      require(work.wait() == -1 && work.reserve(family) == 0 && work.wait() == -1,
        "empty or unconfigured wait must refuse without blocking");
      auto owner = std::make_shared<Probe>(); native = owner;
      require(work.configureSynchronous([owner, status] { return Outcome::done(status); }) == 0 &&
        work.configureSynchronous([] { return Outcome::done(); }) == -1 &&
        work.configure([] { return Outcome::done(); }, [](int) {}) == -1 &&
        work.wait() == -1 && work.enqueue(Queue::enqueue, &queue) == 0,
        "synchronous configuration is exclusive and requires accepted enqueue");
      owner.reset();
      int result = 999;
      if (waitFirst) {
        resultWaitLocked = false;
        std::thread waiter([&] { result = work.wait(); });
        while (!resultWaitLocked.load()) std::this_thread::yield();
        // The waiter owns the result mutex until wait atomically releases it;
        // publication cannot race ahead of this blocked-wait path.
        queue.tasks.front()(); waiter.join();
      } else {
        queue.tasks.front()(); result = work.wait();
      }
      require(result == status && work.wait() == -1 && !native.expired(),
        "result before/after wait, known status or sole waiter contract");
      counts(domain, 1, 0, 0, 1); // Publication/wait cannot refund live job/tails.
    }
    counts(domain, 1, 0, 0, 1);
    queue.tasks.clear();
    require(native.expired(), "known result retained a completed native capture");
    counts(domain, 0, 0, 0, 0); close(*source);
  }
  // CPU synchronization construction failure remains unaccepted rollback.
  {
    auto source = root(); const auto domain = source->domain();
    {
      Family family; PreparedWork work;
      require(Family::beginWork(domain, family) == Result::Ok && work.reserve(family) == 0,
        "failed result setup reservation");
      failResultConstruction = true;
      require(work.configureSynchronous([] { return Outcome::done(); }) == -5 && work.wait() == -1,
        "result construction failure must not accept or wait");
    }
    require(snapshot(domain).phase == Phase::Open, "CPU result constructor failure poisoned source");
    counts(domain, 0, 0, 0, 0); close(*source);
  }
  for (bool seal : {false, true}) {
    auto source = root(); const auto domain = source->domain(); Queue refused; refused.reject = !seal;
    {
      Family family; PreparedWork work;
      require(Family::beginWork(domain, family) == Result::Ok && work.reserve(family) == 0 &&
        work.configureSynchronous([] { return Outcome::done(); }) == 0,
        "refused synchronous task preparation");
      if (seal) require(source->seal() == Result::Ok, "seal before synchronous acceptance");
      const int status = work.enqueue(Queue::enqueue, &refused);
      require(status == (seal ? MGPU_CONTEXT_LEASE_RETIRED : MGPU_CONTEXT_LEASE_INTERNAL) &&
        refused.tasks.empty() && work.wait() == -1,
        "known enqueue rejection or Seal must refuse waiting");
    }
    counts(domain, 0, 0, 0, 0); close(*source);
  }
  // Unknown results wake synchronous callers, retaining actual captures. A
  // thrown action's default INTERNAL status must never look like known rollback.
  for (bool throws : {false, true}) {
    auto source = root(); const auto domain = source->domain(); Queue queue;
    std::weak_ptr<Probe> native;
    {
      Family family; PreparedWork work; auto owner = std::make_shared<Probe>(); native = owner;
      require(Family::beginWork(domain, family) == Result::Ok && work.reserve(family) == 0 &&
        work.configureSynchronous([owner, throws]() -> Outcome {
          if (throws) throw std::runtime_error("unknown native action");
          return Outcome::uncertain();
        }) == 0 && work.enqueue(Queue::enqueue, &queue) == 0, "uncertain synchronous task");
      queue.tasks.front()();
      require(work.wait() == MGPU_CONTEXT_LEASE_QUARANTINED, "unknown result looked like known failure");
    }
    queue.tasks.clear();
    require(!native.expired() && snapshot(domain).phase == Phase::Poisoned,
      "synchronous uncertainty lost actual anchored native owner");
    counts(domain, 1, 0, 0, 1);
  }
  // Accepted wait exceptions remain visible and quarantine their job; no
  // ordinary INTERNAL result can trigger the caller's preacceptance rollback.
  SourceDomain exceptionDomain;
  {
    auto source = root(); const auto domain = source->domain(); Queue queue;
    exceptionDomain = domain;
    Family family; PreparedWork work;
    require(Family::beginWork(domain, family) == Result::Ok && work.reserve(family) == 0 &&
      work.configureSynchronous([] { return Outcome::done(); }) == 0 &&
      work.enqueue(Queue::enqueue, &queue) == 0, "accepted wait exception task");
    failResultWait = true; bool threw = false;
    try { (void)work.wait(); } catch (const std::runtime_error &) { threw = true; }
    require(threw && work.wait() == -1 && snapshot(domain).phase == Phase::Poisoned,
      "accepted synchronization exception became ordinary failure or a repeated wait");
    queue.tasks.front()();
  }
  require(snapshot(exceptionDomain).phase == Phase::Poisoned,
    "accepted wait exception lost poison after worker and caller disposal");
  counts(exceptionDomain, 1, 0, 0, 1);
}
void captureQueueInlineChecks();
int main() {
  captureQueueInlineChecks();
  synchronousResults();
  auto source = root(); const auto domain = source->domain();
  require(validateFrame(domain, 10, 9, 360) == Result::Ok &&
      validateFrame(domain, 9, 10, 360) == Result::Invalid &&
      validateFrame(domain, 10, 9, 359) == Result::Invalid &&
      validateFrame(domain, 0, 9, 0) == Result::Invalid &&
      validateFrame(domain, UINT32_MAX, UINT32_MAX, UINT64_MAX) == Result::Invalid,
      "exact source geometry and overflow reject before native allocation");
  require(validateDispatch(domain, 2, 2, 1) == Result::Ok &&
      validateDispatch(domain, 2, 2, 2) == Result::Invalid &&
      validateDispatch(domain, 1, 2, 1) == Result::Invalid,
      "scoped dispatch exactly covers its immutable 8x8 recipe geometry");
  {
    auto foreign = root(); Family family;
    require(Family::beginWork(domain, family) == Result::Ok &&
      family.matches(domain) && !family.matches(foreign->domain()) &&
      !family.matches({}) && !family.cleanup(), "immutable exact-source Work purpose");
    mgpu::capture_cleanup::PreparedTask cleanup;
    require(cleanup.prepare(family, [] { return true; }, [](int, void *) {}, nullptr) == -1,
      "Work cannot authorize checked cleanup");
    close(*foreign);
  }
  counts(domain, 0, 0, 0, 0);

  // Accepted native role proof cannot bypass final State/member/alias disposal.
  for (const auto kind : {ResourceKind::ImportedFrame, ResourceKind::ExclusiveUse}) {
    bool membersGone = false;
    ResourceCustody role;
    require(role.reserveSourceRole(domain, kind) == Result::Ok, "preallocate resource claim");
    auto owner = std::make_shared<Owner>(std::move(role));
    owner->native = std::make_shared<Probe>();
    owner->native->onDestruction = [&, kind] {
      require(snapshot(domain).resources[static_cast<size_t>(kind)] == 1,
        "source role outlives actual native/member destruction"); membersGone = true;
    };
    require(owner->acceptSourceRole() == Result::Ok, "accept original role");
    auto retained = owner; owner->proveNativeReleased(); owner.reset();
    require(!membersGone && snapshot(domain).resources[static_cast<size_t>(kind)] == 1,
      "public retirement/unroot cannot release retained State aliases");
    retained.reset();
    require(membersGone && snapshot(domain).resources[static_cast<size_t>(kind)] == 0, "final role disposition");
  }

  // Result publication is earlier than the actual caller/worker closure tails.
  Queue queue; bool completed = false, membersGone = false;
  std::function<void()> retained;
  {
    Family family; require(Family::beginWork(domain, family) == Result::Ok, "new Work family");
    PreparedWork work; require(work.reserve(family) == 0, "reserve before caller/native holders");
    auto native = std::make_shared<Probe>();
    native->onDestruction = [&] { counts(domain, 1, 0, 0, 1); membersGone = true; };
    require(work.configure([native] { return Outcome::done(); }, [&](int result) {
      require(result == 0, "native action result"); completed = true;
    }) == 0 && work.enqueue(Queue::enqueue, &queue) == 0, "actual work enqueue");
    require(work.wait() == MGPU_CONTEXT_LEASE_INVALID, "async work cannot acquire a synchronous waiter");
    native.reset(); retained = queue.tasks.front();
    require(source->seal() == Result::Ok, "Seal after work acceptance");
    queue.tasks.front()();
    require(completed && !membersGone, "admitted continuation executes after Seal");
    counts(domain, 1, 0, 0, 1);
  }
  queue.tasks.clear(); counts(domain, 1, 0, 0, 1);
  Family denied; require(Family::beginWork(domain, denied) == Result::Sealed, "later public Work rejects Seal");
  retained = {}; counts(domain, 0, 0, 0, 0); require(membersGone, "last task closure disposed");
  close(*source);

  // Seal before fresh acceptance refuses without entering the native action.
  source = root(); const auto sealedDomain = source->domain();
  {
    Family family; require(Family::beginWork(sealedDomain, family) == Result::Ok, "reserve Work before Seal");
    PreparedWork work; bool ran = false;
    require(work.reserve(family) == 0 && work.configure([&] { ran = true; return Outcome::done(); }, [](int) {}) == 0,
      "prepared unaccepted work");
    source->seal();
    require(work.enqueue(Queue::enqueue, &queue) == MGPU_CONTEXT_LEASE_RETIRED && !ran && queue.tasks.empty(),
      "Seal between reservation/acceptance is ordinary refusal");
  }
  counts(sealedDomain, 0, 0, 0, 0); close(*source);

  // The exact fence and wait objects survive callback publication AND pump return.
  source = root(); const auto waitDomain = source->domain(); source->seal();
  Gate pumpTail, callerTail; std::atomic<unsigned> calls{0};
  {
    Family family; require(Family::begin(waitDomain, family) == Result::Ok, "Cleanup after Seal");
    PreparedWait wait;
    require(wait.prepare(family, [&](WaitSignal &signal) {
      signal.result = 0; signal.completed = true; pumpTail.hold(); return Outcome::done();
    }, [&](int result) { require(result == 0, "completion result"); ++calls; callerTail.hold(); }) == 0 &&
      wait.enqueue(Queue::enqueue, &queue) == 0, "real prepared wait enqueue");
    retained = queue.tasks.front();
    std::thread worker([&] { queue.tasks.front()(); });
    pumpTail.observe(); counts(waitDomain, 0, 1, 1, 1);
    require(!calls.load(), "native publication cannot enter caller before callback/pump return");
    pumpTail.resume(); callerTail.observe(); counts(waitDomain, 0, 1, 1, 1);
    callerTail.resume(); worker.join(); require(calls == 1, "single returned completion");
    counts(waitDomain, 0, 1, 1, 1);
  }
  queue.tasks.clear(); counts(waitDomain, 0, 1, 1, 1);
  retained = {}; counts(waitDomain, 0, 0, 0, 0); close(*source);

  // Known nonacceptance rolls back both prepared continuations, no callback.
  source = root(); const auto rejectedDomain = source->domain();
  queue.reject = true;
  {
    Family family; require(Family::beginWork(rejectedDomain, family) == Result::Ok, "rejected standalone wait");
    PreparedWait wait;
    require(wait.prepare(family, [](WaitSignal &) { require(false, "refused wait cannot run"); return Outcome::done(); },
      [](int) { require(false, "refused wait cannot callback"); }) == 0 && wait.enqueue(Queue::enqueue, &queue) == -5,
      "known fence enqueue refusal");
  }
  counts(rejectedDomain, 0, 0, 0, 0); close(*source); queue.reject = false;

  // Unknown native work and callback exceptions retain actual native captures.
  for (bool callbackThrows : {false, true}) {
    source = root(); const auto unknownDomain = source->domain(); std::weak_ptr<Probe> live;
    {
      Family family; require(Family::beginWork(unknownDomain, family) == Result::Ok, "unknown Work family");
      PreparedWork work; require(work.reserve(family) == 0, "unknown task reservation");
      auto native = std::make_shared<Probe>(); live = native;
      require(work.configure([native, callbackThrows] {
        if (!callbackThrows) throw std::runtime_error("native acceptance uncertain");
        return Outcome::done();
      }, [callbackThrows](int) { if (callbackThrows) throw std::runtime_error("foreign callback uncertain"); }) == 0 &&
        work.enqueue(Queue::enqueue, &queue) == 0, "unknown work accepted");
      queue.tasks.front()();
    }
    queue.tasks.clear();
    require(!live.expired() && snapshot(unknownDomain).phase == Phase::Poisoned, "actual uncertainty anchor retains native owners");
    counts(unknownDomain, 1, 0, 0, 1);
  }
  source = root(); const auto unknownWait = source->domain(); std::weak_ptr<Probe> nativeWait;
  {
    Family family; require(Family::beginWork(unknownWait, family) == Result::Ok, "uncertain wait family");
    PreparedWait wait; auto native = std::make_shared<Probe>(); nativeWait = native;
    require(wait.prepare(family, [native](WaitSignal &) -> Outcome {
      throw std::runtime_error("native callback registration/pump uncertain");
    }, [](int) {}) == 0 && wait.enqueue(Queue::enqueue, &queue) == 0, "uncertain wait accepted");
    queue.tasks.front()();
  }
  queue.tasks.clear(); counts(unknownWait, 0, 1, 1, 1);
  require(!nativeWait.expired() && snapshot(unknownWait).phase == Phase::Poisoned,
    "uncertain callback receipt keeps its actual signal/native owners");

  // A parked wait leaves its worker task with the receipt, fence and owners
  // retained; the worker's later resume settles both exactly once.
  for (bool succeeded : {true, false}) {
    source = root(); const auto parkedDomain = source->domain(); std::weak_ptr<Probe> live;
    WaitSignal *signal = nullptr; std::atomic<unsigned> completions{0}; int seen = 1;
    {
      Family family; require(Family::beginWork(parkedDomain, family) == Result::Ok, "parked wait family");
      PreparedWait wait; auto native = std::make_shared<Probe>(); live = native;
      require(wait.prepare(family, [&signal, native](WaitSignal &armed) {
        armed.parked.linked = true; signal = &armed; // What WebGPUThread::park does.
        return Outcome::deferred();
      }, [&](int result) { seen = result; ++completions; }) == 0 &&
        wait.enqueue(Queue::enqueue, &queue) == 0, "parked wait accepted");
      queue.tasks.front()();
    }
    queue.tasks.clear();
    require(signal && signal->parked.ready == &signal->completed && signal->parked.resume &&
        !completions && !live.expired(), "a parked wait returns unsettled and keeps its owners");
    counts(parkedDomain, 0, 1, 1, 1);
    signal->result = succeeded ? MGPU_CONTEXT_LEASE_OK : MGPU_CONTEXT_LEASE_INTERNAL;
    signal->completed = true;
    auto &node = signal->parked; node.linked = false; node.resume(node);
    require(completions == 1, "a resumed wait completes once");
    if (succeeded) {
      require(seen == MGPU_CONTEXT_LEASE_OK && live.expired(), "a known resume releases every owner");
      counts(parkedDomain, 0, 0, 0, 0); close(*source);
    } else {
      require(seen == MGPU_CONTEXT_LEASE_INTERNAL && !live.expired() &&
          snapshot(parkedDomain).phase == Phase::Poisoned, "a failed resume retains its owners");
      counts(parkedDomain, 0, 1, 1, 1);
    }
  }
  // Deferring without linking the node would never resume: it is uncertain.
  source = root(); const auto unlinkedDomain = source->domain(); int unlinkedResult = 0;
  {
    Family family; require(Family::beginWork(unlinkedDomain, family) == Result::Ok, "unlinked wait family");
    PreparedWait wait;
    require(wait.prepare(family, [](WaitSignal &) { return Outcome::deferred(); },
      [&](int result) { unlinkedResult = result; }) == 0 && wait.enqueue(Queue::enqueue, &queue) == 0,
      "unlinked deferral accepted");
    queue.tasks.front()();
  }
  queue.tasks.clear();
  require(unlinkedResult == MGPU_CONTEXT_LEASE_QUARANTINED &&
      snapshot(unlinkedDomain).phase == Phase::Poisoned, "an unlinked deferral is quarantined");
  std::cout << "PASS production prepared work/fence/wait: exact geometry/source, Seal, real closure and callback/pump tails, role aliases, enqueue refusal, retained uncertainty; CPU only\n";
}
