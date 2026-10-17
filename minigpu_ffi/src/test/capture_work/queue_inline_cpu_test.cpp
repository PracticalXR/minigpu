#include "../../include/minigpu_capture_work_owner.h"
#include "../../include/minigpu_capture_payload.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <mutex>
#ifdef _WIN32
#include <malloc.h>
#endif

namespace {
thread_local bool rejectAllocation = false;
thread_local size_t allocationAttempts = 0;
void *allocate(size_t size, size_t alignment = 0) {
  if (rejectAllocation) { ++allocationAttempts; throw std::bad_alloc(); }
  if (!size) size = 1;
  void *result = nullptr;
  if (!alignment) result = std::malloc(size);
#ifdef _WIN32
  else result = _aligned_malloc(size, alignment);
#else
  else if (posix_memalign(&result, alignment, size) != 0) result = nullptr;
#endif
  if (!result) throw std::bad_alloc();
  return result;
}
void freeAligned(void *p) noexcept {
#ifdef _WIN32
  _aligned_free(p);
#else
  std::free(p);
#endif
}
struct NoAllocation {
  NoAllocation() { rejectAllocation = true; allocationAttempts = 0; }
  ~NoAllocation() { rejectAllocation = false; }
};
void require(bool good, const char *why) {
  if (!good) { std::cerr << "FAIL queue/inline " << why << '\n'; std::exit(1); }
}
using namespace mgpu;
struct Backend {
  task_queue::Core queue;
  std::mutex mutex;
  bool gateHeld = false, stopped = false, violated = false;
  task_queue::Stamp next;
  struct Gate {
    explicit Gate(Backend &owner) : owner(owner), lock(owner.mutex) { owner.gateHeld = true; }
    ~Gate() { owner.gateHeld = false; }
    Backend &owner;
    std::unique_lock<std::mutex> lock;
  };
  static int enqueue(void *context, task_queue::Prepared &task) noexcept {
    auto &owner = *static_cast<Backend *>(context);
    try {
      Gate gate(owner);
      if (owner.stopped) return MGPU_CONTEXT_LEASE_RETIRED;
      return owner.queue.pushScoped(task, owner.next) ? MGPU_CONTEXT_LEASE_OK : MGPU_CONTEXT_LEASE_INVALID;
    } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
  }
  void legacy(std::function<void()> task, task_queue::Stamp stamp) {
    Gate gate(*this);
    queue.pushLegacy(task, stamp);
  }
  bool take(task_queue::Execution &out) {
    Gate gate(*this); return queue.takeFront(out);
  }
};
struct Node {
  explicit Node(Backend &backend) : backend(backend) {}
  ~Node() { if (backend.gateHeld) backend.violated = true; }
  Backend &backend;
  InlineCallable<void()> action;
  task_queue::Entry entry{this, [](void *p) noexcept {
    auto &node = *static_cast<Node *>(p);
    if (node.backend.gateHeld) node.backend.violated = true;
    node.action();
  }};
};
void mixedQueue() {
  Backend backend;
  std::array<int, 5> order{};
  std::array<long long, 5> stamps{};
  size_t count = 0;
  auto two = std::make_shared<Node>(backend), four = std::make_shared<Node>(backend);
  auto five = std::make_shared<Node>(backend);
  std::weak_ptr<Node> heldTwo = two, heldFour = four, heldFive = five;
  task_queue::Prepared second, fourth, nested;
  require(second.prepare(two->entry, two) && fourth.prepare(four->entry, four) &&
      nested.prepare(five->entry, five), "prepare real embedded entries");
  int nestedResult = MGPU_CONTEXT_LEASE_INTERNAL;
  two->action = [&] {
    order[count++] = 2;
    backend.next = {true, 50}; nestedResult = Backend::enqueue(&backend, nested);
  };
  four->action = [&] { order[count++] = 4; };
  five->action = [&] { order[count++] = 5; };
  backend.legacy([&] { if (backend.gateHeld) backend.violated = true; order[count++] = 1; }, {true, 10});
  backend.next = {true, 20};
  require(Backend::enqueue(&backend, second) == 0 && !second, "first scoped transfer");
  backend.legacy([&] { if (backend.gateHeld) backend.violated = true; order[count++] = 3; }, {true, 30});
  backend.next = {true, 40};
  bool completed = false;
  {
    NoAllocation guard;
    completed = Backend::enqueue(&backend, fourth) == 0;
    two.reset(); four.reset(); five.reset();
    size_t index = 0;
    for (;;) {
      task_queue::Execution execution;
      if (!backend.take(execution)) break;
      const auto stamp = execution.stamp();
      if (index >= stamps.size() || !stamp.profiled) { completed = false; break; }
      stamps[index++] = stamp.enqueuedUs;
      execution.run();
      if (backend.take(execution)) completed = false; // Never overwrite a live execution holder.
    }
    completed = completed && index == stamps.size() && allocationAttempts == 0;
  }
  require(completed && nestedResult == 0 && !fourth && !nested && backend.queue.empty() &&
      order == std::array<int, 5>{1, 2, 3, 4, 5} &&
      stamps == std::array<long long, 5>{10, 20, 30, 40, 50}, "mixed FIFO, nested enqueue, profile and no allocation");
  require(!backend.violated && heldTwo.expired() && heldFour.expired() && heldFive.expired(),
      "job invocation/last-owner tails must be outside the queue gate");
}
void refusedQueue() {
  Backend backend;
  auto owner = std::make_shared<Node>(backend);
  owner->action = [] {};
  task_queue::Prepared first, duplicate;
  require(first.prepare(owner->entry, owner) && duplicate.prepare(owner->entry, owner), "duplicate witness preparation");
  backend.stopped = true;
  require(Backend::enqueue(&backend, first) == MGPU_CONTEXT_LEASE_RETIRED && first && backend.queue.empty(),
      "stopped refusal retains exact unpublished token");
  backend.stopped = false;
  require(Backend::enqueue(&backend, first) == 0 && !first &&
      Backend::enqueue(&backend, duplicate) == MGPU_CONTEXT_LEASE_INVALID && duplicate,
      "one entry cannot be linked twice");
  task_queue::Execution execution;
  require(backend.take(execution), "dequeue retained entry"); execution.run();
  bool repeated = false;
  try { execution.run(); } catch (const std::bad_function_call &) { repeated = true; }
  require(repeated && !backend.take(execution) && backend.queue.empty(), "one-shot execution and no live-holder overwrite");
}
void legacyRefusal() {
  Backend backend;
  bool allocationFailed = false;
  // Fill only existing legacy storage until its next block/map allocation is
  // refused. The following payload-bearing append must retain its input.
  {
    NoAllocation guard;
    try {
      std::function<void()> empty;
      for (unsigned n = 0; n != 4096; ++n) backend.legacy(std::move(empty), {});
    } catch (const std::bad_alloc &) { allocationFailed = true; }
  }
  require(allocationFailed, "legacy deque allocation-failure boundary reached");
  auto owner = std::make_shared<Node>(backend);
  owner->action = [] {};
  std::weak_ptr<Node> weak = owner;
  std::function<void()> payload = [owner] { owner->action(); };
  owner.reset(); allocationFailed = false;
  {
    NoAllocation guard;
    try {
      Backend::Gate gate(backend);
      backend.queue.pushLegacy(payload);
    } catch (const std::bad_alloc &) { allocationFailed = true; }
  }
  require(allocationFailed && payload && !weak.expired() && !backend.violated,
      "failed empty-record allocation consumed or destroyed caller payload");
  for (;;) {
    task_queue::Execution execution;
    if (!backend.take(execution)) break;
    execution.run();
  }
  payload = {};
  require(weak.expired() && !backend.violated, "legacy payload disposed outside gate");
}
void preparedProductionWork() {
  using namespace source_scope;
  SourceRootOwner source;
  require(SourceRoot::create({10, 9, 1, 0}, source) == Result::Ok, "source creation");
  const auto domain = source->domain();
  Backend backend;
  bool ran = false, returned = false, noHeap = false;
  {
    capture_cleanup::Family family;
    capture_work::PreparedWork work;
    require(capture_cleanup::Family::beginWork(domain, family) == Result::Ok &&
        work.reserve(family) == 0 && work.configureSynchronous([&] {
          ran = !backend.gateHeld; return capture_work::Outcome::done(23);
        }) == 0, "production job prepared before heap guard");
    {
      NoAllocation guard;
      task_queue::Execution execution;
      noHeap = work.enqueue(Backend::enqueue, &backend) == 0 && backend.take(execution);
      if (noHeap) { execution.run(); returned = work.wait() == 23; }
      noHeap = noHeap && allocationAttempts == 0;
    }
  }
  Snapshot snapshot;
  require(ran && returned && noHeap && backend.queue.empty() && domain.snapshot(snapshot) == Result::Ok &&
      snapshot.operationFamilies == 0 && snapshot.continuations[0] == 0,
      "real source-owned enqueue/dequeue/run/result and known refund without C++ heap allocation");
  source->seal(); JoinReceipt joined;
  require(source->tryJoin(joined) == Result::Ok && source->release(joined) == Result::Ok, "known joined source");
}
void inlineCallable() {
  struct Wide {
    alignas(std::max_align_t) unsigned char bytes[96]{};
    int *result;
    void operator()(int value) { *result += value + bytes[0]; }
  };
  int result = 0;
  bool noHeap = false;
  {
    NoAllocation guard;
    InlineCallable<void(int)> first(Wide{{}, &result});
    InlineCallable<void(int)> moved(std::move(first));
    first = std::move(moved);
    first(19);
    InlineCallable<void()> empty(static_cast<void (*)()>(nullptr));
    noHeap = first && !moved && !empty && allocationAttempts == 0;
  }
  require(noHeap && result == 19, "wide/aligned capture move and invocation must not allocate");
}
void fixedCapturePayload() {
  using namespace source_scope;
  namespace fixed = capture_recipe_v1;
  SourceRootOwner source;
  require(SourceRoot::create({10, 9, 1, 0}, source) == Result::Ok, "payload source creation");
  const auto domain = source->domain();
  std::string callerCode(fixed::wgsl);
  std::array<uint8_t, fixed::parameterBytes> callerDimensions{10, 0, 0, 0, 9};
  bool noHeap = false;
  {
    NoAllocation guard;
    capture_payload::Array<uint8_t, fixed::parameterBytes> initial(true);
    initial.assign(callerDimensions.data(), callerDimensions.data() + callerDimensions.size());
    callerDimensions[0] = 11;
    ResourceKind role;
    require(capture_recipe::classifyResource(domain, fixed::parameterBytes, fixed::parametersUsage,
        initial.data(), static_cast<uint32_t>(initial.size()), role) == Result::Ok &&
        role == ResourceKind::Parameters, "copied initial parameters retain exact geometry");
    initial[8] = 1;
    require(capture_recipe::classifyResource(domain, fixed::parameterBytes, fixed::parametersUsage,
        initial.data(), static_cast<uint32_t>(initial.size()), role) == Result::Invalid,
        "copied parameters still reject nonzero padding");
    initial[8] = 0;

    capture_payload::Kernel prepared(true), committed(true), legacy;
    require(prepared.empty() && !prepared.assign(callerCode.data(), callerCode.size() - 1) &&
        prepared.empty() && prepared.assign(callerCode.data(), callerCode.size()),
        "readiness requires the entire exact recipe");
    callerCode[1] = '\0';
    require(prepared.view() == fixed::wgsl && prepared.data() == fixed::wgsl.data() &&
        !prepared.assign(callerCode.data(), callerCode.size()) && prepared.view() == fixed::wgsl,
        "canonical kernel never borrows mutable caller storage or accepts a changed/NUL module");
    require(!prepared.swap(legacy) && prepared.view() == fixed::wgsl && legacy.empty() &&
        prepared.swap(committed) && prepared.empty() && committed.view() == fixed::wgsl,
        "kernel commit preserves mode and readiness, mismatched modes do not consume input");

    using Binding = capture_recipe::Binding;
    capture_payload::Array<Binding, capture_payload::bindingSlots> pending(true), bindings(true);
    pending.resize(capture_payload::bindingSlots);
    // Prepare by slot while caller descriptor order is reversed, as production does.
    const std::array<Binding, 2> reversed{{
        {fixed::parametersSlot, 3, domain, ResourceKind::Parameters, 0, 16},
        {fixed::packedSlot, 1, domain, ResourceKind::PackedBuffer, 0, 360}}};
    require(capture_recipe::validateSetup(domain, domain, ResourceKind::RecipeShader,
        committed.data(), committed.size(), fixed::frameSlot, reversed) == Result::Ok,
        "fixed storage does not change descriptor-order admission");
    for (const auto &binding : reversed) pending[binding.slot] = binding;
    require(pending.swap(bindings) && pending.empty() && bindings.size() == 3 &&
        bindings[1].role == ResourceKind::PackedBuffer && bindings[2].role == ResourceKind::Parameters,
        "slot-indexed setup commit transfers exact source bindings");
    capture_payload::Array<Binding, capture_payload::bindingSlots> descriptors(true), dynamic;
    descriptors.reserve(bindings.size());
    for (const auto &binding : bindings) if (binding.role != ResourceKind::Count)
      descriptors.push_back(binding);
    bool reserveRefused = false, resizeRefused = false, appendRefused = false;
    try { descriptors.reserve(4); } catch (const std::bad_alloc &) { reserveRefused = true; }
    try { descriptors.resize(4); } catch (const std::bad_alloc &) { resizeRefused = true; }
    descriptors.push_back(Binding{});
    try { descriptors.push_back(Binding{}); } catch (const std::bad_alloc &) { appendRefused = true; }
    require(reserveRefused && resizeRefused && appendRefused && descriptors.size() == 3 &&
        descriptors[0].bytes == 360 && descriptors[1].bytes == 16 &&
        !bindings.swap(dynamic) && dynamic.empty() && bindings.size() == 3,
        "fixed descriptors refuse overflow and mode mismatch without partial mutation or heap fallback");
    noHeap = allocationAttempts == 0;
  }
  require(noHeap, "scoped initial/kernel/binding/descriptor payload storage must not allocate");
  source->seal(); JoinReceipt joined;
  require(source->tryJoin(joined) == Result::Ok && source->release(joined) == Result::Ok, "payload source joined");

  // Unscoped users keep arbitrary WGSL and larger dynamic binding tables.
  capture_payload::Array<unsigned, capture_payload::bindingSlots> legacy;
  legacy.resize(19); legacy[18] = 79;
  const std::string custom(4096, 'x');
  capture_payload::Kernel kernel;
  require(kernel.assign(custom.data(), custom.size()) && kernel == custom && !(custom != kernel) &&
      kernel.data() != custom.data() && legacy.size() == 19 && legacy.legacy()[18] == 79,
      "legacy storage still owns arbitrary code and dynamically sized tables");
}
void fixedPayloadOwners() {
  auto owner = std::make_shared<unsigned>(47);
  std::weak_ptr<unsigned> weak = owner;
  bool noHeap = false;
  {
    NoAllocation guard;
    capture_payload::Array<std::shared_ptr<unsigned>, 3> pending(true), committed(true), wrongMode;
    pending.push_back(owner);
    owner.reset();
    require(!pending.swap(wrongMode) && !weak.expired() && pending.size() == 1,
        "refused payload transfer retains the actual owner");
    require(pending.swap(committed) && pending.empty() && !weak.expired(),
        "committed payload retains the exact owner");
    auto acceptedTail = committed;
    committed.clear();
    require(!weak.expired(), "clearing one payload does not dispose an accepted tail");
    acceptedTail.clear();
    require(weak.expired(), "fixed clear releases owners only at their final tail");
    noHeap = allocationAttempts == 0;
  }
  require(noHeap, "fixed payload ownership transfer and disposal must not allocate");
}
} // namespace

void *operator new(size_t size) { return allocate(size); }
void *operator new[](size_t size) { return allocate(size); }
void *operator new(size_t size, std::align_val_t alignment) { return allocate(size, static_cast<size_t>(alignment)); }
void *operator new[](size_t size, std::align_val_t alignment) { return allocate(size, static_cast<size_t>(alignment)); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, size_t) noexcept { std::free(p); }
void operator delete[](void *p, size_t) noexcept { std::free(p); }
void operator delete(void *p, std::align_val_t) noexcept { freeAligned(p); }
void operator delete[](void *p, std::align_val_t) noexcept { freeAligned(p); }
void operator delete(void *p, size_t, std::align_val_t) noexcept { freeAligned(p); }
void operator delete[](void *p, size_t, std::align_val_t) noexcept { freeAligned(p); }

void captureQueueInlineChecks() {
  inlineCallable(); mixedQueue(); refusedQueue(); legacyRefusal(); preparedProductionWork();
  fixedCapturePayload(); fixedPayloadOwners();
}
