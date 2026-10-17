#include "../include/minigpu_capture_setup_boxes.h"
#include "../include/minigpu_context_owner.h"
#include "../include/minigpu_setup_layout.h"
#include "../include/minigpu_context_limits.h"
#include "../include/minigpu_module_owner.h"
#include "../include/minigpu_default_owner_slot.h"
#include <array>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

namespace {
constexpr uint32_t kMaxContexts = mgpu::native_limits::maxContexts;
constexpr uint32_t kMaxLeases = mgpu::native_limits::maxContextLeases;
std::atomic<uint32_t> liveContexts{0}, liveLeases{0}, quarantinedContexts{0};
std::atomic<uint64_t> nextGeneration{1};
std::atomic<bool> registryClosing{false}, retirementClosing{false};

bool reserve(std::atomic<uint32_t> &count, uint32_t bound) {
  auto value = count.load(std::memory_order_relaxed);
  while (value < bound) {
    if (count.compare_exchange_weak(value, value + 1,
                                    std::memory_order_acq_rel))
      return true;
  }
  return false;
}

// Destruction can be triggered by the final GPU-worker task's lease. Drain and
// join that worker on this separate, bounded retirement worker, never on itself
// or a caller/UI thread. Live counts include queued and quarantined owners.
struct RetirementJob {
  mgpu::MGPU *gpu = nullptr;
  bool owned = true;
};
class RetirementWorker {
public:
  RetirementWorker() {
#ifndef __EMSCRIPTEN__
    worker_ = std::thread([this] { run(); });
#endif
  }
  ~RetirementWorker() {
    retirementClosing.store(true, std::memory_order_release);
#ifndef __EMSCRIPTEN__
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    changed_.notify_one();
    worker_.join();
#endif
  }
  void enqueue(mgpu::MGPU *gpu, bool owned = true) noexcept {
    const RetirementJob job{gpu, owned};
#ifdef __EMSCRIPTEN__
    cleanup(job);
#else
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) { ++quarantinedContexts; return; }
      // Every job owns one of the same 64 charged context reservations.
      if (count_ == jobs_.size()) {
        ++quarantinedContexts; // fail closed; never free under a live worker
        return;
      }
      jobs_[(head_ + count_) % jobs_.size()] = job;
      ++count_;
    }
    changed_.notify_one();
#endif
  }

private:
  static void cleanup(RetirementJob job) noexcept {
    try {
      if (job.owned) {
        job.gpu->destroyContext();
        delete job.gpu;
      } else {
        // Release only our attachment; other renderers keep their device.
        job.gpu->detachContext();
      }
      --liveContexts;
    } catch (...) {
      // A failed drain does not prove that captured pointers are no longer
      // live.
      ++quarantinedContexts;
    }
  }
#ifndef __EMSCRIPTEN__
  void run() {
    for (;;) {
      RetirementJob job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [this] { return count_ || stopping_; });
        if (!count_)
          return;
        job = jobs_[head_];
        head_ = (head_ + 1) % jobs_.size();
        --count_;
      }
      cleanup(job);
    }
  }
  std::array<RetirementJob, kMaxContexts> jobs_{};
  size_t head_ = 0, count_ = 0;
  bool stopping_ = false;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::thread worker_;
#endif
};

RetirementWorker &retirementWorker() {
  static RetirementWorker worker;
  return worker;
}

void retireGpu(mgpu::MGPU *gpu, bool owned) noexcept {
  // Preserve normal DLL worker drain/join. Late static retirement cannot call
  // into an already destroyed worker; unresolved native custody stays charged.
  if (retirementClosing.load(std::memory_order_acquire)) { ++quarantinedContexts; return; }
  retirementWorker().enqueue(gpu, owned);
}
struct Registry {
  std::mutex mutex;
  std::array<std::shared_ptr<mgpu::ContextOwner>, kMaxContexts> active{};
  mgpu::detail::DefaultOwnerSlot<mgpu::ContextOwner> ready{mutex};
  ~Registry() {
    registryClosing.store(true, std::memory_order_release);
    auto retiredDefault = ready.close();
    std::array<std::shared_ptr<mgpu::ContextOwner>, kMaxContexts> retiring{};
    { std::lock_guard<std::mutex> lock(mutex); retiring.swap(active); }
    // Locals die while registry fields and the earlier-created worker live.
    // No payload/free callbacks execute inside the registry lock.
  }
};
Registry &registry() {
  retirementWorker();
  static Registry value;
  return value;
}
void retireDefault(uint64_t generation) noexcept {
  if (registryClosing.load(std::memory_order_acquire)) return;
  try { registry().ready.retire(generation); } catch (...) {}
}
std::shared_ptr<mgpu::ContextOwner> findOwner(MGPUContextHandle *handle) {
  if (registryClosing.load(std::memory_order_acquire)) return {};
  auto &owners = registry();
  std::shared_ptr<mgpu::ContextOwner> found;
  {
    std::lock_guard<std::mutex> lock(owners.mutex);
    for (auto &owner : owners.active) {
      if (owner && reinterpret_cast<MGPUContextHandle *>(&owner->gpu()) == handle) { found = owner; break; }
    }
  }
  if (found) return found;
  auto display = owners.ready.lookup();
  return display && reinterpret_cast<MGPUContextHandle *>(&display->gpu()) == handle ? display : nullptr;
}
} // namespace

namespace mgpu {
namespace detail {
struct ReadyAttachment {
  explicit ReadyAttachment(MGPU &value) noexcept : gpu(value) {}
  ReadyAttachment(const ReadyAttachment &) = delete;
  ReadyAttachment &operator=(const ReadyAttachment &) = delete;
  ~ReadyAttachment() {
    if (attached) retireGpu(&gpu, false);
    else if (charged) --liveContexts;
  }
  MGPU &gpu;
  bool charged = false, attached = false;
  void adopt() noexcept { charged = false; attached = false; }
};
struct PinnedSnapshot {
  PinnedSnapshot() noexcept = default;
  PinnedSnapshot(const PinnedSnapshot &) = delete;
  PinnedSnapshot &operator=(const PinnedSnapshot &) = delete;
  ~PinnedSnapshot() {
    if (queue) wgpuQueueRelease(queue);
    if (device) wgpuDeviceRelease(device);
    if (instance) wgpuInstanceRelease(instance);
    if (charged) --liveLeases;
  }
  WGPUDevice device = nullptr;
  WGPUQueue queue = nullptr;
  WGPUInstance instance = nullptr;
  bool charged = false;
  void adopt() noexcept { device = nullptr; queue = nullptr; instance = nullptr; charged = false; }
};
}
void quarantineContextPin(std::shared_ptr<ContextPin> pin) noexcept {
  if (!pin) return;
  pin->sourceDomain().quarantine();
  struct Quarantine {
    std::mutex mutex;
    std::array<std::shared_ptr<ContextPin>, kMaxLeases> pins{};
  };
  // Intentionally survives static destruction: an unsuccessful native drain
  // must not become a final-owner deletion when a library/thread shuts down.
  auto *retained = &mgpu::detail::moduleOwner<Quarantine>();
  std::lock_guard<std::mutex> lock(retained->mutex);
  for (const auto &existing : retained->pins) {
    if (existing && sameQuarantinedCreation(existing->creationIdentity(),
                                            pin->creationIdentity())) return;
  }
  for (auto &slot : retained->pins) {
    if (!slot) {
      slot = std::move(pin);
      ++quarantinedContexts;
      return;
    }
  }
  // Unreachable: every distinct pin here and the incoming one consume the
  // same kMaxLeases reservation; duplicate scoped roots or legacy generations
  // returned above.
  std::terminate();
}

std::shared_ptr<ContextOwner> retainContextOwner(MGPUContextHandle *handle) {
  auto owner = findOwner(handle);
  // Owned-handle initialization must not repair a borrowed display device.
  return owner && owner->ownsContext() && !owner->retired() ? owner : nullptr;
}
std::shared_ptr<ContextOwner> retainHandleOwner(MGPUContextHandle *handle) noexcept {
  try {
    auto owner = findOwner(handle);
    return owner && !owner->retired() ? owner : nullptr;
  } catch (...) { return nullptr; }
}
ContextOwner::ContextOwner(std::unique_ptr<MGPU> gpu, uint64_t generation)
    : ownedGpu_(std::move(gpu)), gpu_(ownedGpu_.get()), generation_(generation) {}
ContextOwner::ContextOwner(detail::ReadyAttachment &attachment, uint64_t generation, bool funded) noexcept
    : gpu_(&attachment.gpu), generation_(generation), setupFunded_(funded) { attachment.adopt(); }
ContextOwner::~ContextOwner() {
  retireDefault(generation_);
  if (ownedGpu_) retireGpu(ownedGpu_.release(), true);
  else retireGpu(gpu_, false);
}
ContextPin::ContextPin(std::shared_ptr<ContextOwner> owner, detail::PinnedSnapshot &snapshot,
    uint64_t generation, bool funded, source_scope::SourceDomain source, source_scope::SourceClaim claim) noexcept
    : owner_(std::move(owner)), device_(snapshot.device), queue_(snapshot.queue), instance_(snapshot.instance),
      deviceGeneration_(generation), source_(std::move(source)), sourcePin_(std::move(claim)), setupFunded_(funded) {
  snapshot.adopt();
}
ContextPin::~ContextPin() {
  wgpuQueueRelease(queue_);
  wgpuDeviceRelease(device_);
  wgpuInstanceRelease(instance_);
  --liveLeases;
  owner_.reset();
  // The claim follows this one pin through every resource/task alias. Only
  // actual final pin destruction, after its native references, retires it.
  if (sourcePin_) (void)sourcePin_.retire();
}
bool ContextPin::current() const {
  return !owner_->retired() &&
         owner_->gpu().matchesReadyDevice(device_, deviceGeneration_);
}

MGPUContextHandle *createOwnedContext(const char *filter) noexcept {
  if (registryClosing.load(std::memory_order_acquire)) return nullptr;
  std::shared_ptr<ContextOwner> owner;
  bool charged = false;
  try {
    auto &owners = registry();
    const auto generation = detail::nextOwnerGeneration(nextGeneration);
    if (!generation) return nullptr;
    if (!reserve(liveContexts, kMaxContexts))
      return nullptr;
    charged = true;
    auto gpu = std::make_unique<MGPU>();
    if (filter && filter[0])
      gpu->setAdapterFilter(filter);
    owner = std::make_shared<ContextOwner>(std::move(gpu), generation);
    auto *handle = reinterpret_cast<MGPUContextHandle *>(&owner->gpu());
    std::lock_guard<std::mutex> lock(owners.mutex);
    for (auto &entry : owners.active) {
      if (!entry) {
        entry = owner;
        return handle;
      }
    }
  } catch (...) {
    if (charged && !owner)
      --liveContexts;
  }
  return nullptr;
}

void retireOwnedContext(MGPUContextHandle *handle) noexcept {
  if (!handle || registryClosing.load(std::memory_order_acquire)) return;
  std::shared_ptr<ContextOwner> retiring;
  auto &owners = registry();
  {
    std::lock_guard<std::mutex> lock(owners.mutex);
    for (auto &entry : owners.active) {
      if (entry &&
          reinterpret_cast<MGPUContextHandle *>(&entry->gpu()) == handle) {
        entry->retire();
        retiring = std::move(entry);
        break;
      }
    }
  }
  // Final owner release and retirement enqueue happen outside the registry
  // lock.
}

namespace {
int retainPin(std::shared_ptr<ContextOwner> owner, std::shared_ptr<ContextPin> &out,
    setup_custody::SetupTicket ticket = {}) noexcept {
  out.reset();
  detail::PinnedSnapshot snapshot;
  try {
    if (!owner || owner->retired()) return MGPU_CONTEXT_LEASE_RETIRED;
    const bool funded = bool(ticket);
    if (funded && (!ticket.hasRole(MGPU_CAPTURE_SETUP_ORIGINAL_PIN) || !owner->setupFunded()))
      return MGPU_CONTEXT_LEASE_UNSUPPORTED;
    uint64_t generation = 0;
    if (!owner->gpu().retainReadyContextSnapshot(&snapshot.device, &snapshot.queue, &snapshot.instance, &generation))
      return MGPU_CONTEXT_LEASE_NOT_READY;
    if (owner->retired()) return MGPU_CONTEXT_LEASE_RETIRED;
    if (!reserve(liveLeases, kMaxLeases)) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    snapshot.charged = true;
    out = funded ? setup_custody::makeSetupShared<ContextPin>(ticket, std::move(owner), snapshot, generation, true) :
        std::make_shared<ContextPin>(std::move(owner), snapshot, generation, false);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (setup_custody::Error e) {
    return e == setup_custody::Error::Uncertain ? MGPU_CONTEXT_LEASE_QUARANTINED : MGPU_CONTEXT_LEASE_INVALID;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
}
int retainContextPin(MGPUContextHandle *handle, std::shared_ptr<ContextPin> &out) noexcept {
  try { return retainPin(findOwner(handle), out); }
  catch (...) { out.reset(); return MGPU_CONTEXT_LEASE_INTERNAL; }
}
} // namespace mgpu


namespace mgpu {
std::shared_ptr<ContextPin> copyContextLeasePin(MGPUContextLease *lease) {
  return lease ? lease->pin : nullptr;
}
bool adoptSourceContextLease(MGPUContextLease *lease, const source_scope::SourceDomain &source) noexcept {
  if (!lease || lease->sourceManaged || !lease->pin || !source ||
      !sameSourceDomain(lease->pin->sourceDomain(), source)) return false;
  lease->sourceManaged = true;
  return true;
}
bool releaseSourceContextLease(MGPUContextLease *lease) noexcept {
  if (!lease || !lease->sourceManaged) return false;
  setup_custody::SetupDelete<MGPUContextLease>{}(lease);
  return true;
}
int createScopedContextLeaseImpl(
    MGPUContextLease *original, const source_scope::SourceSpec &spec,
    source_scope::SourceRootOwner &outRoot,
    MGPUContextLease **outLease, const setup_custody::SetupTickets *funding) noexcept {
  if (outLease) *outLease = nullptr;
  if (!outLease || outRoot || !original) return MGPU_CONTEXT_LEASE_INVALID;
  source_scope::SourceRootOwner root;
  source_scope::SourceClaim claim;
  setup_custody::SetupUnique<MGPUContextLease> lease;
  std::shared_ptr<ContextPin> scoped;
  detail::PinnedSnapshot snapshot;
  bool entered = false;
  const bool funded = funding != nullptr;
  if (funded && (!funding->valid() || !setup_custody::matchesOriginalRetain(original, *funding)))
    return MGPU_CONTEXT_LEASE_INVALID;
  const auto status = [](source_scope::Result result) {
    switch (result) {
    case source_scope::Result::Ok: return MGPU_CONTEXT_LEASE_OK;
    case source_scope::Result::Invalid: return MGPU_CONTEXT_LEASE_INVALID;
    case source_scope::Result::WouldBlock: return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    case source_scope::Result::Sealed:
    case source_scope::Result::Joined: return MGPU_CONTEXT_LEASE_RETIRED;
    case source_scope::Result::Poisoned: return MGPU_CONTEXT_LEASE_QUARANTINED;
    default: return MGPU_CONTEXT_LEASE_INTERNAL;
    }
  };
  const auto refuse = [&](int result) noexcept -> int {
    // Nothing has been submitted; AddRef/Release are the native refcount API,
    // not fallible GPU work. Clear every completed partial retention first.
    lease.reset();
    scoped.reset();
    // Snapshot guard owns only references/quota not adopted by a constructor.
    if (snapshot.queue) { wgpuQueueRelease(snapshot.queue); snapshot.queue = nullptr; }
    if (snapshot.device) { wgpuDeviceRelease(snapshot.device); snapshot.device = nullptr; }
    if (snapshot.instance) { wgpuInstanceRelease(snapshot.instance); snapshot.instance = nullptr; }
    if (snapshot.charged) { --liveLeases; snapshot.charged = false; }
    if (claim) {
      const auto disposition = entered ? claim.retire() : claim.cancelUnaccepted();
      if (disposition != source_scope::Result::Ok && root)
        root->domain().quarantine();
    }
    if (root) {
      source_scope::JoinReceipt receipt;
      if (root->seal() != source_scope::Result::Ok ||
          root->tryJoin(receipt) != source_scope::Result::Ok ||
          root->release(receipt) != source_scope::Result::Ok) {
        root->domain().quarantine();
        outRoot = std::move(root);
        return MGPU_CONTEXT_LEASE_QUARANTINED;
      }
      root.reset();
    }
    return result;
  };
  try {
    const auto source = copyContextLeasePin(original);
    if (!source || source->sourceDomain() || !source->contextGeneration() ||
        !source->deviceGeneration()) return MGPU_CONTEXT_LEASE_INVALID;
    if (!source->current()) return MGPU_CONTEXT_LEASE_RETIRED;
    if (funded && (!source->setupFunded() || !source->owner_->setupFunded()))
      return MGPU_CONTEXT_LEASE_UNSUPPORTED;
    MGPUCaptureSourceCpuMetadataProfileV1 metadataProfile{};
    const auto profile = mgpuCaptureSourceGetCpuMetadataProfileV1(&metadataProfile, sizeof(metadataProfile));
    if (profile != MGPU_CONTEXT_LEASE_OK) return profile;
    lease = funded ? setup_custody::makeSetupUnique<MGPUContextLease>(funding->roles[MGPU_CAPTURE_SETUP_SCOPED_LEASE]) :
        setup_custody::SetupUnique<MGPUContextLease>(new MGPUContextLease());
    auto result = funded ? source_scope::SourceRoot::createFunded(spec,
        funding->roles[MGPU_CAPTURE_SETUP_ROOT_STATE], funding->roles[MGPU_CAPTURE_SETUP_ROOT_BOX], root) :
        source_scope::SourceRoot::create(spec, root);
    if (result != source_scope::Result::Ok) return refuse(status(result));
    result = root->bindMetadataLayout(metadataProfile.layout);
    if (result != source_scope::Result::Ok) return refuse(MGPU_CONTEXT_LEASE_INTERNAL);
    const auto domain = root->domain();
    result = domain.reserve(source_scope::ResourceKind::ScopedPin, claim);
    if (result != source_scope::Result::Ok) return refuse(status(result));
    if (!reserve(liveLeases, kMaxLeases))
      return refuse(MGPU_CONTEXT_LEASE_WOULD_BLOCK);
    snapshot.charged = true;
    result = claim.accept();
    if (result != source_scope::Result::Ok) return refuse(status(result));
    entered = true;
    wgpuDeviceAddRef(source->device()); snapshot.device = source->device();
    wgpuQueueAddRef(source->queue()); snapshot.queue = source->queue();
    wgpuInstanceAddRef(source->instance()); snapshot.instance = source->instance();
    scoped = funded ? setup_custody::makeSetupShared<ContextPin>(funding->roles[MGPU_CAPTURE_SETUP_SCOPED_PIN],
        source->owner_, snapshot, source->deviceGeneration(), true, domain, std::move(claim)) :
        std::make_shared<ContextPin>(source->owner_, snapshot, source->deviceGeneration(), false, domain, std::move(claim));
    lease->pin = scoped;
    if (!scoped->current()) return refuse(MGPU_CONTEXT_LEASE_RETIRED);
    outRoot = std::move(root);
    *outLease = lease.release();
    return MGPU_CONTEXT_LEASE_OK;
  } catch (setup_custody::Error e) {
    if (e == setup_custody::Error::Uncertain) {
      if (root) root->domain().quarantine();
      const auto result = refuse(MGPU_CONTEXT_LEASE_QUARANTINED);
      return result;
    }
    return refuse(MGPU_CONTEXT_LEASE_INVALID);
  } catch (...) { return refuse(MGPU_CONTEXT_LEASE_INTERNAL); }
}
int createScopedContextLease(MGPUContextLease *original, const source_scope::SourceSpec &spec,
    source_scope::SourceRootOwner &outRoot, MGPUContextLease **outLease) noexcept {
  return createScopedContextLeaseImpl(original, spec, outRoot, outLease, nullptr);
}
int createScopedContextLeaseFunded(MGPUContextLease *original, const source_scope::SourceSpec &spec,
    const setup_custody::SetupTickets &funding, source_scope::SourceRootOwner &outRoot,
    MGPUContextLease **outLease) noexcept {
  return createScopedContextLeaseImpl(original, spec, outRoot, outLease, &funding);
}
namespace {
int retainDefault(MGPU &gpu, MGPUContextLease **outLease, MGPUContextHandle **outContext,
    const setup_custody::SetupTickets *funding) noexcept {
  if (outLease) *outLease = nullptr;
  if (outContext) *outContext = nullptr;
  if (!outLease || !outContext || registryClosing.load(std::memory_order_acquire)) return MGPU_CONTEXT_LEASE_INVALID;
  const bool funded = funding != nullptr;
  if (funded && !funding->valid()) return MGPU_CONTEXT_LEASE_INVALID;
  const auto ticket = [&](MGPUCaptureSetupKindV1 role) { return funded ? funding->roles[role] : setup_custody::SetupTicket{}; };
  setup_custody::PendingSetup ownerPending(ticket(MGPU_CAPTURE_SETUP_CONTEXT_OWNER)),
      pinPending(ticket(MGPU_CAPTURE_SETUP_ORIGINAL_PIN)), leasePending(ticket(MGPU_CAPTURE_SETUP_ORIGINAL_LEASE));
  try {
    auto &owners = registry();
    auto acquisition = owners.ready.acquire(nextGeneration);
    using Decision = detail::DefaultOwnerSlot<ContextOwner>::Decision;
    if (acquisition.decision == Decision::Busy || acquisition.decision == Decision::Exhausted)
      return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    if (acquisition.decision == Decision::Closed) return MGPU_CONTEXT_LEASE_RETIRED;
    struct CreationGuard {
      detail::DefaultOwnerSlot<ContextOwner> &slot; uint64_t generation;
      ~CreationGuard() { if (generation) { try { slot.abandon(generation); } catch (...) {} } }
    } creation{owners.ready, acquisition.decision == Decision::Create ? acquisition.generation : 0};
    acquisition.retired.reset(); // Actual expired block release before another allocation, outside lock.
    auto owner = std::move(acquisition.owner);
    if (owner && (&owner->gpu() != &gpu || (funded && !owner->setupFunded())))
      return MGPU_CONTEXT_LEASE_UNSUPPORTED;
    if (!owner) {
      detail::ReadyAttachment attachment(gpu);
      if (!reserve(liveContexts, kMaxContexts)) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      attachment.charged = true;
      if (!gpu.attachReadyContext()) return MGPU_CONTEXT_LEASE_NOT_READY;
      attachment.attached = true;
      owner = funded ? setup_custody::makeSetupShared<ContextOwner>(ticket(MGPU_CAPTURE_SETUP_CONTEXT_OWNER),
          attachment, acquisition.generation, true) :
          std::make_shared<ContextOwner>(attachment, acquisition.generation, false);
      if (!owners.ready.publish(acquisition.generation, owner)) return MGPU_CONTEXT_LEASE_RETIRED;
    }
    auto lease = funded ? setup_custody::makeSetupUnique<MGPUContextLease>(ticket(MGPU_CAPTURE_SETUP_ORIGINAL_LEASE)) :
        setup_custody::SetupUnique<MGPUContextLease>(new MGPUContextLease());
    const int result = retainPin(owner, lease->pin, ticket(MGPU_CAPTURE_SETUP_ORIGINAL_PIN));
    if (result != MGPU_CONTEXT_LEASE_OK) return result;
    *outLease = lease.release(); *outContext = reinterpret_cast<MGPUContextHandle *>(&gpu);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (setup_custody::Error e) {
    return e == setup_custody::Error::Uncertain ? MGPU_CONTEXT_LEASE_QUARANTINED : MGPU_CONTEXT_LEASE_INVALID;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
}
int retainReadyDefaultContext(MGPU &gpu, MGPUContextLease **lease, MGPUContextHandle **context) noexcept {
  return retainDefault(gpu, lease, context, nullptr);
}
int retainReadyDefaultContextFunded(MGPU &gpu, const setup_custody::SetupTickets &funding,
    MGPUContextLease **lease, MGPUContextHandle **context) noexcept {
  return retainDefault(gpu, lease, context, &funding);
}
} // namespace mgpu
extern "C" {
int mgpuContextRetainLease(MGPUContextHandle *context, MGPUContextLease **out) {
  if (!out)
    return MGPU_CONTEXT_LEASE_INVALID;
  *out = nullptr;
  try {
    auto lease = std::make_unique<MGPUContextLease>();
    const int result = mgpu::retainContextPin(context, lease->pin);
    if (result != MGPU_CONTEXT_LEASE_OK)
      return result;
    *out = lease.release();
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) {
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
}
int mgpuContextLeaseGetInfo(MGPUContextLease *lease, MGPUContextLeaseInfo *out,
                            uint32_t size) {
  if (!out || size < sizeof(*out))
    return MGPU_CONTEXT_LEASE_INVALID;
  std::memset(out, 0, sizeof(*out));
  if (!lease)
    return MGPU_CONTEXT_LEASE_INVALID;
  out->struct_size = sizeof(*out);
  out->version = 1;
  out->context_generation = lease->pin->contextGeneration();
  out->device_generation = lease->pin->deviceGeneration();
  out->current = lease->pin->current() ? 1 : 0;
  return MGPU_CONTEXT_LEASE_OK;
}
void mgpuContextReleaseLease(MGPUContextLease *lease) {
  if (lease && !lease->sourceManaged) mgpu::setup_custody::SetupDelete<MGPUContextLease>{}(lease);
}
int mgpuContextGetLifetimeStats(MGPUContextLifetimeStats *out, uint32_t size) {
  if (!out || size < sizeof(*out))
    return MGPU_CONTEXT_LEASE_INVALID;
  *out = {sizeof(*out),
          1,
          liveContexts.load(),
          liveLeases.load(),
          quarantinedContexts.load(),
          kMaxContexts,
          kMaxLeases,
          0};
  return MGPU_CONTEXT_LEASE_OK;
}
}
namespace mgpu::allocation_layout {
void contextSetupLayout(SetupLayoutBuilder &out) noexcept {
  out.unique<MGPUContextLease>(MGPU_CAPTURE_SETUP_ORIGINAL_LEASE);
  out.shared<ContextPin>(MGPU_CAPTURE_SETUP_ORIGINAL_PIN);
  out.shared<ContextOwner>(MGPU_CAPTURE_SETUP_CONTEXT_OWNER);
  out.unique<MGPUContextLease>(MGPU_CAPTURE_SETUP_SCOPED_LEASE);
  out.shared<ContextPin>(MGPU_CAPTURE_SETUP_SCOPED_PIN);
}
void fundedContextSetupLayout(FundedSetupLayoutBuilder &out) noexcept {
  out.unique<MGPUContextLease>(MGPU_CAPTURE_SETUP_ORIGINAL_LEASE);
  out.shared<ContextPin>(MGPU_CAPTURE_SETUP_ORIGINAL_PIN);
  out.shared<ContextOwner>(MGPU_CAPTURE_SETUP_CONTEXT_OWNER);
  out.unique<MGPUContextLease>(MGPU_CAPTURE_SETUP_SCOPED_LEASE);
  out.shared<ContextPin>(MGPU_CAPTURE_SETUP_SCOPED_PIN);
}
} // namespace mgpu::allocation_layout
