#include "../include/minigpu_context_owner.h"
#include <array>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

namespace {
constexpr uint32_t kMaxContexts = 64;
constexpr uint32_t kMaxLeases = 256;
std::atomic<uint32_t> liveContexts{0}, liveLeases{0}, quarantinedContexts{0};
std::atomic<uint64_t> nextGeneration{1};

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

struct Registry {
  std::mutex mutex;
  std::array<std::shared_ptr<mgpu::ContextOwner>, kMaxContexts> active{};
  // The caller's lease and native consumers own this ready attachment. A weak
  // registry entry must not keep the display device attached indefinitely.
  std::weak_ptr<mgpu::ContextOwner> defaultContext;
};
Registry &registry() {
  // Construct retirement before registry so owners can drain during shutdown.
  retirementWorker();
  static Registry value;
  return value;
}

std::shared_ptr<mgpu::ContextOwner> findOwner(MGPUContextHandle *handle) {
  auto &owners = registry();
  std::lock_guard<std::mutex> lock(owners.mutex);
  for (auto &owner : owners.active) {
    if (owner && reinterpret_cast<MGPUContextHandle *>(&owner->gpu()) == handle)
      return owner;
  }
  auto display = owners.defaultContext.lock();
  if (display &&
      reinterpret_cast<MGPUContextHandle *>(&display->gpu()) == handle)
    return display;
  return {};
}
} // namespace

namespace mgpu {
void quarantineContextPin(std::shared_ptr<ContextPin> pin) noexcept {
  struct Quarantine {
    std::mutex mutex;
    std::array<std::shared_ptr<ContextPin>, kMaxLeases> pins{};
  };
  // Intentionally survives static destruction: an unsuccessful native drain
  // must not become a final-owner deletion when a library/thread shuts down.
  static auto *retained = new Quarantine;
  std::lock_guard<std::mutex> lock(retained->mutex);
  for (const auto &existing : retained->pins) {
    if (existing && existing->contextGeneration() == pin->contextGeneration() &&
        existing->deviceGeneration() == pin->deviceGeneration()) return;
  }
  for (auto &slot : retained->pins) {
    if (!slot) {
      slot = std::move(pin);
      ++quarantinedContexts;
      return;
    }
  }
  // Unreachable: every distinct pin here and the incoming one consume the
  // same kMaxLeases reservation, and duplicate generations returned above.
  std::terminate();
}

std::shared_ptr<ContextOwner> retainContextOwner(MGPUContextHandle *handle) {
  auto owner = findOwner(handle);
  // Owned-handle initialization must not repair a borrowed display device.
  return owner && owner->ownsContext() && !owner->retired() ? owner : nullptr;
}
ContextOwner::ContextOwner(std::unique_ptr<MGPU> gpu, uint64_t generation)
    : ownedGpu_(std::move(gpu)), gpu_(ownedGpu_.get()), generation_(generation) {}
ContextOwner::ContextOwner(MGPU &gpu, uint64_t generation) noexcept
    : gpu_(&gpu), generation_(generation) {}
ContextOwner::~ContextOwner() {
  if (ownedGpu_)
    retirementWorker().enqueue(ownedGpu_.release());
  else
    retirementWorker().enqueue(gpu_, false);
}
ContextPin::ContextPin(std::shared_ptr<ContextOwner> owner, WGPUDevice device,
                       WGPUQueue queue, WGPUInstance instance,
                       uint64_t generation)
    : owner_(std::move(owner)), device_(device), queue_(queue),
      instance_(instance), deviceGeneration_(generation) {}
ContextPin::~ContextPin() {
  wgpuQueueRelease(queue_);
  wgpuDeviceRelease(device_);
  wgpuInstanceRelease(instance_);
  --liveLeases;
}
bool ContextPin::current() const {
  return !owner_->retired() &&
         owner_->gpu().matchesReadyDevice(device_, deviceGeneration_);
}

MGPUContextHandle *createOwnedContext(const char *filter) noexcept {
  std::shared_ptr<ContextOwner> owner;
  bool charged = false;
  try {
    auto &owners = registry();
    if (!reserve(liveContexts, kMaxContexts))
      return nullptr;
    charged = true;
    auto gpu = std::make_unique<MGPU>();
    if (filter && filter[0])
      gpu->setAdapterFilter(filter);
    owner = std::make_shared<ContextOwner>(std::move(gpu), nextGeneration++);
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
  if (!handle)
    return;
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

int retainContextPin(MGPUContextHandle *handle,
                     std::shared_ptr<ContextPin> &out) noexcept {
  out.reset();
  WGPUDevice device = nullptr;
  WGPUQueue queue = nullptr;
  WGPUInstance instance = nullptr;
  bool charged = false;
  auto releaseSnapshot = [&] {
    if (queue)
      wgpuQueueRelease(queue);
    if (device)
      wgpuDeviceRelease(device);
    if (instance)
      wgpuInstanceRelease(instance);
  };
  try {
    if (!handle)
      return MGPU_CONTEXT_LEASE_INVALID;
    auto owner = findOwner(handle);
    if (!owner || owner->retired())
      return MGPU_CONTEXT_LEASE_RETIRED;
    uint64_t generation = 0;
    if (!owner->gpu().retainReadyContextSnapshot(&device, &queue, &instance,
                                                 &generation))
      return MGPU_CONTEXT_LEASE_NOT_READY;
    if (owner->retired()) {
      releaseSnapshot();
      return MGPU_CONTEXT_LEASE_RETIRED;
    }
    if (!reserve(liveLeases, kMaxLeases)) {
      releaseSnapshot();
      return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    }
    charged = true;
    out = std::make_shared<ContextPin>(std::move(owner), device, queue,
                                       instance, generation);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) {
    releaseSnapshot();
    if (charged)
      --liveLeases;
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
}
} // namespace mgpu

struct MGPUContextLease {
  std::shared_ptr<mgpu::ContextPin> pin;
};
namespace mgpu {
std::shared_ptr<ContextPin> copyContextLeasePin(MGPUContextLease *lease) {
  return lease ? lease->pin : nullptr;
}
int retainReadyDefaultContext(MGPU &gpu, MGPUContextLease **outLease,
                              MGPUContextHandle **outContext) noexcept {
  if (outLease) *outLease = nullptr;
  if (outContext) *outContext = nullptr;
  if (!outLease || !outContext) return MGPU_CONTEXT_LEASE_INVALID;
  std::shared_ptr<ContextOwner> owner;
  bool charged = false, attached = false;
  try {
    auto lease = std::make_unique<MGPUContextLease>();
    auto &owners = registry();
    {
      std::lock_guard<std::mutex> lock(owners.mutex);
      owner = owners.defaultContext.lock();
      if (owner && &owner->gpu() != &gpu) return MGPU_CONTEXT_LEASE_INVALID;
      if (!owner) {
        if (!reserve(liveContexts, kMaxContexts))
          return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
        charged = true;
        if (!gpu.attachReadyContext()) {
          --liveContexts;
          return MGPU_CONTEXT_LEASE_NOT_READY;
        }
        attached = true;
        owner = std::make_shared<ContextOwner>(gpu, nextGeneration++);
        owners.defaultContext = owner;
      }
    }
    auto *context = reinterpret_cast<MGPUContextHandle *>(&gpu);
    const int result = retainContextPin(context, lease->pin);
    if (result != MGPU_CONTEXT_LEASE_OK) return result;
    *outLease = lease.release();
    *outContext = context;
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) {
    if (charged && !owner) {
      if (attached)
        retirementWorker().enqueue(&gpu, false);
      else
        --liveContexts;
    }
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
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
void mgpuContextReleaseLease(MGPUContextLease *lease) { delete lease; }
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
