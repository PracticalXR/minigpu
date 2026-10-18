#ifndef MINIGPU_CONTEXT_OWNER_H
#define MINIGPU_CONTEXT_OWNER_H

// Internal native ownership contract; never serialize these objects or
// pointers.
#include "buffer.h"
#include "minigpu_context_lifetime.h"
#include "minigpu_creation_identity.h"
#include "minigpu_inline_callable.h"
#include <atomic>
#include <memory>

namespace mgpu {
namespace capture_cleanup { class Family; class GlobalTaskCustody; }
namespace capture_work { class PreparedWait; }
namespace detail { struct ReadyAttachment; struct PinnedSnapshot; }
class ContextOwner {
public:
  ContextOwner(std::unique_ptr<MGPU> gpu, uint64_t generation);
  // Adoption finishes inside construction, before any allocation receipt.
  ContextOwner(detail::ReadyAttachment &, uint64_t generation, bool funded) noexcept;
  ~ContextOwner();
  MGPU &gpu() const { return *gpu_; }
  bool ownsContext() const { return ownedGpu_ != nullptr; }
  bool setupFunded() const noexcept { return setupFunded_; }
  uint64_t generation() const { return generation_; }
  bool retired() const { return retired_.load(std::memory_order_acquire); }
  void retire() { retired_.store(true, std::memory_order_release); }

private:
  std::unique_ptr<MGPU> ownedGpu_;
  MGPU *gpu_;
  uint64_t generation_;
  std::atomic<bool> retired_{false};
  bool setupFunded_ = false;
};

class ContextPin {
public:
  ContextPin(std::shared_ptr<ContextOwner>, detail::PinnedSnapshot &,
      uint64_t deviceGeneration, bool funded, source_scope::SourceDomain = {},
      source_scope::SourceClaim = {}) noexcept;
  ~ContextPin();
  ContextPin(const ContextPin &) = delete;
  ContextPin &operator=(const ContextPin &) = delete;
  bool current() const;
  bool setupFunded() const noexcept { return setupFunded_; }
  MGPU &gpu() const { return owner_->gpu(); }
  WGPUDevice device() const { return device_; }
  WGPUQueue queue() const { return queue_; }
  WGPUInstance instance() const { return instance_; }
  uint64_t contextGeneration() const { return owner_->generation(); }
  uint64_t deviceGeneration() const { return deviceGeneration_; }
  const source_scope::SourceDomain &sourceDomain() const noexcept { return source_; }
  RetainedCreationIdentity creationIdentity() const noexcept {
    return {&gpu(), device_, queue_, instance_, contextGeneration(),
            deviceGeneration_, source_};
  }

private:
  friend int createScopedContextLeaseImpl(
      MGPUContextLease *, const source_scope::SourceSpec &,
      source_scope::SourceRootOwner &, MGPUContextLease **,
      const setup_custody::SetupTickets *) noexcept;
  std::shared_ptr<ContextOwner> owner_;
  WGPUDevice device_;
  WGPUQueue queue_;
  WGPUInstance instance_;
  uint64_t deviceGeneration_;
  source_scope::SourceDomain source_;
  source_scope::SourceClaim sourcePin_;
  bool setupFunded_ = false;
};

MGPUContextHandle *createOwnedContext(const char *filter) noexcept;
void retireOwnedContext(MGPUContextHandle *context) noexcept;
std::shared_ptr<ContextOwner> retainContextOwner(MGPUContextHandle *context);
// Any registered, unretired context (owned or the default one), or null. Keeps
// its MGPU alive for one call without requiring a ready device, so a scope End
// still closes after a device loss and creation keeps its lazy initialization.
std::shared_ptr<ContextOwner> retainHandleOwner(MGPUContextHandle *context) noexcept;
int retainContextPin(MGPUContextHandle *context,
                     std::shared_ptr<ContextPin> &out) noexcept;
std::shared_ptr<ContextPin> copyContextLeasePin(MGPUContextLease *lease);
// Only the grouped native source owner adopts/releases this borrowed lease.
// Exclusive against use/final release. The ordinary public void Release refuses
// owner-managed leases; it must not bypass source join or delete a borrowed box.
bool adoptSourceContextLease(MGPUContextLease *, const source_scope::SourceDomain &) noexcept;
bool releaseSourceContextLease(MGPUContextLease *) noexcept;
// Internal preparation only, not an installed capture profile or public ABI.
// The original live unscoped lease must not race release. Creates a NEW source
// root/pin from that exact retained snapshot; never initializes or reselects a
// device, and never tags the original. outRoot must be empty. Ordinary refusal
// leaves no outputs. QUARANTINED can return a root custody marker without a lease;
// retain it. Successful join additionally requires the final scoped pin alias.
int createScopedContextLease(
    MGPUContextLease *original, const source_scope::SourceSpec &spec,
    source_scope::SourceRootOwner &outRoot,
    MGPUContextLease **outLease) noexcept;
int createScopedContextLeaseFunded(MGPUContextLease *, const source_scope::SourceSpec &,
    const setup_custody::SetupTickets &, source_scope::SourceRootOwner &,
    MGPUContextLease **) noexcept;
// Native-only exact-queue completion; retains pin through callback. Enqueue may
// throw before accepting work. Callback never runs inside a GPU lock/Dawn event.
void waitForPinnedSubmittedWork(std::shared_ptr<ContextPin> pin,
                                std::function<void(int)> complete);
// Internal retirement continuation. The caller prepares before changing owner
// phase, then enqueues this same family; no recursive public Work admission.
int preparePinnedSubmittedWorkWait(std::shared_ptr<ContextPin>,
    const capture_cleanup::Family &, InlineCallable<void(int)>,
    capture_work::PreparedWait &,
    std::shared_ptr<capture_cleanup::GlobalTaskCustody> = {}) noexcept;
int enqueuePinnedSubmittedWorkWait(std::shared_ptr<ContextPin>,
    capture_work::PreparedWait &) noexcept;
// Failed native destruction cannot release the last creation owner. Retention
// remains charged to the same bounded context/lease quotas for process life.
void quarantineContextPin(std::shared_ptr<ContextPin> pin) noexcept;
int retainReadyDefaultContext(MGPU &gpu, MGPUContextLease **outLease,
                              MGPUContextHandle **outContext) noexcept;
int retainReadyDefaultContextFunded(MGPU &, const setup_custody::SetupTickets &,
    MGPUContextLease **, MGPUContextHandle **) noexcept;
} // namespace mgpu
#endif
