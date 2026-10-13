#ifndef MINIGPU_CONTEXT_OWNER_H
#define MINIGPU_CONTEXT_OWNER_H

// Internal native ownership contract; never serialize these objects or
// pointers.
#include "buffer.h"
#include "minigpu_context_lifetime.h"
#include <atomic>
#include <memory>

namespace mgpu {
class ContextOwner {
public:
  ContextOwner(std::unique_ptr<MGPU> gpu, uint64_t generation);
  // The caller has already charged and retained one ready attachment.
  ContextOwner(MGPU &attachedGpu, uint64_t generation) noexcept;
  ~ContextOwner();
  MGPU &gpu() const { return *gpu_; }
  bool ownsContext() const { return ownedGpu_ != nullptr; }
  uint64_t generation() const { return generation_; }
  bool retired() const { return retired_.load(std::memory_order_acquire); }
  void retire() { retired_.store(true, std::memory_order_release); }

private:
  std::unique_ptr<MGPU> ownedGpu_;
  MGPU *gpu_;
  uint64_t generation_;
  std::atomic<bool> retired_{false};
};

class ContextPin {
public:
  ContextPin(std::shared_ptr<ContextOwner> owner, WGPUDevice device,
             WGPUQueue queue, WGPUInstance instance, uint64_t deviceGeneration);
  ~ContextPin();
  bool current() const;
  MGPU &gpu() const { return owner_->gpu(); }
  WGPUDevice device() const { return device_; }
  WGPUQueue queue() const { return queue_; }
  WGPUInstance instance() const { return instance_; }
  uint64_t contextGeneration() const { return owner_->generation(); }
  uint64_t deviceGeneration() const { return deviceGeneration_; }

private:
  std::shared_ptr<ContextOwner> owner_;
  WGPUDevice device_;
  WGPUQueue queue_;
  WGPUInstance instance_;
  uint64_t deviceGeneration_;
};

MGPUContextHandle *createOwnedContext(const char *filter) noexcept;
void retireOwnedContext(MGPUContextHandle *context) noexcept;
std::shared_ptr<ContextOwner> retainContextOwner(MGPUContextHandle *context);
int retainContextPin(MGPUContextHandle *context,
                     std::shared_ptr<ContextPin> &out) noexcept;
std::shared_ptr<ContextPin> copyContextLeasePin(MGPUContextLease *lease);
// Failed native destruction cannot release the last creation owner. Retention
// remains charged to the same bounded context/lease quotas for process life.
void quarantineContextPin(std::shared_ptr<ContextPin> pin) noexcept;
int retainReadyDefaultContext(MGPU &gpu, MGPUContextLease **outLease,
                              MGPUContextHandle **outContext) noexcept;
} // namespace mgpu
#endif
