#ifndef MINIGPU_CAPTURE_RECIPE_OWNER_H
#define MINIGPU_CAPTURE_RECIPE_OWNER_H

// Internal CPU policy/custody used by native factories. No GPU or public ABI.
#include "minigpu_capture_recipe_v1.h"
#include "minigpu_source_scope_owner.h"
#include <array>
#include <atomic>
#include <cstddef>

namespace mgpu::capture_recipe {
using source_scope::ResourceKind;
using source_scope::Result;
using source_scope::SourceDomain;

Result workStatus(const SourceDomain &) noexcept;
Result classifyResource(const SourceDomain &, uint64_t bytes, uint32_t usage,
                        const void *initial, uint32_t initialBytes,
                        ResourceKind &role) noexcept;
struct Binding {
  uint32_t slot = 0, kind = 0;
  SourceDomain source;
  ResourceKind role = ResourceKind::Count;
  uint64_t offset = 0, bytes = 0;
};
Result validateSetup(const SourceDomain &, const SourceDomain &shaderSource,
                     ResourceKind shaderRole, const char *wgsl, size_t length,
                     uint32_t frameSlot, const std::array<Binding, 2> &) noexcept;

// Use as a lifetime base: base destruction follows ALL derived State members,
// including its original pin and every retained alias. Native-release proof is
// separate from wrapper release; ordinary abandonment poisons, never refunds.
class ResourceCustody {
public:
  ResourceCustody() = default;
  ResourceCustody(ResourceCustody &&) noexcept;
  ~ResourceCustody();
  ResourceCustody(const ResourceCustody &) = delete;
  ResourceCustody &operator=(const ResourceCustody &) = delete;
  Result reserveSourceRole(const SourceDomain &, ResourceKind) noexcept;
  Result acceptSourceRole() noexcept;
  Result reserveResourceMetadata(source_scope::MetadataReservations &) noexcept;
  void proveNativeReleased() noexcept { released_.store(true, std::memory_order_release); }
  void quarantineSourceRole() noexcept;
  ResourceKind sourceRole() const noexcept { return role_; }
  const SourceDomain &resourceSource() const noexcept { return source_; }
protected:
  ResourceCustody(ResourceCustody &&, source_scope::MetadataAllocation) noexcept;
private:
  SourceDomain source_;
  source_scope::SourceClaim claim_;
  source_scope::MetadataAllocation metadata_;
  ResourceKind role_ = ResourceKind::Count;
  bool accepted_ = false;
  std::atomic<bool> released_{false}, uncertain_{false};
};

// Also a lifetime base, not a destructor-body refund. The final shared Job owns
// this until its caller and worker captures AND all derived members are gone.
// Prepared enqueue refusal proves the exact token was never linked; disposal
// follows caller locks. Unknown accepted work is never rolled back.
class JobCustody {
public:
  JobCustody() = default;
  ~JobCustody();
  JobCustody(const JobCustody &) = delete;
  JobCustody &operator=(const JobCustody &) = delete;
  Result prepareSourceJob(const SourceDomain &) noexcept;
  Result reserveJobMetadata(source_scope::MetadataReservations &) const noexcept;
  Result acceptSourceJob() noexcept;
  void proveJobFinished() noexcept { finished_.store(true, std::memory_order_release); }
  void proveEnqueueRejected() noexcept { proveJobFinished(); }
  void quarantineSourceJob() noexcept;
  void setGlobalJobRefund(bool (*refund)() noexcept) noexcept { globalRefund_ = refund; }
protected:
  JobCustody(JobCustody &&, source_scope::MetadataAllocation) noexcept;
private:
  SourceDomain source_;
  source_scope::SourceOperation operation_;
  source_scope::SourceClaim primary_;
  source_scope::MetadataAllocation metadata_;
  bool accepted_ = false;
  std::atomic<bool> finished_{false}, uncertain_{false};
  bool (*globalRefund_)() noexcept = nullptr;
};
} // namespace mgpu::capture_recipe

#endif
