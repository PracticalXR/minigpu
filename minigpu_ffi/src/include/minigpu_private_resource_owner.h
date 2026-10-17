#ifndef MINIGPU_PRIVATE_RESOURCE_OWNER_H
#define MINIGPU_PRIVATE_RESOURCE_OWNER_H
#include "minigpu_context_owner.h"
#include "minigpu_capture_recipe_owner.h"
#include <array>
#include <atomic>

namespace mgpu {
struct PrivateResourceState : capture_recipe::ResourceCustody {
  PrivateResourceState(source_scope::MetadataAllocation allocation,
                       capture_recipe::ResourceCustody &&role) noexcept
      : ResourceCustody(std::move(role), std::move(allocation)) {}
  std::shared_ptr<ContextPin> pin;
  // Scoped shader configuration retains source role aliases through its final
  // State destruction, including if a buffer wrapper releases before shader
  // bind groups/pipelines. No source slot is recycled under those native refs.
  std::array<std::shared_ptr<PrivateResourceState>, 4> bindingHolds{};
  uint64_t bytes = 0;
  uint32_t kind = 0, slot = 0;
  std::atomic<bool> ready{false}, quarantined{false}, configured{false};
  bool nativeReleased = false; // Exclusive creation/cleanup worker only.
};
inline bool samePrivatePin(const PrivateResourceState &s, const ContextPin &pin) {
  return s.pin && samePrivateCreation(s.pin->creationIdentity(), pin.creationIdentity());
}
void releasePrivateResource(const std::shared_ptr<PrivateResourceState> &) noexcept;
struct PrivateResourceAccess;
} // namespace mgpu
#endif
