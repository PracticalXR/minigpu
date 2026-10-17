#ifndef MINIGPU_CAPTURE_OUTPUT_OWNER_H
#define MINIGPU_CAPTURE_OUTPUT_OWNER_H
// CPU policy/custody shared by the actual output chain and its foreign seams.
#include "minigpu_capture_work_owner.h"
#include "minigpu_source_metadata_allocator.h"
#include <utility>

namespace mgpu::capture_output {
source_scope::Result validateWriter(const source_scope::SourceDomain &source,
    const source_scope::SourceDomain &packedSource, source_scope::ResourceKind,
    uint32_t width, uint32_t height, uint64_t bytes, uint64_t usage) noexcept;
source_scope::Result validateOutput(const source_scope::SourceDomain &source,
    const source_scope::SourceDomain &parentSource,
    uint32_t width, uint32_t height, uint64_t bytes) noexcept;

namespace detail { struct UncertainWriteCompletion {}; }
// Called only inside PreparedWork's completion boundary. A successful fence
// does not prove EndAccess: notify the caller, then propagate that uncertainty
// so the executor retains its actual closure/owners/global charge. No allocation
// or exception crosses the public C callback boundary.
template<class Complete>
void completeWrite(const capture_cleanup::Family &family, int result, Complete &&complete) {
  if (result == MGPU_CONTEXT_LEASE_QUARANTINED) family.quarantine();
  std::forward<Complete>(complete)(result);
  if (result == MGPU_CONTEXT_LEASE_QUARANTINED) throw detail::UncertainWriteCompletion{};
}

// Pre-native slot rollback only. Parent pin/role stays alive through this scope.
// A reserve callback must not throw after charging. Commit BEFORE native entry;
// accepted uncertainty then retains its separately anchored real alias owner.
class SlotReservation {
public:
  SlotReservation() = default;
  SlotReservation(const SlotReservation &) = delete;
  SlotReservation &operator=(const SlotReservation &) = delete;
  ~SlotReservation();
  source_scope::Result reserve(const source_scope::SourceDomain &,
      bool (*reserve)(void *), bool (*refund)(void *) noexcept, void *) noexcept;
  void commit() noexcept { refund_ = nullptr; }
private:
  source_scope::SourceDomain source_;
  bool (*refund_)(void *) noexcept = nullptr;
  void *context_ = nullptr;
};

// Lifetime base. The independent root is allocated before any native AddRef,
// transferred before native entry, and dropped only after known native/global
// disposition. A lost/uncertain public box cannot drop its separate COM refs.
template<class Owner>
class AliasCustody : public capture_recipe::ResourceCustody {
public:
  using Anchor = source_scope::MetadataAnchor<Owner>;
  using Root = source_scope::MetadataUnique<Anchor>;
  explicit AliasCustody(capture_recipe::ResourceCustody &&role) noexcept
      : capture_recipe::ResourceCustody(std::move(role)) {}
  AliasCustody(source_scope::MetadataAllocation allocation,
               capture_recipe::ResourceCustody &&role) noexcept
      : capture_recipe::ResourceCustody(std::move(role), std::move(allocation)) {}
  source_scope::Result acceptAlias(Root &root) noexcept {
    if (anchor_ || !root || root->owner.get() != static_cast<Owner *>(this))
      return source_scope::Result::Invalid;
    const auto status = acceptSourceRole();
    if (status == source_scope::Result::Ok) anchor_ = root.release();
    return status;
  }
  void releasedAlias() noexcept {
    proveNativeReleased();
    source_scope::MetadataDelete<Anchor>{}(std::exchange(anchor_, nullptr));
  }
  void quarantineAlias() noexcept { uncertain_ = true; quarantineSourceRole(); }
  bool uncertainAlias() const noexcept { return uncertain_.load(); }
private:
  Anchor *anchor_ = nullptr;
  std::atomic<bool> uncertain_{false};
};
} // namespace mgpu::capture_output
#endif
