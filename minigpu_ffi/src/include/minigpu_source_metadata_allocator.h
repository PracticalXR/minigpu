#ifndef MINIGPU_SOURCE_METADATA_ALLOCATOR_H
#define MINIGPU_SOURCE_METADATA_ALLOCATOR_H
// Internal allocate_shared adapter. Allocation slots, including control blocks,
// survive payload destruction, retained weak references and deallocation itself.
#include "minigpu_source_scope_owner.h"
#include <new>

namespace mgpu::source_scope {
struct MetadataAllocationError { Result result; };
template<class T> class MetadataAllocator {
public:
  using value_type = T;
  MetadataAllocator(MetadataAllocation allocation,
      SourceOperationView operation = {}) noexcept
      : allocation_(std::move(allocation)), operation_(std::move(operation)) {}
  template<class U> MetadataAllocator(const MetadataAllocator<U> &other) noexcept
      : allocation_(other.allocation_), operation_(other.operation_) {}
  T *allocate(size_t count) {
    if (count != 1) throw MetadataAllocationError{Result::Invalid};
    return static_cast<T *>(allocation_.allocateBytes(sizeof(T), alignof(T)));
  }
  void deallocate(T *value, size_t count) noexcept {
    allocation_.deallocateBytes(value, count == 1 ? sizeof(T) : 0, alignof(T));
  }
  template<class U> bool operator==(const MetadataAllocator<U> &other) const noexcept {
    return allocation_.same(other.allocation_);
  }
  template<class U> bool operator!=(const MetadataAllocator<U> &other) const noexcept {
    return !(*this == other);
  }
private:
  template<class> friend class MetadataAllocator;
  MetadataAllocation allocation_;
  // FamilyState's allocator retains the operation beyond its payload members.
  // This is an inline shared owner, not another metadata allocation.
  SourceOperationView operation_;
};

template<class T> constexpr allocation_layout::Expected metadataSharedExtent() noexcept {
  return allocation_layout::sharedExtent<T, MetadataAllocator<T>>();
}

class PendingMetadata {
public:
  explicit PendingMetadata(MetadataAllocation allocation) noexcept
      : allocation_(std::move(allocation)) {}
  ~PendingMetadata() { (void)allocation_.cancelUnallocated(); }
  PendingMetadata(const PendingMetadata &) = delete;
private:
  MetadataAllocation allocation_;
};

// Covered shared records accept their allocation identity as the first argument.
// Mark only the complete CPU payload constructed: a throwing promise/member
// constructor remains known unpublished rollback, rather than false quarantine.
template<class T, class... Args>
std::shared_ptr<T> makeMetadataShared(MetadataReservations &slots, MetadataKind kind,
                                      Args &&...args) {
  auto allocation = slots.take(kind);
  if (!allocation) return std::make_shared<T>(MetadataAllocation{}, std::forward<Args>(args)...);
  PendingMetadata allocationRollback(allocation);
  auto value = std::allocate_shared<T>(MetadataAllocator<T>(allocation, slots.view()),
      allocation, std::forward<Args>(args)...);
  allocation.constructed();
  return value;
}

// An opaque box/anchor retains this inline identity after release() transfers
// the raw pointer. Its deleter copies the identity BEFORE destroying the payload.
struct MetadataUniqueRecord {
  MetadataAllocation allocation;
  SourceOperationView operation;
};

template<class T> struct MetadataDelete {
  void operator()(T *value) const noexcept {
    if (!value) return;
    const auto storage = static_cast<const MetadataUniqueRecord &>(*value);
    if (!storage.allocation) { delete value; return; }
    value->~T();
    storage.allocation.disposed(true); // Actual box/anchor memory, not native completion proof.
    MetadataAllocator<T>(storage.allocation, storage.operation).deallocate(value, 1);
  }
};
template<class T> using MetadataUnique = std::unique_ptr<T, MetadataDelete<T>>;
template<class T, class... Args>
MetadataUnique<T> makeMetadataUnique(MetadataReservations &slots, MetadataKind kind,
                                      Args &&...args) {
  auto allocation = slots.take(kind);
  MetadataUniqueRecord storage{allocation, slots.view()};
  if (!allocation) return MetadataUnique<T>(new T(std::move(storage), std::forward<Args>(args)...));
  PendingMetadata allocationRollback(allocation);
  MetadataAllocator<T> allocator(allocation, slots.view());
  auto *value = allocator.allocate(1);
  try { ::new (value) T(std::move(storage), std::forward<Args>(args)...); }
  catch (...) { allocator.deallocate(value, 1); throw; }
  allocation.constructed();
  return MetadataUnique<T>(value);
}
template<class T> struct MetadataAnchor : MetadataUniqueRecord {
  MetadataAnchor(MetadataUniqueRecord storage, std::shared_ptr<T> held) noexcept
      : MetadataUniqueRecord(std::move(storage)), owner(std::move(held)) {}
  std::shared_ptr<T> owner;
};
} // namespace mgpu::source_scope
#endif
