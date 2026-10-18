#ifndef MINIGPU_SETUP_ALLOCATOR_H
#define MINIGPU_SETUP_ALLOCATOR_H
// Private native mechanism only. Not an authority/guest ABI. The trusted bridge
// must retain the ORIGINAL budget and pin callback code independently. Copies
// retain that same authority; only its original ledger may settle a role.
#include "minigpu_allocation_extent.h"
#include "minigpu_setup_custody_abi.h"
#include <array>
#include <limits>
#include <new>
#include <utility>

namespace mgpu::setup_custody {
using Extent = allocation_layout::Extent;
struct Key { uint64_t generation = 0; uint32_t row = 0, role = 0; };
enum class Event : uint32_t {
  BeginAllocation, Allocated, Constructed, Disposed, BeginDeallocation,
  FreeReturned, AllocationFailed, CancelUnallocated, Quarantine
};
using AuthorityOps = MGPUSetupAuthorityOpsV1;
enum class Error { Refused, Uncertain };

class SetupTicket {
public:
  SetupTicket() noexcept = default;
  // Internal trusted caller keeps context/ops alive while acquiring this copy.
  // No new authority or native allocation follows from a key/extent alone.
  static SetupTicket retain(void *context, const AuthorityOps *ops, Key key, Extent extent) noexcept {
    SetupTicket result;
    if (!context || !ops || !ops->retain || !ops->release || !ops->event || !key.generation ||
        key.role >= 10 || !extent.bytes || extent.bytes > static_cast<uint64_t>((std::numeric_limits<ptrdiff_t>::max)()) ||
        !extent.alignment || (extent.alignment & (extent.alignment - 1)) || extent.bytes % extent.alignment) return result;
    result.context_ = context; result.ops_ = ops; result.key_ = key; result.extent_ = extent;
    ops->retain(context); return result;
  }
  SetupTicket(const SetupTicket &other) noexcept
      : context_(other.context_), ops_(other.ops_), key_(other.key_), extent_(other.extent_) {
    if (context_) ops_->retain(context_);
  }
  SetupTicket(SetupTicket &&other) noexcept
      : context_(std::exchange(other.context_, nullptr)), ops_(other.ops_), key_(other.key_), extent_(other.extent_) {}
  SetupTicket &operator=(SetupTicket other) noexcept { swap(other); return *this; }
  ~SetupTicket() { if (context_) ops_->release(context_); }
  void swap(SetupTicket &other) noexcept {
    std::swap(context_, other.context_); std::swap(ops_, other.ops_);
    std::swap(key_, other.key_); std::swap(extent_, other.extent_);
  }
  explicit operator bool() const noexcept { return context_ != nullptr; }
  bool same(const SetupTicket &other) const noexcept {
    return context_ == other.context_ && ops_ == other.ops_ && key_.generation == other.key_.generation &&
        key_.row == other.key_.row && key_.role == other.key_.role && extent_.bytes == other.extent_.bytes &&
        extent_.alignment == other.extent_.alignment;
  }
  bool hasRole(uint32_t role) const noexcept { return context_ && key_.role == role; }
  bool sameTransaction(const SetupTicket &other) const noexcept {
    return context_ && context_ == other.context_ && ops_ == other.ops_ &&
        key_.generation == other.key_.generation && key_.row == other.key_.row;
  }
  bool notify(Event event, uintptr_t address = 0) const noexcept {
    if (!context_) return false;
    const MGPUSetupEventRecordV1 record{sizeof(MGPUSetupEventRecordV1), 1,
        static_cast<uint32_t>(event), key_.role, key_.generation, key_.row,
        extent_.alignment, address, extent_.bytes};
    return ops_->event(context_, &record) == 1;
  }
  void quarantine() const noexcept { (void)notify(Event::Quarantine); }
  bool matches(size_t bytes, size_t alignment) const noexcept {
    return context_ && extent_.bytes == bytes && extent_.alignment == alignment;
  }
  void *allocateBytes(size_t bytes, size_t alignment) const {
    if (!matches(bytes, alignment) || !notify(Event::BeginAllocation)) throw Error::Refused;
    void *value;
    try {
      value = alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__ ?
          ::operator new(bytes, std::align_val_t(alignment)) : ::operator new(bytes);
    } catch (...) {
      if (!notify(Event::AllocationFailed)) quarantine();
      throw;
    }
    if (!notify(Event::Allocated, reinterpret_cast<uintptr_t>(value))) {
      quarantine(); // Unknown receipt: retain both the block and original debt.
      throw Error::Uncertain;
    }
    return value;
  }
  void deallocateBytes(void *value, size_t bytes, size_t alignment) const noexcept {
    const auto address = reinterpret_cast<uintptr_t>(value);
    if (!value || !matches(bytes, alignment) || !notify(Event::BeginDeallocation, address)) {
      quarantine(); return; // No free on uncertain custody.
    }
    // The invoking ticket copy outlives the block. No dead object is consulted.
    if (alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__) ::operator delete(value, std::align_val_t(alignment));
    else ::operator delete(value);
    if (!notify(Event::FreeReturned, address)) quarantine();
  }
private:
  void *context_ = nullptr;
  const AuthorityOps *ops_ = nullptr;
  Key key_{};
  Extent extent_{};
};

// Stack/inline routing only; S4 authenticates the original host transaction and
// full profile. Role numbers here follow MGPUCaptureSetupKindV1's native10.
struct SetupTickets {
  std::array<SetupTicket, 10> roles;
  bool valid() const noexcept {
    for (uint32_t i = 0; i < roles.size(); ++i)
      if (!roles[i].hasRole(i) || !roles[0].sameTransaction(roles[i])) return false;
    return true;
  }
};

class PendingSetup {
public:
  explicit PendingSetup(SetupTicket value) noexcept : ticket_(std::move(value)) {}
  // Host transition is allowed ONLY for Issued; constructed/terminal/stale roles
  // are a harmless refusal. This destructor is never a free or join receipt.
  ~PendingSetup() { (void)ticket_.notify(Event::CancelUnallocated); }
  PendingSetup(const PendingSetup &) = delete;
private:
  SetupTicket ticket_;
};

template<class T> class SetupAllocator {
public:
  using value_type = T;
  explicit SetupAllocator(SetupTicket ticket) noexcept : ticket_(std::move(ticket)) {}
  template<class U> SetupAllocator(const SetupAllocator<U> &other) noexcept : ticket_(other.ticket_) {}
  T *allocate(size_t count) {
    if (count != 1) throw Error::Refused;
    return static_cast<T *>(ticket_.allocateBytes(sizeof(T), alignof(T)));
  }
  void deallocate(T *value, size_t count) noexcept {
    ticket_.deallocateBytes(value, count == 1 ? sizeof(T) : 0, alignof(T));
  }
  template<class U, class... Args> void construct(U *value, Args &&...args) {
    static_assert(std::is_nothrow_destructible_v<U>, "Funded payload destructors must not throw");
    ::new (static_cast<void *>(value)) U(std::forward<Args>(args)...);
    if (!ticket_.notify(Event::Constructed)) {
      // Placement construction succeeded. STL will assume it did not when we
      // throw, so destroy the complete private object explicitly exactly once.
      ticket_.quarantine(); value->~U(); throw Error::Uncertain;
    }
  }
  template<class U> void destroy(U *value) noexcept {
    static_assert(std::is_nothrow_destructible_v<U>, "Funded payload destructors must not throw");
    value->~U();
    if (!ticket_.notify(Event::Disposed)) ticket_.quarantine();
  }
  template<class U> bool operator==(const SetupAllocator<U> &other) const noexcept { return ticket_.same(other.ticket_); }
  template<class U> bool operator!=(const SetupAllocator<U> &other) const noexcept { return !(*this == other); }
private:
  template<class> friend class SetupAllocator;
  SetupTicket ticket_;
};
template<class T> constexpr allocation_layout::Expected setupSharedExtent() noexcept {
  return allocation_layout::sharedExtent<T, SetupAllocator<T>>();
}
template<class T, class... Args> std::shared_ptr<T> makeSetupShared(SetupTicket ticket, Args &&...args) {
  PendingSetup pending(ticket);
  if (!ticket || setupSharedExtent<T>().status != allocation_layout::Status::Known) throw Error::Refused;
  return std::allocate_shared<T>(SetupAllocator<T>(std::move(ticket)), std::forward<Args>(args)...);
}

struct SetupUniqueRecord { SetupTicket setupTicket; };
template<class T> struct SetupDelete {
  void operator()(T *value) const noexcept {
    if (!value) return;
    const auto record = static_cast<const SetupUniqueRecord &>(*value);
    if (!record.setupTicket) { delete value; return; }
    SetupAllocator<T> allocator(record.setupTicket);
    allocator.destroy(value);
    allocator.deallocate(value, 1);
  }
};
template<class T> using SetupUnique = std::unique_ptr<T, SetupDelete<T>>;
template<class T, class... Args> SetupUnique<T> makeSetupUnique(SetupTicket ticket, Args &&...args) {
  if (!ticket) throw Error::Refused;
  PendingSetup pending(ticket);
  SetupAllocator<T> allocator(ticket);
  auto *value = allocator.allocate(1);
  try { allocator.construct(value, SetupUniqueRecord{ticket}, std::forward<Args>(args)...); }
  catch (...) { allocator.deallocate(value, 1); throw; }
  return SetupUnique<T>(value);
}
} // namespace mgpu::setup_custody
#endif
