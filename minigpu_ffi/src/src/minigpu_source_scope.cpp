#include "../include/minigpu_source_scope_owner.h"
#include "../include/minigpu_metadata_layout.h"
#include "../include/minigpu_setup_layout.h"
#include "../include/minigpu_source_metadata_allocator.h"
#include "../include/minigpu_module_owner.h"
#include <atomic>
#include <limits>
#include <mutex>
#include <utility>

namespace mgpu::source_scope {
namespace detail {
enum class AllocationPhase { Empty, Reserved, Allocating, Allocated, Deallocating, Quarantined };
struct MetadataSlot {
  uint64_t identity = 0, operation = 0;
  AllocationPhase phase = AllocationPhase::Empty;
  void *address = nullptr;
  size_t bytes = 0, alignment = 0;
  bool constructed = false, disposed = false, known = false;
};
struct State {
  State(SourceSpec value, uint64_t bytes) noexcept
      : spec(value), frameBytes(bytes), payloadLimit(5 * bytes + 16) {}
  const SourceSpec spec;
  const uint64_t frameBytes;
  const uint64_t payloadLimit;
  uint64_t identity = 0; // Assigned once, before registry publication.
  size_t registrySlot = 0;
  std::atomic<bool> released{true};
  std::atomic<bool> poisoned{false};
  std::mutex mutex;
  Phase phase = Phase::Open;
  uint64_t payloadBytes = 0;
  std::array<uint32_t, resourceKinds> resources{};
  uint32_t operations = 0;
  uint64_t nextOperation = 1;
  uint64_t activeOperation = 0;
  uint32_t tickets = 0;
  std::array<uint32_t, continuationKinds> continuations{};
  uint64_t nextMetadata = 1;
  std::array<MetadataSlot, metadataSlots> metadata{};
  bool admissionStarted = false, metadataLayoutBound = false;
  MGPUCaptureMetadataLayoutV1 metadataLayout{};
  uint64_t metadataAllocationBytes = 0;
  std::array<uint64_t, metadataKinds> metadataBytesByKind{};
};

struct OperationState {
  OperationState(std::shared_ptr<State> value, uint64_t id,
                 MetadataAllocation allocation) noexcept
      : root(std::move(value)), identity(id), admitted(true), metadata(std::move(allocation)) {
    metadata.constructed();
  }
  ~OperationState() {
    bool known = false;
    try {
      std::lock_guard<std::mutex> lock(root->mutex);
      known = admitted && finished && !unknown.load(std::memory_order_acquire) &&
          root->operations == 1 && root->activeOperation == identity;
    } catch (...) {
      root->poisoned.store(true, std::memory_order_release);
    }
    // The allocator returns capacity after actual storage/control-block release,
    // not while this destructor (or a retained weak control block) still exists.
    metadata.disposed(known);
  }
  std::shared_ptr<State> root;
  uint64_t identity = 0;
  bool admitted = false;
  bool finished = false; // Protected by root->mutex.
  std::atomic<bool> unknown{false};
  MetadataAllocation metadata;
};
} // namespace detail

allocation_layout::Expected operationStateAllocationExtent() noexcept {
  return metadataSharedExtent<detail::OperationState>();
}

namespace {
constexpr std::array<uint32_t, resourceKinds> resourceCaps{
    1, 1, 1, 1, 1, 2, 1, 2, 8, 1};
struct Registry {
  std::mutex mutex;
  std::array<std::shared_ptr<detail::State>, maxSources> roots{};
  uint64_t nextIdentity = 1;
};
Registry &registry() {
  // Failed/abandoned custody must not become an implicit refund at shutdown.
  return mgpu::detail::moduleOwner<Registry>();
}
bool geometry(const SourceSpec &spec, uint64_t &bytes) noexcept {
  if (!spec.width || !spec.height || spec.recipeRevision != 1 || spec.reserved)
    return false;
  const uint64_t row = static_cast<uint64_t>(spec.width) * 4;
  if (row > maxFrameBytes || spec.height > maxFrameBytes / row) return false;
  bytes = row * spec.height;
  return true;
}
uint64_t resourceBytes(ResourceKind kind, uint64_t frameBytes) noexcept {
  switch (kind) {
  case ResourceKind::ImportedFrame:
  case ResourceKind::PackedBuffer:
  case ResourceKind::Writer:
  case ResourceKind::Output: return frameBytes;
  case ResourceKind::Parameters: return 16;
  default: return 0;
  }
}
Result workAdmission(const detail::State &state) noexcept {
  if (state.poisoned.load(std::memory_order_acquire)) return Result::Poisoned;
  if (state.phase == Phase::Joined) return Result::Joined;
  if (state.phase == Phase::Sealed) return Result::Sealed;
  return Result::Ok;
}
void poison(const std::shared_ptr<detail::State> &state) noexcept {
  if (state) state->poisoned.store(true, std::memory_order_release);
}
detail::MetadataSlot *slotFor(detail::State &state, MetadataKind kind, uint64_t id) noexcept {
  const auto index = static_cast<size_t>(kind);
  if (index >= metadataKinds || !id) return nullptr;
  const auto offset = metadataOffset(kind);
  for (size_t i = 0; i < metadataCapacity(kind); ++i)
    if (state.metadata[offset + i].identity == id) return &state.metadata[offset + i];
  return nullptr;
}
detail::MetadataSlot *freeSlot(detail::State &state, MetadataKind kind) noexcept {
  const auto offset = metadataOffset(kind);
  for (size_t i = 0; i < metadataCapacity(kind); ++i)
    if (state.metadata[offset + i].phase == detail::AllocationPhase::Empty)
      return &state.metadata[offset + i];
  return nullptr;
}
bool finishAllocation(detail::State &state, MetadataKind kind,
                      detail::MetadataSlot &slot) noexcept {
  if (kind == MetadataKind::OperationState) {
    if (state.operations != 1 || state.activeOperation != slot.operation) return false;
  }
  const auto index = static_cast<size_t>(kind);
  if (state.metadataLayoutBound && (index >= metadataKinds ||
      state.metadataAllocationBytes < slot.bytes || state.metadataBytesByKind[index] < slot.bytes)) return false;
  // Validate all balances/lineage before changing either counts or byte debt.
  if (kind == MetadataKind::OperationState) {
    state.operations = 0;
    state.activeOperation = 0;
  }
  if (state.metadataLayoutBound) {
    state.metadataAllocationBytes -= slot.bytes;
    state.metadataBytesByKind[index] -= slot.bytes;
  }
  const auto identity = slot.identity;
  slot = {};
  slot.identity = identity; // Stale cancellation cannot cancel a later identity.
  return true;
}
void freeMetadata(void *value, size_t alignment) noexcept {
  if (alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    ::operator delete(value, std::align_val_t(alignment));
  else ::operator delete(value);
}
} // namespace

#ifdef MINIGPU_SOURCE_METADATA_TEST
void testMetadataBeforeAllocate(MetadataKind, size_t bytes, size_t alignment);
void testMetadataBeforeDeallocate(MetadataKind) noexcept;
#endif
void *MetadataAllocation::allocateBytes(size_t bytes, size_t alignment) const {
  if (!state_ || !bytes || !alignment || (alignment & (alignment - 1)))
    throw MetadataAllocationError{Result::Invalid};
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto *slot = slotFor(*state_, kind_, identity_);
    if (!slot || slot->phase != detail::AllocationPhase::Reserved)
      throw MetadataAllocationError{Result::Invalid};
    if (state_->metadataLayoutBound) {
      const auto index = static_cast<size_t>(kind_);
      const auto &entry = state_->metadataLayout.kinds[index];
      bool matches = false;
      for (uint32_t n = 0; n < entry.extent_count; ++n)
        matches = matches || (entry.extents[n].bytes == bytes && entry.extents[n].alignment == alignment);
      if (!matches) throw MetadataAllocationError{Result::Invalid};
      const auto kindLimit = allocation_layout::metadataKindByteLimit(entry);
      const auto totalLimit = state_->metadataLayout.total_requested_bytes;
      if (state_->metadataBytesByKind[index] > kindLimit || state_->metadataAllocationBytes > totalLimit) {
        poison(state_); throw MetadataAllocationError{Result::Internal};
      }
      if (bytes > kindLimit - state_->metadataBytesByKind[index] || bytes > totalLimit - state_->metadataAllocationBytes)
        throw MetadataAllocationError{Result::WouldBlock};
      state_->metadataAllocationBytes += bytes;
      state_->metadataBytesByKind[index] += bytes;
    }
    slot->phase = detail::AllocationPhase::Allocating;
    slot->bytes = bytes; slot->alignment = alignment;
  }
  void *value = nullptr;
  try {
#ifdef MINIGPU_SOURCE_METADATA_TEST
    testMetadataBeforeAllocate(kind_, bytes, alignment);
#endif
    value = alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__
        ? ::operator new(bytes, std::align_val_t(alignment)) : ::operator new(bytes);
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto *slot = slotFor(*state_, kind_, identity_);
    if (!slot || slot->phase != detail::AllocationPhase::Allocating)
      throw MetadataAllocationError{Result::Invalid};
    slot->address = value; slot->phase = detail::AllocationPhase::Allocated;
    return value;
  } catch (...) {
    // No constructor/native call has run. This is a proven allocation rollback.
    if (value) freeMetadata(value, alignment);
    (void)cancelUnallocated();
    throw;
  }
}
Result MetadataAllocation::cancelUnallocated() const noexcept {
  if (!state_) return Result::Invalid;
  try {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto *slot = slotFor(*state_, kind_, identity_);
    if (!slot) return Result::Invalid;
    if (slot->phase == detail::AllocationPhase::Empty) return Result::Ok;
    if (slot->phase != detail::AllocationPhase::Reserved &&
        slot->phase != detail::AllocationPhase::Allocating) return Result::Invalid;
    if (!slot->address && finishAllocation(*state_, kind_, *slot)) return Result::Ok;
    poison(state_);
    return Result::Internal;
  } catch (...) { poison(state_); return Result::Internal; }
}
void MetadataAllocation::constructed() const noexcept {
  if (!state_) return;
  try {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto *slot = slotFor(*state_, kind_, identity_);
    if (!slot || slot->phase != detail::AllocationPhase::Allocated || slot->constructed)
      poison(state_);
    else slot->constructed = true;
  } catch (...) { poison(state_); }
}
void MetadataAllocation::disposed(bool known) const noexcept {
  if (!state_) return;
  try {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto *slot = slotFor(*state_, kind_, identity_);
    if (!slot || slot->phase != detail::AllocationPhase::Allocated || slot->disposed) {
      poison(state_); return;
    }
    // A base/member destructor may run while a pure CPU derived constructor is
    // unwinding. Its factory has not marked the complete payload constructed.
    if (!slot->constructed) return;
    slot->disposed = true; slot->known = known;
    if (!known) poison(state_);
  } catch (...) { poison(state_); }
}
void MetadataAllocation::deallocateBytes(void *value, size_t bytes, size_t alignment) const noexcept {
  if (!state_) return;
  try {
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      auto *slot = slotFor(*state_, kind_, identity_);
      if (!slot || slot->phase != detail::AllocationPhase::Allocated ||
          slot->address != value || slot->bytes != bytes || slot->alignment != alignment) {
        poison(state_); return;
      }
      slot->phase = detail::AllocationPhase::Deallocating;
    }
#ifdef MINIGPU_SOURCE_METADATA_TEST
    testMetadataBeforeDeallocate(kind_);
#endif
    freeMetadata(value, alignment);
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto *slot = slotFor(*state_, kind_, identity_);
    if (!slot || slot->phase != detail::AllocationPhase::Deallocating) {
      poison(state_); return;
    }
    slot->address = nullptr;
    // An unconstructed payload proves constructor failure, before publication.
    if ((!slot->constructed || (slot->disposed && slot->known)) &&
        finishAllocation(*state_, kind_, *slot)) return;
    slot->phase = detail::AllocationPhase::Quarantined;
    poison(state_);
  } catch (...) { poison(state_); }
}

SourceDomain::SourceDomain(std::shared_ptr<detail::State> state) noexcept
    : state_(std::move(state)) {}
bool sameSourceDomain(const SourceDomain &left, const SourceDomain &right) noexcept {
  return left.state_ == right.state_;
}
void SourceDomain::quarantine() const noexcept {
  if (!state_) return;
  try {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->released.load(std::memory_order_acquire)) poison(state_);
  } catch (...) {
    poison(state_);
  }
}
Result SourceDomain::snapshot(Snapshot &out) const noexcept {
  out = Snapshot{};
  if (!state_) return Result::Invalid;
  try {
    std::lock_guard<std::mutex> lock(state_->mutex);
    out.identity = state_->identity;
    out.spec = state_->spec;
    out.phase = state_->poisoned.load(std::memory_order_acquire)
                    ? Phase::Poisoned : state_->phase;
    out.frameBytes = state_->frameBytes;
    out.payloadLimit = state_->payloadLimit;
    out.payloadBytes = state_->payloadBytes;
    out.resources = state_->resources;
    out.operationFamilies = state_->operations;
    out.completionTickets = state_->tickets;
    out.continuations = state_->continuations;
    out.metadataLayoutBound = state_->metadataLayoutBound;
    out.metadataAllocationLimit = state_->metadataLayoutBound ? state_->metadataLayout.total_requested_bytes : 0;
    out.metadataAllocationBytes = state_->metadataAllocationBytes;
    out.metadataBytesByKind = state_->metadataBytesByKind;
    for (size_t i = 0; i < metadataKinds; ++i) {
      const auto kind = static_cast<MetadataKind>(i);
      for (size_t j = 0; j < metadataCapacity(kind); ++j)
        out.metadataAllocations[i] += state_->metadata[metadataOffset(kind) + j].phase != detail::AllocationPhase::Empty;
    }
    return Result::Ok;
  } catch (...) {
    poison(state_);
    return Result::Internal;
  }
}
Result SourceDomain::reserve(ResourceKind kind, SourceClaim &out) const noexcept {
  const auto index = static_cast<size_t>(kind);
  if (!state_ || out || index >= resourceKinds) return Result::Invalid;
  try {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto admitted = workAdmission(*state_);
    if (admitted != Result::Ok) return admitted;
    const auto bytes = resourceBytes(kind, state_->frameBytes);
    if (state_->resources[index] >= resourceCaps[index] ||
        bytes > state_->payloadLimit - state_->payloadBytes)
      return Result::WouldBlock;
    ++state_->resources[index];
    state_->admissionStarted = true;
    state_->payloadBytes += bytes;
    out.state_ = state_;
    out.kind_ = SourceClaim::Kind::Resource;
    out.index_ = index;
    out.bytes_ = bytes;
    return Result::Ok;
  } catch (...) {
    poison(state_);
    return Result::Internal;
  }
}
Result SourceDomain::beginOperation(OperationPurpose purpose,
                                    SourceOperation &out) const noexcept {
  if (!state_ || out ||
      (purpose != OperationPurpose::Work && purpose != OperationPurpose::Cleanup))
    return Result::Invalid;
  MetadataAllocation allocation;
  uint64_t identity = 0;
  try {
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      if (state_->phase == Phase::Joined) return Result::Joined;
      if (purpose == OperationPurpose::Work) {
        const auto admitted = workAdmission(*state_);
        if (admitted != Result::Ok) return admitted;
      }
      // Cleanup is still possible after Seal/Poison, but cannot reuse a family
      // with unknown tails. It never permits new backing resource allocation.
      if (state_->operations ||
          state_->metadata[0].phase != detail::AllocationPhase::Empty ||
          state_->nextOperation == std::numeric_limits<uint64_t>::max() ||
          state_->nextMetadata == std::numeric_limits<uint64_t>::max())
        return Result::WouldBlock;
      identity = state_->nextOperation++;
      auto &slot = state_->metadata[0];
      slot = {};
      slot.identity = state_->nextMetadata++;
      slot.operation = identity;
      slot.phase = detail::AllocationPhase::Reserved;
      allocation = MetadataAllocation(state_, MetadataKind::OperationState, slot.identity);
      state_->operations = 1;
      state_->admissionStarted = true;
      state_->activeOperation = identity;
    }
    out.state_ = std::allocate_shared<detail::OperationState>(
        MetadataAllocator<detail::OperationState>(allocation), state_, identity, allocation);
    return Result::Ok;
  } catch (const MetadataAllocationError &error) {
    (void)allocation.cancelUnallocated();
    return error.result;
  } catch (...) {
    (void)allocation.cancelUnallocated();
    return Result::Internal;
  }
}

SourceClaim::~SourceClaim() { abandon(); }
SourceClaim::SourceClaim(SourceClaim &&other) noexcept { *this = std::move(other); }
SourceClaim &SourceClaim::operator=(SourceClaim &&other) noexcept {
  if (this == &other) return *this;
  abandon();
  state_ = std::move(other.state_);
  operation_ = std::move(other.operation_);
  kind_ = std::exchange(other.kind_, Kind::Empty);
  index_ = std::exchange(other.index_, 0);
  bytes_ = std::exchange(other.bytes_, 0);
  accepted_ = std::exchange(other.accepted_, false);
  metadataIssued_ = std::exchange(other.metadataIssued_, false);
  return *this;
}
void SourceClaim::abandon() noexcept {
  if (!state_) return;
  poison(state_);
  if (operation_) operation_->unknown.store(true, std::memory_order_release);
  operation_.reset();
  state_.reset();
  kind_ = Kind::Empty;
  index_ = 0;
  bytes_ = 0;
  accepted_ = false;
  metadataIssued_ = false;
}
Result SourceClaim::accept() noexcept {
  if (!state_ || accepted_) return Result::Invalid;
  try {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (kind_ == Kind::Resource) {
      const auto admitted = workAdmission(*state_);
      if (admitted != Result::Ok) return admitted;
    } else if (!operation_ || operation_->unknown.load(std::memory_order_acquire) ||
               state_->activeOperation != operation_->identity) {
      return Result::Invalid;
    }
    accepted_ = true;
    return Result::Ok;
  } catch (...) {
    poison(state_);
    return Result::Internal;
  }
}
Result SourceClaim::cancelUnaccepted() noexcept { return finish(true); }
Result SourceClaim::retire() noexcept { return finish(false); }
Result SourceClaim::reserveResourceMetadata(MetadataReservations &out) noexcept {
  if (!state_ || kind_ != Kind::Resource || accepted_ || metadataIssued_ || out.scoped() ||
      index_ >= resourceKinds) return Result::Invalid;
  try {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto admitted = workAdmission(*state_);
    if (admitted != Result::Ok) return admitted;
    const auto role = static_cast<ResourceKind>(index_);
    if (!state_->resources[index_] || bytes_ != resourceBytes(role, state_->frameBytes)) return Result::Invalid;
    std::array<MetadataKind, 3> kinds{};
    size_t count = 2;
    switch (role) {
    case ResourceKind::ImportedFrame:
      kinds = {MetadataKind::ImportedFrameState, MetadataKind::ImportedFrameNative, MetadataKind::ImportedFrameBox}; count = 3; break;
    case ResourceKind::RecipeShader: kinds = {MetadataKind::PrivateResourceState, MetadataKind::ShaderCpu}; break;
    case ResourceKind::PackedBuffer: case ResourceKind::Parameters:
      kinds = {MetadataKind::PrivateResourceState, MetadataKind::BufferCpu}; break;
    case ResourceKind::ExclusiveUse: kinds = {MetadataKind::ExclusiveUseState, MetadataKind::ExclusiveUseBox}; break;
    case ResourceKind::Writer: kinds = {MetadataKind::WriterState, MetadataKind::WriterBox}; break;
    case ResourceKind::Output: kinds = {MetadataKind::OutputState, MetadataKind::OutputBox}; break;
    case ResourceKind::PreviewLoan:
      kinds = {MetadataKind::LoanState, MetadataKind::LoanBox, MetadataKind::LoanAnchor}; count = 3; break;
    case ResourceKind::DescriptorBorrow:
      kinds = {MetadataKind::BorrowState, MetadataKind::BorrowBox, MetadataKind::BorrowAnchor}; count = 3; break;
    default: return Result::Invalid;
    }
    std::array<detail::MetadataSlot *, 3> selected{};
    for (size_t i = 0; i < count; ++i) {
      selected[i] = freeSlot(*state_, kinds[i]);
      if (!selected[i]) return Result::WouldBlock;
    }
    if (count >= std::numeric_limits<uint64_t>::max() - state_->nextMetadata)
      return Result::WouldBlock;
    for (size_t i = 0; i < count; ++i) {
      const auto kind = kinds[i];
      auto &slot = *selected[i];
      slot = {};
      slot.identity = state_->nextMetadata++;
      slot.phase = detail::AllocationPhase::Reserved;
      out.allocations_[static_cast<size_t>(kind)] = MetadataAllocation(state_, kind, slot.identity);
    }
    out.root_ = state_; // Deliberately no operation view for resource lifetime.
    metadataIssued_ = true;
    return Result::Ok;
  } catch (...) { poison(state_); return Result::Internal; }
}
Result SourceClaim::finish(bool rollback) noexcept {
  if (!state_ || rollback == accepted_) return Result::Invalid;
  try {
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      uint32_t *count = nullptr;
      if (kind_ == Kind::Resource && index_ < resourceKinds)
        count = &state_->resources[index_];
      else if (kind_ == Kind::Continuation && index_ < continuationKinds)
        count = &state_->continuations[index_];
      else if (kind_ == Kind::Ticket)
        count = &state_->tickets;
      if (!count || !*count || bytes_ > state_->payloadBytes) {
        poison(state_);
        return Result::Internal;
      }
      --*count;
      state_->payloadBytes -= bytes_;
    }
    // The last operation-state destructor also locks the root. Never drop that
    // tail under its mutex, even when callback completion and release coincide.
    operation_.reset();
    state_.reset();
    kind_ = Kind::Empty;
    index_ = 0;
    bytes_ = 0;
    accepted_ = false;
    metadataIssued_ = false;
    return Result::Ok;
  } catch (...) {
    poison(state_);
    return Result::Internal;
  }
}

SourceOperationView::SourceOperationView(
    std::shared_ptr<detail::OperationState> state) noexcept : state_(std::move(state)) {}
MetadataReservations::~MetadataReservations() {
  for (const auto &allocation : allocations_) (void)allocation.cancelUnallocated();
}
MetadataAllocation MetadataReservations::take(MetadataKind kind) {
  if (!scoped()) return {};
  const auto index = static_cast<size_t>(kind);
  if (index >= metadataKinds || !allocations_[index])
    throw MetadataAllocationError{Result::Invalid};
  return std::move(allocations_[index]);
}
Result SourceOperationView::reserveMetadata(std::initializer_list<MetadataKind> kinds,
                                          MetadataReservations &out) const noexcept {
  if (!state_ || out.scoped() || !kinds.size()) return Result::Invalid;
  const auto &root = state_->root;
  try {
    std::lock_guard<std::mutex> lock(root->mutex);
    if (!state_->admitted || state_->finished || state_->unknown.load() ||
        root->operations != 1 || root->activeOperation != state_->identity)
      return Result::Invalid;
    std::array<bool, metadataKinds> requested{};
    for (auto kind : kinds) {
      const auto index = static_cast<size_t>(kind);
      if (index == 0 || index >= operationMetadataKinds || requested[index]) return Result::Invalid;
      requested[index] = true;
      if (root->metadata[metadataOffset(kind)].phase != detail::AllocationPhase::Empty) return Result::WouldBlock;
    }
    if (kinds.size() >= std::numeric_limits<uint64_t>::max() - root->nextMetadata)
      return Result::WouldBlock;
    for (auto kind : kinds) {
      auto &slot = root->metadata[metadataOffset(kind)];
      slot = {};
      slot.identity = root->nextMetadata++;
      slot.operation = state_->identity;
      slot.phase = detail::AllocationPhase::Reserved;
      out.allocations_[static_cast<size_t>(kind)] = MetadataAllocation(root, kind, slot.identity);
    }
    out.view_ = *this;
    out.root_ = root;
    return Result::Ok;
  } catch (...) { poison(root); return Result::Internal; }
}
Result SourceOperationView::reserveFamilyMetadata(MetadataAllocation &out) const noexcept {
  if (!state_ || out.state_) return Result::Invalid;
  const auto &root = state_->root;
  try {
    std::lock_guard<std::mutex> lock(root->mutex);
    if (!state_->admitted || state_->finished || state_->unknown.load() ||
        root->operations != 1 || root->activeOperation != state_->identity)
      return Result::Invalid;
    auto &slot = root->metadata[metadataOffset(MetadataKind::FamilyState)];
    if (slot.phase != detail::AllocationPhase::Empty ||
        root->nextMetadata == std::numeric_limits<uint64_t>::max()) return Result::WouldBlock;
    slot = {};
    slot.identity = root->nextMetadata++;
    slot.operation = state_->identity;
    slot.phase = detail::AllocationPhase::Reserved;
    out = MetadataAllocation(root, MetadataKind::FamilyState, slot.identity);
    return Result::Ok;
  } catch (...) { poison(root); return Result::Internal; }
}
Result SourceOperationView::reserveContinuation(ContinuationKind kind,
                                                SourceClaim &out) const noexcept {
  return reserve(false, static_cast<size_t>(kind), out);
}
Result SourceOperationView::reserveCompletionTicket(SourceClaim &out) const noexcept {
  return reserve(true, 0, out);
}
Result SourceOperationView::reserve(bool ticket, size_t index,
                                   SourceClaim &out) const noexcept {
  if (!state_ || out || (!ticket && index >= continuationKinds)) return Result::Invalid;
  const auto &root = state_->root;
  try {
    std::lock_guard<std::mutex> lock(root->mutex);
    if (!state_->admitted || state_->unknown.load(std::memory_order_acquire) ||
        root->operations != 1 || root->activeOperation != state_->identity)
      return Result::Invalid;
    auto &count = ticket ? root->tickets : root->continuations[index];
    if (count) return Result::WouldBlock;
    ++count;
    out.state_ = root;
    out.operation_ = state_;
    out.kind_ = ticket ? SourceClaim::Kind::Ticket : SourceClaim::Kind::Continuation;
    out.index_ = index;
    return Result::Ok;
  } catch (...) {
    poison(root);
    return Result::Internal;
  }
}
SourceOperation::~SourceOperation() { abandon(); }
SourceOperation::SourceOperation(SourceOperation &&other) noexcept
    : state_(std::move(other.state_)) {}
SourceOperation &SourceOperation::operator=(SourceOperation &&other) noexcept {
  if (this != &other) {
    abandon();
    state_ = std::move(other.state_);
  }
  return *this;
}
void SourceOperation::abandon() noexcept {
  if (!state_) return;
  poison(state_->root);
  state_->unknown.store(true, std::memory_order_release);
  state_.reset();
}
SourceOperationView SourceOperation::view() const noexcept {
  return SourceOperationView(state_);
}
Result SourceOperation::finish() noexcept {
  if (!state_) return Result::Invalid;
  try {
    {
      std::lock_guard<std::mutex> lock(state_->root->mutex);
      if (state_->finished) return Result::Invalid;
      state_->finished = true;
    }
    state_.reset();
    return Result::Ok;
  } catch (...) {
    poison(state_->root);
    return Result::Internal;
  }
}

SourceRoot::SourceRoot(setup_custody::SetupUniqueRecord record, std::shared_ptr<detail::State> state) noexcept
    : SetupUniqueRecord(std::move(record)), state_(std::move(state)) {}
SourceRoot::~SourceRoot() {
  if (!state_->released.load(std::memory_order_acquire)) poison(state_);
}
Result SourceRoot::create(const SourceSpec &spec, SourceRootOwner &out) noexcept {
  return createImpl(spec, {}, {}, false, out);
}
Result SourceRoot::createFunded(const SourceSpec &spec, setup_custody::SetupTicket state,
    setup_custody::SetupTicket box, SourceRootOwner &out) noexcept {
  return createImpl(spec, std::move(state), std::move(box), true, out);
}
Result SourceRoot::createImpl(const SourceSpec &spec, setup_custody::SetupTicket stateTicket,
    setup_custody::SetupTicket boxTicket, bool funded, SourceRootOwner &out) noexcept {
  setup_custody::PendingSetup statePending(stateTicket), boxPending(boxTicket);
  if (funded && (!stateTicket || !boxTicket)) return Result::Invalid;
  uint64_t bytes = 0;
  if (out || !geometry(spec, bytes)) return Result::Invalid;
  try {
    auto state = funded ? setup_custody::makeSetupShared<detail::State>(stateTicket, spec, bytes) :
        std::make_shared<detail::State>(spec, bytes);
    auto candidate = funded ? setup_custody::makeSetupUnique<SourceRoot>(boxTicket, state) :
        SourceRootOwner(new SourceRoot({}, state));
    auto &owners = registry();
    std::lock_guard<std::mutex> lock(owners.mutex);
    if (owners.nextIdentity == std::numeric_limits<uint64_t>::max())
      return Result::WouldBlock;
    for (size_t i = 0; i < owners.roots.size(); ++i) {
      if (owners.roots[i]) continue;
      state->identity = owners.nextIdentity++;
      state->registrySlot = i;
      owners.roots[i] = state;
      state->released.store(false, std::memory_order_release);
      out = std::move(candidate);
      return Result::Ok;
    }
    return Result::WouldBlock;
  } catch (setup_custody::Error e) {
    return e == setup_custody::Error::Uncertain ? Result::Poisoned : Result::Invalid;
  } catch (...) { return Result::Internal; }
}
SourceDomain SourceRoot::domain() const noexcept { return SourceDomain(state_); }
Result SourceRoot::bindMetadataLayout(const MGPUCaptureMetadataLayoutV1 &profile) noexcept {
  const auto owned = profile; // Validate exactly the retained representation.
  if (!allocation_layout::validateMetadataLayout(owned)) return Result::Invalid;
  try {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto status = workAdmission(*state_);
    if (status != Result::Ok) return status;
    if (state_->released.load(std::memory_order_acquire) || state_->metadataLayoutBound || state_->admissionStarted)
      return Result::Invalid;
    state_->metadataLayout = owned;
    state_->metadataLayoutBound = true;
    return Result::Ok;
  } catch (...) { poison(state_); return Result::Internal; }
}
Result SourceRoot::seal() noexcept {
  try {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->phase == Phase::Joined) return Result::Joined;
    state_->phase = Phase::Sealed;
    return state_->poisoned.load(std::memory_order_acquire)
               ? Result::Poisoned : Result::Ok;
  } catch (...) {
    poison(state_);
    return Result::Internal;
  }
}
Result SourceRoot::tryJoin(JoinReceipt &out) noexcept {
  if (out) return Result::Invalid;
  try {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->poisoned.load(std::memory_order_acquire)) return Result::Poisoned;
    if (state_->phase == Phase::Joined) return Result::Joined;
    if (state_->phase != Phase::Sealed) return Result::Invalid;
    if (state_->payloadBytes || state_->operations || state_->tickets || state_->metadataAllocationBytes)
      return Result::WouldBlock;
    for (auto bytes : state_->metadataBytesByKind) if (bytes) return Result::WouldBlock;
    for (auto count : state_->resources) if (count) return Result::WouldBlock;
    for (auto count : state_->continuations) if (count) return Result::WouldBlock;
    for (const auto &slot : state_->metadata)
      if (slot.phase != detail::AllocationPhase::Empty) return Result::WouldBlock;
    state_->phase = Phase::Joined;
    out.state_ = state_;
    return Result::Ok;
  } catch (...) {
    poison(state_);
    return Result::Internal;
  }
}
Result SourceRoot::release(JoinReceipt &receipt) noexcept {
  if (receipt.state_ != state_) return Result::Invalid;
  try {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->poisoned.load(std::memory_order_acquire)) return Result::Poisoned;
    if (state_->phase != Phase::Joined || state_->released.load(std::memory_order_acquire))
      return Result::Invalid;
    auto &owners = registry();
    std::lock_guard<std::mutex> registryLock(owners.mutex);
    auto &slot = owners.roots[state_->registrySlot];
    if (slot != state_) {
      poison(state_);
      return Result::Internal;
    }
    slot.reset();
    state_->released.store(true, std::memory_order_release);
    receipt.state_.reset();
    return Result::Ok;
  } catch (...) {
    poison(state_);
    return Result::Internal;
  }
}
uint32_t registeredSources() noexcept {
  try {
    auto &owners = registry();
    std::lock_guard<std::mutex> lock(owners.mutex);
    uint32_t count = 0;
    for (const auto &root : owners.roots) if (root) ++count;
    return count;
  } catch (...) {
    return maxSources; // Diagnostic failure never reports free capacity.
  }
}
} // namespace mgpu::source_scope

namespace mgpu::allocation_layout {
void sourceMetadataLayout(MetadataLayoutBuilder &out) noexcept {
  out.shared<source_scope::detail::OperationState>(MetadataKind::OperationState);
}
Expected fundedRootStateExtent() noexcept {
  return setup_custody::setupSharedExtent<source_scope::detail::State>();
}
void rootSetupLayout(SetupLayoutBuilder &out) noexcept {
  out.shared<source_scope::detail::State>(MGPU_CAPTURE_SETUP_ROOT_STATE);
  out.unique<source_scope::SourceRoot>(MGPU_CAPTURE_SETUP_ROOT_BOX);
}
void fundedRootSetupLayout(FundedSetupLayoutBuilder &out) noexcept {
  out.shared<source_scope::detail::State>(MGPU_CAPTURE_SETUP_ROOT_STATE);
  out.unique<source_scope::SourceRoot>(MGPU_CAPTURE_SETUP_ROOT_BOX);
}
} // namespace mgpu::allocation_layout
