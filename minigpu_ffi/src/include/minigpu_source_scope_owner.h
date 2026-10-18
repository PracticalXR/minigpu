#ifndef MINIGPU_SOURCE_SCOPE_OWNER_H
#define MINIGPU_SOURCE_SCOPE_OWNER_H

// Internal, CPU-only admission kernel. This is neither a public ABI nor a
// positive capture resource profile. Callers must still retain actual native
// owners on uncertain cleanup; poisoning preserves accounting, not those owners.
// Shared domain counters are synchronized. Individual mutable claim, operation,
// root-control and receipt handles require exclusive access/external locking;
// never race release, move or destruction of the same handle with its use.
#include "minigpu_context_limits.h"
#include "minigpu_allocation_extent.h"
#include "minigpu_capture_source.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include "minigpu_setup_allocator.h"
#include <initializer_list>
#include <utility>

namespace mgpu::source_scope {
inline constexpr uint32_t maxSources = native_limits::maxContextLeases;
inline constexpr uint64_t maxFrameBytes = 128ull * 1024 * 1024;

enum class Result { Ok, Invalid, WouldBlock, Sealed, Poisoned, Joined, Internal };
enum class Phase { Open, Sealed, Joined, Poisoned };
struct SourceSpec {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t recipeRevision = 1;
  uint32_t reserved = 0;
};
enum class ResourceKind : uint8_t {
  ImportedFrame, RecipeShader, PackedBuffer, Parameters, Writer, Output,
  ExclusiveUse, PreviewLoan, DescriptorBorrow, ScopedPin, Count
};
enum class ContinuationKind : uint8_t { PrimaryTask, QueueFence, QueueWait, Count };
enum class OperationPurpose { Work, Cleanup };
// These exact allocation records only; resource, payload and driver overhead
// outside the declared profile still require separate admission.
enum class MetadataKind : uint8_t {
  OperationState, FamilyState, GlobalTaskCustody, PrimaryJob, PrimaryAnchor,
  FenceJob, FenceAnchor, WaitReceipt, WaitAnchor, PreparedWaitHolder,
  TicketCustody, TicketState, TicketBox,
  ImportedFrameState, ImportedFrameNative, ImportedFrameBox,
  PrivateResourceState, BufferCpu, ShaderCpu, ExclusiveUseState, ExclusiveUseBox,
  WriterState, WriterBox, OutputState, OutputBox,
  LoanState, LoanBox, LoanAnchor, BorrowState, BorrowBox, BorrowAnchor, Count
};
inline constexpr size_t operationMetadataKinds = static_cast<size_t>(MetadataKind::ImportedFrameState);
inline constexpr size_t metadataKinds = static_cast<size_t>(MetadataKind::Count);
inline constexpr uint32_t metadataCapacity(MetadataKind kind) noexcept {
  switch (kind) {
  case MetadataKind::PrivateResourceState: return 3;
  case MetadataKind::BufferCpu:
  case MetadataKind::OutputState: case MetadataKind::OutputBox:
  case MetadataKind::LoanState: case MetadataKind::LoanBox:
  case MetadataKind::LoanAnchor: return 2;
  case MetadataKind::BorrowState: case MetadataKind::BorrowBox:
  case MetadataKind::BorrowAnchor: return 8;
  case MetadataKind::Count: return 0;
  default: return 1;
  }
}
inline constexpr size_t metadataOffset(MetadataKind kind) noexcept {
  size_t offset = 0;
  for (size_t i = 0; i < static_cast<size_t>(kind); ++i)
    offset += metadataCapacity(static_cast<MetadataKind>(i));
  return offset;
}
inline constexpr size_t metadataSlots = metadataOffset(MetadataKind::Count);
static_assert(metadataSlots == 60, "Exact logical allocation inventory changed");
// Exact compiled request size of the real private type, not activation or a
// complete metadata/setup byte profile. Unknown toolchains report Unsupported.
allocation_layout::Expected operationStateAllocationExtent() noexcept;
inline constexpr size_t resourceKinds = static_cast<size_t>(ResourceKind::Count);
inline constexpr size_t continuationKinds =
    static_cast<size_t>(ContinuationKind::Count);

struct Snapshot {
  uint64_t identity = 0;
  SourceSpec spec{};
  Phase phase = Phase::Poisoned;
  uint64_t frameBytes = 0;
  uint64_t payloadLimit = 0;
  uint64_t payloadBytes = 0;
  std::array<uint32_t, resourceKinds> resources{};
  uint32_t operationFamilies = 0;
  uint32_t completionTickets = 0;
  std::array<uint32_t, continuationKinds> continuations{};
  std::array<uint32_t, metadataKinds> metadataAllocations{};
  bool metadataLayoutBound = false;
  uint64_t metadataAllocationLimit = 0, metadataAllocationBytes = 0;
  std::array<uint64_t, metadataKinds> metadataBytesByKind{};
};

namespace detail {
struct State;
struct OperationState;
} // namespace detail
class SourceClaim;
class SourceOperation;
class SourceOperationView;
class SourceRoot;
using SourceRootOwner = setup_custody::SetupUnique<SourceRoot>;
class MetadataAllocation;
class MetadataReservations;

class SourceDomain {
public:
  SourceDomain() = default; // Empty means legacy, never a released scoped root.
  explicit operator bool() const noexcept { return state_ != nullptr; }
  // A native adapter still retains its actual uncertain owners. This only
  // poisons the original source's accounting; it is never refund authority.
  void quarantine() const noexcept;
  Result snapshot(Snapshot &out) const noexcept;
  Result reserve(ResourceKind kind, SourceClaim &out) const noexcept;
  Result beginOperation(OperationPurpose purpose, SourceOperation &out) const noexcept;
  friend bool sameSourceDomain(const SourceDomain &, const SourceDomain &) noexcept;

private:
  friend class SourceRoot;
  explicit SourceDomain(std::shared_ptr<detail::State> state) noexcept;
  std::shared_ptr<detail::State> state_;
};

// Copyable allocator identity, not a refund-on-Drop guard. Copies allocate no
// ownership records. The source roots the slot; only known cancellation or the
// allocator's actual deallocation may return it. Payload constructors mark
// constructed() only after their nonthrowing initialization; destructors record
// disposition, but never return capacity. Mutable allocation calls are exclusive.
class MetadataAllocation {
public:
  MetadataAllocation() = default;
  explicit operator bool() const noexcept { return !!state_; }
  void *allocateBytes(size_t bytes, size_t alignment) const;
  void deallocateBytes(void *, size_t bytes, size_t alignment) const noexcept;
  void constructed() const noexcept;
  void disposed(bool known) const noexcept;
  Result cancelUnallocated() const noexcept;
  bool same(const MetadataAllocation &other) const noexcept {
    return state_ == other.state_ && identity_ == other.identity_ && kind_ == other.kind_;
  }
private:
  friend class SourceDomain;
  friend class SourceOperationView;
  friend class SourceClaim;
  MetadataAllocation(std::shared_ptr<detail::State> state, MetadataKind kind,
                     uint64_t identity) noexcept
      : state_(std::move(state)), kind_(kind), identity_(identity) {}
  std::shared_ptr<detail::State> state_;
  MetadataKind kind_ = MetadataKind::Count;
  uint64_t identity_ = 0;
};

// A claim is not a conventional refund-on-Drop guard. Only known preacceptance
// rollback, or an actual final native/alias/tail disposition, refunds it. An
// abandoned or overwritten claim poisons its process-rooted source permanently.
class SourceClaim {
public:
  SourceClaim() = default;
  ~SourceClaim();
  SourceClaim(const SourceClaim &) = delete;
  SourceClaim &operator=(const SourceClaim &) = delete;
  SourceClaim(SourceClaim &&) noexcept;
  SourceClaim &operator=(SourceClaim &&) noexcept;
  explicit operator bool() const noexcept { return state_ != nullptr; }
  Result accept() noexcept;
  Result cancelUnaccepted() noexcept;
  Result retire() noexcept;
  // Each exact live, unaccepted resource claim issues its fixed batch once.
  // Resource allocation retains the source, not an operation.
  Result reserveResourceMetadata(MetadataReservations &) noexcept;

private:
  friend class SourceDomain;
  friend class SourceOperationView;
  enum class Kind : uint8_t { Empty, Resource, Continuation, Ticket };
  void abandon() noexcept;
  Result finish(bool rollback) noexcept;
  std::shared_ptr<detail::State> state_;
  std::shared_ptr<detail::OperationState> operation_;
  Kind kind_ = Kind::Empty;
  size_t index_ = 0;
  uint64_t bytes_ = 0;
  bool accepted_ = false;
  bool metadataIssued_ = false;
};

// An operation explicitly carries the bounded nested-work permission. A public
// or reentrant operation cannot borrow it. Retained views and child/ticket
// claims keep the family occupied after finish(), until their last tail ends.
class SourceOperationView {
public:
  SourceOperationView() = default;
  explicit operator bool() const noexcept { return !!state_; }
  Result reserveContinuation(ContinuationKind kind, SourceClaim &out) const noexcept;
  Result reserveCompletionTicket(SourceClaim &out) const noexcept;
  Result reserveFamilyMetadata(MetadataAllocation &out) const noexcept;
  Result reserveMetadata(std::initializer_list<MetadataKind>, MetadataReservations &) const noexcept;

private:
  friend class SourceOperation;
  explicit SourceOperationView(std::shared_ptr<detail::OperationState>) noexcept;
  Result reserve(bool ticket, size_t index, SourceClaim &out) const noexcept;
  std::shared_ptr<detail::OperationState> state_;
};

// Stack-only batch rollback. Taking a slot transfers its cancellation obligation
// to the allocation factory; untouched slots cancel on scope exit. No heap owner.
class MetadataReservations {
public:
  MetadataReservations() = default;
  ~MetadataReservations();
  MetadataReservations(const MetadataReservations &) = delete;
  MetadataReservations &operator=(const MetadataReservations &) = delete;
  bool scoped() const noexcept { return !!root_; }
  const SourceOperationView &view() const noexcept { return view_; }
  MetadataAllocation take(MetadataKind);
private:
  friend class SourceOperationView;
  friend class SourceClaim;
  std::shared_ptr<detail::State> root_;
  SourceOperationView view_;
  std::array<MetadataAllocation, metadataKinds> allocations_{};
};
class SourceOperation {
public:
  SourceOperation() = default;
  ~SourceOperation();
  SourceOperation(const SourceOperation &) = delete;
  SourceOperation &operator=(const SourceOperation &) = delete;
  SourceOperation(SourceOperation &&) noexcept;
  SourceOperation &operator=(SourceOperation &&) noexcept;
  explicit operator bool() const noexcept { return state_ != nullptr; }
  SourceOperationView view() const noexcept;
  Result finish() noexcept;

private:
  friend class SourceDomain;
  void abandon() noexcept;
  std::shared_ptr<detail::OperationState> state_;
};

class JoinReceipt {
public:
  explicit operator bool() const noexcept { return state_ != nullptr; }

private:
  friend class SourceRoot;
  std::shared_ptr<detail::State> state_;
};
class SourceRoot : public setup_custody::SetupUniqueRecord {
public:
  static Result create(const SourceSpec &, SourceRootOwner &out) noexcept;
  // Staged internal factory; production exports do not route through it yet.
  static Result createFunded(const SourceSpec &, setup_custody::SetupTicket state,
      setup_custody::SetupTicket box, SourceRootOwner &out) noexcept;
  ~SourceRoot();
  SourceRoot(const SourceRoot &) = delete;
  SourceRoot &operator=(const SourceRoot &) = delete;
  SourceDomain domain() const noexcept;
  // Trusted internal data binding, not authorization/provenance. Native factory
  // supplies only its directly linked compiled query. Once, before ANY claim;
  // source setup bytes and the original host budget remain separate authority.
  Result bindMetadataLayout(const MGPUCaptureMetadataLayoutV1 &) noexcept;
  Result seal() noexcept;
  Result tryJoin(JoinReceipt &out) noexcept;
  Result release(JoinReceipt &receipt) noexcept;

private:
  template<class> friend class setup_custody::SetupAllocator;
  explicit SourceRoot(setup_custody::SetupUniqueRecord, std::shared_ptr<detail::State>) noexcept;
  static Result createImpl(const SourceSpec &, setup_custody::SetupTicket,
      setup_custody::SetupTicket, bool funded, SourceRootOwner &) noexcept;
  std::shared_ptr<detail::State> state_;
};

// Diagnostics only; neither a currentness certificate nor refund authority.
uint32_t registeredSources() noexcept;
} // namespace mgpu::source_scope

#endif
