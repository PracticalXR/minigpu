#include "../include/minigpu_capture_cleanup_owner.h"
#include "../include/minigpu_metadata_layout.h"
#include "../include/minigpu_source_metadata_allocator.h"
#include <utility>

namespace mgpu::capture_cleanup {
using namespace source_scope;
namespace detail {
struct FamilyState {
  FamilyState(SourceDomain value, OperationPurpose why, SourceOperation &&held,
              MetadataAllocation allocation) noexcept
      : source(std::move(value)), operation(std::move(held)), view(operation.view()),
        purpose(why), metadata(std::move(allocation)) { metadata.constructed(); }
  SourceDomain source;
  SourceOperation operation;
  SourceOperationView view;
  OperationPurpose purpose = OperationPurpose::Cleanup;
  std::atomic<bool> admitted{false};
  std::atomic<bool> uncertain{false};
  MetadataAllocation metadata;
  ~FamilyState() {
    const bool known = operation && !uncertain.load() && operation.finish() == Result::Ok;
    if (!known) source.quarantine();
    metadata.disposed(known);
  }
};
struct CleanupJob : TaskCustody {
  CleanupJob(MetadataAllocation allocation, TaskCustody &&held) noexcept
      : TaskCustody(std::move(held), std::move(allocation)) {}
  InlineCallable<bool()> release;
  MGPUResourceCleanupCallback callback = nullptr;
  void *userdata = nullptr;
  task_queue::Entry queue{this, [](void *p) noexcept { static_cast<CleanupJob *>(p)->run(); }};
  MetadataAnchor<CleanupJob> *anchor = nullptr; // Independent heap holder.
  void run() noexcept {
    bool released = false, returned = false;
    try { released = release(); } catch (...) {}
    try { callback(released ? MGPU_CONTEXT_LEASE_OK : MGPU_CONTEXT_LEASE_INTERNAL, userdata); returned = true; }
    catch (...) {}
    if (released && (returned || !scoped())) {
      finished();
      MetadataDelete<MetadataAnchor<CleanupJob>>{}(std::exchange(anchor, nullptr));
    } else {
      quarantine(); // Anchor retains actual release closure, native owner and pin.
    }
  }
};
} // namespace detail

allocation_layout::Expected familyStateAllocationExtent() noexcept {
  return metadataSharedExtent<detail::FamilyState>();
}

int nativeResult(Result result) noexcept {
  switch (result) {
  case Result::Ok: return MGPU_CONTEXT_LEASE_OK;
  case Result::WouldBlock: return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  case Result::Invalid: return MGPU_CONTEXT_LEASE_INVALID;
  case Result::Sealed: case Result::Poisoned: case Result::Joined:
    return MGPU_CONTEXT_LEASE_RETIRED;
  default: return MGPU_CONTEXT_LEASE_INTERNAL;
  }
}
Result Family::begin(const SourceDomain &source, Family &out) noexcept {
  return beginPurpose(source, OperationPurpose::Cleanup, out);
}
Result Family::beginWork(const SourceDomain &source, Family &out) noexcept {
  return beginPurpose(source, OperationPurpose::Work, out);
}
bool Family::cleanup() const noexcept {
  return !state_ || state_->purpose == OperationPurpose::Cleanup;
}
Result Family::beginPurpose(const SourceDomain &source, OperationPurpose purpose, Family &out) noexcept {
  if (out) return Result::Invalid;
  if (!source) return Result::Ok;
  SourceOperation operation;
  const auto result = source.beginOperation(purpose, operation);
  if (result != Result::Ok) return result;
  auto view = operation.view();
  MetadataAllocation allocation;
  const auto reserved = view.reserveFamilyMetadata(allocation);
  if (reserved != Result::Ok) { (void)operation.finish(); return reserved; }
  try {
    out.state_ = std::allocate_shared<detail::FamilyState>(
        MetadataAllocator<detail::FamilyState>(allocation, view),
        source, purpose, std::move(operation), allocation);
    return Result::Ok;
  } catch (const MetadataAllocationError &error) {
    (void)allocation.cancelUnallocated();
    if (operation) (void)operation.finish();
    return error.result;
  } catch (...) {
    (void)allocation.cancelUnallocated();
    if (operation) (void)operation.finish();
    return Result::Internal;
  }
}
bool Family::matches(const SourceDomain &source) const noexcept {
  return sameSourceDomain(source, state_ ? state_->source : SourceDomain{});
}
void Family::quarantine() const noexcept {
  if (state_) { state_->uncertain = true; state_->source.quarantine(); }
}
Result Family::reservePrimary(SourceClaim &out) const noexcept {
  return reserveContinuation(ContinuationKind::PrimaryTask, out);
}
Result Family::reserveContinuation(ContinuationKind kind, SourceClaim &out) const noexcept {
  if (state_ && state_->uncertain.load()) return Result::Poisoned;
  return state_ ? state_->view.reserveContinuation(kind, out) : Result::Ok;
}
Result Family::reserveTicket(SourceClaim &out) const noexcept {
  if (state_ && state_->uncertain.load()) return Result::Poisoned;
  return state_ ? state_->view.reserveCompletionTicket(out) : Result::Ok;
}
Result Family::reserveMetadata(std::initializer_list<MetadataKind> kinds,
                               MetadataReservations &out) const noexcept {
  if (state_ && state_->uncertain.load()) return Result::Poisoned;
  return state_ ? state_->view.reserveMetadata(kinds, out) : Result::Ok;
}
Result Family::accept(SourceClaim &claim) const noexcept {
  if (!state_) return claim ? Result::Invalid : Result::Ok;
  if (state_->uncertain.load()) return Result::Poisoned;
  // Fresh admission linearizes at the locked Open snapshot. An overlapping
  // Seal may return before enqueue does; the already retained family/claims
  // prevent Join across that work. Later public calls must start a new family.
  // Once admitted, actual continuations remain authorized across Seal.
  if (state_->purpose == OperationPurpose::Work && !state_->admitted.load()) {
    Snapshot snapshot;
    const auto status = state_->source.snapshot(snapshot);
    if (status != Result::Ok) return status;
    if (snapshot.phase == Phase::Sealed) return Result::Sealed;
    if (snapshot.phase == Phase::Poisoned) return Result::Poisoned;
    if (snapshot.phase == Phase::Joined) return Result::Joined;
  }
  const auto accepted = claim.accept();
  if (accepted == Result::Ok) state_->admitted = true;
  return accepted;
}
Result GlobalTaskCustody::create(const Family &family, bool (*reserve)(),
    bool (*refund)() noexcept, std::shared_ptr<GlobalTaskCustody> &out) noexcept {
  if (out || !reserve || !refund) return Result::Invalid;
  try {
    MetadataReservations slots;
    const auto reserved = family.reserveMetadata({MetadataKind::GlobalTaskCustody}, slots);
    if (reserved != Result::Ok) return reserved;
    auto held = makeMetadataShared<GlobalTaskCustody>(slots, MetadataKind::GlobalTaskCustody);
    held->family_ = family;
    if (!reserve()) return Result::WouldBlock;
    held->refund_ = refund;
    out = std::move(held);
    return Result::Ok;
  } catch (...) { return Result::Internal; }
}
GlobalTaskCustody::~GlobalTaskCustody() {
  const bool known = !uncertain_.load() && (!refund_ || refund_());
  if (!known) family_.quarantine();
  metadata_.disposed(known);
}
TaskCustody::TaskCustody(TaskCustody &&other, MetadataAllocation allocation) noexcept
    : family_(std::move(other.family_)), global_(std::move(other.global_)),
      primary_(std::move(other.primary_)), metadata_(std::move(allocation)),
      prepared_(std::exchange(other.prepared_, false)),
      accepted_(std::exchange(other.accepted_, false)),
      finished_(other.finished_.exchange(false)), uncertain_(other.uncertain_.exchange(false)) {}
TaskCustody::~TaskCustody() {
  bool known = !uncertain_.load() && (!accepted_ || finished_.load());
  if (primary_ && known)
    known = (accepted_ ? primary_.retire() : primary_.cancelUnaccepted()) == Result::Ok;
  if (!known) {
    if (global_) global_->quarantine();
    family_.quarantine();
  }
  metadata_.disposed(known);
}
Result TaskCustody::prepare(const Family &family, ContinuationKind kind,
                            std::shared_ptr<GlobalTaskCustody> global) noexcept {
  if (prepared_) return Result::Invalid;
  if (global && !global->matches(family)) return Result::Invalid;
  prepared_ = true; family_ = family;
  global_ = std::move(global);
  return family_.reserveContinuation(kind, primary_);
}
Result TaskCustody::accept() noexcept {
  if (!prepared_ || accepted_ || (family_ && !primary_)) return Result::Invalid;
  const auto result = family_.accept(primary_);
  if (result == Result::Ok) accepted_ = true;
  return result;
}
void TaskCustody::quarantine() noexcept {
  uncertain_ = true;
  if (global_) global_->quarantine();
  family_.quarantine();
}

Result TicketCustody::create(const Family &family, std::shared_ptr<TicketCustody> &out) noexcept {
  if (out) return Result::Invalid;
  MetadataReservations slots;
  const auto reserved = family.reserveMetadata({MetadataKind::TicketCustody}, slots);
  return reserved == Result::Ok ? create(family, slots, out) : reserved;
}
Result TicketCustody::create(const Family &family, MetadataReservations &slots,
                             std::shared_ptr<TicketCustody> &out) noexcept {
  if (out) return Result::Invalid;
  if (!family) return Result::Ok;
  SourceClaim ticket;
  const auto reserved = family.reserveTicket(ticket);
  if (reserved != Result::Ok) return reserved;
  try {
    out = makeMetadataShared<TicketCustody>(slots, MetadataKind::TicketCustody,
        family, std::move(ticket));
    return Result::Ok;
  } catch (...) {
    if (ticket) (void)ticket.cancelUnaccepted();
    return Result::Internal;
  }
}
TicketCustody::~TicketCustody() {
  bool known = !uncertain_.load() && (!accepted_ || finished_.load());
  if (ticket_ && known)
    known = (accepted_ ? ticket_.retire() : ticket_.cancelUnaccepted()) == Result::Ok;
  if (!known) family_.quarantine();
  metadata_.disposed(known);
}
Result TicketCustody::accept() noexcept {
  if (accepted_ || !ticket_) return Result::Invalid;
  const auto result = family_.accept(ticket_);
  if (result == Result::Ok) accepted_ = true;
  return result;
}
void TicketCustody::quarantine() noexcept { uncertain_ = true; family_.quarantine(); }

PreparedTask::~PreparedTask() = default;
int PreparedTask::prepare(const Family &family, InlineCallable<bool()> release,
                          MGPUResourceCleanupCallback callback, void *userdata) noexcept {
  if (job_ || !release || !callback || !family.cleanup()) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    TaskCustody held;
    const auto result = held.prepare(family);
    if (result != Result::Ok) return nativeResult(result);
    MetadataReservations slots;
    const auto reserved = family.reserveMetadata({MetadataKind::PrimaryJob, MetadataKind::PrimaryAnchor}, slots);
    if (reserved != Result::Ok) return nativeResult(reserved);
    auto job = makeMetadataShared<detail::CleanupJob>(slots, MetadataKind::PrimaryJob, std::move(held));
    job->release = std::move(release); job->callback = callback; job->userdata = userdata;
    auto anchor = makeMetadataUnique<MetadataAnchor<detail::CleanupJob>>(slots, MetadataKind::PrimaryAnchor, job);
    if (!task_.prepare(job->queue, job)) return MGPU_CONTEXT_LEASE_INVALID;
    job_ = std::move(job); anchor_ = std::move(anchor);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
int PreparedTask::enqueue(Enqueue enqueue, void *context) noexcept {
  if (!job_ || !enqueue || attempted_) return MGPU_CONTEXT_LEASE_INVALID;
  attempted_ = true;
  const auto result = job_->accept();
  if (result != Result::Ok) return nativeResult(result);
  job_->anchor = anchor_.release();
  const auto queued = enqueue(context, task_);
  if (queued != MGPU_CONTEXT_LEASE_OK) {
    // Proven nonacceptance; token/job references survive caller owner locks.
    job_->finished();
    anchor_.reset(std::exchange(job_->anchor, nullptr));
    return queued;
  }
  return MGPU_CONTEXT_LEASE_OK;
}
} // namespace mgpu::capture_cleanup

namespace mgpu::allocation_layout {
void cleanupMetadataLayout(MetadataLayoutBuilder &out) noexcept {
  out.shared<capture_cleanup::detail::FamilyState>(MetadataKind::FamilyState);
  out.shared<capture_cleanup::GlobalTaskCustody>(MetadataKind::GlobalTaskCustody);
  out.shared<capture_cleanup::detail::CleanupJob>(MetadataKind::PrimaryJob, MGPU_CAPTURE_META_VARIANT_CLEANUP);
  out.unique<source_scope::MetadataAnchor<capture_cleanup::detail::CleanupJob>>(MetadataKind::PrimaryAnchor, MGPU_CAPTURE_META_VARIANT_CLEANUP);
  out.shared<capture_cleanup::TicketCustody>(MetadataKind::TicketCustody);
}
} // namespace mgpu::allocation_layout
