#include "../include/minigpu_capture_work_owner.h"
#include "../include/minigpu_metadata_layout.h"
#include "../include/minigpu_inline_result.h"
#include <optional>
#include <utility>

namespace mgpu::capture_work {
using namespace source_scope;
using capture_cleanup::nativeResult;
Result validateFrame(const SourceDomain &source, uint32_t width, uint32_t height,
                     uint64_t bytes) noexcept {
  if (!width || !height || width > maxFrameBytes / 4 ||
      height > maxFrameBytes / (uint64_t(width) * 4) ||
      bytes != uint64_t(width) * height * 4) return Result::Invalid;
  if (!source) return Result::Ok;
  Snapshot snapshot;
  const auto status = source.snapshot(snapshot);
  if (status != Result::Ok) return status;
  return width == snapshot.spec.width && height == snapshot.spec.height &&
      bytes == snapshot.frameBytes ? Result::Ok : Result::Invalid;
}
Result validateDispatch(const SourceDomain &source, int x, int y, int z) noexcept {
  if (x <= 0 || y <= 0 || z <= 0) return Result::Invalid;
  if (!source) return Result::Ok;
  Snapshot snapshot;
  const auto status = source.snapshot(snapshot);
  if (status != Result::Ok) return status;
  constexpr auto side = capture_recipe_v1::workgroupSide;
  return uint32_t(x) == (snapshot.spec.width + side - 1) / side &&
      uint32_t(y) == (snapshot.spec.height + side - 1) / side && z == 1
      ? Result::Ok : Result::Invalid;
}
namespace detail {
struct WorkJob : capture_cleanup::TaskCustody {
  WorkJob(MetadataAllocation allocation, capture_cleanup::TaskCustody &&held) noexcept
      : TaskCustody(std::move(held), std::move(allocation)) {}
  Action action;
  Completion complete;
  std::optional<InlineResult> result;
  task_queue::Entry queue{this, [](void *p) noexcept { static_cast<WorkJob *>(p)->run(); }};
  MetadataAnchor<WorkJob> *anchor = nullptr;
  bool parkable = false; // A PreparedWait fence: its receipt may settle it later.
  void run() noexcept {
    Outcome outcome;
    try { outcome = action(); } catch (...) {}
    if (outcome.parked) {
      // The anchor keeps this job; WaitReceipt::resume settles it.
      if (parkable) return;
      outcome = Outcome::uncertain();
    }
    settle(outcome);
  }
  // May release the last owner of this job: touches nothing after the anchor.
  void settle(Outcome outcome) noexcept {
    if (!outcome.known) quarantine();
    bool returned = false;
    try {
      if (result) returned = result->publish(outcome.known ? outcome.result : MGPU_CONTEXT_LEASE_QUARANTINED);
      else { complete(outcome.result); returned = true; }
    } catch (...) {}
    if (!returned) quarantine();
    if (outcome.known && returned) {
      finished();
      MetadataDelete<MetadataAnchor<WorkJob>>{}(std::exchange(anchor, nullptr));
    }
    // Unknown native/callback disposition retains this actual action, owners
    // and pin through the independent preallocated anchor, not just counters.
  }
};
struct WaitReceipt : capture_cleanup::TaskCustody {
  WaitReceipt(MetadataAllocation allocation, capture_cleanup::TaskCustody &&held) noexcept
      : TaskCustody(std::move(held), std::move(allocation)) {}
  WaitSignal signal;
  WaitAction action;
  MetadataAnchor<WaitReceipt> *anchor = nullptr;
  WorkJob *fence = nullptr; // Owns this receipt through its action.
  Outcome run() noexcept {
    signal.parked.ready = &signal.completed;
    signal.parked.owner = this;
    signal.parked.resume = [](task_queue::ParkedWait &node) noexcept {
      static_cast<WaitReceipt *>(node.owner)->resume();
    };
    Outcome outcome;
    try { outcome = action(signal); } catch (...) {}
    // A linked node is resumed exactly once whatever the action returned; an
    // unlinked one never is, so a deferral without it would hang forever.
    if (signal.parked.linked) return Outcome::deferred();
    if (outcome.parked) outcome = Outcome::uncertain();
    settle(outcome);
    return outcome;
  }
  void settle(const Outcome &outcome) noexcept {
    if (outcome.known) {
      finished();
      MetadataDelete<MetadataAnchor<WaitReceipt>>{}(std::exchange(anchor, nullptr));
    } else quarantine();
  }
  // On the worker, after its pump unlinked the node and left Dawn's callback.
  void resume() noexcept {
    const auto outcome = signal.result == MGPU_CONTEXT_LEASE_OK
        ? Outcome::done() : Outcome{signal.result, false, false};
    auto *job = fence;
    settle(outcome);
    // Last: settling the job can destroy it and, through its action, this.
    if (job) job->settle(outcome);
  }
  void unused() noexcept {
    finished();
  }
};
} // namespace detail

PreparedWork::~PreparedWork() = default;
int PreparedWork::reserve(const Family &family, ContinuationKind kind,
    std::shared_ptr<capture_cleanup::GlobalTaskCustody> global) noexcept {
  if (job_) return MGPU_CONTEXT_LEASE_INVALID;
  const auto jobKind = kind == ContinuationKind::PrimaryTask ? MetadataKind::PrimaryJob : MetadataKind::FenceJob;
  const auto anchorKind = kind == ContinuationKind::PrimaryTask ? MetadataKind::PrimaryAnchor : MetadataKind::FenceAnchor;
  if (kind != ContinuationKind::PrimaryTask && kind != ContinuationKind::QueueFence)
    return MGPU_CONTEXT_LEASE_INVALID;
  try {
    capture_cleanup::TaskCustody held;
    const auto status = held.prepare(family, kind, std::move(global));
    if (status != Result::Ok) return nativeResult(status);
    MetadataReservations slots;
    const auto reserved = family.reserveMetadata({jobKind, anchorKind}, slots);
    if (reserved != Result::Ok) return nativeResult(reserved);
    return reservePrepared(kind, slots, std::move(held));
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
int PreparedWork::reservePrepared(ContinuationKind kind, MetadataReservations &slots,
                                  capture_cleanup::TaskCustody &&held) noexcept {
  if (job_) return MGPU_CONTEXT_LEASE_INVALID;
  const auto jobKind = kind == ContinuationKind::PrimaryTask ? MetadataKind::PrimaryJob : MetadataKind::FenceJob;
  const auto anchorKind = kind == ContinuationKind::PrimaryTask ? MetadataKind::PrimaryAnchor : MetadataKind::FenceAnchor;
  try {
    auto job = makeMetadataShared<detail::WorkJob>(slots, jobKind, std::move(held));
    auto anchor = makeMetadataUnique<MetadataAnchor<detail::WorkJob>>(slots, anchorKind, job);
    if (!task_.prepare(job->queue, job)) return MGPU_CONTEXT_LEASE_INVALID;
    job_ = std::move(job); anchor_ = std::move(anchor);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
int PreparedWork::configure(Action action, Completion complete) noexcept {
  if (!job_ || job_->action || job_->result || attempted_ || !action || !complete)
    return MGPU_CONTEXT_LEASE_INVALID;
  job_->action = std::move(action); job_->complete = std::move(complete);
  return MGPU_CONTEXT_LEASE_OK;
}
int PreparedWork::configureSynchronous(Action action) noexcept {
  if (!job_ || job_->action || job_->result || attempted_ || !action)
    return MGPU_CONTEXT_LEASE_INVALID;
  try {
    job_->result.emplace(); // No CV is constructed for async/fence jobs.
    job_->action = std::move(action);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
int PreparedWork::wait() {
  if (!job_ || !job_->result || !enqueued_ || waitClaimed_)
    return MGPU_CONTEXT_LEASE_INVALID;
  waitClaimed_ = true;
  auto held = job_; // Keep result/lock storage alive through the wait return.
  try { return held->result->wait(); }
  catch (...) { held->quarantine(); throw; }
}
int PreparedWork::enqueue(Enqueue enqueue, void *context) noexcept {
  if (!job_ || !job_->action || !enqueue || attempted_) return MGPU_CONTEXT_LEASE_INVALID;
  attempted_ = true;
  const auto status = job_->accept();
  if (status != Result::Ok) return nativeResult(status);
  job_->anchor = anchor_.release();
  const auto queued = enqueue(context, task_);
  if (queued != MGPU_CONTEXT_LEASE_OK) {
    job_->finished(); // Explicit refusal proves this entry was never linked.
    anchor_.reset(std::exchange(job_->anchor, nullptr)); // Defer destruction past caller locks.
    return queued;
  }
  enqueued_ = true;
  return MGPU_CONTEXT_LEASE_OK;
}

PreparedWait::~PreparedWait() { metadata_.disposed(true); }
int PreparedWait::create(const Family &family, std::shared_ptr<PreparedWait> &out) noexcept {
  if (out) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    MetadataReservations slots;
    const auto status = family.reserveMetadata({MetadataKind::PreparedWaitHolder}, slots);
    if (status != Result::Ok) return nativeResult(status);
    out = makeMetadataShared<PreparedWait>(slots, MetadataKind::PreparedWaitHolder);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
int PreparedWait::prepare(const Family &family,
    WaitAction action, Completion complete,
    std::shared_ptr<capture_cleanup::GlobalTaskCustody> global) noexcept {
  if (wait_ || !action || !complete) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    capture_cleanup::TaskCustody held;
    auto status = held.prepare(family, ContinuationKind::QueueWait, global);
    if (status != Result::Ok) return nativeResult(status);
    capture_cleanup::TaskCustody fenceHeld;
    status = fenceHeld.prepare(family, ContinuationKind::QueueFence, std::move(global));
    if (status != Result::Ok) return nativeResult(status);
    MetadataReservations slots;
    status = family.reserveMetadata({MetadataKind::WaitReceipt, MetadataKind::WaitAnchor,
        MetadataKind::FenceJob, MetadataKind::FenceAnchor}, slots);
    if (status != Result::Ok) return nativeResult(status);
    auto wait = makeMetadataShared<detail::WaitReceipt>(slots, MetadataKind::WaitReceipt, std::move(held));
    wait->action = std::move(action);
    auto anchor = makeMetadataUnique<MetadataAnchor<detail::WaitReceipt>>(slots, MetadataKind::WaitAnchor, wait);
    const auto reserved = fence_.reservePrepared(ContinuationKind::QueueFence, slots, std::move(fenceHeld));
    if (reserved != MGPU_CONTEXT_LEASE_OK) return reserved;
    fence_.job_->parkable = true;
    wait->fence = fence_.job_.get();
    const auto configured = fence_.configure([wait] { return wait->run(); }, std::move(complete));
    if (configured != MGPU_CONTEXT_LEASE_OK) return configured;
    wait_ = std::move(wait); anchor_ = std::move(anchor);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
int PreparedWait::enqueue(PreparedWork::Enqueue enqueue, void *context) noexcept {
  if (!wait_ || !enqueue || attempted_) return MGPU_CONTEXT_LEASE_INVALID;
  attempted_ = true;
  const auto status = wait_->accept();
  if (status != Result::Ok) return nativeResult(status);
  wait_->anchor = anchor_.release();
  const auto enqueued = fence_.enqueue(enqueue, context);
  if (enqueued != MGPU_CONTEXT_LEASE_OK) {
    wait_->unused();
    anchor_.reset(std::exchange(wait_->anchor, nullptr));
  }
  return enqueued;
}
} // namespace mgpu::capture_work

namespace mgpu::allocation_layout {
void workMetadataLayout(MetadataLayoutBuilder &out) noexcept {
  out.shared<capture_work::detail::WorkJob>(MetadataKind::PrimaryJob, MGPU_CAPTURE_META_VARIANT_WORK);
  out.unique<source_scope::MetadataAnchor<capture_work::detail::WorkJob>>(MetadataKind::PrimaryAnchor, MGPU_CAPTURE_META_VARIANT_WORK);
  out.shared<capture_work::detail::WorkJob>(MetadataKind::FenceJob);
  out.unique<source_scope::MetadataAnchor<capture_work::detail::WorkJob>>(MetadataKind::FenceAnchor);
  out.shared<capture_work::detail::WaitReceipt>(MetadataKind::WaitReceipt);
  out.unique<source_scope::MetadataAnchor<capture_work::detail::WaitReceipt>>(MetadataKind::WaitAnchor);
  out.shared<capture_work::PreparedWait>(MetadataKind::PreparedWaitHolder);
}
} // namespace mgpu::allocation_layout
