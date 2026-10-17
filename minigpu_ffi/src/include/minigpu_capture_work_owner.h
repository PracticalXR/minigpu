#ifndef MINIGPU_CAPTURE_WORK_OWNER_H
#define MINIGPU_CAPTURE_WORK_OWNER_H
// CPU-only custody used by the real import/use/fence/wait production jobs.
#include "minigpu_capture_cleanup_owner.h"
#include "minigpu_capture_recipe_owner.h"

namespace mgpu::capture_work {
using capture_cleanup::Family;
struct Outcome {
  int result = MGPU_CONTEXT_LEASE_INTERNAL;
  bool known = false;
  // Only a WaitAction returns this, after linking its WaitSignal's `parked`
  // node into the worker's pump; the receipt settles when the pump resumes it.
  bool parked = false;
  static Outcome done(int result = MGPU_CONTEXT_LEASE_OK) noexcept { return {result, true, false}; }
  static Outcome uncertain() noexcept { return {MGPU_CONTEXT_LEASE_QUARANTINED, false, false}; }
  static Outcome deferred() noexcept { return {MGPU_CONTEXT_LEASE_OK, false, true}; }
};
using Action = InlineCallable<Outcome()>;
using Completion = InlineCallable<void(int)>;
struct WaitSignal;
using WaitAction = InlineCallable<Outcome(WaitSignal &)>;
source_scope::Result validateFrame(const source_scope::SourceDomain &,
    uint32_t width, uint32_t height, uint64_t bytes) noexcept;
source_scope::Result validateDispatch(const source_scope::SourceDomain &,
    int x, int y, int z) noexcept;
namespace detail { struct WorkJob; struct WaitReceipt; }

// Reserve before allocating resource holders or changing owner phase.
// Actions return explicit known disposition; exceptions retain the actual job.
class PreparedWork {
public:
  using Enqueue = capture_cleanup::PreparedTask::Enqueue;
  PreparedWork() = default;
  ~PreparedWork();
  PreparedWork(const PreparedWork &) = delete;
  PreparedWork &operator=(const PreparedWork &) = delete;
  int reserve(const Family &, source_scope::ContinuationKind =
      source_scope::ContinuationKind::PrimaryTask,
      std::shared_ptr<capture_cleanup::GlobalTaskCustody> = {}) noexcept;
  int configure(Action action, Completion complete) noexcept;
  // Creates the inline acknowledgement before owner locks/acceptance. Mutable
  // PreparedWork methods require exclusive access; wait is claimed only once.
  int configureSynchronous(Action action) noexcept;
  int enqueue(Enqueue, void *) noexcept;
  // Invalid/unaccepted/async/repeated waits refuse without blocking. Accepted
  // synchronization exceptions propagate to the caller's uncertainty handling.
  int wait();
private:
  friend class PreparedWait;
  int reservePrepared(source_scope::ContinuationKind, source_scope::MetadataReservations &,
                       capture_cleanup::TaskCustody &&) noexcept;
  std::shared_ptr<detail::WorkJob> job_;
  source_scope::MetadataUnique<source_scope::MetadataAnchor<detail::WorkJob>> anchor_;
  task_queue::Prepared task_;
  bool attempted_ = false;
  bool enqueued_ = false, waitClaimed_ = false;
};

// Dawn receives this exact preallocated object. The pump must return from the
// native callback before the receipt settles. A stalled pump keeps all owners.
// An action either pumps inline and returns a known/uncertain outcome, or links
// `parked` into its worker and returns Outcome::deferred(): the worker resumes
// the receipt once `completed` is set, settling it as done when `result` is OK
// and as uncertain (quarantined) otherwise. The receipt fills in `parked`.
struct WaitSignal {
  std::atomic<bool> completed{false};
  int result = MGPU_CONTEXT_LEASE_INTERNAL;
  task_queue::ParkedWait parked;
};
class PreparedWait {
public:
  PreparedWait() = default;
  explicit PreparedWait(source_scope::MetadataAllocation allocation) noexcept
      : metadata_(std::move(allocation)) {}
  static int create(const Family &, std::shared_ptr<PreparedWait> &) noexcept;
  ~PreparedWait();
  PreparedWait(const PreparedWait &) = delete;
  PreparedWait &operator=(const PreparedWait &) = delete;
  int prepare(const Family &, WaitAction submitAndPump,
              Completion complete,
              std::shared_ptr<capture_cleanup::GlobalTaskCustody> = {}) noexcept;
  int enqueue(PreparedWork::Enqueue, void *) noexcept;
private:
  PreparedWork fence_; // Destroy after receipt/anchor; its action retains receipt.
  std::shared_ptr<detail::WaitReceipt> wait_;
  source_scope::MetadataUnique<source_scope::MetadataAnchor<detail::WaitReceipt>> anchor_;
  bool attempted_ = false;
  source_scope::MetadataAllocation metadata_;
};
} // namespace mgpu::capture_work
#endif
