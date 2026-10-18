#ifndef MINIGPU_CAPTURE_CLEANUP_OWNER_H
#define MINIGPU_CAPTURE_CLEANUP_OWNER_H

// Internal CPU custody. No exported family handles or renderer initialization.
#include "minigpu_context_lifetime.h"
#include "minigpu_source_scope_owner.h"
#include "minigpu_source_metadata_allocator.h"
#include "minigpu_inline_callable.h"
#include "minigpu_task_queue.h"
#include <atomic>
#include <functional>
#include <memory>

namespace mgpu::capture_cleanup {
namespace detail { struct FamilyState; struct CleanupJob; }
int nativeResult(source_scope::Result) noexcept;
allocation_layout::Expected familyStateAllocationExtent() noexcept;

class Family {
public:
  static source_scope::Result begin(const source_scope::SourceDomain &, Family &) noexcept;
  static source_scope::Result beginWork(const source_scope::SourceDomain &, Family &) noexcept;
  bool cleanup() const noexcept;
  explicit operator bool() const noexcept { return !!state_; }
  bool matches(const source_scope::SourceDomain &) const noexcept;
  bool sameOperation(const Family &other) const noexcept { return state_ == other.state_; }
  void quarantine() const noexcept;
  source_scope::Result accept(source_scope::SourceClaim &) const noexcept;
  source_scope::Result reservePrimary(source_scope::SourceClaim &) const noexcept;
  source_scope::Result reserveContinuation(source_scope::ContinuationKind,
                                            source_scope::SourceClaim &) const noexcept;
  source_scope::Result reserveTicket(source_scope::SourceClaim &) const noexcept;
  source_scope::Result reserveMetadata(std::initializer_list<source_scope::MetadataKind>,
                                       source_scope::MetadataReservations &) const noexcept;
#ifdef MINIGPU_SOURCE_METADATA_TEST
  std::weak_ptr<void> allocationWeakForTest() const noexcept { return state_; }
#endif
private:
  static source_scope::Result beginPurpose(const source_scope::SourceDomain &,
      source_scope::OperationPurpose, Family &) noexcept;
  std::shared_ptr<detail::FamilyState> state_;
};

// One existing global job charge shared by the real primary/fence/wait owners.
// No references back to jobs. Its family survives the final global refund.
// The reserve callback must not throw after charging.
class GlobalTaskCustody {
public:
  explicit GlobalTaskCustody(source_scope::MetadataAllocation allocation) noexcept
      : metadata_(std::move(allocation)) {}
  static source_scope::Result create(const Family &, bool (*reserve)(),
      bool (*refund)() noexcept, std::shared_ptr<GlobalTaskCustody> &) noexcept;
  ~GlobalTaskCustody();
  bool matches(const Family &family) const noexcept { return family_.sameOperation(family); }
  void quarantine() noexcept { uncertain_ = true; family_.quarantine(); }
private:
  Family family_;
  bool (*refund_)() noexcept = nullptr;
  std::atomic<bool> uncertain_{false};
  source_scope::MetadataAllocation metadata_;
};

// Lifetime base: the actual job's native members and closure copies finish first.
class TaskCustody {
public:
  TaskCustody() = default;
  ~TaskCustody();
  source_scope::Result prepare(const Family &,
      source_scope::ContinuationKind = source_scope::ContinuationKind::PrimaryTask,
      std::shared_ptr<GlobalTaskCustody> = {}) noexcept;
  source_scope::Result accept() noexcept;
  void finished() noexcept { finished_.store(true, std::memory_order_release); }
  void quarantine() noexcept;
protected:
  TaskCustody(TaskCustody &&, source_scope::MetadataAllocation) noexcept;
  bool scoped() const noexcept { return !!family_; }
private:
  Family family_;
  std::shared_ptr<GlobalTaskCustody> global_;
  source_scope::SourceClaim primary_;
  source_scope::MetadataAllocation metadata_;
  bool prepared_ = false, accepted_ = false;
  std::atomic<bool> finished_{false}, uncertain_{false};
};

// Shared by State AND every counted NativeRef tail. It never owns either one.
class TicketCustody {
public:
  TicketCustody(source_scope::MetadataAllocation allocation, Family family,
      source_scope::SourceClaim ticket) noexcept
      : family_(std::move(family)), ticket_(std::move(ticket)), metadata_(std::move(allocation)) {}
  static source_scope::Result create(const Family &, std::shared_ptr<TicketCustody> &) noexcept;
  static source_scope::Result create(const Family &, source_scope::MetadataReservations &,
                                     std::shared_ptr<TicketCustody> &) noexcept;
  ~TicketCustody();
  source_scope::Result accept() noexcept;
  void finished() noexcept { finished_.store(true, std::memory_order_release); }
  void quarantine() noexcept;
private:
  Family family_;
  source_scope::SourceClaim ticket_;
  source_scope::MetadataAllocation metadata_;
  bool accepted_ = false;
  std::atomic<bool> finished_{false}, uncertain_{false};
};

// The same prepared job and queue boundary are used by native cleanup and CPU
// checks. OK consumes the exact prepared token; refusal never links/invokes it
// and leaves it intact. No throwing after acceptance or hidden queue allocation.
// Unknown cleanup keeps the independent preallocated anchor.
class PreparedTask {
public:
  using Enqueue = int (*)(void *, task_queue::Prepared &) noexcept;
  PreparedTask() = default;
  ~PreparedTask();
  PreparedTask(const PreparedTask &) = delete;
  PreparedTask &operator=(const PreparedTask &) = delete;
  int prepare(const Family &, InlineCallable<bool()> release,
              MGPUResourceCleanupCallback, void *) noexcept;
  int enqueue(Enqueue, void *) noexcept;
private:
  std::shared_ptr<detail::CleanupJob> job_; // Destroy after task and anchor.
  source_scope::MetadataUnique<source_scope::MetadataAnchor<detail::CleanupJob>> anchor_;
  task_queue::Prepared task_;
  bool attempted_ = false;
};
} // namespace mgpu::capture_cleanup
#endif
