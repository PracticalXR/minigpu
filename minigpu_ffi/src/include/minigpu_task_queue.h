#ifndef MINIGPU_TASK_QUEUE_H
#define MINIGPU_TASK_QUEUE_H

// CPU-only queue mechanism shared by the real WebGPU worker and focused checks.
// The caller serializes Core access. No callback or last-owner release belongs
// inside that gate. Core/entries cannot move and must outlive their linked work.
#include <atomic>
#include <cassert>
#include <deque>
#include <functional>
#include <memory>
#include <utility>

namespace mgpu::task_queue {
struct Stamp { bool profiled = false; long long enqueuedUs = 0; };
// A wait whose native callback fires while its instance is pumped. A task
// hands it to its worker instead of blocking in it: the worker pumps
// `instance` between tasks and while idle, and calls `resume` once, on the
// worker and outside the callback, after `ready` is set. The node lives in
// its owner, which stays alive until resume; it is linked only while parked.
struct ParkedWait {
  const std::atomic<bool> *ready = nullptr;
  void (*resume)(ParkedWait &) noexcept = nullptr;
  void *owner = nullptr;
  void *instance = nullptr; // Referenced by the worker while parked.
  ParkedWait *next = nullptr;
  bool linked = false;
};
namespace detail {
struct Hook {
  explicit Hook(bool scoped) noexcept : scoped(scoped) {}
  Hook *next = nullptr;
  const bool scoped;
  Stamp stamp;
};
struct LegacyRecord : Hook {
  LegacyRecord() noexcept : Hook(false) {}
  LegacyRecord(const LegacyRecord &) = delete;
  LegacyRecord &operator=(const LegacyRecord &) = delete;
  std::function<void()> task;
};
}
class Core;
class Prepared;
class Execution;
class Entry : private detail::Hook {
public:
  using Invoke = void (*)(void *) noexcept;
  Entry(void *target, Invoke invoke) noexcept : Hook(true), invoke_(invoke), target_(target) {}
  Entry(const Entry &) = delete;
  Entry &operator=(const Entry &) = delete;
private:
  friend class Core;
  friend class Prepared;
  friend class Execution;
  Invoke invoke_;
  void *target_;
  std::shared_ptr<void> queuedOwner_; // Empty before acceptance and after dequeue.
  bool submitted_ = false;
};
class Prepared {
public:
  Prepared() = default;
  Prepared(const Prepared &) = delete;
  Prepared &operator=(const Prepared &) = delete;
  bool prepare(Entry &entry, std::shared_ptr<void> owner) noexcept {
    if (entry_ || owner_ || !owner || owner.get() != entry.target_ ||
        !entry.invoke_ || entry.submitted_) return false;
    entry_ = &entry; owner_ = std::move(owner);
    return true;
  }
  explicit operator bool() const noexcept { return entry_ && owner_; }
private:
  friend class Core;
  friend class Execution;
  Entry *entry_ = nullptr;
  std::shared_ptr<void> owner_;
};
class Execution {
public:
  Execution() = default;
  Execution(const Execution &) = delete;
  Execution &operator=(const Execution &) = delete;
  bool empty() const noexcept { return !present_; }
  Stamp stamp() const noexcept { return stamp_; }
  void run() {
    if (!present_ || std::exchange(invoked_, true)) throw std::bad_function_call();
    if (invoke_) invoke_(target_);
    else if (legacy_) legacy_();
    // Retention lasts through the caller's execution-holder destruction, not
    // merely callback return. Payloads are never cleared under the queue gate.
  }
  bool takeInline(Prepared &task, Stamp stamp = {}) noexcept {
    if (!empty() || !task || task.entry_->submitted_ || task.entry_->queuedOwner_) return false;
    auto &entry = *task.entry_;
    entry.submitted_ = true;
    invoke_ = entry.invoke_; target_ = entry.target_; stamp_ = stamp;
    scopedOwner_ = std::move(task.owner_); task.entry_ = nullptr;
    present_ = true;
    return true;
  }
private:
  friend class Core;
  std::function<void()> legacy_;
  std::shared_ptr<void> scopedOwner_;
  Entry::Invoke invoke_ = nullptr;
  void *target_ = nullptr;
  Stamp stamp_;
  bool present_ = false, invoked_ = false;
};
class Core {
public:
  Core() = default;
  Core(const Core &) = delete;
  Core &operator=(const Core &) = delete;
  ~Core() { assert(empty()); } // Worker shutdown drains, never silently drops.
  bool empty() const noexcept { return head_ == nullptr; }
  bool pushScoped(Prepared &task, Stamp stamp = {}) noexcept {
    if (!task || task.entry_->submitted_ || task.entry_->queuedOwner_) return false;
    auto &entry = *task.entry_;
    entry.queuedOwner_ = std::move(task.owner_); entry.submitted_ = true;
    task.entry_ = nullptr;
    link(entry, stamp);
    return true;
  }
  // Empty record allocation can throw before any user payload transfer. Deque
  // end insertion preserves pointers to earlier records. No iterators escape.
  void pushLegacy(std::function<void()> &task, Stamp stamp = {}) {
    legacy_.emplace_back();
    auto &record = legacy_.back();
    record.task.swap(task);
    link(record, stamp);
  }
  bool takeFront(Execution &out) noexcept {
    if (!head_ || !out.empty()) return false;
    auto *entry = head_;
    head_ = entry->next;
    if (!head_) tail_ = nullptr;
    out.stamp_ = entry->stamp; out.present_ = true;
    if (entry->scoped) {
      auto &scoped = *static_cast<Entry *>(entry);
      scoped.next = nullptr;
      out.invoke_ = scoped.invoke_; out.target_ = scoped.target_;
      out.scopedOwner_ = std::move(scoped.queuedOwner_);
    } else {
      assert(!legacy_.empty() && &legacy_.front() == entry);
      out.legacy_.swap(legacy_.front().task);
      legacy_.pop_front(); // Empty callable: no user destruction under gate.
    }
    return true;
  }
private:
  void link(detail::Hook &entry, Stamp stamp) noexcept {
    entry.next = nullptr; entry.stamp = stamp;
    if (tail_) tail_->next = &entry;
    else head_ = &entry;
    tail_ = &entry;
  }
  std::deque<detail::LegacyRecord> legacy_;
  detail::Hook *head_ = nullptr, *tail_ = nullptr;
};
} // namespace mgpu::task_queue
#endif
