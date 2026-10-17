#ifndef MINIGPU_INLINE_RESULT_H
#define MINIGPU_INLINE_RESULT_H

#include <condition_variable>
#include <mutex>

namespace mgpu::capture_work {
#ifdef MINIGPU_INLINE_RESULT_TEST
void testInlineResultConstruct();
void testInlineResultWait();
void testInlineResultLockedWait();
#endif
// One-shot CPU acknowledgement embedded in an already owned job. No separate
// promise/shared-state owner, reset, timeout or native completion/refund proof.
// The job must outlive both publish and wait, including their lock/notify tails.
// Platform/library synchronization internals still require host qualification.
class InlineResult {
public:
  InlineResult() {
#ifdef MINIGPU_INLINE_RESULT_TEST
    testInlineResultConstruct();
#endif
  }
  InlineResult(const InlineResult &) = delete;
  InlineResult &operator=(const InlineResult &) = delete;
  bool publish(int result) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (ready_) return false;
      result_ = result;
      ready_ = true;
    }
    changed_.notify_all();
    return true;
  }
  int wait() {
#ifdef MINIGPU_INLINE_RESULT_TEST
    testInlineResultWait();
#endif
    std::unique_lock<std::mutex> lock(mutex_);
#ifdef MINIGPU_INLINE_RESULT_TEST
    testInlineResultLockedWait();
#endif
    changed_.wait(lock, [this] { return ready_; });
    return result_;
  }
private:
  std::mutex mutex_;
  std::condition_variable changed_;
  bool ready_ = false;
  int result_ = 0;
};
} // namespace mgpu::capture_work

#endif
