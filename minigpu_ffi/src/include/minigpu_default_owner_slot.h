#ifndef MINIGPU_DEFAULT_OWNER_SLOT_H
#define MINIGPU_DEFAULT_OWNER_SLOT_H
// Inline registry state only. Owner/weak destruction always follows unlock;
// creating a replacement never needs the expired owner's allocation budget.
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>

namespace mgpu::detail {
inline uint64_t nextOwnerGeneration(std::atomic<uint64_t> &sequence) noexcept {
  auto value = sequence.load(std::memory_order_relaxed);
  while (value && value != (std::numeric_limits<uint64_t>::max)()) {
    if (sequence.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
  }
  return 0; // Permanently exhausted; never resurrect a zero/old generation.
}
template<class T> class DefaultOwnerSlot {
public:
  enum class Decision { Ready, Create, Busy, Exhausted, Closed };
  struct Acquisition {
    Decision decision = Decision::Closed;
    uint64_t generation = 0;
    std::shared_ptr<T> owner;
    std::weak_ptr<T> retired;
  };
  explicit DefaultOwnerSlot(std::mutex &mutex) noexcept : mutex_(mutex) {}
  DefaultOwnerSlot(const DefaultOwnerSlot &) = delete;
  DefaultOwnerSlot &operator=(const DefaultOwnerSlot &) = delete;
  Acquisition acquire(std::atomic<uint64_t> &sequence) {
    Acquisition result;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closing_) result.decision = Decision::Closed;
      else if (creating_) result.decision = Decision::Busy;
      else {
        result.owner = current_.lock();
        if (result.owner) result.decision = Decision::Ready;
        else {
          result.retired.swap(current_);
          generation_ = nextOwnerGeneration(sequence);
          result.generation = generation_;
          creating_ = generation_ != 0;
          result.decision = creating_ ? Decision::Create : Decision::Exhausted;
        }
      }
    }
    return result;
  }
  std::shared_ptr<T> lookup() {
    std::shared_ptr<T> result;
    { std::lock_guard<std::mutex> lock(mutex_); if (!closing_) result = current_.lock(); }
    return result;
  }
  bool publish(uint64_t generation, const std::shared_ptr<T> &owner) {
    if (!owner || !generation) return false;
    std::weak_ptr<T> incoming(owner);
    bool published = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!closing_ && creating_ && generation_ == generation) {
        incoming.swap(current_); creating_ = false; published = true;
      }
    }
    return published;
  }
  void abandon(uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (creating_ && generation_ == generation) { creating_ = false; generation_ = 0; }
  }
  void retire(uint64_t generation) {
    std::weak_ptr<T> retired;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!closing_ && !creating_ && generation_ == generation && current_.expired()) {
        retired.swap(current_); generation_ = 0;
      }
    }
  }
  std::weak_ptr<T> close() {
    std::weak_ptr<T> retired;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closing_ = true; creating_ = false; generation_ = 0; retired.swap(current_);
    }
    return retired;
  }
private:
  std::mutex &mutex_;
  std::weak_ptr<T> current_;
  uint64_t generation_ = 0;
  bool creating_ = false, closing_ = false;
};
} // namespace mgpu::detail
#endif
