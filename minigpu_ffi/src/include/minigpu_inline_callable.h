#ifndef MINIGPU_INLINE_CALLABLE_H
#define MINIGPU_INLINE_CALLABLE_H

#include <cstddef>
#include <functional>
#include <new>
#include <type_traits>
#include <utility>

namespace mgpu {
namespace inline_callable_detail {
template<class T> struct IsStdFunction : std::false_type {};
template<class R, class... A> struct IsStdFunction<std::function<R(A...)>> : std::true_type {};
}

// Internal fixed storage, never a heap fallback. Captures may themselves own
// payloads; their allocation and native disposition are separate obligations.
template<class Signature, size_t Capacity = 128> class InlineCallable;
template<class R, size_t Capacity, class... A>
class InlineCallable<R(A...), Capacity> {
public:
  InlineCallable() noexcept = default;
  template<class F, class T = std::decay_t<F>,
      std::enable_if_t<!std::is_same_v<T, InlineCallable> &&
          std::is_invocable_r_v<R, T &, A...>, int> = 0>
  InlineCallable(F &&value) {
    static_assert(!inline_callable_detail::IsStdFunction<T>::value,
        "Scoped callables must arrive typed, before std::function conversion");
    static_assert(sizeof(T) <= Capacity, "Simplify the scoped capture; no heap fallback");
    static_assert(alignof(T) <= alignof(std::max_align_t), "Over-aligned scoped capture");
    static_assert(std::is_nothrow_move_constructible_v<T> && std::is_nothrow_destructible_v<T>,
        "Scoped captures must move and destroy without throwing");
    if constexpr (std::is_pointer_v<T> || std::is_member_pointer_v<T>) {
      if (!value) return;
    }
    ::new(static_cast<void *>(storage_)) T(std::forward<F>(value));
    invoke_ = [](void *p, A... args) -> R {
      return std::invoke(*static_cast<T *>(p), std::forward<A>(args)...);
    };
    move_ = [](void *to, void *from) noexcept {
      auto *value = static_cast<T *>(from);
      ::new(to) T(std::move(*value));
      value->~T();
    };
    destroy_ = [](void *p) noexcept { static_cast<T *>(p)->~T(); };
  }
  InlineCallable(const InlineCallable &) = delete;
  InlineCallable &operator=(const InlineCallable &) = delete;
  InlineCallable(InlineCallable &&other) noexcept { take(other); }
  InlineCallable &operator=(InlineCallable &&other) noexcept {
    if (this != &other) { reset(); take(other); }
    return *this;
  }
  ~InlineCallable() { reset(); }
  explicit operator bool() const noexcept { return invoke_ != nullptr; }
  R operator()(A... args) {
    if (!invoke_) throw std::bad_function_call();
    return invoke_(storage_, std::forward<A>(args)...);
  }
private:
  void reset() noexcept {
    if (auto destroy = std::exchange(destroy_, nullptr)) {
      invoke_ = nullptr; move_ = nullptr;
      destroy(storage_);
    }
  }
  void take(InlineCallable &other) noexcept {
    if (!other.invoke_) return;
    auto move = other.move_;
    invoke_ = std::exchange(other.invoke_, nullptr);
    move_ = std::exchange(other.move_, nullptr);
    destroy_ = std::exchange(other.destroy_, nullptr);
    move(storage_, other.storage_);
  }
  alignas(std::max_align_t) unsigned char storage_[Capacity];
  R (*invoke_)(void *, A...) = nullptr;
  void (*move_)(void *, void *) noexcept = nullptr;
  void (*destroy_)(void *) noexcept = nullptr;
};
} // namespace mgpu
#endif
