#ifndef MINIGPU_CAPTURE_PAYLOAD_H
#define MINIGPU_CAPTURE_PAYLOAD_H

// Fixed capture recipe storage, not a general allocator or byte-admission grant.
// Mode comes from the retained source domain. Legacy payloads remain dynamic.
#include "minigpu_capture_recipe_v1.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mgpu::capture_payload {
inline constexpr size_t bindingSlots = 3;
static_assert(capture_recipe_v1::parameterBytes == 16 &&
    capture_recipe_v1::frameSlot == 0 && capture_recipe_v1::packedSlot == 1 &&
    capture_recipe_v1::parametersSlot == 2, "Reprice/review a changed capture recipe");

template<class T, size_t Capacity> class Array {
  struct Fixed { std::array<T, Capacity> values{}; size_t size = 0; };
  std::variant<std::vector<T>, Fixed> storage_;
public:
  explicit Array(bool scoped = false) {
    if (scoped) storage_.template emplace<1>();
  }
  bool scoped() const noexcept { return storage_.index() == 1; }
  size_t size() const noexcept {
    return scoped() ? std::get<1>(storage_).size : std::get<0>(storage_).size();
  }
  bool empty() const noexcept { return size() == 0; }
  T *data() noexcept {
    return scoped() ? std::get<1>(storage_).values.data() : std::get<0>(storage_).data();
  }
  const T *data() const noexcept {
    return scoped() ? std::get<1>(storage_).values.data() : std::get<0>(storage_).data();
  }
  T *begin() noexcept { return data(); }
  const T *begin() const noexcept { return data(); }
  T *end() noexcept { return empty() ? data() : data() + size(); }
  const T *end() const noexcept { return empty() ? data() : data() + size(); }
  T &operator[](size_t index) noexcept { return data()[index]; }
  const T &operator[](size_t index) const noexcept { return data()[index]; }
  void reserve(size_t size) {
    if (scoped()) { if (size > Capacity) throw std::bad_alloc(); }
    else std::get<0>(storage_).reserve(size);
  }
  void resize(size_t size) {
    if (!scoped()) { std::get<0>(storage_).resize(size); return; }
    if (size > Capacity) throw std::bad_alloc(); // No heap/error-string fallback.
    auto &fixed = std::get<1>(storage_);
    for (size_t i = std::min(size, fixed.size); i < std::max(size, fixed.size); ++i)
      fixed.values[i] = T{};
    fixed.size = size;
  }
  void clear() { resize(0); }
  void assign(const T *first, const T *last) {
    if (!scoped()) { std::get<0>(storage_).assign(first, last); return; }
    const auto count = first == last ? 0 : static_cast<size_t>(last - first);
    if (count > Capacity) throw std::bad_alloc();
    resize(count);
    std::copy_n(first, count, data());
  }
  void push_back(const T &value) {
    if (!scoped()) { std::get<0>(storage_).push_back(value); return; }
    auto &fixed = std::get<1>(storage_);
    if (fixed.size == Capacity) throw std::bad_alloc();
    fixed.values[fixed.size] = value;
    ++fixed.size;
  }
  bool swap(Array &other) noexcept {
    static_assert(std::is_nothrow_swappable_v<T>, "Fixed bindings must swap without throwing");
    if (scoped() != other.scoped()) return false;
    if (scoped()) {
      auto &a = std::get<1>(storage_), &b = std::get<1>(other.storage_);
      a.values.swap(b.values); std::swap(a.size, b.size);
    } else std::get<0>(storage_).swap(std::get<0>(other.storage_));
    return true;
  }
  const std::vector<T> &legacy() const noexcept {
    assert(!scoped()); return std::get<0>(storage_);
  }
};

class Kernel {
public:
  explicit Kernel(bool scoped = false) noexcept : scoped_(scoped) {}
  bool scoped() const noexcept { return scoped_; }
  bool assign(const char *text, size_t size) {
    if (scoped_) {
      if (!text || size != capture_recipe_v1::wgsl.size() ||
          std::string_view(text, size) != capture_recipe_v1::wgsl) return false;
      canonical_ = true; // Never retain guest/caller storage after validation.
    } else owned_.assign(text, size);
    return true;
  }
  bool swap(Kernel &other) noexcept {
    if (scoped_ != other.scoped_) return false;
    if (scoped_) std::swap(canonical_, other.canonical_);
    else owned_.swap(other.owned_);
    return true;
  }
  std::string_view view() const noexcept {
    if (scoped_) return canonical_ ? capture_recipe_v1::wgsl : std::string_view{};
    return owned_;
  }
  bool empty() const noexcept { return view().empty(); }
  size_t size() const noexcept { return view().size(); }
  size_t length() const noexcept { return size(); }
  const char *data() const noexcept { return view().data(); }
  const char *c_str() const noexcept { return scoped_ ? data() : owned_.c_str(); }
  char operator[](size_t index) const noexcept { return view()[index]; }
  size_t find(char ch, size_t pos = 0) const noexcept { return view().find(ch, pos); }
  std::string_view substr(size_t pos, size_t size) const { return view().substr(pos, size); }
  Kernel &operator=(const std::string &text) {
    assert(!scoped_); owned_ = text; return *this;
  }
  Kernel &operator=(std::string &&text) noexcept {
    assert(!scoped_); owned_ = std::move(text); return *this;
  }
  friend bool operator==(const Kernel &a, const std::string &b) noexcept { return a.view() == std::string_view(b); }
  friend bool operator!=(const std::string &a, const Kernel &b) noexcept { return std::string_view(a) != b.view(); }
private:
  std::string owned_;
  bool scoped_ = false, canonical_ = false;
};
} // namespace mgpu::capture_payload
#endif
