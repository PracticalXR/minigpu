#ifndef MINIGPU_ALLOCATION_EXTENT_H
#define MINIGPU_ALLOCATION_EXTENT_H

// Internal requested-allocation layout only, NOT a budget/physical profile.
// No payload construction, exception probe, heap or runtime registration.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

// This vendor-private adapter names the control blocks of the VS 2022 C++
// library (STL v143, inspected at 14.40 and 14.42): allocate_shared allocates
// one _Ref_count_obj_alloc3, make_shared news one _Ref_count_obj2. Any other
// STL reports Unsupported (CMake warns), and a metadata request that differs
// from its bound extent is still refused at run time. C++20 is the library's
// mode; the C++17 CPU tests build the Unsupported path.
#if defined(_WIN32) && defined(_M_X64) && !defined(__clang__) && \
    defined(_MSVC_STL_VERSION) && _MSVC_STL_VERSION == 143 && \
    defined(_MSVC_LANG) && _MSVC_LANG >= 202002L && \
    !defined(MINIGPU_ALLOCATION_LAYOUT_FORCE_UNSUPPORTED)
#define MGPU_KNOWN_SHARED_ALLOCATION_LAYOUT 1
#else
#define MGPU_KNOWN_SHARED_ALLOCATION_LAYOUT 0
#endif

namespace mgpu::allocation_layout {
enum class Status : uint32_t { Unsupported = 0, Known = 1 };
struct Extent {
  uint64_t bytes = 0;
  uint32_t alignment = 0;
};
struct Expected {
  Status status = Status::Unsupported;
  Extent extent{};
};
struct Identity {
  uint32_t version = 1;
  uint32_t compilerFullVersion = 0;
  uint32_t stlVersion = 0, stlUpdate = 0;
  uint32_t iteratorDebugLevel = 0;
  uint32_t languageVersion = 0, buildFlags = 0;
  uint32_t pointerBytes = sizeof(void *);
  uint32_t defaultNewAlignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
};
inline constexpr bool sharedLayoutSupported = MGPU_KNOWN_SHARED_ALLOCATION_LAYOUT != 0;
constexpr Identity identity() noexcept {
  Identity result;
#ifdef _MSC_FULL_VER
  result.compilerFullVersion = _MSC_FULL_VER;
#endif
#ifdef _MSVC_STL_VERSION
  result.stlVersion = _MSVC_STL_VERSION;
  result.stlUpdate = _MSVC_STL_UPDATE;
  result.iteratorDebugLevel = _ITERATOR_DEBUG_LEVEL;
#endif
#ifdef _MSVC_LANG
  result.languageVersion = _MSVC_LANG;
#else
  result.languageVersion = __cplusplus;
#endif
#ifdef NDEBUG
  result.buildFlags |= 1;
#endif
#ifdef _DEBUG
  result.buildFlags |= 2;
#endif
#ifdef _MT
  result.buildFlags |= 4;
#endif
#ifdef _DLL
  result.buildFlags |= 8;
#endif
#ifdef _WIN32
  result.buildFlags |= 16;
#endif
#ifdef _M_X64
  result.buildFlags |= 32;
#endif
  return result;
}
template<class T> constexpr Expected uniqueExtent() noexcept {
  static_assert(!std::is_array_v<T>, "One object, not an array allocation");
  static_assert(alignof(T) <= UINT32_MAX, "Alignment must fit the layout record");
  return {Status::Known, {sizeof(T), alignof(T)}};
}
template<class T, class Allocator> constexpr Expected sharedExtent() noexcept {
  static_assert(!std::is_array_v<T>, "Only ordinary allocate_shared is covered");
#if MGPU_KNOWN_SHARED_ALLOCATION_LAYOUT
  // MSVC memory: allocate_shared selects this exact type with the ORIGINAL
  // allocator type, then allocates one rebound block. Never replace T or the
  // allocator with a same-size surrogate or guess control-block overhead.
  using Block = std::_Ref_count_obj_alloc3<std::remove_cv_t<T>, Allocator>;
  static_assert(std::is_base_of_v<std::_Ref_count_base, Block> &&
      sizeof(Block) >= sizeof(std::_Ref_count_base) + sizeof(T), "Block holds the counts and T");
  return uniqueExtent<Block>();
#else
  return {}; // Unknown is explicit; zero is never a known extent.
#endif
}
template<class T> constexpr Expected defaultSharedExtent() noexcept {
  static_assert(!std::is_array_v<T>, "Only ordinary make_shared is covered");
#if MGPU_KNOWN_SHARED_ALLOCATION_LAYOUT
  // MSVC make_shared uses new _Ref_count_obj2<T> directly. Alignment is the
  // actual block type's requirement, not an invented argument to ordinary
  // operator new. Over-aligned types select aligned new normally.
  using Block = std::_Ref_count_obj2<T>;
  static_assert(std::is_base_of_v<std::_Ref_count_base, Block> &&
      sizeof(Block) >= sizeof(std::_Ref_count_base) + sizeof(T), "Block holds the counts and T");
  return uniqueExtent<Block>();
#else
  return {}; // Do not instantiate a private type or require complete T.
#endif
}
} // namespace mgpu::allocation_layout
#undef MGPU_KNOWN_SHARED_ALLOCATION_LAYOUT
#endif
