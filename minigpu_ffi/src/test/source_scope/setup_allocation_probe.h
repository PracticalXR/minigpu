#ifndef MINIGPU_SETUP_ALLOCATION_PROBE_H
#define MINIGPU_SETUP_ALLOCATION_PROBE_H
#include <array>
#include <cstdlib>
#include <new>
#ifdef _WIN32
#include <malloc.h>
#endif
namespace probe {
struct Entry { uintptr_t address = 0; size_t bytes = 0, alignmentArgument = 0; bool freed = false; };
struct Probe { bool active = false, overflow = false; size_t count = 0; std::array<Entry, 4> entries{}; };
inline thread_local Probe state{};
inline thread_local bool failNext = false, insideDelete = false;
inline thread_local void (*freeHook)(uintptr_t, bool) noexcept = nullptr;
inline void allocated(void *pointer, size_t bytes, size_t alignment) noexcept {
  if (!state.active) return;
  if (state.count == state.entries.size()) { state.overflow = true; return; }
  state.entries[state.count++] = {reinterpret_cast<uintptr_t>(pointer), bytes, alignment, false};
}
inline void freed(uintptr_t address) noexcept {
  for (auto &entry : state.entries) if (address && entry.address == address && !entry.freed) entry.freed = true;
}
}
void *operator new(size_t bytes) {
  if (probe::failNext) { probe::failNext = false; throw std::bad_alloc(); }
  void *pointer = std::malloc(bytes ? bytes : 1);
  if (!pointer) throw std::bad_alloc();
  probe::allocated(pointer, bytes, 0); return pointer;
}
void *operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete(void *pointer) noexcept {
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  probe::insideDelete = true;
  if (probe::freeHook) probe::freeHook(address, false);
  std::free(pointer);
  if (probe::freeHook) probe::freeHook(address, true);
  probe::freed(address); probe::insideDelete = false;
}
void operator delete[](void *pointer) noexcept { ::operator delete(pointer); }
void operator delete(void *pointer, size_t) noexcept { ::operator delete(pointer); }
void operator delete[](void *pointer, size_t) noexcept { ::operator delete(pointer); }
void *operator new(size_t bytes, std::align_val_t alignment) {
  if (probe::failNext) { probe::failNext = false; throw std::bad_alloc(); }
  void *pointer = nullptr; const auto required = static_cast<size_t>(alignment);
#ifdef _WIN32
  pointer = _aligned_malloc(bytes ? bytes : 1, required);
#else
  if (posix_memalign(&pointer, required, bytes ? bytes : 1) != 0) pointer = nullptr;
#endif
  if (!pointer) throw std::bad_alloc();
  probe::allocated(pointer, bytes, required); return pointer;
}
void *operator new[](size_t bytes, std::align_val_t alignment) { return ::operator new(bytes, alignment); }
void operator delete(void *pointer, std::align_val_t) noexcept {
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  probe::insideDelete = true;
  if (probe::freeHook) probe::freeHook(address, false);
#ifdef _WIN32
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
  if (probe::freeHook) probe::freeHook(address, true);
  probe::freed(address); probe::insideDelete = false;
}
void operator delete[](void *pointer, std::align_val_t a) noexcept { ::operator delete(pointer, a); }
void operator delete(void *pointer, size_t, std::align_val_t a) noexcept { ::operator delete(pointer, a); }
void operator delete[](void *pointer, size_t, std::align_val_t a) noexcept { ::operator delete(pointer, a); }
#endif
