#ifndef MINIGPU_MODULE_OWNER_H
#define MINIGPU_MODULE_OWNER_H

#include <new>

namespace mgpu::detail {
// Fixed module storage for intentionally retained registry roots. Initialize on
// first use with the normal thread-safe static guard, without a heap allocation
// for the registry itself. Do not register a destructor: module/process teardown
// is not evidence that quarantined native owners have actually retired.
// The registry accessor has one instantiation in its owning translation unit.
// Payload/control-block/driver allocations and fixed module bytes still need
// their own accounting; this helper provides no activation or refund authority.
template<class T> static T &moduleOwner() {
  alignas(T) static unsigned char storage[sizeof(T)];
  static T *const value = ::new (static_cast<void *>(storage)) T;
  return *value;
}
} // namespace mgpu::detail

#endif
