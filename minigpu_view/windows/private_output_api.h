#ifndef MINIGPU_VIEW_PRIVATE_OUTPUT_API_H_
#define MINIGPU_VIEW_PRIVATE_OUTPUT_API_H_
// Frozen V1 dynamic interop with minigpu_private_output.h. Avoid a source-tree
// sibling include/link dependency in this separately published plugin package.
#include <cstdint>
namespace minigpu_view::private_preview {
// The producer's frozen ABI uses a different busy/unsupported numbering from
// MGPUViewPrivateResult. Never compare or forward these namespaces implicitly.
enum class NativeResult : int {
  ok = 0, invalid = -1, notReady = -2, retired = -3, wouldBlock = -4,
  internal = -5, unsupported = -6, quarantined = -7
};
struct OutputInfo {
  uint32_t size, version, kind, phase;
  int32_t result;
  uint32_t flags;
  uint64_t id, context_generation, device_generation, bytes;
  uint32_t width, height, operations, borrows;
  uint64_t parent;
};
struct BorrowInfo {
  uint32_t size, version;
  uint64_t output_id, loan_id, borrow_id;
  uint32_t width, height, format, reserved;
  uint64_t handle, device_generation;
};
static_assert(sizeof(OutputInfo) == 80 && sizeof(BorrowInfo) == 64);
using Query = int (*)(void *, OutputInfo *, uint32_t);
using Acquire = int (*)(void *, void **);
using Borrow = int (*)(void *, void **, BorrowInfo *, uint32_t);
using Unary = int (*)(void *);
using Release = int (*)(void *, uint64_t);
struct Api {
  Query output = nullptr, loan = nullptr;
  Acquire acquire = nullptr;
  Borrow borrow = nullptr;
  Unary consume = nullptr, seal = nullptr, quarantine = nullptr;
  Release release = nullptr;
};
} // namespace minigpu_view::private_preview
#endif
