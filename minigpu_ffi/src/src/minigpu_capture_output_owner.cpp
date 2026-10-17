#include "../include/minigpu_capture_output_owner.h"

namespace mgpu::capture_output {
using namespace source_scope;
Result validateWriter(const SourceDomain &source, const SourceDomain &packed,
    ResourceKind role, uint32_t width, uint32_t height, uint64_t bytes,
    uint64_t usage) noexcept {
  const auto geometry = capture_work::validateFrame(source, width, height, bytes);
  if (geometry != Result::Ok) return geometry;
  if (!sameSourceDomain(source, packed)) return Result::Invalid;
  if (!source) return (usage & 0x80u) ? Result::Ok : Result::Invalid;
  return role == ResourceKind::PackedBuffer && usage == capture_recipe_v1::packedUsage
      ? Result::Ok : Result::Invalid;
}
Result validateOutput(const SourceDomain &source, const SourceDomain &parent,
    uint32_t width, uint32_t height, uint64_t bytes) noexcept {
  if (!sameSourceDomain(source, parent)) return Result::Invalid;
  return capture_work::validateFrame(source, width, height, bytes);
}
SlotReservation::~SlotReservation() {
  if (refund_ && !refund_(context_)) source_.quarantine();
}
Result SlotReservation::reserve(const SourceDomain &source, bool (*reserve)(void *),
    bool (*refund)(void *) noexcept, void *context) noexcept {
  if (refund_ || !reserve || !refund) return Result::Invalid;
  source_ = source;
  try {
    if (!reserve(context)) return Result::WouldBlock;
    context_ = context; refund_ = refund;
    return Result::Ok;
  } catch (...) { return Result::Internal; }
}
} // namespace mgpu::capture_output
