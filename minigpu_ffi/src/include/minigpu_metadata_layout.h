#ifndef MINIGPU_METADATA_LAYOUT_H
#define MINIGPU_METADATA_LAYOUT_H

// Closed compiled layout inventory. No runtime registration, owner construction,
// allocation, byte reservation or activation authority. Enforcement is separate.
#include "minigpu_capture_source.h"
#include "minigpu_source_metadata_allocator.h"
#include <limits>

namespace mgpu::allocation_layout {
using source_scope::MetadataKind;
static_assert(source_scope::metadataKinds == MGPU_CAPTURE_META_KIND_COUNT_V1);
static_assert(static_cast<uint32_t>(MetadataKind::OperationState) == MGPU_CAPTURE_META_OPERATION_STATE);
static_assert(static_cast<uint32_t>(MetadataKind::FamilyState) == MGPU_CAPTURE_META_FAMILY_STATE);
static_assert(static_cast<uint32_t>(MetadataKind::GlobalTaskCustody) == MGPU_CAPTURE_META_GLOBAL_TASK);
static_assert(static_cast<uint32_t>(MetadataKind::PrimaryJob) == MGPU_CAPTURE_META_PRIMARY_JOB);
static_assert(static_cast<uint32_t>(MetadataKind::PrimaryAnchor) == MGPU_CAPTURE_META_PRIMARY_ANCHOR);
static_assert(static_cast<uint32_t>(MetadataKind::FenceJob) == MGPU_CAPTURE_META_FENCE_JOB);
static_assert(static_cast<uint32_t>(MetadataKind::FenceAnchor) == MGPU_CAPTURE_META_FENCE_ANCHOR);
static_assert(static_cast<uint32_t>(MetadataKind::WaitReceipt) == MGPU_CAPTURE_META_WAIT_RECEIPT);
static_assert(static_cast<uint32_t>(MetadataKind::WaitAnchor) == MGPU_CAPTURE_META_WAIT_ANCHOR);
static_assert(static_cast<uint32_t>(MetadataKind::PreparedWaitHolder) == MGPU_CAPTURE_META_WAIT_HOLDER);
static_assert(static_cast<uint32_t>(MetadataKind::TicketCustody) == MGPU_CAPTURE_META_TICKET_CUSTODY);
static_assert(static_cast<uint32_t>(MetadataKind::TicketState) == MGPU_CAPTURE_META_TICKET_STATE);
static_assert(static_cast<uint32_t>(MetadataKind::TicketBox) == MGPU_CAPTURE_META_TICKET_BOX);
static_assert(static_cast<uint32_t>(MetadataKind::ImportedFrameState) == MGPU_CAPTURE_META_FRAME_STATE);
static_assert(static_cast<uint32_t>(MetadataKind::ImportedFrameNative) == MGPU_CAPTURE_META_FRAME_NATIVE);
static_assert(static_cast<uint32_t>(MetadataKind::ImportedFrameBox) == MGPU_CAPTURE_META_FRAME_BOX);
static_assert(static_cast<uint32_t>(MetadataKind::PrivateResourceState) == MGPU_CAPTURE_META_RESOURCE_STATE);
static_assert(static_cast<uint32_t>(MetadataKind::BufferCpu) == MGPU_CAPTURE_META_BUFFER_CPU);
static_assert(static_cast<uint32_t>(MetadataKind::ShaderCpu) == MGPU_CAPTURE_META_SHADER_CPU);
static_assert(static_cast<uint32_t>(MetadataKind::ExclusiveUseState) == MGPU_CAPTURE_META_USE_STATE);
static_assert(static_cast<uint32_t>(MetadataKind::ExclusiveUseBox) == MGPU_CAPTURE_META_USE_BOX);
static_assert(static_cast<uint32_t>(MetadataKind::WriterState) == MGPU_CAPTURE_META_WRITER_STATE);
static_assert(static_cast<uint32_t>(MetadataKind::WriterBox) == MGPU_CAPTURE_META_WRITER_BOX);
static_assert(static_cast<uint32_t>(MetadataKind::OutputState) == MGPU_CAPTURE_META_OUTPUT_STATE);
static_assert(static_cast<uint32_t>(MetadataKind::OutputBox) == MGPU_CAPTURE_META_OUTPUT_BOX);
static_assert(static_cast<uint32_t>(MetadataKind::LoanState) == MGPU_CAPTURE_META_LOAN_STATE);
static_assert(static_cast<uint32_t>(MetadataKind::LoanBox) == MGPU_CAPTURE_META_LOAN_BOX);
static_assert(static_cast<uint32_t>(MetadataKind::LoanAnchor) == MGPU_CAPTURE_META_LOAN_ANCHOR);
static_assert(static_cast<uint32_t>(MetadataKind::BorrowState) == MGPU_CAPTURE_META_BORROW_STATE);
static_assert(static_cast<uint32_t>(MetadataKind::BorrowBox) == MGPU_CAPTURE_META_BORROW_BOX);
static_assert(static_cast<uint32_t>(MetadataKind::BorrowAnchor) == MGPU_CAPTURE_META_BORROW_ANCHOR);
constexpr uint32_t requiredMetadataVariants(MetadataKind kind) noexcept {
  if (kind == MetadataKind::PrimaryJob)
    return (1u << MGPU_CAPTURE_META_VARIANT_WORK) | (1u << MGPU_CAPTURE_META_VARIANT_CLEANUP) |
        (1u << MGPU_CAPTURE_META_VARIANT_PRIVATE);
  if (kind == MetadataKind::PrimaryAnchor)
    return (1u << MGPU_CAPTURE_META_VARIANT_WORK) | (1u << MGPU_CAPTURE_META_VARIANT_CLEANUP);
  return 1u << MGPU_CAPTURE_META_VARIANT_GENERIC;
}
inline uint64_t metadataKindByteLimit(const MGPUCaptureMetadataKindLayoutV1 &entry) noexcept {
  // Only an already validated immutable entry. Its product was checked at bind.
  uint64_t maximum = 0;
  for (const auto &extent : entry.extents) if (extent.bytes > maximum) maximum = extent.bytes;
  return maximum * entry.capacity;
}
// The table check of a Known build. An Unsupported build accepts no table.
inline bool validateKnownMetadataLayout(const MGPUCaptureMetadataLayoutV1 &table) noexcept {
  if (table.struct_size != sizeof(table) || table.version != 1 ||
      table.layout_revision != 1 || table.kind_count != source_scope::metadataKinds ||
      table.slot_count != source_scope::metadataSlots || table.flags != MGPU_CAPTURE_META_LAYOUT_REQUEST_SIZES ||
      table.reserved) return false;
  const auto build = identity();
  if (table.compiler_full_version != build.compilerFullVersion || table.stl_version != build.stlVersion ||
      table.stl_update != build.stlUpdate || table.iterator_debug_level != build.iteratorDebugLevel ||
      table.language_version != build.languageVersion || table.build_flags != build.buildFlags ||
      table.pointer_bytes != build.pointerBytes || table.default_new_alignment != build.defaultNewAlignment)
    return false;
  uint64_t total = 0;
  uint32_t slots = 0;
  for (size_t i = 0; i < source_scope::metadataKinds; ++i) {
    const auto &entry = table.kinds[i];
    const auto kind = static_cast<MetadataKind>(i);
    if (entry.kind != i || entry.capacity != source_scope::metadataCapacity(kind) || entry.reserved ||
        !entry.extent_count || entry.extent_count > 3) return false;
    uint64_t maximum = 0;
    uint32_t variants = 0, prior = 0;
    for (uint32_t n = 0; n < 3; ++n) {
      const auto &extent = entry.extents[n];
      if (n >= entry.extent_count) {
        if (extent.bytes || extent.alignment || extent.variant) return false;
        continue;
      }
      if (!extent.bytes || extent.bytes > (std::numeric_limits<size_t>::max)() || !extent.alignment ||
          (extent.alignment & (extent.alignment - 1)) || extent.bytes % extent.alignment ||
          extent.variant < MGPU_CAPTURE_META_VARIANT_GENERIC || extent.variant > MGPU_CAPTURE_META_VARIANT_PRIVATE ||
          extent.variant <= prior) return false;
      variants |= 1u << extent.variant; prior = extent.variant;
      if (extent.bytes > maximum) maximum = extent.bytes;
    }
    if (variants != requiredMetadataVariants(kind) ||
        maximum > (std::numeric_limits<uint64_t>::max)() / entry.capacity) return false;
    const auto bytes = maximum * entry.capacity;
    if (bytes > (std::numeric_limits<uint64_t>::max)() - total) return false;
    total += bytes; slots += entry.capacity;
  }
  return total == table.total_requested_bytes && slots == table.slot_count;
}
inline bool validateMetadataLayout(const MGPUCaptureMetadataLayoutV1 &table) noexcept {
  return sharedLayoutSupported && validateKnownMetadataLayout(table);
}
inline bool validateCpuMetadataProfile(const MGPUCaptureSourceCpuMetadataProfileV1 &profile) noexcept {
  const auto &logical = profile.logical;
  return profile.struct_size == sizeof(profile) && profile.version == 1 &&
      profile.enforcement_revision == 1 && profile.flags == MGPU_CAPTURE_CPU_METADATA_ENFORCED &&
      !profile.reserved && logical.struct_size == sizeof(logical) && logical.version == 1 &&
      logical.profile_revision == 4 && logical.coverage == 15 &&
      logical.operation_state_cap == 1 && logical.family_state_cap == 1 && !logical.reserved &&
      validateMetadataLayout(profile.layout);
}
class MetadataLayoutBuilder {
public:
  template<class T> void shared(MetadataKind kind, uint32_t variant = MGPU_CAPTURE_META_VARIANT_GENERIC) noexcept {
    add(kind, variant, source_scope::metadataSharedExtent<T>());
  }
  template<class T> void unique(MetadataKind kind, uint32_t variant = MGPU_CAPTURE_META_VARIANT_GENERIC) noexcept {
    add(kind, variant, uniqueExtent<T>());
  }
  bool finish(MGPUCaptureMetadataLayoutV1 &out) noexcept {
    if (!valid_ || !sharedLayoutSupported) return false;
    uint64_t total = 0;
    uint32_t slots = 0;
    for (size_t i = 0; i < source_scope::metadataKinds; ++i) {
      const auto &entry = table_.kinds[i];
      if (!entry.extent_count) return false;
      uint64_t maximum = 0;
      uint32_t variants = 0;
      for (uint32_t n = 0; n < entry.extent_count; ++n) {
        variants |= 1u << entry.extents[n].variant;
        if (entry.extents[n].bytes > maximum) maximum = entry.extents[n].bytes;
      }
      if (variants != requiredMetadataVariants(static_cast<MetadataKind>(i))) return false;
      if (maximum > (std::numeric_limits<uint64_t>::max)() / entry.capacity) return false;
      const auto bytes = maximum * entry.capacity;
      if (bytes > (std::numeric_limits<uint64_t>::max)() - total) return false;
      total += bytes; slots += entry.capacity;
    }
    const auto build = identity();
    table_.struct_size = sizeof(table_); table_.version = 1; table_.layout_revision = 1;
    table_.kind_count = static_cast<uint32_t>(source_scope::metadataKinds); table_.slot_count = slots;
    table_.flags = MGPU_CAPTURE_META_LAYOUT_REQUEST_SIZES; table_.total_requested_bytes = total;
    table_.compiler_full_version = build.compilerFullVersion;
    table_.stl_version = build.stlVersion; table_.stl_update = build.stlUpdate;
    table_.iterator_debug_level = build.iteratorDebugLevel;
    table_.language_version = build.languageVersion; table_.build_flags = build.buildFlags;
    table_.pointer_bytes = build.pointerBytes; table_.default_new_alignment = build.defaultNewAlignment;
    if (!validateMetadataLayout(table_)) return false;
    out = table_; return true;
  }
private:
  void add(MetadataKind kind, uint32_t variant, Expected expected) noexcept {
    const auto index = static_cast<size_t>(kind);
    if (index >= source_scope::metadataKinds || expected.status != Status::Known ||
        !expected.extent.bytes || !expected.extent.alignment ||
        (expected.extent.alignment & (expected.extent.alignment - 1)) ||
        expected.extent.bytes % expected.extent.alignment ||
        variant < MGPU_CAPTURE_META_VARIANT_GENERIC || variant > MGPU_CAPTURE_META_VARIANT_PRIVATE) {
      valid_ = false; return;
    }
    if (!(requiredMetadataVariants(kind) & (1u << variant))) { valid_ = false; return; }
    auto &entry = table_.kinds[index];
    for (uint32_t n = 0; n < entry.extent_count; ++n)
      if (entry.extents[n].variant == variant) { valid_ = false; return; }
    if (entry.extent_count == 3) { valid_ = false; return; }
    entry.kind = static_cast<uint32_t>(index); entry.capacity = source_scope::metadataCapacity(kind);
    auto pos = entry.extent_count++;
    while (pos && entry.extents[pos - 1].variant > variant) {
      entry.extents[pos] = entry.extents[pos - 1]; --pos;
    }
    entry.extents[pos] = {expected.extent.bytes, expected.extent.alignment, variant};
  }
  MGPUCaptureMetadataLayoutV1 table_{};
  bool valid_ = true;
};
void sourceMetadataLayout(MetadataLayoutBuilder &) noexcept;
void cleanupMetadataLayout(MetadataLayoutBuilder &) noexcept;
void workMetadataLayout(MetadataLayoutBuilder &) noexcept;
void ticketMetadataLayout(MetadataLayoutBuilder &) noexcept;
void importedMetadataLayout(MetadataLayoutBuilder &) noexcept;
void privateResourceMetadataLayout(MetadataLayoutBuilder &) noexcept;
void shaderUseMetadataLayout(MetadataLayoutBuilder &) noexcept;
void outputMetadataLayout(MetadataLayoutBuilder &) noexcept;
} // namespace mgpu::allocation_layout
#endif
