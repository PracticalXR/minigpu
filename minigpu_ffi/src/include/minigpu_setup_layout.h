#ifndef MINIGPU_SETUP_LAYOUT_H
#define MINIGPU_SETUP_LAYOUT_H
#include "minigpu_allocation_extent.h"
#include "minigpu_capture_source.h"
#include "minigpu_setup_allocator.h"
#include <limits>

namespace mgpu::allocation_layout {
constexpr uint32_t setupAllocationKind(uint32_t kind) noexcept {
  return kind == MGPU_CAPTURE_SETUP_ROOT_STATE || kind == MGPU_CAPTURE_SETUP_PREPARED_STATE ||
      kind == MGPU_CAPTURE_SETUP_ORIGINAL_PIN || kind == MGPU_CAPTURE_SETUP_CONTEXT_OWNER ||
      kind == MGPU_CAPTURE_SETUP_SCOPED_PIN ? MGPU_CAPTURE_SETUP_DEFAULT_SHARED : MGPU_CAPTURE_SETUP_DEFAULT_UNIQUE;
}
// The table check of a Known build. An Unsupported build accepts no table.
inline bool validateKnownSetupLayout(const MGPUCaptureSetupLayoutV1 &out) noexcept {
  if (out.struct_size != sizeof(out) || out.version != 1 ||
      out.layout_revision != 1 || out.kind_count != MGPU_CAPTURE_SETUP_KIND_COUNT_V1 ||
      out.allocation_count != MGPU_CAPTURE_SETUP_KIND_COUNT_V1 ||
      out.flags != MGPU_CAPTURE_SETUP_LAYOUT_REQUEST_SIZES || out.reserved) return false;
  const auto build = identity();
  if (out.compiler_full_version != build.compilerFullVersion || out.stl_version != build.stlVersion ||
      out.stl_update != build.stlUpdate || out.iterator_debug_level != build.iteratorDebugLevel ||
      out.language_version != build.languageVersion || out.build_flags != build.buildFlags ||
      out.pointer_bytes != build.pointerBytes || out.default_new_alignment != build.defaultNewAlignment) return false;
  uint64_t total = 0;
  for (uint32_t i = 0; i < MGPU_CAPTURE_SETUP_KIND_COUNT_V1; ++i) {
    const auto &entry = out.kinds[i];
    if (entry.kind != i || entry.allocation_kind != setupAllocationKind(i) || entry.reserved ||
        !entry.bytes || entry.bytes > (std::numeric_limits<size_t>::max)() || !entry.alignment ||
        (entry.alignment & (entry.alignment - 1)) || entry.bytes % entry.alignment ||
        entry.bytes > (std::numeric_limits<uint64_t>::max)() - total) return false;
    total += entry.bytes;
  }
  return total == out.total_requested_bytes;
}
inline bool validateSetupLayout(const MGPUCaptureSetupLayoutV1 &out) noexcept {
  return sharedLayoutSupported && validateKnownSetupLayout(out);
}
class SetupLayoutBuilder {
public:
  template<class T> void shared(MGPUCaptureSetupKindV1 kind) noexcept {
    add(kind, MGPU_CAPTURE_SETUP_DEFAULT_SHARED, defaultSharedExtent<T>());
  }
  template<class T> void unique(MGPUCaptureSetupKindV1 kind) noexcept {
    add(kind, MGPU_CAPTURE_SETUP_DEFAULT_UNIQUE, uniqueExtent<T>());
  }
  bool finish(MGPUCaptureSetupLayoutV1 &out) noexcept {
    if (!valid_ || !sharedLayoutSupported) return false;
    table_.struct_size = sizeof(table_); table_.version = 1; table_.layout_revision = 1;
    table_.kind_count = table_.allocation_count = MGPU_CAPTURE_SETUP_KIND_COUNT_V1;
    table_.flags = MGPU_CAPTURE_SETUP_LAYOUT_REQUEST_SIZES;
    const auto build = identity();
    table_.compiler_full_version = build.compilerFullVersion; table_.stl_version = build.stlVersion;
    table_.stl_update = build.stlUpdate; table_.iterator_debug_level = build.iteratorDebugLevel;
    table_.language_version = build.languageVersion; table_.build_flags = build.buildFlags;
    table_.pointer_bytes = build.pointerBytes; table_.default_new_alignment = build.defaultNewAlignment;
    uint64_t total = 0;
    for (const auto &entry : table_.kinds) {
      if (entry.bytes > (std::numeric_limits<uint64_t>::max)() - total) return false;
      total += entry.bytes;
    }
    table_.total_requested_bytes = total;
    if (!validateSetupLayout(table_)) return false;
    out = table_; return true;
  }
private:
  void add(MGPUCaptureSetupKindV1 kind, uint32_t allocationKind, Expected expected) noexcept {
    const auto i = static_cast<uint32_t>(kind);
    if (!valid_ || i >= MGPU_CAPTURE_SETUP_KIND_COUNT_V1 || table_.kinds[i].bytes ||
        expected.status != Status::Known || !expected.extent.bytes ||
        allocationKind != setupAllocationKind(i)) { valid_ = false; return; }
    table_.kinds[i] = {i, allocationKind, expected.extent.bytes, expected.extent.alignment, 0};
  }
  bool valid_ = true;
  MGPUCaptureSetupLayoutV1 table_{};
};
void rootSetupLayout(SetupLayoutBuilder &) noexcept;
// Private staged funded extents; no public supported-profile/permission claim.
Expected fundedRootStateExtent() noexcept;
Expected fundedPreparedStateExtent() noexcept;
Expected preparedAnchorExtent() noexcept;
void preparedSetupLayout(SetupLayoutBuilder &) noexcept;
void contextSetupLayout(SetupLayoutBuilder &) noexcept;

// Complete funded native10 types. The distinct funded profile/export group
// uses this table; default-factory reporting and source authority stay separate.
struct FundedSetupLayout {
  Identity build{};
  std::array<Extent, MGPU_CAPTURE_SETUP_KIND_COUNT_V1> kinds{};
  uint64_t totalRequestedBytes = 0;
};
class FundedSetupLayoutBuilder {
public:
  template<class T> void shared(MGPUCaptureSetupKindV1 kind) noexcept {
    add(kind, setup_custody::setupSharedExtent<T>());
  }
  template<class T> void unique(MGPUCaptureSetupKindV1 kind) noexcept { add(kind, uniqueExtent<T>()); }
  bool finish(FundedSetupLayout &out) noexcept {
    out = {};
    if (!valid_ || !sharedLayoutSupported) return false;
    uint64_t total = 0;
    for (const auto &extent : table_.kinds) {
      if (!extent.bytes || extent.bytes > static_cast<uint64_t>((std::numeric_limits<ptrdiff_t>::max)()) || !extent.alignment ||
          (extent.alignment & (extent.alignment - 1)) || extent.bytes % extent.alignment ||
          extent.bytes > (std::numeric_limits<uint64_t>::max)() - total) return false;
      total += extent.bytes;
    }
    table_.build = identity(); table_.totalRequestedBytes = total; out = table_; return true;
  }
private:
  void add(MGPUCaptureSetupKindV1 kind, Expected expected) noexcept {
    const auto index = static_cast<size_t>(kind);
    if (index >= table_.kinds.size() || table_.kinds[index].bytes || expected.status != Status::Known) {
      valid_ = false; return;
    }
    table_.kinds[index] = expected.extent;
  }
  bool valid_ = true;
  FundedSetupLayout table_{};
};
void fundedRootSetupLayout(FundedSetupLayoutBuilder &) noexcept;
void fundedPreparedSetupLayout(FundedSetupLayoutBuilder &) noexcept;
void fundedContextSetupLayout(FundedSetupLayoutBuilder &) noexcept;
bool fundedSetupLayout(FundedSetupLayout &) noexcept;
} // namespace mgpu::allocation_layout
#endif
