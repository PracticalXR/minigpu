// Exported C/C++ ABI/refusal paths only. Never initialize context/GPU/capture.
#include "../../include/minigpu.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
extern "C" int capture_source_c_abi(void);
static void require(bool value, const char *why) {
  if (!value) { std::cerr << "FAIL " << why << '\n'; std::exit(1); }
}
int main() {
  MGPUContextLifetimeStats before{}, after{}; const auto refs = mgpuContextRefCount();
  require(capture_source_c_abi() && mgpuContextGetLifetimeStats(&before, sizeof(before)) == 0,
    "actual C symbols/layouts and initial cold counters");
  MGPUCaptureSourceLogicalMetadataProfileV1 profile;
  std::memset(&profile, 0xa5, sizeof(profile));
  const auto originalProfile = profile;
  require(mgpuCaptureSourceGetLogicalMetadataProfileV1(nullptr, sizeof(profile)) == -1 &&
    mgpuCaptureSourceGetLogicalMetadataProfileV1(&profile, sizeof(profile) - 1) == -1 &&
    std::memcmp(&profile, &originalProfile, sizeof(profile)) == 0,
    "partial metadata profile refuses missing/short storage without mutation");
  require(mgpuCaptureSourceGetLogicalMetadataProfileV1(&profile, sizeof(profile)) == 0 &&
    profile.struct_size == 32 && profile.version == 1 && profile.profile_revision == 4 &&
    profile.coverage == 15 && profile.operation_state_cap == 1 && profile.family_state_cap == 1 &&
    profile.reserved == 0, "exact sixty-slot partial metadata inventory");
  MGPUCaptureMetadataLayoutV1 layout;
  std::memset(&layout, 0xa5, sizeof(layout));
  const auto initialLayout = layout;
  require(mgpuCaptureSourceGetMetadataLayoutV1(nullptr, sizeof(layout)) == -1 &&
      mgpuCaptureSourceGetMetadataLayoutV1(&layout, sizeof(layout) - 1) == -1 &&
      std::memcmp(&layout, &initialLayout, sizeof(layout)) == 0,
      "layout missing/short output refuses without mutation");
  require(mgpuCaptureSourceGetMetadataLayoutV1(&layout, sizeof(layout)) == 0 &&
      layout.struct_size == sizeof(layout) && layout.version == 1 && layout.layout_revision == 1 &&
      layout.kind_count == 31 && layout.slot_count == 60 &&
      layout.flags == MGPU_CAPTURE_META_LAYOUT_REQUEST_SIZES && layout.reserved == 0 &&
      layout.compiler_full_version == 194033818 && layout.stl_version == 143 &&
      layout.stl_update == 202402 && layout.iterator_debug_level == 0 &&
      layout.language_version == 202002 && layout.build_flags == 61 &&
      layout.pointer_bytes == 8 && layout.default_new_alignment == 16,
      "qualified all-kind layout is request reporting only, with exact build identity");
  uint64_t total = 0;
  uint32_t slots = 0, variants = 0;
  for (uint32_t i = 0; i < layout.kind_count; ++i) {
    const auto &entry = layout.kinds[i];
    const uint32_t capacity = i == MGPU_CAPTURE_META_RESOURCE_STATE ? 3 :
        (i == MGPU_CAPTURE_META_BUFFER_CPU || (i >= MGPU_CAPTURE_META_OUTPUT_STATE &&
         i <= MGPU_CAPTURE_META_LOAN_ANCHOR)) ? 2 : i >= MGPU_CAPTURE_META_BORROW_STATE ? 8 : 1;
    const uint32_t alternatives = i == MGPU_CAPTURE_META_PRIMARY_JOB ? 3 :
        i == MGPU_CAPTURE_META_PRIMARY_ANCHOR ? 2 : 1;
    require(entry.kind == i && entry.capacity == capacity && entry.extent_count == alternatives &&
        entry.reserved == 0, "every stable public kind has its exact existing capacity and variants");
    uint64_t maximum = 0;
    uint32_t priorVariant = 0;
    for (uint32_t n = 0; n < 3; ++n) {
      const auto &extent = entry.extents[n];
      if (n >= alternatives) {
        require(!extent.bytes && !extent.alignment && !extent.variant, "unused extent slots are canonical zero");
        continue;
      }
      require(extent.bytes && extent.alignment && !(extent.alignment & (extent.alignment - 1)) &&
          extent.bytes % extent.alignment == 0 && extent.variant > priorVariant,
          "actual type extents are aligned and explicit variants ordered uniquely");
      priorVariant = extent.variant;
      if (extent.bytes > maximum) maximum = extent.bytes;
    }
    total += capacity * maximum; slots += capacity; variants += alternatives;
  }
  require(total == layout.total_requested_bytes && slots == 60 && variants == 34,
      "all-kind aggregate uses maxima of named alternatives, never adds extra capacities");
  MGPUCaptureMetadataLayoutV1 repeated{};
  require(mgpuCaptureSourceGetMetadataLayoutV1(&repeated, sizeof(repeated)) == 0 &&
      std::memcmp(&layout, &repeated, sizeof(layout)) == 0, "cold layout query is deterministic");
  std::cout << "metadata layout: kinds=" << layout.kind_count << " variants=" << variants
      << " slots=" << slots << " requested-bytes=" << total << " reporting-only\n";
  MGPUCaptureSourceCpuMetadataProfileV1 cpuProfile;
  std::memset(&cpuProfile, 0xa5, sizeof(cpuProfile));
  const auto initialCpuProfile = cpuProfile;
  require(mgpuCaptureSourceGetCpuMetadataProfileV1(nullptr, sizeof(cpuProfile)) == -1 &&
      mgpuCaptureSourceGetCpuMetadataProfileV1(&cpuProfile, sizeof(cpuProfile) - 1) == -1 &&
      std::memcmp(&cpuProfile, &initialCpuProfile, sizeof(cpuProfile)) == 0,
      "CPU metadata attestation refuses short/missing storage without mutation");
  require(mgpuCaptureSourceGetCpuMetadataProfileV1(&cpuProfile, sizeof(cpuProfile)) == 0 &&
      cpuProfile.struct_size == 2112 && cpuProfile.version == 1 && cpuProfile.enforcement_revision == 1 &&
      cpuProfile.flags == MGPU_CAPTURE_CPU_METADATA_ENFORCED && !cpuProfile.reserved &&
      std::memcmp(&cpuProfile.logical, &profile, sizeof(profile)) == 0 &&
      std::memcmp(&cpuProfile.layout, &layout, sizeof(layout)) == 0,
      "attestation contains the exact unchanged logical and reporting tables, not setup/host authority");
  struct ExtendedProfile { MGPUCaptureSourceCpuMetadataProfileV1 profile; uint64_t suffix[2]; } extended{};
  extended.suffix[0] = 123; extended.suffix[1] = 456;
  require(mgpuCaptureSourceGetCpuMetadataProfileV1(&extended.profile, sizeof(extended)) == 0 &&
      std::memcmp(&cpuProfile, &extended.profile, sizeof(cpuProfile)) == 0 &&
      extended.suffix[0] == 123 && extended.suffix[1] == 456,
      "attestation repeats deterministically and preserves larger-buffer suffix");
  std::cout << "CPU metadata attestation: revision=1 covered-slots=60 requested-bytes=" << total
      << " setup/host/physical excluded\n";
  MGPUCaptureSetupLayoutV1 setup;
  std::memset(&setup, 0xa5, sizeof(setup)); const auto initialSetup = setup;
  require(mgpuCaptureSourceGetSetupAllocationLayoutV1(nullptr, sizeof(setup)) == -1 &&
      mgpuCaptureSourceGetSetupAllocationLayoutV1(&setup, sizeof(setup) - 1) == -1 &&
      std::memcmp(&setup, &initialSetup, sizeof(setup)) == 0, "setup layout short/missing storage untouched");
  require(mgpuCaptureSourceGetSetupAllocationLayoutV1(&setup, sizeof(setup)) == 0 &&
      setup.struct_size == 312 && setup.version == 1 && setup.layout_revision == 1 &&
      setup.kind_count == 10 && setup.allocation_count == 10 && setup.flags == MGPU_CAPTURE_SETUP_LAYOUT_REQUEST_SIZES &&
      !setup.reserved && setup.compiler_full_version == layout.compiler_full_version &&
      setup.stl_version == layout.stl_version && setup.stl_update == layout.stl_update &&
      setup.iterator_debug_level == layout.iterator_debug_level && setup.language_version == layout.language_version &&
      setup.build_flags == layout.build_flags && setup.pointer_bytes == layout.pointer_bytes &&
      setup.default_new_alignment == layout.default_new_alignment, "native setup10 is exact build reporting only");
  uint64_t setupTotal = 0;
  for (uint32_t i = 0; i < 10; ++i) {
    const auto &entry = setup.kinds[i];
    const auto allocation = i == 0 || i == 2 || i == 6 || i == 7 || i == 9 ?
        MGPU_CAPTURE_SETUP_DEFAULT_SHARED : MGPU_CAPTURE_SETUP_DEFAULT_UNIQUE;
    require(entry.kind == i && entry.allocation_kind == uint32_t(allocation) && !entry.reserved &&
        entry.bytes && entry.alignment && !(entry.alignment & (entry.alignment - 1)) &&
        entry.bytes % entry.alignment == 0 && entry.bytes <= UINT64_MAX - setupTotal,
        "every setup role uses its exact allocator kind and representable extent");
    setupTotal += entry.bytes;
  }
  MGPUCaptureSetupLayoutV1 repeatedSetup{};
  require(setupTotal == setup.total_requested_bytes &&
      mgpuCaptureSourceGetSetupAllocationLayoutV1(&repeatedSetup, sizeof(repeatedSetup)) == 0 &&
      std::memcmp(&setup, &repeatedSetup, sizeof(setup)) == 0, "setup aggregate and repeated query match");
  std::cout << "setup layout: native-allocations=10 requested-bytes=" << setupTotal
      << " reporting-only, Rust3/refund/enforcement excluded\n";
  auto *sentinel = reinterpret_cast<MGPUContextLease *>(uintptr_t(1));
  auto *handle = reinterpret_cast<MGPUCaptureSource *>(uintptr_t(1)); auto *lease = sentinel;
  MGPUCaptureSourceDesc d{48, 1, 10, 9, 1, 0, 17, 19, 360};
  require(mgpuCaptureSourceCreateScoped(nullptr, &d, 48, &handle, &lease) == -1 && !handle && !lease,
    "missing original clears both outputs without initialization");
  lease = sentinel;
  require(mgpuCaptureSourceCreateScoped(sentinel, &d, 48, nullptr, &lease) == -1 && !lease,
    "missing storage refuses before touching unknown original");
  d.frame_bytes = 359;
  require(mgpuCaptureSourceCreateScoped(sentinel, &d, 48, &handle, &lease) == -1 && !handle && !lease,
    "exact geometry refuses before original pin access");
  d.frame_bytes = 360; d.reserved = 1;
  require(mgpuCaptureSourceCreateScoped(sentinel, &d, 48, &handle, &lease) == -1,
    "reserved fields refuse before any source creation");
  MGPUCaptureSourceInfo info{}; std::memset(&info, 0xa5, sizeof(info)); const auto saved = info;
  require(mgpuCaptureSourceGetInfo(reinterpret_cast<MGPUCaptureSource *>(uintptr_t(1)), &info, 63) == -1 &&
    std::memcmp(&info, &saved, sizeof(info)) == 0, "short output untouched before owner access");
  require(mgpuCaptureSourceGetInfo(nullptr, &info, 64) == -1 && !info.struct_size && !info.source_id &&
    mgpuCaptureSourceSeal(nullptr) == -1, "null owner refuses with cleared full output");
  MGPUCaptureSourceJoined receipt{}; std::memset(&receipt, 0xa5, sizeof(receipt)); const auto prior = receipt;
  require(mgpuCaptureSourceTryReleaseJoined(reinterpret_cast<MGPUCaptureSource *>(uintptr_t(1)), &receipt, 31) == -1 &&
    std::memcmp(&receipt, &prior, sizeof(receipt)) == 0, "short joined storage untouched before owner access");
  require(mgpuCaptureSourceTryReleaseJoined(nullptr, &receipt, 32) == -1 && !receipt.source_id && !receipt.struct_size,
    "missing owner never manufactures joined receipt");
  MGPUCaptureFundedSetupProfileV1 funded;
  std::memset(&funded, 0xa5, sizeof(funded));
  const auto initialFunded = funded;
  require(mgpuCaptureSourceGetFundedSetupProfileV1(nullptr, sizeof(funded)) == -1 &&
      mgpuCaptureSourceGetFundedSetupProfileV1(&funded, sizeof(funded) - 1) == -1 &&
      std::memcmp(&funded, &initialFunded, sizeof(funded)) == 0,
      "full funded profile missing/short output is untouched");
  require(mgpuCaptureSourceGetFundedSetupProfileV1(&funded, sizeof(funded)) == 0 &&
      funded.struct_size == sizeof(funded) && funded.version == 1 && funded.enforcement_revision == 2 &&
      funded.flags == (MGPU_CAPTURE_SETUP_ORIGINAL_TICKET_ENFORCED | MGPU_CAPTURE_SETUP_EXPLICIT_CREATE_OUTCOME) && !funded.reserved &&
      std::memcmp(&funded.metadata, &cpuProfile, sizeof(cpuProfile)) == 0,
      "full actual funded native10 plus exact original metadata60 profile");
  const auto &fundedLayout = funded.setup;
  require(fundedLayout.struct_size == 312 && fundedLayout.version == 1 && fundedLayout.layout_revision == 1 &&
      fundedLayout.kind_count == 10 && fundedLayout.allocation_count == 10 &&
      fundedLayout.flags == MGPU_CAPTURE_SETUP_LAYOUT_FUNDED_REQUESTS && !fundedLayout.reserved &&
      fundedLayout.compiler_full_version == layout.compiler_full_version && fundedLayout.stl_version == layout.stl_version &&
      fundedLayout.stl_update == layout.stl_update && fundedLayout.iterator_debug_level == layout.iterator_debug_level &&
      fundedLayout.language_version == layout.language_version && fundedLayout.build_flags == layout.build_flags &&
      fundedLayout.pointer_bytes == layout.pointer_bytes && fundedLayout.default_new_alignment == layout.default_new_alignment,
      "funded table retains exact known compiler/configuration and distinct allocator kinds/flag");
  uint64_t fundedTotal = 0;
  for (uint32_t i = 0; i < 10; ++i) {
    const auto &entry = fundedLayout.kinds[i];
    const bool shared = i == 0 || i == 2 || i == 6 || i == 7 || i == 9;
    require(entry.kind == i && entry.allocation_kind == (shared ? MGPU_CAPTURE_SETUP_FUNDED_SHARED : MGPU_CAPTURE_SETUP_FUNDED_UNIQUE) &&
        !entry.reserved && entry.bytes && entry.bytes <= uint64_t(INT64_MAX) && entry.alignment &&
        !(entry.alignment & (entry.alignment - 1)) && entry.bytes % entry.alignment == 0 &&
        entry.bytes <= UINT64_MAX - fundedTotal, "every actual complete funded role is explicit/aligned/finite");
    fundedTotal += entry.bytes;
  }
  require(fundedTotal == fundedLayout.total_requested_bytes, "funded aggregate uses all ten actual object/block extents");
  MGPUCaptureFundedSetupProfileV1 repeatedFunded{};
  require(mgpuCaptureSourceGetFundedSetupProfileV1(&repeatedFunded, sizeof(repeatedFunded)) == 0 &&
      std::memcmp(&funded, &repeatedFunded, sizeof(funded)) == 0,
      "cold complete funded query deterministic");
  std::cout << "funded setup profile: native10=" << fundedTotal << " metadata60="
      << funded.metadata.layout.total_requested_bytes << " requested-bytes, native route/ticket enforcement only; Rust3/host/physical excluded\n";
  // Only query/linkage above. DO NOT invoke the new Retain/Create entrypoints.
  require(mgpuContextGetLifetimeStats(&after, sizeof(after)) == 0 && refs == mgpuContextRefCount() &&
    std::memcmp(&before, &after, sizeof(before)) == 0, "cold context/ref accounting unchanged");
  std::cout << "PASS actual source C/C++ ABI/layouts and cold missing-storage/owner/geometry refusals; no GPU/engine/capture\n";
}
