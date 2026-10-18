#include "../include/minigpu_capture_setup_boxes.h"
#include "../include/minigpu_capture_setup_bridge.h"
#include "../include/minigpu_capture_source_owner.h"
#include "../include/minigpu_context_owner.h"
#include "../include/minigpu_metadata_layout.h"
#include "../include/minigpu_setup_layout.h"
#include <cstddef>

static_assert(sizeof(MGPUCaptureSourceDesc) == 48 && offsetof(MGPUCaptureSourceDesc, context_generation) == 24);
static_assert(sizeof(MGPUCaptureSourceInfo) == 64 && offsetof(MGPUCaptureSourceInfo, source_id) == 16);
static_assert(sizeof(MGPUCaptureSourceJoined) == 32 && offsetof(MGPUCaptureSourceJoined, source_id) == 8);
static_assert(sizeof(MGPUCaptureSourceLogicalMetadataProfileV1) == 32 &&
    offsetof(MGPUCaptureSourceLogicalMetadataProfileV1, reserved) == 24);
static_assert(sizeof(MGPUCaptureMetadataExtentV1) == 16 && sizeof(MGPUCaptureMetadataKindLayoutV1) == 64);
static_assert(sizeof(MGPUCaptureMetadataLayoutV1) == 2056 && offsetof(MGPUCaptureMetadataLayoutV1, kinds) == 72);
static_assert(sizeof(MGPUCaptureSourceCpuMetadataProfileV1) == 2112 &&
    offsetof(MGPUCaptureSourceCpuMetadataProfileV1, logical) == 16 &&
    offsetof(MGPUCaptureSourceCpuMetadataProfileV1, layout) == 48 &&
    offsetof(MGPUCaptureSourceCpuMetadataProfileV1, reserved) == 2104);
static_assert(sizeof(MGPUCaptureSetupExtentV1) == 24 && sizeof(MGPUCaptureSetupLayoutV1) == 312 &&
    offsetof(MGPUCaptureSetupLayoutV1, kinds) == 72);

namespace {
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
int nativeCreate(void *original, const MGPUCaptureSourceDesc &desc,
    mgpu::source_scope::SourceRootOwner &root, MGPUContextLease *&lease) {
  auto *source = static_cast<MGPUContextLease *>(original);
  const auto pin = mgpu::copyContextLeasePin(source);
  if (!pin || pin->sourceDomain() || pin->contextGeneration() != desc.context_generation ||
      pin->deviceGeneration() != desc.device_generation) return MGPU_CONTEXT_LEASE_INVALID;
  // The exact output cell is pre-rooted before the native factory is entered.
  const auto result = mgpu::createScopedContextLease(source,
      {desc.width, desc.height, desc.recipe_revision, 0}, root, &lease);
  if (lease && (!root || !mgpu::adoptSourceContextLease(lease, root->domain()))) {
    if (root) root->domain().quarantine();
    return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
  return result;
}
struct FundedCreate {
  MGPUContextLease *original;
  const mgpu::setup_custody::SetupTickets *tickets;
};
int nativeCreateFunded(void *raw, const MGPUCaptureSourceDesc &desc,
    mgpu::source_scope::SourceRootOwner &root, MGPUContextLease *&lease) {
  const auto &context = *static_cast<const FundedCreate *>(raw);
  const auto pin = mgpu::copyContextLeasePin(context.original);
  if (!pin || pin->sourceDomain() || pin->contextGeneration() != desc.context_generation ||
      pin->deviceGeneration() != desc.device_generation) return MGPU_CONTEXT_LEASE_INVALID;
  const auto result = mgpu::createScopedContextLeaseFunded(context.original,
      {desc.width, desc.height, desc.recipe_revision, 0}, *context.tickets, root, &lease);
  if (lease && (!root || !mgpu::adoptSourceContextLease(lease, root->domain()))) {
    if (root) root->domain().quarantine();
    return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
  return result;
}
bool nativeRelease(MGPUContextLease *lease) noexcept {
  return mgpu::releaseSourceContextLease(lease);
}
#endif
}
extern "C" int mgpuCaptureSourceGetLogicalMetadataProfileV1(
    MGPUCaptureSourceLogicalMetadataProfileV1 *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  static_assert(mgpu::source_scope::metadataSlots == 60, "priced revision4 metadata inventory");
  *out = {sizeof(*out), 1, 4, 15, 1, 1, 0};
  return MGPU_CONTEXT_LEASE_OK;
}
extern "C" int mgpuCaptureSourceGetMetadataLayoutV1(MGPUCaptureMetadataLayoutV1 *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  *out = {};
  if (!mgpu::allocation_layout::sharedLayoutSupported) return MGPU_CONTEXT_LEASE_UNSUPPORTED;
  mgpu::allocation_layout::MetadataLayoutBuilder layout;
  using namespace mgpu::allocation_layout;
  sourceMetadataLayout(layout); cleanupMetadataLayout(layout); workMetadataLayout(layout);
  ticketMetadataLayout(layout); importedMetadataLayout(layout); privateResourceMetadataLayout(layout);
  shaderUseMetadataLayout(layout); outputMetadataLayout(layout);
  return layout.finish(*out) ? MGPU_CONTEXT_LEASE_OK : MGPU_CONTEXT_LEASE_UNSUPPORTED;
}
extern "C" int mgpuCaptureSourceGetCpuMetadataProfileV1(
    MGPUCaptureSourceCpuMetadataProfileV1 *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  *out = {};
  MGPUCaptureSourceCpuMetadataProfileV1 profile{};
  const auto logical = mgpuCaptureSourceGetLogicalMetadataProfileV1(&profile.logical, sizeof(profile.logical));
  if (logical != MGPU_CONTEXT_LEASE_OK) return logical;
  const auto layout = mgpuCaptureSourceGetMetadataLayoutV1(&profile.layout, sizeof(profile.layout));
  if (layout != MGPU_CONTEXT_LEASE_OK) return layout;
  profile.struct_size = sizeof(profile); profile.version = 1;
  profile.enforcement_revision = 1; profile.flags = MGPU_CAPTURE_CPU_METADATA_ENFORCED;
  if (!mgpu::allocation_layout::validateCpuMetadataProfile(profile)) return MGPU_CONTEXT_LEASE_UNSUPPORTED;
  *out = profile;
  return MGPU_CONTEXT_LEASE_OK;
}
extern "C" int mgpuCaptureSourceGetSetupAllocationLayoutV1(MGPUCaptureSetupLayoutV1 *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  *out = {};
  if (!mgpu::allocation_layout::sharedLayoutSupported) return MGPU_CONTEXT_LEASE_UNSUPPORTED;
  using namespace mgpu::allocation_layout;
  SetupLayoutBuilder builder;
  rootSetupLayout(builder); preparedSetupLayout(builder); contextSetupLayout(builder);
  builder.unique<MGPUCaptureSource>(MGPU_CAPTURE_SETUP_SOURCE_BOX);
  return builder.finish(*out) ? MGPU_CONTEXT_LEASE_OK : MGPU_CONTEXT_LEASE_UNSUPPORTED;
}
static_assert(sizeof(MGPUCaptureFundedSetupProfileV1) == 2448 &&
    offsetof(MGPUCaptureFundedSetupProfileV1, setup) == 16 &&
    offsetof(MGPUCaptureFundedSetupProfileV1, metadata) == 328 &&
    offsetof(MGPUCaptureFundedSetupProfileV1, reserved) == 2440);
extern "C" int mgpuCaptureSourceGetFundedSetupProfileV1(MGPUCaptureFundedSetupProfileV1 *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  *out = {};
  using namespace mgpu::allocation_layout;
  FundedSetupLayout actual{};
  if (!fundedSetupLayout(actual)) return MGPU_CONTEXT_LEASE_UNSUPPORTED;
  MGPUCaptureFundedSetupProfileV1 profile{};
  const auto result = mgpuCaptureSourceGetCpuMetadataProfileV1(&profile.metadata, sizeof(profile.metadata));
  if (result != MGPU_CONTEXT_LEASE_OK) return result;
  profile.struct_size = sizeof(profile); profile.version = 1; profile.enforcement_revision = 2;
  profile.flags = MGPU_CAPTURE_SETUP_ORIGINAL_TICKET_ENFORCED | MGPU_CAPTURE_SETUP_EXPLICIT_CREATE_OUTCOME;
  auto &layout = profile.setup;
  layout.struct_size = sizeof(layout); layout.version = layout.layout_revision = 1;
  layout.kind_count = layout.allocation_count = MGPU_CAPTURE_SETUP_KIND_COUNT_V1;
  layout.flags = MGPU_CAPTURE_SETUP_LAYOUT_FUNDED_REQUESTS;
  layout.total_requested_bytes = actual.totalRequestedBytes;
  const auto &build = actual.build;
  layout.compiler_full_version = build.compilerFullVersion; layout.stl_version = build.stlVersion;
  layout.stl_update = build.stlUpdate; layout.iterator_debug_level = build.iteratorDebugLevel;
  layout.language_version = build.languageVersion; layout.build_flags = build.buildFlags;
  layout.pointer_bytes = build.pointerBytes; layout.default_new_alignment = build.defaultNewAlignment;
  for (uint32_t i = 0; i < layout.kind_count; ++i) {
    const auto shared = setupAllocationKind(i) == MGPU_CAPTURE_SETUP_DEFAULT_SHARED;
    layout.kinds[i] = {i, static_cast<uint32_t>(shared ? MGPU_CAPTURE_SETUP_FUNDED_SHARED : MGPU_CAPTURE_SETUP_FUNDED_UNIQUE),
        actual.kinds[i].bytes, actual.kinds[i].alignment, 0};
  }
  *out = profile;
  return MGPU_CONTEXT_LEASE_OK;
}
namespace mgpu::setup_custody {
int retainAuthority(const MGPUSetupAuthorityV1 *desc, uint32_t size, SetupTickets &out) noexcept {
  if (!desc || size < sizeof(*desc) || desc->struct_size != sizeof(*desc) || desc->version != 1 ||
      !desc->context || !desc->ops || !desc->ops->retain || !desc->ops->release || !desc->ops->event ||
      !desc->generation || desc->role_count != MGPU_CAPTURE_SETUP_KIND_COUNT_V1 || desc->reserved)
    return MGPU_CONTEXT_LEASE_INVALID;
  for (const auto &ticket : out.roles) if (ticket) return MGPU_CONTEXT_LEASE_INVALID;
  allocation_layout::FundedSetupLayout actual{};
  if (!allocation_layout::fundedSetupLayout(actual)) return MGPU_CONTEXT_LEASE_UNSUPPORTED;
  // Validate EVERY role before any callback. Same-size default allocator blocks
  // are still rejected if they do not match these exact funded requests.
  for (uint32_t i = 0; i < desc->role_count; ++i) if (desc->roles[i].reserved ||
      desc->roles[i].bytes != actual.kinds[i].bytes || desc->roles[i].alignment != actual.kinds[i].alignment)
    return MGPU_CONTEXT_LEASE_INVALID;
  for (uint32_t i = 0; i < desc->role_count; ++i) out.roles[i] = SetupTicket::retain(desc->context, desc->ops,
      {desc->generation, desc->row, i}, actual.kinds[i]);
  return out.valid() ? MGPU_CONTEXT_LEASE_OK : MGPU_CONTEXT_LEASE_INVALID;
}
}
namespace {
int createSource(MGPUContextLease *original, const MGPUCaptureSourceDesc *desc, uint32_t size,
    MGPUCaptureSource **out, MGPUContextLease **borrowed, const mgpu::setup_custody::SetupTickets *funding,
    uint32_t *outcome = nullptr) noexcept {
  if (outcome) *outcome = MGPU_CAPTURE_SETUP_OUTCOME_UNKNOWN;
  bool sourceAttempted = false;
  const auto finite = [outcome](int result) noexcept {
    if (outcome) *outcome = MGPU_CAPTURE_SETUP_OUTCOME_FINITE_REFUSED;
    return result;
  };
  if (out) *out = nullptr;
  if (borrowed) *borrowed = nullptr;
  if (!out || !borrowed || !original) return finite(MGPU_CONTEXT_LEASE_INVALID);
  const auto valid = mgpu::capture_source::validate(desc, size);
  if (valid != 0) return finite(valid);
  if (funding && (!funding->valid() || !mgpu::setup_custody::matchesOriginalRetain(original, *funding)))
    return finite(MGPU_CONTEXT_LEASE_INVALID);
#if !defined(_WIN32) || defined(__EMSCRIPTEN__)
  return finite(MGPU_CONTEXT_LEASE_UNSUPPORTED);
#else
  try {
    const auto pin = mgpu::copyContextLeasePin(original);
    if (!pin || pin->sourceDomain() || pin->contextGeneration() != desc->context_generation ||
        pin->deviceGeneration() != desc->device_generation) return finite(MGPU_CONTEXT_LEASE_INVALID);
    if (!pin->current()) return finite(MGPU_CONTEXT_LEASE_RETIRED);
    if (funding && !pin->setupFunded()) return finite(MGPU_CONTEXT_LEASE_UNSUPPORTED);
    mgpu::capture_source::PreparedSource prepared;
    const auto reserved = funding ? prepared.prepareFunded(*desc,
        funding->roles[MGPU_CAPTURE_SETUP_PREPARED_STATE], funding->roles[MGPU_CAPTURE_SETUP_PREPARED_ANCHOR]) : prepared.prepare(*desc);
    if (reserved != 0) return reserved == MGPU_CONTEXT_LEASE_QUARANTINED ? reserved : finite(reserved);
    auto box = funding ? mgpu::setup_custody::makeSetupUnique<MGPUCaptureSource>(
        funding->roles[MGPU_CAPTURE_SETUP_SOURCE_BOX], prepared.owner()) :
        mgpu::setup_custody::SetupUnique<MGPUCaptureSource>(new MGPUCaptureSource({}, prepared.owner()));
    // Synchronous only. All retained tickets are inline in real owners; this
    // adapter and the caller's ticket array are never retained by a source.
    FundedCreate adapter{original, funding};
    sourceAttempted = true;
    const auto result = funding ? prepared.create(&adapter, nativeCreateFunded, nativeRelease) :
        prepared.create(original, nativeCreate, nativeRelease);
    if (result == 0) *borrowed = prepared.borrowedLease();
    if (result == 0 || result == MGPU_CONTEXT_LEASE_QUARANTINED) *out = box.release();
    if (result != 0 && prepared.knownUnpublished()) return finite(result);
    return result;
  } catch (mgpu::setup_custody::Error error) {
    return error == mgpu::setup_custody::Error::Uncertain ? MGPU_CONTEXT_LEASE_QUARANTINED :
        (sourceAttempted ? MGPU_CONTEXT_LEASE_INVALID : finite(MGPU_CONTEXT_LEASE_INVALID));
  } catch (...) { return sourceAttempted ? MGPU_CONTEXT_LEASE_INTERNAL : finite(MGPU_CONTEXT_LEASE_INTERNAL); }
#endif
}
}
extern "C" int mgpuCaptureSourceCreateScoped(MGPUContextLease *original,
    const MGPUCaptureSourceDesc *desc, uint32_t size, MGPUCaptureSource **out, MGPUContextLease **borrowed) {
  return createSource(original, desc, size, out, borrowed, nullptr);
}
extern "C" int mgpuCaptureSourceCreateScopedFundedV1(MGPUContextLease *original,
    const MGPUCaptureSourceDesc *desc, uint32_t size, const MGPUSetupAuthorityV1 *authority,
    uint32_t authoritySize, MGPUCaptureSource **out, MGPUContextLease **borrowed) {
  if (out) *out = nullptr;
  if (borrowed) *borrowed = nullptr;
  if (!out || !borrowed || !original) return MGPU_CONTEXT_LEASE_INVALID;
  const auto valid = mgpu::capture_source::validate(desc, size);
  if (valid != MGPU_CONTEXT_LEASE_OK) return valid;
  mgpu::setup_custody::SetupTickets tickets;
  const auto retained = mgpu::setup_custody::retainAuthority(authority, authoritySize, tickets);
  if (retained != MGPU_CONTEXT_LEASE_OK) return retained;
  return createSource(original, desc, size, out, borrowed, &tickets);
}
static_assert(sizeof(MGPUCaptureSetupCreateOutcomeV1) == 64 &&
    offsetof(MGPUCaptureSetupCreateOutcomeV1, generation) == 16 &&
    offsetof(MGPUCaptureSetupCreateOutcomeV1, context_generation) == 32);
extern "C" int mgpuCaptureSourceCreateScopedFundedOutcomeV1(MGPUContextLease *original,
    const MGPUCaptureSourceDesc *desc, uint32_t size, const MGPUSetupAuthorityV1 *authority,
    uint32_t authoritySize, MGPUCaptureSource **out, MGPUContextLease **borrowed,
    MGPUCaptureSetupCreateOutcomeV1 *outcome, uint32_t outcomeSize) {
  if (out) *out = nullptr;
  if (borrowed) *borrowed = nullptr;
  if (!outcome || outcomeSize < sizeof(*outcome)) return MGPU_CONTEXT_LEASE_INVALID;
  *outcome = {};
  if (!out || !borrowed || !original || mgpu::capture_source::validate(desc, size) != 0)
    return MGPU_CONTEXT_LEASE_INVALID;
  mgpu::setup_custody::SetupTickets tickets;
  const auto retained = mgpu::setup_custody::retainAuthority(authority, authoritySize, tickets);
  if (retained != MGPU_CONTEXT_LEASE_OK) return retained;
  uint32_t disposition = MGPU_CAPTURE_SETUP_OUTCOME_UNKNOWN;
  const auto result = createSource(original, desc, size, out, borrowed, &tickets, &disposition);
  uint64_t sourceId = 0;
  if (result == 0 && *out && *borrowed) {
    MGPUCaptureSourceInfo info{};
    if (mgpu::capture_source::getInfo((*out)->owner, info) == 0 && info.source_id &&
        info.context_generation == desc->context_generation && info.device_generation == desc->device_generation) {
      disposition = MGPU_CAPTURE_SETUP_OUTCOME_SOURCE_OWNED; sourceId = info.source_id;
    }
  }
  *outcome = {sizeof(*outcome), 1, disposition, 0, authority->generation, authority->row,
      result, desc->context_generation, desc->device_generation, sourceId, 0};
  return result;
}
namespace mgpu::capture_source {
int createScopedFunded(MGPUContextLease *original, const MGPUCaptureSourceDesc &desc,
    const setup_custody::SetupTickets &funding, MGPUCaptureSource **out, MGPUContextLease **borrowed) noexcept {
  return createSource(original, &desc, sizeof(desc), out, borrowed, &funding);
}
}
namespace mgpu::allocation_layout {
bool fundedSetupLayout(FundedSetupLayout &out) noexcept {
  FundedSetupLayoutBuilder builder;
  fundedRootSetupLayout(builder); fundedPreparedSetupLayout(builder); fundedContextSetupLayout(builder);
  builder.unique<MGPUCaptureSource>(MGPU_CAPTURE_SETUP_SOURCE_BOX);
  return builder.finish(out);
}
}
extern "C" int mgpuCaptureSourceGetInfo(MGPUCaptureSource *handle, MGPUCaptureSourceInfo *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  *out = {};
  if (!handle) return MGPU_CONTEXT_LEASE_INVALID;
  return mgpu::capture_source::getInfo(handle->owner, *out);
}
extern "C" int mgpuCaptureSourceSeal(MGPUCaptureSource *handle) {
  return handle ? mgpu::capture_source::seal(handle->owner) : MGPU_CONTEXT_LEASE_INVALID;
}
extern "C" int mgpuCaptureSourceTryReleaseJoined(MGPUCaptureSource *handle,
    MGPUCaptureSourceJoined *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  *out = {};
  if (!handle) return MGPU_CONTEXT_LEASE_INVALID;
  auto owner = handle->owner; // Real C-call tail keeps global quota through return.
  const auto result = mgpu::capture_source::tryReleaseJoined(owner, *out);
  if (result == 0) mgpu::setup_custody::SetupDelete<MGPUCaptureSource>{}(handle);
  return result;
}
