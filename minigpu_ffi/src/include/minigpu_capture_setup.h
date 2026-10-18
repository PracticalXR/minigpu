#ifndef MINIGPU_CAPTURE_SETUP_H
#define MINIGPU_CAPTURE_SETUP_H
#include "minigpu_capture_source.h"
#include "minigpu_setup_custody_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Trusted host only. This profile describes actual funded native10 plus the
 * covered metadata60. It excludes Rust3, shared host/engine, allocator overhead,
 * platform/driver/WGC costs, source authentication and capture/Start authority.
 * Exact requested extents are enforced by the funded entrypoints and original
 * host callbacks; the ORIGINAL host budget must reserve the full source quote
 * before every covered allocation. Default V1 reporting is not this profile. */
enum {
  MGPU_CAPTURE_SETUP_FUNDED_UNIQUE = 3, MGPU_CAPTURE_SETUP_FUNDED_SHARED = 4,
  MGPU_CAPTURE_SETUP_LAYOUT_FUNDED_REQUESTS = 2,
  MGPU_CAPTURE_SETUP_ORIGINAL_TICKET_ENFORCED = 1,
  MGPU_CAPTURE_SETUP_EXPLICIT_CREATE_OUTCOME = 2
};
typedef struct MGPUCaptureFundedSetupProfileV1 {
  uint32_t struct_size, version, enforcement_revision, flags;
  MGPUCaptureSetupLayoutV1 setup;
  MGPUCaptureSourceCpuMetadataProfileV1 metadata;
  uint64_t reserved;
} MGPUCaptureFundedSetupProfileV1;
enum MGPUCaptureSetupDispositionV1 {
  MGPU_CAPTURE_SETUP_OUTCOME_UNKNOWN = 0,
  MGPU_CAPTURE_SETUP_OUTCOME_FINITE_REFUSED = 1,
  MGPU_CAPTURE_SETUP_OUTCOME_SOURCE_OWNED = 2
};
typedef struct MGPUCaptureSetupCreateOutcomeV1 {
  uint32_t struct_size, version, disposition, reserved;
  uint64_t generation;
  uint32_t row;
  int32_t native_result;
  uint64_t context_generation, device_generation, source_id, reserved_tail;
} MGPUCaptureSetupCreateOutcomeV1;
/* Exact same-module synchronous outcome. FINITE_REFUSED proves no source
 * obligation survives this call, only after known cleanup; CPU/creator-owner
 * tails remain separately charged until actual frees. SOURCE_OWNED supplies
 * cleanup identity even if later admission/queries fail. UNKNOWN is never a
 * refund receipt. The result is bound to original row/generation and retained
 * descriptor generations, not guest input. Short/missing outcome storage
 * refuses BEFORE any invocation and is untouched. No callback/timeout inference. */
EXPORT int mgpuCaptureSourceCreateScopedFundedOutcomeV1(MGPUContextLease *,
    const MGPUCaptureSourceDesc *, uint32_t, const MGPUSetupAuthorityV1 *, uint32_t,
    MGPUCaptureSource **, MGPUContextLease **, MGPUCaptureSetupCreateOutcomeV1 *, uint32_t);
/* Stack/POD only: no owner/context construction or initialization. Null/short
 * storage is untouched. Other refusal clears the complete known record. */
EXPORT int mgpuCaptureSourceGetFundedSetupProfileV1(MGPUCaptureFundedSetupProfileV1 *, uint32_t);
/* Synchronous trusted in-process routing. Callbacks/static ops MUST remain pinned
 * independently of this DLL; never pass pointers from guests/room metadata.
 * Descriptor extents must equal this module's complete current funded native10.
 * Refuses legacy/unfunded owners; never initializes or selects a device. Both
 * outputs clear on refusal. Actual context generations come from retained lease,
 * not from a prepaid ticket. Caller must fence admission/consent/source entry. */
EXPORT int mgpuDefaultContextRetainLeaseFundedV1(const MGPUSetupAuthorityV1 *, uint32_t,
    MGPUContextLease **, MGPUContextHandle **);
/* Original retained lease and grouped Create share the same authority/row/gen.
 * Only an existing ContextOwner may keep another creator's original charge.
 * Preserve normal source join/borrowed lease/quarantine contracts. Native
 * allocation receipts do not prove source join or known unpublished failure. */
EXPORT int mgpuCaptureSourceCreateScopedFundedV1(MGPUContextLease *,
    const MGPUCaptureSourceDesc *, uint32_t, const MGPUSetupAuthorityV1 *, uint32_t,
    MGPUCaptureSource **, MGPUContextLease **);
#ifdef __cplusplus
}
#endif
#endif
