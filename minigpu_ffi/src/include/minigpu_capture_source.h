#ifndef MINIGPU_CAPTURE_SOURCE_H
#define MINIGPU_CAPTURE_SOURCE_H
#include "minigpu_context_lifetime.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Private source ownership only. Neither these symbols nor a joined receipt
 * grants capture, renderer qualification or resource-budget authority. The
 * trusted host prepays metadata and supplies its exact retained original lease.
 * No context/device initialization or source selection occurs here. */
typedef struct MGPUCaptureSource MGPUCaptureSource;
typedef struct MGPUCaptureSourceDesc {
  uint32_t struct_size, version;
  uint32_t width, height, recipe_revision, reserved;
  uint64_t context_generation, device_generation, frame_bytes;
} MGPUCaptureSourceDesc;
typedef enum MGPUCaptureSourcePhase {
  MGPU_CAPTURE_SOURCE_CREATING = 0, MGPU_CAPTURE_SOURCE_OPEN = 1,
  MGPU_CAPTURE_SOURCE_SEALED = 2, MGPU_CAPTURE_SOURCE_DRAINING = 3,
  MGPU_CAPTURE_SOURCE_QUARANTINED = 4, MGPU_CAPTURE_SOURCE_JOINED = 5
} MGPUCaptureSourcePhase;
typedef struct MGPUCaptureSourceInfo {
  uint32_t struct_size, version, phase, flags;
  uint64_t source_id, context_generation, device_generation, frame_bytes;
  uint32_t width, height, recipe_revision, reserved;
} MGPUCaptureSourceInfo;
typedef struct MGPUCaptureSourceJoined {
  uint32_t struct_size, version;
  uint64_t source_id, context_generation, device_generation;
} MGPUCaptureSourceJoined;

/* Exact partial logical inventory, NOT a complete native/physical cost profile.
 * Revision4/coverage15 fixes 31 allocation kinds with 60 simultaneous slots:
 * OperationState,
 * FamilyState, GlobalTaskCustody, PrimaryJob/PrimaryAnchor, FenceJob/FenceAnchor,
 * WaitReceipt/WaitAnchor, PreparedWaitHolder, TicketCustody/TicketState/TicketBox.
 * PrimaryJob is shared by WorkJob, CleanupJob and private-resource Job, never
 * three extra capacities. Coverage bit0 is the two operation records; bit1 is
 * this fixed eleven-role task/ticket group. Bit2 is ImportedFrameState,
 * ImportedFrameNative and ImportedFrameBox. Resource allocations retain their
 * source without occupying an operation. Bit3 adds PrivateResourceState (3),
 * Buffer CPU (2), Shader CPU (1), ExclusiveUse State/Box (1 each), Writer
 * State/Box (1 each), Output State/Box (2 each), Loan State/Box/independent
 * anchor (2 each), and Borrow State/Box/independent anchor (8 each).
 * All unqualified caps above are 1. Each exact unaccepted resource claim may
 * issue its fixed allocation batch only once. Caps include control-block/free
 * tails. ScopedPin/setup records are already separately paid; no double charge.
 * Covered jobs now embed results/callables/queue entries. Exact scoped recipe
 * payloads use 16 inline parameter bytes, three binding slots and static WGSL;
 * this removes those separate heaps, not all legacy/global/platform storage,
 * allocator-byte or driver costs. Missing/unknown profiles
 * must refuse; profile evidence is neither original consent nor Start authority. */
typedef struct MGPUCaptureSourceLogicalMetadataProfileV1 {
  uint32_t struct_size, version, profile_revision, coverage;
  uint32_t operation_state_cap, family_state_cap;
  uint64_t reserved;
} MGPUCaptureSourceLogicalMetadataProfileV1;
/* Metadata-only original-module query; never initializes a context or device. */
EXPORT int mgpuCaptureSourceGetLogicalMetadataProfileV1(
    MGPUCaptureSourceLogicalMetadataProfileV1 *, uint32_t);

/* Allocation-request layout ONLY, not enforcement, consent, a budget or Start
 * authority. Exact supported compiler/STL/configuration; otherwise UNSUPPORTED.
 * No payload/owner construction or native initialization. Missing/short output
 * storage refuses without mutation; other refusal clears the complete record.
 * IDs are versioned and independent of private C++ type names. */
typedef enum MGPUCaptureMetadataKindV1 {
  MGPU_CAPTURE_META_OPERATION_STATE = 0, MGPU_CAPTURE_META_FAMILY_STATE,
  MGPU_CAPTURE_META_GLOBAL_TASK, MGPU_CAPTURE_META_PRIMARY_JOB, MGPU_CAPTURE_META_PRIMARY_ANCHOR,
  MGPU_CAPTURE_META_FENCE_JOB, MGPU_CAPTURE_META_FENCE_ANCHOR,
  MGPU_CAPTURE_META_WAIT_RECEIPT, MGPU_CAPTURE_META_WAIT_ANCHOR, MGPU_CAPTURE_META_WAIT_HOLDER,
  MGPU_CAPTURE_META_TICKET_CUSTODY, MGPU_CAPTURE_META_TICKET_STATE, MGPU_CAPTURE_META_TICKET_BOX,
  MGPU_CAPTURE_META_FRAME_STATE, MGPU_CAPTURE_META_FRAME_NATIVE, MGPU_CAPTURE_META_FRAME_BOX,
  MGPU_CAPTURE_META_RESOURCE_STATE, MGPU_CAPTURE_META_BUFFER_CPU, MGPU_CAPTURE_META_SHADER_CPU,
  MGPU_CAPTURE_META_USE_STATE, MGPU_CAPTURE_META_USE_BOX,
  MGPU_CAPTURE_META_WRITER_STATE, MGPU_CAPTURE_META_WRITER_BOX,
  MGPU_CAPTURE_META_OUTPUT_STATE, MGPU_CAPTURE_META_OUTPUT_BOX,
  MGPU_CAPTURE_META_LOAN_STATE, MGPU_CAPTURE_META_LOAN_BOX, MGPU_CAPTURE_META_LOAN_ANCHOR,
  MGPU_CAPTURE_META_BORROW_STATE, MGPU_CAPTURE_META_BORROW_BOX, MGPU_CAPTURE_META_BORROW_ANCHOR,
  MGPU_CAPTURE_META_KIND_COUNT_V1
} MGPUCaptureMetadataKindV1;
enum {
  MGPU_CAPTURE_META_VARIANT_GENERIC = 1, MGPU_CAPTURE_META_VARIANT_WORK = 2,
  MGPU_CAPTURE_META_VARIANT_CLEANUP = 3, MGPU_CAPTURE_META_VARIANT_PRIVATE = 4,
  MGPU_CAPTURE_META_LAYOUT_REQUEST_SIZES = 1 /* No enforcement bit exists in V1. */
};
typedef struct MGPUCaptureMetadataExtentV1 {
  uint64_t bytes;
  uint32_t alignment, variant;
} MGPUCaptureMetadataExtentV1;
typedef struct MGPUCaptureMetadataKindLayoutV1 {
  uint32_t kind, capacity, extent_count, reserved;
  MGPUCaptureMetadataExtentV1 extents[3];
} MGPUCaptureMetadataKindLayoutV1;
typedef struct MGPUCaptureMetadataLayoutV1 {
  uint32_t struct_size, version, layout_revision, kind_count;
  uint32_t slot_count, flags;
  uint64_t total_requested_bytes; /* Sum(capacity * max(explicit variants)). */
  uint32_t compiler_full_version, stl_version, stl_update, iterator_debug_level;
  uint32_t language_version, build_flags, pointer_bytes, default_new_alignment;
  uint64_t reserved;
  MGPUCaptureMetadataKindLayoutV1 kinds[MGPU_CAPTURE_META_KIND_COUNT_V1];
} MGPUCaptureMetadataLayoutV1;
EXPORT int mgpuCaptureSourceGetMetadataLayoutV1(MGPUCaptureMetadataLayoutV1 *, uint32_t);

/* Original-module attestation of the COVERED metadata subset only. Every scoped
 * source binds this validated table before its first claim and enforces exact
 * requested extents plus per-kind/aggregate byte limits through actual weak/free
 * tails; uncertain disposition retains debt. The layout's own flag remains
 * REQUEST_SIZES. No setup13/Rust allocation, allocator overhead, module/OS/driver
 * memory, consent, host budget or renderer/Start authority is attested here.
 * Query is stack/POD only, with no owner construction or native initialization.
 * Missing/short output is untouched; other refusal clears the complete record. */
enum { MGPU_CAPTURE_CPU_METADATA_ENFORCED = 1 };
typedef struct MGPUCaptureSourceCpuMetadataProfileV1 {
  uint32_t struct_size, version, enforcement_revision, flags;
  MGPUCaptureSourceLogicalMetadataProfileV1 logical;
  MGPUCaptureMetadataLayoutV1 layout;
  uint64_t reserved;
} MGPUCaptureSourceCpuMetadataProfileV1;
EXPORT int mgpuCaptureSourceGetCpuMetadataProfileV1(
    MGPUCaptureSourceCpuMetadataProfileV1 *, uint32_t);

/* Requested-size reporting ONLY for the native10 portion of existing setup13.
 * Each role is one allocation, including the actual default make_shared block.
 * These allocations are not part of metadata31/60. Rust Arc3/control/shared-
 * engine/OS/driver bytes, funding, enforcement and final-free/refund proof are
 * NOT supplied. A ContextOwner's weak default-registry tail can outlive source
 * join; that join alone must not refund future setup bytes. */
typedef enum MGPUCaptureSetupKindV1 {
  MGPU_CAPTURE_SETUP_ROOT_STATE = 0, MGPU_CAPTURE_SETUP_ROOT_BOX,
  MGPU_CAPTURE_SETUP_PREPARED_STATE, MGPU_CAPTURE_SETUP_PREPARED_ANCHOR,
  MGPU_CAPTURE_SETUP_SOURCE_BOX, MGPU_CAPTURE_SETUP_ORIGINAL_LEASE,
  MGPU_CAPTURE_SETUP_ORIGINAL_PIN, MGPU_CAPTURE_SETUP_CONTEXT_OWNER,
  MGPU_CAPTURE_SETUP_SCOPED_LEASE, MGPU_CAPTURE_SETUP_SCOPED_PIN,
  MGPU_CAPTURE_SETUP_KIND_COUNT_V1
} MGPUCaptureSetupKindV1;
enum {
  MGPU_CAPTURE_SETUP_DEFAULT_UNIQUE = 1, MGPU_CAPTURE_SETUP_DEFAULT_SHARED = 2,
  MGPU_CAPTURE_SETUP_LAYOUT_REQUEST_SIZES = 1
};
typedef struct MGPUCaptureSetupExtentV1 {
  uint32_t kind, allocation_kind;
  uint64_t bytes;
  uint32_t alignment, reserved; /* Actual object/block requirement; ordinary new may over-align. */
} MGPUCaptureSetupExtentV1;
typedef struct MGPUCaptureSetupLayoutV1 {
  uint32_t struct_size, version, layout_revision, kind_count;
  uint32_t allocation_count, flags;
  uint64_t total_requested_bytes;
  uint32_t compiler_full_version, stl_version, stl_update, iterator_debug_level;
  uint32_t language_version, build_flags, pointer_bytes, default_new_alignment;
  uint64_t reserved;
  MGPUCaptureSetupExtentV1 kinds[MGPU_CAPTURE_SETUP_KIND_COUNT_V1];
} MGPUCaptureSetupLayoutV1;
/* Stack/POD only; unsupported build refuses. Null/short untouched, other
 * refusal clears the known record. No native initialization or owner creation. */
EXPORT int mgpuCaptureSourceGetSetupAllocationLayoutV1(MGPUCaptureSetupLayoutV1 *, uint32_t);

/* All output cells are cleared after their required storage is validated.
 * OK returns an owner AND borrowed scoped lease. The owner owns that lease;
 * mgpuContextReleaseLease cannot release it. Original lease remains caller-owned.
 * A nonnull owner on ANY failure is unresolved custody, never retry authority.
 * Known finite refusal with null outputs has accepted no native source owner. */
EXPORT int mgpuCaptureSourceCreateScoped(MGPUContextLease *original,
    const MGPUCaptureSourceDesc *desc, uint32_t desc_size,
    MGPUCaptureSource **out_owner, MGPUContextLease **out_borrowed_lease);
EXPORT int mgpuCaptureSourceGetInfo(MGPUCaptureSource *, MGPUCaptureSourceInfo *, uint32_t);
/* CPU-only synchronous fence; already admitted operations keep their real tails.
 * This may overlap info/seal, but caller must retain the public owner throughout. */
EXPORT int mgpuCaptureSourceSeal(MGPUCaptureSource *);
/* Original worker, exclusive against all uses of its borrowed lease and final
 * public-owner consumption. Require prior Seal and actual child retirement.
 * First call releases the owned lease ONCE. WOULD_BLOCK retains the owner while
 * actual pin/member/task/ticket aliases prevent join; do not use the borrowed
 * lease again after calling. OK alone consumes owner and returns the exact
 * joined receipt. Unknown/failure retains actual owners; never timeout/refund. */
EXPORT int mgpuCaptureSourceTryReleaseJoined(MGPUCaptureSource *, MGPUCaptureSourceJoined *, uint32_t);
#ifdef __cplusplus
}
#endif
#endif
