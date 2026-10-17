#ifndef MINIGPU_PRIVATE_OUTPUT_H
#define MINIGPU_PRIVATE_OUTPUT_H
#include "minigpu_context_lifetime.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGPUPrivateOutputWriter MGPUPrivateOutputWriter;
typedef struct MGPUPrivateOutput MGPUPrivateOutput;
typedef struct MGPUPrivatePreviewLoan MGPUPrivatePreviewLoan;
typedef struct MGPUPrivatePreviewBorrow MGPUPrivatePreviewBorrow;

enum MGPUPrivateOutputPhase {
  MGPU_PRIVATE_OUTPUT_EMPTY = 1,
  MGPU_PRIVATE_OUTPUT_READY = 2,
  MGPU_PRIVATE_OUTPUT_COPYING = 3,
  MGPU_PRIVATE_OUTPUT_RETIRING = 4,
  MGPU_PRIVATE_OUTPUT_RETIRED = 5,
  MGPU_PRIVATE_OUTPUT_QUARANTINED = 6
};
enum MGPUPrivateOutputKind {
  MGPU_PRIVATE_OUTPUT_WRITER = 1,
  MGPU_PRIVATE_OUTPUT_SNAPSHOT = 2,
  MGPU_PRIVATE_OUTPUT_PREVIEW = 3
};
enum MGPUPrivateOutputFlags {
  MGPU_PRIVATE_OUTPUT_SEALED = 1,
  MGPU_PRIVATE_OUTPUT_PREVIEW_CREATED = 2
};
typedef struct MGPUPrivateOutputInfo {
  uint32_t struct_size, version, kind, phase;
  int32_t last_result;
  uint32_t flags;
  uint64_t resource_id, context_generation, device_generation, charged_bytes;
  uint32_t width, height, pending_operations, pending_borrows;
  uint64_t parent_output_id;
} MGPUPrivateOutputInfo; /* V1: 80 bytes; metadata, never authority */
typedef struct MGPUPrivateOutputStats {
  uint32_t struct_size, version, writers, outputs, preview_loans, descriptor_borrows;
  uint32_t pending_jobs, max_writers, max_outputs, max_preview_loans;
  uint32_t max_descriptor_borrows, reserved;
  uint64_t charged_bytes, max_bytes, max_frame_bytes;
} MGPUPrivateOutputStats; /* V1: 72 bytes */
typedef struct MGPUPrivatePreviewBorrowInfo {
  uint32_t struct_size, version;
  uint64_t output_id, loan_id, borrow_id;
  uint32_t width, height, pixel_format, reserved;
  uint64_t legacy_shared_handle, device_generation;
} MGPUPrivatePreviewBorrowInfo; /* V1: 64 bytes; pixel_format=1 BGRA8888 */

/* Trusted-host resource primitives only. No capture/source, input, encode,
 * publication or private-ready authority. Never serialize pointers/handles or
 * these native descriptors to guests/room messages. All objects must come from
 * the same installed module. Pin the code while objects/jobs/callbacks live.
 * Final Release must not race calls on that same raw pointer.
 *
 * First native profile: original ready D3D11 device, BGRA storage support,
 * fixed dimensions. Writer retains a caller-owned packed RGBA8 u32 buffer
 * (R low8, G next8, then B,A), exact width*height*4 bytes. Caller exclusively
 * owns it and prevents explicit BufferDestroy, CPU/map access or competing
 * writes through each completed Write; retain it until WriterRetire succeeds.
 * Writer reuses its pipeline/intermediate/bind group; each output is written
 * once and published at most once, permanently immutable. No slot recycling.
 *
 * Factories are synchronous worker operations; never call on UI/realtime or
 * the creation GPU worker. They do not initialize/migrate/select adapters or
 * fall back. Outputs clear on refusal, except QUARANTINED returns a NONNULL
 * charged owner: retain it, including after lost source authorization.
 * 64 writers/256 outputs/256 preview loans/1024 descriptor borrows (4 per loan),
 * 512 pending jobs and 512MiB
 * combined retained logical image bytes, at most 128MiB per image. These are
 * additional bounds, not physical GPU accounting or participant limits. Host
 * admission separately includes input, packed buffers and other allocations.
 * Non-Windows/Emscripten refuse without GPU work. */
EXPORT int mgpuPrivateOutputWriterCreate(
    MGPUContextLease *, MGPUBuffer *, uint32_t width, uint32_t height,
    MGPUPrivateOutputWriter **out_writer);
EXPORT int mgpuPrivateOutputCreate(
    MGPUPrivateOutputWriter *, MGPUPrivateOutput **out_output);
/* Callback rules match the native queue callback. Null callback touches nothing.
 * One callback per other call, including inline refusal, always outside locks/
 * Dawn events. OK proves checked producer access/submission/completion/EndAccess,
 * not publication authority or consumer completion. Keep buffer ownership through
 * callback return. Unknown accepted execution retains both owners in quarantine.
 * Already admitted writes may finish after Seal; no later publication is allowed. */
EXPORT void mgpuPrivateOutputWriteAsync(
    MGPUPrivateOutputWriter *, MGPUPrivateOutput *,
    MGPUQueueCompletionCallback, void *userdata);
EXPORT int mgpuPrivateOutputWriterSeal(MGPUPrivateOutputWriter *);
EXPORT int mgpuPrivateOutputSeal(MGPUPrivateOutput *);
/* Retirement seals first. Busy work reports WOULD_BLOCK; retry only after that
 * work acknowledges. Successful retirement releases this producer's actual native
 * references on its exact worker. This is not physical reclamation or consumer
 * retirement. Loss alone is no completion proof. No timeout/direct fallback.
 * Known unused/previously completed old-generation refs may be released.
 * Final Release consumes only RETIRED owners. Output Release may precede preview
 * removal: its allocation/pin/bytes stay charged until the last loan closes. */
EXPORT void mgpuPrivateOutputWriterRetireAsync(
    MGPUPrivateOutputWriter *, MGPUQueueCompletionCallback, void *userdata);
EXPORT void mgpuPrivateOutputRetireAsync(
    MGPUPrivateOutput *, MGPUQueueCompletionCallback, void *userdata);
EXPORT int mgpuPrivateOutputWriterRelease(MGPUPrivateOutputWriter *);
EXPORT int mgpuPrivateOutputRelease(MGPUPrivateOutput *);
EXPORT int mgpuPrivateOutputWriterGetInfo(
    MGPUPrivateOutputWriter *, MGPUPrivateOutputInfo *, uint32_t out_size);
EXPORT int mgpuPrivateOutputGetInfo(
    MGPUPrivateOutput *, MGPUPrivateOutputInfo *, uint32_t out_size);
EXPORT int mgpuPrivateOutputGetStats(MGPUPrivateOutputStats *, uint32_t out_size);

/* One independently retained registration loan per successful immutable output.
 * No raw handle getter without a descriptor borrow. Owning this loan does NOT
 * consume/refund its allocation charge. SDK must retain the original source/
 * consent and module through confirmed preview removal and all borrow cleanup.
 * Producer retirement may proceed: registration owns an independent COM ref. */
EXPORT int mgpuPrivateOutputAcquirePreviewLoan(
    MGPUPrivateOutput *, MGPUPrivatePreviewLoan **out_loan);
EXPORT int mgpuPrivatePreviewLoanGetInfo(
    MGPUPrivatePreviewLoan *, MGPUPrivateOutputInfo *, uint32_t out_size);
/* Each descriptor owns an independent COM borrow. Keep it through Flutter's
 * release_callback (including binding failure), then call BorrowConsumed once.
 * That receipt is handle-borrow consumption, NOT display/GPU completion. The
 * legacy shared handle is not an NT handle: never CloseHandle/DuplicateHandle.
 * Never write/recycle the immutable allocation while any consumer may retain it. */
EXPORT int mgpuPrivatePreviewBorrowDescriptor(
    MGPUPrivatePreviewLoan *, MGPUPrivatePreviewBorrow **out_borrow,
    MGPUPrivatePreviewBorrowInfo *, uint32_t out_size);
EXPORT int mgpuPrivatePreviewBorrowConsumed(MGPUPrivatePreviewBorrow *);
EXPORT int mgpuPrivatePreviewLoanSeal(MGPUPrivatePreviewLoan *);
EXPORT int mgpuPrivatePreviewLoanQuarantine(MGPUPrivatePreviewLoan *);
/* TRUSTED PRESENTER PRECONDITION: gate closed and actual registration removal
 * independently confirmed by a qualified consumer profile. IDs cannot prove it;
 * stock present/unregister callbacks alone do not. This function checks only
 * identity, sealed state and zero live descriptor borrows, then releases its own
 * COM reference. OK consumes the loan. Failed/ambiguous cleanup retains the
 * owner and charge; do not retry partially completed native cleanup. Never call
 * merely because a timeout or unknown callback occurred. Ambiguous removal
 * must quarantine instead. */
EXPORT int mgpuPrivatePreviewLoanReleaseAfterRemoval(
    MGPUPrivatePreviewLoan *, uint64_t expected_loan_id);

#ifdef __cplusplus
}
#endif
#endif
