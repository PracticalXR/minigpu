#ifndef MINIGPU_COMPLETION_TICKET_H
#define MINIGPU_COMPLETION_TICKET_H
#include "minigpu_context_lifetime.h"
#include "minigpu_imported_frame.h"
#include "minigpu_private_output.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGPUCompletionTicket MGPUCompletionTicket;
enum MGPUCompletionKind {
  MGPU_COMPLETION_WRITE_OUTPUT = 1, MGPU_COMPLETION_RETIRE_USE = 2,
  MGPU_COMPLETION_RETIRE_FRAME = 3, MGPU_COMPLETION_RETIRE_OUTPUT = 4,
  MGPU_COMPLETION_RETIRE_WRITER = 5, MGPU_COMPLETION_DESTROY_SHADER = 6,
  MGPU_COMPLETION_DESTROY_BUFFER = 7
};
enum MGPUCompletionPhase {
  MGPU_COMPLETION_PENDING = 1, MGPU_COMPLETION_CALLBACK = 2,
  MGPU_COMPLETION_COMPLETE = 3, MGPU_COMPLETION_COLLECTED = 4,
  MGPU_COMPLETION_QUARANTINED = 5
};
enum MGPUCompletionFlags { MGPU_COMPLETION_RESOURCE_CONSUMED = 1 };
typedef struct MGPUCompletionInfo {
  uint32_t struct_size, version, kind, phase;
  int32_t result;
  uint32_t flags;
  uint64_t ticket_id;
} MGPUCompletionInfo; /*32 bytes. Metadata is not a completion receipt. */
typedef struct MGPUCompletionReceipt {
  uint32_t struct_size, version, kind;
  int32_t result;
  uint32_t flags, reserved;
  uint64_t ticket_id;
} MGPUCompletionReceipt; /*32 bytes. result is the underlying operation result. */
typedef struct MGPUCompletionStats {
  uint32_t struct_size, version, live_tickets, max_tickets;
  uint32_t quarantined_tickets, released_with_native_users;
  uint64_t next_ticket_id;
} MGPUCompletionStats; /*32 bytes. Released epilogues still consume quota. */

/* Fixed native operations only; no arbitrary operation dispatcher. These add
 * proof that the caller's SDK callback actually RETURNED. This is NOT proof that
 * backend jobs/last pins were destroyed, native pending-job counts reached zero,
 * memory was physically reclaimed, or source/engine authority may be released.
 * The independently retained backend states protect their own remaining tails.
 * In particular the packed buffer stays owned until writer retirement succeeds.
 *
 * Uses MGPUContextLeaseResult. All storage/holder/callback custody is rooted
 * before invoking an operation. Max128 live tickets; monotonic IDs never wrap
 * and fit uintptr_t. Callback cookies carry only IDs, never recycled addresses.
 * Keep this module, userdata and all required original owners alive. Callback
 * may run inline/before this entrypoint returns; it must not throw/block. The
 * ticket is published at return. The caller retains its synchronous call guard
 * until this entrypoint itself returns; do not race final release with creation.
 * No pointer/ticket/callback or native resource enters guest or room data.
 * Emscripten refuses UNSUPPORTED before operation/callback/owner acquisition.
 *
 * Write/Retire OK means the completion ticket is accepted, NOT operation success.
 * Checked Destroy preserves its original synchronous semantics: non-OK consumes
 * nothing, schedules no callback and returns no ticket; OK consumes the resource
 * pointer immediately, including on a later failure callback. Never retry it.
 * A callback admitted before checked refusal, an exception or unknown acceptance
 * returns a nonnull quarantined ticket: retain its source, resource accounting
 * and userdata; no timeout is proof. Callbacks after terminal refusal/release
 * cannot acquire their former ID or access caller storage.
 */
EXPORT int mgpuPrivateOutputWriteTrackedAsync(MGPUPrivateOutputWriter *, MGPUPrivateOutput *,
    MGPUQueueCompletionCallback, void *, MGPUCompletionTicket **);
EXPORT int mgpuComputeShaderUseRetireTrackedAsync(MGPUComputeShaderUse *,
    MGPUQueueCompletionCallback, void *, MGPUCompletionTicket **);
EXPORT int mgpuImportedFrameRetireTrackedAsync(MGPUImportedFrame *,
    MGPUQueueCompletionCallback, void *, MGPUCompletionTicket **);
EXPORT int mgpuPrivateOutputRetireTrackedAsync(MGPUPrivateOutput *,
    MGPUQueueCompletionCallback, void *, MGPUCompletionTicket **);
EXPORT int mgpuPrivateOutputWriterRetireTrackedAsync(MGPUPrivateOutputWriter *,
    MGPUQueueCompletionCallback, void *, MGPUCompletionTicket **);
EXPORT int mgpuComputeShaderDestroyTrackedAsync(MGPUContextLease *, MGPUComputeShader *,
    MGPUResourceCleanupCallback, void *, MGPUCompletionTicket **);
EXPORT int mgpuBufferDestroyTrackedAsync(MGPUContextLease *, MGPUBuffer *,
    MGPUResourceCleanupCallback, void *, MGPUCompletionTicket **);
EXPORT int mgpuCompletionTicketGetInfo(MGPUCompletionTicket *, MGPUCompletionInfo *, uint32_t);
/* Exactly one Collect after the underlying call AND SDK callback return. OK
 * transfers this identity-bound receipt even when receipt.result is failure.
 * Failure still does not permit resource reuse/refund. Outputs clear on refusal.
 * The private proxy will never subsequently access caller callback/userdata.
 * Its remaining CPU bookkeeping retains native storage and quota separately. */
EXPORT int mgpuCompletionTicketCollect(MGPUCompletionTicket *, MGPUCompletionReceipt *, uint32_t);
/* Exclusive final public-handle consumption after Collect. Excludes other
 * calls/waits on that raw handle. Quota remains until all native users finish. */
EXPORT int mgpuCompletionTicketRelease(MGPUCompletionTicket *);
EXPORT int mgpuCompletionTicketGetStats(MGPUCompletionStats *, uint32_t);
#ifdef __cplusplus
}
#endif
#endif
