#ifndef MINIGPU_VIEW_PRIVATE_DISPATCH_H_
#define MINIGPU_VIEW_PRIVATE_DISPATCH_H_
#include "minigpu_view_private_preview.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGPUViewPrivateDispatcher MGPUViewPrivateDispatcher;
typedef struct MGPUViewPrivateSource MGPUViewPrivateSource;
typedef struct MGPUViewPrivateDispatch MGPUViewPrivateDispatch;
typedef struct MGPUViewPrivateSourceBinding {
  uint32_t struct_size, version;
  uint64_t source_epoch, valid_until_tick_ms;
  void *original_minigpu_query;
  uint64_t reserved;
} MGPUViewPrivateSourceBinding; /* Win64 V1: 40 bytes. */

enum MGPUViewDispatchKind { MGPU_VIEW_DISPATCH_REGISTER = 1, MGPU_VIEW_DISPATCH_RETIRE = 2 };
enum MGPUViewDispatchPhase {
  MGPU_VIEW_DISPATCH_QUEUED = 1, MGPU_VIEW_DISPATCH_EXECUTING = 2,
  MGPU_VIEW_DISPATCH_CALLBACK = 3, MGPU_VIEW_DISPATCH_COMPLETE = 4,
  MGPU_VIEW_DISPATCH_COLLECTED = 5, MGPU_VIEW_DISPATCH_QUARANTINE = 6
};
typedef struct MGPUViewPrivateDispatchInfo {
  uint32_t struct_size, version, kind, phase;
  int32_t result;
  uint32_t reserved;
  uint64_t request_id, source_epoch, registration_id, wake_event;
} MGPUViewPrivateDispatchInfo; /* 56 bytes. Event borrowed until final release. */

/* Native callback records/wakes only. It must not throw, block, run foreign
 * cleanup or release this request's userdata/output/source. It may run on the
 * platform thread before Post returns, or while dispatcher shutdown drains the
 * queue. Collect remains WOULD_BLOCK until the callback has actually returned.
 * Callback code is pinned. This is NOT an ownership or display receipt. */
typedef void (*MGPUViewPrivateDispatchCallback)(uint64_t source_epoch,
                                               uint64_t request_id, void *userdata);

/* Trusted host-local API only. No raw pointer/handle/function enters Dart, guest
 * or room messages. Create/Close run on the qualified platform thread, whose
 * message loop must continue pumping. A message-only window dispatches bounded
 * owned commands; its messages contain no command or callback pointers.
 * One dispatcher per host; 32 live sources, 128 pending requests process-wide,
 * four requests and two registrations per source. Quarantine stays charged.
 * Final Release must not race calls/waits on that raw handle. Caller retains
 * the actual engine/registrar instance through all children and receipts. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewCreatePrivateDispatcher(
    MGPUViewTrustedHost *, MGPUViewPrivateDispatcher **);
/* Native source owner supplies its original once-consumed consent's fixed
 * GetTickCount64 deadline and epoch, never an extended/renewed duration. The
 * exact producer anchor must match the host BEFORE any output is dereferenced.
 * Epochs open in strictly increasing order on this dispatcher, never reopen.
 * These identity checks do not create permission or authenticate an asset. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewOpenPrivateSource(
    MGPUViewPrivateDispatcher *, const MGPUViewPrivateSourceBinding *, MGPUViewPrivateSource **);
/* Any thread: irreversible new-admission fence before return. Already-admitted
 * calls/borrows remain owned until their actual cleanup receipts. No UI post,
 * callback, elapsed time, source fence or Stop supplies those receipts. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewFencePrivateSource(MGPUViewPrivateSource *);
/* Output and callback userdata remain caller-owned and alive through successful
 * Collect (including callback return). Synchronous non-OK accepts nothing and
 * schedules no callback. OK returns a counted request even if completion raced.
 * On collected error a nonnull registration MUST still be adopted and drained. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewPostPrivateRegister(
    MGPUViewPrivateSource *, void *original_output, uint64_t expected_output_id,
    MGPUViewPrivateDispatchCallback, void *userdata, MGPUViewPrivateDispatch **);
/* Allowed after source fencing. Retains the caller's registration borrow until
 * Collect; source identity must match. This posts Begin only: worker Collect of
 * actual removal/native cleanup remains a separate operation. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewPostPrivateRetirement(
    MGPUViewPrivateSource *, MGPUViewPrivateRegistration *,
    MGPUViewPrivateDispatchCallback, void *userdata, MGPUViewPrivateDispatch **);
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewGetPrivateDispatchInfo(
    MGPUViewPrivateDispatch *, MGPUViewPrivateDispatchInfo *, uint32_t out_size);
/* Exactly one transfer after execution AND callback return. Return OK means
 * transfer completed; info.result is the operation result. It can be non-OK
 * with a nonnull owner. With valid output buffers, all outputs clear on
 * WOULD_BLOCK/invalid/repeated calls.
 * Receipt IDs bind request/source only; registration has its own removal receipt. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewCollectPrivateDispatch(
    MGPUViewPrivateDispatch *, MGPUViewPrivateDispatchInfo *, uint32_t info_size,
    MGPUViewPrivateRegistration **, MGPUViewPrivateInfo *, uint32_t preview_size);
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewReleasePrivateDispatch(MGPUViewPrivateDispatch *);
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewReleasePrivateSource(MGPUViewPrivateSource *);
/* Unexpected shutdown: fence all sources, cancel never-started commands with
 * callbacks, quarantine unresolved registrations, destroy only the message
 * window. No successful image/engine cleanup is implied. Orderly owners drain
 * and release all sources first. Registrar shutdown invokes this automatically. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewClosePrivateDispatcher(MGPUViewPrivateDispatcher *);
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewReleasePrivateDispatcher(MGPUViewPrivateDispatcher *);
#ifdef __cplusplus
}
#endif
#endif
