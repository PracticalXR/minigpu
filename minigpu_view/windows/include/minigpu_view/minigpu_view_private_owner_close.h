#ifndef MINIGPU_VIEW_PRIVATE_OWNER_CLOSE_H_
#define MINIGPU_VIEW_PRIVATE_OWNER_CLOSE_H_
#include "minigpu_view_registrar_association.h"
#include "minigpu_view_private_dispatch.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct MGPUViewPrivateOwnerClose MGPUViewPrivateOwnerClose;
enum {
  MGPU_VIEW_OWNER_CLOSE_READY = 1, MGPU_VIEW_OWNER_CLOSE_QUEUED = 2,
  MGPU_VIEW_OWNER_CLOSE_EXECUTING = 3, MGPU_VIEW_OWNER_CLOSE_COMPLETE = 4,
  MGPU_VIEW_OWNER_CLOSE_FAILED = 5
};
typedef struct MGPUViewPrivateOwnerCloseInfo {
  uint32_t struct_size, version, phase;
  int32_t result;
  uint64_t request_id, association_generation, wake_event;
  uint32_t has_host, has_dispatcher;
} MGPUViewPrivateOwnerCloseInfo; /* Win64:48 bytes; event is borrowed. */
typedef struct MGPUViewPrivateOwnerCloseReceipt {
  uint32_t struct_size, version;
  int32_t result;
  uint32_t reserved;
  uint64_t request_id, association_generation;
  uint32_t had_host, had_dispatcher;
} MGPUViewPrivateOwnerCloseReceipt; /*40 bytes. */
/* Trusted native cleanup only. Caller already retains the actual engine root.
 * Create runs on association platform BEFORE host Bind, providing cleanup even
 * if dispatcher creation fails. 32 outstanding objects; failed state is charged.
 * No arbitrary callback/command, capture/input/publication or qualification grant.
 * Begin/Collect are worker-safe. Final Release excludes other calls/waits.
 * Keep raw host/dispatcher handles alive through Begin; strong native states
 * are retained thereafter. They require separate final release AFTER receipt.
 * This closes an engine-scoped parent only after ALL its source children drain;
 * an individual source must never close a parent still serving another source.
 * Raw pointers/events must never enter Dart, guest or room messages. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewCreatePrivateOwnerClose(
    MGPUViewRegistrarAssociation *, MGPUViewPrivateOwnerClose **,
    MGPUViewPrivateOwnerCloseInfo *, uint32_t info_size);
/* One-shot. A dispatcher requires its exact associated host. No sources,
 * requests or registrations may remain. A refusal accepts no owners and can
 * be retried. An accepted post failure stays charged without a close receipt.
 * Both-null closes an unused cleanup owner. No timeout implies completion. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewBeginPrivateOwnerClose(
    MGPUViewPrivateOwnerClose *, MGPUViewTrustedHost *, MGPUViewPrivateDispatcher *);
/* Proof of returned platform close calls and actual window removal ONLY.
 * Missing pumping, failure or unknown completion never supplies this receipt.
 * Actual GPU/registration/context/engine release have separate protocols. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewCollectPrivateOwnerClose(
    MGPUViewPrivateOwnerClose *, MGPUViewPrivateOwnerCloseReceipt *, uint32_t receipt_size);
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewReleasePrivateOwnerClose(MGPUViewPrivateOwnerClose *);
#ifdef __cplusplus
}
#endif
#endif
