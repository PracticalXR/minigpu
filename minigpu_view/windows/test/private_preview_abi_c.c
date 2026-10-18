#include "../include/minigpu_view/minigpu_view_private_preview.h"
#include "../include/minigpu_view/minigpu_view_private_dispatch.h"
#include "../include/minigpu_view/minigpu_view_registrar_association.h"
#include "../include/minigpu_view/minigpu_view_private_owner_close.h"
#include <stddef.h>
#include <minigpu_context_lifetime.h>
typedef char binding_size[(sizeof(MGPUViewHostBinding) == 80) ? 1 : -1];
typedef char info_size[(sizeof(MGPUViewPrivateInfo) == 88) ? 1 : -1];
typedef char receipt_size[(sizeof(MGPUViewPrivateReceipt) == 48) ? 1 : -1];
typedef char stats_size[(sizeof(MGPUViewPrivateStats) == 40) ? 1 : -1];
typedef char event_offset[(offsetof(MGPUViewPrivateInfo, wake_event) == 72) ? 1 : -1];
typedef char source_binding_size[(sizeof(MGPUViewPrivateSourceBinding) == 40) ? 1 : -1];
typedef char dispatch_info_size[(sizeof(MGPUViewPrivateDispatchInfo) == 56) ? 1 : -1];
typedef char dispatch_event_offset[(offsetof(MGPUViewPrivateDispatchInfo, wake_event) == 48) ? 1 : -1];
typedef char association_info_size[(sizeof(MGPUViewRegistrarAssociationInfo) == 32) ? 1 : -1];
typedef char association_generation_offset[(offsetof(MGPUViewRegistrarAssociationInfo, association_generation) == 16) ? 1 : -1];
typedef char owner_close_info_size[(sizeof(MGPUViewPrivateOwnerCloseInfo) == 48) ? 1 : -1];
typedef char owner_close_receipt_size[(sizeof(MGPUViewPrivateOwnerCloseReceipt) == 40) ? 1 : -1];
typedef char owner_close_event_offset[(offsetof(MGPUViewPrivateOwnerCloseInfo, wake_event) == 32) ? 1 : -1];
int private_preview_c_abi(void) {
  MGPUViewTrustedHost *host = (MGPUViewTrustedHost *)(uintptr_t)1;
  MGPUViewPrivateDispatcher *dispatcher = (MGPUViewPrivateDispatcher *)(uintptr_t)1;
  MGPUViewPrivateSource *source = (MGPUViewPrivateSource *)(uintptr_t)1;
  MGPUViewPrivateDispatch *request = (MGPUViewPrivateDispatch *)(uintptr_t)1;
  MGPUViewPrivateDispatchInfo info = {0};
  MGPUViewRegistrarAssociation *association = (MGPUViewRegistrarAssociation *)(uintptr_t)1;
  MGPUViewRegistrarAssociationInfo association_info = {0};
  MGPUViewPrivateOwnerClose *close_owner = (MGPUViewPrivateOwnerClose *)(uintptr_t)1;
  MGPUViewPrivateOwnerCloseInfo close_info = {0};
  MGPUViewPrivateOwnerCloseReceipt close_receipt = {0};
  return MGPUViewBindTrustedHost(NULL, NULL, &host) == MGPU_VIEW_INVALID && !host &&
      MGPUViewCreatePrivateDispatcher(NULL, &dispatcher) == MGPU_VIEW_INVALID && !dispatcher &&
      MGPUViewOpenPrivateSource(NULL, NULL, &source) == MGPU_VIEW_INVALID && !source &&
      MGPUViewPostPrivateRegister(NULL, NULL, 0, NULL, NULL, &request) == MGPU_VIEW_INVALID && !request &&
      MGPUViewGetPrivateDispatchInfo(NULL, &info, sizeof(info)) == MGPU_VIEW_INVALID && !info.request_id &&
      MGPUViewAcquireRegistrarAssociation(NULL, &association) == MGPU_VIEW_INVALID && !association &&
      MGPUViewBindAssociatedTrustedHost(NULL, NULL, &host) == MGPU_VIEW_INVALID && !host &&
      MGPUViewGetRegistrarAssociationInfo(NULL, &association_info, sizeof(association_info)) ==
          MGPU_VIEW_INVALID && !association_info.association_generation &&
      MGPUViewCreatePrivateOwnerClose(NULL, &close_owner, &close_info, sizeof(close_info)) ==
          MGPU_VIEW_INVALID && !close_owner && !close_info.request_id &&
      MGPUViewBeginPrivateOwnerClose(NULL, NULL, NULL) == MGPU_VIEW_INVALID &&
      MGPUViewCollectPrivateOwnerClose(NULL, &close_receipt, sizeof(close_receipt)) ==
          MGPU_VIEW_INVALID && !close_receipt.request_id &&
      MGPUViewReleasePrivateOwnerClose(NULL) == MGPU_VIEW_INVALID;
}
int private_preview_native_busy(void) { return MGPU_CONTEXT_LEASE_WOULD_BLOCK; }
int private_preview_native_unsupported(void) { return MGPU_CONTEXT_LEASE_UNSUPPORTED; }
