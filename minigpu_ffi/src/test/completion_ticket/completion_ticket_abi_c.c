#include "../../include/minigpu_completion_ticket.h"
#include <stddef.h>
typedef char completion_info_abi[(sizeof(MGPUCompletionInfo) == 32) ? 1 : -1];
typedef char completion_receipt_abi[(sizeof(MGPUCompletionReceipt) == 32) ? 1 : -1];
typedef char completion_stats_abi[(sizeof(MGPUCompletionStats) == 32) ? 1 : -1];
typedef char completion_identity_abi[(offsetof(MGPUCompletionReceipt, ticket_id) == 24) ? 1 : -1];
int completion_ticket_c_abi(void) {
  return mgpuPrivateOutputWriteTrackedAsync && mgpuComputeShaderUseRetireTrackedAsync &&
    mgpuImportedFrameRetireTrackedAsync && mgpuPrivateOutputRetireTrackedAsync &&
    mgpuPrivateOutputWriterRetireTrackedAsync && mgpuComputeShaderDestroyTrackedAsync &&
    mgpuBufferDestroyTrackedAsync && mgpuCompletionTicketGetInfo &&
    mgpuCompletionTicketCollect && mgpuCompletionTicketRelease && mgpuCompletionTicketGetStats;
}
