/* Actual exported declarations and layout, without C++/Dawn headers. */
#include "../include/minigpu_private_output.h"
#include <stddef.h>
typedef char output_info_size[(sizeof(MGPUPrivateOutputInfo) == 80) ? 1 : -1];
typedef char output_info_id[(offsetof(MGPUPrivateOutputInfo, resource_id) == 24) ? 1 : -1];
typedef char output_stats_size[(sizeof(MGPUPrivateOutputStats) == 72) ? 1 : -1];
typedef char output_stats_bytes[(offsetof(MGPUPrivateOutputStats, charged_bytes) == 48) ? 1 : -1];
typedef char preview_borrow_size[(sizeof(MGPUPrivatePreviewBorrowInfo) == 64) ? 1 : -1];
typedef char preview_borrow_handle[(offsetof(MGPUPrivatePreviewBorrowInfo,
    legacy_shared_handle) == 48) ? 1 : -1];
int private_output_c_abi(void) {
  MGPUPrivateOutputWriter *writer = (MGPUPrivateOutputWriter *)(uintptr_t)1;
  MGPUPrivateOutput *output = (MGPUPrivateOutput *)(uintptr_t)1;
  MGPUPrivatePreviewLoan *loan = (MGPUPrivatePreviewLoan *)(uintptr_t)1;
  return mgpuPrivateOutputWriterCreate(NULL, NULL, 0, 0, &writer) ==
             MGPU_CONTEXT_LEASE_INVALID && writer == NULL &&
         mgpuPrivateOutputCreate(NULL, &output) == MGPU_CONTEXT_LEASE_INVALID &&
             output == NULL &&
         mgpuPrivateOutputAcquirePreviewLoan(NULL, &loan) ==
             MGPU_CONTEXT_LEASE_INVALID && loan == NULL;
}
