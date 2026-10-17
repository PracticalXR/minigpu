/* Plain C consumer of the installed declarations; no Dawn or C++ dependency. */
#include "../include/minigpu_context_lifetime.h"
#include <stddef.h>

int resource_cleanup_c_abi(void) {
  int (*shader)(MGPUContextLease *, MGPUComputeShader *,
                MGPUResourceCleanupCallback, void *) =
      mgpuComputeShaderDestroyCheckedAsync;
  int (*buffer)(MGPUContextLease *, MGPUBuffer *,
                MGPUResourceCleanupCallback, void *) =
      mgpuBufferDestroyCheckedAsync;
  return shader(NULL, NULL, NULL, NULL) == MGPU_CONTEXT_LEASE_INVALID &&
         buffer(NULL, NULL, NULL, NULL) == MGPU_CONTEXT_LEASE_INVALID;
}
