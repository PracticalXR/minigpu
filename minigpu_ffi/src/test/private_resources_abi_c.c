#include "../include/minigpu_private_resources.h"
#include <stddef.h>
typedef char buffer_desc_size[(sizeof(MGPUPrivateBufferDescV1) == 40) ? 1 : -1];
typedef char buffer_binding_size[(sizeof(MGPUPrivateBufferBindingV1) == 32) ? 1 : -1];
typedef char shader_setup_size[(sizeof(MGPUPrivateShaderSetupV1) == 40) ? 1 : -1];
typedef char initial_offset[(offsetof(MGPUPrivateBufferDescV1, initial_data) == 24) ? 1 : -1];
typedef char reserved_offset[(offsetof(MGPUPrivateBufferDescV1, reserved) == 32) ? 1 : -1];
typedef char frame_slot_offset[(offsetof(MGPUPrivateShaderSetupV1, frame_slot) == 32) ? 1 : -1];
int private_resources_c_abi(void) {
  return mgpuContextLeaseCreateComputeShaderChecked != 0 &&
    mgpuContextLeaseCreateBufferChecked != 0 && mgpuComputeShaderSetupChecked != 0;
}
