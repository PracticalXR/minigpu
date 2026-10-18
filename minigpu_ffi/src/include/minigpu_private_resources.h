#ifndef MINIGPU_PRIVATE_RESOURCES_H
#define MINIGPU_PRIVATE_RESOURCES_H
#include "minigpu_context_lifetime.h"
#ifdef __cplusplus
extern "C" {
#endif

enum MGPUPrivateBufferKind {
  MGPU_PRIVATE_STORAGE_BUFFER = 1,
  MGPU_PRIVATE_READ_ONLY_STORAGE_BUFFER = 2,
  MGPU_PRIVATE_UNIFORM_BUFFER = 3
};
/* Host-local C layouts, never wire/guest data. Fixed x64 layouts are 40/32/40.
 * Only aligned unpacked u32 storage/uniform buffers in this first profile.
 * initial_data is copied before enqueue, at most 64 KiB; NULL iff size is zero.
 * usage permits Storage or Uniform (exactly one), and optional CopySrc/CopyDst.
 */
typedef struct MGPUPrivateBufferDescV1 {
  uint32_t struct_size, version;
  uint64_t byte_size;
  uint32_t usage, initial_size;
  const void *initial_data;
  uint64_t reserved;
} MGPUPrivateBufferDescV1;
typedef struct MGPUPrivateBufferBindingV1 {
  uint32_t slot, kind;
  MGPUBuffer *buffer;
  uint64_t offset, size;
} MGPUPrivateBufferBindingV1;
typedef struct MGPUPrivateShaderSetupV1 {
  uint32_t struct_size, version;
  const char *wgsl;
  uint32_t wgsl_size, binding_count;
  const MGPUPrivateBufferBindingV1 *bindings;
  uint32_t frame_slot, reserved;
} MGPUPrivateShaderSetupV1;

/* Exact original lease only. No initialization, replacement, global device,
 * CPU image/readback fallback or legacy load/bind/write/destruction allowed.
 * First profile: Windows native. Other platforms refuse before ownership.
 * Copy the lease's exact pin from birth; resources remain owner/worker-bound.
 * Native caps: 32 shaders, 128 buffers, 32 pending jobs, 512 MiB buffer bytes;
 * 128 MiB/buffer, 64 KiB initial bytes/WGSL, 4 bindings in 8 slots. Quarantine is charged.
 * Calls synchronously await their creation worker, refusing on that worker.
 * Do not hold authority locks, race calls/destruction, move C++ wrappers or
 * unload code. Keep supplied resources alive through setup, and through every
 * later use/writer retirement. No capture/input/encode/publication grant follows.
 *
 * OK returns a completed nonnull resource. Ordinary non-OK returns NULL and
 * leaves no accepted work/partial resource. QUARANTINED returns a nonnull custody
 * marker: keep source, charges and original lease; no reuse/retry/destruction.
 * Setup is one-shot before exclusive acquisition. Precommit failure leaves the
 * shader unchanged; uncertain mutation quarantines shader and referenced buffers.
 * Success is configuration, NOT pipeline compilation or dispatch/GPU completion.
 * Use the checked destruction/returned-callback protocols for final cleanup.
 */
EXPORT int mgpuContextLeaseCreateComputeShaderChecked(MGPUContextLease *, MGPUComputeShader **);
EXPORT int mgpuContextLeaseCreateBufferChecked(MGPUContextLease *, const MGPUPrivateBufferDescV1 *,
    uint32_t desc_size, MGPUBuffer **);
EXPORT int mgpuComputeShaderSetupChecked(MGPUContextLease *, MGPUComputeShader *,
    const MGPUPrivateShaderSetupV1 *, uint32_t desc_size);
#ifdef __cplusplus
}
#endif
#endif
