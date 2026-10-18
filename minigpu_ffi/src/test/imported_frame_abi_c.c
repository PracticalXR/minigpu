/* Compile and call the actual new C ABI, without C++/Dawn headers. */
#include "../include/minigpu_imported_frame.h"
#include <stddef.h>
typedef char frame_info_size[(sizeof(MGPUImportedFrameInfo) == 48) ? 1 : -1];
typedef char frame_info_generations[(offsetof(MGPUImportedFrameInfo,
    context_generation) == 16) ? 1 : -1];
typedef char frame_stats_size[(sizeof(MGPUImportedFrameLifetimeStats) == 40) ? 1 : -1];
#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
#define FRAME_SIZE(T, N) typedef char external_size_##T[(sizeof(T) == (N)) ? 1 : -1]
#define FRAME_OFFSET(T, F, N) \
    typedef char external_offset_##T##_##F[(offsetof(T, F) == (N)) ? 1 : -1]
FRAME_SIZE(MGPUExternalPlane, 40);
FRAME_SIZE(MGPUExternalFence, 32);
FRAME_SIZE(MGPUExternalVideoBuffer, 224);
FRAME_OFFSET(MGPUExternalPlane, width, 8);
FRAME_OFFSET(MGPUExternalPlane, stride_bytes, 16);
FRAME_OFFSET(MGPUExternalPlane, dmabuf_fd, 28);
FRAME_OFFSET(MGPUExternalPlane, drm_format_modifier, 32);
FRAME_OFFSET(MGPUExternalFence, d3d11_fence, 8);
FRAME_OFFSET(MGPUExternalFence, metal_shared_event, 16);
FRAME_OFFSET(MGPUExternalFence, metal_fence_value, 24);
FRAME_OFFSET(MGPUExternalVideoBuffer, pixel_format, 4);
FRAME_OFFSET(MGPUExternalVideoBuffer, width, 8);
FRAME_OFFSET(MGPUExternalVideoBuffer, height, 12);
FRAME_OFFSET(MGPUExternalVideoBuffer, num_planes, 16);
FRAME_OFFSET(MGPUExternalVideoBuffer, planes, 24);
FRAME_OFFSET(MGPUExternalVideoBuffer, fence, 184);
FRAME_OFFSET(MGPUExternalVideoBuffer, timestamp_us, 216);
typedef char external_tags[(MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE == 1 &&
    MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32 == 2) ? 1 : -1];
#undef FRAME_SIZE
#undef FRAME_OFFSET
#endif
int imported_frame_c_abi(void) {
  MGPUImportedFrame *owner = (MGPUImportedFrame *)(uintptr_t)1;
  return mgpuContextLeaseImportImmutableD3D11BgraFrame(
      NULL, NULL, 0, &owner) == MGPU_CONTEXT_LEASE_INVALID && owner == NULL;
}
