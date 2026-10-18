#ifndef MINIGPU_IMPORTED_FRAME_H
#define MINIGPU_IMPORTED_FRAME_H
#include "minigpu_context_lifetime.h"
#include "minigpu_external.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGPUImportedFrame MGPUImportedFrame;
enum MGPUImportedFramePhase {
  MGPU_IMPORTED_FRAME_READY = 1,
  MGPU_IMPORTED_FRAME_BINDING = 2,
  MGPU_IMPORTED_FRAME_ATTACHED = 3,
  MGPU_IMPORTED_FRAME_RETIRING = 4,
  MGPU_IMPORTED_FRAME_RETIRED = 5,
  MGPU_IMPORTED_FRAME_QUARANTINED = 6
};
typedef struct MGPUImportedFrameInfo {
  uint32_t struct_size, version, phase;
  int32_t last_result;
  uint64_t context_generation, device_generation, charged_bytes;
  uint32_t width, height;
} MGPUImportedFrameInfo; /* V1: 48 bytes */
typedef struct MGPUImportedFrameLifetimeStats {
  uint32_t struct_size, version, live_frames, max_frames;
  uint64_t charged_bytes, max_bytes, max_frame_bytes;
} MGPUImportedFrameLifetimeStats; /* V1: 40 bytes */

/* Native trusted-host resource primitive, not capture/source authority. Exact
 * sizeof descriptor required. Pins an already-ready original creation context;
 * no initialization, migration, adapter search, copying or CPU fallback.
 * First profile: unique completed never-recycled single-plane BGRA32 D3D11
 * snapshot, immutable-ready tag, zero stride/offset/subresource, no fences.
 * Producer retains its handle through this synchronous call. Successful import
 * then owns independent native references and permits producer-buffer release.
 * Execute from a host consumer worker, never UI or this creation GPU worker.
 *
 * Output initially clears. OK returns READY. Ordinary refusal leaves null.
 * QUARANTINED returns a NON-NULL charged owner: retain it even if subsequent
 * source/authority checks fail. It cannot be released or silently retried.
 * Native limits: 256 owners, 128 MiB/frame, 512 MiB logical retained frame bytes;
 * these are additional resource bounds, not physical GPU accounting/admission.
 * No raw native handle/view getter. Emscripten/non-Windows refuse GPU work.
 * Valid raw owners must not race their final Release. Do not unload code while
 * a frame, scope, queued job or callback remains live. */
EXPORT int mgpuContextLeaseImportImmutableD3D11BgraFrame(
    MGPUContextLease *lease, const MGPUExternalVideoBuffer *descriptor,
    uint32_t descriptor_size, MGPUImportedFrame **out_frame);
/* One frame once per ACTIVE/unsealed exclusive shader scope, before its first
 * Dispatch. Slot must be unoccupied with a compatible texture layout. Dispatch
 * refuses while binding is pending. Identical actual creation identity is
 * required. No replacement/sharing/rebinding. Supporting resources stay owned
 * by the dedicated caller. Synchronous; same GPU-worker restriction as import. */
EXPORT int mgpuComputeShaderUseBindImportedFrame(
    MGPUComputeShaderUse *use, uint32_t binding_slot, MGPUImportedFrame *frame);
EXPORT int mgpuImportedFrameGetInfo(MGPUImportedFrame *frame,
    MGPUImportedFrameInfo *out_info, uint32_t out_size);
EXPORT int mgpuImportedFrameGetLifetimeStats(
    MGPUImportedFrameLifetimeStats *out_stats, uint32_t out_size);
/* Only unattached READY frames. BINDING/ATTACHED must retire through their
 * shader scope. Actual queue success, checked exactly-once EndAccess, EndAccess
 * member cleanup and native reference release precede OK. Failed/lost/ambiguous
 * completion quarantines; timeout/device loss/void destruction is no receipt.
 * Callback rules match the native queue callback. Retire before logical
 * renderer-context retirement; a non-current pin conservatively quarantines. */
EXPORT void mgpuImportedFrameRetireAsync(MGPUImportedFrame *frame,
    MGPUQueueCompletionCallback callback, void *userdata);
/* Exclusive handle consumption only after successful RETIRED. */
EXPORT int mgpuImportedFrameRelease(MGPUImportedFrame *frame);

#ifdef __cplusplus
}
#endif
#endif
