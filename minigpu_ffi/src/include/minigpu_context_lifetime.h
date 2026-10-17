#ifndef MINIGPU_CONTEXT_LIFETIME_H
#define MINIGPU_CONTEXT_LIFETIME_H

#include "export.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGPUContextHandle MGPUContextHandle;
typedef struct MGPUContextLease MGPUContextLease;
typedef struct MGPUComputeShader MGPUComputeShader;
typedef struct MGPUBuffer MGPUBuffer;
typedef struct MGPUComputeShaderUse MGPUComputeShaderUse;

enum MGPUContextLeaseResult {
  MGPU_CONTEXT_LEASE_OK = 0,
  MGPU_CONTEXT_LEASE_INVALID = -1,
  MGPU_CONTEXT_LEASE_NOT_READY = -2,
  MGPU_CONTEXT_LEASE_RETIRED = -3,
  MGPU_CONTEXT_LEASE_WOULD_BLOCK = -4,
  MGPU_CONTEXT_LEASE_INTERNAL = -5,
  MGPU_CONTEXT_LEASE_UNSUPPORTED = -6,
  MGPU_CONTEXT_LEASE_QUARANTINED = -7
};

typedef struct MGPUContextLeaseInfo {
  uint32_t struct_size;
  uint32_t version;
  uint64_t context_generation;
  uint64_t device_generation;
  uint32_t current;
  uint32_t reserved;
} MGPUContextLeaseInfo;

typedef struct MGPUContextLifetimeStats {
  uint32_t struct_size;
  uint32_t version;
  uint32_t live_contexts;
  uint32_t live_leases;
  uint32_t quarantined_contexts;
  uint32_t max_contexts;
  uint32_t max_leases;
  uint32_t reserved;
} MGPUContextLifetimeStats;

/* Pins an already-ready handle's exact creation device and worker. Never
 * initializes, repairs or migrates a context. Outputs are cleared on refusal.
 * Retention requires a live handle and must not race its destruction. Once
 * retained, a pin safely observes later retirement. Lease release must not race
 * use of that same lease pointer. The process-global legacy context is not a
 * handle. Do not unload the library while owners/leases remain live. */
EXPORT int mgpuContextRetainLease(MGPUContextHandle *context,
                                  MGPUContextLease **out_lease);
/* Trusted host interop with the already-attached process-global display
 * context. Never initializes, repairs, replaces or selects a device. Bypass
 * mode and the final-detach window refuse. The lease retains an attachment,
 * exact device and worker. Concurrent consumers share the same generation.
 * out_context is a borrowed capability-query/pool-creation token, valid only
 * while this lease or a consumer's synchronously retained pin remains live;
 * it is NOT an owned handle and must not be initialized or destroyed. The
 * final pin releases only its own attachment on the retirement worker. Both
 * outputs are cleared on refusal. Never serialize either pointer. */
EXPORT int mgpuDefaultContextRetainLease(MGPUContextLease **out_lease,
                                         MGPUContextHandle **out_context);
EXPORT int mgpuContextLeaseGetInfo(MGPUContextLease *lease,
                                   MGPUContextLeaseInfo *out_info,
                                   uint32_t out_size);
/* Local hardware identity for same-adapter host recovery. Currently kind=1 is
 * a Windows DXGI LUID (low/high words), backend=1 D3D11 or 2 D3D12. LUIDs are
 * local to this boot, not persistent or room/public identifiers. Other backends
 * refuse; adapter names/vendor IDs never substitute for physical identity.
 * Reads the already-ready creation device; never initializes or migrates it. */
typedef struct MGPUDeviceAdapterIdentity {
  uint32_t struct_size, version;
  uint64_t device_generation;
  uint32_t backend, kind, low, high;
} MGPUDeviceAdapterIdentity;
EXPORT int mgpuContextLeaseGetAdapterIdentity(
    MGPUContextLease *lease, MGPUDeviceAdapterIdentity *out_identity,
    uint32_t out_size);
/* Logical context destruction makes current false immediately. The creation
 * device/worker remain charged and retained through the last lease. Releasing
 * a lease is not evidence that GPU work completed: asynchronous consumers must
 * retain their own lease through actual completion. A borrowed lease returned
 * by the private capture-source owner is owner-managed: this call does not
 * delete it; the source's joined-release operation owns that disposition. */
EXPORT void mgpuContextReleaseLease(MGPUContextLease *lease);
/* Fences this lease's exact creation queue, including after logical retirement.
 * The queued completion owns its own pin; the caller may then release the
 * lease. Dart native-port result uses the existing (token << 1) | success
 * format. */
EXPORT void
mgpuContextLeaseWaitForSubmittedWorkAsyncToPort(MGPUContextLease *lease,
                                                int64_t port, int64_t token);
/* Native embedder form of the same exact-creation-queue fence. Completion is
 * OK only after OnSubmittedWorkDone reports success, never at dispatch return,
 * submission, timeout or device loss. Refusal/failure is not release authority.
 * The queued job retains its own pin, so the caller may release its lease after
 * this call; this does not retire caller-owned shaders, bindings or resources.
 * Callback/userdata must remain live until the single callback. It may run
 * inline on refusal or on the owning GPU worker, outside GPU locks and outside
 * Dawn's event callback. It must not throw, block or synchronously wait for that
 * worker. A null callback refuses without touching the lease. Native code only:
 * Dart must use the port form; Emscripten reports UNSUPPORTED without GPU work. */
typedef void (*MGPUQueueCompletionCallback)(int result, void *userdata);
EXPORT void mgpuContextLeaseWaitForSubmittedWorkAsync(
    MGPUContextLease *lease, MGPUQueueCompletionCallback callback,
    void *userdata);
/* Exclusive native shader resource scope. Uses MGPUContextLeaseResult. Never
 * initializes/migrates a device; Acquire checks the actual shader owner and
 * exact creation lease and clears out_use on refusal. At most 256 live scopes
 * process-wide (resource quota, not a participant limit). Synchronous Acquire
 * refuses on that GPU worker. Existing raw-pointer calls must not race final
 * shader destruction; UseRelease must not race any call on that token.
 *
 * The caller must own the dedicated shader AND all buffers/inputs/outputs until
 * successful retirement. A shader hold/AddRef cannot prevent BufferDestroy.
 * minigpu_imported_frame.h extends this scope with one checked native frame
 * attachment/access-retirement contract. Neither API grants source authority,
 * capture Start, input, encode or publication.
 *
 * Raw shader mutations refuse during reservation/use; status APIs return busy,
 * legacy void calls do nothing. Such interference keeps the shader closed even
 * after Release: destroy it rather than reuse mismatched host shadow state.
 * ShaderDestroy seals immediately and defers deletion while a scope exists.
 * Dispatch OK means CPU admission only; it is NOT GPU completion. Asynchronous
 * execution failure quarantines the scope. Retirement seals submissions, drops
 * cached texture/sampler/bind-group references and awaits actual creation-queue
 * completion. Only OK retirement permits Release; every failure or missing
 * acknowledgement retains the charged shader/context/resources. Never time out
 * to success. Retired textures/samplers require rebinding before shader reuse.
 * Callback rules are identical to the native queue callback above. Reentrant
 * retirement reports WOULD_BLOCK; no callback is invoked under native locks.
 * Native only; Emscripten refuses without GPU work. */
EXPORT int mgpuComputeShaderAcquireExclusiveUse(
    MGPUContextLease *lease, MGPUComputeShader *shader,
    MGPUComputeShaderUse **out_use);
EXPORT int mgpuComputeShaderUseDispatch(MGPUComputeShaderUse *use,
                                       int groups_x, int groups_y, int groups_z);
EXPORT void mgpuComputeShaderUseRetireAsync(
    MGPUComputeShaderUse *use, MGPUQueueCompletionCallback callback,
    void *userdata);
EXPORT int mgpuComputeShaderUseRelease(MGPUComputeShaderUse *use);
/* Checked final cleanup of a dedicated native shader or buffer. This is a
 * native-reference-release receipt, NOT GPU completion, physical reclamation,
 * render success, input retirement or permission to release source consent.
 * First retire all uses through their own completion protocols. All shader-use
 * tokens, including RETIRED tokens, must have been released. Shader-owned
 * textures/samplers are outside this narrow profile and refuse.
 *
 * The caller must exclusively own the resource: no concurrent raw-pointer
 * calls/destruction and no pending buffer CPU/map/readback users. Buffer has no
 * comprehensive busy-user counter; a strong wrapper reference is insufficient.
 * The exact retained lease must match the actual resource owner/creation device.
 * Old-generation references may be cleaned after loss; loss does not prove
 * prior use retirement. Never initializes, repairs, migrates or falls back.
 *
 * Synchronous non-OK consumes nothing and schedules NO callback. Keep the
 * resource, charges and original lease. Synchronous OK consumes the pointer;
 * never use/retry/destroy it again, including after a failure callback. Native
 * cleanup owns its exact creation pin through one callback. Failed/unknown
 * cleanup retains the allocated job, remaining resource and pin in quarantine;
 * do not refund charges or consent on a failure/missing acknowledgement.
 *
 * Register userdata before the call: the callback can run on the creation
 * worker before this API returns, outside native locks and Dawn events. It must
 * not throw/block/use the consumed pointer/wait on that worker. The callback
 * may enqueue other cleanup or release the caller's lease. Native only; Dart
 * must not pass isolate callbacks. Emscripten refuses without touching owners.
 * Both results use MGPUContextLeaseResult. Legacy void destroys are unchanged. */
typedef void (*MGPUResourceCleanupCallback)(int result, void *userdata);
EXPORT int mgpuComputeShaderDestroyCheckedAsync(
    MGPUContextLease *lease, MGPUComputeShader *shader,
    MGPUResourceCleanupCallback callback, void *userdata);
EXPORT int mgpuBufferDestroyCheckedAsync(
    MGPUContextLease *lease, MGPUBuffer *buffer,
    MGPUResourceCleanupCallback callback, void *userdata);
/* Trusted host recovery primitive, native Dawn only. The host must fence all
 * producers first. Pins the lease's exact creation generation, immediately
 * closes its readiness, and destroys that device on its own GPU worker. An OK
 * result means native GPU operations ended (possibly aborted), NOT successful
 * rendering, completed external/CPU readers, or permission to discard owners.
 * The context, exported resources and other leases are not released here.
 * Unknown/failed reader retirement still prevents context replacement.
 *
 * No timeout-to-success. A hang retains the pin/job. Old leases never target
 * another generation. Repeated calls are safe. The callback variant belongs to
 * embedders with process-lifetime callbacks: it may run inline on refusal or on
 * the GPU worker after native destruction. Dart must use the port variant.
 * Emscripten refuses without changing device readiness: browser device.lost is
 * not this native destruction receipt. */
typedef void (*MGPUDeviceRetirementCallback)(int result, void *userdata);
EXPORT void mgpuContextLeaseRetireDeviceAsync(
    MGPUContextLease *lease, MGPUDeviceRetirementCallback callback,
    void *userdata);
EXPORT void mgpuContextLeaseRetireDeviceAsyncToPort(
    MGPUContextLease *lease, int64_t port, int64_t token);
/* Non-owning observation of FINAL native context teardown for this lease's
 * exact generation. Register before releasing the lease. Registration does not
 * fence producers, destroy a device, retain an attachment or wait on a worker.
 * Pending leases/consumers keep this pending. Success follows native context
 * and external-cache teardown; failure is terminal and cannot authorize reset.
 * At most 64 observers per generation. Rejected registration reports failure.
 * The callback may run inline on refusal or on the final-detach thread; it must
 * not throw/block. Dart uses the port form. Never wait for this inside an owner
 * Stop that is being joined before the root releases its own attachment.
 * This is native-only and does not attest browser device retirement. */
EXPORT void mgpuContextLeaseObserveRetirementAsync(
    MGPUContextLease *lease, MGPUDeviceRetirementCallback callback,
    void *userdata);
EXPORT void mgpuContextLeaseObserveRetirementAsyncToPort(
    MGPUContextLease *lease, int64_t port, int64_t token);
EXPORT int mgpuContextGetLifetimeStats(MGPUContextLifetimeStats *out_stats,
                                       uint32_t out_size);

#ifdef __cplusplus
}
#endif
#endif
