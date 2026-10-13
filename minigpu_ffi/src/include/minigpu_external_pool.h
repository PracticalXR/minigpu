#ifndef MINIGPU_EXTERNAL_POOL_H
#define MINIGPU_EXTERNAL_POOL_H

#include "minigpu_context_lifetime.h"
#ifdef __cplusplus
extern "C" {
#endif

enum MGPUExternalPoolBackend {
  MGPU_EXTERNAL_POOL_BACKEND_OTHER = 0,
  MGPU_EXTERNAL_POOL_BACKEND_D3D11 = 1,
  MGPU_EXTERNAL_POOL_BACKEND_D3D12 = 2,
  MGPU_EXTERNAL_POOL_BACKEND_METAL = 3,
  MGPU_EXTERNAL_POOL_BACKEND_VULKAN = 4
};
enum MGPUExternalPoolFacility {
  MGPU_EXTERNAL_POOL_D3D11_DEVICE5_CONTEXT4 = 1,
  MGPU_EXTERNAL_POOL_DAWN_TEXTURE_IMPORT = 2,
  MGPU_EXTERNAL_POOL_DAWN_MULTITHREAD_PROTECTED = 4
};
enum MGPUExternalPoolRefusal {
  MGPU_EXTERNAL_POOL_FACILITIES_READY = 0,
  MGPU_EXTERNAL_POOL_BACKEND_UNQUALIFIED = 1,
  MGPU_EXTERNAL_POOL_NATIVE_FEATURE_MISSING = 2
};
typedef struct MGPUExternalPoolCapabilities {
  uint32_t struct_size;
  uint32_t version;
  uint64_t context_generation;
  uint64_t device_generation;
  uint32_t backend;
  uint32_t refusal;
  uint32_t facility_flags;
  uint32_t max_texture_dimension_2d;
  uint32_t adapter_luid_low;
  int32_t adapter_luid_high;
} MGPUExternalPoolCapabilities;

/* Queries an already-ready receiving context. Returned facilities describe
 * this exact device, not authorization, resource registration or publication.
 * Other backends report their real type with an explicit refusal. No implicit
 * context initialization, CPU fallback or backend-specific unchecked cast. */
EXPORT int mgpuExternalPoolQueryCapabilities(MGPUContextHandle *context,
                                             MGPUExternalPoolCapabilities *out,
                                             uint32_t out_size);

typedef struct MGPUExternalPool MGPUExternalPool;
typedef struct MGPUComputeShader MGPUComputeShader;
enum MGPUExternalPoolResult {
  MGPU_EXTERNAL_POOL_EMPTY = 1,
  MGPU_EXTERNAL_POOL_PENDING = 2,
  MGPU_EXTERNAL_POOL_UNSUPPORTED = -6,
  MGPU_EXTERNAL_POOL_TIMEOUT = -7,
  MGPU_EXTERNAL_POOL_CANCELED = -8,
  MGPU_EXTERNAL_POOL_DEVICE_LOST = -9,
  MGPU_EXTERNAL_POOL_CONFLICT = -10
};
enum MGPUExternalPoolPixelFormat {
  MGPU_EXTERNAL_POOL_RGBA8 = 1,
  MGPU_EXTERNAL_POOL_BGRA8 = 2
};
enum MGPUExternalPoolFencePurpose {
  MGPU_EXTERNAL_POOL_PRODUCER_FENCE = 1,
  MGPU_EXTERNAL_POOL_HOST_CONSUMER_FENCE = 2
};
enum MGPUExternalPoolConsumerKind {
  MGPU_EXTERNAL_POOL_DAWN_CONSUMER = 1,
  MGPU_EXTERNAL_POOL_NATIVE_FENCE_CONSUMER = 2
};
enum MGPUExternalPoolEventKind {
  MGPU_EXTERNAL_POOL_SOURCE_DONE = 1,
  MGPU_EXTERNAL_POOL_OUTPUT_READY = 2,
  MGPU_EXTERNAL_POOL_OUTPUT_DONE = 3,
  MGPU_EXTERNAL_POOL_FAULT = 4
};
typedef struct MGPUExternalPoolConfig {
  uint32_t struct_size, version;
  uint32_t width, height, pixel_format, reserved;
  /* Reserves aggregate host + registered source texture bytes, <=256 MiB.
   * Three host outputs are allocated once. Global pools reserve <=1 GiB.
   * Issued sources are initialized to transparent black before NT export.
   * Creation temporarily uses one zeroed CPU image, <= max_bytes / 6; that
   * image is released before the pool returns and is absent from frame delivery. */
  uint64_t max_bytes;
} MGPUExternalPoolConfig;
typedef struct MGPUExternalPoolSubmission {
  uint32_t struct_size, version;
  uint64_t texture_id, producer_fence_id, producer_fence_value;
  uint64_t media_sequence, lease_token, deadline_micros;
} MGPUExternalPoolSubmission;
typedef struct MGPUExternalPoolEvent {
  uint32_t struct_size, version, kind;
  int32_t result;
  uint64_t lease_token, media_sequence, producer_fence_value, copy_fence_value;
} MGPUExternalPoolEvent;
typedef struct MGPUExternalPoolOutput {
  uint32_t struct_size, version, width, height, pixel_format, reserved;
  uint64_t lease_token, consumer_id, context_generation, device_generation;
  /* Borrowed local objects, never resource tickets or wire values. Every user
   * retains a consumer before use. Initial consumer is Dawn; encoder consumers
   * release against a separately registered HOST_CONSUMER fence. */
  void *wgpu_texture;
  void *wgpu_texture_view;
  void *d3d11_texture;
} MGPUExternalPoolOutput;
typedef struct MGPUExternalPoolStats {
  uint32_t struct_size, version, retiring, faulted;
  uint32_t texture_count, fence_count, pending_sources, occupied_outputs;
  uint32_t quarantined_outputs, host_allocations, host_imports, queued_copies;
  uint64_t reserved_bytes, texture_bytes, gpu_copies;
} MGPUExternalPoolStats;

typedef struct MGPUExternalPoolLimits {
  uint32_t struct_size, version, max_pools, max_source_textures;
  uint32_t max_fences, output_slots, max_consumers, event_records;
  /* Format n uses bit (n-1); zero means this native pool profile is
   * unavailable. */
  uint32_t pixel_format_mask, live_pools;
  uint64_t max_pool_bytes, max_aggregate_bytes, reserved_bytes;
} MGPUExternalPoolLimits;
/* Native host-issued producer access. Local borrowed NT handles must only be
 * duplicated by trusted host code into the original approved peer namespace.
 * IDs/generations and the lease are pool-local ownership, never room grants. */
typedef struct MGPUExternalPoolProducer {
  uint32_t struct_size, version, width, height, pixel_format, slot_count;
  uint64_t context_generation, device_generation, export_lease;
  uint64_t texture_ids[3], producer_fence_id;
  void *nt_texture_handles[3];
  void *nt_producer_fence_handle;
} MGPUExternalPoolProducer;
/* Retain the ready context lease before transferring work to another thread.
 * These calls use that exact pin and safely refuse later logical retirement.
 * No raw context handle is borrowed by the background allocation. */
EXPORT int mgpuExternalPoolQueryLeaseCapabilities(
    MGPUContextLease *lease, MGPUExternalPoolCapabilities *out_caps, uint32_t size);
EXPORT int mgpuExternalPoolCreateIssuedFromLease(
    MGPUContextLease *lease, const MGPUExternalPoolConfig *config,
    MGPUExternalPool **out_pool);
EXPORT int mgpuExternalPoolGetLimits(MGPUExternalPoolLimits *out_limits,
                                     uint32_t out_size);

/* Native host integration API, not a guest authority. Creation/registration
 * run on a host worker and require a live receiving handle; do not race that
 * handle's destruction. First implementation is same-adapter Dawn D3D11 only.
 * NT handles are borrowed for the registration call. A non-inheritable duplicate
 * pins the exact kernel object used for import; retained COM objects and that
 * identity survive until safe unregister/cleanup. Aliases of registered kernel
 * objects across live/retiring pools refuse, including changed fence purposes. This
 * verifies local kernel object identity, not canonical GPU allocation identity:
 * fresh shared handles can represent a different kernel object for the same GPU
 * resource. Caller must verify canonical aliases and authenticated app/helper
 * allocation lineage before promoting an external resource or fence purpose.
 * No raw object/ID may cross a room wire.
 * Caller owns privacy, route authorization and exact registration identity.
 * At most 16 pools, three source textures/fences per pool, three output slots,
 * four consumers per output, and 16 completion records. */
EXPORT int mgpuExternalPoolCreate(MGPUContextHandle *context,
                                  const MGPUExternalPoolConfig *config,
                                  MGPUExternalPool **out_pool);
/* Additive issued profile: creates three shared render targets and a producer
 * fence inside the six-texture byte reservation. Existing imported-source pools
 * keep their behavior. Issued sources/fence cannot be replaced or unregistered.
 * Three mutex-free host outputs isolate source mutation with one GPU copy. */
EXPORT int mgpuExternalPoolCreateIssued(MGPUContextHandle *context,
                                        const MGPUExternalPoolConfig *config,
                                        MGPUExternalPool **out_pool);
/* One outstanding export lease, reserved before exposing borrowed handles.
 * Retire does not release it; final Release refuses until actual peer release.
 * A failed/partial delivery keeps the exact owner charged. */
EXPORT int mgpuExternalPoolAcquireProducer(MGPUExternalPool *pool,
                                           MGPUExternalPoolProducer *out,
                                           uint32_t out_size);
/* Trusted integration completion only: original peer kernel exit or verified
 * release of its actual borrowed/duplicated resources and queued GPU work.
 * A timeout, app notice or failed write is not proof. This consumes the exact
 * export lease and fences submissions, not source/output GPU completion. */
EXPORT int mgpuExternalPoolReleaseProducerAfterPeerRelease(
    MGPUExternalPool *pool, uint64_t export_lease);
EXPORT int mgpuExternalPoolRegisterTexture(MGPUExternalPool *pool,
                                           void *nt_handle,
                                           uint64_t *out_texture_id);
EXPORT int mgpuExternalPoolRegisterFence(MGPUExternalPool *pool,
                                         void *nt_handle, uint32_t purpose,
                                         uint64_t *out_fence_id);
EXPORT int mgpuExternalPoolUnregisterTexture(MGPUExternalPool *pool,
                                             uint64_t id);
EXPORT int mgpuExternalPoolUnregisterFence(MGPUExternalPool *pool, uint64_t id);
/* Trusted host encoder integration. The fence is created on the pool's exact
 * creation device and cannot alias a foreign producer fence. Device is borrowed:
 * retain the pool/allocation before making encoder/device cache references. */
typedef struct MGPUExternalPoolHostFence {
  uint32_t struct_size, version;
  uint64_t fence_id;
  void *d3d11_device;
} MGPUExternalPoolHostFence;
EXPORT int mgpuExternalPoolCreateHostConsumerFence(
    MGPUExternalPool *pool, MGPUExternalPoolHostFence *out_fence, uint32_t out_size);
/* Queues the actual creation-context signal after prior consumer commands,
 * flushes, and returns without waiting for GPU completion. Values increase.
 * Cleanup signaling remains available after retirement, for already queued
 * reads only. Only pool-created host fences qualify, never imported fences. */
EXPORT int mgpuExternalPoolSignalHostConsumerFence(
    MGPUExternalPool *pool, uint64_t fence_id, uint64_t value);
EXPORT uint64_t mgpuExternalPoolNowMicros(void);
/* Successful submission reserves ownership and returns PENDING. Per-pool media
 * sequence, lease tokens and per-producer-fence values strictly increase. Token
 * reuse is rejected without retaining unbounded history. Deadline is
 * host-chosen, >now and at most five seconds away. Full admission returns
 * WOULD_BLOCK without consuming any ownership. Fixed mutex handoff: producer
 * key 0 -> consumer key 1
 * -> key 0. Foreign readiness polls never enter Dawn's worker queue. */
EXPORT int mgpuExternalPoolSubmit(MGPUExternalPool *pool,
                                  const MGPUExternalPoolSubmission *submission);
EXPORT int mgpuExternalPoolCancel(MGPUExternalPool *pool, uint64_t lease_token);
EXPORT int mgpuExternalPoolPoll(MGPUExternalPool *pool,
                                MGPUExternalPoolEvent *out_event,
                                uint32_t out_size);
EXPORT int mgpuExternalPoolAcquireOutput(MGPUExternalPool *pool,
                                         uint64_t lease_token,
                                         MGPUExternalPoolOutput *out_output,
                                         uint32_t out_size);
EXPORT int mgpuExternalPoolRetainOutput(MGPUExternalPool *pool,
                                        uint64_t lease_token, uint32_t kind,
                                        uint64_t *out_consumer_id);
/* Dawn release queues an actual creation-queue completion and requires zero
 * fence arguments. Native release requires an exact registered host-consumer
 * fence/value; producer fences cannot complete host output leases. */
EXPORT int mgpuExternalPoolReleaseOutputConsumer(MGPUExternalPool *pool,
                                                 uint64_t lease_token,
                                                 uint64_t consumer_id,
                                                 uint64_t fence_id,
                                                 uint64_t fence_value);
EXPORT int mgpuExternalPoolBindOutput(MGPUExternalPool *pool,
                                      uint64_t lease_token,
                                      uint64_t consumer_id,
                                      MGPUComputeShader *shader, int binding);
/* Binding is queued in FIFO order. A shader may not dispatch a retained binding
 * after consumer release; replace that binding or destroy the shader before
 * final pool release. A retained view alone is not a retained consumer. */
EXPORT int mgpuExternalPoolGetStats(MGPUExternalPool *pool,
                                    MGPUExternalPoolStats *out_stats,
                                    uint32_t out_size);
/* Retire cancels unsubmitted work and stops admission; claimed consumers still
 * release normally. Post-submit timeout/device loss faults admission and keeps
 * uncertain resources charged. Poll events and finish consumers, then release.
 * Release is final/nonconcurrent and returns WOULD_BLOCK while ownership or
 * completion records remain. It never invents GPU completion. */
EXPORT int mgpuExternalPoolRetire(MGPUExternalPool *pool);
EXPORT int mgpuExternalPoolRelease(MGPUExternalPool *pool);
#ifdef __cplusplus
}
#endif
#endif
