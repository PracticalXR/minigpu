#ifndef MINIGPU_H
#define MINIGPU_H

#include "export.h"
#include "stdint.h"
#include "minigpu_external.h"
#ifdef __cplusplus
#include "../include/buffer.h"
#include "../include/compute_shader.h"

#define LOG(level, msg, ...)                                                   \
  do {                                                                         \
    /* Simple logging or remove entirely for performance */                    \
  } while (0)

extern "C" {
#endif

typedef struct MGPUComputeShader MGPUComputeShader;
typedef struct MGPUBuffer MGPUBuffer;
typedef void (*MGPUCallback)(void);
typedef void (*MGPULogCallback)(int level, const char* message);

EXPORT void mgpuInitializeContext();
EXPORT void mgpuInitializeContextAsync(MGPUCallback callback);
EXPORT void mgpuDestroyContext();
/// Install a log callback. [callback] is invoked on the thread that produces
/// the log line. Pass NULL to revert to the default stderr output.
/// [level]: 0=DEBUG 1=INFO 2=WARN 3=ERROR  (-1 = silence all)
///
/// NOT FOR DART. This registry is PROCESS-GLOBAL, so a Dart NativeCallable
/// installed here outlives the isolate that created it: once that isolate
/// exits the VM has deleted the trampoline while this library still holds
/// the pointer, and the next line a Dawn worker thread logs aborts the whole
/// process ("Callback invoked after it has been deleted"). Kept for non-Dart
/// embedders, which own their own function's lifetime. Dart must use
/// mgpuInitDartApi + mgpuSetLogPort below.
EXPORT void mgpuSetLogCallback(MGPULogCallback callback);
EXPORT void mgpuSetLogLevel(int level);

/// --- Dart native-port log delivery ---------------------------------------
/// A Dart port id is inert once its isolate is gone, so this is the only
/// delivery path a whole-suite `dart test` run (one process, one isolate per
/// test FILE) survives.
///
/// Call mgpuInitDartApi(NativeApi.initializeApiDLData) once, then
/// mgpuSetLogPort(receivePort.sendPort.nativePort). Messages arrive as
/// [int32 level, Uint8List utf8Message] — the bytes are COPIED into the
/// message, so there is nothing to free and mgpuFreeLogMessage does not
/// apply. Pass 0 to stop delivery.
///
/// LAST WRITER WINS across isolates, exactly like the function pointer.
/// A registered port takes precedence over any MGPULogCallback.
/// mgpuInitDartApi returns 0 on success, non-zero when this build has no
/// Dart API (the Emscripten/web build) or the SDK version does not match.
EXPORT int mgpuInitDartApi(void* initialize_api_dl_data);
EXPORT void mgpuSetLogPort(int64_t port);
/// Free a log message string returned via the log callback.
/// The log callback passes a heap-allocated copy of each message so the
/// pointer stays valid until the asynchronous Dart listener reads it.
/// Dart MUST call this after consuming the string.
EXPORT void mgpuFreeLogMessage(const char* msg);

/// --- Dart native-port COMPLETIONS ----------------------------------------
/// The `MGPUCallback` variants below are for embedders that own their
/// function's lifetime. DART MUST USE THESE INSTEAD.
///
/// A Dart `NativeCallable` can only be released by `close()`, which deletes
/// the trampoline, and this library cannot be told to forget a pointer it
/// already holds — so a completion arriving after the close aborts the whole
/// process with "Callback invoked after it has been deleted". Isolate teardown
/// (hot restart, a worker isolate exiting) deletes them too, which no
/// Dart-side discipline can cover. Posting to a port that is closed or whose
/// isolate is gone is a defined, silent no-op.
///
/// Call `mgpuInitDartApi(NativeApi.initializeApiDLData)` once per isolate
/// first (same call the log port uses; 0 = success). Then pass
/// `receivePort.sendPort.nativePort` and a token you allocate.
///
/// WIRE FORMAT: one int64 per completion, `(token << 1) | ok`. Tokens are
/// yours; never reuse one, so a late completion is a lookup miss rather than
/// somebody else's resolved future. Every entry point posts EXACTLY ONCE,
/// including on argument-validation failures — never leave a caller awaiting.
///
/// Element type codes for mgpuReadAsyncToPort. Deliberately NOT either
/// BufferDataType enum: the Dart and C++ enums are ordered DIFFERENTLY, so an
/// `.index` crossing this boundary is ambiguous.
typedef enum {
  MGPU_ELEM_I8 = 0,
  MGPU_ELEM_U8 = 1,
  MGPU_ELEM_I16 = 2,
  MGPU_ELEM_U16 = 3,
  MGPU_ELEM_I32 = 4,
  MGPU_ELEM_U32 = 5,
  MGPU_ELEM_I64 = 6,
  MGPU_ELEM_U64 = 7,
  MGPU_ELEM_F32 = 8,
  MGPU_ELEM_F64 = 9
} MGPUElementType;

EXPORT void mgpuInitializeContextAsyncToPort(int64_t port, int64_t token);
EXPORT void mgpuDispatchAsyncToPort(MGPUComputeShader *shader, int groupsX,
                                    int groupsY, int groupsZ, int64_t port,
                                    int64_t token);
EXPORT void mgpuReadAsyncToPort(MGPUBuffer *buffer, void *outputData,
                                size_t elementCount, size_t elementOffset,
                                int elementType, int64_t port, int64_t token);

/// Blocks until every task already queued on the WebGPU worker thread has
/// run. Ordering aid for teardown: destroying a resource that an enqueued
/// task still references frees it under that task. No-op when called from the
/// worker thread itself (it cannot drain a queue it is the head of).
EXPORT void mgpuDrainWorkQueue(void);

/// Pre-init hint (Windows): make mgpuInitializeContext() bind Dawn to the
/// adapter driving the PRIMARY display, so screen capture (Desktop
/// Duplication / WGC), GPU processing and any D3D11 encoder created on
/// Dawn's adapter share one GPU — same-adapter zero-copy import — even on
/// multi-output hybrid systems where the discrete GPU also drives a monitor
/// (there the automatic topology detection cannot see the capture/compute
/// split). The MGPU_ADAPTER_NAME env var still overrides. No-op on
/// non-Windows platforms.
///
/// Call BEFORE mgpuInitializeContext(). Returns 0 when stored before init;
/// 1 when the context was already initialized (the hint is kept for a future
/// re-init but the live context is unchanged).
EXPORT int mgpuPreferDisplayAdapter(int enable);

/// Copies the name of the adapter Dawn selected into [out] as a
/// NUL-terminated UTF-8 string (truncated to [cap] bytes including the NUL).
/// Returns the full (untruncated) name length in bytes, or 0 when the
/// context is not initialized.
EXPORT int mgpuGetSelectedAdapterName(char* out, int cap);

/// Attach to the PROCESS-GLOBAL context, initializing it on the first attach.
/// Safe to call concurrently from any number of threads/isolates: callers that
/// arrive while an initialization is in flight wait for it and then share the
/// result instead of starting a second one.
EXPORT void mgpuInitializeContext();
EXPORT void mgpuInitializeContextAsync(MGPUCallback callback);
/// Detach. The context is torn down only when the LAST attached consumer
/// detaches — one isolate finishing must not destroy a device another isolate
/// is still using. Single-consumer behaviour is unchanged (attach -> 1 -> init,
/// detach -> 0 -> teardown).
EXPORT void mgpuDestroyContext();
/// Number of explicit attaches currently outstanding on the process-global
/// context (0 when it is torn down). Diagnostics and tests only; lazy internal
/// re-initialization does not count.
EXPORT int mgpuContextRefCount();
EXPORT MGPUComputeShader *mgpuCreateComputeShader();
EXPORT void mgpuDestroyComputeShader(MGPUComputeShader *shader);
EXPORT void mgpuLoadKernel(MGPUComputeShader *shader, const char *kernelString);
EXPORT int mgpuHasKernel(MGPUComputeShader *shader);
EXPORT MGPUBuffer *mgpuCreateBuffer(int bufferSize, int dataType);
EXPORT void mgpuDestroyBuffer(MGPUBuffer *buffer);
EXPORT void mgpuSetBuffer(MGPUComputeShader *shader, int tag,
                          MGPUBuffer *buffer);
EXPORT void mgpuCreateKernel(MGPUComputeShader *shader, int groupsX,
                             int groupsY, int groupsZ);
/* ── Multi-GPU context handles ─────────────────────────────────────────────
 * A context handle is an independent MGPU instance bound to its own adapter
 * (own device, queue, WebGPU thread).  Buffers and shaders created FROM a
 * handle stay bound to it; every existing per-object call (setBuffer,
 * dispatch, read, write, destroy, ...) already routes through the object's
 * stored context, so only creation needs handle-aware entry points.  The
 * historical global-context API is untouched and remains the default
 * context. */
typedef struct MGPUContextHandle MGPUContextHandle;

/* Creates an UNINITIALIZED context bound to the adapter whose name contains
 * [adapterFilter] (case-insensitive substring; e.g. "3090").  Pass NULL or
 * "" for automatic selection (discrete > integrated > any). */
EXPORT MGPUContextHandle *mgpuCreateContextHandle(const char *adapterFilter);
/* Initializes the context; [callback] fires from the context's WebGPU
 * thread when done. */
EXPORT void mgpuContextInitializeAsync(MGPUContextHandle *handle,
                                       MGPUCallback callback);
/* Port-delivered completion — the form Dart must use. See the wire format
 * note above mgpuInitializeContextAsyncToPort. */
EXPORT void mgpuContextInitializeAsyncToPort(MGPUContextHandle *handle,
                                             int64_t port, int64_t token);
/* Destroys the context and frees the handle.  All buffers/shaders created
 * from it must already be destroyed. */
EXPORT void mgpuDestroyContextHandle(MGPUContextHandle *handle);
/* Name of the adapter this context selected (empty until initialized).
 * Returns the full name length; writes up to cap-1 chars + NUL. */
EXPORT int mgpuContextGetAdapterName(MGPUContextHandle *handle, char *out,
                                     int cap);
EXPORT MGPUBuffer *mgpuContextCreateBuffer(MGPUContextHandle *handle,
                                           int bufferSize, int dataType);
EXPORT MGPUComputeShader *
mgpuContextCreateComputeShader(MGPUContextHandle *handle);

/* Ordered bind: enqueues the binding update on the WebGPU thread so it
 * executes in FIFO order with dispatches/reads/writes.  Use when rebinding a
 * shader between fire-and-forget dispatches. */
EXPORT void mgpuSetBufferFire(MGPUComputeShader *shader, int tag,
                              MGPUBuffer *buffer);
EXPORT void mgpuDispatch(MGPUComputeShader *shader, int groupsX, int groupsY,
                         int groupsZ);
EXPORT void mgpuDispatchAsync(MGPUComputeShader *shader, int groupsX,
                              int groupsY, int groupsZ, MGPUCallback callback);

// Signed Integer Types
EXPORT void mgpuReadAsyncInt8(MGPUBuffer *buffer, int8_t *outputData,
                                    size_t size, size_t offset,
                                    MGPUCallback callback);
EXPORT void mgpuReadAsyncInt16(MGPUBuffer *buffer, int16_t *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
EXPORT void mgpuReadAsyncInt32(MGPUBuffer *buffer, int32_t *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
EXPORT void mgpuReadAsyncInt64(MGPUBuffer *buffer, int64_t *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
// Unsigned Integer Types
EXPORT void mgpuReadAsyncUint8(MGPUBuffer *buffer, uint8_t *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
EXPORT void mgpuReadAsyncUint16(MGPUBuffer *buffer, uint16_t *outputData,
                                      size_t size, size_t offset,
                                      MGPUCallback callback);
EXPORT void mgpuReadAsyncUint32(MGPUBuffer *buffer, uint32_t *outputData,
                                      size_t size, size_t offset,
                                      MGPUCallback callback);
EXPORT void mgpuReadAsyncUint64(MGPUBuffer *buffer, uint64_t *outputData,
                                      size_t size, size_t offset,
                                      MGPUCallback callback);
// Floating Point Number Types
EXPORT void mgpuReadAsyncFloat(MGPUBuffer *buffer, float *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
EXPORT void mgpuReadAsyncDouble(MGPUBuffer *buffer, double *outputData,
                                      size_t size, size_t offset,
                                      MGPUCallback callback);
// Sync Read Methods
EXPORT void mgpuReadSync(MGPUBuffer *buffer, void *outputData,
                               size_t size, size_t offset);
EXPORT void mgpuReadSyncInt8(MGPUBuffer *buffer, int8_t *outputData,
                                   size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncUint8(MGPUBuffer *buffer, uint8_t *outputData,
                                    size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncInt16(MGPUBuffer *buffer, int16_t *outputData,
                                    size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncUint16(MGPUBuffer *buffer, uint16_t *outputData,
                                     size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncInt32(MGPUBuffer *buffer, int32_t *outputData,
                                    size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncUint32(MGPUBuffer *buffer, uint32_t *outputData,
                                     size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncInt64(MGPUBuffer *buffer, int64_t *outputData,
                                    size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncUint64(MGPUBuffer *buffer, uint64_t *outputData,
                                     size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncFloat32(MGPUBuffer *buffer, float *outputData,
                                      size_t elementCount,
                                      size_t elementOffset);
EXPORT void mgpuReadSyncFloat64(MGPUBuffer *buffer, double *outputData,
                                      size_t elementCount,
                                      size_t elementOffset);
// Signed Integer Types
EXPORT void mgpuWriteInt8(MGPUBuffer *buffer, const int8_t *inputData,
                                  size_t byteSize);
EXPORT void mgpuWriteInt16(MGPUBuffer *buffer, const int16_t *inputData,
                                   size_t byteSize);
EXPORT void mgpuWriteInt32(MGPUBuffer *buffer, const int32_t *inputData,
                                   size_t byteSize);
EXPORT void mgpuWriteInt64(MGPUBuffer *buffer, const int64_t *inputData,
                                   size_t byteSize);
// Unsigned Integer Types
EXPORT void mgpuWriteUint8(MGPUBuffer *buffer, const uint8_t *inputData,
                                   size_t byteSize);
EXPORT void mgpuWriteUint16(MGPUBuffer *buffer,
                                    const uint16_t *inputData, size_t byteSize);
EXPORT void mgpuWriteUint32(MGPUBuffer *buffer,
                                    const uint32_t *inputData, size_t byteSize);
EXPORT void mgpuWriteUint64(MGPUBuffer *buffer,
                                    const uint64_t *inputData, size_t byteSize);
// Partial RAW write at a byte offset (dirty-region upload): copies byteSize
// bytes into the buffer at dstByteOffset. Both must be 4-byte aligned and stay
// within the buffer. Bypasses per-type packing (caller supplies final bytes).
EXPORT void mgpuWriteBufferAt(MGPUBuffer *buffer, const void *inputData,
                              size_t byteSize, size_t dstByteOffset);

/* ── BATCHED STAGING UPLOADS (additive; every other write path unchanged) ───
 *
 * WHY. `mgpuWriteBufferAt` / `mgpuWrite*` each take the device mutex AND
 * flush the pending compute batch, i.e. one `wgpuQueueSubmit` per range. A
 * producer that pushes many scattered dirty runs per frame therefore pays N
 * submits and N lock round-trips on its own thread, and the cost shows up as
 * a producer/consumer stall rather than as bandwidth.
 *
 * WHAT. Ranges staged between Begin and End are copied into a HOST arena
 * (no lock, no GPU call). `mgpuEndUploads` then, once and asynchronously on
 * minigpu's WebGPU thread, issues ONE `wgpuQueueWriteBuffer` into a
 * PERSISTENT staging buffer and records the ranges as `copyBufferToBuffer`
 * commands into the command encoder the batch is already building — no
 * submit, no batch flush, no per-range mutex.
 *
 * ORDERING. Identical to `mgpuWriteBufferAt`: `mgpuEndUploads` runs
 * synchronously on the calling thread under the same device mutex, so
 * commands already recorded into the batch execute before the copies and a
 * dispatch that has not been enqueued yet cannot overtake them. What the
 * batched path removes is the SUBMIT, not the ordering: a
 * `copyBufferToBuffer` recorded after the pending dispatches is ordered by
 * the encoder, whereas a queue write is ordered by submission and therefore
 * had to flush the batch first.
 *
 * LIFETIME. Staged bytes are copied out of [src] before the call returns; the
 * source may be reused or freed immediately. The staging allocation is
 * created on first use, grows on demand (never per call), is reused for the
 * life of the context and is released by `mgpuDestroyContext`.
 *
 * THREADING. One scope per context at a time, begun and ended on the SAME
 * thread (the frame producer). Nested Begin/End pairs are reference-counted
 * and collapse into one emission.
 *
 * FALLBACK. With no scope open — or for a range whose offset/size is not
 * 4-byte aligned — `mgpuStageWrite` performs the ordinary inline
 * `mgpuWriteBufferAt`, so a caller can route every push through it
 * unconditionally. */
/* Opens/re-enters an upload scope on the default context. Returns the nesting
 * depth (>=1), or 0 if the context is unavailable. */
EXPORT int mgpuBeginUploads(void);
/* Stages one range. Returns 1 = batched, 0 = performed inline (fallback),
 * -1 = bad arguments. */
EXPORT int mgpuStageWrite(MGPUBuffer *buffer, size_t dstByteOffset,
                          const void *inputData, size_t byteSize);
/* Hints the total byte size of the scope so the host arena does not grow
 * mid-frame. Optional. */
EXPORT void mgpuStageReserve(size_t byteSize);
/* Closes one nesting level; at depth 0 emits the batch. Returns the number of
 * copy commands emitted (adjacent ranges coalesce). */
EXPORT int mgpuEndUploads(void);
/* 1 when this build has the batched staging path. */
EXPORT int mgpuUploadsSupported(void);
/* Emission attribution for the default context, accumulated only while the
 * env var MGPU_UPLOAD_PROF=1 is set (zero cost otherwise). Needs n >= 10:
 *   [0] us in the host arena memcpy   [1] us waiting for the device mutex
 *   [2] us in wgpuQueueWriteBuffer    [3] us recording copyBufferToBuffer
 *   [4] us in forced submits          [5] ranges staged
 *   [6] copies emitted                [7] scopes emitted
 *   [8] forced submits                [9] bytes staged
 * Returns 1 on success, 0 if [out] is NULL or too small. */
EXPORT int mgpuUploadStats(long long *out, int n);
/* Context-handle variants (multi-GPU). `mgpuStageWrite` needs no variant: it
 * routes through the buffer's own context. */
EXPORT int mgpuContextBeginUploads(MGPUContextHandle *handle);
EXPORT int mgpuContextEndUploads(MGPUContextHandle *handle);

/* ── BATCHED READBACKS (additive; every other read path unchanged) ──────────
 *
 * WHY, AND WHY IT IS NOT THE UPLOAD PROBLEM. `mgpuReadSync*` exists only per
 * buffer, and each call takes the device mutex, flushes the pending compute
 * batch with its OWN `wgpuQueueSubmit`, creates an encoder, maps its own
 * staging buffer and blocks for GPU completion. A frame that reads N results
 * pays N of each. MEASURED (gsplats420 resident spike, 4K): eight reads
 * totalling 2.72 MB cost 5.85-6.10 ms; with EVERY DISPATCH SKIPPED the same
 * eight reads still cost 2.60 ms -- 1.05 GB/s on an RTX 4090, i.e. the
 * transfer is not the cost. Collapsing the eight calls to two saved 1.18 ms
 * with identical kernels: ~0.2 ms of FIXED COST PER CALL.
 * Note the asymmetry with the upload twin, which measured BYTE-bound
 * (11.9 GB/s at the margin) and call-count-free. Uploads needed the submit
 * storm removed; readbacks need the CALL removed.
 *
 * WHAT. Ranges staged between Begin and End are recorded as
 * `copyBufferToBuffer` into the command encoder the compute batch is ALREADY
 * building, targeting ONE persistent readback buffer. `mgpuEndReadbacks` then
 * submits ONCE (that submit also carries the pending dispatches), maps ONCE,
 * and memcpys each staged range to its destination. N reads = 1 submit +
 * 1 fence + 1 map.
 *
 * RAW BYTES. Offsets and sizes are BYTES and no per-type unpacking happens --
 * the mirror of `mgpuWriteBufferAt`, not of `mgpuReadSyncUint8` (which
 * un-packs 8/16-bit buffers). For 32-bit buffers the two are identical.
 *
 * LIFETIME / VISIBILITY -- the difference from the upload twin that bites.
 * A staged WRITE is copied out of the caller's pointer before the call
 * returns; a staged READ is not filled until End. Destinations must stay
 * alive for the whole scope and MUST NOT be inspected before
 * `mgpuEndReadbacks` returns. And since the copies are recorded at End, every
 * staged read observes its source as of END: do not open a scope across a
 * dispatch that overwrites something already staged.
 *
 * ORDERING / THREADING. `mgpuEndReadbacks` runs INLINE on the calling thread
 * under the same device mutex `mgpuReadSync*` takes, so ordering versus
 * dispatches, inline reads and inline writes is exactly what the per-buffer
 * path already had. (Emitting on minigpu's WebGPU thread would be a race:
 * the inline read/write entry points do not join that FIFO.)
 *
 * FALLBACK. With no scope open -- or for a range whose offset/size is not
 * 4-byte aligned -- `mgpuStageRead` performs the ordinary inline read before
 * returning, so a caller can route every read through it unconditionally. */
/* Opens/re-enters a readback scope on the default context. Returns depth. */
EXPORT int mgpuBeginReadbacks(void);
/* Stages one raw byte range. Returns 1 = batched (destination filled at End),
 * 0 = performed inline (destination already filled), -1 = bad arguments. */
EXPORT int mgpuStageRead(MGPUBuffer *buffer, size_t srcByteOffset, void *dst,
                         size_t byteSize);
/* Optional: pre-grows the persistent readback allocation. */
EXPORT void mgpuReadbackReserve(size_t byteSize);
/* Closes one nesting level; at depth 0 resolves every staged read. Returns the
 * number of reads filled. */
EXPORT int mgpuEndReadbacks(void);
/* 1 when this build has the batched readback path. */
EXPORT int mgpuReadbacksSupported(void);
/* Resolve attribution for the default context, accumulated only while the env
 * var MGPU_READBACK_PROF=1 is set (zero cost otherwise). Needs n >= 12:
 *   [0] us staging bookkeeping     [1] us waiting for the device mutex
 *   [2] us recording the copies    [3] us in the ONE submit
 *   [4] us in map+wait (the GPU completion this readback forces)
 *   [5] us memcpy out of the mapped range
 *   [6] reads staged               [7] copy commands emitted
 *   [8] scopes resolved            [9] bytes staged
 *  [10] readback-buffer grows     [11] scopes that fell back to per-read
 * Returns 1 on success, 0 if [out] is NULL or too small. */
EXPORT int mgpuReadbackStats(long long *out, int n);
/* Context-handle variants (multi-GPU). `mgpuStageRead` needs no variant: it
 * routes through the buffer's own context. */
EXPORT int mgpuContextBeginReadbacks(MGPUContextHandle *handle);
EXPORT int mgpuContextEndReadbacks(MGPUContextHandle *handle);
// Floating Point Number Types
EXPORT void mgpuWriteFloat(MGPUBuffer *buffer, const float *inputData,
                                   size_t byteSize);
EXPORT void mgpuWriteDouble(MGPUBuffer *buffer, const double *inputData,
                                    size_t byteSize);
// Returns dedicated VRAM usage in bytes for the primary GPU (DXGI on Windows).
// Returns -1 on unsupported platforms or if the query fails.
EXPORT int64_t mgpuQueryVramBytes();
/* Enumerates hardware adapters: up to [cap] entries of UTF-8 name
 * (namesOut, cap*128 bytes), total dedicated VRAM and current usage in
 * bytes.  Returns the number of hardware adapters. */
EXPORT int mgpuEnumAdapters(char *namesOut, int64_t *totalOut,
                            int64_t *usedOut, int cap);
#ifdef __cplusplus
}
#endif

#endif // MINIGPU_H
