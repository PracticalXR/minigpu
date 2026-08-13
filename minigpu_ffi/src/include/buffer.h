#ifndef BUFFER_H
#define BUFFER_H

#include "webgpu.h"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "./mutex.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

namespace mgpu {

// Forward declarations
class MGPU;

// ── Pre-init adapter preference ─────────────────────────────────────────────
// When enabled BEFORE MGPU::initializeContext(), Dawn binds to the adapter
// that drives the PRIMARY display (Windows only; ignored elsewhere). This
// aligns capture (Desktop Duplication / WGC), GPU processing and any D3D11
// encoder created on Dawn's adapter onto the display GPU → same-adapter
// zero-copy import even on multi-output hybrid systems where the discrete
// GPU also drives a monitor (there the "dGPU has no outputs" heuristic can't
// detect the capture/compute split). MGPU_ADAPTER_NAME still overrides.
void setPreferDisplayAdapter(bool enable);
bool preferDisplayAdapterEnabled();

// Name of the adapter Dawn actually selected (the Dawn adapter-info device
// string). Empty until initializeContext() completes; cleared on
// destroyContext().
std::string selectedAdapterName();

// Defined in minigpu_external.cpp. Drops the process-global D3D11 device and
// immediate-context caches that ALIAS the context being torn down (on the
// D3D11 backend the cached ID3D11Device IS Dawn's own device). Without this
// they survive a destroy/re-init cycle pointing at the dead device, and the
// next createSharedOutputTexture() silently takes the cross-device path and
// returns null. No-op unless [context] is the process-global one that built
// them, and on non-Windows.
void externalOnContextDestroyed(const void *context);

// no gpu library dependency
struct BufferData {
  WGPUBuffer buffer = nullptr;
  WGPUBufferUsage usage = WGPUBufferUsage_None;
  size_t size = 0;
};

// no gpu library dependency
enum BufferDataType {
  kFloat32,
  kInt32,
  kUInt32,
  kInt8,
  kUInt8,
  kInt16,
  kUInt16,
  kFloat64,
  kInt64,
  kUInt64,
  kUnknownType
};

// managed by MGPU
class WebGPUThread {
private:
  std::thread worker;
  std::queue<std::function<void()>> tasks;
  mgpu::mutex queueMutex;
  std::condition_variable condition;
  std::atomic<bool> stop{false};

public:
  WebGPUThread();
  ~WebGPUThread();

  // Identity of the worker thread. Used by the context state machine to
  // recognise "I am the thread the teardown is about to enqueueSync onto" and
  // refuse to block there (that wait would deadlock the teardown).
  std::thread::id threadId() const { return worker.get_id(); }

  // returns immediately
  void enqueueAsync(std::function<void()> task);

  // waits for completion
  template <typename T> T enqueueSync(std::function<T()> task) {
    auto promise = std::make_shared<std::promise<T>>();
    auto future = promise->get_future();

    enqueueAsync([task, promise]() {
      try {
        if constexpr (std::is_void_v<T>) {
          task();
          promise->set_value();
        } else {
          promise->set_value(task());
        }
      } catch (...) {
        promise->set_exception(std::current_exception());
      }
    });

    return future.get();
  }
};

class MGPU {
public:
  // Define the Context struct here
  struct Context {
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    bool initialized = false;
    // Owned dawn::native::Instance* (non-Emscripten only). Stored as void* so
    // buffer.h need not include dawn/native headers — the cast is done in
    // buffer.cpp where the include is available.
#ifndef __EMSCRIPTEN__
    void* dawnNativeInstance = nullptr;
#endif
  };

  // ── Context lifecycle ────────────────────────────────────────────────────
  // The native context is PROCESS-global; the Dart wrapper that drives it is
  // PER-ISOLATE.  `dart test` runs every suite as an isolate inside ONE
  // process, so N isolates concurrently believe they must create and later
  // destroy "their" context.  Two things follow, and both are handled here:
  //
  //   1. LIFECYCLE RACE.  initializeContext() used to bail out only on
  //      `ctx && ctx->initialized`.  Thread A allocated a fresh Context and
  //      then spent milliseconds in the adapter/device request with
  //      `initialized == false`; thread B's lazy getDevice() saw exactly that,
  //      called initializeContext(), fell through the guard and re-assigned
  //      `ctx` — FREEING A's live Context while A still held raw pointers into
  //      it.  Access violation.  The fix is a state machine (below) whose lock
  //      is held only across the STATE TRANSITIONS, never across the slow
  //      device request: holding a lock across the work deadlocks, because
  //      initializeContextAsync runs the init ON webgpuThread while
  //      destroyContext enqueueSync()s onto that same thread.
  //
  //   2. PREMATURE TEARDOWN.  One isolate finishing and destroying the context
  //      pulled the device out from under every other isolate still using it
  //      (observed as createSharedOutputTexture() returning null in a later
  //      suite of a serialized run).  attach/detach reference-count the
  //      EXPLICIT consumers so teardown happens only when the last one leaves.
  //
  // State transitions are serialized; the work is not.
  enum class CtxState { Uninitialized, Initializing, Ready, Destroying };

  // Idempotent, concurrency-safe.  Blocks while another thread is initializing
  // or destroying, then returns the resulting context (or does the work).
  // Does NOT take a reference — this is the LAZY path (getDevice, getQueue,
  // isDeviceValid, ensureDeviceValid).
  void initializeContext();
  void initializeContextAsync(std::function<void()> callback);
  // Unconditional teardown.  Ignores the reference count — internal use and
  // ensureDeviceValid()'s device-loss recovery.  Blocks while an init is in
  // flight so it can never free a Context another thread is still building.
  void destroyContext();

  // ── Explicit attach / detach (the exported C-API boundary) ───────────────
  // mgpuInitializeContext / mgpuInitializeContextAsync attach; mgpuDestroy-
  // Context detaches.  Teardown happens only on the transition to zero, so a
  // second isolate's destroy cannot kill a first isolate's live device.
  // SINGLE-CONSUMER BEHAVIOUR IS UNCHANGED: attach -> 1 -> init,
  // detach -> 0 -> real teardown.  The lazy re-init paths deliberately do NOT
  // attach — counting them would make the count depend on how many buffers
  // happened to be created before the device existed.
  void attachContext();
  void attachContextAsync(std::function<void()> callback);
  void detachContext();
  int attachCount() const;

  bool isDeviceValid();
  void ensureDeviceValid();

  // Per-instance adapter selection (multi-GPU): when non-empty, overrides the
  // MGPU_ADAPTER_NAME env var during initializeContext (substring,
  // case-insensitive). Set BEFORE initialization.
  void setAdapterFilter(const std::string &filter) { adapterFilter = filter; }
  // Name of the adapter THIS instance selected (empty until initialized).
  std::string adapterName() const { return selectedName; }

  WebGPUThread &getWebGPUThread() { return webgpuThread; }

  WGPUDevice getDevice() const;
  WGPUQueue getQueue() const;
  WGPUInstance getInstance() const;
  // Returns the instance without attempting to reinitialize the context.
  WGPUInstance tryGetInstance() const noexcept {
    return (ctx && ctx->initialized) ? ctx->instance : nullptr;
  }
  // Returns the queue without attempting to reinitialize the context.
  WGPUQueue tryGetQueue() const noexcept {
    return (ctx && ctx->initialized) ? ctx->queue : nullptr;
  }

  mgpu::mutex &getGpuMutex() { return gpuOperationMutex; }

  // ── Batched compute recording ────────────────────────────────────────────
  // Fire-and-forget dispatches record into ONE open compute pass instead of
  // creating an encoder + queue submit each (a 35B-model decode step fires
  // ~2000 dispatches; per-dispatch submits dominated the token time).
  // WebGPU guarantees sequential memory effects between dispatches within a
  // pass, so batching preserves semantics.  The batch MUST be flushed
  // (submitted) before anything else touches the queue out-of-band:
  // queue writes, readback copies, external-texture blits, buffer/context
  // teardown.  All fields are guarded by gpuOperationMutex.
  WGPUCommandEncoder batchEncoder = nullptr;
  WGPUComputePassEncoder batchPass = nullptr;
  int batchCount = 0;
  // Submits any pending batched work.  Caller must hold gpuOperationMutex.
  void flushBatchLocked();
  // Same, but acquires the mutex itself.  Use from paths that submit their
  // own command buffers to the queue without already holding the mutex
  // (external video/shared-texture blits).
  void flushBatch() {
    mgpu::lock_guard<mgpu::mutex> lock(gpuOperationMutex);
    flushBatchLocked();
  }

  // ── Batched staging uploads (ADDITIVE — see mgpuBeginUploads) ─────────────
  // Problem this solves: every `writeBytesAt` / `write` takes the device mutex
  // AND flushes the pending batch (one wgpuQueueSubmit per range).  A producer
  // pushing N scattered dirty runs per frame therefore pays N submits and N
  // lock round-trips on its own thread.
  //
  // The batched path instead accumulates the ranges in a HOST arena (no lock,
  // no GPU call), then — once, asynchronously, on the WebGPU thread — does ONE
  // wgpuQueueWriteBuffer into a persistent staging buffer and records N
  // copyBufferToBuffer commands into the command encoder the batch is ALREADY
  // building.  No submit, no per-range mutex.
  //
  // Ordering: IDENTICAL to the inline `writeBytesAt` path.  The emission runs
  // synchronously on the caller's thread under the same mutex; commands
  // already recorded into the batch execute before the copies, and a dispatch
  // that has not been enqueued yet cannot overtake them.  The difference is
  // only that the batch does not have to be SUBMITTED for the ordering to
  // hold — a `copyBufferToBuffer` recorded after the pending dispatches is
  // ordered by the encoder, whereas a queue write is ordered by submission and
  // therefore had to flush.  The staging write itself is safe against the
  // pending batch because the staging buffer is private to this path (nothing
  // else ever reads it) and its cursor only advances until a submit resets it.
  //
  // Threading: one scope per context at a time; begin/stage/end must run on
  // ONE thread (typically the frame producer).  Nested begin/end pairs are
  // reference-counted and collapse into a single emission.  Staged bytes are
  // copied out of the caller's pointer immediately, so the source may die as
  // soon as the stage call returns.
  struct StagedCopy {
    WGPUBuffer dst = nullptr;
    uint64_t dstOffset = 0;
    uint64_t srcOffset = 0; // into the scope's host arena
    uint64_t size = 0;
  };
  struct UploadScope {
    std::vector<uint8_t> host;
    std::vector<StagedCopy> copies;
  };

  // Opens (or re-enters) an upload scope.  Returns the new nesting depth.
  int beginUploads();
  // Stages one range.  Returns 1 when it was batched, 0 when it fell back to
  // an inline write (no scope open / cross-context buffer / unaligned range),
  // -1 on error.  [dst] must belong to THIS context.
  int stageWrite(WGPUBuffer dst, size_t dstByteOffset, size_t dstBufferSize,
                 const void *src, size_t byteSize);
  // Closes one nesting level; emits at depth 0.  Returns the copies emitted.
  int endUploads();
  // Pre-grows the host arena of the open scope (avoids mid-frame realloc).
  void reserveUploads(size_t byteSize);
  bool uploadsSupported() const { return true; }
  int uploadScopeDepth() const { return uploadDepth; }

private:
  // Runs on the WebGPU thread with gpuOperationMutex held.
  void applyUploadScopeLocked(UploadScope &scope);
  void uploadFallbackLocked(UploadScope &scope);
  void recycleScope(UploadScope *scope);
  UploadScope *acquireScope();

  // Caller-thread state (single-scope contract, see above).
  std::unique_ptr<UploadScope> uploadOpen;
  int uploadDepth = 0;
  // Scope free-list, shared with the WebGPU thread.
  std::vector<std::unique_ptr<UploadScope>> uploadPool;
  mgpu::mutex uploadPoolMutex;
  // WebGPU-thread state, guarded by gpuOperationMutex.
  WGPUBuffer stagingBuffer = nullptr;
  size_t stagingCapacity = 0;
  size_t stagingCursor = 0;

public:
  // Releases the persistent staging allocation (called by destroyContext).
  // Caller must hold gpuOperationMutex.
  void releaseStagingLocked();

  // ── Batched readbacks (ADDITIVE — see mgpuBeginReadbacks) ─────────────────
  // The DOWNLOAD twin of the upload scope above, and NOT the same problem.
  // Uploads measured BYTE-bound (11.9 GB/s at the margin, call-count-free);
  // downloads measure CALL-bound: eight `readSync` calls totalling 2.72 MB cost
  // 2.60 ms with every dispatch skipped (1.05 GB/s on an RTX 4090), and
  // collapsing them to two saved 1.18 ms with identical kernels — ~0.2 ms of
  // FIXED cost per call.  The bytes are not the bill; the per-call
  // submit-and-wait is: every `Buffer::read*` takes the device mutex, flushes
  // the pending batch with its own `wgpuQueueSubmit`, creates an encoder, maps
  // its own staging buffer and blocks for GPU completion — N times.
  //
  // The batched path stages N ranges into ONE persistent, mapped-on-demand
  // readback buffer: the copies are recorded into the command encoder the batch
  // is ALREADY building, submitted ONCE (that submit also carries the pending
  // dispatches), waited on ONCE, and mapped ONCE.  N reads therefore cost one
  // submit + one fence + one map instead of N of each.
  //
  // SEMANTIC DIFFERENCE FROM THE UPLOAD TWIN — the one that bites.  Staged
  // WRITES are copied out of the caller's pointer immediately; staged READS are
  // not filled until End.  The destinations must stay alive across the scope and
  // MUST NOT be inspected before `endReadbacks()` returns.  And because the
  // copies are recorded at End, every staged read observes the source buffer as
  // of END, not as of the stage call — do not open a scope across a dispatch
  // that overwrites something already staged.
  //
  // Threading: End runs INLINE on the caller's thread, deliberately — the same
  // posture `endUploads` was forced into.  `mgpuReadSync*` / `mgpuWrite*` run
  // inline and do NOT join the WebGPU thread's FIFO, so emitting there would
  // race every caller that mixes the two.
  struct StagedRead {
    void *dst = nullptr;      // caller destination, filled at End
    uint64_t slotOffset = 0;  // into the readback buffer
    uint64_t size = 0;
  };
  struct ReadCopy {
    WGPUBuffer src = nullptr;
    uint64_t srcOffset = 0;
    uint64_t slotOffset = 0;
    uint64_t size = 0;
  };
  struct ReadbackScope {
    std::vector<StagedRead> reads;
    std::vector<ReadCopy> copies;
    uint64_t total = 0;
  };

  // Opens (or re-enters) a readback scope.  Returns the new nesting depth.
  int beginReadbacks();
  // Stages one raw byte range.  Returns 1 when it was batched, 0 when the
  // caller must fall back to an inline read (no scope open / unaligned range /
  // no device), -1 on bad arguments.
  int stageRead(WGPUBuffer src, size_t srcByteOffset, size_t srcBufferSize,
                void *dst, size_t byteSize);
  // Closes one nesting level; resolves at depth 0.  Returns the reads filled.
  int endReadbacks();
  // Pre-grows the persistent readback allocation (avoids a mid-frame realloc).
  void reserveReadbacks(size_t byteSize);
  bool readbacksSupported() const { return true; }
  int readbackScopeDepth() const { return readbackDepth; }
  // Releases the persistent readback allocation (called by destroyContext).
  // Caller must hold gpuOperationMutex.
  void releaseReadbackLocked();

  // ── Readback profiler (MGPU_READBACK_PROF=1) ────────────────────────────
  // Splits the resolve so "the readback costs X ms" becomes "of X, Y is the
  // submit, Z is the GPU completion the map waits on and W is the memcpy out".
  struct ReadbackStats {
    long long usStage = 0;   // bookkeeping in stageRead
    long long usLock = 0;    // waiting for gpuOperationMutex in End
    long long usCopy = 0;    // recording copyBufferToBuffer into the batch
    long long usSubmit = 0;  // the ONE flush/submit
    long long usMap = 0;     // mapAsync + wait (= the GPU completion forced)
    long long usOut = 0;     // memcpy from the mapped range to the callers
    long long reads = 0, copies = 0, scopes = 0, bytes = 0, grows = 0,
              fallbacks = 0;
  };
  ReadbackStats readbackStats;
  static bool readbackProfEnabled();

private:
  // Maps [0,size) of [buf] for read and blocks until the map completes.
  bool waitMapReadLocked(WGPUBuffer buf, size_t size);
  // One inline read through a throwaway staging buffer (the End-time fallback
  // when the persistent allocation cannot be created).
  bool readOneLocked(WGPUBuffer src, uint64_t srcOffset, void *dst,
                     uint64_t size);
  void applyReadbackScopeLocked(ReadbackScope &scope);

  std::unique_ptr<ReadbackScope> readbackOpen;
  int readbackDepth = 0;
  // WebGPU-thread/locked state.
  WGPUBuffer readbackBuffer = nullptr;
  size_t readbackCapacity = 0;

public:

  // ── Upload profiler (MGPU_UPLOAD_PROF=1) ────────────────────────────────
  // Attributes the emission, because "the batched upload costs X" is not
  // actionable and "of X, Y is the arena memcpy, Z is waiting for the device
  // mutex and W is Dawn's queue write" is.  Zero cost when the env var is
  // unset (one branch on a cached int).  Counters are microseconds / bytes.
  struct UploadStats {
    long long usStage = 0;  // host memcpy into the arena (mgpuStageWrite)
    long long usLock = 0;   // waiting for gpuOperationMutex in End
    long long usWrite = 0;  // wgpuQueueWriteBuffer of the arena into staging
    long long usCopy = 0;   // recording copyBufferToBuffer into the batch
    long long usFlush = 0;  // forced submits (arena wrap / staging grow)
    long long ranges = 0, copies = 0, scopes = 0, flushes = 0, bytes = 0;
  };
  UploadStats uploadStats;
  static bool uploadProfEnabled();

private:
  // The unguarded body of initializeContext() / destroyContext(): runs with NO
  // lock held (the device request takes milliseconds and destroyContext joins
  // the WebGPU thread), serialized purely by ctxState.
  void initializeContextUnlocked();
  void destroyContextUnlocked();

  std::unique_ptr<Context> ctx;
  mgpu::mutex gpuOperationMutex;
  WebGPUThread webgpuThread;
  std::string adapterFilter;
  std::string selectedName;

  // ── Context lifecycle state (see the enum above) ─────────────────────────
  // Guards ONLY the state word, the owner id and the reference count — never
  // held across device creation or teardown.  Distinct from gpuOperationMutex
  // on purpose: teardown takes gpuOperationMutex on the WebGPU thread, so
  // sharing one mutex would reintroduce the deadlock this design avoids.
  mutable mgpu::mutex ctxStateMutex;
  std::condition_variable ctxStateCv;
  CtxState ctxState = CtxState::Uninitialized;
  // Thread currently performing the Initializing/Destroying work, so a
  // re-entrant call from that same thread returns instead of waiting on
  // itself.
  std::thread::id ctxOwner;
  int ctxAttachCount = 0;
};
class Buffer {
public:
  explicit Buffer(MGPU &mgpu_ref);
  ~Buffer();

  Buffer(Buffer &&other) noexcept;
  Buffer &operator=(Buffer &&other) noexcept;

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  void createBuffer(size_t byteSize, BufferDataType dataType);

  void write(const float *inputData, size_t elementCount);
  void write(const int32_t *inputData, size_t elementCount);
  void write(const uint32_t *inputData, size_t elementCount);
  void write(const int8_t *inputData, size_t elementCount);
  void write(const uint8_t *inputData, size_t elementCount);
  void write(const int16_t *inputData, size_t elementCount);
  void write(const uint16_t *inputData, size_t elementCount);
  void write(const double *inputData, size_t elementCount);
  void write(const int64_t *inputData, size_t elementCount);
  void write(const uint64_t *inputData, size_t elementCount);

  // Partial RAW write: copy [byteSize] bytes of [inputData] into this buffer
  // starting at [dstByteOffset] bytes. Bypasses the per-type packing path --
  // the caller supplies bytes already in final buffer layout. Used for
  // dirty-region uploads (write only the tiles that changed instead of the
  // whole plane). WebGPU requires dstByteOffset and byteSize to be multiples
  // of 4; dstByteOffset + byteSize must be <= getSize().
  void writeBytesAt(const void *inputData, size_t byteSize,
                    size_t dstByteOffset);

  // Partial RAW read at a byte offset — the read-side mirror of writeBytesAt,
  // and the inline fallback for `mgpuStageRead`.  Copies [byteSize] bytes
  // starting at [srcByteOffset] into [outputData] with NO per-type unpacking:
  // the caller gets the buffer's bytes as they are laid out on the device, so
  // the answer is identical whether the read was batched or not.
  void readBytesAt(void *outputData, size_t byteSize, size_t srcByteOffset);

  void read(float *outputData, size_t elementCount, size_t offset = 0);
  void read(int32_t *outputData, size_t elementCount, size_t offset = 0);
  void read(uint32_t *outputData, size_t elementCount, size_t offset = 0);
  void read(int8_t *outputData, size_t elementCount, size_t offset = 0);
  void read(uint8_t *outputData, size_t elementCount, size_t offset = 0);
  void read(int16_t *outputData, size_t elementCount, size_t offset = 0);
  void read(uint16_t *outputData, size_t elementCount, size_t offset = 0);
  void read(double *outputData, size_t elementCount, size_t offset = 0);
  void read(int64_t *outputData, size_t elementCount, size_t offset = 0);
  void read(uint64_t *outputData, size_t elementCount, size_t offset = 0);

  void readAsync(float *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(int32_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(uint32_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(int8_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(uint8_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(int16_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(uint16_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(double *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(int64_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(uint64_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);

  void release();

  size_t getLength() const { return elementCount; }
  size_t getSize() const { return bufferData.size; }
  // The context this buffer was created from (batched-upload routing).
  MGPU &getMGPU() const { return mgpu; }
  BufferDataType getDataType() const { return dataType; }
  WGPUBuffer getWGPUBuffer() const { return bufferData.buffer; }

  BufferData bufferData;

private:
  MGPU &mgpu;
  BufferDataType dataType = kUnknownType;
  size_t elementCount = 0;
  bool isPacked = false; // For types that need packing (8/16-bit, 64-bit)

  // Persistent staging buffer reused across readDirect calls to avoid
  // allocating/freeing a large (e.g. 5 MB) CopyDst|MapRead buffer on every
  // texture readback (at 20 fps that is ~100 MB/s of GPU heap churn).
  WGPUBuffer _readStagingBuffer = nullptr;
  size_t _readStagingBufferSize = 0;

  void releaseInternal();
  size_t getElementSize(BufferDataType type) const;
  bool needsPacking(BufferDataType type) const;

  template <typename T>
  void writeDirect(const T *inputData, size_t byteSize, BufferDataType type);

  template <typename T>
  void writePacked(const T *inputData, size_t byteSize, BufferDataType type);

  template <typename T>
  void readDirect(T *outputData, size_t elementCount, size_t offset);

  template <typename T>
  void readDirectChunk(T *outputData, size_t elementCount, size_t offset);

  template <typename T>
  void readPacked(T *outputData, size_t elementCount, size_t offset);

  template <typename T>
  void readAsyncImpl(T *outputData, size_t elementCount, size_t offset,
                     BufferDataType type, std::function<void()> callback);
};

} // namespace mgpu

#endif // BUFFER_H