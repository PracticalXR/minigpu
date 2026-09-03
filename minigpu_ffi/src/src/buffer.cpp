#include "../include/buffer.h"
#include "../include/compute_shader.h"
#include "../include/log.h"
#include "../include/mutex.h"
#include "../include/platform_sleep.h"
#include "../include/shader_cache.h"
#include "../include/wait_prof.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#ifndef __EMSCRIPTEN__
// Dawn C++ native API — lets us enumerate all adapters and pick by name
// without any platform-specific DXGI/Metal code.
#include "dawn/native/DawnNative.h"
#endif

#ifdef _WIN32
#include <dxgi1_4.h>
#include <d3d12.h>
#include <wrl/client.h>
#endif

namespace mgpu {

// ── Wait profiler (see wait_prof.h) ─────────────────────────────────────────
// Off unless MGPU_WAIT_PROF is set. Counters are relaxed atomics because the
// worker thread and the caller thread both write them and an exact total is
// not worth a lock in an instrument.
namespace {

const char *const kWaitSiteNames[WP_SITE_COUNT] = {
    "rd.lock",  "rd.flush", "rd.alloc", "rd.submit", "rd.map",
    "rd.copy",  "rd.events", "rd.TOTAL", "wr.lock",  "wr.flush",
    "wr.write", "wr.TOTAL", "rb.map",   "q.lat",     "drain",
};

struct WaitSiteAcc {
  std::atomic<long long> n{0};
  std::atomic<long long> us{0};
  std::atomic<long long> maxUs{0};
  std::atomic<long long> iters{0};
  // Time in events at or over the log threshold — "how much of the total is
  // tail", which is the whole question for a spike hunt.
  std::atomic<long long> tailN{0};
  std::atomic<long long> tailUs{0};
};

WaitSiteAcc g_waitAcc[WP_SITE_COUNT];
long long g_waitLogUs = 0;
long long g_waitT0 = 0;

int waitProfInit() {
  const char *s = std::getenv("MGPU_WAIT_PROF");
  const bool on = (s && s[0] && s[0] != '0');
  if (!on) return 0;
  const char *t = std::getenv("MGPU_WAIT_PROF_MS");
  // Default: report anything at or over 5 ms. At 60 fps a 5 ms wait is a third
  // of the frame, so nothing below that can be a spike on its own.
  long long ms = 5;
  if (t && t[0]) {
    char *end = nullptr;
    const long v = std::strtol(t, &end, 10);
    if (end && *end == '\0' && v >= 0) ms = v;
  }
  g_waitLogUs = ms * 1000;
  g_waitT0 = waitProfNowUs();
  std::atexit([]() { waitProfDump(); });
  std::fprintf(stderr,
               "[mgpu waitprof] ON — logging individual waits >= %lld ms\n",
               g_waitLogUs / 1000);
  return 1;
}

} // namespace

bool waitProfEnabled() {
  static const int on = waitProfInit();
  return on != 0;
}

void waitProfNote(int site, long long us, long long iters) {
  if (site < 0 || site >= WP_SITE_COUNT) return;
  WaitSiteAcc &a = g_waitAcc[site];
  a.n.fetch_add(1, std::memory_order_relaxed);
  a.us.fetch_add(us, std::memory_order_relaxed);
  a.iters.fetch_add(iters, std::memory_order_relaxed);
  long long prev = a.maxUs.load(std::memory_order_relaxed);
  while (us > prev &&
         !a.maxUs.compare_exchange_weak(prev, us, std::memory_order_relaxed)) {
  }
  if (g_waitLogUs > 0 && us >= g_waitLogUs) {
    a.tailN.fetch_add(1, std::memory_order_relaxed);
    a.tailUs.fetch_add(us, std::memory_order_relaxed);
    // t= is milliseconds since the first instrumented event, so a burst can be
    // lined up against the consumer's own per-frame log. spin= is the loop
    // iteration count where the site has one: a long wall against a tiny spin
    // count is the OS descheduling us, not the GPU being busy.
    std::fprintf(stderr, "[mgpu waitprof] t=%8.1f  %-9s %8.3f ms  spin=%lld\n",
                 (double)(waitProfNowUs() - g_waitT0) / 1000.0,
                 kWaitSiteNames[site], (double)us / 1000.0, iters);
  }
}

void waitProfDump() {
  if (!waitProfEnabled()) return;
  std::fprintf(stderr,
               "\n[mgpu waitprof] site        count      total ms"
               "     mean ms      max ms      spin/ev   >=thr  thr ms\n");
  for (int i = 0; i < WP_SITE_COUNT; ++i) {
    const long long n = g_waitAcc[i].n.load(std::memory_order_relaxed);
    if (n == 0) continue;
    const long long us = g_waitAcc[i].us.load(std::memory_order_relaxed);
    const long long mx = g_waitAcc[i].maxUs.load(std::memory_order_relaxed);
    const long long it = g_waitAcc[i].iters.load(std::memory_order_relaxed);
    const long long tn = g_waitAcc[i].tailN.load(std::memory_order_relaxed);
    const long long tu = g_waitAcc[i].tailUs.load(std::memory_order_relaxed);
    std::fprintf(stderr,
                 "[mgpu waitprof] %-9s %8lld  %12.3f %11.4f %11.3f %12lld "
                 "%7lld %8.1f\n",
                 kWaitSiteNames[i], n, (double)us / 1000.0,
                 (double)us / 1000.0 / (double)n, (double)mx / 1000.0, it / n,
                 tn, (double)tu / 1000.0);
  }
  std::fflush(stderr);
}

// ── Pre-init adapter preference (see buffer.h) ──────────────────────────────
static std::atomic<bool> g_preferDisplayAdapter{false};
static mgpu::mutex       g_selectedAdapterNameMutex;
static std::string       g_selectedAdapterName;

void setPreferDisplayAdapter(bool enable) {
  g_preferDisplayAdapter.store(enable, std::memory_order_relaxed);
}
bool preferDisplayAdapterEnabled() {
  return g_preferDisplayAdapter.load(std::memory_order_relaxed);
}
std::string selectedAdapterName() {
  mgpu::lock_guard<mgpu::mutex> lock(g_selectedAdapterNameMutex);
  return g_selectedAdapterName;
}
static void set_selected_adapter_name(const std::string& name) {
  mgpu::lock_guard<mgpu::mutex> lock(g_selectedAdapterNameMutex);
  g_selectedAdapterName = name;
}

WebGPUThread::WebGPUThread() {
#ifndef __EMSCRIPTEN__
  // Only create actual thread on native platforms
  worker = std::thread([this]() {
    while (true) {
      std::function<void()> task;
      {
        mgpu::unique_lock<mgpu::mutex> lock(queueMutex);
        condition.wait(lock, [this] { return !tasks.empty() || stop; });

        if (stop && tasks.empty())
          break;

        task = std::move(tasks.front());
        tasks.pop();
      }
      task();
    }
  });
#endif
}

WebGPUThread::~WebGPUThread() {
#ifndef __EMSCRIPTEN__
  {
    mgpu::lock_guard<mgpu::mutex> lock(queueMutex);
    stop = true;
  }
  condition.notify_all();
  if (worker.joinable()) {
    worker.join();
  }
#endif
}

void WebGPUThread::enqueueAsync(std::function<void()> task) {
#ifdef __EMSCRIPTEN__
  // In Emscripten, execute immediately on main thread
  if (task) {
    task();
  }
#else
  // Native platforms: use actual thread
  if (waitProfEnabled()) {
    // Queue latency (enqueue -> the task actually starting on the worker) is
    // the one component a caller can never see and the first thing a
    // wandering spike gets blamed on, so measure it rather than argue.
    const long long t0 = waitProfNowUs();
    std::function<void()> inner = std::move(task);
    task = [inner, t0]() {
      waitProfNote(WP_Q_LAT, waitProfNowUs() - t0, 0);
      if (inner) inner();
    };
  }
  {
    mgpu::lock_guard<mgpu::mutex> lock(queueMutex);
    tasks.push(std::move(task));
  }
  condition.notify_one();
#endif
}

void MGPU::flushBatchLocked() {
  if (batchPass) {
    wgpuComputePassEncoderEnd(batchPass);
    wgpuComputePassEncoderRelease(batchPass);
    batchPass = nullptr;
  }
  if (batchEncoder) {
    WGPUCommandBuffer commands =
        wgpuCommandEncoderFinish(batchEncoder, nullptr);
    wgpuCommandEncoderRelease(batchEncoder);
    batchEncoder = nullptr;
    if (commands) {
      wgpuQueueSubmit(getQueue(), 1, &commands);
      wgpuCommandBufferRelease(commands);
    }
  }
  batchCount = 0;
  // Everything recorded against the staging arena has now been SUBMITTED, so
  // its bytes are free to be overwritten: queue operations execute in
  // submission order, and the next staging write is issued after this submit.
  stagingCursor = 0;
}

// ── Batched staging uploads ────────────────────────────────────────────────
// Design notes live in buffer.h next to the declarations.

// Minimum persistent staging allocation.  Sized so a typical frame's worth of
// scattered pushes fits many times over: the arena is a bump allocator that
// only rewinds on a submit, so a bigger buffer means fewer forced submits.
static constexpr size_t kMinStagingBytes = 8u << 20; // 8 MB

bool MGPU::uploadProfEnabled() {
  static int on = -1;
  if (on < 0) {
    const char *s = std::getenv("MGPU_UPLOAD_PROF");
    on = (s && s[0] && s[0] != '0') ? 1 : 0;
  }
  return on != 0;
}

static inline long long mgpu_now_us() {
  return (long long)std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

MGPU::UploadScope *MGPU::acquireScope() {
  {
    mgpu::lock_guard<mgpu::mutex> lock(uploadPoolMutex);
    if (!uploadPool.empty()) {
      UploadScope *s = uploadPool.back().release();
      uploadPool.pop_back();
      s->host.clear();   // keeps capacity
      s->copies.clear(); // keeps capacity
      return s;
    }
  }
  return new UploadScope();
}

void MGPU::recycleScope(UploadScope *scope) {
  if (!scope) return;
  for (const StagedCopy &c : scope->copies) {
    if (c.dst) wgpuBufferRelease(c.dst);
  }
  scope->host.clear();
  scope->copies.clear();
  mgpu::lock_guard<mgpu::mutex> lock(uploadPoolMutex);
  if (uploadPool.size() < 4) {
    uploadPool.emplace_back(scope);
  } else {
    delete scope;
  }
}

int MGPU::beginUploads() {
  if (uploadDepth == 0) {
    if (!uploadOpen) uploadOpen.reset(acquireScope());
  }
  return ++uploadDepth;
}

void MGPU::reserveUploads(size_t byteSize) {
  if (uploadOpen && byteSize) uploadOpen->host.reserve(byteSize);
}

int MGPU::stageWrite(WGPUBuffer dst, size_t dstByteOffset,
                     size_t dstBufferSize, const void *src, size_t byteSize) {
  if (!dst || !src || byteSize == 0) return -1;
  // A staged range becomes a copyBufferToBuffer, which (like queue writes)
  // requires 4-byte aligned offsets and size.  Anything else must take the
  // inline path so behaviour is bit-identical to before.
  if ((byteSize & 3u) || (dstByteOffset & 3u)) return 0;
  if (dstByteOffset + byteSize > dstBufferSize) return -1;
  if (uploadDepth <= 0 || !uploadOpen) return 0;

  UploadScope &s = *uploadOpen;
  const bool prof = uploadProfEnabled();
  const long long t0 = prof ? mgpu_now_us() : 0;
  const uint64_t srcOffset = (uint64_t)s.host.size();
  const uint8_t *p = static_cast<const uint8_t *>(src);
  s.host.insert(s.host.end(), p, p + byteSize);
  if (prof) {
    uploadStats.usStage += mgpu_now_us() - t0;
    uploadStats.ranges++;
    uploadStats.bytes += (long long)byteSize;
  }
  // Coalesce with the previous range when it is contiguous in BOTH the arena
  // and the destination (dirty runs bridged by the producer land this way).
  if (!s.copies.empty()) {
    StagedCopy &last = s.copies.back();
    if (last.dst == dst && last.dstOffset + last.size == dstByteOffset &&
        last.srcOffset + last.size == srcOffset) {
      last.size += byteSize;
      return 1;
    }
  }
  StagedCopy c;
  c.dst = dst;
  c.dstOffset = dstByteOffset;
  c.srcOffset = srcOffset;
  c.size = byteSize;
  // The emission is asynchronous: hold a reference so a destination destroyed
  // between End and the WebGPU thread picking the scope up stays alive.
  wgpuBufferAddRef(dst);
  s.copies.push_back(c);
  return 1;
}

int MGPU::endUploads() {
  if (uploadDepth <= 0) return 0;
  if (--uploadDepth > 0) return 0;
  if (!uploadOpen) return 0;
  UploadScope *raw = uploadOpen.release();
  const int n = (int)raw->copies.size();
  if (n == 0) {
    recycleScope(raw);
    return 0;
  }
  // Emitted INLINE on the caller's thread — deliberately, so the ordering
  // contract is bit-for-bit the one the inline write path already has:
  // commands already RECORDED into the batch execute before these copies,
  // and a dispatch that has not been enqueued yet cannot overtake them.
  // (Emitting on the WebGPU thread instead would put the upload behind the
  // dispatch FIFO, which the inline read/write entry points do not join —
  // that would be a new race for every caller that mixes the two.)
  // What is saved is the per-range mutex round trip and the per-range
  // wgpuQueueSubmit, not the ordering.
  const bool prof = uploadProfEnabled();
  const long long tLock0 = prof ? mgpu_now_us() : 0;
  {
    mgpu::lock_guard<mgpu::mutex> lock(gpuOperationMutex);
    if (prof) uploadStats.usLock += mgpu_now_us() - tLock0;
    applyUploadScopeLocked(*raw);
  }
  if (prof) {
    uploadStats.scopes++;
    uploadStats.copies += n;
  }
  recycleScope(raw);
  return n;
}

void MGPU::uploadFallbackLocked(UploadScope &scope) {
  // Correct but unbatched: used only when the staging allocation could not be
  // created.  Queue writes are ordered by SUBMISSION time, so pending batched
  // dispatches must go first (this is what the inline path always does).
  WGPUQueue queue = tryGetQueue();
  if (!queue) return;
  flushBatchLocked();
  for (const StagedCopy &c : scope.copies) {
    wgpuQueueWriteBuffer(queue, c.dst, c.dstOffset,
                         scope.host.data() + c.srcOffset, (size_t)c.size);
  }
}

void MGPU::applyUploadScopeLocked(UploadScope &scope) {
  if (scope.copies.empty() || scope.host.empty()) return;
  if (!ctx || !ctx->initialized) return;
  WGPUDevice device = ctx->device;
  WGPUQueue queue = ctx->queue;
  if (!device || !queue) return;

  const size_t total = scope.host.size();
  if (stagingCapacity < total) {
    // Growing orphans copies already recorded against the OLD buffer, so the
    // pending batch must be submitted before the swap.
    flushBatchLocked();
    if (stagingBuffer) {
      wgpuBufferRelease(stagingBuffer);
      stagingBuffer = nullptr;
      stagingCapacity = 0;
    }
    size_t cap = kMinStagingBytes;
    while (cap < total * 2 && cap < (size_t)1 << 31) cap <<= 1;
    if (cap < total) cap = total;
    WGPUBufferDescriptor desc = {};
    desc.size = (uint64_t)cap;
    desc.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    desc.mappedAtCreation = false;
    stagingBuffer = wgpuDeviceCreateBuffer(device, &desc);
    if (!stagingBuffer) {
      LOG_ERROR("staging upload: failed to allocate %zu byte staging buffer",
                cap);
      stagingCapacity = 0;
      stagingCursor = 0;
      uploadFallbackLocked(scope);
      return;
    }
    stagingCapacity = cap;
    stagingCursor = 0;
  }
  const bool prof = uploadProfEnabled();
  if (stagingCursor + total > stagingCapacity) {
    // The arena would wrap onto bytes that recorded-but-unsubmitted copies
    // still read.  Submitting rewinds the cursor (see flushBatchLocked).
    const long long t = prof ? mgpu_now_us() : 0;
    flushBatchLocked();
    if (prof) {
      uploadStats.usFlush += mgpu_now_us() - t;
      uploadStats.flushes++;
    }
  }

  const long long tW = prof ? mgpu_now_us() : 0;
  const uint64_t base = (uint64_t)stagingCursor;
  wgpuQueueWriteBuffer(queue, stagingBuffer, base, scope.host.data(), total);
  stagingCursor = (size_t)base + total;
  if (prof) uploadStats.usWrite += mgpu_now_us() - tW;

  const long long tC = prof ? mgpu_now_us() : 0;
  if (!batchEncoder) {
    batchEncoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    if (!batchEncoder) {
      LOG_ERROR("staging upload: could not create command encoder");
      return;
    }
  }
  // copyBufferToBuffer is an encoder-level command: the open compute pass has
  // to close.  The next dispatch reopens one on the SAME encoder, so this
  // costs a pass boundary, not a submit.
  if (batchPass) {
    wgpuComputePassEncoderEnd(batchPass);
    wgpuComputePassEncoderRelease(batchPass);
    batchPass = nullptr;
  }
  for (const StagedCopy &c : scope.copies) {
    wgpuCommandEncoderCopyBufferToBuffer(batchEncoder, stagingBuffer,
                                         base + c.srcOffset, c.dst, c.dstOffset,
                                         c.size);
  }
  if (prof) uploadStats.usCopy += mgpu_now_us() - tC;
}

void MGPU::releaseStagingLocked() {
  if (stagingBuffer) {
    wgpuBufferRelease(stagingBuffer);
    stagingBuffer = nullptr;
  }
  stagingCapacity = 0;
  stagingCursor = 0;
}

// ── Batched readbacks ──────────────────────────────────────────────────────
// Design notes live in buffer.h next to the declarations.  The short version:
// the upload twin removes a SUBMIT STORM but is byte-bound at the margin; this
// removes a SUBMIT-AND-WAIT storm and is call-bound, so the win is per CALL.

static constexpr size_t kMinReadbackBytes = 4u << 20; // 4 MB

bool MGPU::readbackProfEnabled() {
  static int on = -1;
  if (on < 0) {
    const char *s = std::getenv("MGPU_READBACK_PROF");
    on = (s && s[0] && s[0] != '0') ? 1 : 0;
  }
  return on != 0;
}

int MGPU::beginReadbacks() {
  if (readbackDepth == 0 && !readbackOpen) {
    readbackOpen.reset(new ReadbackScope());
  }
  return ++readbackDepth;
}

void MGPU::reserveReadbacks(size_t byteSize) {
  if (!byteSize) return;
  mgpu::lock_guard<mgpu::mutex> lock(gpuOperationMutex);
  if (readbackCapacity >= byteSize) return;
  if (!ctx || !ctx->initialized || !ctx->device) return;
  if (readbackBuffer) {
    wgpuBufferRelease(readbackBuffer);
    readbackBuffer = nullptr;
    readbackCapacity = 0;
  }
  size_t cap = kMinReadbackBytes;
  while (cap < byteSize && cap < ((size_t)1 << 31)) cap <<= 1;
  if (cap < byteSize) cap = byteSize;
  WGPUBufferDescriptor desc = {};
  desc.size = (uint64_t)cap;
  desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
  desc.mappedAtCreation = false;
  readbackBuffer = wgpuDeviceCreateBuffer(ctx->device, &desc);
  readbackCapacity = readbackBuffer ? cap : 0;
  if (readbackProfEnabled() && readbackBuffer) readbackStats.grows++;
}

int MGPU::stageRead(WGPUBuffer src, size_t srcByteOffset, size_t srcBufferSize,
                    void *dst, size_t byteSize) {
  if (!src || !dst || byteSize == 0) return -1;
  // A staged range becomes a copyBufferToBuffer: 4-byte aligned offset AND
  // size.  Anything else takes the inline path so the answer is unchanged.
  if ((byteSize & 3u) || (srcByteOffset & 3u)) return 0;
  if (srcByteOffset + byteSize > srcBufferSize) return -1;
  if (readbackDepth <= 0 || !readbackOpen) return 0;

  ReadbackScope &s = *readbackOpen;
  const bool prof = readbackProfEnabled();
  const long long t0 = prof ? mgpu_now_us() : 0;
  const uint64_t slot = s.total;
  s.total += byteSize;

  StagedRead r;
  r.dst = dst;
  r.slotOffset = slot;
  r.size = byteSize;
  s.reads.push_back(r);

  // Coalesce with the previous copy when it is contiguous in BOTH the source
  // buffer and the readback slot.  The memcpy-out stays per staged read, so a
  // caller may split one contiguous device range across several destinations
  // and still pay a single copy command.
  if (!s.copies.empty()) {
    ReadCopy &last = s.copies.back();
    if (last.src == src && last.srcOffset + last.size == srcByteOffset &&
        last.slotOffset + last.size == slot) {
      last.size += byteSize;
      if (prof) {
        readbackStats.usStage += mgpu_now_us() - t0;
        readbackStats.reads++;
        readbackStats.bytes += (long long)byteSize;
      }
      return 1;
    }
  }
  ReadCopy c;
  c.src = src;
  c.srcOffset = srcByteOffset;
  c.slotOffset = slot;
  c.size = byteSize;
  // The source must survive until End even if the caller destroys the buffer
  // in between (same contract the upload twin has for destinations).
  wgpuBufferAddRef(src);
  s.copies.push_back(c);
  if (prof) {
    readbackStats.usStage += mgpu_now_us() - t0;
    readbackStats.reads++;
    readbackStats.bytes += (long long)byteSize;
  }
  return 1;
}

int MGPU::endReadbacks() {
  if (readbackDepth <= 0) return 0;
  if (--readbackDepth > 0) return 0;
  if (!readbackOpen) return 0;
  std::unique_ptr<ReadbackScope> scope(readbackOpen.release());
  const int n = (int)scope->reads.size();
  if (n == 0) return 0;

  const bool prof = readbackProfEnabled();
  const long long tLock0 = prof ? mgpu_now_us() : 0;
  {
    // INLINE on the caller's thread, under the same mutex Buffer::read takes —
    // so the ordering versus dispatches and inline reads/writes is exactly the
    // one the per-buffer path already had.
    mgpu::unique_lock<mgpu::mutex> lock(gpuOperationMutex);
    if (prof) readbackStats.usLock += mgpu_now_us() - tLock0;
    applyReadbackScopeLocked(*scope);
  }
  for (const ReadCopy &c : scope->copies) {
    if (c.src) wgpuBufferRelease(c.src);
  }
  if (prof) {
    readbackStats.scopes++;
    readbackStats.copies += (long long)scope->copies.size();
  }
  return n;
}

bool MGPU::waitMapReadLocked(WGPUBuffer buf, size_t size) {
  struct ReadState {
    bool completed = false;
    WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Success;
    mgpu::mutex mutex;
  };
  ReadState state;
  WGPUBufferMapCallbackInfo cb = {};
  cb.mode = WGPUCallbackMode_AllowSpontaneous;
  cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void *u1,
                   void *) {
    ReadState *st = static_cast<ReadState *>(u1);
    mgpu::lock_guard<mgpu::mutex> lk(st->mutex);
    st->status = status;
    st->completed = true;
  };
  cb.userdata1 = &state;

  WaitScope wpMap(WP_RB_MAP);
  WGPUFuture future = wgpuBufferMapAsync(buf, WGPUMapMode_Read, 0, size, cb);
  WGPUInstance instance = tryGetInstance();
  WGPUFutureWaitInfo waitInfo = {};
  waitInfo.future = future;
  if (instance) wgpuInstanceWaitAny(instance, 1, &waitInfo, 0);
  if (!waitInfo.completed) {
    // Same hot-poll as Buffer::readDirect: Sleep(1) actually sleeps ~15 ms
    // under the default Windows timer resolution, which would dominate.
    while (!state.completed) {
      platformSleep(0, instance);
      ++wpMap.iters;
    }
  }
  return state.status == WGPUMapAsyncStatus_Success;
}

bool MGPU::readOneLocked(WGPUBuffer src, uint64_t srcOffset, void *dst,
                         uint64_t size) {
  if (!ctx || !ctx->initialized) return false;
  WGPUDevice device = ctx->device;
  WGPUQueue queue = ctx->queue;
  if (!device || !queue) return false;
  WGPUBufferDescriptor desc = {};
  desc.size = size;
  desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
  WGPUBuffer tmp = wgpuDeviceCreateBuffer(device, &desc);
  if (!tmp) return false;
  flushBatchLocked();
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, nullptr);
  if (!enc) {
    wgpuBufferRelease(tmp);
    return false;
  }
  wgpuCommandEncoderCopyBufferToBuffer(enc, src, srcOffset, tmp, 0, size);
  WGPUCommandBuffer cmds = wgpuCommandEncoderFinish(enc, nullptr);
  if (cmds) {
    wgpuQueueSubmit(queue, 1, &cmds);
    wgpuCommandBufferRelease(cmds);
  }
  wgpuCommandEncoderRelease(enc);
  bool ok = false;
  if (waitMapReadLocked(tmp, (size_t)size)) {
    const void *p = wgpuBufferGetConstMappedRange(tmp, 0, (size_t)size);
    if (p) {
      std::memcpy(dst, p, (size_t)size);
      ok = true;
    }
    wgpuBufferUnmap(tmp);
  }
  wgpuBufferDestroy(tmp);
  wgpuBufferRelease(tmp);
  return ok;
}

void MGPU::applyReadbackScopeLocked(ReadbackScope &scope) {
  if (scope.reads.empty()) return;
  if (!ctx || !ctx->initialized) return;
  WGPUDevice device = ctx->device;
  WGPUQueue queue = ctx->queue;
  if (!device || !queue) return;

  const bool prof = readbackProfEnabled();
  const size_t total = (size_t)scope.total;

  if (readbackCapacity < total) {
    // Growing orphans nothing (the buffer is only ever read within one scope),
    // but the old allocation must be gone before the new one is recorded into.
    if (readbackBuffer) {
      wgpuBufferRelease(readbackBuffer);
      readbackBuffer = nullptr;
      readbackCapacity = 0;
    }
    size_t cap = kMinReadbackBytes;
    while (cap < total && cap < ((size_t)1 << 31)) cap <<= 1;
    if (cap < total) cap = total;
    WGPUBufferDescriptor desc = {};
    desc.size = (uint64_t)cap;
    desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    desc.mappedAtCreation = false;
    readbackBuffer = wgpuDeviceCreateBuffer(device, &desc);
    if (!readbackBuffer) {
      LOG_ERROR("batched readback: failed to allocate %zu byte buffer", cap);
      readbackCapacity = 0;
      if (prof) readbackStats.fallbacks++;
      // Correct but unbatched: exactly what the per-buffer path would have done.
      for (const ReadCopy &c : scope.copies) {
        // Resolve straight into the destinations that map onto this copy.
        for (const StagedRead &r : scope.reads) {
          if (r.slotOffset < c.slotOffset ||
              r.slotOffset + r.size > c.slotOffset + c.size)
            continue;
          readOneLocked(c.src, c.srcOffset + (r.slotOffset - c.slotOffset),
                        r.dst, r.size);
        }
      }
      return;
    }
    readbackCapacity = cap;
    if (prof) readbackStats.grows++;
  }

  // ── ONE submit ───────────────────────────────────────────────────────────
  // The copies go into the encoder the compute batch is ALREADY building, so
  // the single flush below carries the pending dispatches AND every copy: the
  // per-read `flushBatchLocked() + own encoder + own submit` disappears.
  const long long tC = prof ? mgpu_now_us() : 0;
  if (!batchEncoder) {
    batchEncoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    if (!batchEncoder) {
      LOG_ERROR("batched readback: could not create command encoder");
      return;
    }
  }
  if (batchPass) {
    wgpuComputePassEncoderEnd(batchPass);
    wgpuComputePassEncoderRelease(batchPass);
    batchPass = nullptr;
  }
  for (const ReadCopy &c : scope.copies) {
    wgpuCommandEncoderCopyBufferToBuffer(batchEncoder, c.src, c.srcOffset,
                                         readbackBuffer, c.slotOffset, c.size);
  }
  if (prof) readbackStats.usCopy += mgpu_now_us() - tC;

  const long long tS = prof ? mgpu_now_us() : 0;
  flushBatchLocked();
  if (prof) readbackStats.usSubmit += mgpu_now_us() - tS;

  // ── ONE fence + ONE map ──────────────────────────────────────────────────
  const long long tM = prof ? mgpu_now_us() : 0;
  const bool mapped = waitMapReadLocked(readbackBuffer, total);
  if (prof) readbackStats.usMap += mgpu_now_us() - tM;
  if (!mapped) {
    LOG_ERROR("batched readback: map failed");
    return;
  }
  const long long tO = prof ? mgpu_now_us() : 0;
  const uint8_t *base = static_cast<const uint8_t *>(
      wgpuBufferGetConstMappedRange(readbackBuffer, 0, total));
  if (base) {
    for (const StagedRead &r : scope.reads) {
      std::memcpy(r.dst, base + r.slotOffset, (size_t)r.size);
    }
  } else {
    LOG_ERROR("batched readback: could not get mapped range");
  }
  wgpuBufferUnmap(readbackBuffer);
  if (prof) readbackStats.usOut += mgpu_now_us() - tO;
}

void MGPU::releaseReadbackLocked() {
  if (readbackBuffer) {
    wgpuBufferRelease(readbackBuffer);
    readbackBuffer = nullptr;
  }
  readbackCapacity = 0;
}

// ── Context lifecycle serialization ─────────────────────────────────────────
// See the long comment on MGPU::CtxState in buffer.h for WHY. The shape:
//
//   lock -> wait out any in-flight Initializing/Destroying -> claim the state
//        -> UNLOCK -> do the slow work -> relock -> publish the result -> notify
//
// The lock is never held across the work, so destroyContext()'s
// webgpuThread.enqueueSync() can always make progress. A scope guard publishes
// the terminal state so a throw out of the device request cannot strand the
// state at Initializing (which would hang every other thread forever).
namespace {
// MGPU_UNSAFE_NO_CTX_LOCK=1 restores the pre-fix behaviour: no state machine,
// no reference counting. It exists so the regression can be demonstrated to
// come BACK on demand (positive control) without reverting the source. Never
// set it in production — it is the bug.
bool ctxLockBypassEnabled() {
  static int on = -1;
  if (on < 0) {
    const char *e = std::getenv("MGPU_UNSAFE_NO_CTX_LOCK");
    on = (e && e[0] && e[0] != '0') ? 1 : 0;
    if (on) {
      MGPU_LOG(mgpu::LOG_WARN,
          "[mgpu] MGPU_UNSAFE_NO_CTX_LOCK=1: context lifecycle serialization "
          "and reference counting are DISABLED (positive-control mode).");
    }
  }
  return on != 0;
}
} // namespace

void MGPU::initializeContext() {
  if (ctxLockBypassEnabled()) {
    if (ctx && ctx->initialized) return;
    initializeContextUnlocked();
    return;
  }

  const std::thread::id self = std::this_thread::get_id();
  {
    mgpu::unique_lock<mgpu::mutex> lk(ctxStateMutex);
    for (;;) {
      if (ctxState != CtxState::Initializing && ctxState != CtxState::Destroying)
        break;
      // Re-entrant call from the thread already doing the work: it is mid-init,
      // so there is nothing to wait for and waiting would self-deadlock.
      if (ctxOwner == self) return;
      // The WebGPU thread must never block here: destroyContext() runs its
      // drain with webgpuThread.enqueueSync(), so a lazy re-init that parked
      // this thread would stall the very teardown it is waiting on.
      if (ctxState == CtxState::Destroying && self == webgpuThread.threadId()) {
        MGPU_LOG(mgpu::LOG_WARN,
            "[mgpu] initializeContext() on the WebGPU thread during teardown - "
            "not waiting (would deadlock the teardown).");
        return;
      }
      ctxStateCv.wait(lk);
    }
    // Ready AND actually usable -> nothing to do. `ctx->initialized` can be
    // false while the state is Ready when the device-lost callback fired; that
    // falls through and re-initializes, exactly as before.
    if (ctxState == CtxState::Ready && ctx && ctx->initialized) return;
    ctxState = CtxState::Initializing;
    ctxOwner = self;
  }

  // ── Slow work, NO lock held ───────────────────────────────────────────────
  bool ok = false;
  struct Publish {
    MGPU *m;
    const bool *ok;
    ~Publish() {
      mgpu::lock_guard<mgpu::mutex> lk(m->ctxStateMutex);
      m->ctxState = *ok ? MGPU::CtxState::Ready : MGPU::CtxState::Uninitialized;
      m->ctxOwner = std::thread::id();
      m->ctxStateCv.notify_all();
    }
  } publish{this, &ok};

  initializeContextUnlocked(); // may throw -> Publish restores Uninitialized
  ok = ctx && ctx->initialized;
}

void MGPU::initializeContextUnlocked() {
  if (ctx && ctx->initialized) {
    return; // Already initialized
  }
  MGPU_LOG(mgpu::LOG_DEBUG,
      "[mgpu] initializeContext() entering: ctx=%p ctx_initialized=%d",
      (void*)ctx.get(),
      ctx ? (int)ctx->initialized : -1);

  ctx = std::make_unique<Context>();
  // Local alias for the Context being built. Only ONE thread reaches here at a
  // time (the caller holds the Initializing state), so `c` stays valid for the
  // whole function; using it instead of re-reading `ctx` also documents that
  // everything below belongs to THIS initialization.
  Context *const c = ctx.get();

  // Create WebGPU instance via dawn::native::Instance on native platforms so
  // we can use EnumerateAdapters for MGPU_ADAPTER_NAME selection below.
  WGPUInstanceDescriptor instanceDesc = {};
  instanceDesc.nextInChain = nullptr;
#ifndef __EMSCRIPTEN__
  {
    auto* nativeInst = new dawn::native::Instance(&instanceDesc);
    c->dawnNativeInstance = nativeInst;
    c->instance = nativeInst->Get();
    // Take our own ref so destroyContext's wgpuInstanceRelease is balanced
    // regardless of whether the native wrapper also holds one.
    wgpuInstanceAddRef(c->instance);
  }
#else
  c->instance = wgpuCreateInstance(&instanceDesc);
#endif

  if (!c->instance) {
    throw std::runtime_error("Failed to create WebGPU instance");
  }

  // Request adapter - Updated API
  WGPURequestAdapterOptions adapterOptions = {};
#ifndef __EMSCRIPTEN__
  adapterOptions.powerPreference = WGPUPowerPreference_HighPerformance;
  // Default to Dawn's D3D11 backend on Windows so cross-API texture sharing
  // with WGC capture (D3D11) and FFmpeg D3D11VA encoders never has to cross
  // the D3D11/D3D12 boundary (which on NVIDIA fails the d3d11on12 wrap and
  // creates GPU hazards). The user can opt back into D3D12 by setting
  // MGPU_BACKEND=d3d12 / vulkan / metal / opengl / undefined.
  adapterOptions.backendType = WGPUBackendType_Undefined;
  // Set by Win32 detection when cross-adapter is detected but Tier B
  // (D3D12 cross-adapter bridge) is unavailable (e.g. AMD iGPU).
  // When non-empty, adapter selection below prefers the display adapter
  // so Dawn and capture share the same GPU → Tier A zero-copy (Tier A*).
  std::string autoDisplayAdapterName;
#ifdef _WIN32
  // Windows backend selection.
  //
  // We detect the likely GPU topology at startup using DXGI and pick the
  // best strategy automatically.  Three cases:
  //
  //   Tier A  (most systems): capture and compute on the SAME adapter.
  //     D3D11 backend, dGPU selected.  OpenSharedResource1 works because
  //     the NT handle was created on the same device.
  //
  //   Tier B  (Optimus-style, Intel iGPU + dGPU, iGPU drives display):
  //     dGPU has no outputs AND it supports D3D12
  //     CrossAdapterRowMajorTextureSupported.  Use D3D12 backend so
  //     minigpu_external can do a GPU-only cross-adapter copy.
  //
  //   Tier A* (Optimus-style, AMD/other iGPU + dGPU, iGPU drives display,
  //     NO CrossAdapterRowMajorTextureSupported):
  //     Keep D3D11 backend but let adapter selection pick the display
  //     adapter (iGPU) so capture and compute are on the same device →
  //     standard Tier A zero-copy.  Compute runs on the iGPU rather than
  //     the dGPU, but that avoids a CPU staging round-trip per frame.
  //     The user can override with MGPU_ADAPTER_NAME=<dGPU substring> to
  //     get dGPU compute at the cost of Tier C CPU bridge.
  //
  // MGPU_BACKEND / MGPU_ADAPTER_NAME env vars override everything.
  {
    bool needsCrossAdapterBridge = false;
    Microsoft::WRL::ComPtr<IDXGIFactory1> fac;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&fac)))) {
      // Explicit pre-init preference (mgpuPreferDisplayAdapter): bind Dawn to
      // the adapter driving the PRIMARY display (the attached output whose
      // desktop coordinates contain the origin; fallback: first adapter with
      // any attached output). Used by capture/record apps so screen capture,
      // GPU processing and the HW encoder share one adapter (zero-copy) even
      // on multi-output hybrid systems where the discrete GPU also drives a
      // monitor — there the "dGPU has no outputs" heuristic below can't see
      // the capture/compute split. When the resolve succeeds the topology
      // auto-detect below is skipped (its guard checks autoDisplayAdapterName).
      if (preferDisplayAdapterEnabled()) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> a;
        std::string firstOutputName;
        for (UINT i = 0; autoDisplayAdapterName.empty() &&
                         SUCCEEDED(fac->EnumAdapters1(i, &a)); ++i, a.Reset()) {
          DXGI_ADAPTER_DESC1 d{};
          a->GetDesc1(&d);
          if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
          Microsoft::WRL::ComPtr<IDXGIOutput> o;
          for (UINT j = 0; SUCCEEDED(a->EnumOutputs(j, &o)); ++j, o.Reset()) {
            DXGI_OUTPUT_DESC od{};
            if (FAILED(o->GetDesc(&od)) || !od.AttachedToDesktop) continue;
            char name[128]{};
            WideCharToMultiByte(CP_UTF8, 0, d.Description, -1,
                                 name, sizeof(name), nullptr, nullptr);
            if (firstOutputName.empty()) firstOutputName = name;
            if (od.DesktopCoordinates.left == 0 && od.DesktopCoordinates.top == 0) {
              autoDisplayAdapterName = name; // primary display
              break;
            }
          }
        }
        if (autoDisplayAdapterName.empty()) autoDisplayAdapterName = firstOutputName;
        if (autoDisplayAdapterName.empty()) {
          MGPU_LOG(mgpu::LOG_WARN,
              "[mgpu] preferDisplayAdapter: no attached display output found - "
              "using normal auto-select.");
        }
      }
      // Pick the highest-perf real adapter (= the one we'd normally use for
      // compute): largest dedicated VRAM, discrete preferred.
      Microsoft::WRL::ComPtr<IDXGIAdapter1> bestAdapter;
      SIZE_T bestVram = 0;
      bool   bestIsDiscrete = false;
      UINT   realCount = 0;
      Microsoft::WRL::ComPtr<IDXGIAdapter1> a;
      for (UINT i = 0; SUCCEEDED(fac->EnumAdapters1(i, &a)); ++i, a.Reset()) {
        DXGI_ADAPTER_DESC1 d{};
        a->GetDesc1(&d);
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        ++realCount;
        bool isDiscrete = d.DedicatedVideoMemory > (256ull * 1024 * 1024)
                       && d.DedicatedVideoMemory >= d.SharedSystemMemory / 2;
        bool better = false;
        if (!bestAdapter)                                better = true;
        else if (isDiscrete && !bestIsDiscrete)          better = true;
        else if (isDiscrete == bestIsDiscrete &&
                 d.DedicatedVideoMemory > bestVram)      better = true;
        if (better) { bestAdapter = a; bestVram = d.DedicatedVideoMemory; bestIsDiscrete = isDiscrete; }
      }
      // Cross-adapter case: hybrid system AND best adapter has no outputs.
      // (Skipped when preferDisplayAdapter already resolved a display adapter.)
      if (autoDisplayAdapterName.empty() && realCount > 1 && bestAdapter) {
        Microsoft::WRL::ComPtr<IDXGIOutput> firstOut;
        HRESULT outHr = bestAdapter->EnumOutputs(0, &firstOut);
        if (outHr == DXGI_ERROR_NOT_FOUND) {
          // The compute (dGPU) adapter has no outputs, so the display — and
          // therefore the capture producer (Desktop Duplication / WGC) — lives
          // on a DIFFERENT adapter (the iGPU).  Find that display adapter first;
          // it is the producer whose capability actually gates Tier B.
          Microsoft::WRL::ComPtr<IDXGIAdapter1> displayAdapter;
          std::string displayName;
          {
            Microsoft::WRL::ComPtr<IDXGIAdapter1> dispA;
            for (UINT i = 0; SUCCEEDED(fac->EnumAdapters1(i, &dispA)); ++i, dispA.Reset()) {
              DXGI_ADAPTER_DESC1 d{};
              dispA->GetDesc1(&d);
              if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
              Microsoft::WRL::ComPtr<IDXGIOutput> o;
              if (SUCCEEDED(dispA->EnumOutputs(0, &o))) {
                char name[128]{};
                WideCharToMultiByte(CP_UTF8, 0, d.Description, -1,
                                     name, sizeof(name), nullptr, nullptr);
                displayName    = name;
                displayAdapter = dispA;
                break;
              }
            }
          }

          // Determine which cross-adapter strategy is available.
          //
          // Tier B allocates the cross-adapter committed texture on the
          // PRODUCER's D3D12 device (see ensure_cross_adapter_texture() in
          // minigpu_external.cpp — CreateCommittedResource with
          // ALLOW_CROSS_ADAPTER runs on info->b12Dev, the producer adapter).
          // So the capability that gates Tier B is the PRODUCER's (display /
          // iGPU) CrossAdapterRowMajorTextureSupported, NOT the dGPU's — the
          // dGPU only OPENS the shared handle, which needs no such capability.
          // Checking bestAdapter (dGPU) here made us commit to the D3D12 /
          // Tier-B backend on machines where the iGPU cannot allocate the
          // shared texture, then fall to the Tier C CPU bridge on every frame.
          bool tierBAvailable = false;
          if (displayAdapter) {
            Microsoft::WRL::ComPtr<ID3D12Device> d12;
            if (SUCCEEDED(D3D12CreateDevice(displayAdapter.Get(),
                    D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d12))) && d12) {
              D3D12_FEATURE_DATA_D3D12_OPTIONS opts{};
              d12->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,
                                        &opts, sizeof(opts));
              tierBAvailable = opts.CrossAdapterRowMajorTextureSupported;
            }
          }

          // 🔴 TIER B NO LONGER SWITCHES THE BACKEND — see the D3D11 default
          // below. It is still DETECTED, because the topology is worth
          // reporting and it is what MGPU_BACKEND=d3d12 turns back on; it just
          // no longer decides anything.
          needsCrossAdapterBridge = tierBAvailable;
          if (!displayName.empty()) {
            // Tier A*: prefer the display adapter (iGPU) for Dawn so capture and
            // compute share the same device → zero-copy, and the FFmpeg HW
            // encoder (aligned to Dawn's adapter) stays on the iGPU too.
            autoDisplayAdapterName = displayName;
          }
        }
      }
    }
    // 🔴 **D3D11 IS THE DEFAULT ON WINDOWS, UNCONDITIONALLY.**
    //
    // Tier B used to select D3D12 so `minigpu_external` could do a GPU-only
    // cross-adapter copy on hybrid laptops. That optimised INGEST while
    // silently breaking PRESENT: `create_shared_output_texture` is implemented
    // only for the D3D11 backend and returns null on D3D12, so a Tier B machine
    // got a fast capture path and no GPU present surface at all — every decoded
    // frame fell back to a full readback (measured at 84-99 ms/frame).
    //
    // The asymmetry is not arbitrary. Windows fixes the API at both ends: the
    // producers (WGC, Media Foundation) hand out D3D11 textures, and Flutter's
    // external-texture API takes `kFlutterDesktopGpuSurfaceTypeD3d11Texture2D`.
    // Dawn's backend is the only free choice, so choosing D3D11 is what makes
    // capture -> compute -> present one API on one adapter with no bridges.
    // Choosing D3D12 buys one bridge and owes two.
    //
    // The cost is real and accepted: on a Tier B laptop the cross-adapter
    // INGEST copy now goes through Tier C rather than the D3D12 bridge. That is
    // the cheaper thing to lose — a copy on the way in is slower, whereas no
    // present surface is a broken picture. `MGPU_BACKEND=d3d12` restores the
    // old behaviour for anyone who wants that trade.
    adapterOptions.backendType = WGPUBackendType_D3D11;
    if (needsCrossAdapterBridge) {
      MGPU_LOG(mgpu::LOG_INFO,
          "[mgpu] backend auto-select: D3D11 (cross-adapter topology detected, but "
          "D3D12's Tier B bridge would disable GPU present - set MGPU_BACKEND=d3d12 "
          "to prefer ingest speed over presentation)");
    } else if (!autoDisplayAdapterName.empty() && preferDisplayAdapterEnabled()) {
      MGPU_LOG(mgpu::LOG_INFO,
          "[mgpu] backend auto-select: D3D11 (preferDisplayAdapter — binding to "
          "primary-display adapter '%s' for same-adapter capture zero-copy)",
          autoDisplayAdapterName.c_str());
    } else if (!autoDisplayAdapterName.empty()) {
      MGPU_LOG(mgpu::LOG_INFO,
          "[mgpu] backend auto-select: D3D11 (Tier A* iGPU zero-copy — dGPU lacks "
          "CrossAdapterRowMajorTextureSupported; preferring display adapter '%s')",
          autoDisplayAdapterName.c_str());
    } else {
      MGPU_LOG(mgpu::LOG_INFO, "[mgpu] backend auto-select: D3D11 (Tier A same-adapter zero-copy)");
    }
  }
#endif
  // MGPU_BACKEND env var: override the auto-detected backend.
  // Values: d3d11 | d3d12 | vulkan | metal | opengl | opengles | undefined
  if (const char* be = std::getenv("MGPU_BACKEND")) {
    std::string s(be);
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    if      (s == "d3d11")     adapterOptions.backendType = WGPUBackendType_D3D11;
    else if (s == "d3d12")     adapterOptions.backendType = WGPUBackendType_D3D12;
    else if (s == "vulkan")    adapterOptions.backendType = WGPUBackendType_Vulkan;
    else if (s == "metal")     adapterOptions.backendType = WGPUBackendType_Metal;
    else if (s == "opengl")    adapterOptions.backendType = WGPUBackendType_OpenGL;
    else if (s == "opengles")  adapterOptions.backendType = WGPUBackendType_OpenGLES;
    else if (s == "undefined") adapterOptions.backendType = WGPUBackendType_Undefined;
  }
  // Adapter selection via dawn::native::EnumerateAdapters.
  //
  // Automatic preference order (no env vars required):
  //   1. Discrete GPU  (dedicated adapter — NVIDIA, AMD, etc.)
  //   2. Integrated GPU
  //   3. Any non-CPU adapter
  //
  // This is more reliable than WGPUPowerPreference_HighPerformance, which is
  // only a hint and has been observed to pick the wrong adapter on Optimus
  // laptops. By checking adapterType explicitly we always land on the
  // dedicated GPU when one is present.
  //
  // FFmpeg's D3D11VA encoder receives the SAME ID3D11Device via
  // createD3D11DeviceOnDawnAdapter(), so the encoder is automatically
  // aligned to whichever GPU minigpu selected — no separate env var needed.
  //
  // Power-user override: MGPU_ADAPTER_NAME=<substring> (case-insensitive
  // match against the device name, e.g. "Intel" or "NVIDIA"). Useful when
  // the display is on the iGPU and cross-adapter sharing causes issues.
  {
    auto* nativeInst = static_cast<dawn::native::Instance*>(c->dawnNativeInstance);

    // Enumerate without powerPreference so we see ALL adapters for this
    // backend and can sort by adapterType ourselves.
    WGPURequestAdapterOptions enumOpts = adapterOptions;
    enumOpts.powerPreference = WGPUPowerPreference_Undefined;
    auto adapters = nativeInst->EnumerateAdapters(&enumOpts);

    // Collect info for all adapters up-front and log them.
    struct Entry {
      WGPUAdapter        handle = nullptr;
      WGPUAdapterType    type   = WGPUAdapterType_Unknown;
      std::string        name;
    };
    std::vector<Entry> entries;
    entries.reserve(adapters.size());
    for (std::size_t i = 0; i < adapters.size(); ++i) {
      Entry e;
      e.handle = adapters[i].Get();
      WGPUAdapterInfo info{};
      if (wgpuAdapterGetInfo(e.handle, &info) == WGPUStatus_Success) {
        e.type = info.adapterType;
        if (info.device.data && info.device.length)
          e.name = std::string(info.device.data, info.device.length);
        MGPU_LOG(mgpu::LOG_DEBUG,
            "[mgpu] adapter[%zu]: '%s' type=%d backend=%d vendor=0x%X device=0x%X",
            i, e.name.c_str(), (int)info.adapterType, (int)info.backendType,
            (unsigned)info.vendorID, (unsigned)info.deviceID);
        wgpuAdapterInfoFreeMembers(info);
      }
      entries.push_back(std::move(e));
    }

    // 1. Adapter-name override (substring, case-insensitive): the
    //    per-instance filter (multi-GPU contexts) wins over the
    //    MGPU_ADAPTER_NAME env var.
    const char* envFilter = std::getenv("MGPU_ADAPTER_NAME");
    const std::string activeFilter =
        !adapterFilter.empty() ? adapterFilter
                               : (envFilter ? std::string(envFilter) : "");
    if (!activeFilter.empty()) {
      std::string filter(activeFilter);
      for (auto& c : filter) c = (char)std::tolower((unsigned char)c);
      for (auto& e : entries) {
        if (e.type == WGPUAdapterType_CPU) continue;
        std::string lower = e.name;
        for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
        if (lower.find(filter) != std::string::npos) {
          c->adapter = e.handle;
          wgpuAdapterAddRef(c->adapter);
          MGPU_LOG(mgpu::LOG_INFO, "[mgpu] adapter filter '%s': selected '%s'",
              activeFilter.c_str(), e.name.c_str());
          break;
        }
      }
      if (!c->adapter)
        MGPU_LOG(mgpu::LOG_WARN,
            "[mgpu] adapter filter '%s': no match - using auto-select",
            activeFilter.c_str());
    }

    // 1.5. Display-adapter binding. autoDisplayAdapterName is set either by
    //      the explicit preferDisplayAdapter pre-init hint (primary-display
    //      adapter) or by Tier A* auto-detect (cross-adapter topology where
    //      Tier B is unavailable — dGPU lacks
    //      CrossAdapterRowMajorTextureSupported, e.g. AMD iGPU + dGPU).
    //      Either way, prefer that display adapter so capture (WGC, Desktop
    //      Duplication) and Dawn compute share the same device → Tier A
    //      zero-copy, same as a single-GPU system. Compute runs on the
    //      display GPU rather than the dGPU, but avoids a CPU staging
    //      round-trip per frame.
    //
    //      The user can force dGPU compute (accepting Tier C CPU bridge) by
    //      setting MGPU_ADAPTER_NAME to a substring of the dGPU name.
    if (!c->adapter && !autoDisplayAdapterName.empty()) {
      // Case-insensitive exact match against the DXGI description stored at
      // detection time.  Dawn's device name comes from the same driver string
      // so they should match exactly.
      std::string target = autoDisplayAdapterName;
      for (auto& c : target) c = (char)std::tolower((unsigned char)c);
      for (auto& e : entries) {
        if (e.type == WGPUAdapterType_CPU) continue;
        std::string lower = e.name;
        for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
        if (lower == target) {
          c->adapter = e.handle;
          wgpuAdapterAddRef(c->adapter);
          if (preferDisplayAdapterEnabled()) {
            MGPU_LOG(mgpu::LOG_INFO,
                "[mgpu] preferDisplayAdapter: selected primary-display adapter "
                "'%s' for zero-copy capture import. "
                "(Set MGPU_ADAPTER_NAME=<name> to override.)",
                e.name.c_str());
          } else {
            MGPU_LOG(mgpu::LOG_INFO,
                "[mgpu] Tier A*: selected display adapter '%s' for zero-copy capture import. "
                "(dGPU lacks CrossAdapterRowMajorTextureSupported — iGPU compute used. "
                "Set MGPU_ADAPTER_NAME=<dGPU> for dGPU compute with Tier C CPU bridge.)",
                e.name.c_str());
          }
          break;
        }
      }
    }

    // 2. Automatic: discrete GPU first, then integrated, then any non-CPU.
    //    Exception: when Tier A* was intended (autoDisplayAdapterName set) but
    //    the exact display-name match above didn't bind, we still want the
    //    iGPU — matching by name can miss if Dawn's device string differs from
    //    the DXGI description.  Preferring Integrated first keeps capture and
    //    compute on the same (display) adapter → zero-copy instead of Tier C.
    if (!c->adapter) {
      const bool preferIntegrated = !autoDisplayAdapterName.empty();
      const WGPUAdapterType kDiscreteFirst[] = {
        WGPUAdapterType_DiscreteGPU,   // dedicated adapter (NVIDIA/AMD)
        WGPUAdapterType_IntegratedGPU, // built-in (Intel/Apple)
        WGPUAdapterType_Unknown,       // catch-all (matches any non-CPU)
      };
      const WGPUAdapterType kIntegratedFirst[] = {
        WGPUAdapterType_IntegratedGPU, // Tier A*: the display adapter is the iGPU
        WGPUAdapterType_DiscreteGPU,
        WGPUAdapterType_Unknown,
      };
      const WGPUAdapterType* kPasses = preferIntegrated ? kIntegratedFirst : kDiscreteFirst;
      for (auto want : {kPasses[0], kPasses[1], kPasses[2]}) {
        if (c->adapter) break;
        for (auto& e : entries) {
          if (e.type == WGPUAdapterType_CPU) continue;
          // For the catch-all pass accept anything; otherwise require exact type.
          if (want != WGPUAdapterType_Unknown && e.type != want) continue;
          c->adapter = e.handle;
          wgpuAdapterAddRef(c->adapter);
          MGPU_LOG(mgpu::LOG_INFO,
              "[mgpu] auto-selected '%s' (adapterType=%d"
              " - 1=Discrete 2=Integrated 4=Unknown)",
              e.name.c_str(), (int)e.type);
          break;
        }
      }
    }

    if (!c->adapter)
      MGPU_LOG(mgpu::LOG_WARN,
          "[mgpu] EnumerateAdapters: no usable adapter found"
          " - falling back to wgpuInstanceRequestAdapter");
  }
#endif

  struct AdapterRequestState {
    WGPUAdapter adapter = nullptr;
    bool completed = false;
    mgpu::mutex mutex;
    std::condition_variable cv;
  };

  LOG_INFO("Requesting WebGPU adapter...");

  AdapterRequestState adapterState;

  // Updated callback structure for newer WebGPU API
  WGPURequestAdapterCallbackInfo adapterCallbackInfo = {};
  adapterCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
  adapterCallbackInfo.callback = [](WGPURequestAdapterStatus status,
                                    WGPUAdapter adapter, WGPUStringView message,
                                    void *userdata1, void *userdata2) {
    AdapterRequestState *state = static_cast<AdapterRequestState *>(userdata1);
    mgpu::lock_guard<mgpu::mutex> lock(state->mutex);

    if (status == WGPURequestAdapterStatus_Success) {
      state->adapter = adapter;
    }
    state->completed = true;
    state->cv.notify_one();
  };
  adapterCallbackInfo.userdata1 = &adapterState;

  // Only call wgpuInstanceRequestAdapter if EnumerateAdapters above found
  // nothing (e.g. very old driver, Emscripten, unsupported platform).
  WGPUFuture adapterFuture{};
  if (!c->adapter) {
    adapterFuture = wgpuInstanceRequestAdapter(
        c->instance, &adapterOptions, adapterCallbackInfo);
    // Process events until the adapter request completes
    while (!adapterState.completed) {
      platformSleep(1, c->instance);
    }
  } else {
    // Adapter already selected via EnumerateAdapters — synthesise completed state.
    adapterState.adapter = c->adapter;
    adapterState.completed = true;
  }

  LOG_INFO("WebGPU adapter request completed");

  if (!adapterState.adapter) {
    wgpuInstanceRelease(c->instance);
    throw std::runtime_error("Failed to get WebGPU adapter");
  }

  c->adapter = adapterState.adapter;

  // Log which adapter was selected so it's easy to confirm the right GPU.
  // adapterType: 1=DiscreteGpu 2=IntegratedGpu 3=Cpu 4=Unknown
  //
  // These four also feed the shader cache's isolation key below, so they are
  // copied out here — WGPUAdapterInfo's strings are freed by
  // wgpuAdapterInfoFreeMembers before the device descriptor is built.
  uint32_t cacheVendorId = 0;
  uint32_t cacheDeviceId = 0;
  std::string cacheAdapterName;
  std::string cacheDriverDesc;
  {
    WGPUAdapterInfo info{};
    if (wgpuAdapterGetInfo(c->adapter, &info) == WGPUStatus_Success) {
      // WGPUStringView.data is NOT null-terminated; copy into std::string first.
      std::string devStr = (info.device.data && info.device.length)
          ? std::string(info.device.data, info.device.length) : "";
      std::string vendorStr = (info.vendor.data && info.vendor.length)
          ? std::string(info.vendor.data, info.vendor.length) : "";
      MGPU_LOG(mgpu::LOG_INFO,
        "[mgpu adapter] SELECTED: '%s' (%s) backendType=%d adapterType=%d "
        "vendorID=0x%X deviceID=0x%X "
        "(adapterType: 1=Discrete 2=Integrated 3=CPU 4=Unknown) "
        "Set MGPU_ADAPTER_NAME=<substring> to override",
        devStr.c_str(), vendorStr.c_str(),
        (int)info.backendType, (int)info.adapterType,
        (unsigned)info.vendorID, (unsigned)info.deviceID);
      set_selected_adapter_name(devStr);  // queryable via mgpuGetSelectedAdapterName
      selectedName = devStr;              // per-instance (multi-GPU contexts)
      cacheVendorId = (uint32_t)info.vendorID;
      cacheDeviceId = (uint32_t)info.deviceID;
      cacheAdapterName = devStr;
      // On D3D this is "D3D11 driver version 32.0.15.7688" — a driver update
      // therefore changes the shader cache key, which is exactly what must
      // happen (a blob compiled by another driver is not safe to reuse).
      cacheDriverDesc = (info.description.data && info.description.length)
          ? std::string(info.description.data, info.description.length) : "";
      wgpuAdapterInfoFreeMembers(info);
    }
  }

  // Request device - Updated API
  WGPUDeviceDescriptor deviceDesc = {};
  deviceDesc.nextInChain = nullptr;
  static const char kMainDeviceLabel[] = "MGPU.MainDevice";
  deviceDesc.label.data = kMainDeviceLabel;
  deviceDesc.label.length = sizeof(kMainDeviceLabel) - 1;

  // ── Persistent shader cache ────────────────────────────────────────────────
  // Dawn compiles WGSL to a backend shader at pipeline creation and caches the
  // result in its BlobCache — but only writes through to storage if the
  // embedder supplies these two callbacks. Without them every launch pays full
  // compilation again, which on the D3D11 backend means FXC, whose optimiser
  // cost grows superlinearly with kernel size.
  //
  // Both the descriptor and the isolation key must outlive
  // wgpuAdapterRequestDevice below; they are function-scope locals, and the
  // request is awaited before this function returns. The callbacks themselves
  // point at a never-destroyed global, because Dawn keeps calling them for the
  // life of the device — from its own threads.
#ifndef __EMSCRIPTEN__
  WGPUDawnCacheDeviceDescriptor cacheDesc = WGPU_DAWN_CACHE_DEVICE_DESCRIPTOR_INIT;
  std::string cacheIsolationKey;
  if (mgpu::shaderCacheIsActive()) {
    cacheIsolationKey = mgpu::shaderCacheIsolationKey(
        cacheVendorId, cacheDeviceId, cacheAdapterName + "|" + cacheDriverDesc);
    cacheDesc.isolationKey.data = cacheIsolationKey.data();
    cacheDesc.isolationKey.length = cacheIsolationKey.length();
    cacheDesc.loadDataFunction = &mgpu_shader_cache_dawn_load;
    cacheDesc.storeDataFunction = &mgpu_shader_cache_dawn_store;
    cacheDesc.functionUserdata = nullptr;  // global singleton, nothing to own
    // Chain-preserving even though nothing else chains here today.
    cacheDesc.chain.next = deviceDesc.nextInChain;
    deviceDesc.nextInChain = &cacheDesc.chain;
  }
#endif

  // ── Optionally enable platform-specific shared-texture-memory features so
  //    minigpu_external can import D3D11 NT handles, IOSurfaces, DMA-BUFs and
  //    AHardwareBuffers.  We probe the adapter and only enable what it
  //    actually supports — the device request would fail otherwise.
  std::vector<WGPUFeatureName> wantedFeatures;
#if !defined(__EMSCRIPTEN__)
  auto adapterHas = [&](WGPUFeatureName f) -> bool {
    bool has = wgpuAdapterHasFeature(c->adapter, f) != 0;
    MGPU_LOG(mgpu::LOG_DEBUG,
        "[mgpu adapter feature] %d => %d", (int)f, (int)has);
    return has;
  };
#ifdef _WIN32
  if (adapterHas(WGPUFeatureName_SharedTextureMemoryDXGISharedHandle))
    wantedFeatures.push_back(
        WGPUFeatureName_SharedTextureMemoryDXGISharedHandle);
  if (adapterHas(WGPUFeatureName_SharedTextureMemoryD3D11Texture2D))
    wantedFeatures.push_back(
        WGPUFeatureName_SharedTextureMemoryD3D11Texture2D);
  if (adapterHas(WGPUFeatureName_SharedTextureMemoryD3D12Resource))
    wantedFeatures.push_back(
        WGPUFeatureName_SharedTextureMemoryD3D12Resource);
  if (adapterHas(WGPUFeatureName_SharedFenceDXGISharedHandle))
    wantedFeatures.push_back(WGPUFeatureName_SharedFenceDXGISharedHandle);
  if (adapterHas(WGPUFeatureName_D3D11MultithreadProtected))
    wantedFeatures.push_back(WGPUFeatureName_D3D11MultithreadProtected);
  // Allow BGRA8Unorm storage textures so SharedOutputTexture can be
  // created in BGRA layout, matching what NVENC / AMF / QSV D3D11VA
  // hwframes pools require for zero-copy encoding.
  if (adapterHas(WGPUFeatureName_BGRA8UnormStorage))
    wantedFeatures.push_back(WGPUFeatureName_BGRA8UnormStorage);
#endif
  // Multi-planar YUV formats (NV12 / P010) — required to create per-plane
  // views on and sample imported hardware YUV textures (camera frames +
  // hardware video decoders like the Media Foundation D3D11 decoder). Without
  // it the imported R8BG8Biplanar420Unorm texture is invalid at bind time and
  // samples as black. Cross-platform (D3D11 NV12 on Windows, IOSurface NV12 on
  // Apple, dmabuf NV12 on Linux); adapter-gated so the device request never
  // fails where it's unsupported.
  if (adapterHas(WGPUFeatureName_DawnMultiPlanarFormats))
    wantedFeatures.push_back(WGPUFeatureName_DawnMultiPlanarFormats);
#ifdef __APPLE__
  if (adapterHas(WGPUFeatureName_SharedTextureMemoryIOSurface))
    wantedFeatures.push_back(WGPUFeatureName_SharedTextureMemoryIOSurface);
#endif
#if defined(__linux__) && !defined(__ANDROID__)
  if (adapterHas(WGPUFeatureName_SharedTextureMemoryDmaBuf))
    wantedFeatures.push_back(WGPUFeatureName_SharedTextureMemoryDmaBuf);
  if (adapterHas(WGPUFeatureName_SharedFenceSyncFD))
    wantedFeatures.push_back(WGPUFeatureName_SharedFenceSyncFD);
#endif
#ifdef __ANDROID__
  if (adapterHas(WGPUFeatureName_SharedTextureMemoryAHardwareBuffer))
    wantedFeatures.push_back(
        WGPUFeatureName_SharedTextureMemoryAHardwareBuffer);
#endif
#endif // !__EMSCRIPTEN__
  if (!wantedFeatures.empty()) {
    deviceDesc.requiredFeatures = wantedFeatures.data();
    deviceDesc.requiredFeatureCount = wantedFeatures.size();
  }

  // ── Request the adapter's FULL limits instead of the spec defaults.
  //    Defaults cap maxStorageBufferBindingSize at 128 MiB and maxBufferSize
  //    at 256 MiB; ML workloads (gpu_ml expert stacks, large weight
  //    matrices) bind multi-hundred-MB storage buffers, which previously
  //    failed CreateBindGroup with an uncaptured validation error and
  //    silently produced zero outputs.  Requesting exactly what the adapter
  //    reported cannot fail the device request.
  WGPULimits adapterLimits = {};
  bool haveAdapterLimits =
      wgpuAdapterGetLimits(c->adapter, &adapterLimits) ==
      WGPUStatus_Success;
  if (haveAdapterLimits) {
    adapterLimits.nextInChain = nullptr; // request base limits only
    deviceDesc.requiredLimits = &adapterLimits;
    MGPU_LOG(mgpu::LOG_INFO,
        "[mgpu limits] requesting adapter limits: maxBufferSize=%llu "
        "maxStorageBufferBindingSize=%llu maxComputeWorkgroupStorageSize=%u",
        (unsigned long long)adapterLimits.maxBufferSize,
        (unsigned long long)adapterLimits.maxStorageBufferBindingSize,
        (unsigned)adapterLimits.maxComputeWorkgroupStorageSize);
  } else {
    MGPU_LOG(mgpu::LOG_WARN,
        "[mgpu limits] wgpuAdapterGetLimits failed; using default limits "
        "(128MiB storage bindings)");
  }

  // Set up device lost callback info
  WGPUDeviceLostCallbackInfo deviceLostCallbackInfo = {};
  deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
  deviceLostCallbackInfo.callback =
      [](WGPUDevice const *device, WGPUDeviceLostReason reason,
         WGPUStringView message, void *userdata1, void *userdata2) {
        MGPU *mgpu = static_cast<MGPU *>(userdata1);
        const char *reasonStr = "Unknown";

        switch (reason) {
        case WGPUDeviceLostReason_Unknown:
          reasonStr = "Unknown";
          break;
        case WGPUDeviceLostReason_Destroyed:
          reasonStr = "Destroyed";
          break;
        case WGPUDeviceLostReason_FailedCreation:
          reasonStr = "Failed Creation";
          break;
        }

        LOG_ERROR("WebGPU Device Lost: %s - %.*s", reasonStr,
                  (int)message.length, message.data);
        MGPU_LOG(mgpu::LOG_ERROR,
            "[wgpu DEVICE LOST] reason=%s msg=%.*s",
            reasonStr, (int)message.length,
            message.data ? message.data : "");

        // Mark context as uninitialized so it can be recreated. Reads the
        // PUBLISHED context deliberately: this callback can fire long after
        // this function returned, and `c` (the private build-time pointer)
        // may by then belong to a Context that was replaced.
        if (mgpu->ctx) {
          mgpu->ctx->initialized = false;
        }
      };
  deviceLostCallbackInfo.userdata1 = this;

  // Add the device lost callback to the device descriptor
  deviceDesc.deviceLostCallbackInfo = deviceLostCallbackInfo;

  // Surface uncaptured validation/internal errors so we don't silently
  // produce zero output when a compute pass / bind group fails validation.
  WGPUUncapturedErrorCallbackInfo uncapErr = {};
  uncapErr.callback = [](WGPUDevice const*, WGPUErrorType type,
                          WGPUStringView message, void*, void*) {
    const char* t = "?";
    switch (type) {
      case WGPUErrorType_Validation: t = "Validation"; break;
      case WGPUErrorType_OutOfMemory: t = "OutOfMemory"; break;
      case WGPUErrorType_Internal: t = "Internal"; break;
      case WGPUErrorType_Unknown: t = "Unknown"; break;
      default: break;
    }
    MGPU_LOG(mgpu::LOG_ERROR,
        "[wgpu uncaptured %s] %.*s",
        t, (int)message.length, message.data ? message.data : "");
  };
  deviceDesc.uncapturedErrorCallbackInfo = uncapErr;

  struct DeviceRequestState {
    WGPUDevice device = nullptr;
    bool completed = false;
    mgpu::mutex mutex;
    std::condition_variable cv;
  };

  DeviceRequestState deviceState;

  // Updated callback structure for newer WebGPU API
  WGPURequestDeviceCallbackInfo deviceCallbackInfo = {};
  deviceCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
  deviceCallbackInfo.callback = [](WGPURequestDeviceStatus status,
                                   WGPUDevice device, WGPUStringView message,
                                   void *userdata1, void *userdata2) {
    DeviceRequestState *state = static_cast<DeviceRequestState *>(userdata1);
    mgpu::lock_guard<mgpu::mutex> lock(state->mutex);

    if (status == WGPURequestDeviceStatus_Success) {
      state->device = device;
    }
    state->completed = true;
    state->cv.notify_one();
  };
  deviceCallbackInfo.userdata1 = &deviceState;

  LOG_INFO("Requesting WebGPU device...");

  WGPUFuture deviceFuture =
      wgpuAdapterRequestDevice(c->adapter, &deviceDesc, deviceCallbackInfo);

  // Process events until the device request completes
  while (!deviceState.completed) {
    platformSleep(1, c->instance);
  }

  LOG_INFO("WebGPU device request completed");

  if (!deviceState.device) {
    wgpuAdapterRelease(c->adapter);
    wgpuInstanceRelease(c->instance);
    throw std::runtime_error("Failed to get WebGPU device");
  }

  c->device = deviceState.device;

  // Get the device queue
  c->queue = wgpuDeviceGetQueue(c->device);
  if (!c->queue) {
    wgpuDeviceRelease(c->device);
    wgpuAdapterRelease(c->adapter);
    wgpuInstanceRelease(c->instance);
    throw std::runtime_error("Failed to get WebGPU queue");
  }

  c->initialized = true;
  LOG_INFO("WebGPU context initialized successfully");
#ifndef __EMSCRIPTEN__
  // ONE line per device creation at INFO. Per-entry hit/miss detail is DEBUG:
  // this channel defaults to INFO, and a line per blob would bury real
  // warnings the way per-buffer logging once buried them.
  MGPU_LOG(mgpu::LOG_INFO, "%s", mgpu::shaderCacheSummaryLine().c_str());
#endif
#ifdef __EMSCRIPTEN__
  // Publish the WebGPU device so that JavaScript consumers (e.g. the view
  // package's canvas blit) can resolve the device without a separate init
  // call.
  //
  // 🔴 `globalThis`, NOT `window`. A dedicated Worker has no `window`, so this
  // threw there and took the rest of context init with it — an assignment
  // whose only purpose is convenience was enough to make the whole backend
  // main-thread-only. `globalThis` is the same object on a page and the right
  // one in a worker.
  EM_ASM({
      globalThis.gpuDevice = WebGPU.getJsObject($0);
  }, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(c->device)));
#endif
}

// The asynchronous init runs on a ONE-SHOT thread, deliberately NOT on
// webgpuThread. destroyContext() drains webgpuThread with enqueueSync(), so an
// init parked on that thread waiting for a concurrent teardown to finish would
// block the very drain it is waiting for. Dawn device creation was never bound
// to the WebGPU thread anyway — the lazy paths (getDevice / isDeviceValid)
// have always run it on whatever caller thread happened to touch a buffer.
void MGPU::initializeContextAsync(std::function<void()> callback) {
  auto body = [this, callback]() {
    try {
      initializeContext();
    } catch (...) {
      // Error occurred during initialization; the callback still fires so the
      // caller is not left awaiting forever.
    }
    if (callback) {
      callback();
    }
  };
#ifdef __EMSCRIPTEN__
  body(); // single-threaded: run inline, as enqueueAsync would have
#else
  std::thread(body).detach();
#endif
}

void MGPU::attachContext() {
  if (ctxLockBypassEnabled()) {
    initializeContext();
    return;
  }
  {
    mgpu::lock_guard<mgpu::mutex> lk(ctxStateMutex);
    ++ctxAttachCount;
  }
  try {
    initializeContext();
  } catch (...) {
    mgpu::lock_guard<mgpu::mutex> lk(ctxStateMutex);
    if (ctxAttachCount > 0) --ctxAttachCount;
    throw;
  }
}

void MGPU::attachContextAsync(std::function<void()> callback) {
  if (ctxLockBypassEnabled()) {
    initializeContextAsync(std::move(callback));
    return;
  }
  {
    mgpu::lock_guard<mgpu::mutex> lk(ctxStateMutex);
    ++ctxAttachCount;
  }
  auto body = [this, callback]() {
    bool ok = false;
    try {
      initializeContext();
      ok = ctx && ctx->initialized;
    } catch (...) {
      ok = false;
    }
    if (!ok) {
      // Give the reference back — an attach that never produced a device must
      // not keep the context pinned alive forever.
      mgpu::lock_guard<mgpu::mutex> lk(ctxStateMutex);
      if (ctxAttachCount > 0) --ctxAttachCount;
    }
    if (callback) callback();
  };
#ifdef __EMSCRIPTEN__
  body();
#else
  std::thread(body).detach();
#endif
}

void MGPU::detachContext() {
  if (ctxLockBypassEnabled()) {
    destroyContext();
    return;
  }
  {
    mgpu::lock_guard<mgpu::mutex> lk(ctxStateMutex);
    if (ctxAttachCount == 0) {
      // Never attached (or already fully detached): nothing of ours to tear
      // down. Matches the old no-op-when-!ctx behaviour for that case.
      return;
    }
    if (--ctxAttachCount > 0) {
      MGPU_LOG(mgpu::LOG_INFO,
          "[mgpu] destroyContext: %d consumer(s) still attached - keeping the "
          "process-global context alive.",
          ctxAttachCount);
      return;
    }
  }
  destroyContext();
}

int MGPU::attachCount() const {
  mgpu::lock_guard<mgpu::mutex> lk(ctxStateMutex);
  return ctxAttachCount;
}

void MGPU::destroyContext() {
  if (ctxLockBypassEnabled()) {
    destroyContextUnlocked();
    return;
  }

  const std::thread::id self = std::this_thread::get_id();
  {
    mgpu::unique_lock<mgpu::mutex> lk(ctxStateMutex);
    for (;;) {
      if (ctxState != CtxState::Initializing && ctxState != CtxState::Destroying)
        break;
      if (ctxOwner == self) return; // re-entrant; the owner will finish
      ctxStateCv.wait(lk);
    }
    if (ctxState != CtxState::Ready || !ctx) return;
    ctxState = CtxState::Destroying;
    ctxOwner = self;
  }

  // ── Teardown, NO lock held (it joins the WebGPU thread) ───────────────────
  struct Publish {
    MGPU *m;
    ~Publish() {
      mgpu::lock_guard<mgpu::mutex> lk(m->ctxStateMutex);
      m->ctxState = MGPU::CtxState::Uninitialized;
      m->ctxOwner = std::thread::id();
      // ctxAttachCount is deliberately NOT cleared: ensureDeviceValid()'s
      // device-loss recovery calls destroyContext() directly and then
      // re-initializes, and the consumers attached across that recovery are
      // still attached afterwards.
      m->ctxStateCv.notify_all();
    }
  } publish{this};

  destroyContextUnlocked();
}

void MGPU::destroyContextUnlocked() {
  if (!ctx) {
    return;
  }

  // Drain all pending WebGPU-thread tasks (buffer cleanups, dispatch completions)
  // before releasing the handles they reference, and discard any batched
  // work still recording.
  // Drop any upload scope the producer left open; its staged bytes describe a
  // device that is going away.
  if (uploadOpen) {
    UploadScope *open = uploadOpen.release();
    recycleScope(open); // releases the destination refs it holds
  }
  uploadDepth = 0;
  // Same for a readback scope: its destinations describe a device going away,
  // and its copies hold source refs that must be dropped.
  if (readbackOpen) {
    std::unique_ptr<ReadbackScope> open(readbackOpen.release());
    for (const ReadCopy &c : open->copies) {
      if (c.src) wgpuBufferRelease(c.src);
    }
  }
  readbackDepth = 0;
  webgpuThread.enqueueSync<void>([this]{
    mgpu::lock_guard<mgpu::mutex> lock(gpuOperationMutex);
    flushBatchLocked();
    releaseStagingLocked();
    releaseReadbackLocked();
  });

  if (ctx->queue) {
    wgpuQueueRelease(ctx->queue);
    ctx->queue = nullptr;
  }

  if (ctx->device) {
    wgpuDeviceRelease(ctx->device);
    ctx->device = nullptr;
  }

  if (ctx->adapter) {
    wgpuAdapterRelease(ctx->adapter);
    ctx->adapter = nullptr;
  }

  if (ctx->instance) {
    wgpuInstanceRelease(ctx->instance);
    ctx->instance = nullptr;
  }

#ifndef __EMSCRIPTEN__
  // Delete the dawn::native::Instance wrapper AFTER all WGPUInstance users
  // have released their references so the final refcount drop is orderly.
  if (ctx->dawnNativeInstance) {
    delete static_cast<dawn::native::Instance*>(ctx->dawnNativeInstance);
    ctx->dawnNativeInstance = nullptr;
  }
#endif

  ctx->initialized = false;
  ctx.reset();
  set_selected_adapter_name("");  // no adapter selected until next init

  // The external-texture layer caches an ID3D11Device (and its immediate
  // context) that on the D3D11 backend IS the device just released. Drop it
  // here or the next init hands out a dangling cache. (Skipped under the
  // positive-control bypass, which restores the pre-fix behaviour whole.)
  if (!ctxLockBypassEnabled()) {
    externalOnContextDestroyed(this);
  }
}

WGPUDevice MGPU::getDevice() const {
  if (!ctx || !ctx->initialized || !ctx->device) {
    LOG_WARN("Device invalid, attempting to reinitialize context");
    // Cast away const to allow reinitialization
    const_cast<MGPU *>(this)->initializeContext();
  }
  return ctx ? ctx->device : nullptr;
}

WGPUQueue MGPU::getQueue() const {
  if (!ctx || !ctx->initialized || !ctx->queue) {
    LOG_WARN("Queue invalid, attempting to reinitialize context");
    // Cast away const to allow reinitialization
    const_cast<MGPU *>(this)->initializeContext();
  }
  return ctx ? ctx->queue : nullptr;
}

WGPUInstance MGPU::getInstance() const {
  if (!ctx || !ctx->instance) {
    LOG_WARN("Instance invalid, attempting to reinitialize context");
    // Cast away const to allow reinitialization
    const_cast<MGPU *>(this)->initializeContext();
  }
  return ctx ? ctx->instance : nullptr;
}

bool MGPU::isDeviceValid() {
  if (!ctx || !ctx->initialized || !ctx->device || !ctx->queue) {
    LOG_WARN("Device/context invalid, attempting to reinitialize");
    initializeContext();
    return ctx && ctx->initialized && ctx->device && ctx->queue;
  }
  return true;
}

void MGPU::ensureDeviceValid() {
  if (!isDeviceValid()) {
    LOG_INFO("Device lost or uninitialized, attempting to recreate...");
    destroyContext();
    try {
      initializeContext();
      LOG_INFO("Device successfully recreated");
    } catch (const std::exception &e) {
      LOG_ERROR("Failed to recreate device: %s", e.what());
      throw;
    }
  }
}

Buffer::Buffer(MGPU &mgpu_ref) : mgpu(mgpu_ref) {}

Buffer::~Buffer() { release(); }

Buffer::Buffer(Buffer &&other) noexcept
    : mgpu(other.mgpu), bufferData(std::move(other.bufferData)),
      dataType(other.dataType), elementCount(other.elementCount),
      isPacked(other.isPacked),
      _readStagingBuffer(other._readStagingBuffer),
      _readStagingBufferSize(other._readStagingBufferSize) {
  other.bufferData.buffer = nullptr;
  other.bufferData.size = 0;
  other.elementCount = 0;
  other._readStagingBuffer = nullptr;
  other._readStagingBufferSize = 0;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (this != &other) {
    release();
    bufferData = std::move(other.bufferData);
    dataType = other.dataType;
    elementCount = other.elementCount;
    isPacked = other.isPacked;
    _readStagingBuffer = other._readStagingBuffer;
    _readStagingBufferSize = other._readStagingBufferSize;
    other.bufferData.buffer = nullptr;
    other.bufferData.size = 0;
    other.elementCount = 0;
    other._readStagingBuffer = nullptr;
    other._readStagingBufferSize = 0;
  }
  return *this;
}

void Buffer::createBuffer(size_t byteSize, BufferDataType dataType) {
  LOG_INFO("createBuffer called: byteSize=%zu, dataType=%d", byteSize,
           (int)dataType);

  try {

    // Validate device before proceeding
    WGPUDevice device = mgpu.getDevice();
    if (!device) {
      throw std::runtime_error("WebGPU device not initialized");
    }

    this->dataType = dataType;
    this->isPacked = needsPacking(dataType);
    this->elementCount = byteSize / getElementSize(dataType);

    LOG_INFO("isPacked=%s", isPacked ? "true" : "false");

    size_t physicalByteSize;

    if (isPacked) {
      // For PACKED types, byteSize is ELEMENT COUNT
      LOG_INFO("Packed type: elementCount=%zu", elementCount);

      switch (dataType) {
      case kInt8:
      case kUInt8:
        physicalByteSize = ((elementCount + 3) / 4) * 4;
        LOG_INFO("8-bit packed: physicalByteSize=%zu", physicalByteSize);
        break;
      case kInt16:
      case kUInt16:
        physicalByteSize = ((elementCount + 1) / 2) * 4;
        LOG_INFO("16-bit packed: physicalByteSize=%zu", physicalByteSize);
        break;
      case kInt64:
      case kUInt64:
      case kFloat64:
        physicalByteSize = elementCount * 8;
        LOG_INFO("64-bit packed: physicalByteSize=%zu", physicalByteSize);
        break;
      default:
        physicalByteSize = elementCount * getElementSize(dataType);
        LOG_INFO("Default packed: physicalByteSize=%zu", physicalByteSize);
        break;
      }
    } else {
      size_t elementSize = getElementSize(dataType);
      physicalByteSize = elementCount * elementSize;
      LOG_INFO("Direct type: elementCount=%zu, elementSize=%zu, "
               "physicalByteSize=%zu",
               elementCount, elementSize, physicalByteSize);
    }

    // Apply WebGPU requirements
    size_t originalSize = physicalByteSize;
    bufferData.size = std::max<size_t>(physicalByteSize, 4);
    bufferData.size = (bufferData.size + 3) & ~3;

    LOG_INFO("Buffer size: original=%zu, aligned=%zu", originalSize,
             bufferData.size);

    // Create the buffer
    bufferData.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                       WGPUBufferUsage_CopySrc;

    WGPUBufferDescriptor bufferDesc = {};
    bufferDesc.nextInChain = nullptr;
    bufferDesc.size = bufferData.size;
    bufferDesc.usage = bufferData.usage;
    bufferDesc.mappedAtCreation = false;

    try {
      bufferData.buffer = wgpuDeviceCreateBuffer(device, &bufferDesc);
      if (!bufferData.buffer) {
        throw std::runtime_error("Failed to create WebGPU buffer");
      }
      LOG_INFO("Buffer created successfully: buffer=%p, size=%zu",
               bufferData.buffer, bufferData.size);
    } catch (const std::exception &e) {
      LOG_ERROR("Exception creating buffer: %s", e.what());
      throw;
    } catch (...) {
      LOG_ERROR("Unknown exception creating buffer");
      throw std::runtime_error("Failed to create WebGPU buffer");
    }
  } catch (const std::exception &e) {
    LOG_ERROR("Exception in createBuffer: %s", e.what());
    throw;
  } catch (...) {
    LOG_ERROR("Unknown exception in createBuffer");
    throw std::runtime_error("Failed to create WebGPU buffer");
  }
}

void Buffer::release() {
  if (bufferData.buffer) {
    WGPUBuffer bufferHandle = bufferData.buffer;
    WGPUBuffer stagingHandle = _readStagingBuffer;
    WGPUInstance instance = mgpu.tryGetInstance();
    WGPUQueue queue = mgpu.tryGetQueue();
    mgpu::mutex &gpuMutex = mgpu.getGpuMutex();
    MGPU *owner = &mgpu;

    auto cleanupTask = [bufferHandle, stagingHandle, queue, instance, &gpuMutex,
                        owner]() {
      LOG_INFO("Releasing buffer on WebGPU thread: %p", bufferHandle);
      {
        mgpu::lock_guard<mgpu::mutex> lock(gpuMutex);
        // Pending batched dispatches may still reference this buffer.
        owner->flushBatchLocked();
        if (queue) {
          wgpuQueueSubmit(queue, 0, nullptr);
        }
        wgpuBufferDestroy(bufferHandle);
        wgpuBufferRelease(bufferHandle);
        if (stagingHandle) {
          wgpuBufferDestroy(stagingHandle);
          wgpuBufferRelease(stagingHandle);
        }
      }
      if (instance) {
        wgpuInstanceProcessEvents(instance);
      }
      LOG_INFO("Buffer released and events processed");
    };

    try {
      mgpu.getWebGPUThread().enqueueAsync(cleanupTask);
    } catch (...) {
      // GPU thread already shut down — release directly as a last resort.
      wgpuBufferDestroy(bufferHandle);
      wgpuBufferRelease(bufferHandle);
      if (stagingHandle) {
        wgpuBufferDestroy(stagingHandle);
        wgpuBufferRelease(stagingHandle);
      }
    }

    bufferData.buffer = nullptr;
    bufferData.size = 0;
    elementCount = 0;
    dataType = kUnknownType;
    isPacked = false;
    _readStagingBuffer = nullptr;
    _readStagingBufferSize = 0;
  }
}

void Buffer::releaseInternal() {
  if (bufferData.buffer) {
    try {
      LOG_INFO("Releasing buffer: %p", bufferData.buffer);
      wgpuBufferDestroy(bufferData.buffer);
      wgpuBufferRelease(bufferData.buffer);
      bufferData.buffer = nullptr;
      LOG_INFO("Buffer released successfully");
    } catch (const std::exception &e) {
      LOG_ERROR("Exception releasing buffer: %s", e.what());
      bufferData.buffer = nullptr;
    } catch (...) {
      LOG_ERROR("Unknown exception releasing buffer");
      bufferData.buffer = nullptr;
    }
  }

  bufferData.size = 0;
  bufferData.usage = WGPUBufferUsage_None;
  elementCount = 0;
  dataType = kUnknownType;
  isPacked = false;
}

size_t Buffer::getElementSize(BufferDataType type) const {
  switch (type) {
  case kFloat32:
  case kInt32:
  case kUInt32:
    return 4;
  case kFloat64:
  case kInt64:
  case kUInt64:
    return 8;
  case kInt16:
  case kUInt16:
    return 2;
  case kInt8:
  case kUInt8:
    return 1;
  default:
    return 4;
  }
}

bool Buffer::needsPacking(BufferDataType type) const {
  return type == kInt8 || type == kUInt8 || type == kInt16 || type == kUInt16 ||
         type == kInt64 || type == kUInt64 || type == kFloat64;
}

void Buffer::write(const float *inputData, size_t elementCount) {
  size_t byteSize = elementCount * sizeof(float);
  LOG_INFO("write(float): elementCount=%zu, byteSize=%zu", elementCount,
           byteSize);
  writeDirect(inputData, byteSize, kFloat32);
}

void Buffer::write(const int32_t *inputData, size_t elementCount) {
  size_t byteSize = elementCount * sizeof(int32_t);
  writeDirect(inputData, byteSize, kInt32);
}

void Buffer::write(const uint32_t *inputData, size_t elementCount) {
  size_t byteSize = elementCount * sizeof(uint32_t);
  writeDirect(inputData, byteSize, kUInt32);
}

void Buffer::write(const int8_t *inputData, size_t elementCount) {
  size_t byteSize = elementCount * sizeof(int8_t);
  if (needsPacking(kInt8)) {
    writePacked(inputData, byteSize, kInt8);
  } else {
    writeDirect(inputData, byteSize, kInt8);
  }
}

void Buffer::write(const uint8_t *inputData, size_t elementCount) {
  size_t byteSize = elementCount * sizeof(uint8_t);
  if (needsPacking(kUInt8)) {
    writePacked(inputData, byteSize, kUInt8);
  } else {
    writeDirect(inputData, byteSize, kUInt8);
  }
}

void Buffer::write(const int16_t *inputData, size_t elementCount) {
  size_t byteSize = elementCount * sizeof(int16_t);
  if (needsPacking(kInt16)) {
    writePacked(inputData, byteSize, kInt16);
  } else {
    writeDirect(inputData, byteSize, kInt16);
  }
}

void Buffer::write(const uint16_t *inputData, size_t elementCount) {
  size_t byteSize = elementCount * sizeof(uint16_t);
  if (needsPacking(kUInt16)) {
    writePacked(inputData, byteSize, kUInt16);
  } else {
    writeDirect(inputData, byteSize, kUInt16);
  }
}

void Buffer::write(const double *inputData, size_t elementCount) {
  size_t byteSize = elementCount * sizeof(double);
  if (needsPacking(kFloat64)) {
    writePacked(inputData, byteSize, kFloat64);
  } else {
    writeDirect(inputData, byteSize, kFloat64);
  }
}

void Buffer::write(const int64_t *inputData, size_t elementCount) {
  size_t byteSize = elementCount * sizeof(int64_t);
  if (needsPacking(kInt64)) {
    writePacked(inputData, byteSize, kInt64);
  } else {
    writeDirect(inputData, byteSize, kInt64);
  }
}

void Buffer::write(const uint64_t *inputData, size_t elementCount) {
  size_t byteSize = elementCount * sizeof(uint64_t);
  if (needsPacking(kUInt64)) {
    writePacked(inputData, byteSize, kUInt64);
  } else {
    writeDirect(inputData, byteSize, kUInt64);
  }
}

void Buffer::read(float *outputData, size_t elementCount, size_t offset) {
  readDirect(outputData, elementCount, offset);
}

void Buffer::read(int32_t *outputData, size_t elementCount, size_t offset) {
  readDirect(outputData, elementCount, offset);
}

void Buffer::read(uint32_t *outputData, size_t elementCount, size_t offset) {
  readDirect(outputData, elementCount, offset);
}

void Buffer::read(int8_t *outputData, size_t elementCount, size_t offset) {
  if (needsPacking(kInt8)) {
    readPacked(outputData, elementCount, offset);
  } else {
    readDirect(outputData, elementCount, offset);
  }
}

void Buffer::read(uint8_t *outputData, size_t elementCount, size_t offset) {
  if (needsPacking(kUInt8)) {
    readPacked(outputData, elementCount, offset);
  } else {
    readDirect(outputData, elementCount, offset);
  }
}

void Buffer::read(int16_t *outputData, size_t elementCount, size_t offset) {
  if (needsPacking(kInt16)) {
    readPacked(outputData, elementCount, offset);
  } else {
    readDirect(outputData, elementCount, offset);
  }
}

void Buffer::read(uint16_t *outputData, size_t elementCount, size_t offset) {
  if (needsPacking(kUInt16)) {
    readPacked(outputData, elementCount, offset);
  } else {
    readDirect(outputData, elementCount, offset);
  }
}

void Buffer::read(double *outputData, size_t elementCount, size_t offset) {
  if (needsPacking(kFloat64)) {
    readPacked(outputData, elementCount, offset);
  } else {
    readDirect(outputData, elementCount, offset);
  }
}

void Buffer::read(int64_t *outputData, size_t elementCount, size_t offset) {
  if (needsPacking(kInt64)) {
    readPacked(outputData, elementCount, offset);
  } else {
    readDirect(outputData, elementCount, offset);
  }
}

void Buffer::read(uint64_t *outputData, size_t elementCount, size_t offset) {
  if (needsPacking(kUInt64)) {
    readPacked(outputData, elementCount, offset);
  } else {
    readDirect(outputData, elementCount, offset);
  }
}

void Buffer::readAsync(float *outputData, size_t elementCount, size_t offset,
                       std::function<void()> callback) {
  readAsyncImpl(outputData, elementCount, offset, kFloat32, callback);
}

void Buffer::readAsync(int32_t *outputData, size_t elementCount, size_t offset,
                       std::function<void()> callback) {
  readAsyncImpl(outputData, elementCount, offset, kInt32, callback);
}

void Buffer::readAsync(uint32_t *outputData, size_t elementCount, size_t offset,
                       std::function<void()> callback) {
  readAsyncImpl(outputData, elementCount, offset, kUInt32, callback);
}

void Buffer::readAsync(int8_t *outputData, size_t elementCount, size_t offset,
                       std::function<void()> callback) {
  readAsyncImpl(outputData, elementCount, offset, kInt8, callback);
}

void Buffer::readAsync(uint8_t *outputData, size_t elementCount, size_t offset,
                       std::function<void()> callback) {
  readAsyncImpl(outputData, elementCount, offset, kUInt8, callback);
}

void Buffer::readAsync(int16_t *outputData, size_t elementCount, size_t offset,
                       std::function<void()> callback) {
  readAsyncImpl(outputData, elementCount, offset, kInt16, callback);
}

void Buffer::readAsync(uint16_t *outputData, size_t elementCount, size_t offset,
                       std::function<void()> callback) {
  readAsyncImpl(outputData, elementCount, offset, kUInt16, callback);
}

void Buffer::readAsync(double *outputData, size_t elementCount, size_t offset,
                       std::function<void()> callback) {
  readAsyncImpl(outputData, elementCount, offset, kFloat64, callback);
}

void Buffer::readAsync(int64_t *outputData, size_t elementCount, size_t offset,
                       std::function<void()> callback) {
  readAsyncImpl(outputData, elementCount, offset, kInt64, callback);
}

void Buffer::readAsync(uint64_t *outputData, size_t elementCount, size_t offset,
                       std::function<void()> callback) {
  readAsyncImpl(outputData, elementCount, offset, kUInt64, callback);
}

void Buffer::readBytesAt(void *outputData, size_t byteSize,
                         size_t srcByteOffset) {
  if (!outputData || byteSize == 0) return;
  // readDirect<uint8_t> is a RAW byte read: elementSize == 1, so elementCount
  // is the byte count and offset is the byte offset.  No per-type unpacking —
  // identical bytes to what the batched path copies out of the mapped range.
  readDirect<uint8_t>(static_cast<uint8_t *>(outputData), byteSize,
                      srcByteOffset);
}

void Buffer::writeBytesAt(const void *inputData, size_t byteSize,
                          size_t dstByteOffset) {
  if (!inputData || byteSize == 0) {
    return;
  }
  if (!mgpu.isDeviceValid()) {
    LOG_ERROR("MGPU context is not valid, cannot perform buffer operation");
    throw std::runtime_error("WebGPU context not valid for buffer operation");
  }
  if (dstByteOffset + byteSize > bufferData.size) {
    throw std::runtime_error(
        "writeBytesAt out of range: offset=" + std::to_string(dstByteOffset) +
        " size=" + std::to_string(byteSize) +
        " buffer=" + std::to_string(bufferData.size));
  }
  WaitScope wpTotal(WP_WR_TOTAL);
  // 4-byte alignment is a WebGPU requirement for queue writes.
  mgpu::unique_lock<mgpu::mutex> lock = [&] {
    WaitScope wp(WP_WR_LOCK);
    return mgpu::unique_lock<mgpu::mutex>(mgpu.getGpuMutex());
  }();
  // Queue writes are ordered by SUBMISSION time — flush pending batched
  // dispatches first so they execute before this write, not after.
  {
    WaitScope wp(WP_WR_FLUSH);
    mgpu.flushBatchLocked();
  }
  WGPUQueue queue = mgpu.getQueue();
  if (!queue || !bufferData.buffer) {
    LOG_ERROR("Invalid WebGPU handles for writeBytesAt");
    throw std::runtime_error("WebGPU handles not valid for buffer operation");
  }
  {
    WaitScope wp(WP_WR_WRITE);
    wgpuQueueWriteBuffer(queue, bufferData.buffer, dstByteOffset, inputData,
                         byteSize);
  }
}

template <typename T>
void Buffer::writeDirect(const T *inputData, size_t byteSize,
                         BufferDataType type) {
  if (!inputData || byteSize == 0) {
    return;
  }

  if (!mgpu.isDeviceValid()) {
    LOG_ERROR("MGPU context is not valid, cannot perform buffer operation");
    throw std::runtime_error("WebGPU context not valid for buffer operation");
  }

  if (bufferData.size < byteSize) {
    throw std::runtime_error(
        "Buffer size mismatch: allocated=" + std::to_string(bufferData.size) +
        " bytes, trying to upload=" + std::to_string(byteSize) + " bytes");
  }

  LOG_INFO("setDataDirect: byteSize=%zu, bufferSize=%zu, inputData=%p",
           byteSize, bufferData.size, inputData);

  // Acquire WebGPU operation lock to prevent conflicts with compute dispatch
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
  // Queue writes are ordered by SUBMISSION time — flush pending batched
  // dispatches first so they execute before this write, not after.
  mgpu.flushBatchLocked();

  WGPUDevice device = mgpu.getDevice();
  WGPUQueue queue = mgpu.getQueue();

  if (!device || !queue || !bufferData.buffer) {
    LOG_ERROR("Invalid WebGPU handles: device=%p, queue=%p, buffer=%p", device,
              queue, bufferData.buffer);
    throw std::runtime_error("WebGPU handles not valid for buffer operation");
  }

  // Write the buffer data - protected by mutex from compute operations
  wgpuQueueWriteBuffer(queue, bufferData.buffer, 0, inputData, byteSize);
  LOG_INFO("setDataDirect completed successfully");
}

template <typename T>
void Buffer::writePacked(const T *inputData, size_t byteSize,
                         BufferDataType type) {
  if (!inputData || byteSize == 0) {
    return;
  }

  size_t numElements = byteSize / sizeof(T);
  std::vector<uint32_t> packedData;

  switch (type) {
  case kInt8:
  case kUInt8: {
    // Pack 4 bytes into each uint32 - size should match createBuffer
    // calculation
    size_t packedElements = (numElements + 3) / 4;
    packedData.resize(packedElements, 0);

    for (size_t i = 0; i < numElements; ++i) {
      size_t packedIndex = i / 4;
      size_t byteOffset = i % 4;
      uint32_t value =
          static_cast<uint32_t>(static_cast<uint8_t>(inputData[i]));
      packedData[packedIndex] |= (value << (byteOffset * 8));
    }
    break;
  }

  case kInt16:
  case kUInt16: {
    // Pack 2 shorts into each uint32 - size should match createBuffer
    // calculation
    size_t packedElements = (numElements + 1) / 2;
    packedData.resize(packedElements, 0);

    for (size_t i = 0; i < numElements; ++i) {
      size_t packedIndex = i / 2;
      size_t shortOffset = i % 2;
      uint32_t value =
          static_cast<uint32_t>(static_cast<uint16_t>(inputData[i]));
      packedData[packedIndex] |= (value << (shortOffset * 16));
    }
    break;
  }

  case kInt64:
  case kUInt64:
  case kFloat64: {
    // Convert 64-bit values to pairs of uint32
    packedData.resize(numElements * 2);

    for (size_t i = 0; i < numElements; ++i) {
      uint64_t value;
      if (type == kFloat64) {
        double doubleVal = static_cast<double>(inputData[i]);
        std::memcpy(&value, &doubleVal, sizeof(uint64_t));
      } else {
        value = static_cast<uint64_t>(inputData[i]);
      }

      packedData[i * 2] = static_cast<uint32_t>(value & 0xFFFFFFFF);
      packedData[i * 2 + 1] = static_cast<uint32_t>(value >> 32);
    }
    break;
  }

  default:
    writeDirect(inputData, byteSize, type);
    return;
  }

  // Upload packed data - this size should now match what was allocated
  size_t packedByteSize = packedData.size() * sizeof(uint32_t);
  writeDirect(packedData.data(), packedByteSize, kUInt32);
}

template <typename T>
void Buffer::readDirect(T *outputData, size_t elementCount, size_t offset) {
  LOG_INFO("readDirect: elementCount=%zu, offset=%zu, sizeof(T)=%zu",
           elementCount, offset, sizeof(T));

  const size_t elementSize = sizeof(T); // Use actual template type size, not logical dataType size
  size_t byteOffset = offset * elementSize;
  size_t readBytes = elementCount * elementSize;

  LOG_INFO("readDirect: byteOffset=%zu, readBytes=%zu, bufferSize=%zu",
           byteOffset, readBytes, bufferData.size);

  if (byteOffset + readBytes > bufferData.size) {
    LOG_ERROR("Read would exceed buffer bounds: %zu + %zu > %zu", byteOffset,
              readBytes, bufferData.size);
    return;
  }

  if (!mgpu.isDeviceValid()) {
    LOG_ERROR("MGPU context is not valid, cannot perform buffer read");
    throw std::runtime_error("WebGPU context not valid for buffer read");
  }

  WaitScope wpTotal(WP_RD_TOTAL);

  // Acquire WebGPU operation lock to prevent conflicts with compute dispatch.
  // We use unique_lock (vs lock_guard) so we can unlock it explicitly before
  // calling wgpuInstanceProcessEvents at the end, which must run outside the
  // lock to avoid blocking buffer-release cleanup tasks that also acquire it.
  mgpu::unique_lock<mgpu::mutex> lock =
      [&] {
        WaitScope wp(WP_RD_LOCK);
        return mgpu::unique_lock<mgpu::mutex>(mgpu.getGpuMutex());
      }();
  // Readbacks copy via their own encoder — flush pending batched dispatches
  // first so their results are visible to this read.
  {
    WaitScope wp(WP_RD_FLUSH);
    mgpu.flushBatchLocked();
  }

  WGPUDevice device = mgpu.getDevice();
  WGPUQueue queue = mgpu.getQueue();

  if (!device || !queue || !bufferData.buffer) {
    LOG_ERROR("Invalid WebGPU handles for read: device=%p, queue=%p, buffer=%p",
              device, queue, bufferData.buffer);
    throw std::runtime_error("WebGPU handles not valid for buffer read");
  }

  // Snapshot the handle while we hold the GPU mutex.  A concurrent
  // Buffer::release() on the Dart thread nulls bufferData.buffer without
  // the mutex (it only enqueues the actual WGPUBuffer destruction on the
  // WebGPU thread).  Using a local copy avoids a TOCTOU race where
  // bufferData.buffer becomes nullptr between the check above and the
  // wgpuCommandEncoderCopyBufferToBuffer call below.
  // Safety: the cleanup task enqueued by release() is always ordered *after*
  // this readDirect task on the WebGPU thread, so localBuffer stays alive
  // for the full duration of this function.
  const WGPUBuffer localBuffer = bufferData.buffer;

  // Reuse the persistent staging buffer when it already has the right size.
  // This eliminates the ~100 MB/s of GPU heap churn that occurs when reading
  // a 5 MB texture tensor at 20 fps (one 5 MB alloc+free per frame).
  // ── Staging buffer: GROW-ON-DEMAND, never resized to fit ──────────────────
  // This used to keep a staging buffer of EXACTLY readBytes and destroy +
  // recreate it whenever the next read asked for a different length. That is
  // fine for a fixed-size tensor and pathological for the case this path
  // actually serves: a compacted, rate-controlled payload whose length changes
  // every single frame. Every read then paid a GPU buffer destroy + create,
  // and the trailing wgpuInstanceProcessEvents below existed only to stop the
  // orphans piling up. It is the same shape as the raw-upload scratch that used
  // to malloc per call — a per-frame allocate/free cycle wrapped around a
  // transfer, costing more than the transfer.
  //
  // Capacity only ever grows, to the next power of two, and is bounded by the
  // source buffer's own size (a read cannot exceed it), so the worst case is
  // one staging copy of a buffer that is being fully read back — which is what
  // an exact-fit buffer cost anyway. Released with the buffer.
  {
    WaitScope wpAlloc(WP_RD_ALLOC);
    if (_readStagingBufferSize < readBytes) {
      if (_readStagingBuffer) {
        wgpuBufferDestroy(_readStagingBuffer);
        wgpuBufferRelease(_readStagingBuffer);
        _readStagingBuffer = nullptr;
        _readStagingBufferSize = 0;
      }
      size_t cap = 65536;
      while (cap < readBytes && cap < ((size_t)1 << 31)) cap <<= 1;
      if (cap < readBytes) cap = readBytes;
      if (cap > bufferData.size) cap = bufferData.size;
      if (cap < readBytes) cap = readBytes;
      // MapRead buffers must be a multiple of 4; the doubling keeps that, the
      // clamp to bufferData.size might not.
      cap = (cap + 3) & ~(size_t)3;

      WGPUBufferDescriptor stagingDesc = {};
      stagingDesc.size = cap;
      stagingDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
      stagingDesc.mappedAtCreation = false;

      _readStagingBuffer = wgpuDeviceCreateBuffer(device, &stagingDesc);
      if (!_readStagingBuffer) {
        LOG_ERROR("Failed to create staging buffer");
        throw std::runtime_error("Failed to create staging buffer");
      }
      _readStagingBufferSize = cap;
    }
  }
  WGPUBuffer stagingBuffer = _readStagingBuffer;

  // Copy from our buffer to staging buffer - protected by mutex
  const long long wpSubmit0 = waitProfEnabled() ? waitProfNowUs() : 0;
  WGPUCommandEncoderDescriptor encoderDesc = {};
  WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(device, &encoderDesc);
  if (!encoder) {
    LOG_ERROR("Failed to create command encoder");
    wgpuBufferRelease(stagingBuffer);
    throw std::runtime_error("Failed to create command encoder");
  }

  wgpuCommandEncoderCopyBufferToBuffer(encoder, localBuffer, byteOffset,
                                       stagingBuffer, 0, readBytes);

  WGPUCommandBufferDescriptor cmdBufferDesc = {};
  WGPUCommandBuffer commands =
      wgpuCommandEncoderFinish(encoder, &cmdBufferDesc);
  if (!commands) {
    LOG_ERROR("Failed to finish command encoder");
    wgpuCommandEncoderRelease(encoder);
    wgpuBufferRelease(stagingBuffer);
    throw std::runtime_error("Failed to finish command encoder");
  }

  wgpuQueueSubmit(queue, 1, &commands);
  if (waitProfEnabled())
    waitProfNote(WP_RD_SUBMIT, waitProfNowUs() - wpSubmit0, 0);

  // Map and read the staging buffer synchronously
  struct ReadState {
    bool completed = false;
    WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Success;
    mgpu::mutex mutex;
#ifndef __EMSCRIPTEN__
    std::condition_variable cv;
#endif
  };

  ReadState readState;

  WGPUBufferMapCallbackInfo mapCallbackInfo = {};
  mapCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
  mapCallbackInfo.callback = [](WGPUMapAsyncStatus status,
                                WGPUStringView message, void *userdata1,
                                void *userdata2) {
    ReadState *state = static_cast<ReadState *>(userdata1);
    mgpu::lock_guard<mgpu::mutex> lock(state->mutex);
    state->status = status;
    state->completed = true;

#ifndef __EMSCRIPTEN__
    state->cv.notify_one();
#endif
  };
  mapCallbackInfo.userdata1 = &readState;

  // mapAsync wants a 4-byte multiple. readBytes always was one when the
  // staging buffer was created to fit it exactly (a MapRead buffer must be a
  // multiple of 4 too); now that the buffer can be larger than the request,
  // round the MAPPED range up rather than let an odd raw-byte read start
  // failing validation. Only readBytes are ever copied out.
  size_t mapBytes = (readBytes + 3) & ~(size_t)3;
  if (mapBytes > _readStagingBufferSize) mapBytes = _readStagingBufferSize;
  auto future = wgpuBufferMapAsync(stagingBuffer, WGPUMapMode_Read, 0, mapBytes,
                                   mapCallbackInfo);

  WGPUInstance instance = mgpu.getInstance();

  // Wait for the map operation
  {
    WaitScope wpMap(WP_RD_MAP);
    WGPUFutureWaitInfo waitInfo = {};
    waitInfo.future = future;
    wgpuInstanceWaitAny(instance, 1, &waitInfo, 0);
    if (!waitInfo.completed) {
      // Hot-poll: Sleep(1) on Windows actually sleeps ~15ms under the default
      // timer resolution which makes per-frame read-back dominate encoder
      // throughput. Use ms=0 so platformSleep just calls
      // wgpuInstanceProcessEvents + Sleep(0) (yield only). The GPU side
      // typically completes within microseconds for small read-backs.
      while (!readState.completed) {
        platformSleep(0, instance);
        ++wpMap.iters;
      }
    }
  }

  const long long wpCopy0 = waitProfEnabled() ? waitProfNowUs() : 0;
  if (readState.status == WGPUMapAsyncStatus_Success) {
    const void *mappedData =
        wgpuBufferGetConstMappedRange(stagingBuffer, 0, mapBytes);
    if (mappedData) {
      std::memcpy(outputData, mappedData, readBytes);
      LOG_INFO("Successfully read %zu bytes", readBytes);
    } else {
      LOG_ERROR("Failed to get mapped range");
    }
  } else {
    LOG_ERROR("Buffer mapping failed with status %d", (int)readState.status);
  }
  wgpuBufferUnmap(stagingBuffer);
  if (waitProfEnabled()) waitProfNote(WP_RD_COPY, waitProfNowUs() - wpCopy0, 0);

  // Keep the staging buffer alive for reuse on the next readDirect call.
  // It will be released in Buffer::release() when the buffer is destroyed.
  wgpuCommandBufferRelease(commands);
  wgpuCommandEncoderRelease(encoder);

  // Release the GPU mutex before pumping events.  wgpuInstanceProcessEvents
  // may deliver Dawn-internal callbacks (e.g. from prior buffer releases) that
  // themselves need the GPU mutex; holding it here would deadlock.
  lock.unlock();

  // Pump Dawn's event loop so pending deferred deallocations and any callbacks
  // chained after our map can run.
  //
  // This used to exist because the staging buffer was destroyed and recreated
  // whenever the read length changed, so without a pump the orphans piled up a
  // frame at a time. The staging buffer is persistent now, so that reason is
  // gone; the pump is kept for the OTHER deferred work Dawn has queued, and it
  // is cheap (measured at 1.5 us per read). It must stay OUTSIDE the lock: a
  // callback delivered here may itself need the GPU mutex.
  {
    WaitScope wpEv(WP_RD_EVENTS);
    wgpuInstanceProcessEvents(instance);
  }
}
template <typename T>
void Buffer::readPacked(T *outputData, size_t elementCount, size_t offset) {
  if (!outputData || elementCount == 0) {
    return;
  }

  switch (dataType) {
  case kInt8:
  case kUInt8: {
    // Read packed uint32 data
    size_t packedElements = (elementCount + 3) / 4;
    std::vector<uint32_t> packedData(packedElements);

    readDirect(packedData.data(), packedElements, offset / 4);

    // Unpack the data
    for (size_t i = 0; i < elementCount; ++i) {
      size_t packedIndex = i / 4;
      size_t byteOffset = i % 4;
      uint32_t packedValue = packedData[packedIndex];
      uint8_t value =
          static_cast<uint8_t>((packedValue >> (byteOffset * 8)) & 0xFF);
      outputData[i] = static_cast<T>(value);
    }
    break;
  }

  case kInt16:
  case kUInt16: {
    // Read packed uint32 data
    size_t packedElements = (elementCount + 1) / 2;
    std::vector<uint32_t> packedData(packedElements);

    readDirect(packedData.data(), packedElements, offset / 2);

    // Unpack the data
    for (size_t i = 0; i < elementCount; ++i) {
      size_t packedIndex = i / 2;
      size_t shortOffset = i % 2;
      uint32_t packedValue = packedData[packedIndex];
      uint16_t value =
          static_cast<uint16_t>((packedValue >> (shortOffset * 16)) & 0xFFFF);
      outputData[i] = static_cast<T>(value);
    }
    break;
  }

  case kInt64:
  case kUInt64:
  case kFloat64: {
    // Read pairs of uint32 values
    std::vector<uint32_t> packedData(elementCount * 2);
    readDirect(packedData.data(), elementCount * 2, offset * 2);

    // Unpack the data
    for (size_t i = 0; i < elementCount; ++i) {
      uint64_t value = static_cast<uint64_t>(packedData[i * 2]) |
                       (static_cast<uint64_t>(packedData[i * 2 + 1]) << 32);

      if (dataType == kFloat64) {
        double doubleVal;
        std::memcpy(&doubleVal, &value, sizeof(double));
        outputData[i] = static_cast<T>(doubleVal);
      } else {
        outputData[i] = static_cast<T>(value);
      }
    }
    break;
  }

  default:
    // Fallback to direct read
    readDirect(outputData, elementCount, offset);
    break;
  }
}

template <typename T>
void Buffer::readAsyncImpl(T *outputData, size_t elementCount, size_t offset,
                           BufferDataType type,
                           std::function<void()> callback) {
  if (!outputData || elementCount == 0) {
    if (callback)
      callback();
    return;
  }

  mgpu.getWebGPUThread().enqueueAsync(
      [this, outputData, elementCount, offset, type, callback]() {
        try {
          if (needsPacking(type)) {
            readPacked(outputData, elementCount, offset);
          } else {
            readDirect(outputData, elementCount, offset);
          }
          if (callback)
            callback();
        } catch (...) {
          if (callback)
            callback();
        }
      });
}

template void Buffer::writeDirect<int8_t>(const int8_t *, size_t,
                                          BufferDataType);
template void Buffer::writeDirect<uint8_t>(const uint8_t *, size_t,
                                           BufferDataType);
template void Buffer::writeDirect<int16_t>(const int16_t *, size_t,
                                           BufferDataType);
template void Buffer::writeDirect<uint16_t>(const uint16_t *, size_t,
                                            BufferDataType);
template void Buffer::writeDirect<int32_t>(const int32_t *, size_t,
                                           BufferDataType);
template void Buffer::writeDirect<uint32_t>(const uint32_t *, size_t,
                                            BufferDataType);
template void Buffer::writeDirect<int64_t>(const int64_t *, size_t,
                                           BufferDataType);
template void Buffer::writeDirect<uint64_t>(const uint64_t *, size_t,
                                            BufferDataType);
template void Buffer::writeDirect<float>(const float *, size_t, BufferDataType);
template void Buffer::writeDirect<double>(const double *, size_t,
                                          BufferDataType);

template void Buffer::writePacked<int8_t>(const int8_t *, size_t,
                                          BufferDataType);
template void Buffer::writePacked<uint8_t>(const uint8_t *, size_t,
                                           BufferDataType);
template void Buffer::writePacked<int16_t>(const int16_t *, size_t,
                                           BufferDataType);
template void Buffer::writePacked<uint16_t>(const uint16_t *, size_t,
                                            BufferDataType);
template void Buffer::writePacked<int64_t>(const int64_t *, size_t,
                                           BufferDataType);
template void Buffer::writePacked<uint64_t>(const uint64_t *, size_t,
                                            BufferDataType);
template void Buffer::writePacked<double>(const double *, size_t,
                                          BufferDataType);

template void Buffer::readDirect<int8_t>(int8_t *, size_t, size_t);
template void Buffer::readDirect<uint8_t>(uint8_t *, size_t, size_t);
template void Buffer::readDirect<int16_t>(int16_t *, size_t, size_t);
template void Buffer::readDirect<uint16_t>(uint16_t *, size_t, size_t);
template void Buffer::readDirect<int32_t>(int32_t *, size_t, size_t);
template void Buffer::readDirect<uint32_t>(uint32_t *, size_t, size_t);
template void Buffer::readDirect<int64_t>(int64_t *, size_t, size_t);
template void Buffer::readDirect<uint64_t>(uint64_t *, size_t, size_t);
template void Buffer::readDirect<float>(float *, size_t, size_t);
template void Buffer::readDirect<double>(double *, size_t, size_t);

template void Buffer::readPacked<int8_t>(int8_t *, size_t, size_t);
template void Buffer::readPacked<uint8_t>(uint8_t *, size_t, size_t);
template void Buffer::readPacked<int16_t>(int16_t *, size_t, size_t);
template void Buffer::readPacked<uint16_t>(uint16_t *, size_t, size_t);
template void Buffer::readPacked<int64_t>(int64_t *, size_t, size_t);
template void Buffer::readPacked<uint64_t>(uint64_t *, size_t, size_t);
template void Buffer::readPacked<double>(double *, size_t, size_t);

template void Buffer::readAsyncImpl<int8_t>(int8_t *, size_t, size_t,
                                            BufferDataType,
                                            std::function<void()>);
template void Buffer::readAsyncImpl<uint8_t>(uint8_t *, size_t, size_t,
                                             BufferDataType,
                                             std::function<void()>);
template void Buffer::readAsyncImpl<int16_t>(int16_t *, size_t, size_t,
                                             BufferDataType,
                                             std::function<void()>);
template void Buffer::readAsyncImpl<uint16_t>(uint16_t *, size_t, size_t,
                                              BufferDataType,
                                              std::function<void()>);
template void Buffer::readAsyncImpl<int32_t>(int32_t *, size_t, size_t,
                                             BufferDataType,
                                             std::function<void()>);
template void Buffer::readAsyncImpl<uint32_t>(uint32_t *, size_t, size_t,
                                              BufferDataType,
                                              std::function<void()>);
template void Buffer::readAsyncImpl<int64_t>(int64_t *, size_t, size_t,
                                             BufferDataType,
                                             std::function<void()>);
template void Buffer::readAsyncImpl<uint64_t>(uint64_t *, size_t, size_t,
                                              BufferDataType,
                                              std::function<void()>);
template void Buffer::readAsyncImpl<float>(float *, size_t, size_t,
                                           BufferDataType,
                                           std::function<void()>);
template void Buffer::readAsyncImpl<double>(double *, size_t, size_t,
                                            BufferDataType,
                                            std::function<void()>);

} // namespace mgpu