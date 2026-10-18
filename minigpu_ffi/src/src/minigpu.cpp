#include "../include/minigpu.h"
#include "../include/minigpu_context_owner.h"
#include "../include/minigpu_capture_setup_bridge.h"
#include "../include/minigpu_capture_cleanup_adapter.h"
#include "../include/minigpu_capture_work_owner.h"
#include "../include/minigpu_imported_frame_owner.h"
#include "../include/dart_port.h"
#include "../include/log.h"
#include "../include/platform_sleep.h"
#include <atomic>
#include <chrono>
#include <exception>
#include <new>
#include <thread>
#ifdef __EMSCRIPTEN__
#include <unordered_map>
#endif
#ifdef MINIGPU_HAVE_DART_DL
// Vendored Dart SDK dynamic-linking API (third_party/dart_dl).
#include "dart_api_dl.h"
#endif
#ifdef _WIN32
#include <dxgi1_4.h>
#pragma comment(lib, "dxgi.lib")
#endif

// --- Dart native-port log delivery ------------------------------------------
//
// mgpu's log registry is PROCESS-GLOBAL. A Dart `NativeCallable` installed
// through mgpuSetLogCallback belongs to exactly ONE isolate; when that isolate
// exits the VM deletes the trampoline while this library still holds the
// pointer, and the next Dawn worker thread that logs runs it on a thread with
// no isolate group at all -> "Callback invoked after it has been deleted" ->
// the whole process aborts. Posting to a Dart port is defined, silent and
// thread-safe even after the port has closed, so the port is the supported
// Dart delivery path. The function-pointer API stays for non-Dart embedders.
namespace mgpu {

#ifdef MINIGPU_HAVE_DART_DL
// Atomic: any thread that logs reads it while an isolate sets or clears it.
static std::atomic<Dart_Port> g_log_port{ILLEGAL_PORT};
static bool g_dart_api_ready = false;
#endif

bool logPostToPort(int level, const char *message, size_t len) {
#ifdef MINIGPU_HAVE_DART_DL
  const Dart_Port port = g_log_port.load(std::memory_order_acquire);
  if (port == ILLEGAL_PORT || !g_dart_api_ready || message == nullptr)
    return false;

  // Bytes rather than Dart_CObject_kString: Dawn/driver strings can carry
  // non-UTF-8 (Latin-1) sequences and kString demands valid UTF-8.
  // Dart_PostCObject_DL COPIES the typed data, so nothing crosses ownership
  // and there is nothing for Dart to free.
  Dart_CObject c_level;
  c_level.type = Dart_CObject_kInt32;
  c_level.value.as_int32 = static_cast<int32_t>(level);

  Dart_CObject c_msg;
  c_msg.type = Dart_CObject_kTypedData;
  c_msg.value.as_typed_data.type = Dart_TypedData_kUint8;
  c_msg.value.as_typed_data.length = static_cast<intptr_t>(len);
  c_msg.value.as_typed_data.values =
      reinterpret_cast<uint8_t *>(const_cast<char *>(message));

  Dart_CObject *parts[2] = {&c_level, &c_msg};

  Dart_CObject payload;
  payload.type = Dart_CObject_kArray;
  payload.value.as_array.length = 2;
  payload.value.as_array.values = parts;

  // A false return just means the port is gone (its isolate exited). That is
  // the entire reason this path exists — swallow it, and keep reporting
  // "delivered" so a dead registration silences output instead of falling
  // back to a stale function pointer.
  (void)Dart_PostCObject_DL(port, &payload);
  return true;
#else
  (void)level;
  (void)message;
  (void)len;
  return false;
#endif
}

intptr_t logInitDartApi(void *initialize_api_dl_data) {
#ifdef MINIGPU_HAVE_DART_DL
  // Idempotent: Dart_InitializeApiDL re-populates the same function table.
  const intptr_t rc = Dart_InitializeApiDL(initialize_api_dl_data);
  if (rc == 0)
    g_dart_api_ready = true;
  return rc;
#else
  (void)initialize_api_dl_data;
  return -1;
#endif
}

void logSetPort(int64_t port) {
#ifdef MINIGPU_HAVE_DART_DL
  g_log_port.store(static_cast<Dart_Port>(port), std::memory_order_release);
#else
  (void)port;
#endif
}

bool logClearPortIf(int64_t port) {
#ifdef MINIGPU_HAVE_DART_DL
  // An isolate that stops logging clears only its own port, never a newer
  // isolate's registration (which a plain set to 0 would silence).
  Dart_Port expected = static_cast<Dart_Port>(port);
  return expected != ILLEGAL_PORT &&
         g_log_port.compare_exchange_strong(expected, ILLEGAL_PORT,
                                            std::memory_order_acq_rel);
#else
  (void)port;
  return false;
#endif
}

// --- Dart native-port completions -------------------------------------------
// Doctrine and wire format: dart_port.h. Unlike the log port there is no
// registration step and no global — the destination rides along with each
// call, so completions cannot be misrouted between isolates and there is no
// last-writer-wins hazard.
void completionPost(int64_t port, int64_t token, bool ok) {
#ifdef MINIGPU_HAVE_DART_DL
  if (port == ILLEGAL_PORT || !g_dart_api_ready) return;
  // A false return means the port is gone (isolate exited, hot restart, the
  // app is shutting down). That is the entire reason this path exists: the
  // completion is dropped instead of aborting the process.
  (void)Dart_PostInteger_DL(static_cast<Dart_Port>(port),
                            (token << 1) | (ok ? 1 : 0));
#else
  (void)port;
  (void)token;
  (void)ok;
#endif
}

bool dartApiReady() {
#ifdef MINIGPU_HAVE_DART_DL
  return g_dart_api_ready;
#else
  return false;
#endif
}

} // namespace mgpu

namespace {
#ifdef __EMSCRIPTEN__
// Browser callbacks must not suspend an Asyncify stack while another call
// detaches the context. Polling these inert receipts from JS avoids both that
// reentrancy and a Dart function-pointer lifetime. No receipt owns a GPU ref.
std::mutex webObservationMutex;
std::unordered_map<uint32_t, std::shared_ptr<std::atomic<int>>> webObservations;
uint32_t nextWebObservation = 1;
#endif

// Ownership remains with the worker task until Dawn invokes the callback. A
// driver hang is therefore an intentionally pending Future, never a timeout
// that frees callback userdata while Dawn can still dereference it.
void finishSubmittedWorkWait(WGPUQueueWorkDoneStatus status, WGPUStringView,
                             void *userdata1, void *) {
  auto *wait = static_cast<mgpu::capture_work::WaitSignal *>(userdata1);
  wait->result = status == WGPUQueueWorkDoneStatus_Success
      ? MGPU_CONTEXT_LEASE_OK : MGPU_CONTEXT_LEASE_INTERNAL;
  wait->completed.store(true, std::memory_order_release);
}

int prepareSubmittedWorkWait(mgpu::MGPU *gpu, std::shared_ptr<mgpu::ContextPin> pin,
    const mgpu::capture_cleanup::Family &family, mgpu::capture_work::Completion complete,
    mgpu::capture_work::PreparedWait &prepared,
    std::shared_ptr<mgpu::capture_cleanup::GlobalTaskCustody> global = {}) noexcept {
  if (!gpu || !complete || !family.matches(pin ? pin->sourceDomain()
                                             : mgpu::source_scope::SourceDomain{}))
    return MGPU_CONTEXT_LEASE_INVALID;
  try {
    return prepared.prepare(family, [gpu, pin](mgpu::capture_work::WaitSignal &wait) {
    using mgpu::capture_work::Outcome;
    WGPUQueue queue = nullptr;
    WGPUInstance instance = nullptr;
    {
      mgpu::lock_guard<mgpu::mutex> lock(gpu->getGpuMutex());
      queue = pin ? pin->queue() : gpu->tryGetQueue();
      instance = pin ? pin->instance() : gpu->tryGetInstance();
      // Fire-and-forget dispatches remain in an open compute batch until a
      // queue boundary. Submit them before this callback becomes the fence.
      if (!queue || !instance) return Outcome::done(MGPU_CONTEXT_LEASE_NOT_READY);
      if (pin && !gpu->matchesReadyDevice(pin->device(), pin->deviceGeneration()))
        return Outcome::done(MGPU_CONTEXT_LEASE_RETIRED);
      gpu->flushBatchLocked();
    }
    WGPUQueueWorkDoneCallbackInfo info{};
    info.mode = WGPUCallbackMode_AllowProcessEvents;
    info.callback = finishSubmittedWorkWait;
    info.userdata1 = &wait;
    wgpuQueueOnSubmittedWorkDone(queue, info);

    // Dawn invokes this callback while its instance is pumped. On the worker
    // the worker's own pump does that between later tasks, so they no longer
    // queue behind the GPU; the receipt resumes there once the callback fired.
    if (gpu->getWebGPUThread().park(wait.parked, instance))
      return Outcome::deferred();
    // No worker to hand it to (the web runs queued work inline): pump here.
    // The backoff yields only briefly, then sleeps in 0.25 ms steps: an 8 ms
    // yield burned a core, and Sleep(1) overslept by up to a 15.6 ms tick.
    const auto start = std::chrono::steady_clock::now();
    while (!wait.completed.load(std::memory_order_acquire)) {
      wgpuInstanceProcessEvents(instance);
      if (!wait.completed.load(std::memory_order_acquire)) {
        mgpu::platformWaitBackoff(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start)
                .count());
      }
    }
    // Retain the exact creation pin through this acknowledgement, and leave
    // Dawn's event callback before entering any foreign native consumer.
    return wait.result == MGPU_CONTEXT_LEASE_OK
        ? Outcome::done() : Outcome{wait.result, false};
    }, std::move(complete), std::move(global));
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
int enqueueSubmittedWork(void *context, mgpu::task_queue::Prepared &task) noexcept {
  return static_cast<mgpu::MGPU *>(context)->getWebGPUThread().enqueuePrepared(task);
}
template<class Complete>
void waitForSubmittedWorkWithCompletion(mgpu::MGPU *gpu,
    std::shared_ptr<mgpu::ContextPin> pin, Complete complete) {
  // A standalone public wait admits fresh Work. Internal work and retirement
  // pass their already admitted Work/Cleanup family to the prepared overload.
  if (!gpu) { complete(MGPU_CONTEXT_LEASE_INVALID); return; }
  mgpu::capture_cleanup::Family family;
  const auto admitted = mgpu::capture_cleanup::Family::beginWork(
      pin ? pin->sourceDomain() : mgpu::source_scope::SourceDomain{}, family);
  if (admitted != mgpu::source_scope::Result::Ok) {
    complete(mgpu::capture_cleanup::nativeResult(admitted)); return;
  }
  mgpu::capture_work::PreparedWait prepared;
  auto result = prepareSubmittedWorkWait(gpu, pin, family, complete, prepared);
  if (result == MGPU_CONTEXT_LEASE_OK) result = prepared.enqueue(enqueueSubmittedWork, gpu);
  if (result != MGPU_CONTEXT_LEASE_OK) complete(result);
}
void waitForSubmittedWork(mgpu::MGPU *gpu, int64_t port, int64_t token,
                         std::shared_ptr<mgpu::ContextPin> pin = {},
                         std::function<void(int)> nativeComplete = {}) {
  // Legacy internal erased callbacks retain their API; the public C callback
  // route below reaches the typed helper directly, before any erasure.
  waitForSubmittedWorkWithCompletion(gpu, pin,
      [nativeComplete = std::move(nativeComplete), port, token](int result) {
    if (nativeComplete) nativeComplete(result);
    else mgpu::completionPost(port, token, result == MGPU_CONTEXT_LEASE_OK);
  });
}

void retireCreationDevice(std::shared_ptr<mgpu::ContextPin> pin,
                          std::function<void(int)> complete) {
#ifdef __EMSCRIPTEN__
  // WebGPU loss promises resolve before GPU work necessarily finishes. Never
  // export native Dawn's synchronous-destruction guarantee to this backend.
  complete(MGPU_CONTEXT_LEASE_UNSUPPORTED);
#else
  if (!pin) {
    complete(MGPU_CONTEXT_LEASE_INVALID);
    return;
  }
  if (!pin->gpu().invalidateDevice(pin->device(), pin->deviceGeneration())) {
    complete(MGPU_CONTEXT_LEASE_RETIRED);
    return;
  }
  try {
    pin->gpu().getWebGPUThread().enqueueAsync([pin, complete]() {
      int result = MGPU_CONTEXT_LEASE_INTERNAL;
      try {
        // Dawn DeviceBase::Destroy drains pending tasks and the native GPU
        // timeline before returning. Its loss callback can run much earlier;
        // only this return is the retirement evidence for this creation pin.
        wgpuDeviceDestroy(pin->device());
        result = MGPU_CONTEXT_LEASE_OK;
      } catch (...) {
        // A failed destruction is never a successful completion receipt.
        mgpu::quarantineContextPin(pin);
      }
      complete(result);
    });
  } catch (...) {
    mgpu::quarantineContextPin(pin);
    complete(MGPU_CONTEXT_LEASE_INTERNAL);
  }
#endif
}

constexpr uint32_t kBufferDescriptorUsageMask = 0xfc;
constexpr int kMaxBufferDescriptorSize = 2147483644;

bool validBufferDescriptor(int sizeBytes, uint32_t usage) {
  return sizeBytes >= 4 && sizeBytes <= kMaxBufferDescriptorSize &&
         (sizeBytes & 3) == 0 && usage != 0 &&
         (usage & ~kBufferDescriptorUsageMask) == 0;
}

MGPUBuffer *createBufferWithUsage(mgpu::MGPU &gpu, int sizeBytes,
                                  uint32_t usage) {
  if (!validBufferDescriptor(sizeBytes, usage)) return nullptr;

  try {
    WGPUDevice device = gpu.getDevice();
    if (!device) return nullptr;

    WGPULimits limits = {};
    if (wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
        static_cast<uint64_t>(sizeBytes) > limits.maxBufferSize) {
      return nullptr;
    }

    auto *buf = new mgpu::Buffer(gpu);
    try {
      buf->createBuffer(static_cast<size_t>(sizeBytes), mgpu::kUInt32,
                        static_cast<WGPUBufferUsage>(usage));
    } catch (...) {
      delete buf;
      throw;
    }
    return reinterpret_cast<MGPUBuffer *>(buf);
  } catch (...) {
    return nullptr;
  }
}

int currentDeviceBufferLimits(mgpu::MGPU &gpu, uint64_t *maxBufferSize,
                              uint64_t *maxStorageBufferBindingSize,
                              uint64_t *deviceGeneration) {
  if (!maxBufferSize || !maxStorageBufferBindingSize || !deviceGeneration)
    return 0;
  uint64_t generation = 0;
  WGPUDevice device = gpu.retainReadyDeviceGeneration(&generation);
  if (!device) return 0;
  WGPULimits limits = {};
  const bool queried = wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success;
  const bool current = gpu.matchesReadyDevice(device, generation);
  wgpuDeviceRelease(device);
  if (!queried || !current || !limits.maxBufferSize ||
      !limits.maxStorageBufferBindingSize) return 0;
  *maxBufferSize = limits.maxBufferSize;
  *maxStorageBufferBindingSize = limits.maxStorageBufferBindingSize;
  *deviceGeneration = generation;
  return 1;
}

int currentDeviceTextureLimits(mgpu::MGPU &gpu, uint64_t *maxDimension2D,
                               uint64_t *maxArrayLayers,
                               uint64_t *deviceGeneration) {
  if (!maxDimension2D || !maxArrayLayers || !deviceGeneration) return 0;
  uint64_t generation = 0;
  WGPUDevice device = gpu.retainReadyDeviceGeneration(&generation);
  if (!device) return 0;
  WGPULimits limits = {};
  const bool queried = wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success;
  const bool current = gpu.matchesReadyDevice(device, generation);
  wgpuDeviceRelease(device);
  if (!queried || !current || !limits.maxTextureDimension2D ||
      !limits.maxTextureArrayLayers) return 0;
  *maxDimension2D = limits.maxTextureDimension2D;
  *maxArrayLayers = limits.maxTextureArrayLayers;
  *deviceGeneration = generation;
  return 1;
}

} // namespace

namespace mgpu {
void waitForPinnedSubmittedWork(std::shared_ptr<ContextPin> pin,
                                std::function<void(int)> complete) {
  waitForSubmittedWork(pin ? &pin->gpu() : nullptr, 0, 0, pin,
                       std::move(complete));
}
int preparePinnedSubmittedWorkWait(std::shared_ptr<ContextPin> pin,
    const capture_cleanup::Family &family, InlineCallable<void(int)> complete,
    capture_work::PreparedWait &prepared,
    std::shared_ptr<capture_cleanup::GlobalTaskCustody> global) noexcept {
  return prepareSubmittedWorkWait(pin ? &pin->gpu() : nullptr, pin, family,
                                   std::move(complete), prepared, std::move(global));
}
int enqueuePinnedSubmittedWorkWait(std::shared_ptr<ContextPin> pin,
    capture_work::PreparedWait &prepared) noexcept {
  if (!pin) return MGPU_CONTEXT_LEASE_INVALID;
  return prepared.enqueue(enqueueSubmittedWork, &pin->gpu());
}
} // namespace mgpu

namespace mgpu::capture_cleanup {
int prepareFamily(MGPUContextLease *lease, Family &family) noexcept {
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  try {
    const auto pin = copyContextLeasePin(lease);
    if (!pin) return MGPU_CONTEXT_LEASE_INVALID;
    return nativeResult(Family::begin(pin->sourceDomain(), family));
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
#endif
}
int destroyShader(MGPUContextLease *lease, MGPUComputeShader *shader,
                  MGPUResourceCleanupCallback callback, void *userdata,
                  const Family &family) {
  if (!lease || !shader || !callback) return MGPU_CONTEXT_LEASE_INVALID;
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  try {
    auto pin = copyContextLeasePin(lease);
    if (!pin || !family.matches(pin->sourceDomain())) return MGPU_CONTEXT_LEASE_INVALID;
    return reinterpret_cast<ComputeShader *>(shader)->destroyCheckedAsync(
        std::move(pin), callback, userdata, family);
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
#endif
}
int destroyBuffer(MGPUContextLease *lease, MGPUBuffer *buffer,
                  MGPUResourceCleanupCallback callback, void *userdata,
                  const Family &family) {
  if (!lease || !buffer || !callback) return MGPU_CONTEXT_LEASE_INVALID;
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  try {
    auto pin = copyContextLeasePin(lease);
    if (!pin || !family.matches(pin->sourceDomain())) return MGPU_CONTEXT_LEASE_INVALID;
    return reinterpret_cast<Buffer *>(buffer)->destroyCheckedAsync(
        std::move(pin), callback, userdata, family);
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
#endif
}
int prepareUseFamily(MGPUComputeShaderUse *use, Family &out) noexcept {
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  try {
    return use && use->state ? nativeResult(Family::begin(use->state->pin->sourceDomain(), out))
                             : MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
#endif
}
void retireUse(MGPUComputeShaderUse *use, MGPUQueueCompletionCallback callback,
               void *userdata, const Family &family) {
  if (!callback) return;
#ifdef __EMSCRIPTEN__
  callback(MGPU_CONTEXT_LEASE_UNSUPPORTED, userdata);
#else
  if (!use || !use->state) { callback(MGPU_CONTEXT_LEASE_INVALID, userdata); return; }
  const auto state = use->state;
  state->shader->retireExclusiveUse(state, callback, userdata, family);
#endif
}
} // namespace mgpu::capture_cleanup

namespace {
// An MGPUElementType code as a BufferDataType; false for an unknown code.
bool elementTypeFromCode(int code, mgpu::BufferDataType &type) {
  switch (static_cast<MGPUElementType>(code)) {
  case MGPU_ELEM_I8: type = mgpu::kInt8; return true;
  case MGPU_ELEM_U8: type = mgpu::kUInt8; return true;
  case MGPU_ELEM_I16: type = mgpu::kInt16; return true;
  case MGPU_ELEM_U16: type = mgpu::kUInt16; return true;
  case MGPU_ELEM_I32: type = mgpu::kInt32; return true;
  case MGPU_ELEM_U32: type = mgpu::kUInt32; return true;
  case MGPU_ELEM_I64: type = mgpu::kInt64; return true;
  case MGPU_ELEM_U64: type = mgpu::kUInt64; return true;
  case MGPU_ELEM_F32: type = mgpu::kFloat32; return true;
  case MGPU_ELEM_F64: type = mgpu::kFloat64; return true;
  default: return false;
  }
}
} // namespace

#ifdef __cplusplus
using namespace mgpu;
extern "C" {
#endif

MGPU minigpu;

void mgpuSetLogCallback(MGPULogCallback callback) {
  SET_LOG_CALLBACK(callback);
}

int mgpuInitDartApi(void *initialize_api_dl_data) {
  return (int)mgpu::logInitDartApi(initialize_api_dl_data);
}

void mgpuSetLogPort(int64_t port) { mgpu::logSetPort(port); }

int32_t mgpuClearLogPortIf(int64_t port) {
  return mgpu::logClearPortIf(port) ? 1 : 0;
}

void mgpuSetLogLevel(int lvl) {
  SET_LOG_LEVEL(static_cast<mgpu::LogLevel>(lvl));
}

void mgpuFreeLogMessage(const char* msg) {
  free(const_cast<char*>(msg));
}

int mgpuPreferDisplayAdapter(int enable) {
  mgpu::setPreferDisplayAdapter(enable != 0);
  // tryGetInstance() probes without triggering lazy re-initialization.
  return minigpu.tryGetInstance() ? 1 : 0;
}

int mgpuGetSelectedAdapterName(char* out, int cap) {
  try {
    const std::string name = mgpu::selectedAdapterName();
    if (out && cap > 0) {
      const int n = (int)name.size() < cap - 1 ? (int)name.size() : cap - 1;
      memcpy(out, name.c_str(), (size_t)n);
      out[n] = '\0';
    }
    return (int)name.size();
  } catch (...) {
    return 0;
  }
}

// ── Attach / detach, NOT raw init / destroy ─────────────────────────────────
// `minigpu` is process-global; the Dart wrapper driving it is per-isolate, and
// `dart test` puts N isolates in ONE process. Counting the EXPLICIT attaches
// here (and only here — the lazy re-init inside getDevice/getQueue must not
// count) is what stops isolate A's teardown from destroying the device isolate
// B is still dispatching on.
void mgpuInitializeContext() {
  try {
    minigpu.attachContext();
  } catch (...) {
    // This legacy void ABI reports failure through readiness. Never let a
    // lost-generation refusal unwind across C/FFI.
    LOG_ERROR("%s", "mgpuInitializeContext: initialization refused");
  }
}

void mgpuInitializeContextAsync(MGPUCallback callback) {
  try {
    minigpu.attachContextAsync(callback);
  } catch (...) {
    // The init thread never started (nor the callback): report through
    // readiness, as an init that failed would.
    LOG_ERROR("%s", "mgpuInitializeContextAsync: initialization not started");
    if (callback) callback();
  }
}

void mgpuDestroyContext() {
  try { minigpu.detachContext(); } catch (...) {
    // The context remains quarantined and its final observer reports failure.
    // C++ teardown errors must not unwind through a C/FFI caller.
    MGPU_LOG(mgpu::LOG_ERROR, "Native context teardown failed; retained in quarantine");
  }
}

int mgpuContextRefCount() { return minigpu.attachCount(); }

#ifdef __EMSCRIPTEN__
uint32_t mgpuWebObserveContextRetirement(uint32_t expectedDevice) {
  uint64_t generation = 0;
  WGPUDevice device = minigpu.retainReadyDeviceGeneration(&generation);
  if (!device) return 0;
  struct ReleaseDevice {
    WGPUDevice device;
    ~ReleaseDevice() { wgpuDeviceRelease(device); }
  } release{device};
  if (reinterpret_cast<uintptr_t>(device) != expectedDevice) return 0;
  uint32_t token = 0;
  try {
    auto observation = std::make_shared<std::atomic<int>>(0);
    {
      std::lock_guard<std::mutex> lock(webObservationMutex);
      if (webObservations.size() >= 64 || nextWebObservation >= 0x7fffffffu)
        return 0;
      token = nextWebObservation++;
      webObservations.emplace(token, observation);
    }
    if (minigpu.observeContextRetirement(device, generation,
        [observation](bool succeeded) {
          observation->store(succeeded ? 1 : 2, std::memory_order_release);
        })) return token;
  } catch (...) {
    // Refusal cannot certify successful teardown or charge another attachment.
  }
  if (token) mgpuWebReleaseContextObservation(token);
  return 0;
}

int mgpuWebContextRetirementStatus(uint32_t token) {
  std::lock_guard<std::mutex> lock(webObservationMutex);
  const auto found = webObservations.find(token);
  return found == webObservations.end() ? -1
      : found->second->load(std::memory_order_acquire);
}

void mgpuWebReleaseContextObservation(uint32_t token) {
  std::lock_guard<std::mutex> lock(webObservationMutex);
  webObservations.erase(token);
}
#endif

int mgpuGetCurrentDeviceBufferLimits(uint64_t *maxBufferSize,
                                     uint64_t *maxStorageBufferBindingSize,
                                     uint64_t *deviceGeneration) {
  return currentDeviceBufferLimits(minigpu, maxBufferSize,
                                   maxStorageBufferBindingSize, deviceGeneration);
}

int mgpuGetCurrentDeviceTextureLimits(uint64_t *maxDimension2D,
                                      uint64_t *maxArrayLayers,
                                      uint64_t *deviceGeneration) {
  return currentDeviceTextureLimits(minigpu, maxDimension2D, maxArrayLayers,
                                    deviceGeneration);
}

int mgpuDefaultContextRetainLease(MGPUContextLease **outLease,
                                 MGPUContextHandle **outContext) {
  return mgpu::retainReadyDefaultContext(minigpu, outLease, outContext);
}

int mgpuDefaultContextRetainLeaseFundedV1(const MGPUSetupAuthorityV1 *authority,
    uint32_t size, MGPUContextLease **outLease, MGPUContextHandle **outContext) {
  if (outLease) *outLease = nullptr;
  if (outContext) *outContext = nullptr;
  if (!outLease || !outContext) return MGPU_CONTEXT_LEASE_INVALID;
  mgpu::setup_custody::SetupTickets tickets;
  const auto retained = mgpu::setup_custody::retainAuthority(authority, size, tickets);
  if (retained != MGPU_CONTEXT_LEASE_OK) return retained;
  return mgpu::retainReadyDefaultContextFunded(minigpu, tickets, outLease, outContext);
}

MGPUComputeShader *mgpuCreateComputeShader() {
  try {
    return reinterpret_cast<MGPUComputeShader *>(
        new mgpu::ComputeShader(minigpu));
  } catch (...) {
    LOG_ERROR("mgpuCreateComputeShader: allocation failed");
    return nullptr;
  }
}

void mgpuDestroyComputeShader(MGPUComputeShader *shader) {
  if (!shader) return;
  // NOT an inline delete: binds are queued and capture the shader pointer, so
  // deleting here would free it under a pending bind. See destroyQueued().
  try {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->destroyQueued();
  } catch (...) {
    LOG_ERROR("mgpuDestroyComputeShader: exception queueing the delete");
  }
}

int mgpuComputeShaderDestroyCheckedAsync(
    MGPUContextLease *lease, MGPUComputeShader *shader,
    MGPUResourceCleanupCallback callback, void *userdata) {
  if (!lease || !shader || !callback) return MGPU_CONTEXT_LEASE_INVALID;
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  try {
    auto pin = mgpu::copyContextLeasePin(lease);
    if (!pin) return MGPU_CONTEXT_LEASE_INVALID;
    return reinterpret_cast<mgpu::ComputeShader *>(shader)->destroyCheckedAsync(
        std::move(pin), callback, userdata);
  } catch (...) {
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#endif
}

void mgpuLoadKernel(MGPUComputeShader *shader, const char *kernelString) {
  if (!shader) {
    // Simple error - no LOG dependency
    return;
  }

  if (!kernelString || strlen(kernelString) == 0) {
    return;
  }

  try {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->loadKernelString(
        kernelString);
  } catch (...) {
    LOG_ERROR("mgpuLoadKernel: exception storing the kernel");
  }
}

int mgpuHasKernel(MGPUComputeShader *shader) {
  if (shader) {
    return reinterpret_cast<mgpu::ComputeShader *>(shader)->hasKernel();
  } else {
    return 0;
  }
}

int mgpuComputeShaderBuildPipeline(MGPUComputeShader *shader) {
  if (!shader) return 0;
  try {
    return reinterpret_cast<mgpu::ComputeShader *>(shader)->buildPipeline() ? 1
                                                                             : 0;
  } catch (...) {
    LOG_ERROR("mgpuComputeShaderBuildPipeline: exception building the pipeline");
    return 0;
  }
}

BufferDataType mapIntToBufferDataType(int dataType) {
  switch (dataType) {
  case 0:
    return kFloat32; // f16 -> f32 for simplicity
  case 1:
    return kFloat32;
  case 2:
    return kFloat64;
  case 3:
    return kInt8;
  case 4:
    return kInt16;
  case 5:
    return kInt32;
  case 6:
    return kInt64;
  case 7:
    return kUInt8;
  case 8:
    return kUInt16;
  case 9:
    return kUInt32;
  case 10:
    return kUInt64;
  default:
    return kFloat32; // Default fallback
  }
}

bool needsPacking(BufferDataType dataType) {
  switch (dataType) {
  case kInt8:
  case kUInt8:
  case kInt16:
  case kUInt16:
  case kInt64:
  case kUInt64:
  case kFloat64:
    return true;
  default:
    return false;
  }
}

size_t getElementSize(BufferDataType dataType) {
  switch (dataType) {
  case kInt8:
  case kUInt8:
    return 1;
  case kInt16:
  case kUInt16:
    return 2;
  case kInt32:
  case kUInt32:
  case kFloat32:
    return 4;
  case kInt64:
  case kUInt64:
  case kFloat64:
    return 8;
  default:
    return 4; // Default to 4 bytes
  }
}

MGPUBuffer *mgpuCreateBuffer(int byteSize, int dataType) {
  LOG_INFO("mgpuCreateBuffer: byteSize=%d, dataType=%d", byteSize,
           dataType);
  // Negative would reach Dawn as a size_t near 2^64 and come back as an
  // error buffer, which looks like success.
  if (byteSize < 0) {
    LOG_ERROR("mgpuCreateBuffer: negative byteSize %d", byteSize);
    return nullptr;
  }

  BufferDataType mappedType = mapIntToBufferDataType(dataType);
  LOG_INFO("mappedType=%d", (int)mappedType);

  size_t elementSize = getElementSize(mappedType);
  size_t elementCount = static_cast<size_t>(byteSize) / elementSize;

  mgpu::Buffer *buf = nullptr;
  try {
    buf = new mgpu::Buffer(minigpu);
    if (needsPacking(mappedType)) {
      LOG_INFO("Creating packed buffer with byteSize=%d", byteSize);
      buf->createBuffer(static_cast<size_t>(byteSize), mappedType);
    } else {
      LOG_INFO("Creating direct buffer with byteSize=%d (elementCount=%zu * "
               "elementSize=%zu)",
               byteSize, elementCount, getElementSize(mappedType));
      buf->createBuffer(static_cast<size_t>(byteSize), mappedType);
    }
  } catch (...) {
    LOG_ERROR("Exception in mgpuCreateBuffer");
    delete buf;
    return nullptr;
  }
  return reinterpret_cast<MGPUBuffer *>(buf);
}

uint32_t mgpuBufferDescriptorVersion(void) { return 1; }

MGPUBuffer *mgpuCreateBufferWithUsage(int sizeBytes, uint32_t usage) {
  return createBufferWithUsage(minigpu, sizeBytes, usage);
}

void mgpuDestroyBuffer(MGPUBuffer *buffer) {
  if (buffer) {
    // Deferred behind any queued read of this buffer (see destroyQueued).
    reinterpret_cast<mgpu::Buffer *>(buffer)->destroyQueued();
  }
}

int mgpuBufferDestroyCheckedAsync(
    MGPUContextLease *lease, MGPUBuffer *buffer,
    MGPUResourceCleanupCallback callback, void *userdata) {
  if (!lease || !buffer || !callback) return MGPU_CONTEXT_LEASE_INVALID;
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  try {
    auto pin = mgpu::copyContextLeasePin(lease);
    if (!pin) return MGPU_CONTEXT_LEASE_INVALID;
    return reinterpret_cast<mgpu::Buffer *>(buffer)->destroyCheckedAsync(
        std::move(pin), callback, userdata);
  } catch (...) {
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#endif
}

// ── Multi-GPU context handles ───────────────────────────────────────────────
// An MGPUContextHandle identifies a registered heap-allocated MGPU owner. All
// per-object entry points (setBuffer/dispatch/read/write/destroy/...) already
// route through the MGPU& the object captured at creation, so only creation
// needs these handle-aware variants. Explicit pins retain the creation device
// and worker after logical retirement; physical teardown runs off that worker.

MGPUContextHandle *mgpuCreateContextHandle(const char *adapterFilter) {
  return mgpu::createOwnedContext(adapterFilter);
}

void mgpuContextInitializeAsync(MGPUContextHandle *handle,
                                MGPUCallback callback) {
  // The callback carries no status, so a refused handle or an init that never
  // started still calls it (as the port twin posts false): the caller must not
  // wait forever, and checks the context afterwards.
  try {
    if (auto owner = mgpu::retainContextOwner(handle)) {
      owner->gpu().initializeContextAsync([owner, callback]() {
        if (callback) callback();
      });
      return;
    }
  } catch (...) {
    // The init thread was never started.
  }
  if (callback) callback();
}

void mgpuContextInitializeAsyncToPort(MGPUContextHandle *handle, int64_t port,
                                      int64_t token) {
  try {
    auto owner = mgpu::retainContextOwner(handle);
    if (!owner) {
      mgpu::completionPost(port, token, false);
      return;
    }
    owner->gpu().initializeContextAsync([owner, port, token]() {
      auto device = owner->gpu().retainReadyDevice();
      mgpu::completionPost(port, token, !owner->retired() && device);
      if (device) wgpuDeviceRelease(device);
    });
  } catch (...) {
    // The init thread was never started.
    mgpu::completionPost(port, token, false);
  }
}

void mgpuContextWaitForSubmittedWorkAsyncToPort(MGPUContextHandle *handle,
                                                int64_t port, int64_t token) {
  std::shared_ptr<mgpu::ContextPin> pin;
  mgpu::retainContextPin(handle, pin);
  try {
    waitForSubmittedWork(pin ? &pin->gpu() : nullptr, port, token, pin);
  } catch (...) {
    // Thrown before the wait was queued: nothing has posted yet.
    mgpu::completionPost(port, token, false);
  }
}

void mgpuContextLeaseWaitForSubmittedWorkAsyncToPort(MGPUContextLease *lease,
                                                   int64_t port, int64_t token) {
  auto pin = mgpu::copyContextLeasePin(lease);
  try {
    waitForSubmittedWork(pin ? &pin->gpu() : nullptr, port, token, pin);
  } catch (...) {
    mgpu::completionPost(port, token, false);
  }
}

void mgpuContextLeaseWaitForSubmittedWorkAsync(
    MGPUContextLease *lease, MGPUQueueCompletionCallback callback,
    void *userdata) {
  if (!callback) return;
#ifdef __EMSCRIPTEN__
  callback(MGPU_CONTEXT_LEASE_UNSUPPORTED, userdata);
#else
  std::shared_ptr<mgpu::ContextPin> pin;
  try {
    pin = mgpu::copyContextLeasePin(lease);
  } catch (...) {
    callback(MGPU_CONTEXT_LEASE_INTERNAL, userdata);
    return;
  }
  if (!pin) {
    callback(MGPU_CONTEXT_LEASE_INVALID, userdata);
    return;
  }
  try {
    waitForSubmittedWorkWithCompletion(&pin->gpu(), pin,
        [callback, userdata](int result) { callback(result, userdata); });
  } catch (...) {
    // Enqueue did not accept the job. No success or GPU-retirement receipt.
    callback(MGPU_CONTEXT_LEASE_INTERNAL, userdata);
  }
#endif
}

int mgpuComputeShaderAcquireExclusiveUse(
    MGPUContextLease *lease, MGPUComputeShader *shader,
    MGPUComputeShaderUse **out_use) {
  if (!out_use) return MGPU_CONTEXT_LEASE_INVALID;
  *out_use = nullptr;
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  if (!lease || !shader) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    auto pin = mgpu::copyContextLeasePin(lease);
    if (!pin) return MGPU_CONTEXT_LEASE_INVALID;
    mgpu::source_scope::MetadataUnique<MGPUComputeShaderUse> use;
    const int result = reinterpret_cast<mgpu::ComputeShader *>(shader)
        ->acquireExclusiveUse(std::move(pin), use);
    if (result != MGPU_CONTEXT_LEASE_OK) return result;
    *out_use = use.release();
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) {
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#endif
}

int mgpuComputeShaderUseDispatch(MGPUComputeShaderUse *use,
                                 int x, int y, int z) {
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  if (!use || !use->state) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    auto state = use->state;
    return state->shader->dispatchExclusiveUse(state, x, y, z);
  } catch (...) {
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#endif
}

int mgpuComputeShaderUseBindImportedFrame(MGPUComputeShaderUse *use,
                                         uint32_t slot, MGPUImportedFrame *frame) {
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  if (!use || !use->state || !frame) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    auto state = use->state;
    return state->shader->bindImportedFrame(state, slot,
                                            mgpu::copyImportedFrameState(frame));
  } catch (...) {
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#endif
}

void mgpuComputeShaderUseRetireAsync(
    MGPUComputeShaderUse *use, MGPUQueueCompletionCallback callback,
    void *userdata) {
  if (!callback) return;
#ifdef __EMSCRIPTEN__
  callback(MGPU_CONTEXT_LEASE_UNSUPPORTED, userdata);
#else
  mgpu::capture_cleanup::Family family;
  const auto prepared = mgpu::capture_cleanup::prepareUseFamily(use, family);
  if (prepared != MGPU_CONTEXT_LEASE_OK) { callback(prepared, userdata); return; }
  mgpu::capture_cleanup::retireUse(use, callback, userdata, family);
#endif
}

int mgpuComputeShaderUseRelease(MGPUComputeShaderUse *use) {
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  if (!use || !use->state) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    auto state = use->state;
    const int result = state->shader->releaseExclusiveUse(state);
    if (result == MGPU_CONTEXT_LEASE_OK) mgpu::source_scope::MetadataDelete<MGPUComputeShaderUse>{}(use);
    return result;
  } catch (...) {
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#endif
}

void mgpuContextLeaseRetireDeviceAsync(
    MGPUContextLease *lease, MGPUDeviceRetirementCallback callback,
    void *userdata) {
  try {
    retireCreationDevice(mgpu::copyContextLeasePin(lease),
        [callback, userdata](int result) {
          if (callback) callback(result, userdata);
        });
  } catch (...) {
    // retireCreationDevice reports its own failures; this was thrown before
    // it ran, so the callback has not run either.
    if (callback) callback(MGPU_CONTEXT_LEASE_INTERNAL, userdata);
  }
}

void mgpuContextLeaseRetireDeviceAsyncToPort(MGPUContextLease *lease,
                                           int64_t port, int64_t token) {
  try {
    retireCreationDevice(mgpu::copyContextLeasePin(lease),
        [port, token](int result) {
          mgpu::completionPost(port, token, result == MGPU_CONTEXT_LEASE_OK);
        });
  } catch (...) {
    mgpu::completionPost(port, token, false);
  }
}

void mgpuContextLeaseObserveRetirementAsync(
    MGPUContextLease *lease, MGPUDeviceRetirementCallback callback,
    void *userdata) {
  if (!callback) return;
#ifdef __EMSCRIPTEN__
  callback(MGPU_CONTEXT_LEASE_UNSUPPORTED, userdata);
#else
  int refused = MGPU_CONTEXT_LEASE_INVALID;
  try {
    auto pin = mgpu::copyContextLeasePin(lease);
    if (pin) {
      if (pin->gpu().observeContextRetirement(
              pin->device(), pin->deviceGeneration(),
              [callback, userdata](bool succeeded) {
                callback(succeeded ? MGPU_CONTEXT_LEASE_OK
                                   : MGPU_CONTEXT_LEASE_INTERNAL, userdata);
              })) return;
      refused = MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    }
  } catch (...) { refused = MGPU_CONTEXT_LEASE_INTERNAL; }
  callback(refused, userdata);
#endif
}

void mgpuContextLeaseObserveRetirementAsyncToPort(MGPUContextLease *lease,
                                                int64_t port, int64_t token) {
#ifdef __EMSCRIPTEN__
  mgpu::completionPost(port, token, false);
#else
  try {
    auto pin = mgpu::copyContextLeasePin(lease);
    if (pin && pin->gpu().observeContextRetirement(
          pin->device(), pin->deviceGeneration(), [port, token](bool succeeded) {
            mgpu::completionPost(port, token, succeeded);
          })) return;
  } catch (...) {}
  mgpu::completionPost(port, token, false);
#endif
}

void mgpuDestroyContextHandle(MGPUContextHandle *handle) {
  mgpu::retireOwnedContext(handle);
}

int mgpuContextGetAdapterName(MGPUContextHandle *handle, char *out, int cap) {
  // A destroyed or unknown handle is refused, not dereferenced; the owner is
  // kept alive for the call.
  auto owner = mgpu::retainHandleOwner(handle);
  if (!owner) return 0;
  try {
    const std::string name = owner->gpu().adapterName();
    if (out && cap > 0) {
      const int n = (int)name.size() < cap - 1 ? (int)name.size() : cap - 1;
      memcpy(out, name.c_str(), (size_t)n);
      out[n] = '\0';
    }
    return (int)name.size();
  } catch (...) {
    return 0;
  }
}

MGPUBuffer *mgpuContextCreateBuffer(MGPUContextHandle *handle, int byteSize,
                                    int dataType) {
  auto owner = mgpu::retainHandleOwner(handle);
  if (!owner) return nullptr;
  if (byteSize < 0) {
    LOG_ERROR("mgpuContextCreateBuffer: negative byteSize %d", byteSize);
    return nullptr;
  }
  BufferDataType mappedType = mapIntToBufferDataType(dataType);
  mgpu::Buffer *buf = nullptr;
  try {
    buf = new mgpu::Buffer(owner->gpu());
    buf->createBuffer(static_cast<size_t>(byteSize), mappedType);
  } catch (...) {
    LOG_ERROR("Exception in mgpuContextCreateBuffer");
    delete buf;
    return nullptr;
  }
  return reinterpret_cast<MGPUBuffer *>(buf);
}

int mgpuContextGetCurrentDeviceBufferLimits(
    MGPUContextHandle *handle, uint64_t *maxBufferSize,
    uint64_t *maxStorageBufferBindingSize, uint64_t *deviceGeneration) {
  std::shared_ptr<mgpu::ContextPin> pin;
  if (mgpu::retainContextPin(handle, pin) != MGPU_CONTEXT_LEASE_OK || !pin)
    return 0;
  if (!pin->current()) return 0;
  const int result = currentDeviceBufferLimits(
      pin->gpu(), maxBufferSize, maxStorageBufferBindingSize, deviceGeneration);
  return pin->current() ? result : 0;
}

MGPUBuffer *mgpuContextCreateBufferWithUsage(MGPUContextHandle *handle,
                                             int sizeBytes, uint32_t usage) {
  auto owner = mgpu::retainHandleOwner(handle);
  if (!owner) return nullptr;
  return createBufferWithUsage(owner->gpu(), sizeBytes, usage);
}

int mgpuContextGetCurrentDeviceTextureLimits(
    MGPUContextHandle *handle, uint64_t *maxDimension2D,
    uint64_t *maxArrayLayers, uint64_t *deviceGeneration) {
  std::shared_ptr<mgpu::ContextPin> pin;
  if (mgpu::retainContextPin(handle, pin) != MGPU_CONTEXT_LEASE_OK || !pin)
    return 0;
  if (!pin->current()) return 0;
  const int result = currentDeviceTextureLimits(
      pin->gpu(), maxDimension2D, maxArrayLayers, deviceGeneration);
  return pin->current() ? result : 0;
}

MGPUComputeShader *mgpuContextCreateComputeShader(MGPUContextHandle *handle) {
  auto owner = mgpu::retainHandleOwner(handle);
  if (!owner) return nullptr;
  try {
    return reinterpret_cast<MGPUComputeShader *>(
        new mgpu::ComputeShader(owner->gpu()));
  } catch (...) {
    LOG_ERROR("%s", "mgpuContextCreateComputeShader: allocation failed");
    return nullptr;
  }
}

void mgpuSetBuffer(MGPUComputeShader *shader, int tag, MGPUBuffer *buffer) {
  if (shader && tag >= 0 && buffer) {
    try {
      reinterpret_cast<mgpu::ComputeShader *>(shader)->setBuffer(
          tag, *reinterpret_cast<mgpu::Buffer *>(buffer));
    } catch (...) {
      LOG_ERROR("mgpuSetBuffer: exception binding the buffer");
    }
  }
}

void mgpuSetBufferFire(MGPUComputeShader *shader, int tag,
                       MGPUBuffer *buffer) {
  if (shader && tag >= 0 && buffer) {
    try {
      reinterpret_cast<mgpu::ComputeShader *>(shader)->setBufferQueued(
          tag, *reinterpret_cast<mgpu::Buffer *>(buffer));
    } catch (...) {
      LOG_ERROR("mgpuSetBufferFire: exception queueing the bind");
    }
  }
}

void mgpuDispatch(MGPUComputeShader *shader, int groupsX, int groupsY,
                  int groupsZ) {
  if (shader) {
    try {
      reinterpret_cast<mgpu::ComputeShader *>(shader)->dispatch(
          groupsX, groupsY, groupsZ);
    } catch (...) {
      LOG_ERROR("mgpuDispatch: exception queueing the dispatch");
    }
  }
}

void mgpuDispatchAsync(MGPUComputeShader *shader, int groupsX, int groupsY,
                       int groupsZ, MGPUCallback callback) {
  if (shader) {
    try {
      reinterpret_cast<mgpu::ComputeShader *>(shader)->dispatchAsync(
          groupsX, groupsY, groupsZ, callback);
    } catch (...) {
      // Thrown before the work was queued (a refused queue calls back
      // itself), so the callback has not run.
      LOG_ERROR("mgpuDispatchAsync: exception queueing the dispatch");
      if (callback) callback();
    }
  } else if (callback) {
    callback();
  }
}

// ── Port-based completions ──────────────────────────────────────────────────
// Same work, same threads; the only difference is that the completion travels
// as an int64 on a Dart port instead of through a function pointer whose
// lifetime the caller cannot safely manage. See dart_port.h.
//
// Every one of these ALWAYS posts exactly once, including on the argument
// checks — a caller awaiting a completion that silently never arrives is a
// hang, which is worse to debug than a failure.

void mgpuDispatchAsyncToPort(MGPUComputeShader *shader, int groupsX,
                             int groupsY, int groupsZ, int64_t port,
                             int64_t token) {
  if (!shader) {
    mgpu::completionPost(port, token, false);
    return;
  }
  try {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->dispatchAsyncChecked(
        groupsX, groupsY, groupsZ,
        [port, token](bool ok) { mgpu::completionPost(port, token, ok); });
  } catch (...) {
    // Thrown before the work was queued: nothing has posted yet.
    LOG_ERROR("mgpuDispatchAsyncToPort: exception queueing the dispatch");
    mgpu::completionPost(port, token, false);
  }
}

void mgpuComputeShaderBuildPipelineAsyncToPort(MGPUComputeShader *shader,
                                               int64_t port, int64_t token) {
  if (!shader) {
    mgpu::completionPost(port, token, false);
    return;
  }
  try {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->buildPipelineAsync(
        [port, token](bool ok) { mgpu::completionPost(port, token, ok); });
  } catch (...) {
    // Thrown before the build was queued: nothing has posted yet.
    LOG_ERROR("mgpuComputeShaderBuildPipelineAsyncToPort: exception queueing "
              "the build");
    mgpu::completionPost(port, token, false);
  }
}

void mgpuReadAsyncToPort(MGPUBuffer *buffer, void *outputData,
                         size_t elementCount, size_t elementOffset,
                         int elementType, int64_t port, int64_t token) {
  if (!buffer || !outputData) {
    mgpu::completionPost(port, token, false);
    return;
  }
  auto *buf = reinterpret_cast<mgpu::Buffer *>(buffer);
  mgpu::BufferDataType type;
  if (!elementTypeFromCode(elementType, type)) {
    LOG_ERROR("mgpuReadAsyncToPort: unknown element type %d", elementType);
    mgpu::completionPost(port, token, false);
    return;
  }
  try {
    // A read that did not land (bad range, no device, failed map) completes
    // with ok=false: the destination holds nothing the caller asked for.
    buf->readAsyncResult(outputData, elementCount, elementOffset, type,
                         [port, token](bool ok) {
                           mgpu::completionPost(port, token, ok);
                         });
  } catch (...) {
    LOG_ERROR("mgpuReadAsyncToPort: exception issuing the read");
    mgpu::completionPost(port, token, false);
  }
}

void mgpuInitializeContextAsyncToPort(int64_t port, int64_t token) {
  try {
    minigpu.attachContextAsyncResult([port, token](bool attached) {
      mgpu::completionPost(port, token, attached);
    });
  } catch (...) {
    // The init thread never started, so nothing has posted yet.
    LOG_ERROR("mgpuInitializeContextAsyncToPort: initialization not started");
    mgpu::completionPost(port, token, false);
  }
}

uint32_t mgpuBufferBindingVersion(void) { return 1; }

int mgpuSetBufferRange(MGPUComputeShader *shader, int slot, MGPUBuffer *buffer,
                       int kind, int offsetBytes, int sizeBytes) {
  if (!shader || !buffer || slot < 0 || slot > 31 || kind < 0 || kind > 2 ||
      offsetBytes < 0 || sizeBytes <= 0 || offsetBytes > 2147483644 ||
      sizeBytes > 2147483644 || (offsetBytes & 3) != 0 ||
      (sizeBytes & 3) != 0) {
    return 1;
  }

  try {
    auto *nativeShader = reinterpret_cast<mgpu::ComputeShader *>(shader);
    auto *nativeBuffer = reinterpret_cast<mgpu::Buffer *>(buffer);
    mgpu::MGPU &gpu = nativeShader->getMGPU();
    if (&gpu != &nativeBuffer->getMGPU()) return 1;

    WGPUBuffer handle = nativeBuffer->getWGPUBuffer();
    const uint64_t bufferSize = nativeBuffer->getSize();
    const uint64_t offset = static_cast<uint64_t>(offsetBytes);
    const uint64_t size = static_cast<uint64_t>(sizeBytes);
    if (!handle || offset > bufferSize || size > bufferSize - offset) return 1;

    const WGPUBufferUsage usage = nativeBuffer->bufferData.usage;
    mgpu::BindingKind bindingKind;
    if (kind == 2) {
      if ((usage & WGPUBufferUsage_Uniform) == 0) return 1;
      bindingKind = mgpu::BindingKind::kUniformBuffer;
    } else {
      if ((usage & WGPUBufferUsage_Storage) == 0) return 1;
      bindingKind = kind == 0 ? mgpu::BindingKind::kStorageBuffer
                              : mgpu::BindingKind::kReadOnlyStorageBuffer;
    }

    if (!gpu.tryGetQueue()) return 2;
    WGPUDevice device = gpu.retainReadyDevice();
    if (!device) return 2;
    WGPULimits limits = {};
    const bool queried = wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success;
    wgpuDeviceRelease(device);
    if (!queried) return 3;
    if (static_cast<uint32_t>(slot) >= limits.maxBindingsPerBindGroup) return 1;

    const uint32_t alignment =
        kind == 2 ? limits.minUniformBufferOffsetAlignment
                  : limits.minStorageBufferOffsetAlignment;
    const uint64_t maxBindingSize =
        kind == 2 ? limits.maxUniformBufferBindingSize
                  : limits.maxStorageBufferBindingSize;
    if ((alignment != 0 && offset % alignment != 0) ||
        size > maxBindingSize) {
      return 1;
    }

    return nativeShader->setBufferRangeQueued(slot, handle,
                                       static_cast<size_t>(size),
                                       static_cast<size_t>(offset), bindingKind)
        ? 0 : MGPU_SHADER_BIND_BUSY;
  } catch (...) {
    return 3;
  }
}

void mgpuWaitForSubmittedWorkAsyncToPort(int64_t port, int64_t token) {
  try {
    waitForSubmittedWork(&minigpu, port, token);
  } catch (...) {
    // Thrown before the wait was queued: nothing has posted yet.
    mgpu::completionPost(port, token, false);
  }
}

void mgpuFlushBatch() {
  try {
    minigpu.flushBatch();
  } catch (...) {
    LOG_ERROR("mgpuFlushBatch: exception flushing the batch");
  }
}

void mgpuDrainWorkQueue() {
  auto &thread = minigpu.getWebGPUThread();
  // Called FROM the worker (a completion handler, say) the wait could never be
  // satisfied — the queue cannot drain while we are the thing draining it.
  if (std::this_thread::get_id() == thread.threadId()) return;
  try {
    thread.enqueueSync<int>([]() { return 0; });
  } catch (...) {
    // The worker has exited: there is no queue left to drain.
  }
}

void mgpuReadSync(MGPUBuffer *buffer, void *outputData, size_t size,
                        size_t offset) {
  if (buffer && outputData) {
    try {
      // Use float as default - could be improved with type info
      reinterpret_cast<mgpu::Buffer *>(buffer)->read(
          static_cast<float *>(outputData), size / sizeof(float), offset);
    } catch (...) {
      LOG_ERROR("mgpuReadSync: exception reading the buffer");
    }
  }
}

void mgpuReadAsyncFloat(MGPUBuffer *buffer, float *outputData,
                              size_t elementCount, size_t elementOffset,
                              MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          outputData, elementCount, elementOffset, callback);
    } catch (...) {
      // Thrown before the read was queued (a refused queue calls back
      // itself): call back so the caller is not left waiting.
      callback();
    }
  }
}

void mgpuReadAsyncInt8(MGPUBuffer *buffer, int8_t *outputData,
                             size_t elementCount, size_t elementOffset,
                             MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          outputData, elementCount, elementOffset, callback);
    } catch (...) {
      // Thrown before the read was queued (a refused queue calls back
      // itself): call back so the caller is not left waiting.
      callback();
    }
  }
}

void mgpuReadAsyncUint8(MGPUBuffer *buffer, uint8_t *outputData,
                              size_t elementCount, size_t elementOffset,
                              MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          outputData, elementCount, elementOffset, callback);
    } catch (...) {
      // Thrown before the read was queued (a refused queue calls back
      // itself): call back so the caller is not left waiting.
      callback();
    }
  }
}

void mgpuReadAsyncInt16(MGPUBuffer *buffer, int16_t *outputData,
                              size_t elementCount, size_t elementOffset,
                              MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          outputData, elementCount, elementOffset, callback);
    } catch (...) {
      // Thrown before the read was queued (a refused queue calls back
      // itself): call back so the caller is not left waiting.
      callback();
    }
  }
}

void mgpuReadAsyncUint16(MGPUBuffer *buffer, uint16_t *outputData,
                               size_t elementCount, size_t elementOffset,
                               MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          outputData, elementCount, elementOffset, callback);
    } catch (...) {
      // Thrown before the read was queued (a refused queue calls back
      // itself): call back so the caller is not left waiting.
      callback();
    }
  }
}

void mgpuReadAsyncInt32(MGPUBuffer *buffer, int32_t *outputData,
                              size_t elementCount, size_t elementOffset,
                              MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          outputData, elementCount, elementOffset, callback);
    } catch (...) {
      // Thrown before the read was queued (a refused queue calls back
      // itself): call back so the caller is not left waiting.
      callback();
    }
  }
}

void mgpuReadAsyncUint32(MGPUBuffer *buffer, uint32_t *outputData,
                               size_t elementCount, size_t elementOffset,
                               MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          outputData, elementCount, elementOffset, callback);
    } catch (...) {
      // Thrown before the read was queued (a refused queue calls back
      // itself): call back so the caller is not left waiting.
      callback();
    }
  }
}

void mgpuReadAsyncInt64(MGPUBuffer *buffer, int64_t *outputData,
                              size_t elementCount, size_t elementOffset,
                              MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          outputData, elementCount, elementOffset, callback);
    } catch (...) {
      // Thrown before the read was queued (a refused queue calls back
      // itself): call back so the caller is not left waiting.
      callback();
    }
  }
}

void mgpuReadAsyncUint64(MGPUBuffer *buffer, uint64_t *outputData,
                               size_t elementCount, size_t elementOffset,
                               MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          outputData, elementCount, elementOffset, callback);
    } catch (...) {
      // Thrown before the read was queued (a refused queue calls back
      // itself): call back so the caller is not left waiting.
      callback();
    }
  }
}

void mgpuWriteFloat(MGPUBuffer *buffer, const float *inputData,
                            size_t byteSize) {
  LOG_INFO("mgpuWriteFloat: byteSize=%zu", byteSize);
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(float);
    LOG_INFO("Converting to elementCount=%zu", elementCount);
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
    } catch (...) {
      LOG_ERROR("mgpuWriteFloat: exception writing the buffer");
    }
  }
}

void mgpuReadAsyncDouble(MGPUBuffer *buffer, double *outputData,
                               size_t elementCount, size_t elementOffset,
                               MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          outputData, elementCount, elementOffset, callback);
    } catch (...) {
      // Thrown before the read was queued (a refused queue calls back
      // itself): call back so the caller is not left waiting.
      callback();
    }
  }
}

void mgpuReadSyncInt8(MGPUBuffer *buffer, int8_t *outputData,
                            size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->read(outputData, elementCount,
                                                     elementOffset);
    } catch (...) {
      // Handle error silently
    }
  }
}

void mgpuReadSyncUint8(MGPUBuffer *buffer, uint8_t *outputData,
                             size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->read(outputData, elementCount,
                                                     elementOffset);
    } catch (...) {
      // Handle error silently
    }
  }
}

void mgpuReadSyncInt16(MGPUBuffer *buffer, int16_t *outputData,
                             size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->read(outputData, elementCount,
                                                     elementOffset);
    } catch (...) {
      // Handle error silently
    }
  }
}

void mgpuReadSyncUint16(MGPUBuffer *buffer, uint16_t *outputData,
                              size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->read(outputData, elementCount,
                                                     elementOffset);
    } catch (...) {
      // Handle error silently
    }
  }
}

void mgpuReadSyncInt32(MGPUBuffer *buffer, int32_t *outputData,
                             size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->read(outputData, elementCount,
                                                     elementOffset);
    } catch (...) {
      // Handle error silently
    }
  }
}

void mgpuReadSyncUint32(MGPUBuffer *buffer, uint32_t *outputData,
                              size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->read(outputData, elementCount,
                                                     elementOffset);
    } catch (...) {
      // Handle error silently
    }
  }
}

void mgpuReadSyncInt64(MGPUBuffer *buffer, int64_t *outputData,
                             size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->read(outputData, elementCount,
                                                     elementOffset);
    } catch (...) {
      // Handle error silently
    }
  }
}

void mgpuReadSyncUint64(MGPUBuffer *buffer, uint64_t *outputData,
                              size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->read(outputData, elementCount,
                                                     elementOffset);
    } catch (...) {
      // Handle error silently
    }
  }
}

void mgpuReadSyncFloat32(MGPUBuffer *buffer, float *outputData,
                               size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->read(outputData, elementCount,
                                                     elementOffset);
    } catch (...) {
      // Handle error silently
    }
  }
}

void mgpuReadSyncFloat64(MGPUBuffer *buffer, double *outputData,
                               size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->read(outputData, elementCount,
                                                     elementOffset);
    } catch (...) {
      // Handle error silently
    }
  }
}

int mgpuReadSyncChecked(MGPUBuffer *buffer, void *outputData,
                        size_t elementCount, size_t elementOffset,
                        int elementType) {
  if (!buffer || !outputData) return 0;
  mgpu::BufferDataType type;
  if (!elementTypeFromCode(elementType, type)) {
    LOG_ERROR("mgpuReadSyncChecked: unknown element type %d", elementType);
    return 0;
  }
  try {
    return reinterpret_cast<mgpu::Buffer *>(buffer)->readResult(
               outputData, elementCount, elementOffset, type)
               ? 1
               : 0;
  } catch (...) {
    LOG_ERROR("mgpuReadSyncChecked: exception reading the buffer");
    return 0;
  }
}

void mgpuWriteInt8(MGPUBuffer *buffer, const int8_t *inputData,
                           size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(int8_t);
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
    } catch (...) {
      LOG_ERROR("mgpuWriteInt8: exception writing the buffer");
    }
  }
}

void mgpuWriteInt16(MGPUBuffer *buffer, const int16_t *inputData,
                            size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(int16_t);
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
    } catch (...) {
      LOG_ERROR("mgpuWriteInt16: exception writing the buffer");
    }
  }
}

void mgpuWriteInt32(MGPUBuffer *buffer, const int32_t *inputData,
                            size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(int32_t);
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
    } catch (...) {
      LOG_ERROR("mgpuWriteInt32: exception writing the buffer");
    }
  }
}

void mgpuWriteInt64(MGPUBuffer *buffer, const int64_t *inputData,
                            size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(int64_t);
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
    } catch (...) {
      LOG_ERROR("mgpuWriteInt64: exception writing the buffer");
    }
  }
}

void mgpuWriteUint8(MGPUBuffer *buffer, const uint8_t *inputData,
                            size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(uint8_t);
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
    } catch (...) {
      LOG_ERROR("mgpuWriteUint8: exception writing the buffer");
    }
  }
}

void mgpuWriteUint16(MGPUBuffer *buffer, const uint16_t *inputData,
                             size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(uint16_t);
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
    } catch (...) {
      LOG_ERROR("mgpuWriteUint16: exception writing the buffer");
    }
  }
}

void mgpuWriteUint32(MGPUBuffer *buffer, const uint32_t *inputData,
                             size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(uint32_t);
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
    } catch (...) {
      LOG_ERROR("mgpuWriteUint32: exception writing the buffer");
    }
  }
}

void mgpuWriteBufferAt(MGPUBuffer *buffer, const void *inputData,
                       size_t byteSize, size_t dstByteOffset) {
  if (!buffer || !inputData) return;
  // MUST NOT THROW. Letting a C++ exception unwind out of an FFI entry point
  // is undefined behaviour for any caller — it does not become a Dart
  // exception, it corrupts the frame it unwinds through. This entry point is
  // additionally bound as a LEAF call (see mgpuWriteBufferAtLeaf on the Dart
  // side), where the VM has not transitioned out of Dart state at all, so an
  // escaping exception is not merely undefined but reliably fatal.
  // tryWriteBytesAt logs and refuses a bad range or a lost device without
  // throwing; the catch is the backstop.
  try {
    (void)reinterpret_cast<mgpu::Buffer *>(buffer)->tryWriteBytesAt(
        inputData, byteSize, dstByteOffset);
  } catch (const std::exception &e) {
    LOG_ERROR("mgpuWriteBufferAt: %s", e.what());
  } catch (...) {
    LOG_ERROR("mgpuWriteBufferAt: unknown exception");
  }
}

// ── Batched staging uploads (see minigpu.h for the contract) ───────────────
int mgpuUploadsSupported(void) { return 1; }

int mgpuUploadStats(long long *out, int n) {
  if (!out || n < 10) return 0;
  const mgpu::MGPU::UploadStats &s = minigpu.uploadStats;
  out[0] = s.usStage;
  out[1] = s.usLock;
  out[2] = s.usWrite;
  out[3] = s.usCopy;
  out[4] = s.usFlush;
  out[5] = s.ranges;
  out[6] = s.copies;
  out[7] = s.scopes;
  out[8] = s.flushes;
  out[9] = s.bytes;
  return 1;
}

int mgpuBeginUploads(void) {
  try {
    return minigpu.beginUploads();
  } catch (...) {
    return 0; // the scope could not be opened (out of memory)
  }
}

void mgpuStageReserve(size_t byteSize) {
  try {
    minigpu.reserveUploads(byteSize);
  } catch (...) {
    // Only a hint: the arena grows on demand instead.
  }
}

int mgpuEndUploads(void) {
  try {
    return minigpu.endUploads();
  } catch (...) {
    LOG_ERROR("mgpuEndUploads: exception emitting the uploads");
    return -1;
  }
}

int mgpuContextBeginUploads(MGPUContextHandle *handle) {
  auto owner = mgpu::retainHandleOwner(handle);
  if (!owner) return 0;
  try {
    return owner->gpu().beginUploads();
  } catch (...) {
    return 0;
  }
}

int mgpuContextEndUploads(MGPUContextHandle *handle) {
  auto owner = mgpu::retainHandleOwner(handle);
  if (!owner) return 0;
  try {
    return owner->gpu().endUploads();
  } catch (...) {
    LOG_ERROR("mgpuContextEndUploads: exception emitting the uploads");
    return -1;
  }
}

int mgpuStageWrite(MGPUBuffer *buffer, size_t dstByteOffset,
                   const void *inputData, size_t byteSize) {
  if (!buffer || !inputData || byteSize == 0) return -1;
  auto *buf = reinterpret_cast<mgpu::Buffer *>(buffer);
  try {
    const int r = buf->getMGPU().stageWrite(buf->getWGPUBuffer(), dstByteOffset,
                                            buf->getSize(), inputData, byteSize);
    // No scope open on this buffer's context, or the range is not 4-byte
    // aligned: behave exactly like the historical inline write. A write that
    // could not happen is -1, like bad arguments.
    if (r == 0 && !buf->tryWriteBytesAt(inputData, byteSize, dstByteOffset))
      return -1;
    return r;
  } catch (...) {
    LOG_ERROR("mgpuStageWrite: exception staging the write");
    return -1;
  }
}

// ── Batched readbacks (see minigpu.h for the contract) ─────────────────────
int mgpuReadbacksSupported(void) { return 1; }

int mgpuReadbackStats(long long *out, int n) {
  if (!out || n < 12) return 0;
  const mgpu::MGPU::ReadbackStats &s = minigpu.readbackStats;
  out[0] = s.usStage;
  out[1] = s.usLock;
  out[2] = s.usCopy;
  out[3] = s.usSubmit;
  out[4] = s.usMap;
  out[5] = s.usOut;
  out[6] = s.reads;
  out[7] = s.copies;
  out[8] = s.scopes;
  out[9] = s.bytes;
  out[10] = s.grows;
  out[11] = s.fallbacks;
  return 1;
}

int mgpuBeginReadbacks(void) {
  try {
    return minigpu.beginReadbacks();
  } catch (...) {
    return 0; // the scope could not be opened (out of memory)
  }
}

void mgpuReadbackReserve(size_t byteSize) {
  try {
    minigpu.reserveReadbacks(byteSize);
  } catch (...) {
    // Only a hint: the readback buffer grows on demand instead.
  }
}

int mgpuEndReadbacks(void) {
  try {
    return minigpu.endReadbacks();
  } catch (...) {
    LOG_ERROR("mgpuEndReadbacks: exception resolving the readbacks");
    return -1;
  }
}

int mgpuContextBeginReadbacks(MGPUContextHandle *handle) {
  auto owner = mgpu::retainHandleOwner(handle);
  if (!owner) return 0;
  try {
    return owner->gpu().beginReadbacks();
  } catch (...) {
    return 0;
  }
}

int mgpuContextEndReadbacks(MGPUContextHandle *handle) {
  auto owner = mgpu::retainHandleOwner(handle);
  if (!owner) return 0;
  try {
    return owner->gpu().endReadbacks();
  } catch (...) {
    LOG_ERROR("mgpuContextEndReadbacks: exception resolving the readbacks");
    return -1;
  }
}

int mgpuStageRead(MGPUBuffer *buffer, size_t srcByteOffset, void *dst,
                  size_t byteSize) {
  if (!buffer || !dst || byteSize == 0) return -1;
  auto *buf = reinterpret_cast<mgpu::Buffer *>(buffer);
  try {
    const int r = buf->getMGPU().stageRead(buf->getWGPUBuffer(), srcByteOffset,
                                           buf->getSize(), dst, byteSize);
    // No scope open on this buffer's context, or the range is not 4-byte
    // aligned: behave exactly like the historical inline read, and fill the
    // destination NOW (the caller cannot tell which path it got). A read that
    // failed is -1, never a destination silently left as it was.
    if (r == 0 && !buf->tryReadBytesAt(dst, byteSize, srcByteOffset))
      return -1;
    return r;
  } catch (...) {
    LOG_ERROR("mgpuStageRead: exception staging the read");
    return -1;
  }
}

void mgpuWriteUint64(MGPUBuffer *buffer, const uint64_t *inputData,
                             size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(uint64_t);
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
    } catch (...) {
      LOG_ERROR("mgpuWriteUint64: exception writing the buffer");
    }
  }
}

void mgpuWriteDouble(MGPUBuffer *buffer, const double *inputData,
                             size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(double);
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
    } catch (...) {
      LOG_ERROR("mgpuWriteDouble: exception writing the buffer");
    }
  }
}

// Enumerates hardware (non-software) adapters: writes up to [cap] entries —
// namesOut is cap*128 bytes of UTF-8 NUL-terminated names, totalOut/usedOut
// are dedicated-VRAM totals and current usage per adapter.  Returns the
// number of hardware adapters found (0 on non-Windows).
int mgpuEnumAdapters(char *namesOut, int64_t *totalOut, int64_t *usedOut,
                     int cap) {
#ifdef _WIN32
  IDXGIFactory1 *pFactory = nullptr;
  HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&pFactory);
  if (FAILED(hr) || !pFactory) return 0;

  int count = 0;
  for (UINT adapterIdx = 0;; adapterIdx++) {
    IDXGIAdapter1 *pAdapter1 = nullptr;
    // Any failure ends the walk, not only NOT_FOUND (a removed device or an
    // out-of-memory result leaves pAdapter1 null).
    if (FAILED(pFactory->EnumAdapters1(adapterIdx, &pAdapter1)) || !pAdapter1)
      break;

    DXGI_ADAPTER_DESC1 desc = {};
    if (FAILED(pAdapter1->GetDesc1(&desc)) ||
        (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
      pAdapter1->Release();
      continue;
    }

    if (count < cap) {
      if (namesOut) {
        char *slot = namesOut + count * 128;
        int n = WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, slot,
                                    127, nullptr, nullptr);
        slot[n > 0 ? n : 0] = '\0';
      }
      if (totalOut) {
        totalOut[count] = static_cast<int64_t>(desc.DedicatedVideoMemory);
      }
      if (usedOut) {
        usedOut[count] = -1;
        IDXGIAdapter3 *pAdapter3 = nullptr;
        if (SUCCEEDED(pAdapter1->QueryInterface(__uuidof(IDXGIAdapter3),
                                                (void **)&pAdapter3)) &&
            pAdapter3) {
          DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
          if (SUCCEEDED(pAdapter3->QueryVideoMemoryInfo(
                  0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
            usedOut[count] = static_cast<int64_t>(info.CurrentUsage);
          }
          pAdapter3->Release();
        }
      }
    }
    count++;
    pAdapter1->Release();
  }

  pFactory->Release();
  return count;
#else
  (void)namesOut; (void)totalOut; (void)usedOut; (void)cap;
  return 0;
#endif
}

// Returns the current dedicated VRAM usage in bytes for the first non-software
// GPU adapter, using DXGI1_4 QueryVideoMemoryInfo.  Returns -1 on platforms
// where this query is unsupported or fails.
int64_t mgpuQueryVramBytes() {
#ifdef _WIN32
  IDXGIFactory1 *pFactory = nullptr;
  HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&pFactory);
  if (FAILED(hr) || !pFactory) return -1;

  int64_t result = -1;
  for (UINT adapterIdx = 0;; adapterIdx++) {
    IDXGIAdapter1 *pAdapter1 = nullptr;
    if (FAILED(pFactory->EnumAdapters1(adapterIdx, &pAdapter1)) || !pAdapter1)
      break;

    DXGI_ADAPTER_DESC1 desc = {};
    // Skip Microsoft Basic Render Driver (software fallback), and an adapter
    // whose description cannot be read.
    if (FAILED(pAdapter1->GetDesc1(&desc)) ||
        (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
      pAdapter1->Release();
      continue;
    }

    IDXGIAdapter3 *pAdapter3 = nullptr;
    hr = pAdapter1->QueryInterface(__uuidof(IDXGIAdapter3),
                                   (void **)&pAdapter3);
    pAdapter1->Release();

    if (SUCCEEDED(hr) && pAdapter3) {
      DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
      hr = pAdapter3->QueryVideoMemoryInfo(
          0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info);
      pAdapter3->Release();
      if (SUCCEEDED(hr)) {
        result = static_cast<int64_t>(info.CurrentUsage);
      }
      break; // use first non-software adapter
    }
  }

  pFactory->Release();
  return result;
#else
  return -1;
#endif
}

#ifdef __cplusplus
}
#endif // extern "C"
