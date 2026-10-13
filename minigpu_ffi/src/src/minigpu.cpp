#include "../include/minigpu.h"
#include "../include/minigpu_context_owner.h"
#include "../include/dart_port.h"
#include "../include/log.h"
#include <atomic>
#include <chrono>
#include <exception>
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
static Dart_Port g_log_port = ILLEGAL_PORT;
static bool g_dart_api_ready = false;
#endif

bool logPostToPort(int level, const char *message, size_t len) {
#ifdef MINIGPU_HAVE_DART_DL
  const Dart_Port port = g_log_port;
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
  g_log_port = static_cast<Dart_Port>(port);
#else
  (void)port;
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
struct SubmittedWorkWait {
  int64_t port = 0;
  int64_t token = 0;
  std::atomic<bool> completed{false};
};

void finishSubmittedWorkWait(WGPUQueueWorkDoneStatus status, WGPUStringView,
                             void *userdata1, void *) {
  auto *wait = static_cast<SubmittedWorkWait *>(userdata1);
  const bool ok = status == WGPUQueueWorkDoneStatus_Success;
  mgpu::completionPost(wait->port, wait->token, ok);
  wait->completed.store(true, std::memory_order_release);
}

void rejectSubmittedWorkWait(int64_t port, int64_t token) {
  mgpu::completionPost(port, token, false);
}

void waitForSubmittedWork(mgpu::MGPU *gpu, int64_t port, int64_t token,
                         std::shared_ptr<mgpu::ContextPin> pin = {}) {
  if (!gpu) {
    rejectSubmittedWorkWait(port, token);
    return;
  }

  gpu->getWebGPUThread().enqueueAsync([gpu, port, token, pin]() {
    WGPUQueue queue = nullptr;
    WGPUInstance instance = nullptr;
    {
      mgpu::lock_guard<mgpu::mutex> lock(gpu->getGpuMutex());
      queue = pin ? pin->queue() : gpu->tryGetQueue();
      instance = pin ? pin->instance() : gpu->tryGetInstance();
      if (!queue || !instance) {
        rejectSubmittedWorkWait(port, token);
        return;
      }
      // Fire-and-forget dispatches remain in an open compute batch until a
      // queue boundary. Submit them before this callback becomes the fence.
      if (!pin || gpu->matchesReadyDevice(pin->device(), pin->deviceGeneration()))
        gpu->flushBatchLocked();
    }

    auto *wait = new SubmittedWorkWait{port, token};
    WGPUQueueWorkDoneCallbackInfo info{};
    info.mode = WGPUCallbackMode_AllowProcessEvents;
    info.callback = finishSubmittedWorkWait;
    info.userdata1 = wait;
    wgpuQueueOnSubmittedWorkDone(queue, info);

    // Dawn invokes this callback while its instance is pumped. This whole
    // loop is on this context's WebGPU worker, never the Dart UI thread.
    const auto spinUntil = std::chrono::steady_clock::now() +
                           std::chrono::milliseconds(8);
    while (!wait->completed.load(std::memory_order_acquire)) {
      wgpuInstanceProcessEvents(instance);
      if (!wait->completed.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() < spinUntil) {
          std::this_thread::yield();
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }
    }
    delete wait;
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

#ifdef __cplusplus
using namespace mgpu;
extern "C" {
#endif

MGPU minigpu;

mgpu::LogLevel level = mgpu::LOG_INFO;

void mgpuSetLogCallback(MGPULogCallback callback) {
  SET_LOG_CALLBACK(callback);
}

int mgpuInitDartApi(void *initialize_api_dl_data) {
  return (int)mgpu::logInitDartApi(initialize_api_dl_data);
}

void mgpuSetLogPort(int64_t port) { mgpu::logSetPort(port); }

void mgpuSetLogLevel(int lvl) {
  level = static_cast<mgpu::LogLevel>(lvl);
  SET_LOG_LEVEL(level);
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
  const std::string name = mgpu::selectedAdapterName();
  if (out && cap > 0) {
    const int n = (int)name.size() < cap - 1 ? (int)name.size() : cap - 1;
    memcpy(out, name.c_str(), (size_t)n);
    out[n] = '\0';
  }
  return (int)name.size();
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
  minigpu.attachContextAsync(callback);
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

MGPUComputeShader *mgpuCreateComputeShader() {
  return reinterpret_cast<MGPUComputeShader *>(
      new mgpu::ComputeShader(minigpu));
}

void mgpuDestroyComputeShader(MGPUComputeShader *shader) {
  if (!shader) return;
  // NOT an inline delete: binds are queued and capture the shader pointer, so
  // deleting here would free it under a pending bind. See destroyQueued().
  reinterpret_cast<mgpu::ComputeShader *>(shader)->destroyQueued();
}

void mgpuLoadKernel(MGPUComputeShader *shader, const char *kernelString) {
  if (!shader) {
    // Simple error - no LOG dependency
    return;
  }

  if (!kernelString || strlen(kernelString) == 0) {
    return;
  }

  reinterpret_cast<mgpu::ComputeShader *>(shader)->loadKernelString(
      kernelString);
}

int mgpuHasKernel(MGPUComputeShader *shader) {
  if (shader) {
    return reinterpret_cast<mgpu::ComputeShader *>(shader)->hasKernel();
  } else {
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

  BufferDataType mappedType = mapIntToBufferDataType(dataType);
  LOG_INFO("mappedType=%d", (int)mappedType);

  size_t elementSize = getElementSize(mappedType);
  size_t elementCount = byteSize / elementSize;

  auto *buf = new mgpu::Buffer(minigpu);
  try {
    if (needsPacking(mappedType)) {
      LOG_INFO("Creating packed buffer with byteSize=%d", byteSize);
      buf->createBuffer(static_cast<size_t>(byteSize), mappedType);
    } else {
      LOG_INFO("Creating direct buffer with byteSize=%zu (elementCount=%d * "
               "elementSize=%zu)",
               byteSize, elementCount, getElementSize(mappedType));
      buf->createBuffer(byteSize, mappedType);
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
    reinterpret_cast<mgpu::Buffer *>(buffer)->release();
    delete reinterpret_cast<mgpu::Buffer *>(buffer);
  }
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
  if (auto owner = mgpu::retainContextOwner(handle)) {
    owner->gpu().initializeContextAsync([owner, callback]() {
      if (callback) callback();
    });
  }
}

void mgpuContextInitializeAsyncToPort(MGPUContextHandle *handle, int64_t port,
                                      int64_t token) {
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
}

void mgpuContextWaitForSubmittedWorkAsyncToPort(MGPUContextHandle *handle,
                                                int64_t port, int64_t token) {
  std::shared_ptr<mgpu::ContextPin> pin;
  mgpu::retainContextPin(handle, pin);
  waitForSubmittedWork(pin ? &pin->gpu() : nullptr, port, token, pin);
}

void mgpuContextLeaseWaitForSubmittedWorkAsyncToPort(MGPUContextLease *lease,
                                                   int64_t port, int64_t token) {
  auto pin = mgpu::copyContextLeasePin(lease);
  waitForSubmittedWork(pin ? &pin->gpu() : nullptr, port, token, pin);
}

void mgpuContextLeaseRetireDeviceAsync(
    MGPUContextLease *lease, MGPUDeviceRetirementCallback callback,
    void *userdata) {
  retireCreationDevice(mgpu::copyContextLeasePin(lease),
      [callback, userdata](int result) {
        if (callback) callback(result, userdata);
      });
}

void mgpuContextLeaseRetireDeviceAsyncToPort(MGPUContextLease *lease,
                                           int64_t port, int64_t token) {
  retireCreationDevice(mgpu::copyContextLeasePin(lease),
      [port, token](int result) {
        mgpu::completionPost(port, token, result == MGPU_CONTEXT_LEASE_OK);
      });
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
  if (!handle) return 0;
  const std::string name =
      reinterpret_cast<mgpu::MGPU *>(handle)->adapterName();
  if (out && cap > 0) {
    const int n = (int)name.size() < cap - 1 ? (int)name.size() : cap - 1;
    memcpy(out, name.c_str(), (size_t)n);
    out[n] = '\0';
  }
  return (int)name.size();
}

MGPUBuffer *mgpuContextCreateBuffer(MGPUContextHandle *handle, int byteSize,
                                    int dataType) {
  if (!handle) return nullptr;
  auto &m = *reinterpret_cast<mgpu::MGPU *>(handle);
  BufferDataType mappedType = mapIntToBufferDataType(dataType);
  auto *buf = new mgpu::Buffer(m);
  try {
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
  if (!handle) return nullptr;
  return createBufferWithUsage(*reinterpret_cast<mgpu::MGPU *>(handle),
                               sizeBytes, usage);
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
  if (!handle) return nullptr;
  auto &m = *reinterpret_cast<mgpu::MGPU *>(handle);
  return reinterpret_cast<MGPUComputeShader *>(new mgpu::ComputeShader(m));
}

void mgpuSetBuffer(MGPUComputeShader *shader, int tag, MGPUBuffer *buffer) {
  if (shader && tag >= 0 && buffer) {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->setBuffer(
        tag, *reinterpret_cast<mgpu::Buffer *>(buffer));
  }
}

void mgpuSetBufferFire(MGPUComputeShader *shader, int tag,
                       MGPUBuffer *buffer) {
  if (shader && tag >= 0 && buffer) {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->setBufferQueued(
        tag, *reinterpret_cast<mgpu::Buffer *>(buffer));
  }
}

void mgpuDispatch(MGPUComputeShader *shader, int groupsX, int groupsY,
                  int groupsZ) {
  if (shader) {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->dispatch(groupsX, groupsY,
                                                              groupsZ);
  }
}

void mgpuDispatchAsync(MGPUComputeShader *shader, int groupsX, int groupsY,
                       int groupsZ, MGPUCallback callback) {
  if (shader) {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->dispatchAsync(
        groupsX, groupsY, groupsZ, callback);
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
  reinterpret_cast<mgpu::ComputeShader *>(shader)->dispatchAsync(
      groupsX, groupsY, groupsZ,
      [port, token]() { mgpu::completionPost(port, token, true); });
}

void mgpuReadAsyncToPort(MGPUBuffer *buffer, void *outputData,
                         size_t elementCount, size_t elementOffset,
                         int elementType, int64_t port, int64_t token) {
  if (!buffer || !outputData) {
    mgpu::completionPost(port, token, false);
    return;
  }
  auto *buf = reinterpret_cast<mgpu::Buffer *>(buffer);
  auto done = [port, token]() { mgpu::completionPost(port, token, true); };
  try {
    switch (static_cast<MGPUElementType>(elementType)) {
    case MGPU_ELEM_I8:
      buf->readAsync(static_cast<int8_t *>(outputData), elementCount,
                     elementOffset, done);
      break;
    case MGPU_ELEM_U8:
      buf->readAsync(static_cast<uint8_t *>(outputData), elementCount,
                     elementOffset, done);
      break;
    case MGPU_ELEM_I16:
      buf->readAsync(static_cast<int16_t *>(outputData), elementCount,
                     elementOffset, done);
      break;
    case MGPU_ELEM_U16:
      buf->readAsync(static_cast<uint16_t *>(outputData), elementCount,
                     elementOffset, done);
      break;
    case MGPU_ELEM_I32:
      buf->readAsync(static_cast<int32_t *>(outputData), elementCount,
                     elementOffset, done);
      break;
    case MGPU_ELEM_U32:
      buf->readAsync(static_cast<uint32_t *>(outputData), elementCount,
                     elementOffset, done);
      break;
    case MGPU_ELEM_I64:
      buf->readAsync(static_cast<int64_t *>(outputData), elementCount,
                     elementOffset, done);
      break;
    case MGPU_ELEM_U64:
      buf->readAsync(static_cast<uint64_t *>(outputData), elementCount,
                     elementOffset, done);
      break;
    case MGPU_ELEM_F32:
      buf->readAsync(static_cast<float *>(outputData), elementCount,
                     elementOffset, done);
      break;
    case MGPU_ELEM_F64:
      buf->readAsync(static_cast<double *>(outputData), elementCount,
                     elementOffset, done);
      break;
    default:
      LOG_ERROR("mgpuReadAsyncToPort: unknown element type %d", elementType);
      mgpu::completionPost(port, token, false);
      break;
    }
  } catch (...) {
    LOG_ERROR("mgpuReadAsyncToPort: exception issuing the read");
    mgpu::completionPost(port, token, false);
  }
}

void mgpuInitializeContextAsyncToPort(int64_t port, int64_t token) {
  minigpu.attachContextAsyncResult([port, token](bool attached) {
    mgpu::completionPost(port, token, attached);
  });
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
    WGPUDevice device = gpu.getDevice();
    if (!device) return 2;
    WGPULimits limits = {};
    if (wgpuDeviceGetLimits(device, &limits) != WGPUStatus_Success) return 3;
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

    nativeShader->setBufferRangeQueued(slot, handle,
                                       static_cast<size_t>(size),
                                       static_cast<size_t>(offset), bindingKind);
    return 0;
  } catch (...) {
    return 3;
  }
}

void mgpuWaitForSubmittedWorkAsyncToPort(int64_t port, int64_t token) {
  waitForSubmittedWork(&minigpu, port, token);
}

void mgpuFlushBatch() {
  minigpu.flushBatch();
}

void mgpuDrainWorkQueue() {
  auto &thread = minigpu.getWebGPUThread();
  // Called FROM the worker (a completion handler, say) the wait could never be
  // satisfied — the queue cannot drain while we are the thing draining it.
  if (std::this_thread::get_id() == thread.threadId()) return;
  thread.enqueueSync<int>([]() { return 0; });
}

void mgpuReadSync(MGPUBuffer *buffer, void *outputData, size_t size,
                        size_t offset) {
  if (buffer && outputData) {
    // Use float as default - could be improved with type info
    reinterpret_cast<mgpu::Buffer *>(buffer)->read(
        static_cast<float *>(outputData), size / sizeof(float), offset);
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
      // Handle error silently
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
      // Handle error silently
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
      // Handle error silently
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
      // Handle error silently
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
      // Handle error silently
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
      // Handle error silently
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
      // Handle error silently
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
      // Handle error silently
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
      // Handle error silently
    }
  }
}

void mgpuWriteFloat(MGPUBuffer *buffer, const float *inputData,
                            size_t byteSize) {
  LOG_INFO("mgpuWriteFloat: byteSize=%zu", byteSize);
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(float);
    LOG_INFO("Converting to elementCount=%zu", elementCount);
    reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
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
      // Handle error silently
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

void mgpuWriteInt8(MGPUBuffer *buffer, const int8_t *inputData,
                           size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(int8_t);
    reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
  }
}

void mgpuWriteInt16(MGPUBuffer *buffer, const int16_t *inputData,
                            size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(int16_t);
    reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
  }
}

void mgpuWriteInt32(MGPUBuffer *buffer, const int32_t *inputData,
                            size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(int32_t);
    reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
  }
}

void mgpuWriteInt64(MGPUBuffer *buffer, const int64_t *inputData,
                            size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(int64_t);
    reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
  }
}

void mgpuWriteUint8(MGPUBuffer *buffer, const uint8_t *inputData,
                            size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(uint8_t);
    reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
  }
}

void mgpuWriteUint16(MGPUBuffer *buffer, const uint16_t *inputData,
                             size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(uint16_t);
    reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
  }
}

void mgpuWriteUint32(MGPUBuffer *buffer, const uint32_t *inputData,
                             size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(uint32_t);
    reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
  }
}

void mgpuWriteBufferAt(MGPUBuffer *buffer, const void *inputData,
                       size_t byteSize, size_t dstByteOffset) {
  if (!buffer || !inputData) return;
  // MUST NOT THROW. writeBytesAt throws on an invalid context or an
  // out-of-range range, and letting a C++ exception unwind out of an FFI entry
  // point is undefined behaviour for any caller — it does not become a Dart
  // exception, it corrupts the frame it unwinds through. This entry point is
  // additionally bound as a LEAF call (see mgpuWriteBufferAtLeaf on the Dart
  // side), where the VM has not transitioned out of Dart state at all, so an
  // escaping exception is not merely undefined but reliably fatal. Log and
  // return: the same outcome the caller already got, minus the crash.
  try {
    reinterpret_cast<mgpu::Buffer *>(buffer)->writeBytesAt(inputData, byteSize,
                                                           dstByteOffset);
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

int mgpuBeginUploads(void) { return minigpu.beginUploads(); }

void mgpuStageReserve(size_t byteSize) { minigpu.reserveUploads(byteSize); }

int mgpuEndUploads(void) { return minigpu.endUploads(); }

int mgpuContextBeginUploads(MGPUContextHandle *handle) {
  if (!handle) return 0;
  return reinterpret_cast<mgpu::MGPU *>(handle)->beginUploads();
}

int mgpuContextEndUploads(MGPUContextHandle *handle) {
  if (!handle) return 0;
  return reinterpret_cast<mgpu::MGPU *>(handle)->endUploads();
}

int mgpuStageWrite(MGPUBuffer *buffer, size_t dstByteOffset,
                   const void *inputData, size_t byteSize) {
  if (!buffer || !inputData || byteSize == 0) return -1;
  auto *buf = reinterpret_cast<mgpu::Buffer *>(buffer);
  const int r = buf->getMGPU().stageWrite(buf->getWGPUBuffer(), dstByteOffset,
                                          buf->getSize(), inputData, byteSize);
  if (r == 0) {
    // No scope open on this buffer's context, or the range is not 4-byte
    // aligned: behave exactly like the historical inline write.
    buf->writeBytesAt(inputData, byteSize, dstByteOffset);
  }
  return r;
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

int mgpuBeginReadbacks(void) { return minigpu.beginReadbacks(); }

void mgpuReadbackReserve(size_t byteSize) {
  minigpu.reserveReadbacks(byteSize);
}

int mgpuEndReadbacks(void) { return minigpu.endReadbacks(); }

int mgpuContextBeginReadbacks(MGPUContextHandle *handle) {
  if (!handle) return 0;
  return reinterpret_cast<mgpu::MGPU *>(handle)->beginReadbacks();
}

int mgpuContextEndReadbacks(MGPUContextHandle *handle) {
  if (!handle) return 0;
  return reinterpret_cast<mgpu::MGPU *>(handle)->endReadbacks();
}

int mgpuStageRead(MGPUBuffer *buffer, size_t srcByteOffset, void *dst,
                  size_t byteSize) {
  if (!buffer || !dst || byteSize == 0) return -1;
  auto *buf = reinterpret_cast<mgpu::Buffer *>(buffer);
  const int r = buf->getMGPU().stageRead(buf->getWGPUBuffer(), srcByteOffset,
                                         buf->getSize(), dst, byteSize);
  if (r == 0) {
    // No scope open on this buffer's context, or the range is not 4-byte
    // aligned: behave exactly like the historical inline read, and fill the
    // destination NOW (the caller cannot tell which path it got).
    buf->readBytesAt(dst, byteSize, srcByteOffset);
  }
  return r;
}

void mgpuWriteUint64(MGPUBuffer *buffer, const uint64_t *inputData,
                             size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(uint64_t);
    reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
  }
}

void mgpuWriteDouble(MGPUBuffer *buffer, const double *inputData,
                             size_t byteSize) {
  if (buffer && inputData) {
    size_t elementCount = byteSize / sizeof(double);
    reinterpret_cast<mgpu::Buffer *>(buffer)->write(inputData, elementCount);
  }
}

void mgpuWriteAsyncFloat(MGPUBuffer *buffer, const float *data,
                             size_t byteSize, void (*callback)()) {
  if (buffer && data && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->write(
          data, byteSize / sizeof(float));
    } catch (...) {
      // Handle error silently
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
    if (pFactory->EnumAdapters1(adapterIdx, &pAdapter1) ==
        DXGI_ERROR_NOT_FOUND)
      break;

    DXGI_ADAPTER_DESC1 desc;
    pAdapter1->GetDesc1(&desc);
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
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
    if (pFactory->EnumAdapters1(adapterIdx, &pAdapter1) ==
        DXGI_ERROR_NOT_FOUND)
      break;

    DXGI_ADAPTER_DESC1 desc;
    pAdapter1->GetDesc1(&desc);

    // Skip Microsoft Basic Render Driver (software fallback)
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
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
