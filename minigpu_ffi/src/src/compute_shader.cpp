#include "../include/compute_shader.h"
#include "../include/minigpu_metadata_layout.h"
#include "../include/buffer.h"
#include "../include/log.h"
#include "../include/mutex.h"
#include "../include/shader_cache.h"
#include "../include/minigpu_context_owner.h"
#include "../include/minigpu_imported_frame_owner.h"
#include "../include/minigpu_private_resource_owner.h"
#include "../include/minigpu_capture_cleanup_owner.h"
#include "../include/minigpu_capture_work_owner.h"
#include "../include/minigpu_error_scope.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

enum LogLevel { kError = 0, kWarning = 1, kInfo = 2, kDebug = 3 };

const char *kDefLog = "ComputeShader";

namespace mgpu {

namespace {
std::atomic<uint32_t> liveShaderUses{0};
constexpr uint32_t maxShaderUses = 256;
bool chargeShaderUse() {
  auto count = liveShaderUses.load(std::memory_order_relaxed);
  while (count < maxShaderUses) {
    if (liveShaderUses.compare_exchange_weak(count, count + 1)) return true;
  }
  return false;
}

// Highest binding slot a shader accepts, checked before a binding table grows
// to slot + 1: a huge slot threw out of the extern "C" setters, or closed the
// shader for good on the queued path. Buffers, owned textures and samplers use
// the public 0..31 range (what mgpuSetBufferRange and the Dart API check); a
// texture view also takes 32..63, where pool outputs and imported frames bind.
constexpr int maxBufferSlot = 31;
constexpr int maxTextureViewSlot = 63;

// An over-limit dispatch is an invalid command, and one invalid command voids
// the whole shared batch (every other shader's dispatches, uploads and
// readbacks since the last flush), so it is refused before recording.
bool workgroupsWithinLimits(WGPUDevice device, int x, int y, int z) {
  // Queried once per device, not per dispatch: in the browser the query is a
  // JS call that reads every limit.
  thread_local WGPUDevice limitsDevice = nullptr;
  thread_local uint32_t most = 0;
  if (!device) return false;
  if (device != limitsDevice) {
    WGPULimits limits = {};
    if (wgpuDeviceGetLimits(device, &limits) != WGPUStatus_Success)
      return false;
    limitsDevice = device;
    most = limits.maxComputeWorkgroupsPerDimension;
  }
  return static_cast<uint32_t>(x) <= most && static_cast<uint32_t>(y) <= most &&
         static_cast<uint32_t>(z) <= most;
}

// The access a kernel declares for each @group(0) storage buffer binding. A
// layout entry must match its declaration exactly, so a buffer bound without
// a kind takes the declared access: a plain setBuffer on a
// `var<storage, read>` binding used to build no pipeline.
enum class DeclaredStorage : uint8_t { kUnknown, kRead, kReadWrite, kConflict };
using DeclaredStorageTable = std::array<DeclaredStorage, maxBufferSlot + 1>;

bool isWgslIdentChar(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_';
}

size_t skipWgslSpace(std::string_view code, size_t i) {
  while (i < code.size() && (code[i] == ' ' || code[i] == '\t' ||
                             code[i] == '\r' || code[i] == '\n')) {
    ++i;
  }
  return i;
}

std::string_view readWgslIdent(std::string_view code, size_t &i) {
  const size_t start = i;
  while (i < code.size() && isWgslIdentChar(code[i])) ++i;
  return code.substr(start, i - start);
}

// The decimal literal argument of the attribute whose name ends before [i],
// or -1 when it has none (an expression, say): such a binding stays unknown.
int readWgslAttributeInt(std::string_view code, size_t &i) {
  i = skipWgslSpace(code, i);
  if (i >= code.size() || code[i] != '(') return -1;
  i = skipWgslSpace(code, i + 1);
  int value = 0;
  size_t digits = 0;
  while (i < code.size() && code[i] >= '0' && code[i] <= '9' && digits < 9) {
    value = value * 10 + (code[i] - '0');
    ++i;
    ++digits;
  }
  if (digits == 0) return -1;
  if (i < code.size() && (code[i] == 'u' || code[i] == 'i')) ++i;
  i = skipWgslSpace(code, i);
  if (i < code.size() && code[i] == ',') i = skipWgslSpace(code, i + 1);
  if (i >= code.size() || code[i] != ')') return -1;
  ++i;
  return value;
}

// One pass over the module, skipping comments (block comments nest). Two
// declarations of one binding that disagree leave it unknown: only a full
// entry-point analysis could tell which one `main` uses.
DeclaredStorageTable declaredStorageAccess(std::string_view code) {
  DeclaredStorageTable table;
  table.fill(DeclaredStorage::kUnknown);
  int group = -1;
  int binding = -1;
  size_t i = 0;
  while (i < code.size()) {
    if (code.compare(i, 2, "//") == 0) {
      i = code.find('\n', i);
      if (i == std::string_view::npos) break;
      continue;
    }
    if (code.compare(i, 2, "/*") == 0) {
      int depth = 0;
      while (i < code.size()) {
        if (code.compare(i, 2, "/*") == 0) {
          ++depth;
          i += 2;
        } else if (code.compare(i, 2, "*/") == 0) {
          i += 2;
          if (--depth == 0) break;
        } else {
          ++i;
        }
      }
      continue;
    }
    const char c = code[i];
    if (c == ';' || c == '{' || c == '}') {
      group = binding = -1;
      ++i;
      continue;
    }
    if (c == '@') {
      i = skipWgslSpace(code, i + 1);
      const std::string_view name = readWgslIdent(code, i);
      if (name == "group") group = readWgslAttributeInt(code, i);
      if (name == "binding") binding = readWgslAttributeInt(code, i);
      continue;
    }
    if (!isWgslIdentChar(c)) {
      ++i;
      continue;
    }
    if (readWgslIdent(code, i) != "var") continue;
    const int declaredGroup = group;
    const int declaredBinding = binding;
    group = binding = -1;
    i = skipWgslSpace(code, i);
    if (i >= code.size() || code[i] != '<') continue;
    i = skipWgslSpace(code, i + 1);
    const std::string_view space = readWgslIdent(code, i);
    i = skipWgslSpace(code, i);
    std::string_view access;
    if (i < code.size() && code[i] == ',') {
      i = skipWgslSpace(code, i + 1);
      access = readWgslIdent(code, i);
      i = skipWgslSpace(code, i);
      if (i < code.size() && code[i] == ',') i = skipWgslSpace(code, i + 1);
    }
    if (i >= code.size() || code[i] != '>') continue;
    ++i;
    if (space != "storage" || declaredGroup != 0 || declaredBinding < 0 ||
        declaredBinding > maxBufferSlot) {
      continue;
    }
    // WGSL storage defaults to read.
    DeclaredStorage found;
    if (access.empty() || access == "read") {
      found = DeclaredStorage::kRead;
    } else if (access == "read_write") {
      found = DeclaredStorage::kReadWrite;
    } else {
      continue;
    }
    DeclaredStorage &slot = table[static_cast<size_t>(declaredBinding)];
    slot = slot == DeclaredStorage::kUnknown || slot == found
               ? found
               : DeclaredStorage::kConflict;
  }
  return table;
}

WGPUBufferBindingType storageBindingType(const DeclaredStorageTable &declared,
                                         size_t slot, BindingKind kind,
                                         bool accessFromKernel) {
  const DeclaredStorage access = accessFromKernel && slot < declared.size()
                                     ? declared[slot]
                                     : DeclaredStorage::kUnknown;
  if (access == DeclaredStorage::kRead)
    return WGPUBufferBindingType_ReadOnlyStorage;
  if (access == DeclaredStorage::kReadWrite)
    return WGPUBufferBindingType_Storage;
  return kind == BindingKind::kReadOnlyStorageBuffer
             ? WGPUBufferBindingType_ReadOnlyStorage
             : WGPUBufferBindingType_Storage;
}

#ifdef __EMSCRIPTEN__
// The browser answers a popped error scope through a promise, never inside the
// call, so the web build cannot gate recording on it and keeps relying on the
// device's uncaptured-error log.
class CreationErrorScope {
public:
  explicit CreationErrorScope(WGPUDevice) {}
  bool failed() { return false; }
};
#else
// Error scopes around the objects a shader creates, so an error pipeline or
// bind group is released instead of recorded into the shared batch. Dawn keeps
// scopes per thread (the GPU mutex keeps other work off this one meanwhile) and
// resolves a pop inside the call when nothing async is pending, which holds for
// synchronous creation: with AllowSpontaneous the answer arrives before
// PopErrorScope returns, so nothing is pumped under the GPU lock. An answer
// that has not arrived counts as success (the unscoped behavior), and a late
// callback matches no live ticket, so it never writes through a dead frame.
struct ScopeReceipt {
  uintptr_t first = 0;
  bool failed = false;
};
constexpr uintptr_t scopeFilterCount = 3;
thread_local ScopeReceipt *activeScopeReceipt = nullptr;
std::atomic<uintptr_t> nextScopeTicket{1};

void onCreationScopePopped(WGPUPopErrorScopeStatus status, WGPUErrorType type,
                           WGPUStringView message, void *ticket, void *) {
  if (status != WGPUPopErrorScopeStatus_Success || type == WGPUErrorType_NoError)
    return;
  const size_t length = !message.data ? 0
                        : message.length == WGPU_STRLEN ? std::strlen(message.data)
                                                         : message.length;
  LOG_ERROR("shader object creation failed: %.*s",
            static_cast<int>(std::min<size_t>(length, 4096)),
            message.data ? message.data : "");
  ScopeReceipt *receipt = activeScopeReceipt;
  const uintptr_t id = reinterpret_cast<uintptr_t>(ticket);
  if (receipt && id - receipt->first < scopeFilterCount) receipt->failed = true;
}

class CreationErrorScope {
public:
  explicit CreationErrorScope(WGPUDevice device) : device_(device) {
    if (!device_) return;
    wgpuDevicePushErrorScope(device_, WGPUErrorFilter_Validation);
    wgpuDevicePushErrorScope(device_, WGPUErrorFilter_OutOfMemory);
    wgpuDevicePushErrorScope(device_, WGPUErrorFilter_Internal);
  }
  ~CreationErrorScope() { failed(); }
  CreationErrorScope(const CreationErrorScope &) = delete;
  CreationErrorScope &operator=(const CreationErrorScope &) = delete;
  // Pops the scopes on the first call; true when Dawn reported an error.
  bool failed() {
    if (!device_) return failed_;
    ScopeReceipt receipt;
    receipt.first =
        nextScopeTicket.fetch_add(scopeFilterCount, std::memory_order_relaxed);
    activeScopeReceipt = &receipt;
    for (uintptr_t i = 0; i < scopeFilterCount; ++i) {
      WGPUPopErrorScopeCallbackInfo info = {};
      info.mode = WGPUCallbackMode_AllowSpontaneous;
      info.callback = onCreationScopePopped;
      info.userdata1 = reinterpret_cast<void *>(receipt.first + i);
      wgpuDevicePopErrorScope(device_, info);
    }
    activeScopeReceipt = nullptr;
    device_ = nullptr;
    failed_ = receipt.failed;
    return failed_;
  }

private:
  WGPUDevice device_;
  bool failed_ = false;
};

// The answer to a pipeline compiled with the GPU mutex released. It lives on
// the waiting worker's stack, and `done` is the callback's last write, so the
// spontaneous callback (inside the call or on a Dawn thread) never writes
// through a dead frame. Validation and backend failures arrive here, not in an
// error scope.
struct PipelineReceipt {
  std::atomic<bool> done{false};
  WGPUCreatePipelineAsyncStatus status =
      WGPUCreatePipelineAsyncStatus_CallbackCancelled;
  WGPUComputePipeline pipeline = nullptr;
};

void onPipelineCreated(WGPUCreatePipelineAsyncStatus status,
                       WGPUComputePipeline pipeline, WGPUStringView message,
                       void *userdata, void *) {
  if (status != WGPUCreatePipelineAsyncStatus_Success) {
    const size_t length = !message.data ? 0
                          : message.length == WGPU_STRLEN ? std::strlen(message.data)
                                                           : message.length;
    LOG_ERROR("shader object creation failed: %.*s",
              static_cast<int>(std::min<size_t>(length, 4096)),
              message.data ? message.data : "");
  }
  auto &receipt = *static_cast<PipelineReceipt *>(userdata);
  receipt.status = status;
  receipt.pipeline = pipeline;
  receipt.done.store(true, std::memory_order_release);
}
#endif
} // namespace

ShaderUseCustody::~ShaderUseCustody() {
  if (charged_) liveShaderUses.fetch_sub(1, std::memory_order_relaxed);
}

ComputeShader::ComputeShader(MGPU &mgpu)
    : mgpu(mgpu), creationDevice(mgpu.retainReadyDevice()) {}

ComputeShader::~ComputeShader() {
  if (privateResource_) {
    // Checked cleanup has already emptied the exact native fields. A legacy
    // destructor cannot turn an unresolved private owner into release proof.
    releasePrivateResource(privateResource_);
    return;
  }
  // safe to do immediately since it's on the right
  // thread
  auto cleanupTask = [computePipeline = this->computePipeline,
                      bindGroup = this->bindGroup,
                      bindGroupLayout = this->bindGroupLayout,
                      pipelineLayout = this->pipelineLayout,
                      shaderModule = this->shaderModule,
                      creationDevice = this->creationDevice,
                      retainedBindings = std::move(this->bindings),
                      bindGroupCache = std::move(this->bindGroupCache_),
                      bindGroupCacheDevice = this->bindGroupCacheDevice_,
                      pin = this->lifetimePin_]() {
    if (computePipeline) {
      wgpuComputePipelineRelease(computePipeline);
    }
    if (bindGroup) {
      wgpuBindGroupRelease(bindGroup);
    }
    releaseCachedBindGroups(bindGroupCache, bindGroupCacheDevice);
    if (bindGroupLayout) {
      wgpuBindGroupLayoutRelease(bindGroupLayout);
    }
    if (pipelineLayout) {
      wgpuPipelineLayoutRelease(pipelineLayout);
    }
    if (shaderModule) {
      wgpuShaderModuleRelease(shaderModule);
    }
    if (creationDevice) {
      wgpuDeviceRelease(creationDevice);
    }
  };

  if (std::this_thread::get_id() == mgpu.getWebGPUThread().threadId()) {
    cleanupTask();
    return;
  }
  try {
    mgpu.getWebGPUThread().enqueueAsync(cleanupTask);
  } catch (...) {
    // The queue is already closed, so release retained handles here instead.
    cleanupTask();
  }
}

void ComputeShader::cleanup() {
  // defer to destructor
}

void ComputeShader::destroyQueued() {
  mgpu::unique_lock<mgpu::mutex> gate(submissionMutex_);
  destroyRequested_ = true;
  if (exclusiveUse_) {
    exclusiveUse_->sealed = true;
    return;
  }
#ifdef __EMSCRIPTEN__
  // The browser queue executes inline, so never delete a locked mutex.
  gate.unlock();
  source_scope::MetadataDelete<ComputeShader>{}(this);
#else
  queueDeleteLocked();
#endif
}

void ComputeShader::queueDeleteLocked() {
  if (deleteQueued_) return;
  deleteQueued_ = true;
  ComputeShader *self = this;
  try {
    mgpu.getWebGPUThread().enqueueAsync([self, pin = lifetimePin_]() {
      // The admitting caller must finish unlocking this object's gate before
      // deletion. All preceding admitted operations are ahead of us on FIFO.
      { mgpu::lock_guard<mgpu::mutex> gate(self->submissionMutex_); }
      source_scope::MetadataDelete<ComputeShader>{}(self);
    });
  } catch (...) {
    // Allocation/enqueue failure is not proof of a drained worker. Retain the
    // object rather than deleting underneath possibly accepted work.
    deleteQueued_ = false;
    if (lifetimePin_) quarantineContextPin(lifetimePin_);
    LOG_ERROR("shader deletion retained: GPU enqueue failed");
  }
}

int ComputeShader::destroyCheckedAsync(
    std::shared_ptr<ContextPin> pin, MGPUResourceCleanupCallback callback,
    void *userdata) {
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  if (!pin || !callback) return MGPU_CONTEXT_LEASE_INVALID;
  capture_cleanup::Family family;
  const auto result = capture_cleanup::Family::begin(pin->sourceDomain(), family);
  if (result != source_scope::Result::Ok) return capture_cleanup::nativeResult(result);
  return destroyCheckedAsync(std::move(pin), callback, userdata, family);
#endif
}

int ComputeShader::destroyCheckedAsync(
    std::shared_ptr<ContextPin> pin, MGPUResourceCleanupCallback callback,
    void *userdata, const capture_cleanup::Family &family) {
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  if (!pin || !callback || !family.matches(pin->sourceDomain())) return MGPU_CONTEXT_LEASE_INVALID;
  if (pin->sourceDomain() && !privateResource_) return MGPU_CONTEXT_LEASE_INVALID;
  if (privateResource_ && (!samePrivatePin(*privateResource_, *pin) ||
      !privateResource_->ready.load() || privateResource_->quarantined.load()))
    return MGPU_CONTEXT_LEASE_QUARANTINED;
  capture_cleanup::PreparedTask job;
  const auto prepared = job.prepare(family, [shader = this, pin]() {
      // The admitting caller must finish unlocking before this FIFO deletion.
      { mgpu::lock_guard<mgpu::mutex> gate(shader->submissionMutex_); }
      if (shader->releaseCheckedOnWorker(*pin)) {
        source_scope::MetadataDelete<ComputeShader>{}(shader); // Empty native fields.
        return true;
      }
      return false;
  }, callback, userdata);
  if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared;

  mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
  if (&pin->gpu() != &mgpu || !creationDevice ||
      pin->device() != creationDevice)
    return MGPU_CONTEXT_LEASE_INVALID;
  if (exclusiveUse_ || destroyRequested_ || deleteQueued_)
    return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  {
    mgpu::lock_guard<mgpu::mutex> gpuLock(mgpu.getGpuMutex());
    for (const auto &binding : bindings) {
      if (binding.ownedTextureView || binding.ownedSampler)
        return MGPU_CONTEXT_LEASE_UNSUPPORTED;
    }
  }
  destroyRequested_ = true;
  deleteQueued_ = true;
  const auto accepted = job.enqueue([](void *context, mgpu::task_queue::Prepared &task) noexcept {
    return static_cast<MGPU *>(context)->getWebGPUThread().enqueuePrepared(task);
  }, &mgpu);
  if (accepted != MGPU_CONTEXT_LEASE_OK) {
    destroyRequested_ = false;
    deleteQueued_ = false;
  }
  return accepted;
#endif
}

bool ComputeShader::releaseCheckedOnWorker(const ContextPin &pin) {
  if (std::this_thread::get_id() != mgpu.getWebGPUThread().threadId() ||
      &pin.gpu() != &mgpu || pin.device() != creationDevice)
    return false;
  if ((pin.sourceDomain() && !privateResource_) ||
      (privateResource_ && !samePrivatePin(*privateResource_, pin))) return false;
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
  // A previously admitted legacy bind can execute after admission but before
  // this FIFO task. Recheck the narrow profile before releasing any reference.
  for (const auto &binding : bindings) {
    if (binding.ownedTextureView || binding.ownedSampler) return false;
  }
  if (bindGroup) { wgpuBindGroupRelease(bindGroup); bindGroup = nullptr; }
  purgeBindGroupCache(); // Its buffer references too.
  if (computePipeline) {
    wgpuComputePipelineRelease(computePipeline); computePipeline = nullptr;
  }
  if (bindGroupLayout) {
    wgpuBindGroupLayoutRelease(bindGroupLayout); bindGroupLayout = nullptr;
  }
  if (pipelineLayout) {
    wgpuPipelineLayoutRelease(pipelineLayout); pipelineLayout = nullptr;
  }
  if (shaderModule) { wgpuShaderModuleRelease(shaderModule); shaderModule = nullptr; }
  if (creationDevice) { wgpuDeviceRelease(creationDevice); creationDevice = nullptr; }
  bindings.clear();
  buffers.clear();
  if (privateResource_) privateResource_->nativeReleased = true;
  return true;
}

bool ComputeShader::admitLegacyLocked() {
  if (privateResource_) return false;
  if (exclusiveUse_) {
    interference_ = true;
    return false;
  }
  return !destroyRequested_ && !interference_;
}

bool ComputeShader::enqueueLegacy(std::function<void()> task) {
  mgpu::unique_lock<mgpu::mutex> gate(submissionMutex_);
  if (!admitLegacyLocked()) return false;
#ifdef __EMSCRIPTEN__
  // Exclusive native scopes are unsupported here. Preserve browser inline
  // execution/reentrant callback behavior without locking through the task.
  gate.unlock();
#endif
  try {
    mgpu.getWebGPUThread().enqueueAsync([this, task = std::move(task)]() {
      try {
        task();
      } catch (...) {
        // The admitted body's GPU lock has unwound before taking submission.
        mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
        interference_ = true;
        LOG_ERROR("shader operation failed; shader remains closed");
      }
    });
  } catch (...) {
    // Not queued (the worker has exited, or out of memory): refused, like a
    // closed shader, so no exception reaches an export.
    return false;
  }
  return true;
}

void ComputeShader::loadKernelString(const std::string &kernelString) {
  mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
  if (!admitLegacyLocked()) return;
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
  if (kernelString.empty() || shaderCode == kernelString) {
    return; // No change, skip
  }

  shaderCode = kernelString;
  moduleDirty = true;
  markLayoutDirty();
}

void ComputeShader::loadKernelFile(const std::string &path) {
  mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
  if (!admitLegacyLocked()) return;
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open kernel file: " + path);
  }

  std::string kernelString((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
  if (!kernelString.empty() && kernelString != shaderCode) {
    shaderCode = std::move(kernelString);
    moduleDirty = true;
    markLayoutDirty();
  }
}

bool ComputeShader::hasKernel() const {
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
  return !shaderCode.empty();
}

void ComputeShader::setBuffer(int tag, const Buffer &buffer) {
  mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
  if (!admitLegacyLocked()) return;
  if (tag < 0 || buffer.bufferData.buffer == nullptr) {
    return;
  }

  // Synchronize with dispatch/update on GPU thread
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());

  applyBufferBinding(tag, buffer.bufferData.buffer, buffer.bufferData.size, 0,
                     BindingKind::kStorageBuffer, true);
}

void ComputeShader::setBufferQueued(int tag, const Buffer &buffer) {
  // Capture the raw handle + size by value: the Buffer object itself may be
  // gone by the time the task runs, but buffer destruction is enqueued on
  // the same FIFO, so a handle captured before a queued destroy stays valid
  // for this task and any dispatch queued before that destroy.
  WGPUBuffer handle = buffer.bufferData.buffer;
  size_t size = buffer.bufferData.size;
  if (tag < 0 || handle == nullptr) {
    return;
  }
  enqueueLegacy([this, tag, handle, size]() {
    mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
    applyBufferBinding(tag, handle, size, 0, BindingKind::kStorageBuffer, true);
  });
}

bool ComputeShader::setBufferRangeQueued(int slot, WGPUBuffer buffer,
                                         size_t size, size_t offset,
                                         BindingKind kind) {
  return enqueueLegacy(
      [this, slot, buffer, size, offset, kind]() {
        mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
        applyBufferBinding(slot, buffer, size, offset, kind);
        // Explicit range binds are also the Dart wrapper's rebind boundary.
        // A destroyed WebGPU buffer may later be assigned the same raw handle,
        // so identical-looking metadata must still replace the bind group.
        bindingsDirty = true;
      });
}

void ComputeShader::applyBufferBinding(int slot, WGPUBuffer buffer, size_t size,
                                       size_t offset, BindingKind kind,
                                       bool accessFromKernel) {
  if (slot < 0 || slot > maxBufferSlot) {
    LOG_ERROR("buffer binding slot %d ignored: outside 0..%d", slot,
              maxBufferSlot);
    return;
  }
  if (slot >= static_cast<int>(buffers.size())) buffers.resize(slot + 1);
  if (slot >= static_cast<int>(bindings.size())) bindings.resize(slot + 1);

  const BindingEntry &current = bindings[slot];
  if (current.kind == kind && current.accessFromKernel == accessFromKernel &&
      current.buffer == buffer && current.size == size &&
      current.offset == offset) {
    return;
  }

  const bool layoutChanged = current.kind != kind ||
                             current.accessFromKernel != accessFromKernel;
  buffers[slot] = BufferBinding{buffer, size, offset};
  bindings[slot] = BindingEntry{kind, buffer, size, offset, nullptr};
  bindings[slot].accessFromKernel = accessFromKernel;
  bindingsDirty = true;
  if (layoutChanged) markLayoutDirty();
}

bool ComputeShader::setOwnedTextureQueued(
    int slot, std::shared_ptr<OwnedTextureViewState> view, BindingKind kind) {
  if (slot < 0 || slot > maxBufferSlot) return false;
  return enqueueLegacy([this, slot, view = std::move(view), kind]() {
    mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
    WGPUDevice currentDevice = mgpu.retainReadyDevice();
    const bool usable = currentDevice && creationDevice &&
                        currentDevice == creationDevice && view->liveOnWorker() &&
                        view->textureState->owner == &mgpu &&
                        view->textureState->creationDevice == creationDevice;
    if (currentDevice) wgpuDeviceRelease(currentDevice);
    if (!usable) return;
    if (slot >= static_cast<int>(buffers.size())) buffers.resize(slot + 1);
    if (slot >= static_cast<int>(bindings.size())) bindings.resize(slot + 1);
    const BindingEntry &current = bindings[slot];
    const WGPUTextureFormat format = view->textureState->format;
    const bool layoutChanged = current.kind != kind ||
                               current.viewDimension != view->dimension ||
                               current.textureFormat != format;
    BindingEntry next;
    next.kind = kind;
    next.view = view->view;
    next.viewDimension = view->dimension;
    next.textureFormat = format;
    next.ownedTextureView = view;
    bindings[slot] = std::move(next);
    buffers[slot] = BufferBinding{};
    bindingsDirty = true;
    if (layoutChanged) markLayoutDirty();
  });
}

bool ComputeShader::setOwnedSamplerQueued(
    int slot, std::shared_ptr<OwnedSamplerState> sampler, BindingKind kind) {
  if (slot < 0 || slot > maxBufferSlot) return false;
  return enqueueLegacy(
      [this, slot, sampler = std::move(sampler), kind]() {
        mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
        WGPUDevice currentDevice = mgpu.retainReadyDevice();
        const bool usable =
            currentDevice && creationDevice && currentDevice == creationDevice &&
            !sampler->destroyedOnWorker.load(std::memory_order_acquire) &&
            sampler->owner == &mgpu &&
            sampler->creationDevice == creationDevice;
        if (currentDevice) wgpuDeviceRelease(currentDevice);
        if (!usable) return;
        if (slot >= static_cast<int>(buffers.size())) buffers.resize(slot + 1);
        if (slot >= static_cast<int>(bindings.size())) bindings.resize(slot + 1);
        const bool layoutChanged = bindings[slot].kind != kind;
        BindingEntry next;
        next.kind = kind;
        next.sampler = sampler->sampler;
        next.ownedSampler = sampler;
        bindings[slot] = std::move(next);
        buffers[slot] = BufferBinding{};
        bindingsDirty = true;
        if (layoutChanged) markLayoutDirty();
      });
}

bool ComputeShader::ownedBindingsAlive() const {
  bool hasOwned = false;
  for (const BindingEntry &binding : bindings) {
    hasOwned = hasOwned || binding.ownedTextureView || binding.ownedSampler;
  }
  if (!hasOwned) return true;

  WGPUDevice currentDevice = mgpu.retainReadyDevice();
  if (!currentDevice) return false;
  bool valid = creationDevice && currentDevice == creationDevice;
  for (const BindingEntry &binding : bindings) {
    if (binding.ownedTextureView) {
      const auto &view = binding.ownedTextureView;
      // The worker-side flags: a destroy queued after this dispatch must not
      // drop it, and the texture is only destroyed once the dispatch recorded.
      valid = valid && view->liveOnWorker() &&
              view->textureState->creationDevice == creationDevice &&
              view->textureState->owner == &mgpu;
    }
    if (binding.ownedSampler) {
      const auto &sampler = binding.ownedSampler;
      valid = valid && !sampler->destroyedOnWorker.load(std::memory_order_acquire) &&
              sampler->creationDevice == creationDevice &&
              sampler->owner == &mgpu;
    }
  }
  wgpuDeviceRelease(currentDevice);
  return valid;
}

void ComputeShader::setTextureView(int slot, WGPUTextureView view) {
  mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
  if (!admitLegacyLocked()) return;
  if (slot < 0 || !view) return;
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
  applyTextureBinding(slot, view);
}

bool ComputeShader::setTextureViewQueued(int slot, WGPUTextureView view,
                                        std::shared_ptr<void> owner,
                                        std::function<void()> failure) {
  if (slot < 0 || !view) return false;
  return enqueueLegacy([this, slot, view, owner = std::move(owner), failure]() {
    bool failed = false;
    try {
      // The owner may release its view before this binding is replaced.
      wgpuTextureViewAddRef(view);
      std::shared_ptr<void> retained(
          view, [](WGPUTextureView v) { wgpuTextureViewRelease(v); });
      mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
      applyTextureBinding(slot, view, std::move(retained));
    } catch (...) {
      failed = true;
    }
    if (failed) {
      { mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
        interference_ = true; }
      if (failure) failure();
    }
  });
}

void ComputeShader::applyTextureBinding(int slot, WGPUTextureView view,
                                        std::shared_ptr<void> retained) {
  if (slot < 0 || slot > maxTextureViewSlot) {
    LOG_ERROR("texture binding slot %d ignored: outside 0..%d", slot,
              maxTextureViewSlot);
    return;
  }
  if (slot >= static_cast<int>(bindings.size())) bindings.resize(slot + 1);
  const bool layoutChanged = bindings[slot].kind != BindingKind::kTextureView ||
                             bindings[slot].viewDimension !=
                                 WGPUTextureViewDimension_2D;
  BindingEntry next;
  next.kind = BindingKind::kTextureView;
  next.view = view;
  next.viewDimension = WGPUTextureViewDimension_2D;
  next.retainedView = std::move(retained);
  bindings[slot] = std::move(next);
  bindingsDirty = true;
  if (layoutChanged) markLayoutDirty();
}

void ComputeShader::setStorageBuffer(int slot, WGPUBuffer buf,
                                     size_t size, size_t offset) {
  mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
  if (!admitLegacyLocked()) return;
  if (slot < 0 || !buf) return;
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
  applyBufferBinding(slot, buf, size, offset, BindingKind::kStorageBuffer);
}
bool ComputeShader::releaseTextureBindingQueued(int slot) {
  return enqueueLegacy([this, slot]() {
    mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
    if (slot < 0 || slot >= static_cast<int>(bindings.size())) return;
    bindings[slot].view = nullptr;
    bindings[slot].ownedTextureView.reset();
    bindings[slot].retainedView.reset();
    if (bindGroup) {
      wgpuBindGroupRelease(bindGroup);
      bindGroup = nullptr;
    }
    bindingsDirty = true;
  });
}

void ComputeShader::setUniformBuffer(int slot, WGPUBuffer buf, size_t size) {
  mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
  if (!admitLegacyLocked()) return;
  if (slot < 0 || !buf) return;
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
  applyBufferBinding(slot, buf, size, 0, BindingKind::kUniformBuffer);
}

WGPUDevice ComputeShader::pipelineDevice() const {
  // A retained shader may never repair/migrate its creation device. The legacy
  // path keeps its prior initialization behavior until acquisition succeeds.
  return lifetimePin_ ? creationDevice : mgpu.getDevice();
}

size_t ComputeShader::calculateBindingsHash() const {
  size_t hash = bindings.size() ^ buffers.size();
  if (!bindings.empty() && bindings[0].buffer) {
    hash ^= reinterpret_cast<size_t>(bindings[0].buffer);
  }
  return hash;
}

// First non-empty line of the WGSL, stripped of comment slashes and capped —
// kernels here name themselves in a leading comment. Backend errors quote the
// object label, so this is what turns "[ComputePipeline (unlabeled)] ...
// E_OUTOFMEMORY" in a console into the name of the kernel that caused it.
std::string_view ComputeShader::kernelLabel() const {
  size_t pos = 0;
  while (pos < shaderCode.size()) {
    size_t eol = shaderCode.find('\n', pos);
    if (eol == std::string::npos) eol = shaderCode.size();
    size_t b = pos, e = eol;
    while (b < e && (shaderCode[b] == ' ' || shaderCode[b] == '\t' ||
                     shaderCode[b] == '/' || shaderCode[b] == '\r')) {
      ++b;
    }
    while (e > b && (shaderCode[e - 1] == ' ' || shaderCode[e - 1] == '\t' ||
                     shaderCode[e - 1] == '\r')) {
      --e;
    }
    if (e > b) return shaderCode.substr(b, std::min<size_t>(e - b, 96));
    pos = eol + 1;
  }
  return "minigpu kernel";
}

bool ComputeShader::createShaderModule() {
  // Only the kernel text (or another device) invalidates the module; a layout
  // change reuses it.
  WGPUDevice device = pipelineDevice();
  if (!device) return false;
  if (shaderModule && !moduleDirty && moduleDevice == device) {
    return true; // Reuse existing module
  }

  if (shaderModule) {
    wgpuShaderModuleRelease(shaderModule);
    shaderModule = nullptr;
  }

  WGPUShaderSourceWGSL wgslDesc = {};
  wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgslDesc.code.data = shaderCode.c_str();
  wgslDesc.code.length = shaderCode.length();

  WGPUShaderModuleDescriptor shaderModuleDesc = {};
  shaderModuleDesc.nextInChain = &wgslDesc.chain;
  // Dawn copies descriptor strings during the create call, so a local is safe.
  const std::string_view label = kernelLabel();
  shaderModuleDesc.label.data = label.data();
  shaderModuleDesc.label.length = label.size();

  shaderModule = wgpuDeviceCreateShaderModule(device, &shaderModuleDesc);
  if (shaderModule) {
    moduleDirty = false;
    moduleDevice = device;
  }
  return shaderModule != nullptr;
}

bool ComputeShader::createBindGroupLayout() {
  // Only a pipeline rebuild calls this, and it always rebuilds: a buffer bound
  // without a kind takes its kernel's declared access, so a new kernel alone
  // can change the layout.
  if (bindGroupLayout) {
    wgpuBindGroupLayoutRelease(bindGroupLayout);
    bindGroupLayout = nullptr;
  }

  // Prefer the extended bindings vector; fall back to legacy buffers vector
  // when bindings is empty (existing code paths).
  const bool useExtended = !bindings.empty();

  capture_payload::Array<WGPUBindGroupLayoutEntry, capture_payload::bindingSlots> layoutEntries(bindings.scoped());

  // Capture-scoped kernels bind every kind explicitly.
  DeclaredStorageTable declared;
  declared.fill(DeclaredStorage::kUnknown);
  if (!shaderCode.scoped()) declared = declaredStorageAccess(shaderCode.view());

  if (useExtended) {
    layoutEntries.reserve(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
      const auto& b = bindings[i];
      WGPUBindGroupLayoutEntry entry = {};
      entry.binding    = static_cast<uint32_t>(i);
      entry.visibility = WGPUShaderStage_Compute;
      switch (b.kind) {
      case BindingKind::kStorageBuffer:
      case BindingKind::kReadOnlyStorageBuffer:
        entry.buffer.type =
            storageBindingType(declared, i, b.kind, b.accessFromKernel);
        entry.buffer.minBindingSize = 0;
        layoutEntries.push_back(entry);
        break;
      case BindingKind::kUniformBuffer:
        entry.buffer.type = WGPUBufferBindingType_Uniform;
        entry.buffer.minBindingSize = 0;
        layoutEntries.push_back(entry);
        break;
      case BindingKind::kTextureView:
        entry.texture.sampleType    = WGPUTextureSampleType_UnfilterableFloat;
        entry.texture.viewDimension = b.viewDimension;
        entry.texture.multisampled  = false;
        layoutEntries.push_back(entry);
        break;
      case BindingKind::kSampledTextureFloat:
      case BindingKind::kSampledTextureUnfilterable:
      case BindingKind::kDepthTexture:
        entry.texture.sampleType =
            b.kind == BindingKind::kSampledTextureFloat
                ? WGPUTextureSampleType_Float
                : (b.kind == BindingKind::kSampledTextureUnfilterable
                       ? WGPUTextureSampleType_UnfilterableFloat
                       : WGPUTextureSampleType_Depth);
        entry.texture.viewDimension = b.viewDimension;
        entry.texture.multisampled = false;
        layoutEntries.push_back(entry);
        break;
      case BindingKind::kStorageTextureWriteOnly:
        entry.storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
        entry.storageTexture.format = b.textureFormat;
        entry.storageTexture.viewDimension = b.viewDimension;
        layoutEntries.push_back(entry);
        break;
      case BindingKind::kFilteringSampler:
      case BindingKind::kNonFilteringSampler:
      case BindingKind::kComparisonSampler:
        entry.sampler.type =
            b.kind == BindingKind::kFilteringSampler
                ? WGPUSamplerBindingType_Filtering
                : (b.kind == BindingKind::kNonFilteringSampler
                       ? WGPUSamplerBindingType_NonFiltering
                       : WGPUSamplerBindingType_Comparison);
        layoutEntries.push_back(entry);
        break;
      default:
        break; // skip empty slots
      }
    }
  } else {
    // Legacy path: all bindings are storage buffers
    layoutEntries.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      if (buffers[i].buffer) {
        WGPUBindGroupLayoutEntry entry = {};
        entry.binding = static_cast<uint32_t>(i);
        entry.visibility = WGPUShaderStage_Compute;
        entry.buffer.type = storageBindingType(
            declared, i, BindingKind::kStorageBuffer, true);
        entry.buffer.minBindingSize = 0;
        layoutEntries.push_back(entry);
      }
    }
  }

  if (layoutEntries.empty()) return false;

  WGPULimits limits = {};
  WGPUDevice device = pipelineDevice();
  if (!device || wgpuDeviceGetLimits(device, &limits) != WGPUStatus_Success)
    return false;
  uint32_t sampledTextures = 0;
  uint32_t storageTextures = 0;
  uint32_t samplers = 0;
  uint32_t storageBuffers = 0;
  uint32_t uniformBuffers = 0;
  for (const BindingEntry &binding : bindings) {
    switch (binding.kind) {
    case BindingKind::kTextureView:
    case BindingKind::kSampledTextureFloat:
    case BindingKind::kSampledTextureUnfilterable:
    case BindingKind::kDepthTexture:
      ++sampledTextures;
      break;
    case BindingKind::kStorageTextureWriteOnly:
      ++storageTextures;
      break;
    case BindingKind::kFilteringSampler:
    case BindingKind::kNonFilteringSampler:
    case BindingKind::kComparisonSampler:
      ++samplers;
      break;
    case BindingKind::kStorageBuffer:
    case BindingKind::kReadOnlyStorageBuffer:
      ++storageBuffers;
      break;
    case BindingKind::kUniformBuffer:
      ++uniformBuffers;
      break;
    default:
      break;
    }
  }
  // The limit bounds every binding INDEX, not the entry count; entries are
  // pushed in slot order, so the last one carries the highest index.
  if (layoutEntries[layoutEntries.size() - 1].binding >=
          limits.maxBindingsPerBindGroup ||
      sampledTextures > limits.maxSampledTexturesPerShaderStage ||
      storageTextures > limits.maxStorageTexturesPerShaderStage ||
      samplers > limits.maxSamplersPerShaderStage ||
      storageBuffers > limits.maxStorageBuffersPerShaderStage ||
      uniformBuffers > limits.maxUniformBuffersPerShaderStage) {
    LOG_ERROR("bind group layout exceeds device binding limits");
    return false;
  }

  WGPUBindGroupLayoutDescriptor layoutDesc = {};
  layoutDesc.entryCount = static_cast<uint32_t>(layoutEntries.size());
  layoutDesc.entries    = layoutEntries.data();

  bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &layoutDesc);
  return bindGroupLayout != nullptr;
}

bool ComputeShader::createPipelineLayout() {
  if (pipelineLayout && !pipelineDirty) {
    return true; // Reuse existing layout
  }

  if (pipelineLayout) {
    wgpuPipelineLayoutRelease(pipelineLayout);
    pipelineLayout = nullptr;
  }

  WGPUPipelineLayoutDescriptor pipelineLayoutDesc = {};
  pipelineLayoutDesc.bindGroupLayoutCount = 1;
  pipelineLayoutDesc.bindGroupLayouts = &bindGroupLayout;

  pipelineLayout =
      wgpuDeviceCreatePipelineLayout(pipelineDevice(), &pipelineLayoutDesc);
  return pipelineLayout != nullptr;
}

bool ComputeShader::createComputePipeline() {
  if (computePipeline && !pipelineDirty) {
    return true; // Reuse existing pipeline
  }

  if (computePipeline) {
    wgpuComputePipelineRelease(computePipeline);
    computePipeline = nullptr;
  }

  WGPUComputePipelineDescriptor pipelineDesc = {};
  pipelineDesc.layout = pipelineLayout;
  pipelineDesc.compute.module = shaderModule;
  pipelineDesc.compute.entryPoint.data = "main";
  pipelineDesc.compute.entryPoint.length = 4;
  // Same label as the module: creation errors (a D3D12 E_OUTOFMEMORY at
  // pipeline-state creation, say) then NAME the kernel instead of printing
  // "[ComputePipeline (unlabeled)]". Note WebGPU returns an INVALID object,
  // not null, on creation failure — the nullptr check below does not catch
  // it; updatePipelineIfNeeded's error scope does (native), and the message
  // it logs quotes this label. On the web the later SetPipeline validation
  // error is still the only identification the console gets.
  const std::string_view label = kernelLabel();
  pipelineDesc.label.data = label.data();
  pipelineDesc.label.length = label.size();

  // This call is where WGSL becomes a backend shader, so it is the whole cost
  // the persistent shader cache exists to remove. Accumulating it makes the
  // difference measurable from Dart (Minigpu.shaderCacheStats) instead of
  // inferable from wall-clock, and a future regression in compile time shows
  // up here first.
  const auto pipelineT0 = std::chrono::steady_clock::now();
  computePipeline =
      wgpuDeviceCreateComputePipeline(pipelineDevice(), &pipelineDesc);
  shaderCacheNotePipelineMs(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - pipelineT0)
          .count());
  return computePipeline != nullptr;
}

bool ComputeShader::createBindGroup() {
  if (bindGroup && !bindingsDirty) {
    return true; // Reuse existing bind group
  }

  if (bindGroup) {
    wgpuBindGroupRelease(bindGroup);
    bindGroup = nullptr;
  }

  const bool useExtended = !bindings.empty();
  capture_payload::Array<WGPUBindGroupEntry, capture_payload::bindingSlots> bindGroupEntries(bindings.scoped());

  if (useExtended) {
    bindGroupEntries.reserve(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
      const auto& b = bindings[i];
      WGPUBindGroupEntry entry = {};
      entry.binding = static_cast<uint32_t>(i);
      switch (b.kind) {
      case BindingKind::kStorageBuffer:
      case BindingKind::kReadOnlyStorageBuffer:
      case BindingKind::kUniformBuffer:
        entry.buffer = b.buffer;
        entry.offset = b.offset;
        entry.size   = (b.size > 0) ? b.size : WGPU_WHOLE_SIZE;
        bindGroupEntries.push_back(entry);
        break;
      case BindingKind::kTextureView:
      case BindingKind::kSampledTextureFloat:
      case BindingKind::kSampledTextureUnfilterable:
      case BindingKind::kDepthTexture:
      case BindingKind::kStorageTextureWriteOnly:
        entry.textureView = b.view;
        bindGroupEntries.push_back(entry);
        break;
      case BindingKind::kFilteringSampler:
      case BindingKind::kNonFilteringSampler:
      case BindingKind::kComparisonSampler:
        entry.sampler = b.sampler;
        bindGroupEntries.push_back(entry);
        break;
      default:
        break;
      }
    }
  } else {
    bindGroupEntries.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      if (buffers[i].buffer) {
        WGPUBindGroupEntry entry = {};
        entry.binding = static_cast<uint32_t>(i);
        entry.buffer  = buffers[i].buffer;
        entry.offset  = 0;
        entry.size    = WGPU_WHOLE_SIZE;
        bindGroupEntries.push_back(entry);
      }
    }
  }

  if (bindGroupEntries.empty()) return false;

  WGPUBindGroupDescriptor bindGroupDesc = {};
  bindGroupDesc.layout     = bindGroupLayout;
  bindGroupDesc.entryCount = static_cast<uint32_t>(bindGroupEntries.size());
  bindGroupDesc.entries    = bindGroupEntries.data();

  bindGroup = wgpuDeviceCreateBindGroup(pipelineDevice(), &bindGroupDesc);
  return bindGroup != nullptr;
}

bool ComputeShader::updatePipelineIfNeeded() {
  //  only rebuild what's actually dirty
  if (pipelineDirty) {
    // Cached bind groups belong to the layout being replaced.
    purgeBindGroupCache();
    CreationErrorScope scope(pipelineDevice());
    if (!createShaderModule() || !createBindGroupLayout() ||
        !createPipelineLayout() || !createComputePipeline()) {
      return false;
    }
    if (scope.failed()) {
      // Never record an error pipeline: one invalid command voids the whole
      // shared batch. Nor recompile a known failure on every dispatch: a new
      // kernel or binding kind sets pipelineDirty again.
      wgpuComputePipelineRelease(computePipeline);
      computePipeline = nullptr;
      pipelineDirty = false;
      return false;
    }
    pipelineDirty = false;
    bindingsDirty = true; // Force bind group recreation since layout changed
  }
  if (!computePipeline) return false; // Failed creation above.

  if (bindingsDirty) {
    // A buffer-only set seen before (a ping-pong, the stale-alias rebind of
    // the same buffers) takes its cached group; anything else is created.
    const bool cacheable = cacheableBindGroupKey();
    if (cacheable && reuseCachedBindGroup()) {
      bindingsDirty = false;
      return true;
    }
    CreationErrorScope scope(pipelineDevice());
    if (!createBindGroup()) {
      return false;
    }
    if (scope.failed()) {
      // Same rule; the next changed bind sets bindingsDirty again.
      wgpuBindGroupRelease(bindGroup);
      bindGroup = nullptr;
      bindingsDirty = false;
      return false;
    }
    bindingsDirty = false;
    if (cacheable) rememberBindGroup();
  }

  return bindGroup != nullptr;
}

bool ComputeShader::updatePipelineIfNeeded(
    mgpu::unique_lock<mgpu::mutex> &lock) {
#ifndef __EMSCRIPTEN__
  if (pipelineDirty && !compilePipelineOffLock(lock)) return false;
#else
  (void)lock; // The browser has no worker to wait on: it compiles in place.
#endif
  return updatePipelineIfNeeded();
}

#ifndef __EMSCRIPTEN__
// The module and layouts are made under the GPU mutex as before; the backend
// compile, the slow part, runs with it released so other threads' GPU calls
// do not stall behind it. The worker waits for the answer in place: parking
// the task would let later FIFO work overtake this shader's dispatch. False
// only for a definite failure; true with pipelineDirty still set asks the
// caller to compile synchronously under the lock.
bool ComputeShader::compilePipelineOffLock(
    mgpu::unique_lock<mgpu::mutex> &lock) {
  WGPUDevice device = pipelineDevice();
  WGPUInstance instance =
      lifetimePin_ ? lifetimePin_->instance() : mgpu.tryGetInstance();
  if (!device || !instance || !lock.owns_lock() ||
      !mgpu.getWebGPUThread().onWorker())
    return true;
  // Cached bind groups belong to the layout being replaced.
  purgeBindGroupCache();
  {
    CreationErrorScope scope(device);
    if (!createShaderModule() || !createBindGroupLayout() ||
        !createPipelineLayout()) {
      return false;
    }
    if (scope.failed()) {
      // The updatePipelineIfNeeded rule: a known failure waits for a new
      // kernel or binding kind instead of recompiling on every dispatch.
      if (computePipeline) wgpuComputePipelineRelease(computePipeline);
      computePipeline = nullptr;
      pipelineDirty = false;
      return false;
    }
  }
  const uint64_t epoch = layoutEpoch_;
  // Dawn copies descriptor strings inside the call, which now runs while a
  // bind may replace the kernel text: copy the label out first.
  const std::string_view name = kernelLabel();
  char label[96];
  const size_t labelLength = std::min<size_t>(name.size(), sizeof(label));
  std::memcpy(label, name.data(), labelLength);
  WGPUComputePipelineDescriptor pipelineDesc = {};
  pipelineDesc.layout = pipelineLayout;
  pipelineDesc.compute.module = shaderModule;
  pipelineDesc.compute.entryPoint.data = "main";
  pipelineDesc.compute.entryPoint.length = 4;
  pipelineDesc.label.data = label;
  pipelineDesc.label.length = labelLength;
  // Own references for the unlocked stretch.
  wgpuDeviceAddRef(device);
  wgpuInstanceAddRef(instance);
  wgpuShaderModuleAddRef(pipelineDesc.compute.module);
  wgpuPipelineLayoutAddRef(pipelineDesc.layout);
  PipelineReceipt receipt;
  lock.unlock();
  WGPUCreateComputePipelineAsyncCallbackInfo info = {};
  info.mode = WGPUCallbackMode_AllowSpontaneous;
  info.callback = onPipelineCreated;
  info.userdata1 = &receipt;
  // Same accounting as createComputePipeline: this wait is the compile the
  // persistent shader cache exists to remove.
  const auto pipelineT0 = std::chrono::steady_clock::now();
  wgpuDeviceCreateComputePipelineAsync(device, &pipelineDesc, info);
  while (!receipt.done.load(std::memory_order_acquire)) {
    wgpuInstanceProcessEvents(instance);
    if (receipt.done.load(std::memory_order_acquire)) break;
    waitBackoffStep(std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - pipelineT0)
                        .count());
  }
  shaderCacheNotePipelineMs(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - pipelineT0)
          .count());
  wgpuPipelineLayoutRelease(pipelineDesc.layout);
  wgpuShaderModuleRelease(pipelineDesc.compute.module);
  wgpuInstanceRelease(instance);
  lock.lock();
  const bool stale = layoutEpoch_ != epoch || pipelineDevice() != device;
  wgpuDeviceRelease(device);
  if (stale ||
      receipt.status == WGPUCreatePipelineAsyncStatus_CallbackCancelled) {
    // A kernel, binding layout or device changed while unlocked, or no answer
    // came: compile again, under the lock.
    if (receipt.pipeline) wgpuComputePipelineRelease(receipt.pipeline);
    return true;
  }
  if (computePipeline) wgpuComputePipelineRelease(computePipeline);
  computePipeline = nullptr;
  pipelineDirty = false;
  if (receipt.status != WGPUCreatePipelineAsyncStatus_Success ||
      !receipt.pipeline) {
    // Never record an error pipeline; the failure is remembered as above.
    if (receipt.pipeline) wgpuComputePipelineRelease(receipt.pipeline);
    return false;
  }
  computePipeline = receipt.pipeline;
  bindingsDirty = true; // Force bind group recreation since layout changed
  return true;
}
#endif

void ComputeShader::markLayoutDirty() {
  pipelineDirty = true;
  ++layoutEpoch_;
}

// Fills bindGroupProbe_ from the current bindings. False when the set may not
// be cached: a texture or sampler, a capture-scoped shader, no memory.
bool ComputeShader::cacheableBindGroupKey() {
  bindGroupProbe_.clear();
  if (privateResource_ || bindings.scoped()) return false;
  try {
    if (!bindings.empty()) {
      for (size_t i = 0; i < bindings.size(); ++i) {
        const BindingEntry &b = bindings[i];
        switch (b.kind) {
        case BindingKind::kEmpty:
          break;
        case BindingKind::kStorageBuffer:
        case BindingKind::kReadOnlyStorageBuffer:
        case BindingKind::kUniformBuffer:
          if (!b.buffer) return false;
          bindGroupProbe_.push_back({static_cast<uint32_t>(i), b.buffer,
                                     b.offset,
                                     b.size > 0 ? b.size : WGPU_WHOLE_SIZE});
          break;
        default:
          return false;
        }
      }
    } else {
      for (size_t i = 0; i < buffers.size(); ++i) {
        if (buffers[i].buffer)
          bindGroupProbe_.push_back({static_cast<uint32_t>(i),
                                     buffers[i].buffer, 0, WGPU_WHOLE_SIZE});
      }
    }
  } catch (...) {
    return false;
  }
  return !bindGroupProbe_.empty();
}

// Makes the cached group for bindGroupProbe_ current. Every miss is counted.
bool ComputeShader::reuseCachedBindGroup() {
  if (!bindGroupCache_.empty() && bindGroupCacheDevice_ != pipelineDevice())
    purgeBindGroupCache();
  for (CachedBindGroup &entry : bindGroupCache_) {
    if (entry.epoch != layoutEpoch_ || entry.key != bindGroupProbe_) continue;
    entry.lastUse = ++bindGroupClock_;
    if (bindGroup != entry.group) {
      wgpuBindGroupAddRef(entry.group);
      if (bindGroup) wgpuBindGroupRelease(bindGroup);
      bindGroup = entry.group;
    }
    bindGroupHits_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  bindGroupMisses_.fetch_add(1, std::memory_order_relaxed);
  return false;
}

// Adds the bind group just created for bindGroupProbe_, evicting the least
// recently used entry when full.
void ComputeShader::rememberBindGroup() {
  WGPUDevice device = pipelineDevice();
  if (!bindGroup || !device || bindGroupProbe_.empty()) return;
  CachedBindGroup entry;
  try {
    bindGroupCache_.reserve(kBindGroupCacheEntries);
    entry.key = bindGroupProbe_;
  } catch (...) {
    return; // Uncached is still correct.
  }
  if (bindGroupCacheDevice_ != device) {
    purgeBindGroupCache();
    wgpuDeviceAddRef(device);
    bindGroupCacheDevice_ = device;
  }
  if (bindGroupCache_.size() >= kBindGroupCacheEntries) {
    auto oldest = std::min_element(
        bindGroupCache_.begin(), bindGroupCache_.end(),
        [](const CachedBindGroup &a, const CachedBindGroup &b) {
          return a.lastUse < b.lastUse;
        });
    wgpuBindGroupRelease(oldest->group);
    for (const auto &slot : oldest->key) wgpuBufferRelease(slot.buffer);
    bindGroupCache_.erase(oldest);
  }
  for (const auto &slot : entry.key) wgpuBufferAddRef(slot.buffer);
  wgpuBindGroupAddRef(bindGroup);
  entry.group = bindGroup;
  entry.epoch = layoutEpoch_;
  entry.lastUse = ++bindGroupClock_;
  bindGroupCache_.push_back(std::move(entry)); // Capacity reserved above.
}

void ComputeShader::purgeBindGroupCache() {
  releaseCachedBindGroups(bindGroupCache_, bindGroupCacheDevice_);
  bindGroupCache_.clear();
  bindGroupCacheDevice_ = nullptr;
}

void ComputeShader::releaseCachedBindGroups(
    const std::vector<CachedBindGroup> &cache, WGPUDevice device) {
  for (const CachedBindGroup &entry : cache) {
    if (entry.group) wgpuBindGroupRelease(entry.group);
    for (const auto &slot : entry.key) wgpuBufferRelease(slot.buffer);
  }
  if (device) wgpuDeviceRelease(device);
}

// Batch cap: bounds encoder memory and keeps a single submission far under
// the Windows TDR budget while still amortizing submit cost across hundreds
// of dispatches.
static constexpr int kMaxBatchedDispatches = 512;

void ComputeShader::dispatch(int groupsX, int groupsY, int groupsZ) {
  // Fire-and-forget: records into the context's shared batch pass (one
  // encoder + one submit per flush point instead of per dispatch).  WebGPU
  // guarantees sequential memory effects between dispatches in a pass, and
  // reads/writes/teardown flush the batch, so semantics match the previous
  // submit-per-dispatch behavior.
  auto dispatchTask = [this, groupsX, groupsY, groupsZ]() {
    // Ensure consistency with concurrent setBuffer/loadKernelString calls.
    mgpu::unique_lock<mgpu::mutex> lock(mgpu.getGpuMutex());

    if (lifetimePin_ && !lifetimePin_->current()) return;
    if (!ownedBindingsAlive()) {
      LOG_ERROR("dispatch skipped: owned texture or sampler is stale");
      return;
    }

    if (shaderCode.empty() || groupsX <= 0 || groupsY <= 0 || groupsZ <= 0) {
      LOG_ERROR("dispatch skipped: no kernel loaded or bad workgroup counts "
                "(%d,%d,%d)",
                groupsX, groupsY, groupsZ);
      return;
    }

    if (!workgroupsWithinLimits(pipelineDevice(), groupsX, groupsY, groupsZ)) {
      LOG_ERROR("dispatch skipped: workgroup counts (%d,%d,%d) exceed the "
                "device's maxComputeWorkgroupsPerDimension",
                groupsX, groupsY, groupsZ);
      return;
    }

    if (!updatePipelineIfNeeded(lock)) {
      LOG_ERROR("dispatch skipped: pipeline/bind group creation failed "
                "(output buffers were NOT written)");
      return;
    }
    // A compile runs with the GPU mutex released; the context may have
    // retired meanwhile.
    if (lifetimePin_ && !lifetimePin_->current()) return;

    if (!computePipeline || !bindGroup) {
      LOG_ERROR("dispatch skipped: pipeline or bind group missing "
                "(output buffers were NOT written)");
      return;
    }

    if (!mgpu.batchEncoder) {
      mgpu.batchEncoder =
          wgpuDeviceCreateCommandEncoder(pipelineDevice(), nullptr);
      if (!mgpu.batchEncoder) {
        return;
      }
    }
    if (!mgpu.batchPass) {
      mgpu.batchPass =
          wgpuCommandEncoderBeginComputePass(mgpu.batchEncoder, nullptr);
      if (!mgpu.batchPass) {
        return;
      }
    }

    wgpuComputePassEncoderSetPipeline(mgpu.batchPass, computePipeline);
    wgpuComputePassEncoderSetBindGroup(mgpu.batchPass, 0, bindGroup, 0,
                                       nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(
        mgpu.batchPass, static_cast<uint32_t>(groupsX),
        static_cast<uint32_t>(groupsY), static_cast<uint32_t>(groupsZ));

    if (++mgpu.batchCount >= kMaxBatchedDispatches) {
      mgpu.flushBatchLocked();
    }
  };

  enqueueLegacy(std::move(dispatchTask));
}

void ComputeShader::dispatchAsync(int groupsX, int groupsY, int groupsZ,
                                  std::function<void()> callback) {
  dispatchAsyncChecked(groupsX, groupsY, groupsZ, [callback](bool) {
    if (callback) callback();
  });
}

void ComputeShader::dispatchAsyncChecked(int groupsX, int groupsY, int groupsZ,
                                         std::function<void(bool)> callback) {
  bool accepted = false;
  try {
    accepted = enqueueLegacy([this, groupsX, groupsY, groupsZ, callback]() {
      bool ok = false;
      try {
        mgpu::unique_lock<mgpu::mutex> lock(mgpu.getGpuMutex());
        if (!lifetimePin_ || lifetimePin_->current()) {
          // Check arguments before selecting a legacy device: invalid/cold work
          // must not initialize a GPU merely to report failed submission.
          if (!shaderCode.empty() && groupsX > 0 && groupsY > 0 && groupsZ > 0) {
            auto device = pipelineDevice();
            ok = dispatchImmediateLocked(lock, groupsX, groupsY, groupsZ,
                                          device, mgpu.tryGetQueue());
          }
        }
      } catch (...) {
        LOG_ERROR("dispatchAsync failed before submission acknowledgement");
      }
      // Submission result, not GPU completion; always outside both locks.
      if (callback) callback(ok);
    });
  } catch (...) {
    // Queue insertion failed; enqueueAsync cannot throw after accepting work.
  }
  if (!accepted && callback) callback(false);
}

void ComputeShader::buildPipelineAsync(std::function<void(bool)> done) {
  bool accepted = false;
  try {
    accepted = enqueueLegacy([this, done]() {
      bool ok = false;
      try {
        mgpu::unique_lock<mgpu::mutex> lock(mgpu.getGpuMutex());
        // A dispatch's gates minus the workgroup counts; no kernel means no
        // device is selected, as for a dispatch.
        if ((!lifetimePin_ || lifetimePin_->current()) && !shaderCode.empty() &&
            ownedBindingsAlive()) {
          updatePipelineIfNeeded(lock);
          ok = !pipelineDirty && computePipeline;
        }
      } catch (...) {
        LOG_ERROR("pipeline build failed before acknowledgement");
      }
      if (done) done(ok); // Outside both locks.
    });
  } catch (...) {
    // Queue insertion failed; enqueueAsync cannot throw after accepting work.
  }
  if (!accepted && done) done(false);
}

bool ComputeShader::buildPipeline() {
#ifdef __EMSCRIPTEN__
  // The queued task has run inline by the time this returns.
  bool built = false;
  buildPipelineAsync([&built](bool ok) { built = ok; });
  return built;
#else
  if (mgpu.getWebGPUThread().onWorker()) return false;
  auto answer = std::make_shared<std::promise<bool>>();
  std::future<bool> built = answer->get_future();
  buildPipelineAsync([answer](bool ok) { answer->set_value(ok); });
  return built.get();
#endif
}

bool ComputeShader::dispatchImmediateLocked(
    mgpu::unique_lock<mgpu::mutex> &lock, int x, int y, int z,
    WGPUDevice device, WGPUQueue queue) {
  if (!device || !queue || !ownedBindingsAlive() || shaderCode.empty() ||
      x <= 0 || y <= 0 || z <= 0 || !workgroupsWithinLimits(device, x, y, z))
    return false;
  mgpu.flushBatchLocked();
  if (!updatePipelineIfNeeded(lock) || !computePipeline || !bindGroup)
    return false;
  // An exclusive scope never uses getDevice/getQueue. Recheck readiness after
  // potentially expensive pipeline compilation, immediately before encoding.
  if (lifetimePin_ && !lifetimePin_->current()) return false;
  auto encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
  if (!encoder) return false;
  auto pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
  if (!pass) {
    wgpuCommandEncoderRelease(encoder);
    return false;
  }
  wgpuComputePassEncoderSetPipeline(pass, computePipeline);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
  wgpuComputePassEncoderDispatchWorkgroups(pass, static_cast<uint32_t>(x),
                                          static_cast<uint32_t>(y),
                                          static_cast<uint32_t>(z));
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
  auto commands = wgpuCommandEncoderFinish(encoder, nullptr);
  wgpuCommandEncoderRelease(encoder);
  if (!commands) return false;
  const bool current = !lifetimePin_ || lifetimePin_->current();
  if (current) wgpuQueueSubmit(queue, 1, &commands);
  wgpuCommandBufferRelease(commands);
  return current;
}

int ComputeShader::acquireExclusiveUse(std::shared_ptr<ContextPin> pin,
                                      source_scope::MetadataUnique<MGPUComputeShaderUse> &out) {
  out.reset();
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  if (!pin) return MGPU_CONTEXT_LEASE_INVALID;
  if (pin->sourceDomain() && !privateResource_) return MGPU_CONTEXT_LEASE_INVALID;
  if (privateResource_ && (!samePrivatePin(*privateResource_, *pin) ||
      !privateResource_->ready.load() || !privateResource_->configured.load() ||
      privateResource_->quarantined.load())) return MGPU_CONTEXT_LEASE_QUARANTINED;
  if (&pin->gpu() != &mgpu || !creationDevice ||
      creationDevice != pin->device()) return MGPU_CONTEXT_LEASE_INVALID;
  if (!pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
  if (std::this_thread::get_id() == mgpu.getWebGPUThread().threadId())
    return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  capture_recipe::ResourceCustody role;
  auto admitted = role.reserveSourceRole(pin->sourceDomain(), source_scope::ResourceKind::ExclusiveUse);
  if (admitted != source_scope::Result::Ok) return capture_cleanup::nativeResult(admitted);
  source_scope::MetadataReservations resourceSlots;
  admitted = role.reserveResourceMetadata(resourceSlots);
  if (admitted != source_scope::Result::Ok) return capture_cleanup::nativeResult(admitted);
  capture_cleanup::Family family;
  admitted = capture_cleanup::Family::beginWork(pin->sourceDomain(), family);
  if (admitted != source_scope::Result::Ok) return capture_cleanup::nativeResult(admitted);
  capture_work::PreparedWork job;
  auto prepared = job.reserve(family);
  if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared;
  auto holder = source_scope::makeMetadataUnique<MGPUComputeShaderUse>(resourceSlots,
      source_scope::MetadataKind::ExclusiveUseBox);
  std::shared_ptr<ShaderExclusiveUse> use;
  if (!chargeShaderUse()) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  try {
    use = source_scope::makeMetadataShared<ShaderExclusiveUse>(resourceSlots,
        source_scope::MetadataKind::ExclusiveUseState, std::move(role));
  } catch (...) {
    liveShaderUses.fetch_sub(1, std::memory_order_relaxed);
    throw;
  }
  use->setGlobalUseCharge();
  use->shader = this;
  use->pin = std::move(pin);
  prepared = job.configureSynchronous([this, use] {
        using capture_work::Outcome;
        int result = MGPU_CONTEXT_LEASE_INTERNAL;
        try {
          mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
          mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
          if (destroyRequested_ || use->sealed || !use->pin->current()) {
            result = MGPU_CONTEXT_LEASE_RETIRED;
          } else if (interference_) {
            result = MGPU_CONTEXT_LEASE_WOULD_BLOCK;
          } else {
            mgpu.flushBatchLocked();
            lifetimePin_ = use->pin;
            use->phase = ShaderExclusiveUse::Phase::Active;
            result = MGPU_CONTEXT_LEASE_OK;
          }
        } catch (...) {
          use->phase = ShaderExclusiveUse::Phase::Quarantined;
          use->quarantineSourceRole();
          return Outcome::uncertain();
        }
        // An unsuccessful barrier never activated this use. Actual use/pin
        // members and closure copies still precede its final role refund.
        if (result != MGPU_CONTEXT_LEASE_OK) use->proveNativeReleased();
        return Outcome::done(result);
      });
  if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared;
  {
    // All fallible allocation/configuration precedes the submission gate. A
    // racing seal/destroy/use still wins at this final admission boundary.
    mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
    if (destroyRequested_ || interference_ || !use->pin->current())
      return MGPU_CONTEXT_LEASE_RETIRED;
    if (exclusiveUse_) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    prepared = capture_cleanup::nativeResult(use->acceptSourceRole());
    if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared;
    exclusiveUse_ = use;
    prepared = job.enqueue([](void *context, mgpu::task_queue::Prepared &task) noexcept {
      return static_cast<MGPU *>(context)->getWebGPUThread().enqueuePrepared(task);
    }, &mgpu);
    if (prepared != MGPU_CONTEXT_LEASE_OK) {
      exclusiveUse_.reset();
      use->proveNativeReleased(); // Known queue nonacceptance.
      return prepared;
    }
  }
  // Never wait while holding submission or GPU mutexes. No timeout releases an
  // activation whose worker might still be using the shader/context.
  int result;
  try { result = job.wait(); }
  catch (...) {
    use->quarantineSourceRole();
    try {
      mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
      use->sealed = true; use->phase = ShaderExclusiveUse::Phase::Quarantined;
    } catch (...) {}
    return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
  if (result != MGPU_CONTEXT_LEASE_OK) {
    mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
    if (result == MGPU_CONTEXT_LEASE_QUARANTINED) return result;
    exclusiveUse_.reset(); // Known unactivated barrier failure only.
    if (destroyRequested_) queueDeleteLocked();
    return result;
  }
  holder->state = std::move(use);
  out = std::move(holder);
  return MGPU_CONTEXT_LEASE_OK;
#endif
}

int ComputeShader::dispatchExclusiveUse(
    const std::shared_ptr<ShaderExclusiveUse> &use, int x, int y, int z) {
  if (!use || x <= 0 || y <= 0 || z <= 0) return MGPU_CONTEXT_LEASE_INVALID;
  capture_cleanup::Family family;
  capture_work::PreparedWork job;
  auto status = [&]() -> int {
    if (exclusiveUse_ != use || use->shader != this) return MGPU_CONTEXT_LEASE_INVALID;
    if (use->frameBindingPending) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    const auto groups = capture_work::validateDispatch(use->pin->sourceDomain(), x, y, z);
    if (groups != source_scope::Result::Ok) return capture_cleanup::nativeResult(groups);
    if (use->pin->sourceDomain() && !use->frame) return MGPU_CONTEXT_LEASE_INVALID;
    if (destroyRequested_ || use->sealed || use->phase != ShaderExclusiveUse::Phase::Active ||
        !use->pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
    return MGPU_CONTEXT_LEASE_OK;
  };
  {
    mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
    const auto result = status();
    if (result != MGPU_CONTEXT_LEASE_OK) return result;
  }
  const auto admitted = capture_cleanup::Family::beginWork(use->pin->sourceDomain(), family);
  if (admitted != source_scope::Result::Ok) return capture_cleanup::nativeResult(admitted);
  auto prepared = job.reserve(family);
  if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared;
  try {
    prepared = job.configure([this, use, x, y, z] {
      using capture_work::Outcome;
      // Seal shares this gate with the final readiness check and submission.
      mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
      // A later seal preserves previously accepted FIFO work. Destruction and
      // authority loss, however, must prevent work that has not submitted yet.
      if (!use->pin->current() ||
          use->phase == ShaderExclusiveUse::Phase::Quarantined) {
        use->phase = ShaderExclusiveUse::Phase::Quarantined;
        use->quarantineSourceRole();
        quarantineImportedFrame(use->frame);
        return Outcome::uncertain();
      }
      if (destroyRequested_) return Outcome::done(MGPU_CONTEXT_LEASE_RETIRED);
      bool ok = false;
      try {
        mgpu::unique_lock<mgpu::mutex> lock(mgpu.getGpuMutex());
        ok = dispatchImmediateLocked(lock, x, y, z, use->pin->device(),
                                      use->pin->queue());
      } catch (...) { }
      if (!ok) {
        use->sealed = true;
        use->phase = ShaderExclusiveUse::Phase::Quarantined;
        use->quarantineSourceRole();
        quarantineImportedFrame(use->frame);
        return Outcome::uncertain();
      }
      return Outcome::done();
    }, [](int) {});
    if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared;
    mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
    const auto current = status();
    if (current != MGPU_CONTEXT_LEASE_OK) return current;
    prepared = job.enqueue([](void *context, mgpu::task_queue::Prepared &task) noexcept {
      return static_cast<MGPU *>(context)->getWebGPUThread().enqueuePrepared(task);
    }, &mgpu);
    if (prepared == MGPU_CONTEXT_LEASE_OK) use->dispatchAdmitted = true;
  } catch (...) {
    // Actions/callbacks allocate before enqueue; queue itself proves refusal.
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
  if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared;
  return MGPU_CONTEXT_LEASE_OK;
}

int ComputeShader::bindImportedFrame(
    const std::shared_ptr<ShaderExclusiveUse> &use, uint32_t slot,
    const std::shared_ptr<ImportedFrameState> &frame) {
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  if (!use || !frame || slot >= 64) return MGPU_CONTEXT_LEASE_INVALID;
  if (use->pin->sourceDomain() && slot != capture_recipe_v1::frameSlot)
    return MGPU_CONTEXT_LEASE_INVALID;
  if (std::this_thread::get_id() == mgpu.getWebGPUThread().threadId())
    return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  if (!sameFrameCreation(*use->pin, *frame->pin)) return MGPU_CONTEXT_LEASE_INVALID;
  capture_cleanup::Family family;
  const auto admitted = capture_cleanup::Family::beginWork(use->pin->sourceDomain(), family);
  if (admitted != source_scope::Result::Ok) return capture_cleanup::nativeResult(admitted);
  capture_work::PreparedWork job;
  auto prepared = job.reserve(family);
  if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared;
  try {
    prepared = job.configureSynchronous([this, use, frame, slot] {
      int result = MGPU_CONTEXT_LEASE_OK;
      {
        mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
        try {
          std::lock_guard<std::mutex> frameLock(frame->mutex);
          // A later Retire preserves this previously accepted binding. It
          // queues behind us and closes its exact native access afterward.
          if (destroyRequested_ || !use->pin->current() || !frame->pin->current() ||
              frame->phase != MGPU_IMPORTED_FRAME_BINDING ||
              (use->phase != ShaderExclusiveUse::Phase::Active &&
               use->phase != ShaderExclusiveUse::Phase::Retiring)) {
            result = MGPU_CONTEXT_LEASE_QUARANTINED;
          } else {
            mgpu::lock_guard<mgpu::mutex> gpu(mgpu.getGpuMutex());
            applyTextureBinding(static_cast<int>(slot), frame->view);
            frame->phase = MGPU_IMPORTED_FRAME_ATTACHED;
          }
        } catch (...) {
          result = MGPU_CONTEXT_LEASE_QUARANTINED;
        }
        use->frameBindingPending = false;
        if (result != MGPU_CONTEXT_LEASE_OK) {
          use->sealed = true;
          use->phase = ShaderExclusiveUse::Phase::Quarantined;
          use->quarantineSourceRole();
          quarantineImportedFrame(frame);
        }
      }
      return result == MGPU_CONTEXT_LEASE_OK ? capture_work::Outcome::done()
                                            : capture_work::Outcome::uncertain();
    });
  } catch (...) {
    return MGPU_CONTEXT_LEASE_INTERNAL; // No owner phase changed yet.
  }
  if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared;
  {
    mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
    if (exclusiveUse_ != use || use->shader != this)
      return MGPU_CONTEXT_LEASE_INVALID;
    if (destroyRequested_ || use->sealed ||
        use->phase != ShaderExclusiveUse::Phase::Active || !use->pin->current())
      return MGPU_CONTEXT_LEASE_RETIRED;
    if (use->frame || use->frameBindingPending || use->dispatchAdmitted)
      return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    std::lock_guard<std::mutex> frameLock(frame->mutex);
    if (frame->phase == MGPU_IMPORTED_FRAME_QUARANTINED)
      return MGPU_CONTEXT_LEASE_QUARANTINED;
    if (frame->phase != MGPU_IMPORTED_FRAME_READY) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    if (!frame->pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
    if (!sameFrameCreation(*use->pin, *frame->pin) || !frame->view)
      return MGPU_CONTEXT_LEASE_INVALID;
    {
      mgpu::lock_guard<mgpu::mutex> gpu(mgpu.getGpuMutex());
      WGPULimits limits{};
      if (wgpuDeviceGetLimits(use->pin->device(), &limits) != WGPUStatus_Success)
        return MGPU_CONTEXT_LEASE_INTERNAL;
      if (slot >= limits.maxBindingsPerBindGroup) return MGPU_CONTEXT_LEASE_INVALID;
      if (slot < bindings.size()) {
        const auto &b = bindings[slot];
        if ((b.kind != BindingKind::kEmpty && b.kind != BindingKind::kTextureView) ||
            b.buffer || b.view || b.sampler || b.ownedTextureView || b.ownedSampler)
          return MGPU_CONTEXT_LEASE_INVALID;
      }
      if (slot < buffers.size() && buffers[slot].buffer)
        return MGPU_CONTEXT_LEASE_INVALID;
    }
    frame->phase = MGPU_IMPORTED_FRAME_BINDING;
    use->frame = frame;
    use->frameSlot = slot;
    use->frameBindingPending = true;
    prepared = job.enqueue([](void *context, mgpu::task_queue::Prepared &task) noexcept {
      return static_cast<MGPU *>(context)->getWebGPUThread().enqueuePrepared(task);
    }, &mgpu);
    if (prepared != MGPU_CONTEXT_LEASE_OK) {
      // Queue nonacceptance: restore without native binding mutation.
      frame->phase = MGPU_IMPORTED_FRAME_READY;
      use->frame.reset();
      use->frameBindingPending = false;
      return prepared;
    }
  }
  try { return job.wait(); } // Worker acknowledges after releasing locks.
  catch (...) {
    use->quarantineSourceRole(); frame->quarantineSourceRole();
    try {
      mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
      std::lock_guard<std::mutex> frameLock(frame->mutex);
      use->sealed = true; use->phase = ShaderExclusiveUse::Phase::Quarantined;
      frame->phase = MGPU_IMPORTED_FRAME_QUARANTINED;
      frame->result = MGPU_CONTEXT_LEASE_QUARANTINED;
    } catch (...) {}
    return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
#endif
}

void ComputeShader::retireExclusiveUse(
    const std::shared_ptr<ShaderExclusiveUse> &use,
    std::function<void(int)> callback) {
  if (!callback) return;
  capture_cleanup::Family family;
  const auto prepared = capture_cleanup::Family::begin(
      use ? use->pin->sourceDomain() : source_scope::SourceDomain{}, family);
  if (prepared != source_scope::Result::Ok) {
    callback(capture_cleanup::nativeResult(prepared)); return;
  }
  retireExclusiveUsePrepared(use,
      [callback = std::move(callback)](int result) { callback(result); }, family);
}
void ComputeShader::retireExclusiveUse(
    const std::shared_ptr<ShaderExclusiveUse> &use,
    MGPUQueueCompletionCallback callback, void *userdata,
    const capture_cleanup::Family &family) {
  if (!callback) return;
  retireExclusiveUsePrepared(use,
      [callback, userdata](int result) { callback(result, userdata); }, family);
}
template<class Callback>
void ComputeShader::retireExclusiveUsePrepared(
    const std::shared_ptr<ShaderExclusiveUse> &use,
    Callback callback, const capture_cleanup::Family &family) {
  capture_work::PreparedWork job;
  std::shared_ptr<capture_work::PreparedWait> wait;
  int immediate = MGPU_CONTEXT_LEASE_INTERNAL;
  bool enqueued = false, needsWork = false;
  auto status = [&]() -> int {
    if (!use || exclusiveUse_ != use || use->shader != this || !family.cleanup() ||
        !family.matches(use->pin->sourceDomain())) return MGPU_CONTEXT_LEASE_INVALID;
    if (use->phase == ShaderExclusiveUse::Phase::Retired) return MGPU_CONTEXT_LEASE_OK;
    if (use->phase == ShaderExclusiveUse::Phase::Retiring) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    if (use->phase == ShaderExclusiveUse::Phase::Quarantined) return MGPU_CONTEXT_LEASE_QUARANTINED;
    return use->phase == ShaderExclusiveUse::Phase::Active ? MGPU_CONTEXT_LEASE_OK
                                                         : MGPU_CONTEXT_LEASE_INTERNAL;
  };
  {
    mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
    immediate = status();
    needsWork = immediate == MGPU_CONTEXT_LEASE_OK && use->phase == ShaderExclusiveUse::Phase::Active;
  }
  if (needsWork) {
    try {
        // Allocate/copy callback captures on the admitting thread so allocation
        // failure cannot unwind out of the GPU worker without an acknowledgement.
        auto complete = [this, use, callback, family](int result) {
          // Called on the creation worker outside GPU and Dawn callbacks.
          if (result == MGPU_CONTEXT_LEASE_OK && !use->pin->current())
            result = MGPU_CONTEXT_LEASE_RETIRED;
          if (use->frame) {
            try {
              if (result == MGPU_CONTEXT_LEASE_OK)
                result = finishImportedFrameRetirement(use->frame);
              if (result != MGPU_CONTEXT_LEASE_OK) quarantineImportedFrame(use->frame);
            } catch (...) {
              quarantineImportedFrame(use->frame);
              result = MGPU_CONTEXT_LEASE_QUARANTINED;
            }
          }
          if (result == MGPU_CONTEXT_LEASE_OK) {
            // Release completed input owners before permitting token Release.
            use->retiringBindings.clear();
          }
          {
            mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
            use->phase = result == MGPU_CONTEXT_LEASE_OK
                ? ShaderExclusiveUse::Phase::Retired
                : ShaderExclusiveUse::Phase::Quarantined;
            if (result == MGPU_CONTEXT_LEASE_OK) use->proveNativeReleased();
            else { use->quarantineSourceRole(); family.quarantine(); }
          }
          callback(result);
        };
        immediate = job.reserve(family);
        if (immediate == MGPU_CONTEXT_LEASE_OK) {
          immediate = capture_work::PreparedWait::create(family, wait);
          if (immediate == MGPU_CONTEXT_LEASE_OK)
            immediate = preparePinnedSubmittedWorkWait(use->pin, family, complete, *wait);
        }
        if (immediate == MGPU_CONTEXT_LEASE_OK)
          immediate = job.configure([this, use, wait] {
          bool prepared = false;
          try {
            mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
            bool frameReady = !use->frame;
            if (use->frame) {
              std::lock_guard<std::mutex> frameLock(use->frame->mutex);
              frameReady = !use->frameBindingPending &&
                  use->frame->phase == MGPU_IMPORTED_FRAME_ATTACHED;
              if (frameReady) use->frame->phase = MGPU_IMPORTED_FRAME_RETIRING;
            }
            mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
            if (use->phase == ShaderExclusiveUse::Phase::Retiring &&
                use->pin->current() && frameReady) {
              // Hold owned inputs until completion even if dropping their
              // binding references would otherwise destroy the last owner.
              if (bindings.scoped()) {
                // Scoped inputs remain owned by bindingHolds and use->frame.
                // An unexpected generic owned alias cannot be silently dropped.
                for (const auto &binding : bindings)
                  if (binding.ownedTextureView || binding.ownedSampler)
                    return capture_work::Outcome::uncertain();
              } else use->retiringBindings = bindings.legacy();
              for (auto &binding : bindings) {
                binding.view = nullptr;
                binding.sampler = nullptr;
                binding.ownedTextureView.reset();
                binding.ownedSampler.reset();
                binding.retainedView.reset();
              }
              if (bindGroup) wgpuBindGroupRelease(bindGroup);
              bindGroup = nullptr;
              bindingsDirty = true;
              prepared = true;
            }
          } catch (...) { }
          if (!prepared) {
            return capture_work::Outcome::uncertain();
          }
          const auto queued = enqueuePinnedSubmittedWorkWait(use->pin, *wait);
          // Native bindings have been detached: even a proven nested enqueue
          // refusal is an accepted cleanup failure, never rollback/retry.
          return queued == MGPU_CONTEXT_LEASE_OK ? capture_work::Outcome::done()
                                                 : capture_work::Outcome::uncertain();
        }, [complete](int result) {
          if (result != MGPU_CONTEXT_LEASE_OK) complete(result);
        });
      if (immediate == MGPU_CONTEXT_LEASE_OK) {
        mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
        immediate = status();
        if (immediate == MGPU_CONTEXT_LEASE_OK && use->phase == ShaderExclusiveUse::Phase::Active) {
          const bool wasSealed = use->sealed;
          use->sealed = true;
          use->phase = ShaderExclusiveUse::Phase::Retiring;
          immediate = job.enqueue([](void *context, mgpu::task_queue::Prepared &task) noexcept {
            return static_cast<MGPU *>(context)->getWebGPUThread().enqueuePrepared(task);
          }, &mgpu);
          enqueued = immediate == MGPU_CONTEXT_LEASE_OK;
          if (!enqueued) { use->phase = ShaderExclusiveUse::Phase::Active; use->sealed = wasSealed; }
        }
      }
    } catch (...) {
      immediate = MGPU_CONTEXT_LEASE_INTERNAL;
    }
  }
  if (!enqueued) callback(immediate);
}

int ComputeShader::releaseExclusiveUse(
    const std::shared_ptr<ShaderExclusiveUse> &use) {
  mgpu::lock_guard<mgpu::mutex> gate(submissionMutex_);
  if (!use || exclusiveUse_ != use || use->shader != this)
    return MGPU_CONTEXT_LEASE_INVALID;
  if (use->phase != ShaderExclusiveUse::Phase::Retired)
    return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  exclusiveUse_.reset();
  if (destroyRequested_) queueDeleteLocked();
  return MGPU_CONTEXT_LEASE_OK;
}

} // namespace mgpu

namespace mgpu::allocation_layout {
void shaderUseMetadataLayout(MetadataLayoutBuilder &out) noexcept {
  out.shared<ShaderExclusiveUse>(MetadataKind::ExclusiveUseState);
  out.unique<MGPUComputeShaderUse>(MetadataKind::ExclusiveUseBox);
}
} // namespace mgpu::allocation_layout
