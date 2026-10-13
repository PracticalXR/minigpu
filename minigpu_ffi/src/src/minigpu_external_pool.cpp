#include "../include/minigpu_external_pool.h"
#include "../include/compute_shader.h"
#include "../include/minigpu_context_owner.h"
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <d3d11_4.h>
#include <dawn/native/D3D11Backend.h>
#include <dawn/native/D3D12Backend.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#endif

extern "C" int mgpuContextLeaseGetAdapterIdentity(
    MGPUContextLease *lease, MGPUDeviceAdapterIdentity *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  std::memset(out, 0, sizeof(*out));
  try {
    auto pin = mgpu::copyContextLeasePin(lease);
    if (!pin) return MGPU_CONTEXT_LEASE_INVALID;
    if (!pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
#ifdef _WIN32
    WGPUAdapterInfo info{};
    const auto status = wgpuDeviceGetAdapterInfo(pin->device(), &info);
    const auto backend = info.backendType;
    wgpuAdapterInfoFreeMembers(info);
    if (status != WGPUStatus_Success) return MGPU_CONTEXT_LEASE_NOT_READY;
    LUID luid{};
    uint32_t code;
    if (backend == WGPUBackendType_D3D11) {
      using Microsoft::WRL::ComPtr;
      auto device = dawn::native::d3d11::GetD3D11Device(pin->device());
      ComPtr<IDXGIDevice> dxgi;
      ComPtr<IDXGIAdapter> adapter;
      DXGI_ADAPTER_DESC desc{};
      if (!device || FAILED(device.As(&dxgi)) ||
          FAILED(dxgi->GetAdapter(&adapter)) || FAILED(adapter->GetDesc(&desc)))
        return MGPU_CONTEXT_LEASE_NOT_READY;
      luid = desc.AdapterLuid;
      code = 1;
    } else if (backend == WGPUBackendType_D3D12) {
      auto device = dawn::native::d3d12::GetD3D12Device(pin->device());
      if (!device) return MGPU_CONTEXT_LEASE_NOT_READY;
      luid = device->GetAdapterLuid();
      code = 2;
    } else {
      return MGPU_CONTEXT_LEASE_UNSUPPORTED;
    }
    if (!pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
    out->struct_size = sizeof(*out);
    out->version = 1;
    out->device_generation = pin->deviceGeneration();
    out->backend = code;
    out->kind = 1;
    out->low = luid.LowPart;
    out->high = static_cast<uint32_t>(luid.HighPart);
    return MGPU_CONTEXT_LEASE_OK;
#else
    return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#endif
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}

static int queryExternalPoolPin(std::shared_ptr<mgpu::ContextPin> pin,
                                  MGPUExternalPoolCapabilities *out,
                                  uint32_t size) {
  if (!out || size < sizeof(*out))
    return MGPU_CONTEXT_LEASE_INVALID;
  std::memset(out, 0, sizeof(*out));
  try {
    if (!pin) return MGPU_CONTEXT_LEASE_INVALID;
    if (!pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
    out->struct_size = sizeof(*out);
    out->version = 1;
    out->context_generation = pin->contextGeneration();
    out->device_generation = pin->deviceGeneration();
    out->refusal = MGPU_EXTERNAL_POOL_BACKEND_UNQUALIFIED;
    WGPUAdapterInfo adapter{};
    const auto status = wgpuDeviceGetAdapterInfo(pin->device(), &adapter);
    const auto backend = adapter.backendType;
    wgpuAdapterInfoFreeMembers(adapter);
    if (status != WGPUStatus_Success)
      return MGPU_CONTEXT_LEASE_NOT_READY;
    switch (backend) {
    case WGPUBackendType_D3D11:
      out->backend = MGPU_EXTERNAL_POOL_BACKEND_D3D11;
      break;
    case WGPUBackendType_D3D12:
      out->backend = MGPU_EXTERNAL_POOL_BACKEND_D3D12;
      break;
    case WGPUBackendType_Metal:
      out->backend = MGPU_EXTERNAL_POOL_BACKEND_METAL;
      break;
    case WGPUBackendType_Vulkan:
      out->backend = MGPU_EXTERNAL_POOL_BACKEND_VULKAN;
      break;
    default:
      out->backend = MGPU_EXTERNAL_POOL_BACKEND_OTHER;
      break;
    }
    WGPULimits limits{};
    if (wgpuDeviceGetLimits(pin->device(), &limits) != WGPUStatus_Success)
      return MGPU_CONTEXT_LEASE_NOT_READY;
    out->max_texture_dimension_2d = limits.maxTextureDimension2D;
#ifdef _WIN32
    // Dawn's D3D11 helper casts internally: only call after checking its
    // backend.
    if (backend == WGPUBackendType_D3D11) {
      using Microsoft::WRL::ComPtr;
      auto native = dawn::native::d3d11::GetD3D11Device(pin->device());
      ComPtr<IDXGIDevice> dxgi;
      ComPtr<IDXGIAdapter> actualAdapter;
      DXGI_ADAPTER_DESC descriptor{};
      if (!native || FAILED(native.As(&dxgi)) ||
          FAILED(dxgi->GetAdapter(&actualAdapter)) ||
          FAILED(actualAdapter->GetDesc(&descriptor)))
        return MGPU_CONTEXT_LEASE_NOT_READY;
      out->adapter_luid_low = descriptor.AdapterLuid.LowPart;
      out->adapter_luid_high = descriptor.AdapterLuid.HighPart;
      ComPtr<ID3D11Device5> device5;
      ComPtr<ID3D11DeviceContext> immediate;
      ComPtr<ID3D11DeviceContext4> context4;
      native->GetImmediateContext(&immediate);
      if (SUCCEEDED(native.As(&device5)) && immediate &&
          SUCCEEDED(immediate.As(&context4)))
        out->facility_flags |= MGPU_EXTERNAL_POOL_D3D11_DEVICE5_CONTEXT4;
      if (wgpuDeviceHasFeature(
              pin->device(), WGPUFeatureName_SharedTextureMemoryD3D11Texture2D))
        out->facility_flags |= MGPU_EXTERNAL_POOL_DAWN_TEXTURE_IMPORT;
      if (wgpuDeviceHasFeature(pin->device(),
                               WGPUFeatureName_D3D11MultithreadProtected))
        out->facility_flags |= MGPU_EXTERNAL_POOL_DAWN_MULTITHREAD_PROTECTED;
      constexpr uint32_t required =
          MGPU_EXTERNAL_POOL_D3D11_DEVICE5_CONTEXT4 |
          MGPU_EXTERNAL_POOL_DAWN_TEXTURE_IMPORT |
          MGPU_EXTERNAL_POOL_DAWN_MULTITHREAD_PROTECTED;
      out->refusal = out->facility_flags == required
                         ? MGPU_EXTERNAL_POOL_FACILITIES_READY
                         : MGPU_EXTERNAL_POOL_NATIVE_FEATURE_MISSING;
    }
#endif
    // A concurrent logical retirement cannot turn this observation into
    // admission.
    return pin->current() ? MGPU_CONTEXT_LEASE_OK : MGPU_CONTEXT_LEASE_RETIRED;
  } catch (...) {
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
}

extern "C" int mgpuExternalPoolQueryCapabilities(
    MGPUContextHandle *context, MGPUExternalPoolCapabilities *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  std::memset(out, 0, sizeof(*out));
  std::shared_ptr<mgpu::ContextPin> pin;
  const int result = mgpu::retainContextPin(context, pin);
  return result ? result : queryExternalPoolPin(std::move(pin), out, size);
}
extern "C" int mgpuExternalPoolQueryLeaseCapabilities(
    MGPUContextLease *lease, MGPUExternalPoolCapabilities *out, uint32_t size) {
  return queryExternalPoolPin(mgpu::copyContextLeasePin(lease), out, size);
}

namespace {
constexpr unsigned kSlots = 3, kConsumers = 4, kEvents = 16;
constexpr uint64_t kMiB = 1024 * 1024, kMaxPoolBytes = 256 * kMiB;
uint64_t nowMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
template <typename T> bool valid(const T *value) {
  return value && value->struct_size == sizeof(T) && value->version == 1;
}
#ifdef _WIN32
using Microsoft::WRL::ComPtr;
// Fixed, process-local identity storage covers all live/retiring pool sources
// and fences. Handle numbers and COM wrapper addresses are not allocation IDs.
std::mutex identityMutex;
std::array<HANDLE, 16 * kSlots * 2> resourceIdentities{};
class KernelIdentity {
public:
  KernelIdentity() = default;
  KernelIdentity(const KernelIdentity &) = delete;
  KernelIdentity &operator=(const KernelIdentity &) = delete;
  KernelIdentity(KernelIdentity &&other) noexcept
      : handle(std::exchange(other.handle, nullptr)),
        index(std::exchange(other.index, resourceIdentities.size())) {}
  KernelIdentity &operator=(KernelIdentity &&other) noexcept {
    if (this != &other) {
      reset();
      handle = std::exchange(other.handle, nullptr);
      index = std::exchange(other.index, resourceIdentities.size());
    }
    return *this;
  }
  ~KernelIdentity() { reset(); }
  int reserve(HANDLE borrowed) {
    using CompareHandles = BOOL(WINAPI *)(HANDLE, HANDLE);
    static const auto compare = [] {
      auto module = GetModuleHandleW(L"kernelbase.dll");
      return module ? reinterpret_cast<CompareHandles>(
                          GetProcAddress(module, "CompareObjectHandles"))
                    : nullptr;
    }();
    if (!compare)
      return MGPU_EXTERNAL_POOL_UNSUPPORTED;
    HANDLE duplicate = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), borrowed, GetCurrentProcess(),
                         &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS))
      return MGPU_CONTEXT_LEASE_INVALID;
    std::unique_ptr<void, decltype(&CloseHandle)> owner(duplicate, &CloseHandle);
    std::lock_guard<std::mutex> lock(identityMutex);
    size_t free = resourceIdentities.size();
    for (size_t i = 0; i < resourceIdentities.size(); ++i) {
      if (resourceIdentities[i]) {
        if (compare(duplicate, resourceIdentities[i])) {
          return MGPU_EXTERNAL_POOL_CONFLICT;
        }
      } else if (free == resourceIdentities.size()) {
        free = i;
      }
    }
    if (free == resourceIdentities.size()) {
      return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    }
    handle = resourceIdentities[free] = owner.release();
    index = free;
    return 0;
  }
  HANDLE get() const { return handle; }

private:
  void reset() noexcept {
    if (!handle)
      return;
    HANDLE retired = nullptr;
    {
      std::lock_guard<std::mutex> lock(identityMutex);
      resourceIdentities[index] = nullptr;
      retired = std::exchange(handle, nullptr);
      index = resourceIdentities.size();
    }
    CloseHandle(retired);
  }
  HANDLE handle = nullptr;
  size_t index = resourceIdentities.size();
};
std::mutex quotaMutex;
unsigned poolCount = 0;
uint64_t poolBytes = 0;
bool reserveQuota(uint64_t bytes) {
  std::lock_guard<std::mutex> lock(quotaMutex);
  if (poolCount == 16 || bytes > 1024 * kMiB - poolBytes)
    return false;
  ++poolCount;
  poolBytes += bytes;
  return true;
}
void releaseQuota(uint64_t bytes) {
  std::lock_guard<std::mutex> lock(quotaMutex);
  --poolCount;
  poolBytes -= bytes;
}
bool adapterMatches(ID3D11Device *device, const LUID &expected) {
  ComPtr<IDXGIDevice> dxgi;
  ComPtr<IDXGIAdapter> adapter;
  DXGI_ADAPTER_DESC desc{};
  return device && SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dxgi))) &&
         SUCCEEDED(dxgi->GetAdapter(&adapter)) &&
         SUCCEEDED(adapter->GetDesc(&desc)) &&
         desc.AdapterLuid.LowPart == expected.LowPart &&
         desc.AdapterLuid.HighPart == expected.HighPart;
}
enum class Phase {
  Idle,
  Waiting,
  CopyQueued,
  Copying,
  ReturnSource,
  Publishing,
  Ready,
  Leased,
  Ending,
  Quarantine
};
struct Source {
  KernelIdentity identity;
  uint64_t id = 0;
  bool busy = false;
  ComPtr<ID3D11Texture2D> texture;
  ComPtr<IDXGIKeyedMutex> mutex;
};
struct Fence {
  KernelIdentity identity;
  uint64_t id = 0, highValue = 0;
  unsigned uses = 0, purpose = 0;
  ComPtr<ID3D11Fence> object;
  bool hostOwned = false;
};
struct Consumer {
  uint64_t id = 0, fenceId = 0, fenceValue = 0;
  unsigned kind = 0;
  bool released = false, done = false, callbackPending = false;
};
struct Slot {
  ComPtr<ID3D11Texture2D> native;
  WGPUSharedTextureMemory memory = nullptr;
  WGPUTexture texture = nullptr;
  WGPUTextureView view = nullptr;
  Phase phase = Phase::Idle;
  MGPUExternalPoolSubmission frame{};
  uint64_t copyValue = 0;
  bool mutexHeld = false, access = false, canceled = false, uncertain = false;
  int result = 0;
  std::array<Consumer, kConsumers> consumers{};
};
class PoolState : public std::enable_shared_from_this<PoolState> {
public:
  PoolState(std::shared_ptr<mgpu::ContextPin> creation,
            MGPUExternalPoolConfig declared, bool createIssued)
      : pin(std::move(creation)), config(declared), issued(createIssued),
        frameBytes(uint64_t(config.width) * config.height * 4) {}
  ~PoolState() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      wake.notify_all();
    }
    if (coordinator.joinable())
      coordinator.join();
    // Release cleans up on the creation worker after all captures drain. Failed
    // creation has admitted no work and may release thread-safe Dawn references
    // directly. Destruction itself never allocates or enqueues a synchronous
    // job.
    cleanup();
    if (charged)
      releaseQuota(config.max_bytes);
  }
  void cleanup() noexcept {
    for (auto &s : slots) {
      if (s.view)
        wgpuTextureViewRelease(s.view);
      if (s.texture)
        wgpuTextureRelease(s.texture);
      if (s.memory)
        wgpuSharedTextureMemoryRelease(s.memory);
      s.view = nullptr;
      s.texture = nullptr;
      s.memory = nullptr;
      s.native.Reset();
    }
    sources = {};
    fences = {};
    copyFence.Reset();
    context4.Reset();
    device5.Reset();
    native.Reset();
  }
  int initialize(const MGPUExternalPoolCapabilities &caps) {
    luid = {caps.adapter_luid_low, caps.adapter_luid_high};
    // One cold, bounded image initializes every issued source before NT export.
    // CreateTexture2D owns its initial data; do not enqueue a separate clear whose
    // unresolved GPU work would invalidate failed-creation cleanup. Config has
    // already bounded frameBytes to <= max_bytes / six texture planes.
    const std::vector<uint8_t> initialPixels(issued ? size_t(frameBytes) : 0, 0);
    return pin->gpu().getWebGPUThread().enqueueSync<int>([this, &initialPixels]() -> int {
      if (!pin->current())
        return MGPU_CONTEXT_LEASE_RETIRED;
      native = dawn::native::d3d11::GetD3D11Device(pin->device());
      ComPtr<ID3D11DeviceContext> immediate;
      if (!native || FAILED(native.As(&device5)))
        return MGPU_EXTERNAL_POOL_UNSUPPORTED;
      native->GetImmediateContext(&immediate);
      if (!immediate || FAILED(immediate.As(&context4)))
        return MGPU_EXTERNAL_POOL_UNSUPPORTED;
      if (FAILED(device5->CreateFence(0, D3D11_FENCE_FLAG_NONE,
                                      IID_PPV_ARGS(&copyFence))))
        return MGPU_EXTERNAL_POOL_UNSUPPORTED;
      mgpu::lock_guard<mgpu::mutex> gpuLock(pin->gpu().getGpuMutex());
      for (auto &s : slots) {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = config.width;
        desc.Height = config.height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.SampleDesc.Count = 1;
        desc.Format = format();
        desc.Usage = D3D11_USAGE_DEFAULT;
        // VideoProcessor input views require render-target-capable RGBA/BGRA.
        // Keep the same private isolation texture; no extra RGBA staging copy.
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        if (FAILED(native->CreateTexture2D(&desc, nullptr, &s.native)))
          return MGPU_CONTEXT_LEASE_INTERNAL;
        ++allocations;
        dawn::native::d3d11::SharedTextureMemoryD3D11Texture2DDescriptor chain;
        chain.texture = s.native;
        chain.requiresEndAccessFence = false;
        WGPUSharedTextureMemoryDescriptor imported{};
        imported.nextInChain = reinterpret_cast<WGPUChainedStruct *>(&chain);
        s.memory =
            wgpuDeviceImportSharedTextureMemory(pin->device(), &imported);
        WGPUSharedTextureMemoryProperties properties{};
        if (!s.memory || wgpuSharedTextureMemoryGetProperties(
                             s.memory, &properties) != WGPUStatus_Success)
          return MGPU_CONTEXT_LEASE_INTERNAL;
        WGPUTextureDescriptor texture{};
        texture.dimension = WGPUTextureDimension_2D;
        texture.size = {config.width, config.height, 1};
        texture.format = config.pixel_format == MGPU_EXTERNAL_POOL_RGBA8
                             ? WGPUTextureFormat_RGBA8Unorm
                             : WGPUTextureFormat_BGRA8Unorm;
        texture.mipLevelCount = 1;
        texture.sampleCount = 1;
        texture.usage =
            WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc;
        if ((properties.usage & texture.usage) != texture.usage)
          return MGPU_EXTERNAL_POOL_UNSUPPORTED;
        s.texture = wgpuSharedTextureMemoryCreateTexture(s.memory, &texture);
        if (!s.texture)
          return MGPU_CONTEXT_LEASE_INTERNAL;
        s.view = wgpuTextureCreateView(s.texture, nullptr);
        if (!s.view)
          return MGPU_CONTEXT_LEASE_INTERNAL;
        ++imports;
      }
      if (issued) {
        for (auto &s : sources) {
          D3D11_TEXTURE2D_DESC desc{};
          desc.Width = config.width;
          desc.Height = config.height;
          desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
          desc.Format = format();
          desc.Usage = D3D11_USAGE_DEFAULT;
          desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
          desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                           D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
          D3D11_SUBRESOURCE_DATA initial{};
          initial.pSysMem = initialPixels.data();
          initial.SysMemPitch = config.width * 4;
          if (FAILED(native->CreateTexture2D(&desc, &initial, &s.texture)) ||
              FAILED(s.texture.As(&s.mutex)))
            return MGPU_CONTEXT_LEASE_INTERNAL;
          // Establish the creation device's first handoff before a foreign
          // device acquires key zero. Initial data is part of resource creation,
          // and the keyed mutex synchronizes that access across devices.
          if (s.mutex->AcquireSync(0, 0) != S_OK ||
              s.mutex->ReleaseSync(0) != S_OK)
            return MGPU_CONTEXT_LEASE_INTERNAL;
          ++allocations;
          ComPtr<IDXGIResource1> resource;
          HANDLE handle = nullptr;
          if (FAILED(s.texture.As(&resource)) ||
              FAILED(resource->CreateSharedHandle(
                  nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                  nullptr, &handle)))
            return MGPU_CONTEXT_LEASE_INTERNAL;
          std::unique_ptr<void, decltype(&CloseHandle)> owner(handle, &CloseHandle);
          if (int r = s.identity.reserve(handle))
            return r;
          s.id = ++nextId;
        }
        auto &f = fences[0];
        HANDLE handle = nullptr;
        if (FAILED(device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                        IID_PPV_ARGS(&f.object))) ||
            FAILED(f.object->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr,
                                                &handle)))
          return MGPU_CONTEXT_LEASE_INTERNAL;
        std::unique_ptr<void, decltype(&CloseHandle)> owner(handle, &CloseHandle);
        if (int r = f.identity.reserve(handle))
          return r;
        f.id = ++nextId;
        f.purpose = MGPU_EXTERNAL_POOL_PRODUCER_FENCE;
      }
      return 0;
    });
  }
  DXGI_FORMAT format() const {
    return config.pixel_format == MGPU_EXTERNAL_POOL_RGBA8
               ? DXGI_FORMAT_R8G8B8A8_UNORM
               : DXGI_FORMAT_B8G8R8A8_UNORM;
  }
  int admission() const {
    if (stopping || retiring || !pin->current())
      return MGPU_CONTEXT_LEASE_RETIRED;
    if (faulted)
      return MGPU_EXTERNAL_POOL_DEVICE_LOST;
    return 0;
  }
  Source *source(uint64_t id) {
    for (auto &s : sources)
      if (id && s.id == id)
        return &s;
    return nullptr;
  }
  Fence *fence(uint64_t id) {
    for (auto &f : fences)
      if (id && f.id == id)
        return &f;
    return nullptr;
  }
  Slot *slot(uint64_t token) {
    for (auto &s : slots)
      if (s.phase != Phase::Idle && s.frame.lease_token == token)
        return &s;
    return nullptr;
  }
  Consumer *consumer(Slot &s, uint64_t id) {
    for (auto &c : s.consumers)
      if (id && c.id == id)
        return &c;
    return nullptr;
  }
  void emit(Slot &s, unsigned kind, int result) {
    MGPUExternalPoolEvent e{};
    e.struct_size = sizeof(e);
    e.version = 1;
    e.kind = kind;
    e.result = result;
    e.lease_token = s.frame.lease_token;
    e.media_sequence = s.frame.media_sequence;
    e.producer_fence_value = s.frame.producer_fence_value;
    e.copy_fence_value = s.copyValue;
    events[(head + count) % kEvents] = e;
    ++count;
    --reservedEvents;
  }
  void fault(int result) {
    if (faulted)
      return;
    faulted = true;
    MGPUExternalPoolEvent e{};
    e.struct_size = sizeof(e);
    e.version = 1;
    e.kind = MGPU_EXTERNAL_POOL_FAULT;
    e.result = result;
    events[(head + count) % kEvents] = e;
    ++count;
    for (auto &s : slots)
      if (s.phase != Phase::Idle)
        s.canceled = true;
  }
  void quarantineLocked() noexcept {
    // Uncertain partial task/scheduling failure never manufactures a terminal
    // ownership receipt. Fixed storage keeps the fault and resources charged.
    fault(MGPU_CONTEXT_LEASE_INTERNAL);
    for (auto &s : slots)
      if (s.phase != Phase::Idle) {
        s.phase = Phase::Quarantine;
        s.uncertain = true;
      }
  }
  void enqueueJob(std::function<void()> task) {
    auto self = shared_from_this();
    pin->gpu().getWebGPUThread().enqueueAsync([self, task = std::move(task)] {
      try {
        task();
      } catch (...) {
        std::lock_guard<std::mutex> lock(self->mutex);
        self->quarantineLocked();
        self->wake.notify_all();
      }
    });
  }
  void idle(Slot &s) {
    s.frame = {};
    s.consumers = {};
    s.copyValue = 0;
    s.phase = Phase::Idle;
    s.canceled = s.uncertain = s.mutexHeld = false;
    s.result = 0;
  }
  void sourceFinished(Slot &s) {
    auto *src = source(s.frame.texture_id);
    auto *f = fence(s.frame.producer_fence_id);
    if (src)
      src->busy = false;
    if (f)
      --f->uses;
    emit(s, MGPU_EXTERNAL_POOL_SOURCE_DONE, s.result);
  }
  void rollback(Slot &s, int result) {
    s.result = result;
    sourceFinished(s);
    emit(s, MGPU_EXTERNAL_POOL_OUTPUT_READY, result);
    emit(s, MGPU_EXTERNAL_POOL_OUTPUT_DONE, result);
    idle(s);
  }
  void enqueueCopy(Slot &s) {
    s.phase = Phase::CopyQueued;
    auto self = shared_from_this();
    auto token = s.frame.lease_token;
    enqueueJob([self, token] {
      std::lock_guard<std::mutex> lock(self->mutex);
      auto *s = self->slot(token);
      if (!s || s->phase != Phase::CopyQueued)
        return;
      if (s->canceled || self->admission() ||
          nowMicros() >= s->frame.deadline_micros) {
        s->result = s->canceled ? MGPU_EXTERNAL_POOL_CANCELED
                                : MGPU_EXTERNAL_POOL_TIMEOUT;
        s->phase = Phase::ReturnSource;
        self->wake.notify_all();
        return;
      }
      auto *src = self->source(s->frame.texture_id);
      mgpu::lock_guard<mgpu::mutex> gpuLock(self->pin->gpu().getGpuMutex());
      self->pin->gpu().flushBatchLocked();
      s->phase = Phase::Copying;
      self->context4->CopyResource(s->native.Get(), src->texture.Get());
      s->copyValue = ++self->nextCopy;
      ++self->copies;
      if (FAILED(self->context4->Signal(self->copyFence.Get(), s->copyValue))) {
        s->phase = Phase::Quarantine;
        s->uncertain = true;
        self->fault(MGPU_EXTERNAL_POOL_DEVICE_LOST);
      }
      self->context4->Flush();
      self->wake.notify_all();
    });
  }
  void enqueuePublish(Slot &s) {
    s.phase = Phase::Publishing;
    auto self = shared_from_this();
    auto token = s.frame.lease_token;
    enqueueJob([self, token] {
      std::lock_guard<std::mutex> lock(self->mutex);
      auto *s = self->slot(token);
      if (!s || s->phase != Phase::Publishing)
        return;
      if (s->canceled || self->admission()) {
        self->emit(*s, MGPU_EXTERNAL_POOL_OUTPUT_READY,
                   MGPU_EXTERNAL_POOL_CANCELED);
        self->emit(*s, MGPU_EXTERNAL_POOL_OUTPUT_DONE,
                   MGPU_EXTERNAL_POOL_CANCELED);
        self->idle(*s);
        return;
      }
      mgpu::lock_guard<mgpu::mutex> gpuLock(self->pin->gpu().getGpuMutex());
      WGPUSharedTextureMemoryBeginAccessDescriptor access{};
      access.initialized = true;
      if (wgpuSharedTextureMemoryBeginAccess(s->memory, s->texture, &access) !=
          WGPUStatus_Success) {
        // No consumers or Dawn work have been admitted for this slot.
        self->emit(*s, MGPU_EXTERNAL_POOL_OUTPUT_READY,
                   MGPU_CONTEXT_LEASE_INTERNAL);
        self->emit(*s, MGPU_EXTERNAL_POOL_OUTPUT_DONE,
                   MGPU_CONTEXT_LEASE_INTERNAL);
        self->idle(*s);
        self->fault(MGPU_EXTERNAL_POOL_DEVICE_LOST);
        return;
      }
      s->access = true;
      s->phase = Phase::Ready;
      self->emit(*s, MGPU_EXTERNAL_POOL_OUTPUT_READY, 0);
      self->wake.notify_all();
    });
  }
  void enqueueEnd(Slot &s) {
    s.phase = Phase::Ending;
    auto self = shared_from_this();
    auto token = s.frame.lease_token;
    enqueueJob([self, token] {
      std::lock_guard<std::mutex> lock(self->mutex);
      auto *s = self->slot(token);
      if (!s || s->phase != Phase::Ending)
        return;
      mgpu::lock_guard<mgpu::mutex> gpuLock(self->pin->gpu().getGpuMutex());
      WGPUSharedTextureMemoryEndAccessState end{};
      auto status =
          wgpuSharedTextureMemoryEndAccess(s->memory, s->texture, &end);
      wgpuSharedTextureMemoryEndAccessStateFreeMembers(end);
      if (status != WGPUStatus_Success) {
        s->phase = Phase::Quarantine;
        s->uncertain = true;
        self->fault(MGPU_EXTERNAL_POOL_DEVICE_LOST);
        return;
      }
      s->access = false;
      self->emit(*s, MGPU_EXTERNAL_POOL_OUTPUT_DONE,
                 s->canceled ? MGPU_EXTERNAL_POOL_CANCELED : 0);
      self->idle(*s);
      self->wake.notify_all();
    });
  }
  struct Completion {
    std::shared_ptr<PoolState> pool;
    uint64_t token, id;
  };
  void enqueueDawnCompletion(Slot &s, Consumer &c) {
    c.callbackPending = true;
    auto self = shared_from_this();
    auto token = s.frame.lease_token;
    auto id = c.id;
    enqueueJob([self, token, id] {
      std::lock_guard<std::mutex> lock(self->mutex);
      auto *s = self->slot(token);
      if (!s)
        return;
      auto *c = self->consumer(*s, id);
      if (!c || !c->callbackPending)
        return;
      // A retired owner still drains its exact creation queue. Never repair.
      mgpu::lock_guard<mgpu::mutex> gpuLock(self->pin->gpu().getGpuMutex());
      if (self->pin->gpu().matchesReadyDevice(self->pin->device(),
                                              self->pin->deviceGeneration()))
        self->pin->gpu().flushBatchLocked();
      auto data = std::make_unique<Completion>(Completion{self, token, id});
      WGPUQueueWorkDoneCallbackInfo callback{};
      callback.mode = WGPUCallbackMode_AllowProcessEvents;
      callback.userdata1 = data.release();
      callback.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView,
                             void *data, void *) {
        std::unique_ptr<Completion> completed(static_cast<Completion *>(data));
        auto self = completed->pool;
        std::lock_guard<std::mutex> lock(self->mutex);
        auto *s = self->slot(completed->token);
        if (!s)
          return;
        auto *c = self->consumer(*s, completed->id);
        if (!c)
          return;
        c->callbackPending = false;
        if (status == WGPUQueueWorkDoneStatus_Success)
          c->done = true;
        else {
          s->uncertain = true;
          self->fault(MGPU_EXTERNAL_POOL_DEVICE_LOST);
        }
        self->wake.notify_all();
      };
      wgpuQueueOnSubmittedWorkDone(self->pin->queue(), callback);
      self->wake.notify_all();
    });
  }
  void step() {
    if (!pin->current() && !retiring) {
      retiring = true;
      fault(MGPU_CONTEXT_LEASE_RETIRED);
    }
    for (auto &s : slots) {
      if (s.phase == Phase::Idle || s.phase == Phase::Quarantine)
        continue;
      if (retiring || faulted)
        s.canceled = true;
      if (s.phase == Phase::Waiting) {
        if (s.canceled || nowMicros() >= s.frame.deadline_micros) {
          rollback(s, s.canceled ? MGPU_EXTERNAL_POOL_CANCELED
                                 : MGPU_EXTERNAL_POOL_TIMEOUT);
          continue;
        }
        auto *f = fence(s.frame.producer_fence_id);
        auto value = f->object->GetCompletedValue();
        if (value == UINT64_MAX) {
          rollback(s, MGPU_EXTERNAL_POOL_DEVICE_LOST);
          fault(MGPU_EXTERNAL_POOL_DEVICE_LOST);
          continue;
        }
        if (value < s.frame.producer_fence_value)
          continue;
        auto *src = source(s.frame.texture_id);
        const HRESULT result = src->mutex->AcquireSync(1, 0);
        // WAIT_TIMEOUT/ABANDONED are nonnegative: SUCCEEDED is not admission.
        if (result == WAIT_TIMEOUT)
          continue;
        if (result != S_OK) {
          rollback(s, MGPU_EXTERNAL_POOL_DEVICE_LOST);
          fault(MGPU_EXTERNAL_POOL_DEVICE_LOST);
          continue;
        }
        s.mutexHeld = true;
        enqueueCopy(s);
        continue;
      }
      if (s.phase == Phase::Copying) {
        auto value = copyFence->GetCompletedValue();
        if (value == UINT64_MAX) {
          s.uncertain = true;
          s.phase = Phase::Quarantine;
          fault(MGPU_EXTERNAL_POOL_DEVICE_LOST);
          continue;
        }
        if (value < s.copyValue) {
          if (nowMicros() >= s.frame.deadline_micros) {
            s.uncertain = true;
            fault(MGPU_EXTERNAL_POOL_TIMEOUT);
          }
          continue;
        }
        s.uncertain = false;
        s.phase = Phase::ReturnSource;
        s.result = s.canceled ? MGPU_EXTERNAL_POOL_CANCELED : 0;
      }
      if (s.phase == Phase::ReturnSource) {
        auto *src = source(s.frame.texture_id);
        if (s.mutexHeld && src->mutex->ReleaseSync(0) != S_OK) {
          s.uncertain = true;
          s.phase = Phase::Quarantine;
          fault(MGPU_EXTERNAL_POOL_DEVICE_LOST);
          continue;
        }
        s.mutexHeld = false;
        if (s.canceled && !s.result)
          s.result = MGPU_EXTERNAL_POOL_CANCELED;
        sourceFinished(s);
        if (s.result) {
          emit(s, MGPU_EXTERNAL_POOL_OUTPUT_READY, s.result);
          emit(s, MGPU_EXTERNAL_POOL_OUTPUT_DONE, s.result);
          idle(s);
        } else
          enqueuePublish(s);
      }
      if (s.phase == Phase::Ready && s.canceled)
        enqueueEnd(s);
      if (s.phase == Phase::Leased) {
        bool allDone = true;
        for (auto &c : s.consumers) {
          if (!c.id)
            continue;
          if (c.released && !c.done &&
              c.kind == MGPU_EXTERNAL_POOL_NATIVE_FENCE_CONSUMER) {
            auto *f = fence(c.fenceId);
            auto value = f->object->GetCompletedValue();
            if (value == UINT64_MAX) {
              s.uncertain = true;
              fault(MGPU_EXTERNAL_POOL_DEVICE_LOST);
            } else if (value >= c.fenceValue) {
              c.done = true;
              --f->uses;
            }
          }
          allDone &= c.done;
        }
        if (allDone && !s.uncertain)
          enqueueEnd(s);
      }
    }
  }
  void start() {
    coordinator = std::thread([this] {
      for (;;) {
        bool callbacks = false;
        {
          std::unique_lock<std::mutex> lock(mutex);
          if (stopping)
            break;
          try {
            step();
          } catch (...) {
            quarantineLocked();
          }
          for (const auto &s : slots)
            for (const auto &c : s.consumers)
              callbacks |= c.callbackPending;
        }
        if (callbacks)
          wgpuInstanceProcessEvents(pin->instance());
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait_for(lock, std::chrono::milliseconds(1));
      }
    });
  }
  std::shared_ptr<mgpu::ContextPin> pin;
  MGPUExternalPoolConfig config;
  const bool issued;
  uint64_t frameBytes;
  std::mutex mutex;
  std::condition_variable wake;
  std::thread coordinator;
  bool stopping = false, charged = false, retiring = false, faulted = false;
  LUID luid{};
  ComPtr<ID3D11Device> native;
  ComPtr<ID3D11Device5> device5;
  ComPtr<ID3D11DeviceContext4> context4;
  ComPtr<ID3D11Fence> copyFence;
  std::array<Source, kSlots> sources{};
  std::array<Fence, kSlots> fences{};
  std::mutex completionSignalMutex;
  std::array<Slot, kSlots> slots{};
  std::array<MGPUExternalPoolEvent, kEvents> events{};
  unsigned head = 0, count = 0, reservedEvents = 0;
  uint64_t nextId = 0, nextConsumer = 0, nextCopy = 0, highSequence = 0,
           highToken = 0;
  uint64_t producerExport = 0;
  bool producerReleased = false;
  unsigned allocations = 0, imports = 0;
  uint64_t copies = 0;
};
#endif
} // namespace

struct MGPUExternalPool {
#ifdef _WIN32
  std::shared_ptr<PoolState> state;
#endif
};

extern "C" uint64_t mgpuExternalPoolNowMicros() { return nowMicros(); }
extern "C" int mgpuExternalPoolGetLimits(MGPUExternalPoolLimits *out,
                                         uint32_t size) {
  if (!out || size < sizeof(*out))
    return MGPU_CONTEXT_LEASE_INVALID;
  std::memset(out, 0, sizeof(*out));
  out->struct_size = sizeof(*out);
  out->version = 1;
#ifdef _WIN32
  try {
    std::lock_guard<std::mutex> lock(quotaMutex);
    out->max_pools = 16;
    out->max_source_textures = kSlots;
    out->max_fences = kSlots;
    out->output_slots = kSlots;
    out->max_consumers = kConsumers;
    out->event_records = kEvents;
    out->pixel_format_mask = 3;
    out->live_pools = poolCount;
    out->max_pool_bytes = kMaxPoolBytes;
    out->max_aggregate_bytes = 1024 * kMiB;
    out->reserved_bytes = poolBytes;
    return 0;
  } catch (...) {
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
}

static bool validPoolConfig(const MGPUExternalPoolConfig *config, bool issued) {
  return valid(config) && !config->reserved && config->width && config->height &&
         config->width <= 16384 && config->height <= 16384 &&
         (config->pixel_format == MGPU_EXTERNAL_POOL_RGBA8 ||
          config->pixel_format == MGPU_EXTERNAL_POOL_BGRA8) &&
         config->max_bytes <= kMaxPoolBytes &&
         config->max_bytes >= uint64_t(config->width) * config->height * 4 *
                                  kSlots * (issued ? 2 : 1);
}

static int createExternalPoolPin(std::shared_ptr<mgpu::ContextPin> pin,
                              const MGPUExternalPoolConfig *config,
                              MGPUExternalPool **out, bool issued) {
  if (!out)
    return MGPU_CONTEXT_LEASE_INVALID;
  *out = nullptr;
  if (!validPoolConfig(config, issued))
    return MGPU_CONTEXT_LEASE_INVALID;
#ifdef _WIN32
  try {
    MGPUExternalPoolCapabilities caps{};
    int result =
        queryExternalPoolPin(pin, &caps, sizeof(caps));
    if (result)
      return result;
    if (caps.refusal != MGPU_EXTERNAL_POOL_FACILITIES_READY)
      return MGPU_EXTERNAL_POOL_UNSUPPORTED;
    if (config->width > caps.max_texture_dimension_2d ||
        config->height > caps.max_texture_dimension_2d)
      return MGPU_CONTEXT_LEASE_INVALID;
    if (pin->contextGeneration() != caps.context_generation ||
        pin->deviceGeneration() != caps.device_generation)
      return MGPU_CONTEXT_LEASE_RETIRED;
    auto state = std::make_shared<PoolState>(std::move(pin), *config, issued);
    if (!reserveQuota(config->max_bytes))
      return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    state->charged = true;
    result = state->initialize(caps);
    if (result)
      return result;
    if (!state->pin->current())
      return MGPU_CONTEXT_LEASE_RETIRED;
    auto wrapper = std::make_unique<MGPUExternalPool>();
    wrapper->state = state;
    state->start();
    *out = wrapper.release();
    return 0;
  } catch (...) {
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
}

extern "C" int mgpuExternalPoolCreate(MGPUContextHandle *context,
                                      const MGPUExternalPoolConfig *config,
                                      MGPUExternalPool **out) {
  if (!out) return MGPU_CONTEXT_LEASE_INVALID;
  *out = nullptr;
  if (!validPoolConfig(config, false)) return MGPU_CONTEXT_LEASE_INVALID;
  std::shared_ptr<mgpu::ContextPin> pin;
  const int result = mgpu::retainContextPin(context, pin);
  return result ? result : createExternalPoolPin(std::move(pin), config, out, false);
}
extern "C" int mgpuExternalPoolCreateIssued(MGPUContextHandle *context,
                                            const MGPUExternalPoolConfig *config,
                                            MGPUExternalPool **out) {
  if (!out) return MGPU_CONTEXT_LEASE_INVALID;
  *out = nullptr;
  if (!validPoolConfig(config, true)) return MGPU_CONTEXT_LEASE_INVALID;
  std::shared_ptr<mgpu::ContextPin> pin;
  const int result = mgpu::retainContextPin(context, pin);
  return result ? result : createExternalPoolPin(std::move(pin), config, out, true);
}
extern "C" int mgpuExternalPoolCreateIssuedFromLease(
    MGPUContextLease *lease, const MGPUExternalPoolConfig *config,
    MGPUExternalPool **out) {
  return createExternalPoolPin(mgpu::copyContextLeasePin(lease), config, out, true);
}

// No public call may race final Release. Other calls are serialized by the
// pool; borrowed resource/shader objects still obey their ordinary native
// lifetime.
#ifdef _WIN32
#define POOL_BEGIN(pool)                                                       \
  if (!(pool) || !(pool)->state)                                               \
    return MGPU_CONTEXT_LEASE_INVALID;                                         \
  try {                                                                        \
    auto self = (pool)->state;                                                 \
    std::lock_guard<std::mutex> lock(self->mutex);
#define POOL_END                                                               \
  }                                                                            \
  catch (...) {                                                                \
    return MGPU_CONTEXT_LEASE_INTERNAL;                                        \
  }
#else
#define POOL_BEGIN(pool)                                                       \
  if (!(pool))                                                                 \
    return MGPU_CONTEXT_LEASE_INVALID;
#define POOL_END
#endif

extern "C" int mgpuExternalPoolAcquireProducer(MGPUExternalPool *pool,
                                               MGPUExternalPoolProducer *out,
                                               uint32_t size) {
  if (!out || size < sizeof(*out))
    return MGPU_CONTEXT_LEASE_INVALID;
  std::memset(out, 0, sizeof(*out));
  POOL_BEGIN(pool)
#ifdef _WIN32
  if (!self->issued)
    return MGPU_EXTERNAL_POOL_UNSUPPORTED;
  if (int r = self->admission())
    return r;
  if (self->producerExport || self->producerReleased)
    return MGPU_EXTERNAL_POOL_CONFLICT;
  self->producerExport = ++self->nextId;
  out->struct_size = sizeof(*out);
  out->version = 1;
  out->width = self->config.width;
  out->height = self->config.height;
  out->pixel_format = self->config.pixel_format;
  out->slot_count = kSlots;
  out->context_generation = self->pin->contextGeneration();
  out->device_generation = self->pin->deviceGeneration();
  out->export_lease = self->producerExport;
  for (unsigned i = 0; i < kSlots; ++i) {
    out->texture_ids[i] = self->sources[i].id;
    out->nt_texture_handles[i] = self->sources[i].identity.get();
  }
  out->producer_fence_id = self->fences[0].id;
  out->nt_producer_fence_handle = self->fences[0].identity.get();
  return 0;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolReleaseProducerAfterPeerRelease(
    MGPUExternalPool *pool, uint64_t lease) {
  POOL_BEGIN(pool)
#ifdef _WIN32
  if (!self->issued || !lease || self->producerExport != lease)
    return MGPU_CONTEXT_LEASE_INVALID;
  self->producerExport = 0;
  self->producerReleased = true;
  return 0;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}

extern "C" int mgpuExternalPoolRegisterTexture(MGPUExternalPool *pool,
                                               void *handle, uint64_t *out) {
  if (!out)
    return MGPU_CONTEXT_LEASE_INVALID;
  *out = 0;
  if (!handle || handle == reinterpret_cast<void *>(intptr_t(-1)))
    return MGPU_CONTEXT_LEASE_INVALID;
  POOL_BEGIN(pool)
#ifdef _WIN32
  if (int r = self->admission())
    return r;
  if (self->issued)
    return MGPU_EXTERNAL_POOL_CONFLICT;
  Source *free = nullptr;
  unsigned total = kSlots;
  for (auto &s : self->sources) {
    if (!s.id && !free)
      free = &s;
    total += !!s.id;
  }
  if (!free ||
      (uint64_t(total) + 1) * self->frameBytes > self->config.max_bytes)
    return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  KernelIdentity identity;
  if (int r = identity.reserve(static_cast<HANDLE>(handle)))
    return r;
  ComPtr<ID3D11Texture2D> texture;
  if (FAILED(self->device5->OpenSharedResource1(identity.get(),
                                                IID_PPV_ARGS(&texture))))
    return MGPU_CONTEXT_LEASE_INVALID;
  D3D11_TEXTURE2D_DESC desc{};
  texture->GetDesc(&desc);
  constexpr UINT expectedMisc = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                                D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
  ComPtr<ID3D11Device> device;
  texture->GetDevice(&device);
  ComPtr<IDXGIKeyedMutex> mutex;
  if (desc.Width != self->config.width || desc.Height != self->config.height ||
      desc.Format != self->format() || desc.MipLevels != 1 ||
      desc.ArraySize != 1 || desc.SampleDesc.Count != 1 ||
      desc.SampleDesc.Quality || desc.Usage != D3D11_USAGE_DEFAULT ||
      desc.CPUAccessFlags || desc.MiscFlags != expectedMisc ||
      !(desc.BindFlags & D3D11_BIND_SHADER_RESOURCE) ||
      !adapterMatches(device.Get(), self->luid) || FAILED(texture.As(&mutex)))
    return MGPU_CONTEXT_LEASE_INVALID;
  free->texture = std::move(texture);
  free->mutex = std::move(mutex);
  free->identity = std::move(identity);
  free->id = ++self->nextId;
  *out = free->id;
  return 0;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolRegisterFence(MGPUExternalPool *pool,
                                             void *handle, uint32_t purpose,
                                             uint64_t *out) {
  if (!out)
    return MGPU_CONTEXT_LEASE_INVALID;
  *out = 0;
  if (!handle || (purpose != MGPU_EXTERNAL_POOL_PRODUCER_FENCE &&
                  purpose != MGPU_EXTERNAL_POOL_HOST_CONSUMER_FENCE))
    return MGPU_CONTEXT_LEASE_INVALID;
  POOL_BEGIN(pool)
#ifdef _WIN32
  if (int r = self->admission())
    return r;
  if (self->issued && purpose == MGPU_EXTERNAL_POOL_PRODUCER_FENCE)
    return MGPU_EXTERNAL_POOL_CONFLICT;
  Fence *free = nullptr;
  for (auto &f : self->fences)
    if (!f.id) {
      free = &f;
      break;
    }
  if (!free)
    return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  KernelIdentity identity;
  if (int r = identity.reserve(static_cast<HANDLE>(handle)))
    return r;
  ComPtr<ID3D11Fence> fence;
  if (FAILED(self->device5->OpenSharedFence(identity.get(),
                                            IID_PPV_ARGS(&fence))) ||
      fence->GetCompletedValue() == UINT64_MAX)
    return MGPU_CONTEXT_LEASE_INVALID;
  free->object = std::move(fence);
  free->identity = std::move(identity);
  free->purpose = purpose;
  free->id = ++self->nextId;
  *out = free->id;
  return 0;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolCreateHostConsumerFence(
    MGPUExternalPool *pool, MGPUExternalPoolHostFence *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  std::memset(out, 0, sizeof(*out));
  POOL_BEGIN(pool)
#ifdef _WIN32
  if (int r = self->admission()) return r;
  Fence *free = nullptr;
  for (auto &f : self->fences)
    if (!f.id) { free = &f; break; }
  if (!free) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  ComPtr<ID3D11Fence> fence;
  if (FAILED(self->device5->CreateFence(0, D3D11_FENCE_FLAG_NONE,
                                       IID_PPV_ARGS(&fence))))
    return MGPU_CONTEXT_LEASE_INTERNAL;
  free->object = std::move(fence);
  free->purpose = MGPU_EXTERNAL_POOL_HOST_CONSUMER_FENCE;
  free->hostOwned = true;
  free->id = ++self->nextId;
  out->struct_size = sizeof(*out);
  out->version = 1;
  out->fence_id = free->id;
  out->d3d11_device = self->native.Get();
  return 0;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolSignalHostConsumerFence(
    MGPUExternalPool *pool, uint64_t id, uint64_t value) {
#ifdef _WIN32
  if (!pool || !pool->state || !value || value == UINT64_MAX)
    return MGPU_CONTEXT_LEASE_INVALID;
  auto self = pool->state;
  if (self->pin->gpu().getWebGPUThread().threadId() == std::this_thread::get_id())
    return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  std::lock_guard<std::mutex> signalOrder(self->completionSignalMutex);
  try {
    ComPtr<ID3D11Fence> fence;
    {
      std::lock_guard<std::mutex> lock(self->mutex);
      auto *f = self->fence(id);
      if (!f || !f->hostOwned || f->purpose != MGPU_EXTERNAL_POOL_HOST_CONSUMER_FENCE ||
          value <= f->highValue) return MGPU_CONTEXT_LEASE_INVALID;
      if (self->faulted || self->stopping) return MGPU_EXTERNAL_POOL_DEVICE_LOST;
      f->highValue = value;
      ++f->uses;
      fence = f->object;
    }
    // No metadata lock or foreign readiness/GPU wait enters the creation worker.
    const int result = self->pin->gpu().getWebGPUThread().enqueueSync<int>(
        [self, fence, value] {
          mgpu::lock_guard<mgpu::mutex> lock(self->pin->gpu().getGpuMutex());
          if (self->pin->gpu().matchesReadyDevice(self->pin->device(), self->pin->deviceGeneration()))
            self->pin->gpu().flushBatchLocked();
          if (FAILED(self->context4->Signal(fence.Get(), value)))
            return int(MGPU_EXTERNAL_POOL_DEVICE_LOST);
          self->context4->Flush();
          return 0;
        });
    std::lock_guard<std::mutex> lock(self->mutex);
    if (result == 0) --self->fence(id)->uses;
    else { self->faulted = self->retiring = true; self->wake.notify_all(); }
    return result;
  } catch (...) {
    std::lock_guard<std::mutex> lock(self->mutex);
    self->faulted = self->retiring = true;
    self->wake.notify_all();
    return MGPU_CONTEXT_LEASE_INTERNAL; // Unknown signal remains retained.
  }
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
}
extern "C" int mgpuExternalPoolUnregisterTexture(MGPUExternalPool *pool,
                                                 uint64_t id) {
  POOL_BEGIN(pool)
#ifdef _WIN32
  auto *s = self->source(id);
  if (!s)
    return MGPU_CONTEXT_LEASE_INVALID;
  if (self->issued)
    return MGPU_EXTERNAL_POOL_CONFLICT;
  if (s->busy)
    return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  *s = {};
  return 0;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolUnregisterFence(MGPUExternalPool *pool,
                                               uint64_t id) {
  POOL_BEGIN(pool)
#ifdef _WIN32
  auto *f = self->fence(id);
  if (!f)
    return MGPU_CONTEXT_LEASE_INVALID;
  if (self->issued && f->purpose == MGPU_EXTERNAL_POOL_PRODUCER_FENCE)
    return MGPU_EXTERNAL_POOL_CONFLICT;
  if (f->uses)
    return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  *f = {};
  return 0;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolSubmit(MGPUExternalPool *pool,
                                      const MGPUExternalPoolSubmission *frame) {
  if (!valid(frame) || !frame->media_sequence || !frame->lease_token ||
      !frame->producer_fence_value || frame->producer_fence_value == UINT64_MAX)
    return MGPU_CONTEXT_LEASE_INVALID;
  POOL_BEGIN(pool)
#ifdef _WIN32
  if (int r = self->admission())
    return r;
  if (self->issued && !self->producerExport)
    return MGPU_CONTEXT_LEASE_RETIRED;
  auto now = nowMicros();
  if (frame->deadline_micros <= now || frame->deadline_micros - now > 5000000)
    return MGPU_CONTEXT_LEASE_INVALID;
  auto *src = self->source(frame->texture_id);
  auto *f = self->fence(frame->producer_fence_id);
  if (!src || !f || f->purpose != MGPU_EXTERNAL_POOL_PRODUCER_FENCE)
    return MGPU_CONTEXT_LEASE_INVALID;
  if (frame->media_sequence <= self->highSequence ||
      frame->lease_token <= self->highToken ||
      frame->producer_fence_value <= f->highValue)
    return MGPU_EXTERNAL_POOL_CONFLICT;
  Slot *free = nullptr;
  for (auto &s : self->slots)
    if (s.phase == Phase::Idle) {
      free = &s;
      break;
    }
  if (!free || src->busy ||
      self->count + self->reservedEvents + 3 > kEvents - 1)
    return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  free->frame = *frame;
  free->phase = Phase::Waiting;
  src->busy = true;
  ++f->uses;
  f->highValue = frame->producer_fence_value;
  self->highSequence = frame->media_sequence;
  self->highToken = frame->lease_token;
  self->reservedEvents += 3;
  self->wake.notify_all();
  return MGPU_EXTERNAL_POOL_PENDING;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolCancel(MGPUExternalPool *pool, uint64_t token) {
  POOL_BEGIN(pool)
#ifdef _WIN32
  auto *s = self->slot(token);
  if (!s)
    return MGPU_CONTEXT_LEASE_INVALID;
  s->canceled = true;
  self->wake.notify_all();
  return 0;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolPoll(MGPUExternalPool *pool,
                                    MGPUExternalPoolEvent *out, uint32_t size) {
  if (!out || size < sizeof(*out))
    return MGPU_CONTEXT_LEASE_INVALID;
  std::memset(out, 0, sizeof(*out));
  POOL_BEGIN(pool)
#ifdef _WIN32
  if (!self->count)
    return MGPU_EXTERNAL_POOL_EMPTY;
  *out = self->events[self->head];
  self->head = (self->head + 1) % kEvents;
  --self->count;
  return 0;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolAcquireOutput(MGPUExternalPool *pool,
                                             uint64_t token,
                                             MGPUExternalPoolOutput *out,
                                             uint32_t size) {
  if (!out || size < sizeof(*out))
    return MGPU_CONTEXT_LEASE_INVALID;
  std::memset(out, 0, sizeof(*out));
  POOL_BEGIN(pool)
#ifdef _WIN32
  if (int r = self->admission())
    return r;
  auto *s = self->slot(token);
  if (!s || s->canceled)
    return MGPU_CONTEXT_LEASE_INVALID;
  if (s->phase != Phase::Ready)
    return MGPU_EXTERNAL_POOL_PENDING;
  auto &c = s->consumers[0];
  c.id = ++self->nextConsumer;
  c.kind = MGPU_EXTERNAL_POOL_DAWN_CONSUMER;
  s->phase = Phase::Leased;
  out->struct_size = sizeof(*out);
  out->version = 1;
  out->width = self->config.width;
  out->height = self->config.height;
  out->pixel_format = self->config.pixel_format;
  out->lease_token = token;
  out->consumer_id = c.id;
  out->context_generation = self->pin->contextGeneration();
  out->device_generation = self->pin->deviceGeneration();
  out->wgpu_texture = s->texture;
  out->wgpu_texture_view = s->view;
  out->d3d11_texture = s->native.Get();
  return 0;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolRetainOutput(MGPUExternalPool *pool,
                                            uint64_t token, uint32_t kind,
                                            uint64_t *out) {
  if (!out)
    return MGPU_CONTEXT_LEASE_INVALID;
  *out = 0;
  if (kind != MGPU_EXTERNAL_POOL_DAWN_CONSUMER &&
      kind != MGPU_EXTERNAL_POOL_NATIVE_FENCE_CONSUMER)
    return MGPU_CONTEXT_LEASE_INVALID;
  POOL_BEGIN(pool)
#ifdef _WIN32
  if (int r = self->admission())
    return r;
  auto *s = self->slot(token);
  if (!s || s->phase != Phase::Leased || s->canceled)
    return MGPU_CONTEXT_LEASE_INVALID;
  for (auto &c : s->consumers)
    if (c.released)
      return MGPU_EXTERNAL_POOL_CONFLICT;
  for (auto &c : s->consumers)
    if (!c.id) {
      c.id = ++self->nextConsumer;
      c.kind = kind;
      *out = c.id;
      return 0;
    }
  return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolReleaseOutputConsumer(MGPUExternalPool *pool,
                                                     uint64_t token,
                                                     uint64_t id,
                                                     uint64_t fenceId,
                                                     uint64_t value) {
  POOL_BEGIN(pool)
#ifdef _WIN32
  auto *s = self->slot(token);
  if (!s || s->phase != Phase::Leased)
    return MGPU_CONTEXT_LEASE_INVALID;
  auto *c = self->consumer(*s, id);
  if (!c || c->released)
    return MGPU_EXTERNAL_POOL_CONFLICT;
  if (c->kind == MGPU_EXTERNAL_POOL_DAWN_CONSUMER) {
    if (fenceId || value)
      return MGPU_CONTEXT_LEASE_INVALID;
    self->enqueueDawnCompletion(*s, *c);
  } else {
    auto *f = self->fence(fenceId);
    if (!f || f->purpose != MGPU_EXTERNAL_POOL_HOST_CONSUMER_FENCE || !value ||
        value == UINT64_MAX)
      return MGPU_CONTEXT_LEASE_INVALID;
    c->fenceId = fenceId;
    c->fenceValue = value;
    ++f->uses;
  }
  c->released = true;
  self->wake.notify_all();
  return MGPU_EXTERNAL_POOL_PENDING;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolBindOutput(MGPUExternalPool *pool,
                                          uint64_t token, uint64_t id,
                                          MGPUComputeShader *shader,
                                          int binding) {
  if (!shader || binding < 0 || binding >= 64)
    return MGPU_CONTEXT_LEASE_INVALID;
  POOL_BEGIN(pool)
#ifdef _WIN32
  if (int r = self->admission())
    return r;
  auto *s = self->slot(token);
  if (!s || s->phase != Phase::Leased || s->canceled)
    return MGPU_CONTEXT_LEASE_INVALID;
  auto *c = self->consumer(*s, id);
  auto *nativeShader = reinterpret_cast<mgpu::ComputeShader *>(shader);
  if (!c || c->kind != MGPU_EXTERNAL_POOL_DAWN_CONSUMER || c->released ||
      &nativeShader->getMGPU() != &self->pin->gpu() ||
      nativeShader->getCreationDevice() != self->pin->device())
    return MGPU_CONTEXT_LEASE_INVALID;
  auto view = s->view;
  self->enqueueJob([self, nativeShader, binding, view] {
    nativeShader->setTextureView(binding, view);
  });
  return 0;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolGetStats(MGPUExternalPool *pool,
                                        MGPUExternalPoolStats *out,
                                        uint32_t size) {
  if (!out || size < sizeof(*out))
    return MGPU_CONTEXT_LEASE_INVALID;
  std::memset(out, 0, sizeof(*out));
  POOL_BEGIN(pool)
#ifdef _WIN32
  out->struct_size = sizeof(*out);
  out->version = 1;
  out->retiring = self->retiring;
  out->faulted = self->faulted;
  out->reserved_bytes = self->config.max_bytes;
  out->texture_bytes = self->frameBytes * kSlots;
  out->host_allocations = self->allocations;
  out->host_imports = self->imports;
  out->gpu_copies = self->copies;
  for (auto &src : self->sources) {
    out->texture_count += !!src.id;
    if (src.id)
      out->texture_bytes += self->frameBytes;
    out->pending_sources += src.busy;
  }
  for (auto &f : self->fences)
    out->fence_count += !!f.id;
  for (auto &s : self->slots) {
    out->occupied_outputs += s.phase != Phase::Idle;
    out->quarantined_outputs += s.uncertain;
    out->queued_copies += s.phase == Phase::CopyQueued;
  }
  return 0;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolRetire(MGPUExternalPool *pool) {
  POOL_BEGIN(pool)
#ifdef _WIN32
  self->retiring = true;
  for (auto &s : self->slots)
    if (s.phase != Phase::Idle)
      s.canceled = true;
  self->wake.notify_all();
  return 0;
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
  POOL_END
}
extern "C" int mgpuExternalPoolRelease(MGPUExternalPool *pool) {
  if (!pool)
    return MGPU_CONTEXT_LEASE_INVALID;
#ifdef _WIN32
  try {
    {
      auto self = pool->state;
      std::lock_guard<std::mutex> lock(self->mutex);
      if (self->producerExport || self->count || self->reservedEvents)
        return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      for (auto &s : self->slots)
        if (s.phase != Phase::Idle)
          return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      for (auto &f : self->fences)
        if (f.uses) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      if (self->pin->gpu().getWebGPUThread().threadId() ==
          std::this_thread::get_id())
        return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      self->stopping = true;
      self->wake.notify_all();
    }
    // Join the event pump and drain the short remaining job captures while the
    // opaque owner still holds the state. Its destructor cannot join itself or
    // enqueueSync back onto the worker whose last capture it just released.
    if (pool->state->coordinator.joinable())
      pool->state->coordinator.join();
    auto state = pool->state;
    state->pin->gpu().getWebGPUThread().enqueueSync<void>(
        [state] { state->cleanup(); });
    delete pool;
    return 0;
  } catch (...) {
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#else
  return MGPU_EXTERNAL_POOL_UNSUPPORTED;
#endif
}
#undef POOL_BEGIN
#undef POOL_END
