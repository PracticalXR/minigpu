#include "../include/buffer.h"
#include "../include/compute_shader.h"
#include "../include/log.h"
#include "../include/mutex.h"
#include "../include/platform_sleep.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <thread>
#include <vector>

#ifndef __EMSCRIPTEN__
// Dawn C++ native API — lets us enumerate all adapters and pick by name
// without any platform-specific DXGI/Metal code.
#include "dawn/native/DawnNative.h"
#endif

namespace mgpu {

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
  {
    mgpu::lock_guard<mgpu::mutex> lock(queueMutex);
    tasks.push(std::move(task));
  }
  condition.notify_one();
#endif
}

void MGPU::initializeContext() {
  if (ctx && ctx->initialized) {
    return; // Already initialized
  }
  MGPU_LOG(mgpu::LOG_DEBUG,
      "[mgpu] initializeContext() entering: ctx=%p ctx_initialized=%d",
      (void*)ctx.get(),
      ctx ? (int)ctx->initialized : -1);

  ctx = std::make_unique<Context>();

  // Create WebGPU instance via dawn::native::Instance on native platforms so
  // we can use EnumerateAdapters for MGPU_ADAPTER_NAME selection below.
  WGPUInstanceDescriptor instanceDesc = {};
  instanceDesc.nextInChain = nullptr;
#ifndef __EMSCRIPTEN__
  {
    auto* nativeInst = new dawn::native::Instance(&instanceDesc);
    ctx->dawnNativeInstance = nativeInst;
    ctx->instance = nativeInst->Get();
    // Take our own ref so destroyContext's wgpuInstanceRelease is balanced
    // regardless of whether the native wrapper also holds one.
    wgpuInstanceAddRef(ctx->instance);
  }
#else
  ctx->instance = wgpuCreateInstance(&instanceDesc);
#endif

  if (!ctx->instance) {
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
#ifdef _WIN32
  adapterOptions.backendType = WGPUBackendType_D3D11;
#endif
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
    auto* nativeInst = static_cast<dawn::native::Instance*>(ctx->dawnNativeInstance);

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

    // 1. MGPU_ADAPTER_NAME env-var override (substring, case-insensitive).
    if (const char* nameFilter = std::getenv("MGPU_ADAPTER_NAME")) {
      std::string filter(nameFilter);
      for (auto& c : filter) c = (char)std::tolower((unsigned char)c);
      for (auto& e : entries) {
        if (e.type == WGPUAdapterType_CPU) continue;
        std::string lower = e.name;
        for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
        if (lower.find(filter) != std::string::npos) {
          ctx->adapter = e.handle;
          wgpuAdapterAddRef(ctx->adapter);
          MGPU_LOG(mgpu::LOG_INFO, "[mgpu] MGPU_ADAPTER_NAME='%s': selected '%s'",
              nameFilter, e.name.c_str());
          break;
        }
      }
      if (!ctx->adapter)
        MGPU_LOG(mgpu::LOG_WARN,
            "[mgpu] MGPU_ADAPTER_NAME='%s': no match - using auto-select", nameFilter);
    }

    // 2. Automatic: discrete GPU first, then integrated, then any non-CPU.
    if (!ctx->adapter) {
      const WGPUAdapterType kPasses[] = {
        WGPUAdapterType_DiscreteGPU,   // dedicated adapter (NVIDIA/AMD)
        WGPUAdapterType_IntegratedGPU, // built-in (Intel/Apple)
        WGPUAdapterType_Unknown,       // catch-all (matches any non-CPU)
      };
      for (auto want : kPasses) {
        if (ctx->adapter) break;
        for (auto& e : entries) {
          if (e.type == WGPUAdapterType_CPU) continue;
          // For the catch-all pass accept anything; otherwise require exact type.
          if (want != WGPUAdapterType_Unknown && e.type != want) continue;
          ctx->adapter = e.handle;
          wgpuAdapterAddRef(ctx->adapter);
          MGPU_LOG(mgpu::LOG_INFO,
              "[mgpu] auto-selected '%s' (adapterType=%d"
              " - 1=Discrete 2=Integrated 4=Unknown)",
              e.name.c_str(), (int)e.type);
          break;
        }
      }
    }

    if (!ctx->adapter)
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
  if (!ctx->adapter) {
    adapterFuture = wgpuInstanceRequestAdapter(
        ctx->instance, &adapterOptions, adapterCallbackInfo);
    // Process events until the adapter request completes
    while (!adapterState.completed) {
      platformSleep(1, ctx->instance);
    }
  } else {
    // Adapter already selected via EnumerateAdapters — synthesise completed state.
    adapterState.adapter = ctx->adapter;
    adapterState.completed = true;
  }

  LOG_INFO("WebGPU adapter request completed");

  if (!adapterState.adapter) {
    wgpuInstanceRelease(ctx->instance);
    throw std::runtime_error("Failed to get WebGPU adapter");
  }

  ctx->adapter = adapterState.adapter;

  // Log which adapter was selected so it's easy to confirm the right GPU.
  // adapterType: 1=DiscreteGpu 2=IntegratedGpu 3=Cpu 4=Unknown
  {
    WGPUAdapterInfo info{};
    if (wgpuAdapterGetInfo(ctx->adapter, &info) == WGPUStatus_Success) {
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
      wgpuAdapterInfoFreeMembers(info);
    }
  }

  // Request device - Updated API
  WGPUDeviceDescriptor deviceDesc = {};
  deviceDesc.nextInChain = nullptr;
  static const char kMainDeviceLabel[] = "MGPU.MainDevice";
  deviceDesc.label.data = kMainDeviceLabel;
  deviceDesc.label.length = sizeof(kMainDeviceLabel) - 1;

  // ── Optionally enable platform-specific shared-texture-memory features so
  //    minigpu_external can import D3D11 NT handles, IOSurfaces, DMA-BUFs and
  //    AHardwareBuffers.  We probe the adapter and only enable what it
  //    actually supports — the device request would fail otherwise.
  std::vector<WGPUFeatureName> wantedFeatures;
#if !defined(__EMSCRIPTEN__)
  auto adapterHas = [&](WGPUFeatureName f) -> bool {
    bool has = wgpuAdapterHasFeature(ctx->adapter, f) != 0;
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

        // Mark context as uninitialized so it can be recreated
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
      wgpuAdapterRequestDevice(ctx->adapter, &deviceDesc, deviceCallbackInfo);

  // Process events until the device request completes
  while (!deviceState.completed) {
    platformSleep(1, ctx->instance);
  }

  LOG_INFO("WebGPU device request completed");

  if (!deviceState.device) {
    wgpuAdapterRelease(ctx->adapter);
    wgpuInstanceRelease(ctx->instance);
    throw std::runtime_error("Failed to get WebGPU device");
  }

  ctx->device = deviceState.device;

  // Get the device queue
  ctx->queue = wgpuDeviceGetQueue(ctx->device);
  if (!ctx->queue) {
    wgpuDeviceRelease(ctx->device);
    wgpuAdapterRelease(ctx->adapter);
    wgpuInstanceRelease(ctx->instance);
    throw std::runtime_error("Failed to get WebGPU queue");
  }

  ctx->initialized = true;
  LOG_INFO("WebGPU context initialized successfully");
}

void MGPU::initializeContextAsync(std::function<void()> callback) {
  webgpuThread.enqueueAsync([this, callback]() {
    try {
      initializeContext();
      if (callback) {
        callback();
      }
    } catch (...) {
      // Error occurred during initialization
      if (callback) {
        callback();
      }
    }
  });
}

void MGPU::destroyContext() {
  if (!ctx) {
    return;
  }

  // Drain all pending WebGPU-thread tasks (buffer cleanups, dispatch completions)
  // before releasing the handles they reference.
  webgpuThread.enqueueSync<void>([]{});

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

    auto cleanupTask = [bufferHandle, stagingHandle, queue, instance, &gpuMutex]() {
      LOG_INFO("Releasing buffer on WebGPU thread: %p", bufferHandle);
      {
        mgpu::lock_guard<mgpu::mutex> lock(gpuMutex);
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

  // Acquire WebGPU operation lock to prevent conflicts with compute dispatch.
  // We use unique_lock (vs lock_guard) so we can unlock it explicitly before
  // calling wgpuInstanceProcessEvents at the end, which must run outside the
  // lock to avoid blocking buffer-release cleanup tasks that also acquire it.
  mgpu::unique_lock<mgpu::mutex> lock(mgpu.getGpuMutex());

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
  if (_readStagingBuffer && _readStagingBufferSize != readBytes) {
    wgpuBufferDestroy(_readStagingBuffer);
    wgpuBufferRelease(_readStagingBuffer);
    _readStagingBuffer = nullptr;
    _readStagingBufferSize = 0;
  }

  if (!_readStagingBuffer) {
    // Create a staging buffer for reading
    WGPUBufferDescriptor stagingDesc = {};
    stagingDesc.size = readBytes;
    stagingDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    stagingDesc.mappedAtCreation = false;

    _readStagingBuffer = wgpuDeviceCreateBuffer(device, &stagingDesc);
    if (!_readStagingBuffer) {
      LOG_ERROR("Failed to create staging buffer");
      throw std::runtime_error("Failed to create staging buffer");
    }
    _readStagingBufferSize = readBytes;
  }
  WGPUBuffer stagingBuffer = _readStagingBuffer;

  // Copy from our buffer to staging buffer - protected by mutex
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

    auto future = wgpuBufferMapAsync(stagingBuffer, WGPUMapMode_Read, 0, readBytes,
                     mapCallbackInfo);

  WGPUInstance instance = mgpu.getInstance();

  // Wait for the map operation
  WGPUFutureWaitInfo waitInfo = {};
  waitInfo.future = future;
  wgpuInstanceWaitAny(instance, 1, &waitInfo, 0);
  if (waitInfo.completed != true) {
    LOG_WARN("wgpuInstanceWaitAny returned status %d, falling back to polling",
             (int)waitInfo.completed);
    while (!readState.completed) {
      platformSleep(1, instance);
    }
  }

  if (readState.status == WGPUMapAsyncStatus_Success) {
    const void *mappedData =
        wgpuBufferGetConstMappedRange(stagingBuffer, 0, readBytes);
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

  // Keep the staging buffer alive for reuse on the next readDirect call.
  // It will be released in Buffer::release() when the buffer is destroyed.
  wgpuCommandBufferRelease(commands);
  wgpuCommandEncoderRelease(encoder);

  // Release the GPU mutex before pumping events.  wgpuInstanceProcessEvents
  // may deliver Dawn-internal callbacks (e.g. from prior buffer releases) that
  // themselves need the GPU mutex; holding it here would deadlock.
  lock.unlock();

  // Pump Dawn's event loop so it can immediately reclaim the staging buffer's
  // GPU memory and any other pending deferred deallocations.  Without this,
  // ~800 KB staging buffers accumulate each frame until the next GC cycle.
  wgpuInstanceProcessEvents(instance);
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