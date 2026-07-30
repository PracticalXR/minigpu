#include "../include/minigpu.h"
#include "../include/log.h"
#ifdef _WIN32
#include <dxgi1_4.h>
#pragma comment(lib, "dxgi.lib")
#endif
#ifdef __cplusplus
using namespace mgpu;
extern "C" {
#endif

MGPU minigpu;

mgpu::LogLevel level = mgpu::LOG_INFO;

void mgpuSetLogCallback(MGPULogCallback callback) {
  SET_LOG_CALLBACK(callback);
}

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

void mgpuInitializeContext() {
  minigpu.initializeContext();
}

void mgpuInitializeContextAsync(MGPUCallback callback) {
  minigpu.initializeContextAsync(callback);
}

void mgpuDestroyContext() { minigpu.destroyContext(); }

MGPUComputeShader *mgpuCreateComputeShader() {
  return reinterpret_cast<MGPUComputeShader *>(
      new mgpu::ComputeShader(minigpu));
}

void mgpuDestroyComputeShader(MGPUComputeShader *shader) {
  delete reinterpret_cast<mgpu::ComputeShader *>(shader);
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

void mgpuDestroyBuffer(MGPUBuffer *buffer) {
  if (buffer) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->release();
    delete reinterpret_cast<mgpu::Buffer *>(buffer);
  }
}

// ── Multi-GPU context handles ───────────────────────────────────────────────
// An MGPUContextHandle is simply a heap-allocated mgpu::MGPU instance.  All
// per-object entry points (setBuffer/dispatch/read/write/destroy/...) already
// route through the MGPU& the object captured at creation, so only creation
// needs these handle-aware variants.

MGPUContextHandle *mgpuCreateContextHandle(const char *adapterFilter) {
  auto *m = new mgpu::MGPU();
  if (adapterFilter && adapterFilter[0] != '\0') {
    m->setAdapterFilter(adapterFilter);
  }
  return reinterpret_cast<MGPUContextHandle *>(m);
}

void mgpuContextInitializeAsync(MGPUContextHandle *handle,
                                MGPUCallback callback) {
  if (handle) {
    reinterpret_cast<mgpu::MGPU *>(handle)->initializeContextAsync(callback);
  }
}

void mgpuDestroyContextHandle(MGPUContextHandle *handle) {
  if (handle) {
    auto *m = reinterpret_cast<mgpu::MGPU *>(handle);
    m->destroyContext();
    delete m;
  }
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
  if (buffer && inputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->writeBytesAt(inputData, byteSize,
                                                           dstByteOffset);
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