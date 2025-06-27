#include "../include/minigpu.h"
#include "../include/log.h"
#ifdef __cplusplus
using namespace mgpu;
extern "C" {
#endif

MGPU minigpu;

mgpu::LogLevel level = mgpu::LOG_NONE;

void mgpuInitializeContext() {
  SET_LOG_LEVEL(level);
  minigpu.initializeContext();
}

void mgpuInitializeContextAsync(MGPUCallback callback) {
  SET_LOG_LEVEL(level);
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

MGPUBuffer *mgpuCreateBuffer(int elementCount, int dataType) {
  LOG_INFO("mgpuCreateBuffer: elementCount=%d, dataType=%d", elementCount,
           dataType);

  BufferDataType mappedType = mapIntToBufferDataType(dataType);
  LOG_INFO("mappedType=%d", (int)mappedType);

  auto *buf = new mgpu::Buffer(minigpu);
  try {
    if (needsPacking(mappedType)) {
      LOG_INFO("Creating packed buffer with elementCount=%d", elementCount);
      buf->createBuffer(static_cast<size_t>(elementCount), mappedType);
    } else {
      size_t byteSize = elementCount;
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

void mgpuSetBuffer(MGPUComputeShader *shader, int tag, MGPUBuffer *buffer) {
  if (shader && tag >= 0 && buffer) {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->setBuffer(
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
#ifdef __cplusplus
}
#endif // extern "C"