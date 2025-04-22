#include "../include/minigpu.h"
#ifdef __cplusplus
using namespace mgpu;
using namespace gpu;
extern "C" {
#endif

MGPU minigpu;

void mgpuInitializeContext() {
  minigpu.initializeContext();
  setLogLevel(4);
}

void mgpuInitializeContextAsync(MGPUCallback callback) {

  minigpu.initializeContextAsync(callback);
  setLogLevel(4);
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
    gpu::LOG(kDefLog, kError, "Invalid shader pointer (null)");
    return;
  }

  if (!kernelString) {
    gpu::LOG(kDefLog, kError, "Invalid kernelString pointer (null)");
    return;
  }

  if (strlen(kernelString) == 0) {
    gpu::LOG(kDefLog, kError, "Empty kernel string provided");
    return;
  }

  reinterpret_cast<mgpu::ComputeShader *>(shader)->loadKernelString(
      kernelString);
}

int mgpuHasKernel(MGPUComputeShader *shader) {
  if (shader) {
    return reinterpret_cast<mgpu::ComputeShader *>(shader)->hasKernel();
  } else {
    LOG(kDefLog, kError, "Invalid shader pointer");
    return 0;
  }
}

gpu::NumType mapIntToNumType(int dataType) {
  switch (dataType) {
  case 0:
    return gpu::kf16;
  case 1:
    return gpu::kf32;
  case 2:
    return gpu::kf64;
  case 3:
    return gpu::ki8;
  case 4:
    return gpu::ki16;
  case 5:
    return gpu::ki32;
  case 6:
    return gpu::ki64;
  case 7:
    return gpu::ku8;
  case 8:
    return gpu::ku16;
  case 9:
    return gpu::ku32;
  case 10:
    return gpu::ku64;
  default:
    LOG(kDefLog, kError, "Invalid dataType integer: %d. Defaulting to kf32.",
        dataType);
    return gpu::kf32; // Or handle error more strictly
  }
}

MGPUBuffer *mgpuCreateBuffer(int bufferSize, int dataType) {
  if (bufferSize < 0) {
    LOG(kDefLog, kError, "Invalid bufferSize: %d", bufferSize);
    return nullptr;
  }

  // Map the integer dataType to gpu::NumType
  gpu::NumType originalType = mapIntToNumType(dataType);

  auto *buf = new mgpu::Buffer(minigpu);
  try {
    // Call the C++ createBuffer with size and mapped type
    buf->createBuffer(static_cast<size_t>(bufferSize), originalType);
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "Failed to create buffer: %s", e.what());
    delete buf; // Clean up allocated buffer object
    return nullptr;
  } catch (...) {
    LOG(kDefLog, kError, "Failed to create buffer due to unknown exception");
    delete buf; // Clean up allocated buffer object
    return nullptr;
  }

  return reinterpret_cast<MGPUBuffer *>(buf);
}

void mgpuDestroyBuffer(MGPUBuffer *buffer) {
  if (buffer) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->release();
    delete reinterpret_cast<mgpu::Buffer *>(buffer);
  } else {
    LOG(kDefLog, kError, "Invalid buffer pointer");
  }
}

void mgpuSetBuffer(MGPUComputeShader *shader, int tag, MGPUBuffer *buffer) {
  if (shader && tag >= 0 && buffer) {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->setBuffer(
        tag, *reinterpret_cast<mgpu::Buffer *>(buffer));
  } else {
    LOG(kDefLog, kError, "Invalid shader, or buffer pointer");
  }
}

void mgpuDispatch(MGPUComputeShader *shader, int groupsX, int groupsY,
                  int groupsZ) {
  if (shader) {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->dispatch(groupsX, groupsY,
                                                              groupsZ);
  } else {
    LOG(kDefLog, kError, "Invalid shader or kernel pointer");
  }
}

void mgpuDispatchAsync(MGPUComputeShader *shader, int groupsX, int groupsY,
                       int groupsZ, MGPUCallback callback) {
  if (shader) {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->dispatchAsync(
        groupsX, groupsY, groupsZ, callback);
  } else {
    LOG(kDefLog, kError, "Invalid shader or kernel pointer");
  }
}

void mgpuReadBufferSync(MGPUBuffer *buffer, void *outputData, size_t size,
                        size_t offset) {
  if (buffer && outputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(outputData, kf32, size,
                                                       offset ? offset : 0);
  } else {
    LOG(kDefLog, kError, "Invalid buffer or outputData pointer");
  }
}

// Signed Integer Types
void mgpuReadBufferAsyncInt8(MGPUBuffer *buffer, int8_t *outputData,
                             size_t size, size_t offset,
                             MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(outputData, ki8, size,
                                                          offset, callback);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncInt8: Exception: %s", e.what());
      // Optionally, you could try to invoke the callback here to signal an
      // error state, but the current Buffer::readAsync doesn't have an error
      // callback mechanism. The detached thread might finish without calling
      // the success callback.
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncInt8: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "Invalid buffer, outputData, or callback pointer (int8)");
  }
}

void mgpuReadBufferAsyncInt16(MGPUBuffer *buffer, int16_t *outputData,
                              size_t size, size_t offset,
                              MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          static_cast<void *>(outputData), ki16, size, offset, callback);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncInt16: Exception: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncInt16: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "Invalid buffer, outputData, or callback pointer (int16)");
  }
}

void mgpuReadBufferAsyncInt32(MGPUBuffer *buffer, int32_t *outputData,
                              size_t size, size_t offset,
                              MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          static_cast<void *>(outputData), ki32, size, offset, callback);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncInt32: Exception: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncInt32: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "Invalid buffer, outputData, or callback pointer (int32)");
  }
}

void mgpuReadBufferAsyncInt64(MGPUBuffer *buffer, int64_t *outputData,
                              size_t size, size_t offset,
                              MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          static_cast<void *>(outputData), ki64, size, offset, callback);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncInt64: Exception: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncInt64: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "Invalid buffer, outputData, or callback pointer (int64)");
  }
}

// Unsigned Integer Types
void mgpuReadBufferAsyncUint8(MGPUBuffer *buffer, uint8_t *outputData,
                              size_t size, size_t offset,
                              MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(outputData, ku8, size,
                                                          offset, callback);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncUint8: Exception: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncUint8: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "Invalid buffer, outputData, or callback pointer (uint8)");
  }
}

void mgpuReadBufferAsyncUint16(MGPUBuffer *buffer, uint16_t *outputData,
                               size_t size, size_t offset,
                               MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          static_cast<void *>(outputData), ku16, size, offset, callback);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncUint16: Exception: %s",
          e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncUint16: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "Invalid buffer, outputData, or callback pointer (uint16)");
  }
}

void mgpuReadBufferAsyncUint32(MGPUBuffer *buffer, uint32_t *outputData,
                               size_t size, size_t offset,
                               MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          static_cast<void *>(outputData), ku32, size, offset, callback);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncUint32: Exception: %s",
          e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncUint32: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "Invalid buffer, outputData, or callback pointer (uint32)");
  }
}

void mgpuReadBufferAsyncUint64(MGPUBuffer *buffer, uint64_t *outputData,
                               size_t size, size_t offset,
                               MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          static_cast<void *>(outputData), ku64, size, offset, callback);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncUint64: Exception: %s",
          e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncUint64: Unknown exception");
    }
  } else {
    // Note: Original log message said (uint32), corrected to (uint64)
    LOG(kDefLog, kError,
        "Invalid buffer, outputData, or callback pointer (uint64)");
  }
}

// Floating Point Number Types
void mgpuReadBufferAsyncFloat(MGPUBuffer *buffer, float *outputData,
                              size_t size, size_t offset,
                              MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          static_cast<void *>(outputData), kf32, size, offset, callback);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncFloat: Exception: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncFloat: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "Invalid buffer, outputData, or callback pointer (float)");
  }
}

void mgpuReadBufferAsyncDouble(MGPUBuffer *buffer, double *outputData,
                               size_t size, size_t offset,
                               MGPUCallback callback) {
  if (buffer && outputData && callback) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readAsync(
          static_cast<void *>(outputData), kf64, size, offset, callback);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncDouble: Exception: %s",
          e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferAsyncDouble: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "Invalid buffer, outputData, or callback pointer (double)");
  }
}

void mgpuReadBufferSyncInt8(MGPUBuffer *buffer, int8_t *outputData,
                            size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(
          outputData, gpu::ki8, elementCount, elementOffset);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncInt8: Exception: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncInt8: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "mgpuReadBufferSyncInt8: Invalid buffer or outputData pointer");
  }
}

void mgpuReadBufferSyncUint8(MGPUBuffer *buffer, uint8_t *outputData,
                             size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(
          outputData, gpu::ku8, elementCount, elementOffset);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncUint8: Exception: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncUint8: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "mgpuReadBufferSyncUint8: Invalid buffer or outputData pointer");
  }
}

void mgpuReadBufferSyncInt16(MGPUBuffer *buffer, int16_t *outputData,
                             size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(
          outputData, gpu::ki16, elementCount, elementOffset);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncInt16: Exception: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncInt16: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "mgpuReadBufferSyncInt16: Invalid buffer or outputData pointer");
  }
}

void mgpuReadBufferSyncUint16(MGPUBuffer *buffer, uint16_t *outputData,
                              size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(
          outputData, gpu::ku16, elementCount, elementOffset);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncUint16: Exception: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncUint16: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "mgpuReadBufferSyncUint16: Invalid buffer or outputData pointer");
  }
}

void mgpuReadBufferSyncInt32(MGPUBuffer *buffer, int32_t *outputData,
                             size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(
          outputData, gpu::ki32, elementCount, elementOffset);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncInt32: Exception: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncInt32: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "mgpuReadBufferSyncInt32: Invalid buffer or outputData pointer");
  }
}

void mgpuReadBufferSyncUint32(MGPUBuffer *buffer, uint32_t *outputData,
                              size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(
          outputData, gpu::ku32, elementCount, elementOffset);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncUint32: Exception: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncUint32: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "mgpuReadBufferSyncUint32: Invalid buffer or outputData pointer");
  }
}

void mgpuReadBufferSyncInt64(MGPUBuffer *buffer, int64_t *outputData,
                             size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(
          outputData, gpu::ki64, elementCount, elementOffset);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncInt64: Exception: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncInt64: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "mgpuReadBufferSyncInt64: Invalid buffer or outputData pointer");
  }
}

void mgpuReadBufferSyncUint64(MGPUBuffer *buffer, uint64_t *outputData,
                              size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(
          outputData, gpu::ku64, elementCount, elementOffset);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncUint64: Exception: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncUint64: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "mgpuReadBufferSyncUint64: Invalid buffer or outputData pointer");
  }
}

void mgpuReadBufferSyncFloat32(MGPUBuffer *buffer, float *outputData,
                               size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(
          outputData, gpu::kf32, elementCount, elementOffset);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncFloat32: Exception: %s",
          e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncFloat32: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "mgpuReadBufferSyncFloat32: Invalid buffer or outputData pointer");
  }
}

void mgpuReadBufferSyncFloat64(MGPUBuffer *buffer, double *outputData,
                               size_t elementCount, size_t elementOffset) {
  if (buffer && outputData) {
    try {
      reinterpret_cast<mgpu::Buffer *>(buffer)->readSync(
          outputData, gpu::kf64, elementCount, elementOffset);
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncFloat64: Exception: %s",
          e.what());
    } catch (...) {
      LOG(kDefLog, kError, "mgpuReadBufferSyncFloat64: Unknown exception");
    }
  } else {
    LOG(kDefLog, kError,
        "mgpuReadBufferSyncFloat64: Invalid buffer or outputData pointer");
  }
}

// Signed Integer Types
void mgpuSetBufferDataInt8(MGPUBuffer *buffer, const int8_t *inputData,
                           size_t byteSize) {
  if (buffer && inputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->setData(inputData, byteSize);
  } else {
    LOG(kDefLog, kError, "Invalid buffer or inputData pointer");
  }
}

void mgpuSetBufferDataInt16(MGPUBuffer *buffer, const int16_t *inputData,
                            size_t byteSize) {
  if (buffer && inputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->setData(inputData, byteSize);
  } else {
    LOG(kDefLog, kError, "Invalid buffer or inputData pointer");
  }
}

void mgpuSetBufferDataInt32(MGPUBuffer *buffer, const int32_t *inputData,
                            size_t byteSize) {
  if (buffer && inputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->setData(inputData, byteSize);
  } else {
    LOG(kDefLog, kError, "Invalid buffer or inputData pointer");
  }
}

void mgpuSetBufferDataInt64(MGPUBuffer *buffer, const int64_t *inputData,
                            size_t byteSize) {
  if (buffer && inputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->setData(inputData, byteSize);
  } else {
    LOG(kDefLog, kError, "Invalid buffer or inputData pointer");
  }
}

// Unsigned Integer Types
void mgpuSetBufferDataUint8(MGPUBuffer *buffer, const uint8_t *inputData,
                            size_t byteSize) {
  if (buffer && inputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->setData(inputData, byteSize);
  } else {
    LOG(kDefLog, kError, "Invalid buffer or inputData pointer");
  }
}

void mgpuSetBufferDataUint16(MGPUBuffer *buffer, const uint16_t *inputData,
                             size_t byteSize) {
  if (buffer && inputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->setData(inputData, byteSize);
  } else {
    LOG(kDefLog, kError, "Invalid buffer or inputData pointer");
  }
}

void mgpuSetBufferDataUint32(MGPUBuffer *buffer, const uint32_t *inputData,
                             size_t byteSize) {
  if (buffer && inputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->setData(inputData, byteSize);
  } else {
    LOG(kDefLog, kError, "Invalid buffer or inputData pointer");
  }
}

void mgpuSetBufferDataUint64(MGPUBuffer *buffer, const uint64_t *inputData,
                             size_t byteSize) {
  if (buffer && inputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->setData(inputData, byteSize);
  } else {
    LOG(kDefLog, kError, "Invalid buffer or inputData pointer");
  }
}

// Floating Point Number Types
void mgpuSetBufferDataFloat(MGPUBuffer *buffer, const float *inputData,
                            size_t byteSize) {
  if (buffer && inputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->setData(inputData, byteSize);
  } else {
    LOG(kDefLog, kError, "Invalid buffer or inputData pointer");
  }
}

void mgpuSetBufferDataDouble(MGPUBuffer *buffer, const double *inputData,
                             size_t byteSize) {
  if (buffer && inputData) {
    reinterpret_cast<mgpu::Buffer *>(buffer)->setData(inputData, byteSize);
  } else {
    LOG(kDefLog, kError, "Invalid buffer or inputData pointer");
  }
}

#ifdef __cplusplus
}
#endif // extern "C"
