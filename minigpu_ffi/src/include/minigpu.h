#ifndef MINIGPU_H
#define MINIGPU_H

#include "export.h"
#include "stdint.h"
#ifdef __cplusplus
#include "../include/buffer.h"
#include "../include/compute_shader.h"

#define LOG(level, msg, ...)                                                   \
  do {                                                                         \
    /* Simple logging or remove entirely for performance */                    \
  } while (0)

extern "C" {
#endif

typedef struct MGPUComputeShader MGPUComputeShader;
typedef struct MGPUBuffer MGPUBuffer;
typedef void (*MGPUCallback)(void);

EXPORT void mgpuInitializeContext();
EXPORT void mgpuInitializeContextAsync(MGPUCallback callback);
EXPORT void mgpuDestroyContext();
EXPORT MGPUComputeShader *mgpuCreateComputeShader();
EXPORT void mgpuDestroyComputeShader(MGPUComputeShader *shader);
EXPORT void mgpuLoadKernel(MGPUComputeShader *shader, const char *kernelString);
EXPORT int mgpuHasKernel(MGPUComputeShader *shader);
EXPORT MGPUBuffer *mgpuCreateBuffer(int bufferSize, int dataType);
EXPORT void mgpuDestroyBuffer(MGPUBuffer *buffer);
EXPORT void mgpuSetBuffer(MGPUComputeShader *shader, int tag,
                          MGPUBuffer *buffer);
EXPORT void mgpuCreateKernel(MGPUComputeShader *shader, int groupsX,
                             int groupsY, int groupsZ);
EXPORT void mgpuDispatch(MGPUComputeShader *shader, int groupsX, int groupsY,
                         int groupsZ);
EXPORT void mgpuDispatchAsync(MGPUComputeShader *shader, int groupsX,
                              int groupsY, int groupsZ, MGPUCallback callback);

// Signed Integer Types
EXPORT void mgpuReadAsyncInt8(MGPUBuffer *buffer, int8_t *outputData,
                                    size_t size, size_t offset,
                                    MGPUCallback callback);
EXPORT void mgpuReadAsyncInt16(MGPUBuffer *buffer, int16_t *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
EXPORT void mgpuReadAsyncInt32(MGPUBuffer *buffer, int32_t *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
EXPORT void mgpuReadAsyncInt64(MGPUBuffer *buffer, int64_t *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
// Unsigned Integer Types
EXPORT void mgpuReadAsyncUint8(MGPUBuffer *buffer, uint8_t *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
EXPORT void mgpuReadAsyncUint16(MGPUBuffer *buffer, uint16_t *outputData,
                                      size_t size, size_t offset,
                                      MGPUCallback callback);
EXPORT void mgpuReadAsyncUint32(MGPUBuffer *buffer, uint32_t *outputData,
                                      size_t size, size_t offset,
                                      MGPUCallback callback);
EXPORT void mgpuReadAsyncUint64(MGPUBuffer *buffer, uint64_t *outputData,
                                      size_t size, size_t offset,
                                      MGPUCallback callback);
// Floating Point Number Types
EXPORT void mgpuReadAsyncFloat(MGPUBuffer *buffer, float *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
EXPORT void mgpuReadAsyncDouble(MGPUBuffer *buffer, double *outputData,
                                      size_t size, size_t offset,
                                      MGPUCallback callback);
// Sync Read Methods
EXPORT void mgpuReadSync(MGPUBuffer *buffer, void *outputData,
                               size_t size, size_t offset);
EXPORT void mgpuReadSyncInt8(MGPUBuffer *buffer, int8_t *outputData,
                                   size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncUint8(MGPUBuffer *buffer, uint8_t *outputData,
                                    size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncInt16(MGPUBuffer *buffer, int16_t *outputData,
                                    size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncUint16(MGPUBuffer *buffer, uint16_t *outputData,
                                     size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncInt32(MGPUBuffer *buffer, int32_t *outputData,
                                    size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncUint32(MGPUBuffer *buffer, uint32_t *outputData,
                                     size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncInt64(MGPUBuffer *buffer, int64_t *outputData,
                                    size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncUint64(MGPUBuffer *buffer, uint64_t *outputData,
                                     size_t elementCount, size_t elementOffset);
EXPORT void mgpuReadSyncFloat32(MGPUBuffer *buffer, float *outputData,
                                      size_t elementCount,
                                      size_t elementOffset);
EXPORT void mgpuReadSyncFloat64(MGPUBuffer *buffer, double *outputData,
                                      size_t elementCount,
                                      size_t elementOffset);
// Signed Integer Types
EXPORT void mgpuWriteInt8(MGPUBuffer *buffer, const int8_t *inputData,
                                  size_t byteSize);
EXPORT void mgpuWriteInt16(MGPUBuffer *buffer, const int16_t *inputData,
                                   size_t byteSize);
EXPORT void mgpuWriteInt32(MGPUBuffer *buffer, const int32_t *inputData,
                                   size_t byteSize);
EXPORT void mgpuWriteInt64(MGPUBuffer *buffer, const int64_t *inputData,
                                   size_t byteSize);
// Unsigned Integer Types
EXPORT void mgpuWriteUint8(MGPUBuffer *buffer, const uint8_t *inputData,
                                   size_t byteSize);
EXPORT void mgpuWriteUint16(MGPUBuffer *buffer,
                                    const uint16_t *inputData, size_t byteSize);
EXPORT void mgpuWriteUint32(MGPUBuffer *buffer,
                                    const uint32_t *inputData, size_t byteSize);
EXPORT void mgpuWriteUint64(MGPUBuffer *buffer,
                                    const uint64_t *inputData, size_t byteSize);
// Floating Point Number Types
EXPORT void mgpuWriteFloat(MGPUBuffer *buffer, const float *inputData,
                                   size_t byteSize);
EXPORT void mgpuWriteDouble(MGPUBuffer *buffer, const double *inputData,
                                    size_t byteSize);
#ifdef __cplusplus
}
#endif

#endif // MINIGPU_H
