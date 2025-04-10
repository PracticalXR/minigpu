#ifndef MINIGPU_H
#define MINIGPU_H

#include "export.h"
#include "stdint.h"
#ifdef __cplusplus
#include "../include/buffer.h"
#include "../include/compute_shader.h"

extern "C" {
#endif

typedef struct MGPUComputeShader MGPUComputeShader;
typedef struct MGPUBuffer MGPUBuffer;

EXPORT void mgpuInitializeContext();
typedef void (*MGPUCallback)(void);
EXPORT void mgpuInitializeContextAsync(MGPUCallback callback);
EXPORT void mgpuDestroyContext();
EXPORT MGPUComputeShader *mgpuCreateComputeShader();
EXPORT void mgpuDestroyComputeShader(MGPUComputeShader *shader);
EXPORT void mgpuLoadKernel(MGPUComputeShader *shader, const char *kernelString);
EXPORT int mgpuHasKernel(MGPUComputeShader *shader);
EXPORT MGPUBuffer *mgpuCreateBuffer(int bufferSize);
EXPORT void mgpuDestroyBuffer(MGPUBuffer *buffer);
EXPORT void mgpuSetBuffer(MGPUComputeShader *shader, int tag,
                          MGPUBuffer *buffer);
EXPORT void mgpuDispatch(MGPUComputeShader *shader, int groupsX, int groupsY,
                         int groupsZ);
EXPORT void mgpuDispatchAsync(MGPUComputeShader *shader, int groupsX,
                              int groupsY, int groupsZ, MGPUCallback callback);
EXPORT void mgpuReadBufferSync(MGPUBuffer *buffer, void *outputData,
                               size_t size, size_t offset);
// Signed Integer Types
EXPORT void mgpuReadBufferAsyncInt8(MGPUBuffer *buffer, int8_t *outputData,
                                    size_t size, size_t offset,
                                    MGPUCallback callback);
EXPORT void mgpuReadBufferAsyncInt16(MGPUBuffer *buffer, int16_t *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
EXPORT void mgpuReadBufferAsyncInt32(MGPUBuffer *buffer, int32_t *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
EXPORT void mgpuReadBufferAsyncInt64(MGPUBuffer *buffer, int64_t *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
// Unsigned Integer Types
EXPORT void mgpuReadBufferAsyncUint8(MGPUBuffer *buffer, uint8_t *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
EXPORT void mgpuReadBufferAsyncUint16(MGPUBuffer *buffer, uint16_t *outputData,
                                      size_t size, size_t offset,
                                      MGPUCallback callback);
EXPORT void mgpuReadBufferAsyncUint32(MGPUBuffer *buffer, uint32_t *outputData,
                                      size_t size, size_t offset,
                                      MGPUCallback callback);
// Floating Point Number Types
EXPORT void mgpuReadBufferAsyncFloat(MGPUBuffer *buffer, float *outputData,
                                     size_t size, size_t offset,
                                     MGPUCallback callback);
EXPORT void mgpuReadBufferAsyncDouble(MGPUBuffer *buffer, double *outputData,
                                      size_t size, size_t offset,
                                      MGPUCallback callback);
// Signed Integer Types
EXPORT void mgpuSetBufferDataInt8(MGPUBuffer *buffer, const int8_t *inputData,
                                  size_t byteSize);
EXPORT void mgpuSetBufferDataInt16(MGPUBuffer *buffer, const int16_t *inputData,
                                   size_t byteSize);
EXPORT void mgpuSetBufferDataInt32(MGPUBuffer *buffer, const int32_t *inputData,
                                   size_t byteSize);
EXPORT void mgpuSetBufferDataInt64(MGPUBuffer *buffer, const int64_t *inputData,
                                   size_t byteSize);
// Unsigned Integer Types
EXPORT void mgpuSetBufferDataUint8(MGPUBuffer *buffer, const uint8_t *inputData,
                                   size_t byteSize);
EXPORT void mgpuSetBufferDataUint16(MGPUBuffer *buffer,
                                    const uint16_t *inputData, size_t byteSize);
EXPORT void mgpuSetBufferDataUint32(MGPUBuffer *buffer,
                                    const uint32_t *inputData, size_t byteSize);
// Floating Point Number Types
EXPORT void mgpuSetBufferDataFloat(MGPUBuffer *buffer, const float *inputData,
                                   size_t byteSize);
EXPORT void mgpuSetBufferDataDouble(MGPUBuffer *buffer, const double *inputData,
                                    size_t byteSize);

#ifdef __cplusplus
}
#endif

#endif // MINIGPU_H
