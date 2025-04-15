#ifndef BUFFER_H
#define BUFFER_H

#include "gpuh.h"
#include <fstream>
#include <future>
#include <string>
#include <vector>

namespace mgpu {
class MGPU {
public:
  void initializeContext();
  void initializeContextAsync(std::function<void()> callback);
  void destroyContext();

  gpu::Context &getContext() { return *ctx; }

private:
  std::unique_ptr<gpu::Context> ctx;
};

class Buffer {
public:
  Buffer(MGPU &mgpu);
  void createBuffer(int bufferSize);
  void readSync(void *outputData, gpu::NumType dType, size_t size, size_t offset = 0);
  void readAsync(void *outputData, gpu::NumType dType, size_t size, size_t offset,
                 std::function<void()> callback);
  void setData(const double *inputData, size_t byteSize);
  void setData(const float *inputData, size_t byteSize);
  void setData(const half *inputData, size_t byteSize);
  void setData(const uint8_t *inputData, size_t byteSize);
  void setData(const uint16_t *inputData, size_t byteSize);
  void setData(const uint32_t *inputData, size_t byteSize);
  void setData(const uint64_t *inputData, size_t byteSize);
  void setData(const int8_t *inputData, size_t byteSize);
  void setData(const int16_t *inputData, size_t byteSize);
  void setData(const int32_t *inputData, size_t byteSize);
  void setData(const int64_t *inputData, size_t byteSize);
  void release();

  gpu::Array bufferData;
  gpu::NumType bufferType;

private:
  MGPU &mgpu;
};

} // namespace mgpu
#endif // BUFFER_H