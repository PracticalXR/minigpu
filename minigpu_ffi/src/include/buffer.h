#ifndef BUFFER_H
#define BUFFER_H

#include "webgpu.h"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

namespace mgpu {

// Forward declarations
class MGPU;

// no gpu library dependency
struct BufferData {
  WGPUBuffer buffer = nullptr;
  WGPUBufferUsage usage = WGPUBufferUsage_None;
  size_t size = 0;
};

// no gpu library dependency
enum BufferDataType {
  kFloat32,
  kInt32,
  kUInt32,
  kInt8,
  kUInt8,
  kInt16,
  kUInt16,
  kFloat64,
  kInt64,
  kUInt64,
  kUnknownType
};

// managed by MGPU
class WebGPUThread {
private:
  std::thread worker;
  std::queue<std::function<void()>> tasks;
  std::mutex queueMutex;
  std::condition_variable condition;
  std::atomic<bool> stop{false};

public:
  WebGPUThread();
  ~WebGPUThread();

  // returns immediately
  void enqueueAsync(std::function<void()> task);

  // waits for completion
  template <typename T> T enqueueSync(std::function<T()> task) {
    auto promise = std::make_shared<std::promise<T>>();
    auto future = promise->get_future();

    enqueueAsync([task, promise]() {
      try {
        if constexpr (std::is_void_v<T>) {
          task();
          promise->set_value();
        } else {
          promise->set_value(task());
        }
      } catch (...) {
        promise->set_exception(std::current_exception());
      }
    });

    return future.get();
  }
};

class MGPU {
public:
  // Define the Context struct here
  struct Context {
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    bool initialized = false;
  };

  void initializeContext();
  void initializeContextAsync(std::function<void()> callback);
  void destroyContext();
  bool isDeviceValid();
  void ensureDeviceValid();

  WebGPUThread &getWebGPUThread() { return webgpuThread; }

  WGPUDevice getDevice() const;
  WGPUQueue getQueue() const;
  WGPUInstance getInstance() const;

  std::mutex &getGpuMutex() { return gpuOperationMutex; }

private:
  std::unique_ptr<Context> ctx;
  std::mutex gpuOperationMutex;
  WebGPUThread webgpuThread;
};
class Buffer {
public:
  explicit Buffer(MGPU &mgpu_ref);
  ~Buffer();

  Buffer(Buffer &&other) noexcept;
  Buffer &operator=(Buffer &&other) noexcept;

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  void createBuffer(size_t elementCount, BufferDataType dataType);

  void write(const float *inputData, size_t elementCount);
  void write(const int32_t *inputData, size_t elementCount);
  void write(const uint32_t *inputData, size_t elementCount);
  void write(const int8_t *inputData, size_t elementCount);
  void write(const uint8_t *inputData, size_t elementCount);
  void write(const int16_t *inputData, size_t elementCount);
  void write(const uint16_t *inputData, size_t elementCount);
  void write(const double *inputData, size_t elementCount);
  void write(const int64_t *inputData, size_t elementCount);
  void write(const uint64_t *inputData, size_t elementCount);

  void read(float *outputData, size_t elementCount, size_t offset = 0);
  void read(int32_t *outputData, size_t elementCount, size_t offset = 0);
  void read(uint32_t *outputData, size_t elementCount, size_t offset = 0);
  void read(int8_t *outputData, size_t elementCount, size_t offset = 0);
  void read(uint8_t *outputData, size_t elementCount, size_t offset = 0);
  void read(int16_t *outputData, size_t elementCount, size_t offset = 0);
  void read(uint16_t *outputData, size_t elementCount, size_t offset = 0);
  void read(double *outputData, size_t elementCount, size_t offset = 0);
  void read(int64_t *outputData, size_t elementCount, size_t offset = 0);
  void read(uint64_t *outputData, size_t elementCount, size_t offset = 0);

  void readAsync(float *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(int32_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(uint32_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(int8_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(uint8_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(int16_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(uint16_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(double *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(int64_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);
  void readAsync(uint64_t *outputData, size_t elementCount, size_t offset,
                 std::function<void()> callback);

  void release();

  size_t getLength() const { return elementCount; }
  size_t getSize() const { return bufferData.size; }
  BufferDataType getDataType() const { return dataType; }
  WGPUBuffer getWGPUBuffer() const { return bufferData.buffer; }

  BufferData bufferData;

private:
  MGPU &mgpu;
  BufferDataType dataType = kUnknownType;
  size_t elementCount = 0;
  bool isPacked = false; // For types that need packing (8/16-bit, 64-bit)

  void releaseInternal();
  size_t getElementSize(BufferDataType type) const;
  bool needsPacking(BufferDataType type) const;

  template <typename T>
  void writeDirect(const T *inputData, size_t byteSize, BufferDataType type);

  template <typename T>
  void writePacked(const T *inputData, size_t byteSize, BufferDataType type);

  template <typename T>
  void readDirect(T *outputData, size_t elementCount, size_t offset);

  template <typename T>
  void readDirectChunk(T *outputData, size_t elementCount, size_t offset);

  template <typename T>
  void readPacked(T *outputData, size_t elementCount, size_t offset);

  template <typename T>
  void readAsyncImpl(T *outputData, size_t elementCount, size_t offset,
                     BufferDataType type, std::function<void()> callback);
};

} // namespace mgpu

#endif // BUFFER_H