#ifndef BUFFER_H
#define BUFFER_H

#include "gpuh.h"
#include <fstream>
#include <future>
#include <string>
#include <vector>
#include <cstddef> // For size_t
#include <functional> // For std::function
#include <memory> // For std::unique_ptr

namespace mgpu {
class MGPU {
public:
  void initializeContext();
  void initializeContextAsync(std::function<void()> callback);
  void destroyContext();

  // Provide const access if context shouldn't be modified externally
  const gpu::Context &getContext() const {
    if (!ctx) throw std::runtime_error("GPU context not initialized");
    return *ctx;
  }
  // Provide non-const access if needed, maybe rename to getMutableContext
  gpu::Context &getContext() {
     if (!ctx) throw std::runtime_error("GPU context not initialized");
     return *ctx;
  }


private:
  std::unique_ptr<gpu::Context> ctx;
};

class Buffer {
public:
  // --- Constructor / Destructor / Move Semantics ---
  explicit Buffer(MGPU &mgpu); // Use explicit for single-argument constructors
  ~Buffer(); // Destructor to manage resource release

  // Delete copy constructor and assignment operator to prevent accidental copies
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  // Declare move constructor and assignment operator
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;

  // --- Core Operations ---
  // Use size_t for sizes/counts consistently
  void createBuffer(size_t requestedByteSize, gpu::NumType dataType);
  void readSync(void *outputData, gpu::NumType readAsType, size_t readElementCount, size_t readElementOffset = 0);
  void readAsync(void *outputData, gpu::NumType readAsType, size_t readElementCount, size_t readElementOffset,
                 std::function<void()> callback);

  // Update setData signatures to take number of elements (numElements)
  void setData(const double *inputData, size_t numElements);
  void setData(const float *inputData, size_t numElements);
  // void setData(const half *inputData, size_t numElements); // Assuming 'half' type exists
  void setData(const uint8_t *inputData, size_t numElements);
  void setData(const uint16_t *inputData, size_t numElements);
  void setData(const uint32_t *inputData, size_t numElements);
  void setData(const uint64_t *inputData, size_t numElements);
  void setData(const int8_t *inputData, size_t numElements);
  void setData(const int16_t *inputData, size_t numElements);
  void setData(const int32_t *inputData, size_t numElements);
  void setData(const int64_t *inputData, size_t numElements);

  void release(); // Manually release resources if needed before destruction

  // --- Public Members (Consider making some private/protected with accessors) ---
  gpu::Array bufferData;     // Holds the WGPUBuffer and its physical size/usage
  gpu::NumType bufferType = gpu::kUnknown; // The type of data stored (e.g., kf32, ki32)
  size_t length = 0;         // Number of logical elements in the buffer
  bool isPacked = false;     // True if buffer holds unpacked i32 derived from i8

private:
  MGPU &mgpu; // Reference to the main GPU context manager

  // Helper for setData methods
  void ensureBuffer(size_t requiredByteSize, gpu::NumType dataType, size_t logicalLength, bool packedState);
};

} // namespace mgpu
#endif // BUFFER_H