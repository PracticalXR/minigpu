#ifndef BUFFER_H
#define BUFFER_H

#include "gpuh.h"
#include <cstddef> // For size_t
#include <fstream>
#include <functional> // For std::function
#include <future>
#include <memory> // For std::unique_ptr
#include <string>
#include <vector>

namespace mgpu {
class MGPU {
public:
  void initializeContext();
  void initializeContextAsync(std::function<void()> callback);
  void destroyContext();

  // Provide const access if context shouldn't be modified externally
  const gpu::Context &getContext() const {
    if (!ctx)
      throw std::runtime_error("GPU context not initialized");
    return *ctx;
  }
  // Provide non-const access if needed, maybe rename to getMutableContext
  gpu::Context &getContext() {
    if (!ctx)
      throw std::runtime_error("GPU context not initialized");
    return *ctx;
  }

private:
  std::unique_ptr<gpu::Context> ctx;
};

class Buffer {
public:
  // --- Constructor / Destructor / Move Semantics ---
  explicit Buffer(MGPU &mgpu); // Use explicit for single-argument constructors
  ~Buffer();                   // Destructor to manage resource release

  // Delete copy constructor and assignment operator to prevent accidental
  // copies
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  // Declare move constructor and assignment operator
  Buffer(Buffer &&other) noexcept;
  Buffer &operator=(Buffer &&other) noexcept;

  // --- Core Operations ---
  // Use size_t for sizes/counts consistently
  void createBuffer(size_t sizeParam, gpu::NumType requestedDataType);
  void readSync(void *outputData, gpu::NumType readAsType,
                size_t readElementCount, size_t readElementOffset = 0);
  void readAsync(void *outputData, gpu::NumType readAsType,
                 size_t readElementCount, size_t readElementOffset,
                 std::function<void()> callback);

  // Update setData signatures to take number of elements (numElements)
  void setData(const double *inputData, size_t numElements);
  void setData(const float *inputData, size_t numElements);
  // void setData(const half *inputData, size_t numElements);
  void setData(const uint8_t *inputData, size_t numElements);
  void setData(const uint16_t *inputData, size_t numElements);
  void setData(const uint32_t *inputData, size_t numElements);
  void setData(const uint64_t *inputData, size_t numElements);
  void setData(const int8_t *inputData, size_t numElements);
  void setData(const int16_t *inputData, size_t numElements);
  void setData(const int32_t *inputData, size_t numElements);
  void setData(const int64_t *inputData, size_t numElements);

  void release(); // Manually release resources if needed before destruction

  // --- Public Members (Consider making some private/protected with accessors)
  // ---
  gpu::Array bufferData; // Holds the WGPUBuffer and its physical size/usage
  gpu::NumType bufferType =
      gpu::kUnknown; // The internal storage type (e.g., kf32, ki32, ku32)
  gpu::NumType getOriginalDataType() const; // Get the logical data type
  size_t length = 0; // Number of logical elements in the buffer
  bool isPacked =
      false; // True if internal storage differs from logical (e.g., i8->i32)

private:
  MGPU &mgpu; // Reference to the main GPU context manager

  // --- Private Helper Methods ---
  // Helper for setData methods
  void ensureBuffer(size_t requiredLogicalLength,
                    gpu::NumType targetOriginalDataType);

  // Helpers for readSync
  bool validateReadPreconditions(const void *outputData,
                                 gpu::NumType readAsType) const;
  bool validateBufferStateForRead(gpu::NumType readAsType,
                                  gpu::NumType expectedInternalType,
                                  bool expectedPackedState) const;
  bool calculateClampedReadRange(gpu::NumType readAsType,
                                 size_t readElementOffset,
                                 size_t &inOutClampedElementCount) const;
  void readSyncDirect(void *outputData, gpu::NumType readAsType,
                      size_t readElementCount, size_t readElementOffset);
  void readSyncPackedSmallTypes(void *outputData, gpu::NumType readAsType,
                                size_t readElementCount,
                                size_t readElementOffset);
  void readSyncExpandedFloat64(void *outputData, size_t readElementCount,
                               size_t readElementOffset);
  void readSyncExpandedInt64(void *outputData, size_t readElementCount,
                               size_t readElementOffset);
  void readSyncExpandedUint64(void *outputData, size_t readElementCount,
                               size_t readElementOffset);
};

} // namespace mgpu
#endif // BUFFER_H