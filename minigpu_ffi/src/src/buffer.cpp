#include "../include/buffer.h"
#include "../include/compute_shader.h"
#include "../include/conversion_kernels.h"
#include "../include/gpuh.h"

#include <stdexcept>
#include <string>

using namespace gpu;

namespace mgpu {

// --- MGPU Context Management ---

void MGPU::initializeContext() {
  try {
    ctx = std::make_unique<gpu::Context>(std::move(gpu::createContext()));
    LOG(kDefLog, kInfo, "GPU context initialized successfully.");
  } catch (const std::exception &ex) {
    // Log and potentially re-throw or handle context creation failure
    LOG(kDefLog, kError, "Failed to create GPU context: %s", ex.what());
  }
}

void MGPU::initializeContextAsync(std::function<void()> callback) {
  std::thread([this, callback]() {
    try {
      initializeContext();
      if (ctx && callback) { // Only call callback if context is valid
        callback();
      } else if (!ctx) {
        LOG(kDefLog, kError,
            "Async context initialization failed, callback skipped.");
      }
    } catch (const std::exception &ex) {
      // Log error from the async thread
      LOG(kDefLog, kError, "Async context initialization failed: %s",
          ex.what());
    }
  }).detach();
}

void MGPU::destroyContext() {
  if (ctx) {
    ctx.reset(); // Use reset() for unique_ptr to destroy the managed object
    LOG(kDefLog, kInfo, "GPU context destroyed successfully.");
  } else {
    LOG(kDefLog, kWarn,
        "Attempted to destroy GPU context, but it was already destroyed or not "
        "initialized.");
  }
}

// --- Buffer Implementation ---

Buffer::Buffer(MGPU &mgpu_ref) : mgpu(mgpu_ref) {
  // Initialize members to safe defaults
  bufferData.buffer = nullptr;
  bufferData.usage = WGPUBufferUsage_None;
  bufferData.size = 0;
  bufferType = kUnknown;
  length = 0;
  isPacked = false;
}

Buffer::~Buffer() { release(); }

Buffer::Buffer(Buffer &&other) noexcept
    : mgpu(other.mgpu), bufferData(other.bufferData),
      bufferType(other.bufferType), isPacked(other.isPacked),
      length(other.length) {
  other.bufferData.buffer = nullptr;
  other.bufferData.size = 0;
  other.length = 0;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (this != &other) {
    release();
    // mgpu reference doesn't need to be moved/assigned
    bufferData = other.bufferData;
    bufferType = other.bufferType;
    isPacked = other.isPacked;
    length = other.length;
    other.bufferData.buffer = nullptr;
    other.bufferData.size = 0;
    other.length = 0;
  }
  return *this;
}

// sizeParam interpretation depends on requestedDataType:
// - For ki8/ku8/ki16/ku16: sizeParam is the number of logical elements.
// - For kf64: sizeParam is the number of logical f64 elements.
// - For other types: sizeParam is the requested physical size in bytes.
void Buffer::createBuffer(size_t sizeParam, gpu::NumType requestedDataType) {

  size_t physicalByteSize = 0;
  size_t logicalLength = 0;
  gpu::NumType internalDataType = requestedDataType;
  bool packedState = false;
  size_t logicalElementSizeBytes = 0;
  size_t internalElementSizeBytes = 0;

  switch (requestedDataType) {
  case ki8:
  case ku8:
    logicalLength = sizeParam;
    // Internal storage is unpacked i32/u32
    internalDataType = (requestedDataType == ki8) ? ki32 : ku32;
    internalElementSizeBytes = sizeof(int32_t);
    physicalByteSize = logicalLength * internalElementSizeBytes;
    packedState = true;
    LOG(kDefLog, kInfo,
        "Creating buffer for %zu logical %s elements. Internal storage: %s, "
        "isPacked=%d, physical_size=%zu bytes.",
        logicalLength, gpu::toString(requestedDataType).c_str(),
        gpu::toString(internalDataType).c_str(), packedState, physicalByteSize);
    break;

  case ki16:
  case ku16:
    logicalLength = sizeParam;
    // Internal storage is unpacked i32/u32
    internalDataType = (requestedDataType == ki16) ? ki32 : ku32;
    internalElementSizeBytes = sizeof(int32_t);
    physicalByteSize = logicalLength * internalElementSizeBytes;
    packedState = true;
    LOG(kDefLog, kInfo,
        "Creating buffer for %zu logical %s elements. Internal storage: %s, "
        "isPacked=%d, physical_size=%zu bytes.",
        logicalLength, gpu::toString(requestedDataType).c_str(),
        gpu::toString(internalDataType).c_str(), packedState, physicalByteSize);
    break;
  case kf64:
  case ki64: // Assuming these should also use direct storage for now
  case ku64:
    logicalLength = sizeParam;
    if (requestedDataType == kf64) {
      // Store f64 internally as expanded u32 pairs
      internalDataType = ku32;
      internalElementSizeBytes = sizeof(uint32_t);
      physicalByteSize =
          logicalLength * 2 * internalElementSizeBytes; // Size for pairs
      packedState = true; // Mark as packed/expanded state
      LOG(kDefLog, kInfo,
          "Creating buffer for %zu logical f64 elements. Internal storage: %s "
          "(pairs), "
          "isPacked=%d, physical_size=%zu bytes.",
          logicalLength, gpu::toString(internalDataType).c_str(), packedState,
          physicalByteSize);
    } else if (requestedDataType == ki64) { // Handle i64
      internalDataType = ki32;
      internalElementSizeBytes = sizeof(int32_t);
      physicalByteSize = logicalLength * 2 * internalElementSizeBytes;
      packedState = true;
      LOG(kDefLog, kInfo,
          "Creating buffer for %zu logical i64 elements. Internal storage: %s "
          "(pairs), isPacked=%d, physical_size=%zu bytes.",
          logicalLength, gpu::toString(internalDataType).c_str(), packedState,
          physicalByteSize);
    } else { // Handle u64
      internalDataType = ku32;
      internalElementSizeBytes = sizeof(uint32_t);
      physicalByteSize = logicalLength * 2 * internalElementSizeBytes;
      packedState = true;
      LOG(kDefLog, kInfo,
          "Creating buffer for %zu logical u64 elements. Internal storage: %s "
          "(pairs), isPacked=%d, physical_size=%zu bytes.",
          logicalLength, gpu::toString(internalDataType).c_str(), packedState,
          physicalByteSize);
    }
    break;

  // --- Other direct storage types ---
  case kf32:
  case ki32:
  case ku32:
  case kUnknown:
  default:
    // Original logic: sizeParam is physical bytes for these types
    physicalByteSize = sizeParam;
    internalDataType = requestedDataType;
    logicalElementSizeBytes = gpu::sizeBytes(internalDataType);
    packedState = false;

    if (logicalElementSizeBytes > 0) {
      logicalLength = physicalByteSize / logicalElementSizeBytes;
      if (physicalByteSize % logicalElementSizeBytes != 0) {
        LOG(kDefLog, kWarn,
            "Requested byte size (%zu) is not a multiple of element size (%zu) "
            "for type %s. Length calculation might be inaccurate.",
            physicalByteSize, logicalElementSizeBytes,
            gpu::toString(internalDataType).c_str());
      }
    } else if (physicalByteSize > 0 && internalDataType != kUnknown) {
      logicalLength = 0;
      LOG(kDefLog, kError,
          "Cannot determine element size for type %s, but byte size > 0. "
          "Setting length to 0.",
          gpu::toString(internalDataType).c_str());
    } else {
      logicalLength = 0;
    }
    LOG(kDefLog, kInfo,
        "Creating buffer: requested physical size=%zu bytes, type=%s. "
        "Calculated logical length: %zu elements.",
        physicalByteSize, gpu::toString(internalDataType).c_str(),
        logicalLength);
    break;
  }

  // --- Common Buffer Creation Logic ---
  if (logicalLength > 0 && physicalByteSize == 0) {
    LOG(kDefLog, kError,
        "Calculated length is %zu, but physical byte size is 0. Invalid state.",
        logicalLength);
    return; // Prevent creation
  }

  // WGPU requirements/safety adjustments
  if (physicalByteSize > 0 && physicalByteSize < 4) {
    LOG(kDefLog, kWarn,
        "Calculated physical buffer size %zu is less than 4 bytes. Adjusting "
        "to 4 bytes.",
        physicalByteSize);
    physicalByteSize = 4;
  }
  // Alignment to 4 bytes
  physicalByteSize = (physicalByteSize + 3) & ~3;

  if (physicalByteSize == 0 && logicalLength == 0) {
    LOG(kDefLog, kInfo,
        "Requested buffer size results in 0 physical bytes. Buffer not "
        "created.");
    bufferData.buffer = nullptr;
    bufferData.size = 0;
    length = 0;
    bufferType = kUnknown;
    isPacked = false;
    return;
  }

  WGPUBufferUsage usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                          WGPUBufferUsage_CopySrc | WGPUBufferUsage_Vertex |
                          WGPUBufferUsage_Index;
  WGPUBufferDescriptor descriptor = {};
  descriptor.usage = usage;
  descriptor.size = physicalByteSize;
  descriptor.mappedAtCreation = false;

  WGPUBuffer newBuffer =
      wgpuDeviceCreateBuffer(this->mgpu.getContext().device, &descriptor);

  if (newBuffer == nullptr) {
    LOG(kDefLog, kError,
        "Failed to create WGPUBuffer (physical size: %zu bytes)",
        physicalByteSize);
    bufferData.buffer = nullptr;
    bufferData.size = 0;
    length = 0;
    bufferType = kUnknown;
    isPacked = false;
    return;
  }

  LOG(kDefLog, kInfo,
      "Successfully created buffer (physical size: %zu bytes, logical length: "
      "%zu, internal type: %s, isPacked: %d)",
      physicalByteSize, logicalLength, gpu::toString(internalDataType).c_str(),
      packedState);

  // hold the old bufferData if it exists
  if (bufferData.buffer != nullptr) {
    LOG(kDefLog, kInfo,
        "Releasing old buffer before creating new one. Old size: %zu bytes.",
        bufferData.size);
    wgpuBufferRelease(bufferData.buffer);
  }

  this->bufferData = gpu::Array{
      .buffer = newBuffer,
      .usage = usage,
      .size = physicalByteSize,
  };
  this->bufferType = internalDataType;
  this->length = logicalLength;
  this->isPacked = packedState;
}

// Helper to determine the original data type based on buffer state
// Returns kUnknown if state is ambiguous or invalid
gpu::NumType Buffer::getOriginalDataType() const {
  if (!isPacked) {
    return bufferType; // Directly stored type
  }

  if (bufferType == ki32 || bufferType == ku32) {
    size_t internalElementSize = gpu::sizeBytes(bufferType);
    if (internalElementSize == 0)
      return kUnknown;

    size_t expectedPhysicalSize8 = length * internalElementSize;
    size_t expectedPhysicalSize16 = length * internalElementSize;

    // Check if physical size matches expected size for unpacked 8-bit
    if (bufferData.size >= expectedPhysicalSize8 &&
        (bufferData.size < expectedPhysicalSize8 + internalElementSize)) {
      // Check alignment for 8-bit packing (4 per internal element)
      if (length > 0 &&
          bufferData.size >= ((length + 3) / 4) * internalElementSize) {
        return (bufferType == ki32) ? ki8 : ku8;
      }
    }
    // Check if physical size matches expected size for unpacked 16-bit
    if (bufferData.size >= expectedPhysicalSize16 &&
        (bufferData.size < expectedPhysicalSize16 + internalElementSize)) {
      // Check alignment for 16-bit packing (2 per internal element)
      if (length > 0 &&
          bufferData.size >= ((length + 1) / 2) * internalElementSize) {
        return (bufferType == ki32) ? ki16 : ku16;
      }
    }
    // Check if physical size matches expected size for expanded f64 (2 u32 per
    // f64)
    if (bufferType == ku32) {
      size_t expectedPhysicalSizeF64 = length * 2 * internalElementSize;
      if (bufferData.size >= expectedPhysicalSizeF64 &&
          (bufferData.size < expectedPhysicalSizeF64 + internalElementSize)) {
        return kf64;
      }
    }

    LOG(kDefLog, kWarn,
        "getOriginalDataType: Ambiguous packed state. InternalType=%s, "
        "isPacked=true, length=%zu, physicalSize=%zu",
        gpu::toString(bufferType).c_str(), length, bufferData.size);
    return kUnknown; // Ambiguous
  }

  // Add cases for other potential internal types used for packing/expansion if
  // needed

  return kUnknown; // Not a recognized packed state
}

// Validates basic preconditions for reading. Returns false if validation fails.
bool Buffer::validateReadPreconditions(const void *outputData,
                                       NumType readAsType) const {
  if (bufferData.buffer == nullptr) {
    LOG(kDefLog, kError, "readSync(%s): Buffer is null.",
        gpu::toString(readAsType).c_str());
    return false;
  }
  if (outputData == nullptr) {
    LOG(kDefLog, kError, "readSync(%s): Output data pointer is null.",
        gpu::toString(readAsType).c_str());
    return false;
  }
  return true;
}

// Validates buffer state against expected state for a specific read type.
// Returns false if validation fails.
bool Buffer::validateBufferStateForRead(NumType readAsType,
                                        NumType expectedInternalType,
                                        bool expectedPackedState) const {
  // Check 1: Is the buffer's packed state what we expect for this read type?
  if (isPacked != expectedPackedState) {
    LOG(kDefLog, kError,
        "readSync(%s): Buffer packed state mismatch. Expected packed=%d, "
        "Actual: packed=%d.",
        gpu::toString(readAsType).c_str(), expectedPackedState, isPacked);
    return false;
  }

  // Check 2: Is the buffer's internal storage type what we expect for this read
  // type?
  if (bufferType != expectedInternalType) {
    LOG(kDefLog, kError,
        "readSync(%s): Buffer internal type mismatch. Expected "
        "internalType=%s, "
        "Actual: internalType=%s.",
        gpu::toString(readAsType).c_str(),
        gpu::toString(expectedInternalType).c_str(),
        gpu::toString(bufferType).c_str());
    return false;
  }

  // physical size based on the expected state?
  size_t internalElementSize = gpu::sizeBytes(bufferType);
  if (internalElementSize > 0 && length > 0) {
    size_t expectedPhysicalSize = 0;
    if (isPacked) {
      if (readAsType == kf64) { // Expanded f64 (2 internal per logical)
        expectedPhysicalSize = length * 2 * internalElementSize;
      } else { // Unpacked 8/16 bit (1 internal per logical)
        expectedPhysicalSize = length * internalElementSize;
      }
    } else { // Direct storage (1 internal per logical)
      expectedPhysicalSize = length * internalElementSize;
    }

    // Assume createBuffer/ensureBuffer sets the exact size needed.
    if (bufferData.size != expectedPhysicalSize) {
      LOG(kDefLog, kWarn, // Warn because it might still work, but indicates
                          // inconsistency
          "readSync(%s): Buffer physical size (%zu) does not match expected "
          "size (%zu) "
          "based on logical length (%zu), internal type (%s), and packed state "
          "(%d).",
          gpu::toString(readAsType).c_str(), bufferData.size,
          expectedPhysicalSize, length, gpu::toString(bufferType).c_str(),
          isPacked);
      // Allow it but log a warning.
    }
  }

  return true;
}

// Calculates the clamped element count based on offset and buffer length.
// Returns true if there are elements to read, false otherwise.
bool Buffer::calculateClampedReadRange(NumType readAsType,
                                       size_t readElementOffset,
                                       size_t &inOutClampedElementCount) const {
  if (readElementOffset >= this->length) {
    LOG(kDefLog, kWarn,
        "readSync(%s): Read offset (%zu) is out of logical bounds (%zu). "
        "Reading 0 elements.",
        gpu::toString(readAsType).c_str(), readElementOffset, this->length);
    inOutClampedElementCount = 0;
    return false;
  }

  if (readElementOffset + inOutClampedElementCount > this->length) {
    size_t originalCount = inOutClampedElementCount;
    inOutClampedElementCount = this->length - readElementOffset;
    LOG(kDefLog, kWarn,
        "readSync(%s): Read count (%zu) exceeds buffer bounds from offset "
        "(%zu). Clamping to %zu elements.",
        gpu::toString(readAsType).c_str(), originalCount, readElementOffset,
        inOutClampedElementCount);
  }

  if (inOutClampedElementCount == 0) {
    LOG(kDefLog, kInfo, "readSync(%s): Clamped read count is 0.",
        gpu::toString(readAsType).c_str());
    return false;
  }
  return true;
}

// Handles reading directly stored data types (f32, i32, u32, i64, u64).
void Buffer::readSyncDirect(void *outputData, NumType readAsType,
                            size_t readElementCount, size_t readElementOffset) {
  if (!validateBufferStateForRead(readAsType, readAsType, false)) {
    return;
  }

  size_t clampedElementCount = readElementCount;
  if (!calculateClampedReadRange(readAsType, readElementOffset,
                                 clampedElementCount)) {
    return;
  }

  size_t bytesPerLogicalElement = gpu::sizeBytes(readAsType);
  if (bytesPerLogicalElement == 0) {
    LOG(kDefLog, kError, "readSyncDirect(%s): Cannot determine element size.",
        gpu::toString(readAsType).c_str());
    return;
  }

  size_t sourceOffsetBytes = readElementOffset * bytesPerLogicalElement;
  size_t totalBytesToRead = clampedElementCount * bytesPerLogicalElement;

  // Physical Bounds Check
  if (sourceOffsetBytes >= bufferData.size) {
    LOG(kDefLog, kError,
        "readSyncDirect(%s): Calculated byte offset (%zu) exceeds buffer "
        "physical size (%zu).",
        gpu::toString(readAsType).c_str(), sourceOffsetBytes, bufferData.size);
    return;
  }
  if (sourceOffsetBytes + totalBytesToRead > bufferData.size) {
    size_t originalBytes = totalBytesToRead;
    totalBytesToRead = bufferData.size - sourceOffsetBytes;
    LOG(kDefLog, kWarn,
        "readSyncDirect(%s): Calculated bytes to read (%zu) exceeds buffer "
        "physical size from offset (%zu). Clamping to %zu bytes.",
        gpu::toString(readAsType).c_str(), originalBytes, sourceOffsetBytes,
        totalBytesToRead);
    if (totalBytesToRead == 0)
      return;
    // Recalculate clampedElementCount based on clamped bytes
    clampedElementCount = totalBytesToRead / bytesPerLogicalElement;
    if (totalBytesToRead % bytesPerLogicalElement != 0) {
      LOG(kDefLog, kError,
          "readSyncDirect(%s): Clamped byte count (%zu) is not a multiple of "
          "element size (%zu). Read may be incomplete.",
          gpu::toString(readAsType).c_str(), totalBytesToRead,
          bytesPerLogicalElement);
    }
    if (clampedElementCount == 0)
      return; // Ensure we don't proceed if clamping results in 0 elements
  }

  LOG(kDefLog, kInfo,
      "readSyncDirect(%s): Reading %zu logical elements (%zu bytes) from "
      "offset %zu (byte offset %zu).",
      gpu::toString(readAsType).c_str(), clampedElementCount, totalBytesToRead,
      readElementOffset, sourceOffsetBytes);

  gpu::toCPU(this->mgpu.getContext(), bufferData.buffer, readAsType, outputData,
             clampedElementCount, sourceOffsetBytes);
}

// Handles reading packed 8-bit or 16-bit types by reading internal
// representation and unpacking on CPU.
void Buffer::readSyncPackedSmallTypes(void *outputData, NumType readAsType,
                                      size_t readElementCount,
                                      size_t readElementOffset) {
  gpu::NumType expectedInternalType =
      (readAsType == ki8 || readAsType == ki16) ? ki32 : ku32;
  if (!validateBufferStateForRead(readAsType, expectedInternalType, true)) {
    return;
  }

  size_t clampedElementCount = readElementCount;
  if (!calculateClampedReadRange(readAsType, readElementOffset,
                                 clampedElementCount)) {
    return;
  }

  size_t bytesPerInternalElement =
      sizeof(int32_t); // Internal type is always 32-bit
  size_t internalElementsToRead =
      clampedElementCount; // 1 internal element per logical element
  size_t internalElementOffset = readElementOffset;
  size_t sourceOffsetBytes = internalElementOffset * bytesPerInternalElement;
  size_t totalBytesToRead = internalElementsToRead * bytesPerInternalElement;

  // Physical Bounds Check on internal data read
  if (sourceOffsetBytes >= bufferData.size) {
    LOG(kDefLog, kError,
        "readSyncPackedSmallTypes(%s): Calculated byte offset (%zu) exceeds "
        "buffer "
        "physical size (%zu).",
        gpu::toString(readAsType).c_str(), sourceOffsetBytes, bufferData.size);
    return;
  }
  if (sourceOffsetBytes + totalBytesToRead > bufferData.size) {
    size_t originalBytes = totalBytesToRead;
    totalBytesToRead = bufferData.size - sourceOffsetBytes;
    internalElementsToRead =
        totalBytesToRead /
        bytesPerInternalElement; // Recalculate elements based on clamped bytes
    if (totalBytesToRead % bytesPerInternalElement != 0) {
      LOG(kDefLog, kWarn,
          "readSyncPackedSmallTypes(%s): Clamped byte read (%zu) is not "
          "multiple of internal element size (%zu).",
          gpu::toString(readAsType).c_str(), totalBytesToRead,
          bytesPerInternalElement);
      // Adjust internalElementsToRead down if partial element read occurred
      internalElementsToRead = totalBytesToRead / bytesPerInternalElement;
    }
    LOG(kDefLog, kWarn,
        "readSyncPackedSmallTypes(%s): Calculated bytes to read (%zu) exceeds "
        "buffer "
        "physical size from offset (%zu). Clamping to %zu bytes (%zu internal "
        "elements).",
        gpu::toString(readAsType).c_str(), originalBytes, sourceOffsetBytes,
        totalBytesToRead, internalElementsToRead);

    if (internalElementsToRead == 0)
      return;
    // Adjust clampedElementCount to match the number of elements we can
    // actually read internally
    clampedElementCount = internalElementsToRead;
  }

  LOG(kDefLog, kInfo,
      "readSyncPackedSmallTypes(%s): Reading %zu internal %s elements (%zu "
      "bytes) from GPU "
      "buffer (byte offset %zu) into temporary CPU buffer.",
      gpu::toString(readAsType).c_str(), internalElementsToRead,
      gpu::toString(expectedInternalType).c_str(), totalBytesToRead,
      sourceOffsetBytes);

  // Temporary CPU buffer for internal i32/u32 data
  std::vector<uint8_t> tempInternalDataBytes(totalBytesToRead);

  // Read raw internal data (assuming gpu::toCPU can read raw bytes or the
  // correct internal type) Option 1: Read raw bytes
  // gpu::toCPU(this->mgpu.getContext(), bufferData.buffer,
  // tempInternalDataBytes.data(), totalBytesToRead, sourceOffsetBytes); Option
  // 2: Read as internal type (preferred if available)
  gpu::toCPU(this->mgpu.getContext(), bufferData.buffer, expectedInternalType,
             tempInternalDataBytes.data(), internalElementsToRead,
             sourceOffsetBytes);

  // --- CPU-side Unpacking ---
  LOG(kDefLog, kInfo,
      "readSyncPackedSmallTypes(%s): Unpacking %zu internal elements into %zu "
      "logical %s elements on CPU.",
      gpu::toString(readAsType).c_str(), internalElementsToRead,
      clampedElementCount, gpu::toString(readAsType).c_str());

  if (readAsType == ki8) {
    int8_t *outputPtr = static_cast<int8_t *>(outputData);
    int32_t *tempPtr =
        reinterpret_cast<int32_t *>(tempInternalDataBytes.data());
    for (size_t i = 0; i < clampedElementCount; ++i) {
      // Assuming the kernel stored the sign-extended i8 in the i32
      outputPtr[i] = static_cast<int8_t>(tempPtr[i]);
    }
  } else if (readAsType == ku8) {
    uint8_t *outputPtr = static_cast<uint8_t *>(outputData);
    uint32_t *tempPtr =
        reinterpret_cast<uint32_t *>(tempInternalDataBytes.data());
    for (size_t i = 0; i < clampedElementCount; ++i) {
      // Assuming the kernel stored the u8 in the u32
      outputPtr[i] = static_cast<uint8_t>(tempPtr[i]);
    }
  } else if (readAsType == ki16) {
    int16_t *outputPtr = static_cast<int16_t *>(outputData);
    int32_t *tempPtr =
        reinterpret_cast<int32_t *>(tempInternalDataBytes.data());
    for (size_t i = 0; i < clampedElementCount; ++i) {
      // Assuming the kernel stored the sign-extended i16 in the i32
      outputPtr[i] = static_cast<int16_t>(tempPtr[i]);
    }
  } else if (readAsType == ku16) {
    uint16_t *outputPtr = static_cast<uint16_t *>(outputData);
    uint32_t *tempPtr =
        reinterpret_cast<uint32_t *>(tempInternalDataBytes.data());
    for (size_t i = 0; i < clampedElementCount; ++i) {
      // Assuming the kernel stored the u16 in the u32
      outputPtr[i] = static_cast<uint16_t>(tempPtr[i]);
    }
  } else {
    LOG(kDefLog, kError, "readSyncPackedSmallTypes: Unexpected readAsType %s",
        gpu::toString(readAsType).c_str());
  }
}

// Handles reading expanded float64 (stored as u32 pairs).
void Buffer::readSyncExpandedFloat64(void *outputData, size_t readElementCount,
                                     size_t readElementOffset) {
  NumType readAsType = kf64;
  if (!validateBufferStateForRead(readAsType, ku32, true)) {
    return;
  }

  size_t clampedElementCount = readElementCount;
  if (!calculateClampedReadRange(readAsType, readElementOffset,
                                 clampedElementCount)) {
    return;
  }

  size_t u32sPerElement = 2;
  size_t bytesPerInternalElement = sizeof(uint32_t);
  size_t sourceElementOffsetInternal = readElementOffset * u32sPerElement;
  size_t numInternalElementsToRead = clampedElementCount * u32sPerElement;
  size_t sourceOffsetBytes =
      sourceElementOffsetInternal * bytesPerInternalElement;
  size_t totalBytesToRead = numInternalElementsToRead * bytesPerInternalElement;

  // Physical bounds check
  if (sourceOffsetBytes >= bufferData.size) {
    LOG(kDefLog, kError,
        "readSyncExpandedFloat64: Calculated byte offset (%zu) exceeds buffer "
        "physical size (%zu).",
        sourceOffsetBytes, bufferData.size);
    return;
  }
  if (sourceOffsetBytes + totalBytesToRead > bufferData.size) {
    totalBytesToRead = bufferData.size - sourceOffsetBytes;
    numInternalElementsToRead = totalBytesToRead / bytesPerInternalElement;
    LOG(kDefLog, kWarn,
        "readSyncExpandedFloat64: Clamping read bytes to %zu (%zu u32 "
        "elements).",
        totalBytesToRead, numInternalElementsToRead);
    if (numInternalElementsToRead == 0)
      return;
    // Ensure we don't read partial doubles
    if (numInternalElementsToRead % u32sPerElement != 0) {
      LOG(kDefLog, kWarn,
          "readSyncExpandedFloat64: Clamped read size results in partial "
          "double. Truncating.");
      numInternalElementsToRead -= (numInternalElementsToRead % u32sPerElement);
      totalBytesToRead = numInternalElementsToRead * bytesPerInternalElement;
      if (numInternalElementsToRead == 0)
        return;
    }
  }

  // Temporary CPU buffer for u32 data
  std::vector<uint32_t> tempU32Data(numInternalElementsToRead);

  LOG(kDefLog, kInfo,
      "readSyncExpandedFloat64: Reading %zu u32 elements (%zu bytes) from GPU "
      "buffer (byte offset %zu) into temporary CPU buffer.",
      numInternalElementsToRead, totalBytesToRead, sourceOffsetBytes);

  // Read raw u32 data
  gpu::toCPU(this->mgpu.getContext(), bufferData.buffer, tempU32Data.data(),
             totalBytesToRead, sourceOffsetBytes);

  // --- CPU-side Bitcasting ---
  size_t numDoublesRead = numInternalElementsToRead / u32sPerElement;
  LOG(kDefLog, kInfo,
      "readSyncExpandedFloat64: Bitcasting %zu uint32_t elements back to %zu "
      "doubles on CPU.",
      numInternalElementsToRead, numDoublesRead);
  static_assert(sizeof(double) == 2 * sizeof(uint32_t), "Size mismatch");
  memcpy(outputData, tempU32Data.data(), numDoublesRead * sizeof(double));
}

// Handles reading expanded int64 (stored as i32 pairs).
void Buffer::readSyncExpandedInt64(void *outputData, size_t readElementCount,
                                   size_t readElementOffset) {
  NumType readAsType = ki64;
  // Expect internal type ki32 and packed state
  if (!validateBufferStateForRead(readAsType, ki32, true)) {
    return;
  }

  size_t clampedElementCount = readElementCount;
  if (!calculateClampedReadRange(readAsType, readElementOffset,
                                 clampedElementCount)) {
    return;
  }

  size_t i32sPerElement = 2;
  size_t bytesPerInternalElement = sizeof(int32_t);
  size_t sourceElementOffsetInternal = readElementOffset * i32sPerElement;
  size_t numInternalElementsToRead = clampedElementCount * i32sPerElement;
  size_t sourceOffsetBytes =
      sourceElementOffsetInternal * bytesPerInternalElement;
  size_t totalBytesToRead = numInternalElementsToRead * bytesPerInternalElement;

  // Physical bounds check (similar to float64)
  if (sourceOffsetBytes >= bufferData.size) {
    LOG(kDefLog, kError,
        "readSyncExpandedInt64: Calculated byte offset (%zu) exceeds buffer "
        "physical size (%zu).",
        sourceOffsetBytes, bufferData.size);
    return;
  }
  if (sourceOffsetBytes + totalBytesToRead > bufferData.size) {
    totalBytesToRead = bufferData.size - sourceOffsetBytes;
    numInternalElementsToRead = totalBytesToRead / bytesPerInternalElement;
    LOG(kDefLog, kWarn,
        "readSyncExpandedInt64: Clamping read bytes to %zu (%zu i32 "
        "elements).",
        totalBytesToRead, numInternalElementsToRead);
    if (numInternalElementsToRead == 0)
      return;
    if (numInternalElementsToRead % i32sPerElement != 0) {
      LOG(kDefLog, kWarn,
          "readSyncExpandedInt64: Clamped read size results in partial "
          "int64. Truncating.");
      numInternalElementsToRead -= (numInternalElementsToRead % i32sPerElement);
      totalBytesToRead = numInternalElementsToRead * bytesPerInternalElement;
      if (numInternalElementsToRead == 0)
        return;
    }
  }

  // Temporary CPU buffer for i32 data
  std::vector<int32_t> tempI32Data(numInternalElementsToRead);

  LOG(kDefLog, kInfo,
      "readSyncExpandedInt64: Reading %zu i32 elements (%zu bytes) from GPU "
      "buffer (byte offset %zu) into temporary CPU buffer.",
      numInternalElementsToRead, totalBytesToRead, sourceOffsetBytes);

  // Read raw i32 data
  gpu::toCPU(this->mgpu.getContext(), bufferData.buffer, ki32, // Read as i32
             tempI32Data.data(), numInternalElementsToRead, sourceOffsetBytes);

  // --- CPU-side Combining ---
  size_t numInt64sRead = numInternalElementsToRead / i32sPerElement;
  LOG(kDefLog, kInfo,
      "readSyncExpandedInt64: Combining %zu int32_t elements back to %zu "
      "int64s on CPU.",
      numInternalElementsToRead, numInt64sRead);
  static_assert(sizeof(int64_t) == 2 * sizeof(int32_t), "Size mismatch");
  memcpy(outputData, tempI32Data.data(), numInt64sRead * sizeof(int64_t));
}

// Handles reading expanded uint64 (stored as u32 pairs).
void Buffer::readSyncExpandedUint64(void *outputData, size_t readElementCount,
                                    size_t readElementOffset) {
  NumType readAsType = ku64;
  // Expect internal type ku32 and packed state
  if (!validateBufferStateForRead(readAsType, ku32, true)) {
    return;
  }

  size_t clampedElementCount = readElementCount;
  if (!calculateClampedReadRange(readAsType, readElementOffset,
                                 clampedElementCount)) {
    return;
  }

  size_t u32sPerElement = 2;
  size_t bytesPerInternalElement = sizeof(uint32_t);
  size_t sourceElementOffsetInternal = readElementOffset * u32sPerElement;
  size_t numInternalElementsToRead = clampedElementCount * u32sPerElement;
  size_t sourceOffsetBytes =
      sourceElementOffsetInternal * bytesPerInternalElement;
  size_t totalBytesToRead = numInternalElementsToRead * bytesPerInternalElement;

  // Physical bounds check (similar to float64)
  if (sourceOffsetBytes >= bufferData.size) {
    LOG(kDefLog, kError,
        "readSyncExpandedUint64: Calculated byte offset (%zu) exceeds buffer "
        "physical size (%zu).",
        sourceOffsetBytes, bufferData.size);
    return;
  }
  if (sourceOffsetBytes + totalBytesToRead > bufferData.size) {
    totalBytesToRead = bufferData.size - sourceOffsetBytes;
    numInternalElementsToRead = totalBytesToRead / bytesPerInternalElement;
    LOG(kDefLog, kWarn,
        "readSyncExpandedUint64: Clamping read bytes to %zu (%zu u32 "
        "elements).",
        totalBytesToRead, numInternalElementsToRead);
    if (numInternalElementsToRead == 0)
      return;
    if (numInternalElementsToRead % u32sPerElement != 0) {
      LOG(kDefLog, kWarn,
          "readSyncExpandedUint64: Clamped read size results in partial "
          "uint64. Truncating.");
      numInternalElementsToRead -= (numInternalElementsToRead % u32sPerElement);
      totalBytesToRead = numInternalElementsToRead * bytesPerInternalElement;
      if (numInternalElementsToRead == 0)
        return;
    }
  }

  // Temporary CPU buffer for u32 data
  std::vector<uint32_t> tempU32Data(numInternalElementsToRead);

  LOG(kDefLog, kInfo,
      "readSyncExpandedUint64: Reading %zu u32 elements (%zu bytes) from GPU "
      "buffer (byte offset %zu) into temporary CPU buffer.",
      numInternalElementsToRead, totalBytesToRead, sourceOffsetBytes);

  // Read raw u32 data
  gpu::toCPU(this->mgpu.getContext(), bufferData.buffer, ku32, // Read as u32
             tempU32Data.data(), numInternalElementsToRead, sourceOffsetBytes);

  // --- CPU-side Combining ---
  size_t numUint64sRead = numInternalElementsToRead / u32sPerElement;
  LOG(kDefLog, kInfo,
      "readSyncExpandedUint64: Combining %zu uint32_t elements back to %zu "
      "uint64s on CPU.",
      numInternalElementsToRead, numUint64sRead);
  static_assert(sizeof(uint64_t) == 2 * sizeof(uint32_t), "Size mismatch");
  memcpy(outputData, tempU32Data.data(), numUint64sRead * sizeof(uint64_t));
}

// --- Public readSync Method ---
void Buffer::readSync(void *outputData, NumType readAsType,
                      size_t readElementCount, size_t readElementOffset) {

  if (!validateReadPreconditions(outputData, readAsType)) {
    return;
  }

  LOG(kDefLog, kInfo, "readSync(%s): Reading %zu elements at offset %zu.",
      gpu::toString(readAsType).c_str(), readElementCount, readElementOffset);

  switch (readAsType) {
  case ki8:
  case ku8:
  case ki16:
  case ku16:
    readSyncPackedSmallTypes(outputData, readAsType, readElementCount,
                             readElementOffset);
    break;
  case kf64:
    readSyncExpandedFloat64(outputData, readElementCount, readElementOffset);
    break;
  case ki64:
    readSyncExpandedInt64(outputData, readElementCount, readElementOffset);
    break;
  case ku64:
    readSyncExpandedUint64(outputData, readElementCount, readElementOffset);
    break;
  case kf32:
  case ki32:
  case ku32:
    readSyncDirect(outputData, readAsType, readElementCount, readElementOffset);
    break;
  case kUnknown:
  default:
    LOG(kDefLog, kError, "readSync: Unsupported or unknown readAsType: %d",
        readAsType);
    break;
  }

  LOG(kDefLog, kInfo, "readSync(%s) complete.",
      gpu::toString(readAsType).c_str());
}

void Buffer::readAsync(void *outputData, NumType dType, size_t size,
                       size_t offset, std::function<void()> callback) {
  std::thread([=]() {
    try {
      readSync(outputData, dType, size, offset);
      if (callback) {
        callback();
      }
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "Exception in readAsync thread: %s", e.what());
    } catch (...) {
      LOG(kDefLog, kError, "Unknown exception in readAsync thread.");
    }
  }).detach();
}

// --- setData Overloads ---
bool Buffer::validateBufferStateTransition(NumType requestedReadType) const {
  if (bufferData.buffer == nullptr) {
    LOG(kDefLog, kError, "validateBufferStateTransition: Buffer is null");
    return false;
  }

  // If buffer is empty, it's safe
  if (length == 0 || bufferData.size == 0) {
    LOG(kDefLog, kInfo,
        "validateBufferStateTransition: Empty buffer, safe to transition");
    return true;
  }

  // Get the original data type this buffer was created for
  NumType originalType = getOriginalDataType();

  // If we can't determine the original type, be conservative
  if (originalType == kUnknown) {
    LOG(kDefLog, kWarn,
        "validateBufferStateTransition: Cannot determine original buffer type, "
        "assuming unsafe");
    return false;
  }

  // Check if the transition is safe
  if (originalType == requestedReadType) {
    return true; // Same type, always safe
  }

  // Check size compatibility for different types
  size_t originalElementSize = gpu::sizeBytes(originalType);
  size_t requestedElementSize = gpu::sizeBytes(requestedReadType);

  if (originalElementSize == 0 || requestedElementSize == 0) {
    LOG(kDefLog, kError,
        "validateBufferStateTransition: Cannot determine element sizes");
    return false;
  }

  // For packed types, check if the physical buffer can accommodate the new
  // layout
  bool originalIsPacked = isPackedType(originalType);
  bool requestedIsPacked = isPackedType(requestedReadType);

  if (originalIsPacked != requestedIsPacked) {
    LOG(kDefLog, kWarn,
        "validateBufferStateTransition: Packed state mismatch - "
        "original=%s(packed=%d) vs requested=%s(packed=%d)",
        gpu::toString(originalType).c_str(), originalIsPacked,
        gpu::toString(requestedReadType).c_str(), requestedIsPacked);
    return false;
  }

  return true;
}

// Add helper method to check if a type uses packed storage
bool Buffer::isPackedType(NumType type) const {
  switch (type) {
  case ki8:
  case ku8:
  case ki16:
  case ku16:
  case kf64:
  case ki64:
  case ku64:
    return true;
  default:
    return false;
  }
}

// Add safe buffer recreation method
bool Buffer::recreateBufferSafely(size_t newLogicalLength,
                                  NumType newOriginalDataType) {
  LOG(kDefLog, kInfo,
      "recreateBufferSafely: Recreating buffer from type=%s(packed=%d,len=%zu) "
      "to type=%s(len=%zu)",
      gpu::toString(bufferType).c_str(), isPacked, length,
      gpu::toString(newOriginalDataType).c_str(), newLogicalLength);

  // Store old state for cleanup
  WGPUBuffer oldBuffer = bufferData.buffer;
  size_t oldSize = bufferData.size;

  // Reset state before recreation
  bufferData.buffer = nullptr;
  bufferData.size = 0;
  length = 0;
  bufferType = kUnknown;
  isPacked = false;

  try {
    // Create new buffer
    createBuffer(newLogicalLength, newOriginalDataType);

    if (bufferData.buffer == nullptr) {
      LOG(kDefLog, kError, "recreateBufferSafely: Failed to create new buffer");
      return false;
    }

    // Clean up old buffer
    if (oldBuffer != nullptr) {
      LOG(kDefLog, kInfo,
          "recreateBufferSafely: Releasing old buffer (size=%zu)", oldSize);
      wgpuBufferRelease(oldBuffer);
    }

    LOG(kDefLog, kInfo,
        "recreateBufferSafely: Successfully recreated buffer. New state: "
        "type=%s, packed=%d, size=%zu, length=%zu",
        gpu::toString(bufferType).c_str(), isPacked, bufferData.size, length);
    return true;

  } catch (const std::exception &e) {
    LOG(kDefLog, kError,
        "recreateBufferSafely: Exception during recreation: %s", e.what());

    // Restore old buffer if recreation failed
    if (oldBuffer != nullptr) {
      bufferData.buffer = oldBuffer;
      bufferData.size = oldSize;
      // Note: We've lost the original state info, buffer may be in inconsistent
      // state
    }
    return false;
  }
}

// Enhanced ensureBuffer with safe transitions
void Buffer::ensureBuffer(size_t requiredLogicalLength,
                          gpu::NumType targetOriginalDataType) {
  // Check if current buffer is compatible
  if (bufferData.buffer != nullptr) {
    NumType currentOriginalType = getOriginalDataType();

    // If we can determine the current type and it matches what we need
    if (currentOriginalType == targetOriginalDataType &&
        length >= requiredLogicalLength) {
      // Buffer is compatible, just update length if needed
      if (length != requiredLogicalLength) {
        LOG(kDefLog, kInfo,
            "ensureBuffer: Updating length from %zu to %zu for compatible "
            "buffer",
            length, requiredLogicalLength);
        length = requiredLogicalLength;
      }
      return;
    }

    // If types don't match or buffer is too small, we need to recreate
    if (currentOriginalType != targetOriginalDataType) {
      LOG(kDefLog, kInfo,
          "ensureBuffer: Type change detected (%s -> %s), recreating buffer",
          gpu::toString(currentOriginalType).c_str(),
          gpu::toString(targetOriginalDataType).c_str());
    } else {
      LOG(kDefLog, kInfo,
          "ensureBuffer: Size increase needed (%zu -> %zu), recreating buffer",
          length, requiredLogicalLength);
    }

    if (!recreateBufferSafely(requiredLogicalLength, targetOriginalDataType)) {
      throw std::runtime_error(
          "Failed to safely recreate buffer in ensureBuffer");
    }
    return;
  }

  // No existing buffer, create new one
  LOG(kDefLog, kInfo,
      "ensureBuffer: No existing buffer, creating new one for type=%s, "
      "length=%zu",
      gpu::toString(targetOriginalDataType).c_str(), requiredLogicalLength);

  createBuffer(requiredLogicalLength, targetOriginalDataType);

  if (bufferData.buffer == nullptr) {
    throw std::runtime_error("Failed to create new buffer in ensureBuffer");
  }
}

// --- setData Implementations ---
void Buffer::setData(const float *inputData, size_t byteSize) {
  if (inputData == nullptr && byteSize > 0) {
    throw std::invalid_argument("setData(float): inputData is null");
  }
  size_t numElements = byteSize / sizeof(float);
  if (bufferData.buffer == nullptr) {
    LOG(kDefLog, kError, "setData(float): Buffer is null before upload");
    return;
  }
  gpu::NumType originalDataType = kf32;
  LOG(kDefLog, kInfo, "setData(float): %zu elements", numElements);
  try {
    ensureBuffer(
        numElements,
        originalDataType); // Ensure buffer exists with direct storage state
    if (numElements > 0) {
      size_t uploadBytes = numElements * sizeof(float);

      // Validate buffer size before upload
      if (uploadBytes > bufferData.size) {
        LOG(kDefLog, kError,
            "setData(float): Upload size (%zu bytes) exceeds buffer size (%zu "
            "bytes)",
            uploadBytes, bufferData.size);
        return;
      }

      LOG(kDefLog, kInfo,
          "setData(float): Uploading %zu bytes to buffer of size %zu",
          uploadBytes, bufferData.size);
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 uploadBytes);
    }
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(float): Failed - %s", e.what());
  }
}

void Buffer::setData(const int32_t *inputData, size_t byteSize) {
  if (inputData == nullptr && byteSize > 0)
    throw std::invalid_argument("setData(int32_t): inputData is null");
  size_t numElements = byteSize / sizeof(int32_t);
  gpu::NumType originalDataType = ki32;
  LOG(kDefLog, kInfo, "setData(int32_t): %zu elements", numElements);
  try {
    ensureBuffer(numElements, originalDataType);
    if (numElements > 0) {
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 numElements * sizeof(int32_t));
    }
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(int32_t): Failed - %s", e.what());
  }
}

void Buffer::setData(const uint32_t *inputData, size_t byteSize) {
  if (inputData == nullptr && byteSize > 0)
    throw std::invalid_argument("setData(uint32_t): inputData is null");
  size_t numElements = byteSize / sizeof(uint32_t);
  gpu::NumType originalDataType = ku32;
  LOG(kDefLog, kInfo, "setData(uint32_t): %zu elements", numElements);
  try {
    ensureBuffer(numElements, originalDataType);
    if (numElements > 0) {
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 numElements * sizeof(uint32_t));
    }
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(uint32_t): Failed - %s", e.what());
  }
}

// --- setData for Packed/Expanded Types ---
void Buffer::setData(const int8_t *inputData, size_t byteSize) {
  if (inputData == nullptr && byteSize > 0)
    throw std::invalid_argument("setData(int8_t): inputData is null");
  size_t numElements = byteSize / sizeof(int8_t);
  gpu::NumType originalDataType = ki8;
  gpu::NumType internalDataType = ki32;
  size_t packingRatio = 4;
  LOG(kDefLog, kInfo, "setData(int8_t): %zu elements (logical length)",
      numElements);

  // --- CPU-side Padding ---
  size_t paddedNumElements = (numElements + packingRatio - 1) &
                             ~(packingRatio - 1); // Align up to multiple of 4
  std::vector<int8_t> paddedInputData;
  const int8_t *dataToUpload =
      inputData; // Use original data if no padding needed

  if (paddedNumElements != numElements) {
    LOG(kDefLog, kInfo,
        "setData(int8_t): Padding CPU data from %zu to %zu elements.",
        numElements, paddedNumElements);
    paddedInputData.resize(paddedNumElements,
                           0); // Allocate and zero-initialize
    memcpy(paddedInputData.data(), inputData, numElements * sizeof(int8_t));
    dataToUpload = paddedInputData.data(); // Point to the padded data
  } else {
    LOG(kDefLog, kInfo,
        "setData(int8_t): No CPU padding needed for %zu elements.",
        numElements);
  }
  // --- End CPU-side Padding ---

  // 1. Temporary buffer for initial packed upload via gpu::toGPU
  //    Size calculation remains based on the *padded* element count now.
  size_t packedUploadElementCount = paddedNumElements / packingRatio;
  size_t packedUploadByteSize = packedUploadElementCount * sizeof(int32_t);
  Buffer initialPackedUploadBuffer(mgpu);
  LOG(kDefLog, kInfo,
      "setData(int8_t): Creating temporary packed upload buffer (size: %zu "
      "bytes, type: %s, based on %zu padded elements)",
      packedUploadByteSize, gpu::toString(internalDataType).c_str(),
      paddedNumElements);
  initialPackedUploadBuffer.createBuffer(
      packedUploadByteSize, internalDataType); // Create based on physical size
  if (initialPackedUploadBuffer.bufferData.buffer == nullptr) {
    LOG(kDefLog, kError,
        "setData(int8_t): Failed to create temporary packed buffer.");
    return;
  }

  // Upload CPU int8 data (potentially padded), packing it into the temporary
  // buffer
  if (paddedNumElements > 0) { // Use padded count for upload
    LOG(kDefLog, kInfo,
        "setData(int8_t): Uploading %zu padded int8 elements (packed via "
        "gpu::toGPU) "
        "to temporary buffer",
        paddedNumElements);
    // Pass original type, *padded* data pointer, and *padded* logical length to
    // gpu::toGPU
    gpu::toGPU(this->mgpu.getContext(),
               dataToUpload, // Use potentially padded data
               initialPackedUploadBuffer.bufferData.buffer,
               paddedNumElements); // Use padded element count
  }

  // Ensure main buffer exists and is sized for UNPACKED int32 data.
  //    This still uses the *original* numElements for logical length.
  LOG(kDefLog, kInfo,
      "setData(int8_t): Ensuring main buffer exists for unpacked data (logical "
      "length %zu).",
      numElements);
  try {
    ensureBuffer(
        numElements,       // Use original logical length
        originalDataType); // Ensure buffer exists with correct internal state
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(int8_t): Failed ensuring main buffer - %s",
        e.what());
    return;
  }

  // Dispatch the unpack kernel: initialPackedUploadBuffer ->
  // this->bufferData
  //    The kernel reads based on packed count, writes based on logical length.
  if (numElements >
      0) { // Kernel dispatch depends on actual logical elements needed
    LOG(kDefLog, kInfo,
        "setData(int8_t): Dispatching kernel to unpack data from temporary "
        "buffer into main buffer (logical length %zu)",
        numElements);
    kernels::dispatchPackedI8toI32(mgpu, initialPackedUploadBuffer, *this);
  }
  initialPackedUploadBuffer.release();

  LOG(kDefLog, kInfo,
      "setData(int8_t) complete. Main buffer state: type=%s, isPacked=%d, "
      "physical_size=%zu, logical_length=%zu",
      gpu::toString(bufferType).c_str(), isPacked, bufferData.size,
      this->length); // Length should be original numElements
}

void Buffer::setData(const uint8_t *inputData, size_t byteSize) {
  if (inputData == nullptr && byteSize > 0)
    throw std::invalid_argument("setData(uint8_t): inputData is null");
  size_t numElements = byteSize / sizeof(uint8_t);
  gpu::NumType originalDataType = ku8;
  gpu::NumType internalDataType = ku32;
  size_t packingRatio = 4;
  LOG(kDefLog, kInfo, "setData(uint8_t): %zu elements (logical length)",
      numElements);

  // --- CPU-side Padding ---
  size_t paddedNumElements = (numElements + packingRatio - 1) &
                             ~(packingRatio - 1); // Align up to multiple of 4
  std::vector<uint8_t> paddedInputData;
  const uint8_t *dataToUpload = inputData;

  if (paddedNumElements != numElements) {
    LOG(kDefLog, kInfo,
        "setData(uint8_t): Padding CPU data from %zu to %zu elements.",
        numElements, paddedNumElements);
    paddedInputData.resize(paddedNumElements, 0);
    memcpy(paddedInputData.data(), inputData, numElements * sizeof(uint8_t));
    dataToUpload = paddedInputData.data();
  } else {
    LOG(kDefLog, kInfo,
        "setData(uint8_t): No CPU padding needed for %zu elements.",
        numElements);
  }
  // --- End CPU-side Padding ---

  size_t packedUploadElementCount = paddedNumElements / packingRatio;
  size_t packedUploadByteSize = packedUploadElementCount * sizeof(uint32_t);
  Buffer initialPackedUploadBuffer(mgpu);
  LOG(kDefLog, kInfo,
      "setData(uint8_t): Creating temporary packed upload buffer (size: %zu "
      "bytes, type: %s, based on %zu padded elements)",
      packedUploadByteSize, gpu::toString(internalDataType).c_str(),
      paddedNumElements);
  initialPackedUploadBuffer.createBuffer(packedUploadByteSize,
                                         internalDataType);
  if (initialPackedUploadBuffer.bufferData.buffer == nullptr) {
    LOG(kDefLog, kError,
        "setData(uint8_t): Failed to create temporary packed buffer.");
    return;
  }

  if (paddedNumElements > 0) {
    LOG(kDefLog, kInfo,
        "setData(uint8_t): Uploading %zu padded uint8 elements (packed via "
        "gpu::toGPU) to temporary buffer",
        paddedNumElements);
    gpu::toGPU(this->mgpu.getContext(),
               dataToUpload, // Use potentially padded data
               initialPackedUploadBuffer.bufferData.buffer,
               paddedNumElements); // Use padded element count
  }

  LOG(kDefLog, kInfo,
      "setData(uint8_t): Ensuring main buffer exists for unpacked data "
      "(logical length %zu).",
      numElements);
  try {
    ensureBuffer(numElements, originalDataType); // Use original logical length
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(uint8_t): Failed ensuring main buffer - %s",
        e.what());
    return;
  }

  if (numElements > 0) {
    LOG(kDefLog, kInfo,
        "setData(uint8_t): Dispatching kernel to unpack data from temporary "
        "buffer into main buffer (logical length %zu)",
        numElements);
    kernels::dispatchPackedU8toU32(mgpu, initialPackedUploadBuffer, *this);
    initialPackedUploadBuffer.release();
  }

  LOG(kDefLog, kInfo,
      "setData(uint8_t) complete. Main buffer state: type=%s, isPacked=%d, "
      "physical_size=%zu, logical_length=%zu",
      gpu::toString(bufferType).c_str(), isPacked, bufferData.size,
      this->length); // Length should be original numElements
}

void Buffer::setData(const int16_t *inputData, size_t byteSize) {
  if (inputData == nullptr && byteSize > 0)
    throw std::invalid_argument("setData(int16_t): inputData is null");
  size_t numElements = byteSize / sizeof(int16_t);
  gpu::NumType originalDataType = ki16;
  gpu::NumType internalDataType = ki32;
  size_t packingRatio = 2;
  LOG(kDefLog, kInfo, "setData(int16_t): %zu elements (logical length)",
      numElements);

  size_t packedUploadElementCount =
      (numElements + packingRatio - 1) / packingRatio;
  size_t packedUploadByteSize = packedUploadElementCount * sizeof(int32_t);
  Buffer initialPackedUploadBuffer(mgpu);
  LOG(kDefLog, kInfo,
      "setData(int16_t): Creating temporary packed upload buffer (size: %zu "
      "bytes, type: %s)",
      packedUploadByteSize, gpu::toString(internalDataType).c_str());
  initialPackedUploadBuffer.createBuffer(packedUploadByteSize,
                                         internalDataType);
  if (initialPackedUploadBuffer.bufferData.buffer == nullptr) {
    LOG(kDefLog, kError,
        "setData(int16_t): Failed to create temporary packed buffer.");
    return;
  }

  if (numElements > 0) {
    LOG(kDefLog, kInfo,
        "setData(int16_t): Uploading %zu int16 elements (packed via "
        "gpu::toGPU) to temporary buffer",
        numElements);
    gpu::toGPU(this->mgpu.getContext(), inputData,
               initialPackedUploadBuffer.bufferData.buffer, numElements);
  }

  LOG(kDefLog, kInfo,
      "setData(int16_t): Ensuring main buffer exists for unpacked data.");
  try {
    ensureBuffer(numElements, originalDataType);
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(int16_t): Failed ensuring main buffer - %s",
        e.what());
    return;
  }

  if (numElements > 0) {
    LOG(kDefLog, kInfo,
        "setData(int16_t): Dispatching kernel to unpack data from temporary "
        "buffer into main buffer");
    kernels::dispatchPackedI16toI32(mgpu, initialPackedUploadBuffer, *this);
  }

  initialPackedUploadBuffer.release();

  LOG(kDefLog, kInfo,
      "setData(int16_t) complete. Main buffer state: type=%s, isPacked=%d, "
      "physical_size=%zu, logical_length=%zu",
      gpu::toString(bufferType).c_str(), isPacked, bufferData.size,
      this->length);
}

void Buffer::setData(const uint16_t *inputData, size_t byteSize) {
  if (inputData == nullptr && byteSize > 0)
    throw std::invalid_argument("setData(uint16_t): inputData is null");
  size_t numElements = byteSize / sizeof(uint16_t);
  gpu::NumType originalDataType = ku16;
  gpu::NumType internalDataType = ku32;
  size_t packingRatio = 2;
  LOG(kDefLog, kInfo, "setData(uint16_t): %zu elements (logical length)",
      numElements);

  size_t packedUploadElementCount =
      (numElements + packingRatio - 1) / packingRatio;
  size_t packedUploadByteSize = packedUploadElementCount * sizeof(uint32_t);
  Buffer initialPackedUploadBuffer(mgpu);
  LOG(kDefLog, kInfo,
      "setData(uint16_t): Creating temporary packed upload buffer (size: %zu "
      "bytes, type: %s)",
      packedUploadByteSize, gpu::toString(internalDataType).c_str());
  initialPackedUploadBuffer.createBuffer(packedUploadByteSize,
                                         internalDataType);
  if (initialPackedUploadBuffer.bufferData.buffer == nullptr) {
    LOG(kDefLog, kError,
        "setData(uint16_t): Failed to create temporary packed buffer.");
    return;
  }

  if (numElements > 0) {
    LOG(kDefLog, kInfo,
        "setData(uint16_t): Uploading %zu uint16 elements (packed via "
        "gpu::toGPU) to temporary buffer",
        numElements);
    gpu::toGPU(this->mgpu.getContext(), inputData,
               initialPackedUploadBuffer.bufferData.buffer, numElements);
  }

  LOG(kDefLog, kInfo,
      "setData(uint16_t): Ensuring main buffer exists for unpacked data.");
  try {
    ensureBuffer(numElements, originalDataType);
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(uint16_t): Failed ensuring main buffer - %s",
        e.what());
    return;
  }

  if (numElements > 0) {
    LOG(kDefLog, kInfo,
        "setData(uint16_t): Dispatching kernel to unpack data from temporary "
        "buffer into main buffer");
    kernels::dispatchPackedU16toU32(mgpu, initialPackedUploadBuffer, *this);
  }

  initialPackedUploadBuffer.release();

  LOG(kDefLog, kInfo,
      "setData(uint16_t) complete. Main buffer state: type=%s, isPacked=%d, "
      "physical_size=%zu, logical_length=%zu",
      gpu::toString(bufferType).c_str(), isPacked, bufferData.size,
      this->length);
}

void Buffer::setData(const double *inputData, size_t byteSize) {
  if (inputData == nullptr && byteSize > 0)
    throw std::invalid_argument("setData(double): inputData is null");
  size_t numElements = byteSize / sizeof(double);
  gpu::NumType originalDataType = kf64;
  // Internal state should be ku32, isPacked=true
  LOG(kDefLog, kInfo, "setData(double): %zu elements (logical length)",
      numElements);

  try {
    // Ensure buffer exists with correct internal state (ku32, isPacked=true)
    ensureBuffer(numElements, originalDataType);

    // Upload the data using the packing overload of gpu::toGPU
    if (numElements > 0) {
      LOG(kDefLog, kInfo,
          "setData(double): Uploading %zu double elements (packed via "
          "gpu::toGPU) to main buffer",
          numElements);
      // Call the gpu::toGPU overload that handles double packing
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 numElements);
    }

  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(double): Failed - %s", e.what());
    // Consider re-throwing or returning an error code if appropriate
    return; // Exit if ensureBuffer or upload fails
  }

  LOG(kDefLog, kInfo,
      "setData(double) complete. Main buffer state: type=%s, isPacked=%d, "
      "physical_size=%zu, logical_length=%zu",
      gpu::toString(bufferType).c_str(), isPacked, bufferData.size,
      this->length);
}

// --- setData for 64-bit integers (Direct Upload - No Packing/Expansion Yet)
void Buffer::setData(const int64_t *inputData, size_t byteSize) {
  if (inputData == nullptr && byteSize > 0)
    throw std::invalid_argument("setData(int64_t): inputData is null");

  size_t numElements = byteSize / sizeof(int64_t);

  gpu::NumType originalDataType = ki64;
  // Internal state should be ki32, isPacked=true
  LOG(kDefLog, kInfo, "setData(int64_t): %zu elements (logical length)",
      numElements);

  try {
    // Ensure buffer exists with correct internal state (ki32, isPacked=true)
    ensureBuffer(numElements, originalDataType);

    // Upload the data using the packing overload of gpu::toGPU
    if (numElements > 0) {
      LOG(kDefLog, kInfo,
          "setData(int64_t): Uploading %zu int64 elements (packed via "
          "gpu::toGPU) to main buffer",
          numElements);
      // Call the gpu::toGPU overload that handles int64 packing
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 numElements);
    }

  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(int64_t): Failed - %s", e.what());
    return; // Exit if ensureBuffer or upload fails
  }

  LOG(kDefLog, kInfo,
      "setData(int64_t) complete. Main buffer state: type=%s, isPacked=%d, "
      "physical_size=%zu, logical_length=%zu",
      gpu::toString(bufferType).c_str(), isPacked, bufferData.size,
      this->length);
}

void Buffer::setData(const uint64_t *inputData, size_t byteSize) {
  if (inputData == nullptr && byteSize > 0)
    throw std::invalid_argument("setData(uint64_t): inputData is null");

  size_t numElements = byteSize / sizeof(uint64_t);

  gpu::NumType originalDataType = ku64;
  // Internal state should be ku32, isPacked=true
  LOG(kDefLog, kInfo, "setData(uint64_t): %zu elements (logical length)",
      numElements);

  try {
    // Ensure buffer exists with correct internal state (ku32, isPacked=true)
    ensureBuffer(numElements, originalDataType);

    // Upload the data using the packing overload of gpu::toGPU
    if (numElements > 0) {
      LOG(kDefLog, kInfo,
          "setData(uint64_t): Uploading %zu uint64 elements (packed via "
          "gpu::toGPU) to main buffer",
          numElements);
      // Call the gpu::toGPU overload that handles uint64 packing
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 numElements);
    }

  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(uint64_t): Failed - %s", e.what());
    return; // Exit if ensureBuffer or upload fails
  }

  LOG(kDefLog, kInfo,
      "setData(uint64_t) complete. Main buffer state: type=%s, isPacked=%d, "
      "physical_size=%zu, logical_length=%zu",
      gpu::toString(bufferType).c_str(), isPacked, bufferData.size,
      this->length);
}

void Buffer::release() {
  if (bufferData.buffer != nullptr) {
    LOG(kDefLog, kInfo,
        "Releasing buffer (physical size: %zu, logical length: %zu, internal "
        "type: %s, isPacked: %d)",
        bufferData.size, length, gpu::toString(bufferType).c_str(), isPacked);
    wgpuBufferRelease(bufferData.buffer);
    bufferData.buffer = nullptr;
    bufferData.size = 0;
    bufferData.usage = WGPUBufferUsage_None;
    length = 0;
    bufferType = kUnknown;
    isPacked = false;
  }
}

} // namespace mgpu
