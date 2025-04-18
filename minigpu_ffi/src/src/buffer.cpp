#include "../include/buffer.h"
#include "../include/compute_shader.h"
#include "../include/conversion_kernels.h"
#include "../include/gpuh.h"

#include <stdexcept> // Required for std::runtime_error
#include <string>    // Required for std::to_string

// NOTE: Assumes buffer.h has been updated with:
// class Buffer {
// public:
//   // ... existing members ...
//   size_t length = 0; // Number of logical elements
//   // ... rest of class ...
// };

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
    // Depending on application structure, might want to throw
    // std::runtime_error here throw std::runtime_error("Failed to create GPU
    // context: " + std::string(ex.what()));
  }
}

void MGPU::initializeContextAsync(std::function<void()> callback) {
  // Consider using std::async or a thread pool for true async initialization
  std::thread([this, callback]() {
    try {
      initializeContext();
      if (ctx && callback) { // Only call callback if context is valid
        callback();
      } else if (!ctx) {
        LOG(kDefLog, kError,
            "Async context initialization failed, callback skipped.");
        // Optionally, invoke callback with an error status
      }
    } catch (const std::exception &ex) {
      // Log error from the async thread
      LOG(kDefLog, kError, "Async context initialization failed: %s",
          ex.what());
      // Optionally, invoke callback with an error status
    }
  }).detach(); // Detach thread for fire-and-forget async
}

void MGPU::destroyContext() {
  if (ctx) {
    // Ensure all GPU operations are complete before releasing context resources
    // wgpuDeviceTick(ctx->device); // May be needed depending on usage
    ctx.reset(); // Use reset() for unique_ptr to destroy the managed object
    LOG(kDefLog, kInfo, "GPU context destroyed successfully.");
  } else {
    LOG(kDefLog, kWarn, // Use Warn level for non-critical state info
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

// Destructor to ensure buffer release
Buffer::~Buffer() {
  release(); // Release GPU resources when Buffer object goes out of scope
}

// Move Constructor
Buffer::Buffer(Buffer &&other) noexcept
    : mgpu(other.mgpu), bufferData(other.bufferData),
      bufferType(other.bufferType), isPacked(other.isPacked),
      length(other.length) {
  // Prevent double release by nullifying the moved-from object's buffer
  other.bufferData.buffer = nullptr;
  other.bufferData.size = 0;
  other.length = 0;
}

// Move Assignment Operator
Buffer &Buffer::operator=(Buffer &&other) noexcept {
  if (this != &other) {
    release(); // Release existing resource

    bufferData = other.bufferData;
    bufferType = other.bufferType;
    isPacked = other.isPacked;
    length = other.length;

    // Prevent double release
    other.bufferData.buffer = nullptr;
    other.bufferData.size = 0;
    other.length = 0;
  }
  return *this;
}

void Buffer::createBuffer(size_t sizeParam, gpu::NumType requestedDataType) {
  release(); // Release existing buffer first

  size_t physicalByteSize = 0;
  size_t logicalLength = 0;
  gpu::NumType internalDataType = requestedDataType;
  bool packedState = false;

  if (requestedDataType == ki8 || requestedDataType == ku8) {
    // Treat sizeParam as the number of logical 8-bit elements
    logicalLength = sizeParam;
    internalDataType = ki32; // Store internally as unpacked i32
    physicalByteSize =
        logicalLength * sizeof(int32_t); // Calculate required physical size
    packedState = true;                  // Mark as representing packed data

    LOG(kDefLog, kInfo,
        "Creating buffer for %zu elements of requested type %d. Internal "
        "storage: type=%d, isPacked=%d, physical_size=%zu bytes.",
        logicalLength, requestedDataType, internalDataType, packedState,
        physicalByteSize);

  } else {
    // Treat sizeParam as the requested physical byte size
    physicalByteSize = sizeParam;
    internalDataType = requestedDataType;
    packedState = false;
    size_t elementSizeBytes = gpu::sizeBytes(internalDataType);

    if (elementSizeBytes > 0) {
      logicalLength = physicalByteSize / elementSizeBytes;
      if (physicalByteSize % elementSizeBytes != 0) {
        LOG(kDefLog, kWarn,
            "Requested byte size (%zu) is not a multiple of element size (%zu) "
            "for type %d. Length calculation might be inaccurate.",
            physicalByteSize, elementSizeBytes, internalDataType);
        // Optionally adjust physicalByteSize or throw an error
      }
    } else if (physicalByteSize > 0 && internalDataType != kUnknown) {
      logicalLength = 0; // Unknown element size
      LOG(kDefLog, kError,
          "Cannot determine element size for type %d, but byte size > 0. "
          "Setting length to 0.",
          internalDataType);
    } else {
      logicalLength = 0; // Zero size or unknown type
    }
    LOG(kDefLog, kInfo,
        "Creating buffer: requested physical size=%zu bytes, type=%d. "
        "Calculated logical length: %zu elements.",
        physicalByteSize, internalDataType, logicalLength);
  }

  // --- Common Buffer Creation Logic ---

  // Ensure buffer size is not zero if length is non-zero (should be caught by
  // above logic, but double-check)
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

  // If size is still 0 after adjustments, nothing to create
  if (physicalByteSize == 0 && logicalLength == 0) {
    LOG(kDefLog, kInfo,
        "Requested buffer size results in 0 physical bytes. Buffer not "
        "created.");
    // Ensure state is reset
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
  // descriptor.label = "mgpu::Buffer";

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
      "%zu, type: %d, isPacked: %d)",
      physicalByteSize, logicalLength, internalDataType, packedState);

  // Set the final state of the buffer object
  this->bufferData = gpu::Array{
      .buffer = newBuffer,
      .usage = usage,
      .size = physicalByteSize, // Store the actual physical size
  };
  this->bufferType = internalDataType;
  this->length = logicalLength;
  this->isPacked = packedState;
}

void Buffer::readSync(void *outputData, NumType readAsType,
                      size_t readElementCount, size_t readElementOffset) {

  if (bufferData.buffer == nullptr) {
    LOG(kDefLog, kError, "readSync: Buffer is null or uninitialized.");
    // Consider throwing an exception for invalid state
    // throw std::runtime_error("readSync called on null buffer");
    return;
  }
  if (outputData == nullptr) {
    LOG(kDefLog, kError, "readSync: Output data pointer is null.");
    // throw std::invalid_argument("readSync outputData cannot be null");
    return;
  }

  if (readAsType == ki8) {
    // --- Read as int8_t (Requires Packing) ---
    LOG(kDefLog, kInfo, "readSync(ki8): Reading %zu elements at offset %zu.",
        readElementCount, readElementOffset);

    // State Validation: Expect buffer to hold unpacked int32 derived from int8
    if (bufferType != ki32 || !isPacked) {
      LOG(kDefLog, kError,
          "readSync(ki8): Buffer state invalid. Expected type=ki32 and "
          "isPacked=true, but got type=%d, isPacked=%d.",
          bufferType, isPacked);
      // throw std::runtime_error("readSync(ki8) called on buffer with
      // incompatible state");
      return;
    }

    // Use logical length (number of original int8 elements) for bounds checking
    size_t totalInt8Elements = this->length;
    if (totalInt8Elements == 0) {
      LOG(kDefLog, kWarn,
          "readSync(ki8): Buffer logical length is 0. Nothing to read.");
      return;
    }

    // Bounds check for requested read range (elements)
    if (readElementOffset >= totalInt8Elements) {
      LOG(kDefLog, kError,
          "readSync(ki8): Read offset (%zu) is out of bounds for buffer length "
          "(%zu).",
          readElementOffset, totalInt8Elements);
      return;
    }
    size_t clampedElementCount = readElementCount;
    if (readElementOffset + clampedElementCount > totalInt8Elements) {
      clampedElementCount = totalInt8Elements - readElementOffset;
      LOG(kDefLog, kWarn,
          "readSync(ki8): Read request (%zu elements) exceeds buffer bounds "
          "(%zu available at offset %zu). Clamping read count to %zu elements.",
          readElementCount, clampedElementCount, readElementOffset,
          clampedElementCount);
    }
    if (clampedElementCount == 0) {
      LOG(kDefLog, kInfo,
          "readSync(ki8): Clamped read count is 0. Nothing to read.");
      return;
    }

    // --- Temporary Packed Buffer ---
    // Create a temporary buffer to hold the GPU-packed int8 data (as int32)
    size_t numPackedInt32Elements =
        (totalInt8Elements + 3) / 4; // Total packed elements needed
    size_t packedBufferSize =
        numPackedInt32Elements * sizeof(int32_t); // Physical size

    LOG(kDefLog, kInfo,
        "readSync(ki8): Creating temporary buffer for packing (physical size: "
        "%zu bytes, %zu packed i32 elements).",
        packedBufferSize, numPackedInt32Elements);

    Buffer tempPackedBuffer(mgpu);
    // Create buffer using physical byte size, type is ki32
    tempPackedBuffer.createBuffer(packedBufferSize, ki32);
    if (tempPackedBuffer.bufferData.buffer == nullptr) {
      LOG(kDefLog, kError,
          "readSync(ki8): Failed to create temporary packed buffer.");
      // tempPackedBuffer destructor will handle potential partial creation
      return;
    }

    // --- GPU Packing ---
    LOG(kDefLog, kInfo,
        "readSync(ki8): Dispatching kernel to pack %zu elements from main "
        "buffer into temporary buffer.",
        this->length);
    // Pack from *this (unpacked i32, logical length = totalInt8Elements)
    // into tempPackedBuffer (packed i32, logical length =
    // numPackedInt32Elements) Assumes dispatchI32toPackedI8 uses buffer.length
    // correctly.
    kernels::dispatchI32toPackedI8(mgpu, *this, tempPackedBuffer);

    // --- CPU Read and Unpack ---
    // Calculate byte offset into the *temporary packed* buffer
    size_t sourceOffsetBytes = (readElementOffset / 4) * sizeof(int32_t);
    // Bounds check for read offset against physical size of temp buffer
    if (sourceOffsetBytes >= tempPackedBuffer.bufferData.size) {
      LOG(kDefLog, kError,
          "readSync(ki8): Calculated read byte offset (%zu) is out of bounds "
          "for temporary packed buffer size (%zu).",
          sourceOffsetBytes, tempPackedBuffer.bufferData.size);
      return; // Should not happen if element offset check passed, but good for
              // safety
    }

    LOG(kDefLog, kInfo,
        "readSync(ki8): Reading from temp packed buffer and unpacking %zu int8 "
        "elements (element offset %zu / byte offset %zu).",
        clampedElementCount, readElementOffset, sourceOffsetBytes);

    // gpu::toCPU handles reading the necessary packed i32 data and unpacking to
    // int8
    gpu::toCPU(
        this->mgpu.getContext(), tempPackedBuffer.bufferData.buffer,
        ki8,        // Signal to gpu::toCPU to perform i32->i8 unpacking.
        outputData, // Destination pointer (int8_t*)
        clampedElementCount, // Number of *target* int8 elements to unpack.
        sourceOffsetBytes);  // Byte offset in the source (tempPackedBuffer) GPU
                             // buffer.

    // tempPackedBuffer goes out of scope, its destructor calls release()
    LOG(kDefLog, kInfo, "readSync(ki8) complete.");

  } else {
    // --- Handle non-ki8 cases (Direct Read) ---
    LOG(kDefLog, kInfo,
        "readSync: Reading %zu elements as type %d at offset %zu. Buffer "
        "state: type=%d, isPacked=%d, length=%zu, physical_size=%zu",
        readElementCount, readAsType, readElementOffset, bufferType, isPacked,
        this->length, bufferData.size);

    // State Validation: Check if reading non-ki32 from an unpacked-int8 buffer
    if (isPacked && bufferType == ki32 && readAsType != ki32) {
      LOG(kDefLog, kWarn,
          "readSync: Reading type %d from buffer holding unpacked int32 "
          "representation of int8 data. Data might be misinterpreted.",
          readAsType);
    } else if (isPacked && bufferType != ki32) {
      // This indicates an internal logic error
      LOG(kDefLog, kError,
          "readSync: Buffer marked as 'isPacked' but type is not ki32 (%d). "
          "Invalid state.",
          bufferType);
      // throw std::runtime_error("readSync called on buffer with inconsistent
      // isPacked state");
      return;
    }
    // Optional: Check if readAsType matches bufferType for non-packed buffers
    // if (!isPacked && readAsType != bufferType && bufferType != kUnknown) {
    //     LOG(kDefLog, kWarn, "readSync: Reading type %d from buffer of type
    //     %d.", readAsType, bufferType);
    // }

    // Calculate byte size and offset for the requested type
    size_t elementSizeBytes = gpu::sizeBytes(readAsType);
    if (elementSizeBytes == 0) {
      LOG(kDefLog, kError, "readSync: Unknown size for requested data type %d.",
          readAsType);
      // throw std::invalid_argument("readSync requested with unknown data
      // type");
      return;
    }

    // --- Bounds Checking ---
    // 1. Check element offset and count against logical length
    if (this->length == 0 && (readElementOffset > 0 || readElementCount > 0)) {
      LOG(kDefLog, kWarn,
          "readSync: Attempting to read from buffer with logical length 0.");
      return; // Nothing to read
    }
    if (readElementOffset >= this->length) {
      LOG(kDefLog, kError,
          "readSync: Read element offset (%zu) is out of bounds for buffer "
          "length (%zu).",
          readElementOffset, this->length);
      return;
    }
    size_t clampedElementCount = readElementCount;
    if (readElementOffset + clampedElementCount > this->length) {
      clampedElementCount = this->length - readElementOffset;
      LOG(kDefLog, kWarn,
          "readSync: Read request (%zu elements) exceeds buffer bounds (%zu "
          "available at offset %zu). Clamping read count to %zu elements.",
          readElementCount, clampedElementCount, readElementOffset,
          clampedElementCount);
    }
    if (clampedElementCount == 0) {
      LOG(kDefLog, kInfo,
          "readSync: Clamped read count is 0. Nothing to read.");
      return;
    }

    // 2. Calculate byte range and check against physical size
    size_t totalBytesToRead = clampedElementCount * elementSizeBytes;
    size_t sourceOffsetBytes =
        readElementOffset *
        elementSizeBytes; // Potential issue if bufferType != readAsType

    // Safety check: Ensure calculated byte range fits within physical buffer
    // size
    if (sourceOffsetBytes >= bufferData.size) {
      LOG(kDefLog, kError,
          "readSync: Calculated byte offset (%zu) is out of bounds for buffer "
          "physical size (%zu). (Element offset %zu, Element size %zu)",
          sourceOffsetBytes, bufferData.size, readElementOffset,
          elementSizeBytes);
      return;
    }
    if (sourceOffsetBytes + totalBytesToRead > bufferData.size) {
      size_t availableBytes = bufferData.size - sourceOffsetBytes;
      LOG(kDefLog, kError,
          "readSync: Calculated byte read request (%zu) exceeds physical "
          "buffer size (%zu available at offset %zu). Clamping read bytes to "
          "%zu. This might indicate inconsistent state or read type mismatch.",
          totalBytesToRead, availableBytes, sourceOffsetBytes, availableBytes);
      totalBytesToRead = availableBytes;
      // Recalculate element count based on clamped bytes
      clampedElementCount = totalBytesToRead / elementSizeBytes;
      if (totalBytesToRead == 0)
        return;
    }

    LOG(kDefLog, kInfo,
        "readSync: Reading %zu bytes (offset %zu bytes) from main buffer "
        "directly.",
        totalBytesToRead, sourceOffsetBytes);
    gpu::toCPU(this->mgpu.getContext(), bufferData.buffer, readAsType,
               outputData,
               totalBytesToRead,   // Pass total bytes to read
               sourceOffsetBytes); // Pass offset in bytes
  }
}

void Buffer::readAsync(void *outputData, NumType dType, size_t size,
                       size_t offset, std::function<void()> callback) {
  // Basic async wrapper using std::thread. Consider a thread pool for
  // efficiency.
  std::thread([=]() {
    try {
      readSync(outputData, dType, size, offset);
      if (callback) {
        callback();
      }
    } catch (const std::exception &e) {
      LOG(kDefLog, kError, "Exception in readAsync thread: %s", e.what());
      // How to report error back? Callback could take an error parameter.
      // if (callback) callback(e); // Example
    } catch (...) {
      LOG(kDefLog, kError, "Unknown exception in readAsync thread.");
      // if (callback) callback(std::runtime_error("Unknown async error")); //
      // Example
    }
  }).detach();
}

// --- setData Overloads ---

// Helper to manage buffer creation/resizing and state updates
void Buffer::ensureBuffer(size_t requiredByteSize, gpu::NumType dataType,
                          size_t logicalLength, bool packedState) {
  bool needsRecreation = false;
  if (bufferData.buffer == nullptr) {
    LOG(kDefLog, kInfo, "ensureBuffer: No buffer exists. Creating new one.");
    needsRecreation = true;
  } else if (requiredByteSize > bufferData.size) {
    LOG(kDefLog, kInfo,
        "ensureBuffer: Buffer too small (current: %zu, required: %zu). "
        "Recreating.",
        bufferData.size, requiredByteSize);
    needsRecreation = true;
  } else if (bufferType != dataType || isPacked != packedState) {
    LOG(kDefLog, kInfo,
        "ensureBuffer: Buffer type or packed state mismatch (current: type=%d "
        "packed=%d, required: type=%d packed=%d). Reusing buffer, updating "
        "state.",
        bufferType, isPacked, dataType, packedState);
    // No recreation needed, just update state below
  } else {
    LOG(kDefLog, kInfo,
        "ensureBuffer: Reusing existing buffer (size: %zu, type: %d, packed: "
        "%d).",
        bufferData.size, bufferType, isPacked);
  }

  if (needsRecreation) {
    createBuffer(requiredByteSize, dataType);
    if (bufferData.buffer == nullptr) {
      // createBuffer logs the error
      throw std::runtime_error(
          "Failed to create/recreate buffer in ensureBuffer");
    }
    // createBuffer sets initial length based on physical size and type,
    // but we need to set the intended logical length and packed state.
    this->length = logicalLength;
    this->isPacked = packedState;
    // bufferType is already set by createBuffer
    LOG(kDefLog, kInfo,
        "ensureBuffer: New buffer created. State set: type=%d, packed=%d, "
        "length=%zu",
        bufferType, isPacked, this->length);
  } else {
    // Reusing buffer, ensure state is correct
    this->bufferType = dataType;
    this->length = logicalLength;
    this->isPacked = packedState;
  }
}

void Buffer::setData(const float *inputData, size_t numElements) {
  if (inputData == nullptr && numElements > 0)
    throw std::invalid_argument("setData(float): inputData is null");
  size_t byteSize = numElements * sizeof(float);
  size_t logicalLength = numElements;
  gpu::NumType dataType = kf32;
  bool packedState = false;

  LOG(kDefLog, kInfo, "setData(float): %zu elements (%zu bytes)", logicalLength,
      byteSize);
  try {
    ensureBuffer(byteSize, dataType, logicalLength, packedState);
    if (byteSize > 0) {
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 byteSize);
    }
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(float): Failed - %s", e.what());
    // Handle or re-throw
  }
}

void Buffer::setData(const double *inputData, size_t numElements) {
  if (inputData == nullptr && numElements > 0)
    throw std::invalid_argument("setData(double): inputData is null");
  size_t byteSize = numElements * sizeof(double);
  size_t logicalLength = numElements;
  gpu::NumType dataType = kf64;
  bool packedState = false;

  LOG(kDefLog, kInfo, "setData(double): %zu elements (%zu bytes)",
      logicalLength, byteSize);
  try {
    ensureBuffer(byteSize, dataType, logicalLength, packedState);
    if (byteSize > 0) {
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 byteSize);
    }
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(double): Failed - %s", e.what());
  }
}

void Buffer::setData(const int8_t *inputData, size_t numElements) {
  if (inputData == nullptr && numElements > 0)
    throw std::invalid_argument("setData(int8_t): inputData is null");
  // numElements is the number of int8 elements (logical length)
  size_t logicalLength = numElements;
  gpu::NumType finalDataType = ki32; // Data will be stored as unpacked int32
  bool finalPackedState = true;      // Mark as derived from int8

  LOG(kDefLog, kInfo, "setData(int8_t): %zu elements (logical length)",
      logicalLength);

  // 1. Temporary buffer for initial packed upload via gpu::toGPU
  size_t packedUploadByteSize = (logicalLength + 3) / 4 * sizeof(int32_t);
  Buffer initialPackedUploadBuffer(mgpu);
  LOG(kDefLog, kInfo,
      "setData(int8_t): Creating temporary packed upload buffer (size: %zu "
      "bytes)",
      packedUploadByteSize);
  initialPackedUploadBuffer.createBuffer(packedUploadByteSize, ki32);
  if (initialPackedUploadBuffer.bufferData.buffer == nullptr) {
    LOG(kDefLog, kError,
        "setData(int8_t): Failed to create temporary packed buffer.");
    return; // Or throw
  }

  // 2. Upload CPU int8 data, packing it into the temporary buffer
  if (logicalLength > 0) {
    LOG(kDefLog, kInfo,
        "setData(int8_t): Uploading %zu int8 elements (packed via gpu::toGPU) "
        "to temporary buffer",
        logicalLength);
    // Pass logicalLength (number of int8 elements) to gpu::toGPU for ki8 case
    gpu::toGPU(this->mgpu.getContext(), inputData,
               initialPackedUploadBuffer.bufferData.buffer, logicalLength);
  }

  // 3. Ensure main buffer ('this') exists and is sized for UNPACKED int32 data.
  size_t unpackedByteSize =
      logicalLength * sizeof(int32_t); // Physical size needed
  LOG(kDefLog, kInfo,
      "setData(int8_t): Ensuring main buffer exists for unpacked data "
      "(physical size: %zu bytes)",
      unpackedByteSize);
  try {
    // Ensure buffer exists with correct physical size, type ki32, intended
    // logical length, and packed state
    ensureBuffer(unpackedByteSize, finalDataType, logicalLength,
                 finalPackedState);
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(int8_t): Failed ensuring main buffer - %s",
        e.what());
    return; // Or throw
  }

  // 4. Dispatch the unpack kernel: initialPackedUploadBuffer ->
  // this->bufferData
  if (logicalLength > 0) {
    LOG(kDefLog, kInfo,
        "setData(int8_t): Dispatching kernel to unpack data from temporary "
        "buffer into main buffer");
    // Assumes dispatchPackedI8toI32 uses buffer.length correctly
    kernels::dispatchPackedI8toI32(mgpu, initialPackedUploadBuffer, *this);
  }

  // 5. Final state is set by ensureBuffer

  // 6. initialPackedUploadBuffer goes out of scope and is released.
  LOG(kDefLog, kInfo,
      "setData(int8_t) complete. Main buffer state: type=%d, isPacked=%d, "
      "physical_size=%zu, logical_length=%zu",
      bufferType, isPacked, bufferData.size, this->length);
}

// --- Other setData overloads (Simplified logging, using ensureBuffer) ---

void Buffer::setData(const uint16_t *inputData, size_t numElements) {
  if (inputData == nullptr && numElements > 0)
    throw std::invalid_argument("setData(uint16_t): inputData is null");
  size_t byteSize = numElements * sizeof(uint16_t);
  size_t logicalLength = numElements;
  gpu::NumType dataType = ku16; // Assuming unsigned, use ki16 if signed
  bool packedState = false;

  LOG(kDefLog, kInfo, "setData(uint16_t): %zu elements (%zu bytes)",
      logicalLength, byteSize);
  try {
    ensureBuffer(byteSize, dataType, logicalLength, packedState);
    if (byteSize > 0) {
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 byteSize);
    }
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(uint16_t): Failed - %s", e.what());
  }
}

void Buffer::setData(const uint32_t *inputData, size_t numElements) {
  if (inputData == nullptr && numElements > 0)
    throw std::invalid_argument("setData(uint32_t): inputData is null");
  size_t byteSize = numElements * sizeof(uint32_t);
  size_t logicalLength = numElements;
  gpu::NumType dataType = ku32; // Assuming unsigned, use ki32 if signed
  bool packedState = false;

  LOG(kDefLog, kInfo, "setData(uint32_t): %zu elements (%zu bytes)",
      logicalLength, byteSize);
  try {
    ensureBuffer(byteSize, dataType, logicalLength, packedState);
    if (byteSize > 0) {
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 byteSize);
    }
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(uint32_t): Failed - %s", e.what());
  }
}

void Buffer::setData(const uint64_t *inputData, size_t numElements) {
  if (inputData == nullptr && numElements > 0)
    throw std::invalid_argument("setData(uint64_t): inputData is null");
  size_t byteSize = numElements * sizeof(uint64_t);
  size_t logicalLength = numElements;
  gpu::NumType dataType = ku64; // Assuming unsigned, use ki64 if signed
  bool packedState = false;

  LOG(kDefLog, kInfo, "setData(uint64_t): %zu elements (%zu bytes)",
      logicalLength, byteSize);
  try {
    ensureBuffer(byteSize, dataType, logicalLength, packedState);
    if (byteSize > 0) {
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 byteSize);
    }
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(uint64_t): Failed - %s", e.what());
  }
}

// Overload for uint8_t - Currently just uploads bytes, NO unpacking.
// If unpacking to u32 is needed, implement similar logic to int8_t with a
// specific kernel.
void Buffer::setData(const uint8_t *inputData, size_t numElements) {
  if (inputData == nullptr && numElements > 0)
    throw std::invalid_argument("setData(uint8_t): inputData is null");
  size_t byteSize = numElements * sizeof(uint8_t);
  size_t logicalLength = numElements;
  gpu::NumType dataType = ku8; // Represent as unsigned 8-bit
  bool packedState = false;    // Not using the packed/unpacked mechanism here

  LOG(kDefLog, kInfo,
      "setData(uint8_t): %zu elements (%zu bytes). NOTE: Uploading as raw "
      "bytes, type ku8.",
      logicalLength, byteSize);
  // Warning: The previous implementation attempted unpacking but was flawed.
  // This version simply uploads the bytes. If unpacking to u32 is required,
  // a dedicated kernel (e.g., dispatchPackedU8toU32) and logic similar to
  // setData(int8_t) must be implemented.

  try {
    // Ensure buffer exists, sized for the raw bytes.
    ensureBuffer(byteSize, dataType, logicalLength, packedState);
    if (byteSize > 0) {
      // gpu::toGPU needs to handle ku8 correctly (likely direct byte copy)
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 byteSize);
    }
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(uint8_t): Failed - %s", e.what());
  }
}

void Buffer::setData(const int16_t *inputData, size_t numElements) {
  if (inputData == nullptr && numElements > 0)
    throw std::invalid_argument("setData(int16_t): inputData is null");
  size_t byteSize = numElements * sizeof(int16_t);
  size_t logicalLength = numElements;
  gpu::NumType dataType = ki16; // Assuming signed
  bool packedState = false;

  LOG(kDefLog, kInfo, "setData(int16_t): %zu elements (%zu bytes)",
      logicalLength, byteSize);
  try {
    ensureBuffer(byteSize, dataType, logicalLength, packedState);
    if (byteSize > 0) {
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 byteSize);
    }
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(int16_t): Failed - %s", e.what());
  }
}

void Buffer::setData(const int32_t *inputData, size_t numElements) {
  if (inputData == nullptr && numElements > 0)
    throw std::invalid_argument("setData(int32_t): inputData is null");
  size_t byteSize = numElements * sizeof(int32_t);
  size_t logicalLength = numElements;
  gpu::NumType dataType = ki32; // Assuming signed
  bool packedState = false;

  LOG(kDefLog, kInfo, "setData(int32_t): %zu elements (%zu bytes)",
      logicalLength, byteSize);
  try {
    ensureBuffer(byteSize, dataType, logicalLength, packedState);
    if (byteSize > 0) {
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 byteSize);
    }
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(int32_t): Failed - %s", e.what());
  }
}

void Buffer::setData(const int64_t *inputData, size_t numElements) {
  if (inputData == nullptr && numElements > 0)
    throw std::invalid_argument("setData(int64_t): inputData is null");
  size_t byteSize = numElements * sizeof(int64_t);
  size_t logicalLength = numElements;
  gpu::NumType dataType = ki64; // Assuming signed
  bool packedState = false;

  LOG(kDefLog, kInfo, "setData(int64_t): %zu elements (%zu bytes)",
      logicalLength, byteSize);
  try {
    ensureBuffer(byteSize, dataType, logicalLength, packedState);
    if (byteSize > 0) {
      gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer,
                 byteSize);
    }
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "setData(int64_t): Failed - %s", e.what());
  }
}

void Buffer::release() {
  if (bufferData.buffer != nullptr) {
    LOG(kDefLog, kInfo,
        "Releasing buffer (physical size: %zu, logical length: %zu)",
        bufferData.size, length);
    wgpuBufferRelease(bufferData.buffer);
    // Reset state after release
    bufferData.buffer = nullptr;
    bufferData.size = 0;
    bufferData.usage = WGPUBufferUsage_None;
    length = 0;
    bufferType = kUnknown;
    isPacked = false;
  }
}

} // namespace mgpu