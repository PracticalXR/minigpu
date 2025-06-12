#include "../include/conversion_kernels.h"
#include "../include/buffer.h" // Full definition of mgpu::Buffer
#include "../include/compute_shader.h" // Full definition of mgpu::MGPU and ComputeShader
#include <cstdint>
#include <stdexcept> // For runtime_error

namespace mgpu {
namespace kernels {

// no external dependencies
#define SIMPLE_LOG(level, msg, ...) \
  do { \
    /* Simple logging or remove entirely for performance */ \
  } while(0)

// Helper to calculate workgroup count for a 1D dispatch
static int calculateGroups(size_t numElements, size_t workgroupSize = 256) {
  if (workgroupSize == 0) {
    SIMPLE_LOG("ERROR", "calculateGroups: Workgroup size cannot be zero.");
    return 0; // Avoid division by zero
  }
  return static_cast<int>((numElements + workgroupSize - 1) / workgroupSize);
}

static bool validateBufferSafety(const Buffer &sourceBuffer,
                                const Buffer &destBuffer,
                                size_t expectedSourceElements,
                                size_t expectedDestElements,
                                const std::string &kernelName) {

  // Validate source buffer
  if (sourceBuffer.bufferData.buffer == nullptr) {
    SIMPLE_LOG("ERROR", "validateBufferSafety(%s): Source buffer is null", kernelName.c_str());
    return false;
  }

  if (sourceBuffer.getLength() < expectedSourceElements) {
    SIMPLE_LOG("ERROR", "validateBufferSafety(%s): Source buffer length (%zu) < expected elements (%zu)",
        kernelName.c_str(), sourceBuffer.getLength(), expectedSourceElements);
    return false;
  }

  // Validate destination buffer
  if (destBuffer.bufferData.buffer == nullptr) {
    SIMPLE_LOG("ERROR", "validateBufferSafety(%s): Destination buffer is null", kernelName.c_str());
    return false;
  }

  if (destBuffer.getLength() < expectedDestElements) {
    SIMPLE_LOG("ERROR", "validateBufferSafety(%s): Destination buffer length (%zu) < expected elements (%zu)",
        kernelName.c_str(), destBuffer.getLength(), expectedDestElements);
    return false;
  }

  // Basic size validation
  if (sourceBuffer.getSize() == 0 || destBuffer.getSize() == 0) {
    SIMPLE_LOG("ERROR", "validateBufferSafety(%s): Zero-sized buffer detected", kernelName.c_str());
    return false;
  }

  SIMPLE_LOG("INFO", "validateBufferSafety(%s): PASSED - Source: %zu elements, %zu bytes. Dest: %zu elements, %zu bytes",
      kernelName.c_str(), sourceBuffer.getLength(), sourceBuffer.getSize(),
      destBuffer.getLength(), destBuffer.getSize());

  return true;
}

// Enhanced safety validation for workgroup dispatch limits
static bool validateWorkgroupSafety(size_t numElements,
                                   size_t workgroupSize,
                                   const std::string &kernelName) {
  if (workgroupSize == 0) {
    SIMPLE_LOG("ERROR", "validateWorkgroupSafety(%s): Workgroup size cannot be zero", kernelName.c_str());
    return false;
  }

  size_t numWorkgroups = (numElements + workgroupSize - 1) / workgroupSize;

  // WebGPU limits
  const size_t MAX_WORKGROUPS_X = 65535;
  const size_t MAX_TOTAL_INVOCATIONS = 256 * 65535; // Conservative estimate

  if (numWorkgroups > MAX_WORKGROUPS_X) {
    SIMPLE_LOG("ERROR", "validateWorkgroupSafety(%s): Workgroup count (%zu) exceeds X-dimension limit (%zu)",
        kernelName.c_str(), numWorkgroups, MAX_WORKGROUPS_X);
    return false;
  }

  size_t totalInvocations = numWorkgroups * workgroupSize;
  if (totalInvocations > MAX_TOTAL_INVOCATIONS) {
    SIMPLE_LOG("ERROR", "validateWorkgroupSafety(%s): Total invocations (%zu) exceeds limit (%zu)",
        kernelName.c_str(), totalInvocations, MAX_TOTAL_INVOCATIONS);
    return false;
  }

  SIMPLE_LOG("INFO", "validateWorkgroupSafety(%s): PASSED - %zu elements, %zu workgroups, %zu invocations",
      kernelName.c_str(), numElements, numWorkgroups, totalInvocations);

  return true;
}

void dispatchPackedI8toI32(MGPU &mgpu, Buffer &packed_input, Buffer &unpacked_output) {
  // Basic validation
  if (packed_input.getDataType() != kInt32) {
    SIMPLE_LOG("ERROR", "dispatchPackedI8toI32: Packed input buffer type is not kInt32");
    return;
  }
  if (unpacked_output.getDataType() != kInt32) {
    SIMPLE_LOG("ERROR", "dispatchPackedI8toI32: Unpacked output buffer type is not kInt32");
    return;
  }
  if (unpacked_output.getLength() == 0) {
    SIMPLE_LOG("WARN", "dispatchPackedI8toI32: Unpacked output buffer length is 0. Nothing to unpack.");
    return;
  }

  // Calculate dispatch parameters
  size_t numLogicalElements = unpacked_output.getLength();
  size_t numPackedElements = (numLogicalElements + 3) / 4;

  // Safety validation
  if (!validateBufferSafety(packed_input, unpacked_output, numPackedElements, numLogicalElements, "dispatchPackedI8toI32")) {
    SIMPLE_LOG("ERROR", "dispatchPackedI8toI32: Buffer safety validation FAILED. Aborting.");
    return;
  }

  if (!validateWorkgroupSafety(numPackedElements, 256, "dispatchPackedI8toI32")) {
    SIMPLE_LOG("ERROR", "dispatchPackedI8toI32: Workgroup safety validation FAILED. Aborting.");
    return;
  }

  SIMPLE_LOG("INFO", "dispatchPackedI8toI32: Unpacking %zu logical elements. Dispatching %zu packed elements.",
      numLogicalElements, numPackedElements);

  try {
    ComputeShader shader(mgpu);
    shader.loadKernelString(kPackedInt8ToInt32Kernel);
    shader.setBuffer(0, packed_input);
    shader.setBuffer(1, unpacked_output);

    int groupsX = calculateGroups(numPackedElements);
    SIMPLE_LOG("INFO", "dispatchPackedI8toI32: Dispatching %d workgroups", groupsX);
    shader.dispatch(groupsX, 1, 1);

    SIMPLE_LOG("INFO", "dispatchPackedI8toI32: Kernel dispatch completed successfully");

  } catch (const std::exception &e) {
    SIMPLE_LOG("ERROR", "dispatchPackedI8toI32: Exception during kernel dispatch: %s", e.what());
    throw;
  }
}

void dispatchI32toPackedI8(MGPU &mgpu, Buffer &unpacked_input, Buffer &packed_output) {
  // Validation
  if (unpacked_input.getDataType() != kInt8) {
    SIMPLE_LOG("ERROR", "dispatchI32toPackedI8: Unpacked input buffer type should be kInt8");
    return;
  }
  if (packed_output.getDataType() != kInt32) {
    SIMPLE_LOG("ERROR", "dispatchI32toPackedI8: Packed output buffer type is not kInt32");
    return;
  }
  if (unpacked_input.getLength() == 0) {
    SIMPLE_LOG("WARN", "dispatchI32toPackedI8: Unpacked input buffer length is 0. Nothing to pack.");
    return;
  }

  size_t numLogicalElements = unpacked_input.getLength();
  size_t numPackedElements = (numLogicalElements + 3) / 4;

  SIMPLE_LOG("INFO", "dispatchI32toPackedI8: Packing %zu logical elements into %zu packed elements.",
      numLogicalElements, numPackedElements);

  try {
    ComputeShader shader(mgpu);
    shader.loadKernelString(kInt32ToPackedInt8Kernel);
    shader.setBuffer(0, unpacked_input);
    shader.setBuffer(1, packed_output);

    int groupsX = calculateGroups(numPackedElements);
    shader.dispatch(groupsX, 1, 1);

    SIMPLE_LOG("INFO", "dispatchI32toPackedI8: Kernel dispatch completed successfully");
  } catch (const std::exception &e) {
    SIMPLE_LOG("ERROR", "dispatchI32toPackedI8: Exception during kernel dispatch: %s", e.what());
    throw;
  }
}

void dispatchPackedU8toU32(MGPU &mgpu, Buffer &packed_input, Buffer &unpacked_output) {
  // Similar validation pattern...
  if (packed_input.getDataType() != kUInt32) {
    SIMPLE_LOG("ERROR", "dispatchPackedU8toU32: Packed input buffer type is not kUInt32");
    return;
  }
  if (unpacked_output.getDataType() != kUInt32) {
    SIMPLE_LOG("ERROR", "dispatchPackedU8toU32: Unpacked output buffer type is not kUInt32");
    return;
  }
  if (unpacked_output.getLength() == 0) {
    SIMPLE_LOG("WARN", "dispatchPackedU8toU32: Unpacked output buffer length is 0. Nothing to unpack.");
    return;
  }

  size_t numLogicalElements = unpacked_output.getLength();
  size_t numPackedElements = (numLogicalElements + 3) / 4;

  if (!validateBufferSafety(packed_input, unpacked_output, numPackedElements, numLogicalElements, "dispatchPackedU8toU32")) {
    SIMPLE_LOG("ERROR", "dispatchPackedU8toU32: Buffer safety validation FAILED. Aborting.");
    return;
  }

  if (!validateWorkgroupSafety(numPackedElements, 256, "dispatchPackedU8toU32")) {
    SIMPLE_LOG("ERROR", "dispatchPackedU8toU32: Workgroup safety validation FAILED. Aborting.");
    return;
  }

  try {
    ComputeShader shader(mgpu);
    shader.loadKernelString(kPackedUInt8ToUInt32Kernel);
    shader.setBuffer(0, packed_input);
    shader.setBuffer(1, unpacked_output);
    shader.dispatch(calculateGroups(numPackedElements), 1, 1);

    SIMPLE_LOG("INFO", "dispatchPackedU8toU32: Kernel dispatch completed successfully");
  } catch (const std::exception &e) {
    SIMPLE_LOG("ERROR", "dispatchPackedU8toU32: Exception during kernel dispatch: %s", e.what());
    throw;
  }
}

void dispatchU32toPackedU8(MGPU &mgpu, Buffer &unpacked_input, Buffer &packed_output) {
  // Validation
  if (unpacked_input.getDataType() != kUInt8) {
    SIMPLE_LOG("ERROR", "dispatchU32toPackedU8: Unpacked input buffer type should be kUInt8");
    return;
  }
  if (packed_output.getDataType() != kUInt32) {
    SIMPLE_LOG("ERROR", "dispatchU32toPackedU8: Packed output buffer type is not kUInt32");
    return;
  }
  if (unpacked_input.getLength() == 0) {
    SIMPLE_LOG("WARN", "dispatchU32toPackedU8: Unpacked input buffer length is 0. Nothing to pack.");
    return;
  }

  size_t numLogicalElements = unpacked_input.getLength();
  size_t numPackedElements = (numLogicalElements + 3) / 4;

  SIMPLE_LOG("INFO", "dispatchU32toPackedU8: Packing %zu logical elements into %zu packed elements.",
      numLogicalElements, numPackedElements);

  try {
    ComputeShader shader(mgpu);
    shader.loadKernelString(kUInt32ToPackedUInt8Kernel);
    shader.setBuffer(0, unpacked_input);
    shader.setBuffer(1, packed_output);
    shader.dispatch(calculateGroups(numPackedElements), 1, 1);
  } catch (const std::exception &e) {
    SIMPLE_LOG("ERROR", "dispatchU32toPackedU8: Exception during kernel dispatch: %s", e.what());
    throw;
  }
}

void dispatchPackedI16toI32(MGPU &mgpu, Buffer &packed_input, Buffer &unpacked_output) {
  if (packed_input.getDataType() != kInt32) {
    SIMPLE_LOG("ERROR", "dispatchPackedI16toI32: Packed input buffer type is not kInt32");
    return;
  }
  if (unpacked_output.getDataType() != kInt32) {
    SIMPLE_LOG("ERROR", "dispatchPackedI16toI32: Unpacked output buffer type is not kInt32");
    return;
  }
  if (unpacked_output.getLength() == 0) {
    SIMPLE_LOG("WARN", "dispatchPackedI16toI32: Unpacked output buffer length is 0. Nothing to unpack.");
    return;
  }

  size_t numLogicalElements = unpacked_output.getLength(); // Logical i16 elements
  size_t numPackedElements = (numLogicalElements + 1) / 2; // i32 elements needed for packing

  SIMPLE_LOG("INFO", "dispatchPackedI16toI32: Unpacking %zu logical elements from %zu packed elements.",
      numLogicalElements, numPackedElements);

  try {
    ComputeShader shader(mgpu);
    shader.loadKernelString(kPackedInt16ToInt32Kernel);
    shader.setBuffer(0, packed_input);
    shader.setBuffer(1, unpacked_output);
    shader.dispatch(calculateGroups(numPackedElements), 1, 1);
  } catch (const std::exception &e) {
    SIMPLE_LOG("ERROR", "dispatchPackedI16toI32: Exception during kernel dispatch: %s", e.what());
    throw;
  }
}

void dispatchI32toPackedI16(MGPU &mgpu, Buffer &unpacked_input, Buffer &packed_output) {
  if (unpacked_input.getDataType() != kInt16) {
    SIMPLE_LOG("ERROR", "dispatchI32toPackedI16: Unpacked input buffer type should be kInt16");
    return;
  }
  if (packed_output.getDataType() != kInt32) {
    SIMPLE_LOG("ERROR", "dispatchI32toPackedI16: Packed output buffer type is not kInt32");
    return;
  }
  if (unpacked_input.getLength() == 0) {
    SIMPLE_LOG("WARN", "dispatchI32toPackedI16: Unpacked input buffer length is 0. Nothing to pack.");
    return;
  }

  size_t numLogicalElements = unpacked_input.getLength(); // Logical i16 elements
  size_t numPackedElements = (numLogicalElements + 1) / 2; // i32 elements needed for packing

  SIMPLE_LOG("INFO", "dispatchI32toPackedI16: Packing %zu logical elements into %zu packed elements.",
      numLogicalElements, numPackedElements);

  try {
    ComputeShader shader(mgpu);
    shader.loadKernelString(kInt32ToPackedInt16Kernel);
    shader.setBuffer(0, unpacked_input);
    shader.setBuffer(1, packed_output);
    shader.dispatch(calculateGroups(numPackedElements), 1, 1);
  } catch (const std::exception &e) {
    SIMPLE_LOG("ERROR", "dispatchI32toPackedI16: Exception during kernel dispatch: %s", e.what());
    throw;
  }
}

void dispatchPackedU16toU32(MGPU &mgpu, Buffer &packed_input, Buffer &unpacked_output) {
  if (packed_input.getDataType() != kUInt32) {
    SIMPLE_LOG("ERROR", "dispatchPackedU16toU32: Packed input buffer type is not kUInt32");
    return;
  }
  if (unpacked_output.getDataType() != kUInt32) {
    SIMPLE_LOG("ERROR", "dispatchPackedU16toU32: Unpacked output buffer type is not kUInt32");
    return;
  }
  if (unpacked_output.getLength() == 0) {
    SIMPLE_LOG("WARN", "dispatchPackedU16toU32: Unpacked output buffer length is 0. Nothing to unpack.");
    return;
  }

  size_t numLogicalElements = unpacked_output.getLength(); // Logical u16 elements
  size_t numPackedElements = (numLogicalElements + 1) / 2; // u32 elements needed for packing

  SIMPLE_LOG("INFO", "dispatchPackedU16toU32: Unpacking %zu logical elements from %zu packed elements.",
      numLogicalElements, numPackedElements);

  try {
    ComputeShader shader(mgpu);
    shader.loadKernelString(kPackedUInt16ToUInt32Kernel);
    shader.setBuffer(0, packed_input);
    shader.setBuffer(1, unpacked_output);
    shader.dispatch(calculateGroups(numPackedElements), 1, 1);
  } catch (const std::exception &e) {
    SIMPLE_LOG("ERROR", "dispatchPackedU16toU32: Exception during kernel dispatch: %s", e.what());
    throw;
  }
}

void dispatchU32toPackedU16(MGPU &mgpu, Buffer &unpacked_input, Buffer &packed_output) {
  if (unpacked_input.getDataType() != kUInt16) {
    SIMPLE_LOG("ERROR", "dispatchU32toPackedU16: Unpacked input buffer type should be kUInt16");
    return;
  }
  if (packed_output.getDataType() != kUInt32) {
    SIMPLE_LOG("ERROR", "dispatchU32toPackedU16: Packed output buffer type is not kUInt32");
    return;
  }
  if (unpacked_input.getLength() == 0) {
    SIMPLE_LOG("WARN", "dispatchU32toPackedU16: Unpacked input buffer length is 0. Nothing to pack.");
    return;
  }

  size_t numLogicalElements = unpacked_input.getLength(); // Logical u16 elements
  size_t numPackedElements = (numLogicalElements + 1) / 2; // u32 elements needed for packing

  SIMPLE_LOG("INFO", "dispatchU32toPackedU16: Packing %zu logical elements into %zu packed elements.",
      numLogicalElements, numPackedElements);

  try {
    ComputeShader shader(mgpu);
    shader.loadKernelString(kUInt32ToPackedUInt16Kernel);
    shader.setBuffer(0, unpacked_input);
    shader.setBuffer(1, packed_output);
    shader.dispatch(calculateGroups(numPackedElements), 1, 1);
  } catch (const std::exception &e) {
    SIMPLE_LOG("ERROR", "dispatchU32toPackedU16: Exception during kernel dispatch: %s", e.what());
    throw;
  }
}

void dispatchExpandF64toU32x2(MGPU &mgpu, Buffer &input_f64, Buffer &output_u32x2) {
  if (input_f64.getDataType() != kFloat64) {
    SIMPLE_LOG("ERROR", "dispatchExpandF64toU32x2: Input buffer type is not kFloat64");
    return;
  }
  if (output_u32x2.getDataType() != kUInt32) {
    SIMPLE_LOG("ERROR", "dispatchExpandF64toU32x2: Output buffer type is not kUInt32");
    return;
  }
  if (input_f64.getLength() == 0) {
    SIMPLE_LOG("WARN", "dispatchExpandF64toU32x2: Input buffer length is 0. Nothing to expand.");
    return;
  }

  size_t numLogicalElements = input_f64.getLength(); // Logical f64 elements

  SIMPLE_LOG("INFO", "dispatchExpandF64toU32x2: Expanding %zu f64 elements to u32x2 pairs.", numLogicalElements);

  try {
    ComputeShader shader(mgpu);
    shader.loadKernelString(kExpandFloat64ToUInt32x2Kernel);
    shader.setBuffer(0, input_f64);
    shader.setBuffer(1, output_u32x2);
    // Dispatch based on the number of f64 elements
    shader.dispatch(calculateGroups(numLogicalElements), 1, 1);
  } catch (const std::exception &e) {
    SIMPLE_LOG("ERROR", "dispatchExpandF64toU32x2: Exception during kernel dispatch: %s", e.what());
    throw;
  }
}

void dispatchCombineU32x2toF64(MGPU &mgpu, Buffer &input_u32x2, Buffer &output_f64) {
  if (input_u32x2.getDataType() != kUInt32) {
    SIMPLE_LOG("ERROR", "dispatchCombineU32x2toF64: Input buffer type is not kUInt32");
    return;
  }
  if (output_f64.getDataType() != kFloat64) {
    SIMPLE_LOG("ERROR", "dispatchCombineU32x2toF64: Output buffer type is not kFloat64");
    return;
  }
  if (output_f64.getLength() == 0) {
    SIMPLE_LOG("WARN", "dispatchCombineU32x2toF64: Output buffer length is 0. Nothing to combine.");
    return;
  }

  size_t numLogicalElements = output_f64.getLength(); // Logical f64 elements

  SIMPLE_LOG("INFO", "dispatchCombineU32x2toF64: Combining u32x2 pairs to %zu f64 elements.", numLogicalElements);

  try {
    ComputeShader shader(mgpu);
    shader.loadKernelString(kCombineUInt32x2ToFloat64Kernel);
    shader.setBuffer(0, input_u32x2);
    shader.setBuffer(1, output_f64);
    // Dispatch based on the number of f64 elements to produce
    shader.dispatch(calculateGroups(numLogicalElements), 1, 1);
  } catch (const std::exception &e) {
    SIMPLE_LOG("ERROR", "dispatchCombineU32x2toF64: Exception during kernel dispatch: %s", e.what());
    throw;
  }
}

} // namespace kernels
} // namespace mgpu