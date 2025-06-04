#include "../include/conversion_kernels.h"
#include "../include/buffer.h" // Full definition of mgpu::Buffer
#include "../include/compute_shader.h" // Full definition of mgpu::MGPU and ComputeShader
#include "../include/gpuh.h" // Full definition of gpu::LOG and gpu::KernelCode
#include <cstdint>
#include <stdexcept> // For runtime_error

namespace mgpu {
namespace kernels {

// Helper to calculate workgroup count for a 1D dispatch
static int calculateGroups(size_t numElements, size_t workgroupSize = 256) {
  if (workgroupSize == 0) {
    gpu::LOG(gpu::kDefLog, gpu::kError,
             "calculateGroups: Workgroup size cannot be zero.");
    return 0; // Avoid division by zero
  }
  return static_cast<int>((numElements + workgroupSize - 1) / workgroupSize);
}

// Enhanced kernel dispatch safety validation - specific to kernel operations
static bool validateKernelDispatchSafety(const Buffer &sourceBuffer,
                                         const Buffer &destBuffer,
                                         size_t expectedSourceElements,
                                         size_t expectedDestElements,
                                         const std::string &kernelName) {

  // Validate source buffer
  if (sourceBuffer.bufferData.buffer == nullptr) {
    LOG(gpu::kDefLog, gpu::kError,
        "validateKernelDispatchSafety(%s): Source buffer is null",
        kernelName.c_str());
    return false;
  }

  if (sourceBuffer.length < expectedSourceElements) {
    LOG(gpu::kDefLog, gpu::kError,
        "validateKernelDispatchSafety(%s): Source buffer length (%zu) < "
        "expected elements (%zu)",
        kernelName.c_str(), sourceBuffer.length, expectedSourceElements);
    return false;
  }

  // Validate destination buffer
  if (destBuffer.bufferData.buffer == nullptr) {
    LOG(gpu::kDefLog, gpu::kError,
        "validateKernelDispatchSafety(%s): Destination buffer is null",
        kernelName.c_str());
    return false;
  }

  if (destBuffer.length < expectedDestElements) {
    LOG(gpu::kDefLog, gpu::kError,
        "validateKernelDispatchSafety(%s): Destination buffer length (%zu) < "
        "expected elements (%zu)",
        kernelName.c_str(), destBuffer.length, expectedDestElements);
    return false;
  }

  // Calculate expected physical sizes based on kernel type
  size_t sourceExpectedPhysicalSize = 0;
  size_t destExpectedPhysicalSize = 0;

  if (sourceBuffer.bufferType != gpu::kUnknown) {
    size_t sourceElementSize = gpu::sizeBytes(sourceBuffer.bufferType);
    if (kernelName.find("I8") != std::string::npos) {
      // For I8 packing: 4 i8 elements per i32
      sourceExpectedPhysicalSize =
          ((expectedSourceElements + 3) / 4) * sourceElementSize;
    } else if (kernelName.find("I16") != std::string::npos) {
      // For I16 packing: 2 i16 elements per i32
      sourceExpectedPhysicalSize =
          ((expectedSourceElements + 1) / 2) * sourceElementSize;
    } else {
      sourceExpectedPhysicalSize = expectedSourceElements * sourceElementSize;
    }

    if (sourceBuffer.bufferData.size < sourceExpectedPhysicalSize) {
      LOG(gpu::kDefLog, gpu::kError,
          "validateKernelDispatchSafety(%s): Source buffer physical size (%zu) "
          "< expected (%zu)",
          kernelName.c_str(), sourceBuffer.bufferData.size,
          sourceExpectedPhysicalSize);
      return false;
    }
  }

  if (destBuffer.bufferType != gpu::kUnknown) {
    size_t destElementSize = gpu::sizeBytes(destBuffer.bufferType);
    destExpectedPhysicalSize = expectedDestElements * destElementSize;

    if (destBuffer.bufferData.size < destExpectedPhysicalSize) {
      LOG(gpu::kDefLog, gpu::kError,
          "validateKernelDispatchSafety(%s): Destination buffer physical size "
          "(%zu) < expected (%zu)",
          kernelName.c_str(), destBuffer.bufferData.size,
          destExpectedPhysicalSize);
      return false;
    }
  }

  LOG(gpu::kDefLog, gpu::kInfo,
      "validateKernelDispatchSafety(%s): PASSED - Source: %zu logical, %zu "
      "physical bytes. Dest: %zu logical, %zu physical bytes",
      kernelName.c_str(), sourceBuffer.length, sourceBuffer.bufferData.size,
      destBuffer.length, destBuffer.bufferData.size);

  return true;
}

// Enhanced safety validation for workgroup dispatch limits
static bool validateWorkgroupDispatchSafety(size_t numElements,
                                            size_t workgroupSize,
                                            const std::string &kernelName) {
  if (workgroupSize == 0) {
    LOG(gpu::kDefLog, gpu::kError,
        "validateWorkgroupDispatchSafety(%s): Workgroup size cannot be zero",
        kernelName.c_str());
    return false;
  }

  size_t numWorkgroups = (numElements + workgroupSize - 1) / workgroupSize;

  // WebGPU limits
  const size_t MAX_WORKGROUPS_X = 65535;
  const size_t MAX_TOTAL_INVOCATIONS = 256 * 65535; // Conservative estimate

  if (numWorkgroups > MAX_WORKGROUPS_X) {
    LOG(gpu::kDefLog, gpu::kError,
        "validateWorkgroupDispatchSafety(%s): Workgroup count (%zu) exceeds "
        "X-dimension limit (%zu)",
        kernelName.c_str(), numWorkgroups, MAX_WORKGROUPS_X);
    return false;
  }

  size_t totalInvocations = numWorkgroups * workgroupSize;
  if (totalInvocations > MAX_TOTAL_INVOCATIONS) {
    LOG(gpu::kDefLog, gpu::kError,
        "validateWorkgroupDispatchSafety(%s): Total invocations (%zu) exceeds "
        "limit (%zu)",
        kernelName.c_str(), totalInvocations, MAX_TOTAL_INVOCATIONS);
    return false;
  }

  LOG(gpu::kDefLog, gpu::kInfo,
      "validateWorkgroupDispatchSafety(%s): PASSED - %zu elements, %zu "
      "workgroups, %zu invocations",
      kernelName.c_str(), numElements, numWorkgroups, totalInvocations);

  return true;
}

// Update dispatchPackedI8toI32 with enhanced safety
void dispatchPackedI8toI32(MGPU &mgpu, Buffer &packed_input,
                           Buffer &unpacked_output) {
  // --- State Validation ---
  if (packed_input.bufferType != gpu::ki32) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedI8toI32: Packed input buffer type is not ki32 (is %d).",
        packed_input.bufferType);
    return;
  }
  if (unpacked_output.bufferType != gpu::ki32) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedI8toI32: Unpacked output buffer type is not ki32 (is "
        "%d).",
        unpacked_output.bufferType);
    return;
  }
  if (!unpacked_output.isPacked) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedI8toI32: Unpacked output buffer should have "
        "isPacked=true.");
    return;
  }
  if (unpacked_output.length == 0) {
    LOG(gpu::kDefLog, gpu::kWarn,
        "dispatchPackedI8toI32: Unpacked output buffer logical length is 0. "
        "Nothing to unpack.");
    return;
  }

  // --- Dispatch Calculation ---
  size_t numLogicalElements = unpacked_output.length;
  size_t numPackedElements = (numLogicalElements + 3) / 4;

  // --- CRITICAL: Enhanced Safety Validation ---
  if (!validateKernelDispatchSafety(packed_input, unpacked_output,
                                    numPackedElements, numLogicalElements,
                                    "dispatchPackedI8toI32")) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedI8toI32: Kernel dispatch safety validation FAILED. "
        "Aborting to prevent crash.");
    return;
  }

  if (!validateWorkgroupDispatchSafety(numPackedElements, 256,
                                       "dispatchPackedI8toI32")) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedI8toI32: Workgroup dispatch safety validation FAILED. "
        "Aborting to prevent crash.");
    return;
  }

  LOG(gpu::kDefLog, gpu::kInfo,
      "dispatchPackedI8toI32: SAFETY PASSED - Unpacking %zu logical elements. "
      "Dispatching based on %zu packed elements.",
      numLogicalElements, numPackedElements);

  try {
    ComputeShader shader(mgpu);
    shader.loadKernelString(kPackedInt8ToInt32Kernel);
    shader.setBuffer(0, packed_input);
    shader.setBuffer(1, unpacked_output);

    int groupsX = calculateGroups(numPackedElements);

    LOG(gpu::kDefLog, gpu::kInfo,
        "dispatchPackedI8toI32: Dispatching %d workgroups for %zu packed "
        "elements",
        groupsX, numPackedElements);
    shader.dispatch(groupsX, 1, 1);

    LOG(gpu::kDefLog, gpu::kInfo,
        "dispatchPackedI8toI32: Kernel dispatch completed successfully");

  } catch (const std::exception &e) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedI8toI32: Exception during kernel dispatch: %s",
        e.what());
    throw; // Re-throw to let caller handle
  }
}

// Packs data from 'unpacked_input' (ki32, isPacked=true) into 'packed_output'
// (ki32) Kernel iterates over PACKED elements.
void dispatchI32toPackedI8(MGPU &mgpu, Buffer &unpacked_input,
                           Buffer &packed_output) {
  // --- State Validation ---
  // The input buffer MUST be marked as packed=true because it represents
  // original i8 data.
  if (unpacked_input.bufferType != gpu::ki32 || !unpacked_input.isPacked) {
    gpu::LOG(
        gpu::kDefLog, gpu::kError,
        "dispatchI32toPackedI8: Unpacked input buffer state invalid. Expected "
        "type=ki32 and isPacked=true, but got type=%d, isPacked=%d.",
        unpacked_input.bufferType, unpacked_input.isPacked);
    return;
  }
  if (packed_output.bufferType != gpu::ki32) {
    gpu::LOG(
        gpu::kDefLog, gpu::kError,
        "dispatchI32toPackedI8: Packed output buffer type is not ki32 (is %d).",
        packed_output.bufferType);
    return;
  }
  if (unpacked_input.length == 0) {
    gpu::LOG(gpu::kDefLog, gpu::kWarn,
             "dispatchI32toPackedI8: Unpacked input buffer logical length is "
             "0. Nothing to pack.");
    return;
  }

  // --- Dispatch Calculation ---
  // The kernel iterates based on the *packed* output index.
  // The number of packed elements is determined by the *logical length* of the
  // unpacked input.
  size_t numLogicalElements =
      unpacked_input.length; // The input buffer holds the logical length
  size_t numPackedElements =
      (numLogicalElements + 3) /
      4; // Number of i32 elements in the packed representation

  // --- Physical Size Validation ---
  size_t requiredUnpackedBytes = numLogicalElements * sizeof(int32_t);
  if (unpacked_input.bufferData.size < requiredUnpackedBytes) {
    gpu::LOG(
        gpu::kDefLog, gpu::kError,
        "dispatchI32toPackedI8: Unpacked input buffer physical size (%zu) is "
        "too small for required bytes (%zu) based on its logical length (%zu).",
        unpacked_input.bufferData.size, requiredUnpackedBytes,
        numLogicalElements);
    return;
  }
  size_t requiredPackedBytes = numPackedElements * sizeof(int32_t);
  if (packed_output.bufferData.size < requiredPackedBytes) {
    gpu::LOG(gpu::kDefLog, gpu::kError,
             "dispatchI32toPackedI8: Packed output buffer physical size (%zu) "
             "is too small for required bytes (%zu) based on input logical "
             "length (%zu).",
             packed_output.bufferData.size, requiredPackedBytes,
             numLogicalElements);
    return;
  }

  gpu::LOG(gpu::kDefLog, gpu::kInfo,
           "dispatchI32toPackedI8: Packing %zu logical elements. Dispatching "
           "based on %zu packed elements.",
           numLogicalElements, numPackedElements);

  ComputeShader shader(mgpu);
  shader.loadKernelString(kInt32ToPackedInt8Kernel);
  shader.setBuffer(
      0, unpacked_input); // Binding 0: Source (unpacked i32, isPacked=true)
  shader.setBuffer(1, packed_output); // Binding 1: Destination (packed i32)

  // Dispatch based on the number of PACKED elements the kernel needs to
  // process.
  int groupsX = calculateGroups(numPackedElements);
  shader.dispatch(groupsX, 1, 1);
}

// Apply similar enhanced safety to other dispatch functions...
void dispatchPackedU8toU32(MGPU &mgpu, Buffer &packed_input,
                           Buffer &unpacked_output) {
  // Similar validation pattern...
  if (packed_input.bufferType != gpu::ku32) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedU8toU32: Packed input buffer type is not ku32 (is %d).",
        packed_input.bufferType);
    return;
  }
  if (unpacked_output.bufferType != gpu::ku32 || !unpacked_output.isPacked) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedU8toU32: Unpacked output buffer should have "
        "isPacked=true.");
    return;
  }
  if (unpacked_output.length == 0) {
    LOG(gpu::kDefLog, gpu::kWarn,
        "dispatchPackedU8toU32: Unpacked output buffer logical length is 0. "
        "Nothing to unpack.");
    return;
  }

  size_t numLogicalElements = unpacked_output.length;
  size_t numPackedElements = (numLogicalElements + 3) / 4;

  // Enhanced safety validation
  if (!validateKernelDispatchSafety(packed_input, unpacked_output,
                                    numPackedElements, numLogicalElements,
                                    "dispatchPackedU8toU32")) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedU8toU32: Kernel dispatch safety validation FAILED. "
        "Aborting.");
    return;
  }

  if (!validateWorkgroupDispatchSafety(numPackedElements, 256,
                                       "dispatchPackedU8toU32")) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedU8toU32: Workgroup dispatch safety validation FAILED. "
        "Aborting.");
    return;
  }

  try {
    ComputeShader shader(mgpu);
    shader.loadKernelString(kPackedUInt8ToUInt32Kernel);
    shader.setBuffer(0, packed_input);
    shader.setBuffer(1, unpacked_output);
    shader.dispatch(calculateGroups(numPackedElements), 1, 1);

    LOG(gpu::kDefLog, gpu::kInfo,
        "dispatchPackedU8toU32: Kernel dispatch completed successfully");
  } catch (const std::exception &e) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedU8toU32: Exception during kernel dispatch: %s",
        e.what());
    throw;
  }
}

// Packs data from 'unpacked_input' (ku32, isPacked=true) into 'packed_output'
// (ku32)
void dispatchU32toPackedU8(MGPU &mgpu, Buffer &unpacked_input,
                           Buffer &packed_output) {
  // Validation: Input is unpacked u32 representing u8, Output is packed u32
  if (unpacked_input.bufferType != gpu::ku32 || !unpacked_input.isPacked) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchU32toPackedU8: Unpacked input buffer state invalid. "
        "Expected type=ku32 and isPacked=true, but got type=%d, isPacked=%d.",
        unpacked_input.bufferType, unpacked_input.isPacked);
    return;
  }
  if (packed_output.bufferType != gpu::ku32) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchU32toPackedU8: Packed output buffer type is not ku32 (is "
        "%d).",
        packed_output.bufferType);
    return;
  }
  if (unpacked_input.length == 0) {
    LOG(gpu::kDefLog, gpu::kWarn,
        "dispatchU32toPackedU8: Unpacked input buffer logical length is "
        "0. Nothing to pack.");
    return;
  }

  size_t numLogicalElements = unpacked_input.length;
  size_t numPackedElements = (numLogicalElements + 3) / 4;

  // Physical Size Validation (similar to i8 version)
  if (unpacked_input.bufferData.size < numLogicalElements * sizeof(uint32_t)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchU32toPackedU8: Unpacked input buffer physical size (%zu) "
        "is too small for required bytes (%zu) based on its logical length "
        "(%zu).",
        unpacked_input.bufferData.size, numLogicalElements * sizeof(uint32_t),
        numLogicalElements);
    return;
  }
  if (packed_output.bufferData.size < numPackedElements * sizeof(uint32_t)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchU32toPackedU8: Packed output buffer physical size (%zu) "
        "is too small for required bytes (%zu) based on input logical "
        "length (%zu).",
        packed_output.bufferData.size, numPackedElements * sizeof(uint32_t),
        numLogicalElements);
    return;
  }

  gpu::LOG(gpu::kDefLog, gpu::kInfo,
           "dispatchU32toPackedU8: Packing %zu logical elements. Dispatching "
           "based on %zu packed elements.",
           numLogicalElements, numPackedElements);

  ComputeShader shader(mgpu);
  shader.loadKernelString(kUInt32ToPackedUInt8Kernel);
  shader.setBuffer(0, unpacked_input);
  shader.setBuffer(1, packed_output);
  shader.dispatch(calculateGroups(numPackedElements), 1, 1);
}

// --- 16-bit Dispatch ---

// Unpacks data from 'packed_input' (ki32 representing packed ki16) into
// 'unpacked_output' (ki32)
void dispatchPackedI16toI32(MGPU &mgpu, Buffer &packed_input,
                            Buffer &unpacked_output) {
  // Validation: Input is packed i32, Output is unpacked i32 representing i16
  if (packed_input.bufferType != gpu::ki32) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedI16toI32: Packed input buffer type is not ki32 (is "
        "%d).",
        packed_input.bufferType);
    return;
  }
  if (unpacked_output.bufferType != gpu::ki32 || !unpacked_output.isPacked) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedI16toI32: Unpacked output buffer should have "
        "isPacked=true (indicates it represents original i16 data).");
    return;
  }
  if (unpacked_output.length == 0) {
    LOG(gpu::kDefLog, gpu::kWarn,
        "dispatchPackedI16toI32: Unpacked output buffer logical length is "
        "0. Nothing to unpack.");
    return;
  }

  size_t numLogicalElements = unpacked_output.length; // Logical i16 elements
  size_t numPackedElements =
      (numLogicalElements + 1) / 2; // i32 elements needed for packing

  // Physical Size Validation
  if (packed_input.bufferData.size < numPackedElements * sizeof(int32_t)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedI16toI32: Packed input buffer physical size (%zu) is "
        "too small for required bytes (%zu) based on output logical length "
        "(%zu).",
        packed_input.bufferData.size, numPackedElements * sizeof(int32_t),
        numLogicalElements);
    return;
  }
  if (unpacked_output.bufferData.size < numLogicalElements * sizeof(int32_t)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedI16toI32: Unpacked output buffer physical size (%zu) "
        "is too small for required bytes (%zu) based on its logical length "
        "(%zu).",
        unpacked_output.bufferData.size, numLogicalElements * sizeof(int32_t),
        numLogicalElements);
    return;
  } // Output stores unpacked i32

  gpu::LOG(gpu::kDefLog, gpu::kInfo,
           "dispatchPackedI16toI32: Unpacking %zu logical elements. "
           "Dispatching based on %zu packed elements.",
           numLogicalElements, numPackedElements);

  ComputeShader shader(mgpu);
  shader.loadKernelString(kPackedInt16ToInt32Kernel);
  shader.setBuffer(0, packed_input);
  shader.setBuffer(1, unpacked_output);
  shader.dispatch(calculateGroups(numPackedElements), 1, 1);
}

// Packs data from 'unpacked_input' (ki32, isPacked=true) into 'packed_output'
// (ki32)
void dispatchI32toPackedI16(MGPU &mgpu, Buffer &unpacked_input,
                            Buffer &packed_output) {
  // Validation: Input is unpacked i32 representing i16, Output is packed i32
  if (unpacked_input.bufferType != gpu::ki32 || !unpacked_input.isPacked) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchI32toPackedI16: Unpacked input buffer state invalid. Expected "
        "type=ki32 and isPacked=true, but got type=%d, isPacked=%d.",
        unpacked_input.bufferType, unpacked_input.isPacked);
    return;
  }
  if (packed_output.bufferType != gpu::ki32) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchI32toPackedI16: Packed output buffer type is not ki32 (is "
        "%d).",
        packed_output.bufferType);
    return;
  }
  if (unpacked_input.length == 0) {
    LOG(gpu::kDefLog, gpu::kWarn,
        "dispatchI32toPackedI16: Unpacked input buffer logical length is "
        "0. Nothing to pack.");
    return;
  }

  size_t numLogicalElements = unpacked_input.length; // Logical i16 elements
  size_t numPackedElements =
      (numLogicalElements + 1) / 2; // i32 elements needed for packing

  // Physical Size Validation
  if (unpacked_input.bufferData.size < numLogicalElements * sizeof(int32_t)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchI32toPackedI16: Unpacked input buffer physical size (%zu) is "
        "too small for required bytes (%zu) based on its logical length (%zu).",
        unpacked_input.bufferData.size, numLogicalElements * sizeof(int32_t),
        numLogicalElements);
    return;
  }
  if (packed_output.bufferData.size < numPackedElements * sizeof(int32_t)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchI32toPackedI16: Packed output buffer physical size (%zu) is "
        "too small for required bytes (%zu) based on input logical length "
        "(%zu).",
        packed_output.bufferData.size, numPackedElements * sizeof(int32_t),
        numLogicalElements);
    return;
  }

  gpu::LOG(gpu::kDefLog, gpu::kInfo,
           "dispatchI32toPackedI16: Packing %zu logical elements. Dispatching "
           "based on %zu packed elements.",
           numLogicalElements, numPackedElements);

  ComputeShader shader(mgpu);
  shader.loadKernelString(kInt32ToPackedInt16Kernel);
  shader.setBuffer(0, unpacked_input);
  shader.setBuffer(1, packed_output);
  shader.dispatch(calculateGroups(numPackedElements), 1, 1);
}

// Unpacks data from 'packed_input' (ku32 representing packed ku16) into
// 'unpacked_output' (ku32)
void dispatchPackedU16toU32(MGPU &mgpu, Buffer &packed_input,
                            Buffer &unpacked_output) {
  // Validation: Input is packed u32, Output is unpacked u32 representing u16
  if (packed_input.bufferType != gpu::ku32) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedU16toU32: Packed input buffer type is not ku32 (is %d).",
        packed_input.bufferType);
    return;
  }
  if (unpacked_output.bufferType != gpu::ku32 || !unpacked_output.isPacked) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedU16toU32: Unpacked output buffer should have "
        "isPacked=true (indicates it represents original u16 data).");
    return;
  }
  if (unpacked_output.length == 0) {
    LOG(gpu::kDefLog, gpu::kWarn,
        "dispatchPackedU16toU32: Unpacked output buffer logical length is "
        "0. Nothing to unpack.");
    return;
  }

  size_t numLogicalElements = unpacked_output.length; // Logical u16 elements
  size_t numPackedElements =
      (numLogicalElements + 1) / 2; // u32 elements needed for packing

  // Physical Size Validation
  if (packed_input.bufferData.size < numPackedElements * sizeof(uint32_t)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedU16toU32: Packed input buffer physical size (%zu) is "
        "too small for required bytes (%zu) based on output logical length "
        "(%zu).",
        packed_input.bufferData.size, numPackedElements * sizeof(uint32_t),
        numLogicalElements);
    return;
  }
  if (unpacked_output.bufferData.size < numLogicalElements * sizeof(uint32_t)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchPackedU16toU32: Unpacked output buffer physical size (%zu) is "
        "too small for required bytes (%zu) based on its logical length (%zu).",
        unpacked_output.bufferData.size, numLogicalElements * sizeof(uint32_t),
        numLogicalElements);
    return;
  }

  gpu::LOG(gpu::kDefLog, gpu::kInfo,
           "dispatchPackedU16toU32: Unpacking %zu logical elements. "
           "Dispatching based on %zu packed elements.",
           numLogicalElements, numPackedElements);

  ComputeShader shader(mgpu);
  shader.loadKernelString(kPackedUInt16ToUInt32Kernel);
  shader.setBuffer(0, packed_input);
  shader.setBuffer(1, unpacked_output);
  shader.dispatch(calculateGroups(numPackedElements), 1, 1);
}

// Packs data from 'unpacked_input' (ku32, isPacked=true) into 'packed_output'
// (ku32)
void dispatchU32toPackedU16(MGPU &mgpu, Buffer &unpacked_input,
                            Buffer &packed_output) {
  // Validation: Input is unpacked u32 representing u16, Output is packed u32
  if (unpacked_input.bufferType != gpu::ku32 || !unpacked_input.isPacked) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchU32toPackedU16: Unpacked input buffer state invalid. Expected "
        "type=ku32 and isPacked=true, but got type=%d, isPacked=%d.",
        unpacked_input.bufferType, unpacked_input.isPacked);
    return;
  }
  if (packed_output.bufferType != gpu::ku32) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchU32toPackedU16: Packed output buffer type is not ku32 (is "
        "%d).",
        packed_output.bufferType);
    return;
  }
  if (unpacked_input.length == 0) {
    LOG(gpu::kDefLog, gpu::kWarn,
        "dispatchU32toPackedU16: Unpacked input buffer logical length is "
        "0. Nothing to pack.");
    return;
  }

  size_t numLogicalElements = unpacked_input.length; // Logical u16 elements
  size_t numPackedElements =
      (numLogicalElements + 1) / 2; // u32 elements needed for packing

  // Physical Size Validation
  if (unpacked_input.bufferData.size < numLogicalElements * sizeof(uint32_t)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchU32toPackedU16: Unpacked input buffer physical size (%zu) is "
        "too small for required bytes (%zu) based on its logical length (%zu).",
        unpacked_input.bufferData.size, numLogicalElements * sizeof(uint32_t),
        numLogicalElements);
    return;
  }
  if (packed_output.bufferData.size < numPackedElements * sizeof(uint32_t)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchU32toPackedU16: Packed output buffer physical size (%zu) is "
        "too small for required bytes (%zu) based on input logical length "
        "(%zu).",
        packed_output.bufferData.size, numPackedElements * sizeof(uint32_t),
        numLogicalElements);
    return;
  }

  gpu::LOG(gpu::kDefLog, gpu::kInfo,
           "dispatchU32toPackedU16: Packing %zu logical elements. Dispatching "
           "based on %zu packed elements.",
           numLogicalElements, numPackedElements);

  ComputeShader shader(mgpu);
  shader.loadKernelString(kUInt32ToPackedUInt16Kernel);
  shader.setBuffer(0, unpacked_input);
  shader.setBuffer(1, packed_output);
  shader.dispatch(calculateGroups(numPackedElements), 1, 1);
}

// --- 64-bit Dispatch (f64 example) ---

// Expands f64 data into u32 pairs
void dispatchExpandF64toU32x2(MGPU &mgpu, Buffer &input_f64,
                              Buffer &output_u32x2) {
  // Validation: Input is f64, Output is u32 pairs (isPacked might indicate this
  // expansion) Note: Buffer type system needs extension or careful use of
  // isPacked/length input_f64.bufferType == kf64 and
  // output_u32x2.bufferType == ku32 and output_u32x2.isPacked == true (to
  // signify it holds expanded data) and output_u32x2.length == input_f64.length
  // * 2

  if (input_f64.bufferType != gpu::kf64) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchExpandF64toU32x2: Input buffer type is not kf64 (is %d).",
        input_f64.bufferType);
    return;
  }
  if (output_u32x2.bufferType != gpu::ku32 || !output_u32x2.isPacked) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchExpandF64toU32x2: Output buffer should have isPacked=true "
        "(indicates it represents expanded u32 pairs).");
    return;
  } // Add isPacked check if used
  if (input_f64.length == 0) {
    LOG(gpu::kDefLog, gpu::kWarn,
        "dispatchExpandF64toU32x2: Input buffer logical length is 0. Nothing "
        "to "
        "expand.");
    return;
  }

  size_t numLogicalElements = input_f64.length; // Logical f64 elements

  // Physical Size Validation
  if (input_f64.bufferData.size < numLogicalElements * sizeof(double)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchExpandF64toU32x2: Input f64 buffer physical size (%zu) is too "
        "small for required bytes (%zu) based on its logical length (%zu).",
        input_f64.bufferData.size, numLogicalElements * sizeof(double),
        numLogicalElements);
    return;
  }
  // Output buffer holds vec2<u32> effectively, so size is numLogicalElements *
  // sizeof(vec2<u32>) = numLogicalElements * 2 * sizeof(u32)
  if (output_u32x2.bufferData.size <
      numLogicalElements * 2 * sizeof(uint32_t)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchExpandF64toU32x2: Output u32x2 buffer physical size (%zu) is "
        "too small for required bytes (%zu) based on input logical length "
        "(%zu).",
        output_u32x2.bufferData.size, numLogicalElements * 2 * sizeof(uint32_t),
        numLogicalElements);
    return;
  }

  gpu::LOG(gpu::kDefLog, gpu::kInfo,
           "dispatchExpandF64toU32x2: Expanding %zu f64 elements. Dispatching "
           "based on logical elements.",
           numLogicalElements);

  ComputeShader shader(mgpu);
  shader.loadKernelString(kExpandFloat64ToUInt32x2Kernel);
  shader.setBuffer(0, input_f64);
  shader.setBuffer(1, output_u32x2);
  // Dispatch based on the number of f64 elements
  shader.dispatch(calculateGroups(numLogicalElements), 1, 1);
}

// Combines u32 pairs into f64 data
void dispatchCombineU32x2toF64(MGPU &mgpu, Buffer &input_u32x2,
                               Buffer &output_f64) {
  // Validation: Input is u32 pairs, Output is f64
  // Assuming input_u32x2.bufferType == ku32 and input_u32x2.isPacked == true
  // and output_f64.bufferType == kf64
  // and output_f64.length == input_u32x2.length / 2

  if (input_u32x2.bufferType != gpu::ku32 || !input_u32x2.isPacked) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchCombineU32x2toF64: Input buffer should have isPacked=true "
        "(indicates it represents original u32 pairs).");
    return;
  }
  if (output_f64.bufferType != gpu::kf64) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchCombineU32x2toF64: Output buffer type is not kf64 (is %d).",
        output_f64.bufferType);
    return;
  }
  if (output_f64.length == 0) {
    LOG(gpu::kDefLog, gpu::kWarn,
        "dispatchCombineU32x2toF64: Output buffer logical length is 0. Nothing "
        "to combine.");
    return;
  } // Check output length as it dictates dispatch

  size_t numLogicalElements = output_f64.length; // Logical f64 elements

  // Physical Size Validation
  if (input_u32x2.bufferData.size < numLogicalElements * 2 * sizeof(uint32_t)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchCombineU32x2toF64: Input u32x2 buffer physical size (%zu) is "
        "too small for required bytes (%zu) based on output logical length "
        "(%zu).",
        input_u32x2.bufferData.size, numLogicalElements * 2 * sizeof(uint32_t),
        numLogicalElements);
    return;
  }
  if (output_f64.bufferData.size < numLogicalElements * sizeof(double)) {
    LOG(gpu::kDefLog, gpu::kError,
        "dispatchCombineU32x2toF64: Output f64 buffer physical size (%zu) is "
        "too small for required bytes (%zu) based on its logical length (%zu).",
        output_f64.bufferData.size, numLogicalElements * sizeof(double),
        numLogicalElements);
    return;
  }

  gpu::LOG(gpu::kDefLog, gpu::kInfo,
           "dispatchCombineU32x2toF64: Combining into %zu f64 elements. "
           "Dispatching based on logical elements.",
           numLogicalElements);

  ComputeShader shader(mgpu);
  shader.loadKernelString(kCombineUInt32x2ToFloat64Kernel);
  shader.setBuffer(0, input_u32x2);
  shader.setBuffer(1, output_f64);
  // Dispatch based on the number of f64 elements to produce
  shader.dispatch(calculateGroups(numLogicalElements), 1, 1);
}

} // namespace kernels
} // namespace mgpu
