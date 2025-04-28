#include "../include/compute_shader.h"
#include "../include/buffer.h" // Include buffer.h for Buffer definition
#include <sstream>
#include <stdexcept>
#include <vector> // Include vector for std::vector

using namespace gpu;

namespace mgpu {

ComputeShader::ComputeShader(MGPU &mgpu) : mgpu(mgpu) {}

void ComputeShader::loadKernelString(const std::string &kernelString) {
  // Assuming default workgroup size and type for now, might need adjustment
  code = KernelCode{kernelString, Shape{256, 1, 1}, ki32};
  LOG(kDefLog, kInfo, "Loaded kernel string.");
}

void ComputeShader::loadKernelFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(kDefLog, kError, "Failed to open kernel file: %s", path.c_str());
    throw std::runtime_error("Failed to open kernel file: " + path);
  }
  std::string kernelString((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
  loadKernelString(kernelString);
  LOG(kDefLog, kInfo, "Loaded kernel file: %s", path.c_str());
}

bool ComputeShader::hasKernel() const { return !code.data.empty(); }

void ComputeShader::setBuffer(int tag, const Buffer &buffer) {
  if (tag < 0) {
    LOG(kDefLog, kError, "setBuffer: Invalid negative tag %d.", tag);
    return;
  }
  if (buffer.bufferData.buffer == nullptr) {
    LOG(kDefLog, kError,
        "setBuffer: Attempted to set buffer with tag %d, but buffer is null.",
        tag);
    return;
  }

  if (tag >= static_cast<int>(bindings.size())) {
    bindings.resize(tag + 1); // Resize to accommodate the new tag
  }

  // --- Shape Calculation ---
  // The shape should represent the *logical* number of elements.
  // This information is now stored directly in buffer.length.
  size_t logicalNumElements = buffer.length;

  // Log buffer details for debugging
  LOG(kDefLog, kInfo,
      "setBuffer: Tag=%d, BufferType=%d, IsPacked=%d, LogicalLength=%zu, "
      "PhysicalSize=%zu bytes",
      tag, buffer.bufferType, buffer.isPacked, logicalNumElements,
      buffer.bufferData.size);

  // The Shape for the Tensor binding should reflect the logical element count.
  Shape shape{logicalNumElements};
  LOG(kDefLog, kInfo, "setBuffer: Tag=%d, Setting Tensor shape: [%zu]", tag,
      shape[0]);

  // Create the Tensor binding using the buffer's data and the logical shape.
  bindings[tag] = Tensor{.data = buffer.bufferData, .shape = shape};
}

void ComputeShader::dispatch(int groupsX, int groupsY, int groupsZ) {
  if (!hasKernel()) {
    LOG(kDefLog, kError, "dispatch: No kernel loaded.");
    return;
  }
  if (groupsX <= 0 || groupsY <= 0 || groupsZ <= 0) {
    LOG(kDefLog, kError,
        "dispatch: Invalid group dimensions (%d, %d, %d). Must be positive.",
        groupsX, groupsY, groupsZ);
    return;
  }

  // Create array of view offsets for tensor, size_t all 0
  // This assumes we always bind the entire buffer, starting at offset 0.
  std::vector<size_t> viewOffsets(bindings.size(), 0);

  LOG(kDefLog, kInfo,
      "Dispatching kernel with groups: (%d, %d, %d) and %zu bindings.", groupsX,
      groupsY, groupsZ, bindings.size());

  // Ensure all bindings are valid before creating the kernel
  for (size_t i = 0; i < bindings.size(); ++i) {
    if (bindings[i].data.buffer == nullptr) {
      // This check might be redundant if setBuffer prevents null buffers, but
      // good for safety.
      LOG(kDefLog, kError,
          "dispatch: Binding at tag %zu is null. Cannot dispatch.", i);
      return;
    }
  }

  Kernel kernel =
      createKernel(mgpu.getContext(), code, bindings.data(), bindings.size(),
                   viewOffsets.data(),
                   {static_cast<size_t>(groupsX), static_cast<size_t>(groupsY),
                    static_cast<size_t>(groupsZ)});

  // Check if kernel creation failed (createKernel might return a null/invalid
  // kernel) Assuming createKernel returns a struct where some member indicates
  // validity, e.g., pipeline != nullptr if (kernel.pipeline == nullptr) { //
  // Adjust based on actual Kernel struct definition
  //     LOG(kDefLog, kError, "dispatch: Failed to create kernel object.");
  //     return;
  // }

  dispatchKernel(mgpu.getContext(), kernel);
  LOG(kDefLog, kInfo, "Kernel dispatch complete.");
}

void ComputeShader::dispatchAsync(int groupsX, int groupsY, int groupsZ,
                                  std::function<void()> callback) {
  LOG(kDefLog, kInfo,
      "dispatchAsync called. Dispatching synchronously then calling callback.");
  try {
    dispatch(groupsX, groupsY, groupsZ);
    if (callback) {
      LOG(kDefLog, kInfo, "dispatchAsync: Invoking callback.");
      callback();
    }
  } catch (const std::exception &e) {
    LOG(kDefLog, kError, "dispatchAsync: Exception during dispatch: %s",
        e.what());
  } catch (...) {
    LOG(kDefLog, kError, "dispatchAsync: Unknown exception during dispatch.");
  }
}
} // namespace mgpu
