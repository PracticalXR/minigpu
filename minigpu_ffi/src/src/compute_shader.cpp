#include "../include/compute_shader.h"
#include "../include/buffer.h" // Include buffer.h for Buffer definition
#include <sstream>
#include <stdexcept>
#include <vector> // Include vector for std::vector

using namespace gpu;

namespace mgpu {

ComputeShader::ComputeShader(MGPU &mgpu) : mgpu(mgpu) {}

ComputeShader::~ComputeShader() {
  LOG(kDefLog, kInfo, "ComputeShader destructor: Cleaning up cached kernel.");
  destroyCachedKernel();
}

size_t ComputeShader::calculateBindingsHash() const {
  size_t hash = bindings.size();
  for (const auto &binding : bindings) {
    // Simple hash based on buffer pointer and shape
    hash ^= reinterpret_cast<size_t>(binding.data.buffer) + 0x9e3779b9 +
            (hash << 6) + (hash >> 2);
    if (!binding.shape.data.empty()) {
      hash ^= binding.shape[0] + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
  }
  return hash;
}

bool ComputeShader::needsKernelRecreation(int groupsX, int groupsY,
                                          int groupsZ) const {
  if (!cachedKernel.has_value()) {
    return true; // No cached kernel
  }

  if (lastGroupSize[0] != static_cast<size_t>(groupsX) ||
      lastGroupSize[1] != static_cast<size_t>(groupsY) ||
      lastGroupSize[2] != static_cast<size_t>(groupsZ)) {
    return true; // Group size changed
  }

  size_t currentHash = calculateBindingsHash();
  if (lastBindingsHash != currentHash) {
    return true; // Bindings changed
  }

  if (lastShaderCode != code.data) {
    return true; // Shader code changed
  }

  return false; // Can reuse cached kernel
}

void ComputeShader::destroyCachedKernel() {
  if (cachedKernel.has_value()) {
    LOG(kDefLog, kInfo,
        "Destroying cached kernel and releasing WebGPU resources.");

    // Get reference to kernel before destroying it
    auto kernel = cachedKernel.value();

    if (kernel->bindGroup) {
      wgpuBindGroupRelease(kernel->bindGroup);
      kernel->bindGroup = nullptr;
    }

    // Now reset the cached kernel
    cachedKernel.reset();
    lastBindingsHash = 0;

    LOG(kDefLog, kInfo,
        "Cached kernel destroyed and WebGPU resources released.");
  }
}

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

  // Check if buffer is already bound to avoid unnecessary work
  if (bindings[tag].data.buffer == buffer.bufferData.buffer &&
      bindings[tag].shape.data.size() > 0 &&
      bindings[tag].shape[0] == buffer.length) {
    LOG(kDefLog, kInfo,
        "setBuffer: Tag=%d already bound to same buffer, skipping.", tag);
    return;
  }

  // --- Shape Calculation ---
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

  // Invalidate cached kernel when bindings change
  size_t newHash = calculateBindingsHash();
  if (newHash != lastBindingsHash) {
    LOG(kDefLog, kInfo, "Buffer binding changed, invalidating cached kernel.");
    destroyCachedKernel();
  }
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

  // Ensure all bindings are valid
  for (size_t i = 0; i < bindings.size(); ++i) {
    if (bindings[i].data.buffer == nullptr) {
      LOG(kDefLog, kError,
          "dispatch: Binding at tag %zu is null. Cannot dispatch.", i);
      return;
    }
  }

  // Only create kernel if needed
  if (needsKernelRecreation(groupsX, groupsY, groupsZ)) {
    LOG(kDefLog, kInfo,
        "Creating new kernel for dispatch (%d, %d, %d) with %zu bindings.",
        groupsX, groupsY, groupsZ, bindings.size());

    destroyCachedKernel(); // Clean up old kernel first

    std::vector<size_t> viewOffsets(bindings.size(), 0);

    cachedKernel = createKernel(mgpu.getContext(), code, bindings.data(),
                                bindings.size(), viewOffsets.data(),
                                {static_cast<size_t>(groupsX),
                                 static_cast<size_t>(groupsY),
                                 static_cast<size_t>(groupsZ)});

    // Update cache state
    lastGroupSize = {static_cast<size_t>(groupsX), static_cast<size_t>(groupsY),
                     static_cast<size_t>(groupsZ)};
    lastBindingsHash = calculateBindingsHash();
    lastShaderCode = code.data;

    LOG(kDefLog, kInfo, "Cached new kernel.");
  } else {
    LOG(kDefLog, kInfo, "Reusing cached kernel for dispatch.");
  }

  // Dispatch with cached kernel
  dispatchKernel(mgpu.getContext(), cachedKernel.value());
  LOG(kDefLog, kInfo, "Kernel dispatch complete (cached).");
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
