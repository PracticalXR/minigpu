#include "../include/compute_shader.h"
#include <sstream>
#include <stdexcept>

using namespace gpu;

namespace mgpu {

ComputeShader::ComputeShader(MGPU &mgpu) : mgpu(mgpu) {}

void ComputeShader::loadKernelString(const std::string &kernelString) {
  code = KernelCode{kernelString, Shape{256, 1, 1}, kf32};
}

void ComputeShader::loadKernelFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open kernel file: " + path);
  }
  std::string kernelString((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
  loadKernelString(kernelString);
}

bool ComputeShader::hasKernel() const { return !code.data.empty(); }

void ComputeShader::setBuffer(int tag, const Buffer &buffer) {

  if (tag >= static_cast<int>(bindings.size())) {
    bindings.resize(tag + 1);
  }

  // For the input buffer, we assume it's already been created properly.
  // For output, if buffer.bufferData.size is zero, we use a fallback value.
  size_t numElements = 0;
  if (buffer.bufferData.buffer != nullptr && buffer.bufferData.size > 0) {
    // log buffer type
    LOG(kDefLog, kInfo, "Buffer Type %zu",
      buffer.bufferType);

    numElements = buffer.bufferData.size / sizeBytes(buffer.bufferType);
  }
  Shape shape{numElements};

  bindings[tag] = Tensor{.data = buffer.bufferData, .shape = shape};
}

void ComputeShader::dispatch(int groupsX, int groupsY, int groupsZ) {

  // create array of view offsets for tensor, size_t all 0
  std::vector<size_t> viewOffsets(bindings.size(), 0);

  LOG(kDefLog, kInfo,
      "Dispatching kernel with groups: (%d, %d, %d) and bindings size: %zu",
      groupsX, groupsY, groupsZ, bindings.size());

  Kernel kernel =
      createKernel(mgpu.getContext(), code, bindings.data(), bindings.size(),
                   viewOffsets.data(),
                   {static_cast<size_t>(groupsX), static_cast<size_t>(groupsY),
                    static_cast<size_t>(groupsZ)});

  dispatchKernel(mgpu.getContext(), kernel);
}

void ComputeShader::dispatchAsync(int groupsX, int groupsY, int groupsZ,
                                  std::function<void()> callback) {
  dispatch(groupsX, groupsY, groupsZ);
  if (callback) {
    callback();
  }
}
} // namespace mgpu