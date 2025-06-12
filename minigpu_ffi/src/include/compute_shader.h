#pragma once

#include "buffer.h"
#include "webgpu.h"
#include <string>
#include <vector>
#include <functional>
#include <future>
#include <thread>

namespace mgpu {

class Buffer; // Forward declaration

struct BufferBinding {
  WGPUBuffer buffer = nullptr;
  size_t size = 0;
  size_t offset = 0;
};

class ComputeShader {
public:
  explicit ComputeShader(MGPU &mgpu);
  ~ComputeShader();

  // Delete copy constructor and assignment operator
  ComputeShader(const ComputeShader &) = delete;
  ComputeShader &operator=(const ComputeShader &) = delete;

  void loadKernelString(const std::string &kernelString);
  void loadKernelFile(const std::string &path);
  bool hasKernel() const;

  void setBuffer(int tag, const Buffer &buffer);

  void dispatch(int groupsX, int groupsY, int groupsZ);
  void dispatchAsync(int groupsX, int groupsY, int groupsZ,
                     std::function<void()> callback = nullptr);

private:
  MGPU &mgpu;
  std::string shaderCode;
  std::vector<BufferBinding> buffers;

  // WebGPU resources
  WGPUShaderModule shaderModule = nullptr;
  WGPUBindGroupLayout bindGroupLayout = nullptr;
  WGPUPipelineLayout pipelineLayout = nullptr;
  WGPUComputePipeline computePipeline = nullptr;
  WGPUBindGroup bindGroup = nullptr;

  // State tracking
  bool pipelineDirty = true;
  bool bindingsDirty = true;
  size_t currentBindingsHash = 0;
  bool cleanupScheduled = false;

  // Helper methods
  void cleanup();
  size_t calculateBindingsHash() const;
  bool createShaderModule();
  bool createBindGroupLayout();
  bool createPipelineLayout();
  bool createComputePipeline();
  bool createBindGroup();
  bool updatePipelineIfNeeded();
};

} // namespace mgpu