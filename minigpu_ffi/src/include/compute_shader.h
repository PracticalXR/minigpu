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

enum class BindingKind {
  kEmpty = 0,
  kStorageBuffer,
  kUniformBuffer,
  kTextureView,
};

struct BindingEntry {
  BindingKind        kind   = BindingKind::kEmpty;
  WGPUBuffer         buffer = nullptr;
  size_t             size   = 0;
  size_t             offset = 0;
  WGPUTextureView    view   = nullptr;
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
  // Extended binding setters for video texture interop
  void setTextureView(int slot, WGPUTextureView view);
  void setStorageBuffer(int slot, WGPUBuffer buf, size_t size, size_t offset);
  void setUniformBuffer(int slot, WGPUBuffer buf, size_t size);

  void dispatch(int groupsX, int groupsY, int groupsZ);
  void dispatchAsync(int groupsX, int groupsY, int groupsZ,
                     std::function<void()> callback = nullptr);

private:
  MGPU &mgpu;
  std::string shaderCode;
  std::vector<BufferBinding> buffers;  // legacy: slot → storage buffer (Buffer objects)
  std::vector<BindingEntry>  bindings; // extended: all binding types indexed by slot

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