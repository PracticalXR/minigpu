#pragma once

#include "buffer.h"
#include "minigpu_resources.h"
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
  kReadOnlyStorageBuffer,
  kUniformBuffer,
  kTextureView,
  kSampledTextureFloat,
  kSampledTextureUnfilterable,
  kDepthTexture,
  kStorageTextureWriteOnly,
  kFilteringSampler,
  kNonFilteringSampler,
  kComparisonSampler,
};

struct BindingEntry {
  BindingKind        kind   = BindingKind::kEmpty;
  WGPUBuffer         buffer = nullptr;
  size_t             size   = 0;
  size_t             offset = 0;
  WGPUTextureView    view   = nullptr;
  WGPUSampler sampler = nullptr;
  WGPUTextureViewDimension viewDimension = WGPUTextureViewDimension_Undefined;
  WGPUTextureFormat textureFormat = WGPUTextureFormat_Undefined;
  std::shared_ptr<OwnedTextureViewState> ownedTextureView;
  std::shared_ptr<OwnedSamplerState> ownedSampler;
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
  // Ordered variant: enqueues the binding update on the WebGPU thread so it
  // executes in FIFO order with dispatch/read/write tasks.  Required when
  // rebinding a shader between fire-and-forget dispatches (a plain setBuffer
  // mutates immediately and would race the still-queued earlier dispatch).
  void setBufferQueued(int tag, const Buffer &buffer);
  void setBufferRangeQueued(int slot, WGPUBuffer buffer, size_t size,
                            size_t offset, BindingKind kind);
  void setOwnedTextureQueued(int slot,
                             std::shared_ptr<OwnedTextureViewState> view,
                             BindingKind kind);
  void setOwnedSamplerQueued(int slot,
                             std::shared_ptr<OwnedSamplerState> sampler,
                             BindingKind kind);
  // Extended binding setters for video texture interop
  void setTextureView(int slot, WGPUTextureView view);
  // Drop a completed input's cached view AND bind-group references, retaining
  // its layout/pipeline so a later loan can bind without shader recompilation.
  void releaseTextureBindingQueued(int slot);
  void setStorageBuffer(int slot, WGPUBuffer buf, size_t size, size_t offset);
  void setUniformBuffer(int slot, WGPUBuffer buf, size_t size);

  void dispatch(int groupsX, int groupsY, int groupsZ);
  void dispatchAsync(int groupsX, int groupsY, int groupsZ,
                     std::function<void()> callback = nullptr);

  // Queues `delete this` on the WebGPU FIFO instead of deleting inline.
  //
  // Binds are queued (see setBufferQueued) and capture `this` to mutate the
  // binding tables when they run, so an INLINE delete would free the object out
  // from under a pending bind. Buffers do not have this problem — a queued bind
  // captures only the raw WGPUBuffer handle by value — which is why
  // mgpuDestroyBuffer can delete immediately and this cannot.
  void destroyQueued();

  MGPU &getMGPU() const { return mgpu; }
  WGPUDevice getCreationDevice() const { return creationDevice; }

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
  WGPUDevice creationDevice = nullptr;

  // State tracking
  bool pipelineDirty = true;
  bool bindingsDirty = true;
  size_t currentBindingsHash = 0;
  bool cleanupScheduled = false;

  // Helper methods
  void cleanup();
  void applyBufferBinding(int slot, WGPUBuffer buffer, size_t size,
                          size_t offset, BindingKind kind);
  bool ownedBindingsAlive() const;
  size_t calculateBindingsHash() const;
  std::string kernelLabel() const;
  bool createShaderModule();
  bool createBindGroupLayout();
  bool createPipelineLayout();
  bool createComputePipeline();
  bool createBindGroup();
  bool updatePipelineIfNeeded();
};

} // namespace mgpu
