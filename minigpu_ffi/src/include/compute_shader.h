#pragma once

#include "buffer.h"
#include "minigpu_resources.h"
#include "minigpu_capture_recipe_owner.h"
#include "minigpu_capture_payload.h"
#include "webgpu.h"
#include <string>
#include <vector>
#include <functional>
#include <future>
#include <thread>
#include <utility>

namespace mgpu {

class Buffer; // Forward declaration
class ContextPin;
namespace capture_cleanup { class Family; }
struct ShaderExclusiveUse;
struct ImportedFrameState;
struct PrivateResourceState;
struct PrivateResourceAccess;

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
  // The binding's own reference to a raw `view` whose owner may release it
  // first, so a bind-group rebuild never reads a freed handle.
  std::shared_ptr<void> retainedView;
  // A buffer bound without a kind (setBuffer) is read-only or read-write
  // storage as its kernel declares; an explicit kind is used as given.
  bool accessFromKernel = false;
};

class ComputeShader : public source_scope::MetadataUniqueRecord {
public:
  explicit ComputeShader(MGPU &mgpu);
  explicit ComputeShader(std::shared_ptr<PrivateResourceState>);
  // Private CPU storage uses MetadataDelete<ComputeShader> on every owner path;
  // no class-specific delete may run on failed ordinary legacy construction.
  ComputeShader(source_scope::MetadataUniqueRecord, std::shared_ptr<PrivateResourceState>);
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
  bool setBufferRangeQueued(int slot, WGPUBuffer buffer, size_t size,
                            size_t offset, BindingKind kind);
  bool setOwnedTextureQueued(int slot,
                             std::shared_ptr<OwnedTextureViewState> view,
                             BindingKind kind);
  bool setOwnedSamplerQueued(int slot,
                             std::shared_ptr<OwnedSamplerState> sampler,
                             BindingKind kind);
  // Extended binding setters for video texture interop
  void setTextureView(int slot, WGPUTextureView view);
  // Admission happens now, not inside another owner's queued job. Retain that
  // owner through the mutation; its failure callback runs outside shader/GPU locks.
  bool setTextureViewQueued(int slot, WGPUTextureView view,
                            std::shared_ptr<void> owner = {},
                            std::function<void()> failure = {});
  // Drop a completed input's cached view AND bind-group references, retaining
  // its layout/pipeline so a later loan can bind without shader recompilation.
  bool releaseTextureBindingQueued(int slot);
  void setStorageBuffer(int slot, WGPUBuffer buf, size_t size, size_t offset);
  void setUniformBuffer(int slot, WGPUBuffer buf, size_t size);

  void dispatch(int groupsX, int groupsY, int groupsZ);
  void dispatchAsync(int groupsX, int groupsY, int groupsZ,
                     std::function<void()> callback = nullptr);
  void dispatchAsyncChecked(int groupsX, int groupsY, int groupsZ,
                            std::function<void(bool)> callback);
  // Compiles the pipeline for the current kernel and binding layout on the
  // WebGPU FIFO, the way a dispatch would, but records nothing. `done(ok)` runs
  // exactly once, outside the shader/GPU locks; ok means that pipeline exists
  // now (false for a refused shader, no kernel, no bindings, a failed compile).
  // The bind group is built too when the bindings allow; a binding problem is
  // still reported by the dispatch that meets it.
  void buildPipelineAsync(std::function<void(bool)> done);
  // The same, waiting for the answer (the browser runs the task inline). False
  // on the WebGPU thread itself, which cannot wait on its own FIFO.
  bool buildPipeline();
  // Buffer-only bind groups served from / created into the per-shader cache.
  uint64_t bindGroupCacheHits() const noexcept {
    return bindGroupHits_.load(std::memory_order_relaxed);
  }
  uint64_t bindGroupCacheMisses() const noexcept {
    return bindGroupMisses_.load(std::memory_order_relaxed);
  }

  int acquireExclusiveUse(std::shared_ptr<ContextPin> pin,
                          source_scope::MetadataUnique<MGPUComputeShaderUse> &out);
  int dispatchExclusiveUse(const std::shared_ptr<ShaderExclusiveUse> &use,
                           int x, int y, int z);
  int bindImportedFrame(const std::shared_ptr<ShaderExclusiveUse> &use,
                        uint32_t slot,
                        const std::shared_ptr<ImportedFrameState> &frame);
  void retireExclusiveUse(const std::shared_ptr<ShaderExclusiveUse> &use,
                          std::function<void(int)> callback);
  void retireExclusiveUse(const std::shared_ptr<ShaderExclusiveUse> &use,
      MGPUQueueCompletionCallback callback, void *userdata, const capture_cleanup::Family &);
  int releaseExclusiveUse(const std::shared_ptr<ShaderExclusiveUse> &use);

  // Queues `delete this` on the WebGPU FIFO instead of deleting inline.
  //
  // Binds are queued (see setBufferQueued) and capture `this` to mutate the
  // binding tables when they run, so an INLINE delete would free the object out
  // from under a pending bind. Buffers are deleted the same way
  // (Buffer::destroyQueued), because a queued read captures the Buffer.
  void destroyQueued();
  int destroyCheckedAsync(std::shared_ptr<ContextPin> pin,
                          MGPUResourceCleanupCallback callback, void *userdata);
  int destroyCheckedAsync(std::shared_ptr<ContextPin> pin,
                          MGPUResourceCleanupCallback callback, void *userdata,
                          const capture_cleanup::Family &family);

  MGPU &getMGPU() const { return mgpu; }
  WGPUDevice getCreationDevice() const { return creationDevice; }

private:
  friend struct PrivateResourceAccess;
  MGPU &mgpu;
  // Admission/enqueue are atomic. Lock order is submission -> GPU; admitted
  // legacy worker bodies never reacquire submission while holding GPU.
  mgpu::mutex submissionMutex_;
  std::shared_ptr<ShaderExclusiveUse> exclusiveUse_;
  std::shared_ptr<ContextPin> lifetimePin_;
  std::shared_ptr<PrivateResourceState> privateResource_;
  bool destroyRequested_ = false;
  bool deleteQueued_ = false;
  bool interference_ = false;
  capture_payload::Kernel shaderCode;
  std::vector<BufferBinding> buffers;  // legacy: slot → storage buffer (Buffer objects)
  capture_payload::Array<BindingEntry, capture_payload::bindingSlots> bindings;

  // WebGPU resources
  WGPUShaderModule shaderModule = nullptr;
  WGPUBindGroupLayout bindGroupLayout = nullptr;
  WGPUPipelineLayout pipelineLayout = nullptr;
  WGPUComputePipeline computePipeline = nullptr;
  WGPUBindGroup bindGroup = nullptr;
  WGPUDevice creationDevice = nullptr;

  // State tracking
  // The WGSL module depends only on the kernel text: a binding-kind change
  // rebuilds the layouts and pipeline but keeps the parsed module. Identity
  // only (the module itself holds its device alive), so a legacy shader that
  // meets a re-created context still compiles a module for the new device.
  bool moduleDirty = true;
  WGPUDevice moduleDevice = nullptr;
  bool pipelineDirty = true;
  bool bindingsDirty = true;
  size_t currentBindingsHash = 0;
  bool cleanupScheduled = false;
  // Bumped with every pipelineDirty: a pipeline compiled with the GPU mutex
  // released is installed only if no kernel or layout change landed meanwhile.
  uint64_t layoutEpoch_ = 0;

  // Bind groups for buffer-only binding sets, so a ping-pong between buffer
  // sets (or the forced rebind of the stale-alias rule) stops creating one per
  // dispatch. Each entry owns references to its group AND its key buffers: a
  // live key handle cannot be recycled by a later buffer, so a hit is the same
  // buffers. Textures and samplers never enter (their release paths must be
  // able to drop view refs), and neither do capture-scoped shaders.
  struct CachedBindGroup {
    struct Slot {
      uint32_t binding = 0;
      WGPUBuffer buffer = nullptr;
      uint64_t offset = 0;
      uint64_t size = 0;
      bool operator==(const Slot &) const = default;
    };
    std::vector<Slot> key;
    uint64_t epoch = 0;
    WGPUBindGroup group = nullptr;
    uint64_t lastUse = 0;
  };
  static constexpr size_t kBindGroupCacheEntries = 8;
  std::vector<CachedBindGroup> bindGroupCache_;
  std::vector<CachedBindGroup::Slot> bindGroupProbe_; // reused lookup key
  WGPUDevice bindGroupCacheDevice_ = nullptr; // add-ref'd while entries exist
  uint64_t bindGroupClock_ = 0;
  std::atomic<uint64_t> bindGroupHits_{0};
  std::atomic<uint64_t> bindGroupMisses_{0};

  // Helper methods
  bool admitLegacyLocked();
  template<class Callback> void retireExclusiveUsePrepared(
      const std::shared_ptr<ShaderExclusiveUse> &, Callback,
      const capture_cleanup::Family &);
  bool enqueueLegacy(std::function<void()> task);
  void queueDeleteLocked();
  bool releaseCheckedOnWorker(const ContextPin &pin);
  void applyTextureBinding(int slot, WGPUTextureView view,
                           std::shared_ptr<void> retained = {});
  bool dispatchImmediateLocked(mgpu::unique_lock<mgpu::mutex> &lock, int x,
                               int y, int z, WGPUDevice device,
                               WGPUQueue queue);
  WGPUDevice pipelineDevice() const;
  void cleanup();
  void applyBufferBinding(int slot, WGPUBuffer buffer, size_t size,
                          size_t offset, BindingKind kind,
                          bool accessFromKernel = false);
  bool ownedBindingsAlive() const;
  size_t calculateBindingsHash() const;
  std::string_view kernelLabel() const;
  bool createShaderModule();
  bool createBindGroupLayout();
  bool createPipelineLayout();
  bool createComputePipeline();
  bool createBindGroup();
  bool updatePipelineIfNeeded();
  // Worker paths that hold the GPU mutex in `lock`: the pipeline compile runs
  // with it released (native), everything else as above under it.
  bool updatePipelineIfNeeded(mgpu::unique_lock<mgpu::mutex> &lock);
  bool compilePipelineOffLock(mgpu::unique_lock<mgpu::mutex> &lock);
  void markLayoutDirty();
  bool cacheableBindGroupKey();
  bool reuseCachedBindGroup();
  void rememberBindGroup();
  void purgeBindGroupCache();
  static void releaseCachedBindGroups(const std::vector<CachedBindGroup> &cache,
                                      WGPUDevice device);
};

// Native resource ownership only, never source/publication authority. A shader
// holds this state until successful retirement and exclusive Release. A lost or
// quarantined token deliberately retains its shader/pin and bounded quota.
class ShaderUseCustody : public capture_recipe::ResourceCustody {
public:
  ShaderUseCustody(source_scope::MetadataAllocation allocation,
                   capture_recipe::ResourceCustody &&role) noexcept
      : capture_recipe::ResourceCustody(std::move(role), std::move(allocation)) {}
  ~ShaderUseCustody(); // After State members; source role outlives global refund.
  void setGlobalUseCharge() noexcept { charged_ = true; }
private:
  bool charged_ = false;
};
struct ShaderExclusiveUse : ShaderUseCustody {
  ShaderExclusiveUse(source_scope::MetadataAllocation allocation,
                     capture_recipe::ResourceCustody &&role) noexcept
      : ShaderUseCustody(std::move(allocation), std::move(role)) {}
  enum class Phase { Reserved, Active, Retiring, Retired, Quarantined };
  ComputeShader *shader = nullptr;
  std::shared_ptr<ContextPin> pin;
  Phase phase = Phase::Reserved; // protected by shader's submission mutex
  bool sealed = false;
  bool frameBindingPending = false;
  bool dispatchAdmitted = false;
  uint32_t frameSlot = 0;
  std::shared_ptr<ImportedFrameState> frame;
  std::vector<BindingEntry> retiringBindings;
};

} // namespace mgpu

// Private holder definition shared by the implementation and its C ABI bridge.
// The native acquire reserves claims before allocating this holder.
struct MGPUComputeShaderUse : mgpu::source_scope::MetadataUniqueRecord {
  explicit MGPUComputeShaderUse(mgpu::source_scope::MetadataUniqueRecord storage) noexcept
      : MetadataUniqueRecord(std::move(storage)) {}
  std::shared_ptr<mgpu::ShaderExclusiveUse> state;
};
