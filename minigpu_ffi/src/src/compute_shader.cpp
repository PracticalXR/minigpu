#include "../include/compute_shader.h"
#include "../include/buffer.h"
#include "../include/log.h"
#include "../include/mutex.h"
#include "../include/shader_cache.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

enum LogLevel { kError = 0, kWarning = 1, kInfo = 2, kDebug = 3 };

const char *kDefLog = "ComputeShader";

namespace mgpu {

ComputeShader::ComputeShader(MGPU &mgpu) : mgpu(mgpu) {}

ComputeShader::~ComputeShader() {
  // safe to do immediately since it's on the right
  // thread
  auto cleanupTask = [computePipeline = this->computePipeline,
                      bindGroup = this->bindGroup,
                      bindGroupLayout = this->bindGroupLayout,
                      pipelineLayout = this->pipelineLayout,
                      shaderModule = this->shaderModule]() {
    if (computePipeline) {
      wgpuComputePipelineRelease(computePipeline);
    }
    if (bindGroup) {
      wgpuBindGroupRelease(bindGroup);
    }
    if (bindGroupLayout) {
      wgpuBindGroupLayoutRelease(bindGroupLayout);
    }
    if (pipelineLayout) {
      wgpuPipelineLayoutRelease(pipelineLayout);
    }
    if (shaderModule) {
      wgpuShaderModuleRelease(shaderModule);
    }
  };

  try {
    mgpu.getWebGPUThread().enqueueAsync(cleanupTask);
  } catch (...) {
    // If GPU thread is shutdown, ignore cleanup
  }
}

void ComputeShader::cleanup() {
  // defer to destructor
}

void ComputeShader::destroyQueued() {
  ComputeShader *self = this;
  try {
    // Lands after any bind/dispatch already queued against this shader. The
    // destructor itself enqueues its handle-release task from inside this one;
    // that re-entrant enqueue is safe because the worker pops a task under
    // queueMutex and then runs it with the lock released.
    mgpu.getWebGPUThread().enqueueAsync([self]() { delete self; });
  } catch (...) {
    // GPU thread already shut down: nothing can still be queued against us.
    delete self;
  }
}

void ComputeShader::loadKernelString(const std::string &kernelString) {
  if (kernelString.empty() || shaderCode == kernelString) {
    return; // No change, skip
  }

  // Guard against concurrent dispatch reads of shaderCode/pipelineDirty.
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
  shaderCode = kernelString;
  pipelineDirty = true;
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

bool ComputeShader::hasKernel() const { return !shaderCode.empty(); }

void ComputeShader::setBuffer(int tag, const Buffer &buffer) {
  if (tag < 0 || buffer.bufferData.buffer == nullptr) {
    return;
  }

  // Synchronize with dispatch/update on GPU thread
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());

  // Resize if needed
  if (tag >= static_cast<int>(buffers.size())) {
    buffers.resize(tag + 1);
  }

  // only check pointer, not size
  if (buffers[tag].buffer == buffer.bufferData.buffer) {
    return; // No change
  }

  buffers[tag] =
      BufferBinding{buffer.bufferData.buffer, buffer.bufferData.size, 0};

  // Sync into unified bindings vector
  if (tag >= static_cast<int>(bindings.size())) bindings.resize(tag + 1);
  bindings[tag] = BindingEntry{BindingKind::kStorageBuffer,
                               buffer.bufferData.buffer,
                               buffer.bufferData.size, 0, nullptr};
  bindingsDirty = true;
}

void ComputeShader::setBufferQueued(int tag, const Buffer &buffer) {
  // Capture the raw handle + size by value: the Buffer object itself may be
  // gone by the time the task runs, but buffer destruction is enqueued on
  // the same FIFO, so a handle captured before a queued destroy stays valid
  // for this task and any dispatch queued before that destroy.
  WGPUBuffer handle = buffer.bufferData.buffer;
  size_t size = buffer.bufferData.size;
  if (tag < 0 || handle == nullptr) {
    return;
  }
  mgpu.getWebGPUThread().enqueueAsync([this, tag, handle, size]() {
    mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
    if (tag >= static_cast<int>(buffers.size())) {
      buffers.resize(tag + 1);
    }
    if (buffers[tag].buffer == handle) {
      return; // No change
    }
    buffers[tag] = BufferBinding{handle, size, 0};
    if (tag >= static_cast<int>(bindings.size())) bindings.resize(tag + 1);
    bindings[tag] =
        BindingEntry{BindingKind::kStorageBuffer, handle, size, 0, nullptr};
    bindingsDirty = true;
  });
}

void ComputeShader::setTextureView(int slot, WGPUTextureView view) {
  if (slot < 0 || !view) return;
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
  if (slot >= static_cast<int>(bindings.size())) bindings.resize(slot + 1);
  bindings[slot] = BindingEntry{BindingKind::kTextureView,
                                nullptr, 0, 0, view};
  bindingsDirty = true;
}

void ComputeShader::setStorageBuffer(int slot, WGPUBuffer buf,
                                     size_t size, size_t offset) {
  if (slot < 0 || !buf) return;
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
  if (slot >= static_cast<int>(bindings.size())) bindings.resize(slot + 1);
  bindings[slot] = BindingEntry{BindingKind::kStorageBuffer,
                                buf, size, offset, nullptr};
  bindingsDirty = true;
}

void ComputeShader::setUniformBuffer(int slot, WGPUBuffer buf, size_t size) {
  if (slot < 0 || !buf) return;
  mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
  if (slot >= static_cast<int>(bindings.size())) bindings.resize(slot + 1);
  bindings[slot] = BindingEntry{BindingKind::kUniformBuffer,
                                buf, size, 0, nullptr};
  bindingsDirty = true;
}

size_t ComputeShader::calculateBindingsHash() const {
  size_t hash = bindings.size() ^ buffers.size();
  if (!bindings.empty() && bindings[0].buffer) {
    hash ^= reinterpret_cast<size_t>(bindings[0].buffer);
  }
  return hash;
}

// First non-empty line of the WGSL, stripped of comment slashes and capped —
// kernels here name themselves in a leading comment. Backend errors quote the
// object label, so this is what turns "[ComputePipeline (unlabeled)] ...
// E_OUTOFMEMORY" in a console into the name of the kernel that caused it.
std::string ComputeShader::kernelLabel() const {
  size_t pos = 0;
  while (pos < shaderCode.size()) {
    size_t eol = shaderCode.find('\n', pos);
    if (eol == std::string::npos) eol = shaderCode.size();
    size_t b = pos, e = eol;
    while (b < e && (shaderCode[b] == ' ' || shaderCode[b] == '\t' ||
                     shaderCode[b] == '/' || shaderCode[b] == '\r')) {
      ++b;
    }
    while (e > b && (shaderCode[e - 1] == ' ' || shaderCode[e - 1] == '\t' ||
                     shaderCode[e - 1] == '\r')) {
      --e;
    }
    if (e > b) return shaderCode.substr(b, std::min<size_t>(e - b, 96));
    pos = eol + 1;
  }
  return "minigpu kernel";
}

bool ComputeShader::createShaderModule() {
  // only recreate if shader actually changed
  if (shaderModule && !pipelineDirty) {
    return true; // Reuse existing module
  }

  if (shaderModule) {
    wgpuShaderModuleRelease(shaderModule);
    shaderModule = nullptr;
  }

  WGPUShaderSourceWGSL wgslDesc = {};
  wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgslDesc.code.data = shaderCode.c_str();
  wgslDesc.code.length = shaderCode.length();

  WGPUShaderModuleDescriptor shaderModuleDesc = {};
  shaderModuleDesc.nextInChain = &wgslDesc.chain;
  // Dawn copies descriptor strings during the create call, so a local is safe.
  const std::string label = kernelLabel();
  shaderModuleDesc.label.data = label.data();
  shaderModuleDesc.label.length = label.size();

  shaderModule =
      wgpuDeviceCreateShaderModule(mgpu.getDevice(), &shaderModuleDesc);
  return shaderModule != nullptr;
}

bool ComputeShader::createBindGroupLayout() {

  if (bindGroupLayout && !bindingsDirty) {
    return true;
  }

  if (bindGroupLayout) {
    wgpuBindGroupLayoutRelease(bindGroupLayout);
    bindGroupLayout = nullptr;
  }

  // Prefer the extended bindings vector; fall back to legacy buffers vector
  // when bindings is empty (existing code paths).
  const bool useExtended = !bindings.empty();

  std::vector<WGPUBindGroupLayoutEntry> layoutEntries;

  if (useExtended) {
    layoutEntries.reserve(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
      const auto& b = bindings[i];
      WGPUBindGroupLayoutEntry entry = {};
      entry.binding    = static_cast<uint32_t>(i);
      entry.visibility = WGPUShaderStage_Compute;
      switch (b.kind) {
      case BindingKind::kStorageBuffer:
        entry.buffer.type = WGPUBufferBindingType_Storage;
        entry.buffer.minBindingSize = 0;
        layoutEntries.push_back(entry);
        break;
      case BindingKind::kUniformBuffer:
        entry.buffer.type = WGPUBufferBindingType_Uniform;
        entry.buffer.minBindingSize = 0;
        layoutEntries.push_back(entry);
        break;
      case BindingKind::kTextureView:
        entry.texture.sampleType    = WGPUTextureSampleType_Float;
        entry.texture.viewDimension = WGPUTextureViewDimension_2D;
        entry.texture.multisampled  = false;
        layoutEntries.push_back(entry);
        break;
      default:
        break; // skip empty slots
      }
    }
  } else {
    // Legacy path: all bindings are storage buffers
    layoutEntries.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      if (buffers[i].buffer) {
        WGPUBindGroupLayoutEntry entry = {};
        entry.binding = static_cast<uint32_t>(i);
        entry.visibility = WGPUShaderStage_Compute;
        entry.buffer.type = WGPUBufferBindingType_Storage;
        entry.buffer.minBindingSize = 0;
        layoutEntries.push_back(entry);
      }
    }
  }

  if (layoutEntries.empty()) return false;

  WGPUBindGroupLayoutDescriptor layoutDesc = {};
  layoutDesc.entryCount = static_cast<uint32_t>(layoutEntries.size());
  layoutDesc.entries    = layoutEntries.data();

  bindGroupLayout = wgpuDeviceCreateBindGroupLayout(mgpu.getDevice(), &layoutDesc);
  return bindGroupLayout != nullptr;
}

bool ComputeShader::createPipelineLayout() {
  if (pipelineLayout && !pipelineDirty) {
    return true; // Reuse existing layout
  }

  if (pipelineLayout) {
    wgpuPipelineLayoutRelease(pipelineLayout);
    pipelineLayout = nullptr;
  }

  WGPUPipelineLayoutDescriptor pipelineLayoutDesc = {};
  pipelineLayoutDesc.bindGroupLayoutCount = 1;
  pipelineLayoutDesc.bindGroupLayouts = &bindGroupLayout;

  pipelineLayout =
      wgpuDeviceCreatePipelineLayout(mgpu.getDevice(), &pipelineLayoutDesc);
  return pipelineLayout != nullptr;
}

bool ComputeShader::createComputePipeline() {
  if (computePipeline && !pipelineDirty) {
    return true; // Reuse existing pipeline
  }

  if (computePipeline) {
    wgpuComputePipelineRelease(computePipeline);
    computePipeline = nullptr;
  }

  WGPUComputePipelineDescriptor pipelineDesc = {};
  pipelineDesc.layout = pipelineLayout;
  pipelineDesc.compute.module = shaderModule;
  pipelineDesc.compute.entryPoint.data = "main";
  pipelineDesc.compute.entryPoint.length = 4;
  // Same label as the module: uncaptured errors (a D3D12 E_OUTOFMEMORY at
  // pipeline-state creation, say) then NAME the kernel instead of printing
  // "[ComputePipeline (unlabeled)]". Note WebGPU returns an INVALID object,
  // not null, on creation failure — the nullptr check below does not catch
  // it, so the label on the later SetPipeline validation error is often the
  // only identification the console gets.
  const std::string label = kernelLabel();
  pipelineDesc.label.data = label.data();
  pipelineDesc.label.length = label.size();

  // This call is where WGSL becomes a backend shader, so it is the whole cost
  // the persistent shader cache exists to remove. Accumulating it makes the
  // difference measurable from Dart (Minigpu.shaderCacheStats) instead of
  // inferable from wall-clock, and a future regression in compile time shows
  // up here first.
  const auto pipelineT0 = std::chrono::steady_clock::now();
  computePipeline =
      wgpuDeviceCreateComputePipeline(mgpu.getDevice(), &pipelineDesc);
  shaderCacheNotePipelineMs(
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - pipelineT0)
          .count());
  return computePipeline != nullptr;
}

bool ComputeShader::createBindGroup() {
  if (bindGroup && !bindingsDirty) {
    return true; // Reuse existing bind group
  }

  if (bindGroup) {
    wgpuBindGroupRelease(bindGroup);
    bindGroup = nullptr;
  }

  const bool useExtended = !bindings.empty();
  std::vector<WGPUBindGroupEntry> bindGroupEntries;

  if (useExtended) {
    bindGroupEntries.reserve(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
      const auto& b = bindings[i];
      WGPUBindGroupEntry entry = {};
      entry.binding = static_cast<uint32_t>(i);
      switch (b.kind) {
      case BindingKind::kStorageBuffer:
      case BindingKind::kUniformBuffer:
        entry.buffer = b.buffer;
        entry.offset = b.offset;
        entry.size   = (b.size > 0) ? b.size : WGPU_WHOLE_SIZE;
        bindGroupEntries.push_back(entry);
        break;
      case BindingKind::kTextureView:
        entry.textureView = b.view;
        bindGroupEntries.push_back(entry);
        break;
      default:
        break;
      }
    }
  } else {
    bindGroupEntries.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
      if (buffers[i].buffer) {
        WGPUBindGroupEntry entry = {};
        entry.binding = static_cast<uint32_t>(i);
        entry.buffer  = buffers[i].buffer;
        entry.offset  = 0;
        entry.size    = WGPU_WHOLE_SIZE;
        bindGroupEntries.push_back(entry);
      }
    }
  }

  if (bindGroupEntries.empty()) return false;

  WGPUBindGroupDescriptor bindGroupDesc = {};
  bindGroupDesc.layout     = bindGroupLayout;
  bindGroupDesc.entryCount = static_cast<uint32_t>(bindGroupEntries.size());
  bindGroupDesc.entries    = bindGroupEntries.data();

  bindGroup = wgpuDeviceCreateBindGroup(mgpu.getDevice(), &bindGroupDesc);
  return bindGroup != nullptr;
}

bool ComputeShader::updatePipelineIfNeeded() {
  //  only rebuild what's actually dirty
  if (pipelineDirty) {
    if (!createShaderModule() || !createBindGroupLayout() ||
        !createPipelineLayout() || !createComputePipeline()) {
      return false;
    }
    pipelineDirty = false;
    bindingsDirty = true; // Force bind group recreation since layout changed
  }

  if (bindingsDirty) {
    if (!createBindGroup()) {
      return false;
    }
    bindingsDirty = false;
  }

  return true;
}

// Batch cap: bounds encoder memory and keeps a single submission far under
// the Windows TDR budget while still amortizing submit cost across hundreds
// of dispatches.
static constexpr int kMaxBatchedDispatches = 512;

void ComputeShader::dispatch(int groupsX, int groupsY, int groupsZ) {
  // Fire-and-forget: records into the context's shared batch pass (one
  // encoder + one submit per flush point instead of per dispatch).  WebGPU
  // guarantees sequential memory effects between dispatches in a pass, and
  // reads/writes/teardown flush the batch, so semantics match the previous
  // submit-per-dispatch behavior.
  auto dispatchTask = [this, groupsX, groupsY, groupsZ]() {
    // Ensure consistency with concurrent setBuffer/loadKernelString calls.
    mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());

    if (shaderCode.empty() || groupsX <= 0 || groupsY <= 0 || groupsZ <= 0) {
      LOG_ERROR("dispatch skipped: no kernel loaded or bad workgroup counts "
                "(%d,%d,%d)",
                groupsX, groupsY, groupsZ);
      return;
    }

    if (!updatePipelineIfNeeded()) {
      LOG_ERROR("dispatch skipped: pipeline/bind group creation failed "
                "(output buffers were NOT written)");
      return;
    }

    if (!computePipeline || !bindGroup) {
      LOG_ERROR("dispatch skipped: pipeline or bind group missing "
                "(output buffers were NOT written)");
      return;
    }

    if (!mgpu.batchEncoder) {
      mgpu.batchEncoder =
          wgpuDeviceCreateCommandEncoder(mgpu.getDevice(), nullptr);
      if (!mgpu.batchEncoder) {
        return;
      }
    }
    if (!mgpu.batchPass) {
      mgpu.batchPass =
          wgpuCommandEncoderBeginComputePass(mgpu.batchEncoder, nullptr);
      if (!mgpu.batchPass) {
        return;
      }
    }

    wgpuComputePassEncoderSetPipeline(mgpu.batchPass, computePipeline);
    wgpuComputePassEncoderSetBindGroup(mgpu.batchPass, 0, bindGroup, 0,
                                       nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(
        mgpu.batchPass, static_cast<uint32_t>(groupsX),
        static_cast<uint32_t>(groupsY), static_cast<uint32_t>(groupsZ));

    if (++mgpu.batchCount >= kMaxBatchedDispatches) {
      mgpu.flushBatchLocked();
    }
  };

  mgpu.getWebGPUThread().enqueueAsync(dispatchTask);
}

void ComputeShader::dispatchAsync(int groupsX, int groupsY, int groupsZ,
                                  std::function<void()> callback) {
  auto dispatchTask = [this, groupsX, groupsY, groupsZ, callback]() {
    bool ok = false;
    {
      // Scope the lock so the callback is invoked after it is released.
      // This prevents a deadlock if the callback itself makes GPU calls.
      mgpu::lock_guard<mgpu::mutex> lock(mgpu.getGpuMutex());
      // Awaited dispatches submit immediately — flush any batched
      // fire-and-forget work first to preserve execution order.
      mgpu.flushBatchLocked();

      if (shaderCode.empty() || groupsX <= 0 || groupsY <= 0 || groupsZ <= 0) {
        LOG_ERROR("dispatchAsync skipped: no kernel loaded or bad workgroup "
                  "counts (%d,%d,%d)",
                  groupsX, groupsY, groupsZ);
        // ok stays false
      } else if (!updatePipelineIfNeeded()) {
        LOG_ERROR("dispatchAsync skipped: pipeline/bind group creation failed "
                  "(output buffers were NOT written)");
        // ok stays false
      } else {
        WGPUCommandEncoder commandEncoder =
            wgpuDeviceCreateCommandEncoder(mgpu.getDevice(), nullptr);

        if (commandEncoder) {
          WGPUComputePassEncoder computePassEncoder =
              wgpuCommandEncoderBeginComputePass(commandEncoder, nullptr);

          if (computePassEncoder) {
            wgpuComputePassEncoderSetPipeline(computePassEncoder,
                                             computePipeline);
            wgpuComputePassEncoderSetBindGroup(computePassEncoder, 0,
                                               bindGroup, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(
                computePassEncoder, static_cast<uint32_t>(groupsX),
                static_cast<uint32_t>(groupsY),
                static_cast<uint32_t>(groupsZ));
            wgpuComputePassEncoderEnd(computePassEncoder);
            wgpuComputePassEncoderRelease(computePassEncoder);

            WGPUCommandBuffer commandBuffer =
                wgpuCommandEncoderFinish(commandEncoder, nullptr);
            wgpuCommandEncoderRelease(commandEncoder);

            if (commandBuffer) {
              wgpuQueueSubmit(mgpu.getQueue(), 1, &commandBuffer);
              wgpuCommandBufferRelease(commandBuffer);
              ok = true;
            }
          } else {
            wgpuCommandEncoderRelease(commandEncoder);
          }
        }
      }
    } // lock released here — safe for callback to make GPU calls

    if (callback) {
      callback();
    }
  };

  mgpu.getWebGPUThread().enqueueAsync(dispatchTask);
}

} // namespace mgpu