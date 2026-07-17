#include "../include/compute_shader.h"
#include "../include/buffer.h"
#include "../include/log.h"
#include "../include/mutex.h"
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

  computePipeline =
      wgpuDeviceCreateComputePipeline(mgpu.getDevice(), &pipelineDesc);
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

void ComputeShader::dispatch(int groupsX, int groupsY, int groupsZ) {
  // async dispatch, returns immediately
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

    WGPUCommandEncoder commandEncoder =
        wgpuDeviceCreateCommandEncoder(mgpu.getDevice(), nullptr);

    if (!commandEncoder) {
      return;
    }

    WGPUComputePassEncoder computePassEncoder =
        wgpuCommandEncoderBeginComputePass(commandEncoder, nullptr);

    if (!computePassEncoder) {
      wgpuCommandEncoderRelease(commandEncoder);
      return;
    }

    wgpuComputePassEncoderSetPipeline(computePassEncoder, computePipeline);
    wgpuComputePassEncoderSetBindGroup(computePassEncoder, 0, bindGroup, 0,
                                       nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(
        computePassEncoder, static_cast<uint32_t>(groupsX),
        static_cast<uint32_t>(groupsY), static_cast<uint32_t>(groupsZ));

    wgpuComputePassEncoderEnd(computePassEncoder);
    wgpuComputePassEncoderRelease(computePassEncoder);

    WGPUCommandBuffer commandBuffer =
        wgpuCommandEncoderFinish(commandEncoder, nullptr);
    wgpuCommandEncoderRelease(commandEncoder);

    if (commandBuffer) {
      wgpuQueueSubmit(mgpu.getQueue(), 1, &commandBuffer);
      wgpuCommandBufferRelease(commandBuffer);
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