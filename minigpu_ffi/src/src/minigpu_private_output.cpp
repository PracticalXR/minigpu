#include "../include/minigpu_private_output.h"
#include "../include/minigpu_metadata_layout.h"
#include "../include/minigpu_context_owner.h"
#include "../include/minigpu_error_scope.h"
#include "../include/minigpu_capture_output_owner.h"
#include "../include/minigpu_capture_cleanup_adapter.h"
#include "../include/minigpu_private_resource_owner.h"
#include "../include/minigpu_module_owner.h"
#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#ifdef _WIN32
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <dawn/native/D3D11Backend.h>
#endif

namespace mgpu::private_output {
constexpr uint32_t maxWriters = 64, maxOutputs = 256;
constexpr uint32_t maxLoans = 256, maxBorrows = 1024, maxJobs = 512;
constexpr uint64_t maxFrameBytes = 128ull * 1024 * 1024;
constexpr uint64_t maxBytes = 512ull * 1024 * 1024;

using capture_cleanup::Family;
using capture_cleanup::GlobalTaskCustody;
using capture_cleanup::nativeResult;
using capture_work::Outcome;
using capture_work::PreparedWork;
using capture_work::PreparedWait;
using capture_recipe::ResourceCustody;
using source_scope::ResourceKind;
using source_scope::Result;
using source_scope::MetadataAllocation;
using source_scope::MetadataReservations;
using source_scope::MetadataKind;
using source_scope::MetadataUnique;
using source_scope::MetadataDelete;
using source_scope::makeMetadataShared;
using source_scope::makeMetadataUnique;

struct Base : ResourceCustody {
  Base(MetadataAllocation allocation, ResourceCustody &&role) noexcept
      : ResourceCustody(std::move(role), std::move(allocation)) {}
  std::mutex mutex;
  std::shared_ptr<ContextPin> pin;
  uint64_t id = 0, bytes = 0;
  uint32_t width = 0, height = 0, slot = 0;
  uint32_t phase = MGPU_PRIVATE_OUTPUT_EMPTY;
  int result = MGPU_CONTEXT_LEASE_OK;
  bool sealed = false;
};
struct Writer : Base {
  using Base::Base;
  std::shared_ptr<PrivateResourceState> packedState; // After native release, until final aliases.
  uint32_t creates = 0;
  bool copying = false;
#ifdef _WIN32
  WGPUBuffer input = nullptr;
  WGPUTexture intermediate = nullptr;
  WGPUTextureView view = nullptr;
  WGPUShaderModule module = nullptr;
  WGPUBindGroupLayout bgl = nullptr;
  WGPUPipelineLayout layout = nullptr;
  WGPUComputePipeline pipeline = nullptr;
  WGPUBindGroup group = nullptr;
#endif
};
struct Output : Base {
  using Base::Base;
  bool publicOwner = true, previewCreated = false, previewActive = false;
  uint32_t borrows = 0;
#ifdef _WIN32
  Microsoft::WRL::ComPtr<ID3D11Texture2D> source;
  HANDLE sharedHandle = nullptr; // Legacy DXGI handle, never CloseHandle.
  WGPUSharedTextureMemory memory = nullptr;
  WGPUTexture texture = nullptr;
  bool beginAttempted = false, endAttempted = false;
#endif
};
struct Loan : capture_output::AliasCustody<Loan> {
  Loan(MetadataAllocation allocation, ResourceCustody &&role) noexcept
      : AliasCustody(std::move(allocation), std::move(role)) {}
  std::shared_ptr<Output> output;
  uint64_t id = 0;
  bool sealed = false, quarantined = false;
#ifdef _WIN32
  Microsoft::WRL::ComPtr<ID3D11Texture2D> registration;
  HANDLE sharedHandle = nullptr;
#endif
};
struct Borrow : capture_output::AliasCustody<Borrow> {
  Borrow(MetadataAllocation allocation, ResourceCustody &&role) noexcept
      : AliasCustody(std::move(allocation), std::move(role)) {}
  std::shared_ptr<Loan> loan;
  MGPUPrivatePreviewBorrowInfo info{};
#ifdef _WIN32
  Microsoft::WRL::ComPtr<ID3D11Texture2D> reference;
#endif
};
struct Registry {
  std::mutex mutex;
  std::array<std::shared_ptr<Writer>, maxWriters> writers{};
  std::array<std::shared_ptr<Output>, maxOutputs> outputs{};
  uint32_t writerCount = 0, outputCount = 0, loans = 0, borrows = 0, jobs = 0;
  uint64_t bytes = 0, nextId = 1;
};
Registry &registry() {
  // Fixed process-rooted slots retain every uncertain owner and exact pin.
  // Initialization happens before admitting work; failure paths allocate nothing.
  return mgpu::detail::moduleOwner<Registry>();
}
uint64_t nextIdLocked(Registry &r) {
  return r.nextId == (std::numeric_limits<uint64_t>::max)() ? 0 : r.nextId++;
}
bool imageBytes(uint32_t width, uint32_t height, uint64_t &bytes) {
  if (!width || !height || width > maxFrameBytes / 4 ||
      height > maxFrameBytes / (uint64_t(width) * 4)) return false;
  bytes = uint64_t(width) * height * 4;
  return bytes && bytes <= maxFrameBytes;
}
template<class T, size_t N>
bool reserve(const std::shared_ptr<T> &s,
             std::array<std::shared_ptr<T>, N> &slots, uint32_t &count,
             Registry &r) {
  if (count >= N || s->bytes > maxBytes - r.bytes) return false;
  for (uint32_t i = 0; i < N; ++i) {
    if (!slots[i]) {
      s->id = nextIdLocked(r);
      if (!s->id) return false;
      s->slot = i;
      slots[i] = s;
      ++count;
      r.bytes += s->bytes;
      return true;
    }
  }
  return false;
}
bool reserveWriter(const std::shared_ptr<Writer> &s) {
  auto &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  return reserve(s, r.writers, r.writerCount, r);
}
bool reserveOutput(const std::shared_ptr<Output> &s) {
  auto &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  return reserve(s, r.outputs, r.outputCount, r);
}
bool removeWriter(const std::shared_ptr<Writer> &s) {
  auto &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (s->slot >= maxWriters || r.writers[s->slot] != s) return false;
  r.writers[s->slot].reset();
  --r.writerCount;
  r.bytes -= s->bytes;
  return true;
}
void removeOutputIfUnowned(const std::shared_ptr<Output> &s) {
  // Caller holds output gate and a local strong reference.
  if (s->publicOwner || s->previewActive || s->borrows ||
      s->phase != MGPU_PRIVATE_OUTPUT_RETIRED) return;
  auto &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (s->slot < maxOutputs && r.outputs[s->slot] == s) {
    r.outputs[s->slot].reset();
    --r.outputCount;
    r.bytes -= s->bytes;
  }
}
bool chargeJob() {
  auto &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (r.jobs >= maxJobs) return false;
  ++r.jobs;
  return true;
}
bool finishJob() noexcept {
  try {
    auto &r = registry(); // Already initialized before admission.
    std::lock_guard<std::mutex> lock(r.mutex);
    if (!r.jobs) return false;
    --r.jobs;
    return true;
  } catch (...) { return false; }
}
struct AliasSlot {
  Output *output;
  uint64_t id = 0;
};
bool reserveLoanSlot(void *context) {
  auto &slot = *static_cast<AliasSlot *>(context);
  auto &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (r.loans >= maxLoans || !(slot.id = nextIdLocked(r))) return false;
  ++r.loans;
  return true;
}
bool refundLoanSlot(void *) noexcept {
  try {
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.mutex);
    if (!r.loans) return false;
    --r.loans; return true;
  } catch (...) { return false; }
}
bool reserveBorrowSlot(void *context) {
  auto &slot = *static_cast<AliasSlot *>(context);
  // Admitting caller holds the output gate before this registry gate.
  auto &r = registry(); std::lock_guard<std::mutex> lock(r.mutex);
  if (r.borrows >= maxBorrows || slot.output->borrows >= 4 ||
      !(slot.id = nextIdLocked(r))) return false;
  ++r.borrows; ++slot.output->borrows; return true;
}
bool refundBorrowSlot(void *context) noexcept {
  try {
    auto &slot = *static_cast<AliasSlot *>(context);
    std::lock_guard<std::mutex> outputLock(slot.output->mutex);
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.mutex);
    if (!r.borrows || !slot.output->borrows) return false;
    --r.borrows; --slot.output->borrows; return true;
  } catch (...) { return false; }
}
void notify(MGPUQueueCompletionCallback cb, void *userdata, int result) noexcept {
  try { cb(result, userdata); } catch (...) {} // Caller must not throw.
}
void quarantine(Base &s) {
  s.sealed = true;
  s.phase = MGPU_PRIVATE_OUTPUT_QUARANTINED;
  s.result = MGPU_CONTEXT_LEASE_QUARANTINED;
  s.quarantineSourceRole();
}
void quarantineFactory(const std::shared_ptr<Base> &s) noexcept {
  if (!s) return;
  try {
    std::lock_guard<std::mutex> lock(s->mutex);
    quarantine(*s);
  } catch (...) { s->quarantineSourceRole(); }
}
int enqueueOutput(void *context, task_queue::Prepared &task) noexcept {
  return static_cast<MGPU *>(context)->getWebGPUThread().enqueuePrepared(task);
}
int prepareJob(const Family &family, PreparedWork &job,
               std::shared_ptr<GlobalTaskCustody> &global) noexcept {
  const auto admitted = GlobalTaskCustody::create(family, chargeJob, finishJob, global);
  return admitted == Result::Ok ? job.reserve(family,
      source_scope::ContinuationKind::PrimaryTask, global) : nativeResult(admitted);
}
MGPUPrivateOutputInfo info(const Base &s, uint32_t kind) {
  MGPUPrivateOutputInfo value{};
  value.struct_size = sizeof(value);
  value.version = 1;
  value.kind = kind;
  value.phase = s.phase;
  value.last_result = s.result;
  value.flags = s.sealed ? MGPU_PRIVATE_OUTPUT_SEALED : 0;
  value.resource_id = s.id;
  value.context_generation = s.pin->contextGeneration();
  value.device_generation = s.pin->deviceGeneration();
  value.charged_bytes = s.bytes;
  value.width = s.width;
  value.height = s.height;
  return value;
}

#ifdef _WIN32
constexpr const char *copyShader = R"(
@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var out_tex: texture_storage_2d<bgra8unorm, write>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) p: vec3<u32>) {
  let size = textureDimensions(out_tex);
  if (p.x >= size.x || p.y >= size.y) { return; }
  let packed = pixels[p.y * size.x + p.x];
  let color = vec4<f32>(f32(packed & 255u), f32((packed >> 8u) & 255u),
                       f32((packed >> 16u) & 255u), f32(packed >> 24u)) / 255.0;
  textureStore(out_tex, vec2<i32>(p.xy), color);
})";

int deviceProfile(const std::shared_ptr<ContextPin> &pin,
                   uint32_t width, uint32_t height, uint64_t bytes) {
  if (!pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
  WGPUAdapterInfo adapter{};
  const auto got = wgpuDeviceGetAdapterInfo(pin->device(), &adapter);
  const bool d3d11 = got == WGPUStatus_Success &&
                     adapter.backendType == WGPUBackendType_D3D11;
  wgpuAdapterInfoFreeMembers(adapter);
  if (!d3d11 ||
      !wgpuDeviceHasFeature(pin->device(), WGPUFeatureName_SharedTextureMemoryD3D11Texture2D) ||
      !wgpuDeviceHasFeature(pin->device(), WGPUFeatureName_BGRA8UnormStorage))
    return MGPU_CONTEXT_LEASE_UNSUPPORTED;
  WGPULimits limits{};
  if (wgpuDeviceGetLimits(pin->device(), &limits) != WGPUStatus_Success)
    return MGPU_CONTEXT_LEASE_INTERNAL;
  if (width > limits.maxTextureDimension2D || height > limits.maxTextureDimension2D ||
      bytes > limits.maxStorageBufferBindingSize ||
      (width + 7) / 8 > limits.maxComputeWorkgroupsPerDimension ||
      (height + 7) / 8 > limits.maxComputeWorkgroupsPerDimension)
    return MGPU_CONTEXT_LEASE_INVALID;
  return MGPU_CONTEXT_LEASE_OK;
}
void releaseWriterNative(Writer &s) {
  mgpu::lock_guard<mgpu::mutex> lock(s.pin->gpu().getGpuMutex());
  if (s.group) { wgpuBindGroupRelease(s.group); s.group = nullptr; }
  if (s.pipeline) { wgpuComputePipelineRelease(s.pipeline); s.pipeline = nullptr; }
  if (s.layout) { wgpuPipelineLayoutRelease(s.layout); s.layout = nullptr; }
  if (s.bgl) { wgpuBindGroupLayoutRelease(s.bgl); s.bgl = nullptr; }
  if (s.module) { wgpuShaderModuleRelease(s.module); s.module = nullptr; }
  if (s.view) { wgpuTextureViewRelease(s.view); s.view = nullptr; }
  if (s.intermediate) {
    wgpuTextureDestroy(s.intermediate);
    wgpuTextureRelease(s.intermediate);
    s.intermediate = nullptr;
  }
  // The input belongs to the caller. Release only the writer's independent ref.
  if (s.input) { wgpuBufferRelease(s.input); s.input = nullptr; }
}
void releaseOutputNative(Output &s) {
  mgpu::lock_guard<mgpu::mutex> lock(s.pin->gpu().getGpuMutex());
  if (s.texture) { wgpuTextureRelease(s.texture); s.texture = nullptr; }
  if (s.memory) { wgpuSharedTextureMemoryRelease(s.memory); s.memory = nullptr; }
  s.source.Reset(); // A preview loan has its own independent COM reference.
  s.sharedHandle = nullptr; // Legacy DXGI handle is not closed.
}
int createWriter(Writer &s, WGPUBuffer input) {
  int result = deviceProfile(s.pin, s.width, s.height, s.bytes);
  if (result != MGPU_CONTEXT_LEASE_OK) return result;
  PinnedErrorScopes errors(s.pin);
  {
    mgpu::lock_guard<mgpu::mutex> lock(s.pin->gpu().getGpuMutex());
    result = [&]() {
      if (!s.pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
      s.input = input;
      wgpuBufferAddRef(input);
      WGPUShaderSourceWGSL wgsl{};
      wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
      wgsl.code = {copyShader, std::strlen(copyShader)};
      WGPUShaderModuleDescriptor shader{};
      shader.nextInChain = &wgsl.chain;
      s.module = wgpuDeviceCreateShaderModule(s.pin->device(), &shader);
      WGPUTextureDescriptor texture{};
      texture.usage = WGPUTextureUsage_StorageBinding | WGPUTextureUsage_CopySrc;
      texture.dimension = WGPUTextureDimension_2D;
      texture.size = {s.width, s.height, 1};
      texture.format = WGPUTextureFormat_BGRA8Unorm;
      texture.mipLevelCount = texture.sampleCount = 1;
      s.intermediate = wgpuDeviceCreateTexture(s.pin->device(), &texture);
      if (!s.module || !s.intermediate) return MGPU_CONTEXT_LEASE_INTERNAL;
      WGPUTextureViewDescriptor view{};
      view.dimension = WGPUTextureViewDimension_2D;
      view.format = WGPUTextureFormat_BGRA8Unorm;
      view.mipLevelCount = view.arrayLayerCount = 1;
      view.aspect = WGPUTextureAspect_All;
      s.view = wgpuTextureCreateView(s.intermediate, &view);
      WGPUBindGroupLayoutEntry entries[2]{};
      entries[0].binding = 0;
      entries[0].visibility = WGPUShaderStage_Compute;
      entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
      entries[0].buffer.minBindingSize = s.bytes;
      entries[1].binding = 1;
      entries[1].visibility = WGPUShaderStage_Compute;
      entries[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
      entries[1].storageTexture.format = WGPUTextureFormat_BGRA8Unorm;
      entries[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
      WGPUBindGroupLayoutDescriptor bgl{};
      bgl.entryCount = 2; bgl.entries = entries;
      s.bgl = wgpuDeviceCreateBindGroupLayout(s.pin->device(), &bgl);
      if (!s.view || !s.bgl) return MGPU_CONTEXT_LEASE_INTERNAL;
      WGPUPipelineLayoutDescriptor layout{};
      layout.bindGroupLayoutCount = 1; layout.bindGroupLayouts = &s.bgl;
      s.layout = wgpuDeviceCreatePipelineLayout(s.pin->device(), &layout);
      WGPUComputePipelineDescriptor pipeline{};
      pipeline.layout = s.layout;
      pipeline.compute.module = s.module;
      pipeline.compute.entryPoint = {"main", 4};
      s.pipeline = wgpuDeviceCreateComputePipeline(s.pin->device(), &pipeline);
      WGPUBindGroupEntry bindings[2]{};
      bindings[0].binding = 0; bindings[0].buffer = input; bindings[0].size = s.bytes;
      bindings[1].binding = 1; bindings[1].textureView = s.view;
      WGPUBindGroupDescriptor group{};
      group.layout = s.bgl; group.entryCount = 2; group.entries = bindings;
      s.group = wgpuDeviceCreateBindGroup(s.pin->device(), &group);
      return s.layout && s.pipeline && s.group ?
          MGPU_CONTEXT_LEASE_OK : MGPU_CONTEXT_LEASE_INTERNAL;
    }();
  }
  if (!errors.finish()) result = MGPU_CONTEXT_LEASE_INTERNAL;
  return result;
}
int createOutput(Output &s) {
  int result = deviceProfile(s.pin, s.width, s.height, s.bytes);
  if (result != MGPU_CONTEXT_LEASE_OK) return result;
  auto device = dawn::native::d3d11::GetD3D11Device(s.pin->device());
  if (!device) return MGPU_CONTEXT_LEASE_UNSUPPORTED;
  D3D11_TEXTURE2D_DESC desc{};
  desc.Width = s.width; desc.Height = s.height;
  desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
  desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
  desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
  if (FAILED(device->CreateTexture2D(&desc, nullptr, &s.source)) || !s.source)
    return MGPU_CONTEXT_LEASE_INTERNAL;
  Microsoft::WRL::ComPtr<IDXGIResource> resource;
  if (FAILED(s.source.As(&resource)) ||
      FAILED(resource->GetSharedHandle(&s.sharedHandle)) || !s.sharedHandle)
    return MGPU_CONTEXT_LEASE_INTERNAL;
  PinnedErrorScopes errors(s.pin);
  {
    mgpu::lock_guard<mgpu::mutex> lock(s.pin->gpu().getGpuMutex());
    result = [&]() {
      if (!s.pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
      dawn::native::d3d11::SharedTextureMemoryD3D11Texture2DDescriptor chain;
      chain.texture = s.source;
      WGPUSharedTextureMemoryDescriptor memory{};
      memory.nextInChain = const_cast<WGPUChainedStruct *>(
          reinterpret_cast<const WGPUChainedStruct *>(
              static_cast<const wgpu::ChainedStruct *>(&chain)));
      s.memory = wgpuDeviceImportSharedTextureMemory(s.pin->device(), &memory);
      if (!s.memory) return MGPU_CONTEXT_LEASE_INTERNAL;
      WGPUSharedTextureMemoryProperties properties{};
      if (wgpuSharedTextureMemoryGetProperties(s.memory, &properties) != WGPUStatus_Success ||
          properties.format != WGPUTextureFormat_BGRA8Unorm ||
          properties.size.width != s.width || properties.size.height != s.height ||
          properties.size.depthOrArrayLayers != 1 ||
          !(properties.usage & WGPUTextureUsage_CopyDst))
        return MGPU_CONTEXT_LEASE_INVALID;
      WGPUTextureDescriptor texture{};
      texture.usage = WGPUTextureUsage_CopyDst;
      texture.dimension = WGPUTextureDimension_2D;
      texture.size = properties.size; texture.format = properties.format;
      texture.mipLevelCount = texture.sampleCount = 1;
      s.texture = wgpuSharedTextureMemoryCreateTexture(s.memory, &texture);
      return s.texture ? MGPU_CONTEXT_LEASE_OK : MGPU_CONTEXT_LEASE_INTERNAL;
    }();
  }
  if (!errors.finish()) result = MGPU_CONTEXT_LEASE_INTERNAL;
  return result;
}
int submitCopy(Writer &w, Output &o) {
  if (!w.pin->current()) return MGPU_CONTEXT_LEASE_RETIRED; // No access attempted.
  PinnedErrorScopes errors(w.pin);
  int result = MGPU_CONTEXT_LEASE_INTERNAL;
  {
    mgpu::lock_guard<mgpu::mutex> gpu(w.pin->gpu().getGpuMutex());
    result = [&]() {
      if (!w.pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
      WGPUSharedTextureMemoryBeginAccessDescriptor begin{};
      begin.initialized = false; // First and only complete write.
      begin.concurrentRead = false;
      o.beginAttempted = true; // Failure may follow actual state changes.
      if (wgpuSharedTextureMemoryBeginAccess(o.memory, o.texture, &begin) != WGPUStatus_Success)
        return MGPU_CONTEXT_LEASE_INTERNAL;
      WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(w.pin->device(), nullptr);
      if (!encoder) return MGPU_CONTEXT_LEASE_INTERNAL;
      WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
      if (!pass) { wgpuCommandEncoderRelease(encoder); return MGPU_CONTEXT_LEASE_INTERNAL; }
      wgpuComputePassEncoderSetPipeline(pass, w.pipeline);
      wgpuComputePassEncoderSetBindGroup(pass, 0, w.group, 0, nullptr);
      wgpuComputePassEncoderDispatchWorkgroups(pass, (w.width + 7) / 8, (w.height + 7) / 8, 1);
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass);
      WGPUTexelCopyTextureInfo source{}, destination{};
      source.texture = w.intermediate; source.aspect = WGPUTextureAspect_All;
      destination.texture = o.texture; destination.aspect = WGPUTextureAspect_All;
      WGPUExtent3D extent{w.width, w.height, 1};
      wgpuCommandEncoderCopyTextureToTexture(encoder, &source, &destination, &extent);
      WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
      wgpuCommandEncoderRelease(encoder);
      if (!commands) return MGPU_CONTEXT_LEASE_INTERNAL;
      w.pin->gpu().flushBatchLocked();
      wgpuQueueSubmit(w.pin->queue(), 1, &commands);
      wgpuCommandBufferRelease(commands);
      return MGPU_CONTEXT_LEASE_OK;
    }();
  }
  if (!errors.finish()) result = MGPU_CONTEXT_LEASE_INTERNAL;
  return result;
}
int endOutputAccess(Output &o) {
  if (!o.beginAttempted || o.endAttempted) return MGPU_CONTEXT_LEASE_INTERNAL;
  PinnedErrorScopes errors(o.pin);
  WGPUSharedTextureMemoryEndAccessState ended{};
  WGPUStatus status;
  {
    mgpu::lock_guard<mgpu::mutex> gpu(o.pin->gpu().getGpuMutex());
    o.endAttempted = true;
    status = wgpuSharedTextureMemoryEndAccess(o.memory, o.texture, &ended);
  }
  const bool initialized = ended.initialized;
  wgpuSharedTextureMemoryEndAccessStateFreeMembers(ended);
  return errors.finish() && status == WGPUStatus_Success && initialized ?
      MGPU_CONTEXT_LEASE_OK : MGPU_CONTEXT_LEASE_INTERNAL;
}
int completeCopy(const std::shared_ptr<Writer> &w, const std::shared_ptr<Output> &o, int result) {
  try {
    if (result == MGPU_CONTEXT_LEASE_OK) result = endOutputAccess(*o);
  } catch (...) { result = MGPU_CONTEXT_LEASE_INTERNAL; }
  {
    std::lock_guard<std::mutex> writerLock(w->mutex);
    std::lock_guard<std::mutex> outputLock(o->mutex);
    w->copying = false;
    if (result == MGPU_CONTEXT_LEASE_OK) {
      w->phase = MGPU_PRIVATE_OUTPUT_READY;
      o->phase = MGPU_PRIVATE_OUTPUT_READY;
      w->result = o->result = result;
    } else if (!o->beginAttempted) {
      // Known refusal before touching GPU access: owner cleanup is still safe.
      w->phase = MGPU_PRIVATE_OUTPUT_READY;
      o->phase = MGPU_PRIVATE_OUTPUT_EMPTY;
      w->result = o->result = result;
    } else {
      quarantine(*w); quarantine(*o);
      result = MGPU_CONTEXT_LEASE_QUARANTINED;
    }
  }
  return result;
}
#endif
} // namespace mgpu::private_output

template<class T> struct PrivateOutputBox : mgpu::source_scope::MetadataUniqueRecord {
  PrivateOutputBox(mgpu::source_scope::MetadataUniqueRecord storage,
                   std::shared_ptr<T> held) noexcept
      : MetadataUniqueRecord(std::move(storage)), state(std::move(held)) {}
  std::shared_ptr<T> state;
};
struct MGPUPrivateOutputWriter : PrivateOutputBox<mgpu::private_output::Writer> { using PrivateOutputBox::PrivateOutputBox; };
struct MGPUPrivateOutput : PrivateOutputBox<mgpu::private_output::Output> { using PrivateOutputBox::PrivateOutputBox; };
struct MGPUPrivatePreviewLoan : PrivateOutputBox<mgpu::private_output::Loan> { using PrivateOutputBox::PrivateOutputBox; };
struct MGPUPrivatePreviewBorrow : PrivateOutputBox<mgpu::private_output::Borrow> { using PrivateOutputBox::PrivateOutputBox; };

using namespace mgpu::private_output;

extern "C" int mgpuPrivateOutputWriterCreate(
    MGPUContextLease *lease, MGPUBuffer *buffer, uint32_t width, uint32_t height,
    MGPUPrivateOutputWriter **out) {
  if (out) *out = nullptr;
  uint64_t bytes = 0;
  if (!out || !lease || !buffer || !imageBytes(width, height, bytes))
    return MGPU_CONTEXT_LEASE_INVALID;
#if !defined(_WIN32) || defined(__EMSCRIPTEN__)
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  std::shared_ptr<Writer> s;
  MetadataUnique<MGPUPrivateOutputWriter> box;
  bool accepted = false, charged = false;
  try {
    auto pin = mgpu::copyContextLeasePin(lease);
    if (!pin) return MGPU_CONTEXT_LEASE_INVALID;
    if (!pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
    if (std::this_thread::get_id() == pin->gpu().getWebGPUThread().threadId())
      return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    auto &input = *reinterpret_cast<mgpu::Buffer *>(buffer);
    WGPUBuffer raw = nullptr;
    std::shared_ptr<mgpu::PrivateResourceState> packed;
    {
      mgpu::lock_guard<mgpu::mutex> lock(pin->gpu().getGpuMutex());
      if (&input.getMGPU() != &pin->gpu() || !input.matchesPrivateSourcePin(*pin) ||
          input.bufferData.creationDevice != pin->device() ||
          input.bufferData.size != bytes || !input.bufferData.buffer)
        return MGPU_CONTEXT_LEASE_INVALID;
      packed = input.copyPrivateSourceState(*pin);
      if (pin->sourceDomain() && (!packed || packed->bytes != bytes))
        return MGPU_CONTEXT_LEASE_INVALID;
      const auto valid = mgpu::capture_output::validateWriter(pin->sourceDomain(),
          packed ? packed->resourceSource() : mgpu::source_scope::SourceDomain{},
          packed ? packed->sourceRole() : ResourceKind::Count, width, height, bytes,
          input.bufferData.usage);
      if (valid != Result::Ok) return nativeResult(valid);
      raw = input.bufferData.buffer; // Caller retains usability through return.
    }
    ResourceCustody role;
    auto status = role.reserveSourceRole(pin->sourceDomain(), ResourceKind::Writer);
    if (status != Result::Ok) return nativeResult(status);
    MetadataReservations resourceSlots;
    status = role.reserveResourceMetadata(resourceSlots);
    if (status != Result::Ok) return nativeResult(status);
    Family family;
    status = Family::beginWork(pin->sourceDomain(), family);
    if (status != Result::Ok) return nativeResult(status);
    std::shared_ptr<GlobalTaskCustody> global;
    PreparedWork job;
    int result = prepareJob(family, job, global);
    if (result != MGPU_CONTEXT_LEASE_OK) return result;
    s = makeMetadataShared<Writer>(resourceSlots, MetadataKind::WriterState, std::move(role));
    s->pin = pin; s->packedState = std::move(packed);
    s->width = width; s->height = height; s->bytes = bytes;
    box = makeMetadataUnique<MGPUPrivateOutputWriter>(resourceSlots, MetadataKind::WriterBox, s);
    if (!(charged = reserveWriter(s))) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    result = job.configureSynchronous([s, raw] {
      int result;
      try {
        result = createWriter(*s, raw);
        if (result != MGPU_CONTEXT_LEASE_OK) {
          releaseWriterNative(*s);
          s->proveNativeReleased();
        }
      } catch (...) { result = MGPU_CONTEXT_LEASE_QUARANTINED; }
      {
        std::lock_guard<std::mutex> lock(s->mutex);
        if (result == MGPU_CONTEXT_LEASE_QUARANTINED) quarantine(*s);
        else {
          s->result = result;
          s->phase = result == MGPU_CONTEXT_LEASE_OK ? MGPU_PRIVATE_OUTPUT_READY
                                                    : MGPU_PRIVATE_OUTPUT_RETIRED;
        }
      }
      return result == MGPU_CONTEXT_LEASE_QUARANTINED ? Outcome::uncertain()
                                                     : Outcome::done(result);
    });
    if (result == MGPU_CONTEXT_LEASE_OK) result = nativeResult(s->acceptSourceRole());
    if (result == MGPU_CONTEXT_LEASE_OK) result = job.enqueue(enqueueOutput, &pin->gpu());
    if (result != MGPU_CONTEXT_LEASE_OK) {
      s->proveNativeReleased();
      if (!removeWriter(s)) throw std::runtime_error("writer reservation mismatch");
      charged = false;
      return result;
    }
    accepted = true;
    result = job.wait(); // Actual caller/worker closure tails retain the job.
    if (result == MGPU_CONTEXT_LEASE_OK || result == MGPU_CONTEXT_LEASE_QUARANTINED)
      *out = box.release();
    else {
      if (!removeWriter(s)) throw std::runtime_error("writer disposition mismatch");
      charged = false;
    }
    return result;
  } catch (...) {
    try {
      if (!accepted) {
        if (s) s->proveNativeReleased(); // No queue/native work was accepted.
        if (charged && !removeWriter(s)) throw std::runtime_error("writer rollback mismatch");
        return MGPU_CONTEXT_LEASE_INTERNAL;
      }
    } catch (...) {} // A failed rollback cannot escape the C boundary.
    quarantineFactory(s);
    *out = box.release(); return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
#endif
}

extern "C" int mgpuPrivateOutputCreate(
    MGPUPrivateOutputWriter *writer, MGPUPrivateOutput **out) {
  if (out) *out = nullptr;
  if (!out || !writer) return MGPU_CONTEXT_LEASE_INVALID;
#if !defined(_WIN32) || defined(__EMSCRIPTEN__)
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  std::shared_ptr<Writer> w;
  std::shared_ptr<Output> s;
  MetadataUnique<MGPUPrivateOutput> box;
  bool accepted = false, reserved = false, counted = false;
  try {
    w = writer->state;
    if (std::this_thread::get_id() == w->pin->gpu().getWebGPUThread().threadId())
      return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    const auto valid = mgpu::capture_output::validateOutput(w->pin->sourceDomain(),
        w->resourceSource(), w->width, w->height, w->bytes);
    if (valid != Result::Ok) return nativeResult(valid);
    ResourceCustody role;
    auto status = role.reserveSourceRole(w->pin->sourceDomain(), ResourceKind::Output);
    if (status != Result::Ok) return nativeResult(status);
    MetadataReservations resourceSlots;
    status = role.reserveResourceMetadata(resourceSlots);
    if (status != Result::Ok) return nativeResult(status);
    Family family;
    status = Family::beginWork(w->pin->sourceDomain(), family);
    if (status != Result::Ok) return nativeResult(status);
    std::shared_ptr<GlobalTaskCustody> global;
    PreparedWork job;
    int result = prepareJob(family, job, global);
    if (result != MGPU_CONTEXT_LEASE_OK) return result;
    s = makeMetadataShared<Output>(resourceSlots, MetadataKind::OutputState, std::move(role));
    s->pin = w->pin; s->width = w->width; s->height = w->height; s->bytes = w->bytes;
    box = makeMetadataUnique<MGPUPrivateOutput>(resourceSlots, MetadataKind::OutputBox, s);
    result = job.configureSynchronous([w, s] {
      int result;
      try {
        result = createOutput(*s);
        if (result != MGPU_CONTEXT_LEASE_OK) {
          releaseOutputNative(*s);
          s->proveNativeReleased();
        }
      } catch (...) { result = MGPU_CONTEXT_LEASE_QUARANTINED; }
      {
        std::lock_guard<std::mutex> writerLock(w->mutex);
        std::lock_guard<std::mutex> outputLock(s->mutex);
        --w->creates;
        if (result == MGPU_CONTEXT_LEASE_QUARANTINED) quarantine(*s);
        else {
          s->result = result;
          s->phase = result == MGPU_CONTEXT_LEASE_OK ? MGPU_PRIVATE_OUTPUT_EMPTY
                                                    : MGPU_PRIVATE_OUTPUT_RETIRED;
        }
      }
      return result == MGPU_CONTEXT_LEASE_QUARANTINED ? Outcome::uncertain()
                                                     : Outcome::done(result);
    });
    if (result != MGPU_CONTEXT_LEASE_OK) return result;
    {
      std::lock_guard<std::mutex> lock(w->mutex);
      if (w->sealed || w->phase != MGPU_PRIVATE_OUTPUT_READY || !w->pin->current())
        return MGPU_CONTEXT_LEASE_RETIRED;
      if (!(reserved = reserveOutput(s))) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      result = nativeResult(s->acceptSourceRole());
      if (result == MGPU_CONTEXT_LEASE_OK) {
        ++w->creates; counted = true;
        result = job.enqueue(enqueueOutput, &s->pin->gpu());
      }
      if (result != MGPU_CONTEXT_LEASE_OK) {
        if (counted) { --w->creates; counted = false; }
        s->proveNativeReleased();
        std::lock_guard<std::mutex> outputLock(s->mutex);
        s->phase = MGPU_PRIVATE_OUTPUT_RETIRED; s->publicOwner = false;
        removeOutputIfUnowned(s); reserved = false;
        return result;
      }
    }
    accepted = true;
    result = job.wait();
    if (result == MGPU_CONTEXT_LEASE_OK || result == MGPU_CONTEXT_LEASE_QUARANTINED)
      *out = box.release();
    else {
      std::lock_guard<std::mutex> lock(s->mutex);
      s->publicOwner = false; removeOutputIfUnowned(s); reserved = false;
    }
    return result;
  } catch (...) {
    try {
      if (!accepted) {
        if (counted) { std::lock_guard<std::mutex> lock(w->mutex); --w->creates; }
        if (s) s->proveNativeReleased();
        if (reserved) {
          std::lock_guard<std::mutex> lock(s->mutex);
          s->phase = MGPU_PRIVATE_OUTPUT_RETIRED; s->publicOwner = false;
          removeOutputIfUnowned(s);
        }
        return MGPU_CONTEXT_LEASE_INTERNAL;
      }
    } catch (...) {} // Retain registry/native custody on uncertain rollback.
    quarantineFactory(s);
    *out = box.release(); return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
#endif
}

extern "C" void mgpuPrivateOutputWriteAsync(
    MGPUPrivateOutputWriter *writer, MGPUPrivateOutput *output,
    MGPUQueueCompletionCallback cb, void *userdata) {
  if (!cb) return;
  Family family;
  const auto prepared = mgpu::capture_cleanup::prepareWriteFamily(writer, output, family);
  if (prepared != MGPU_CONTEXT_LEASE_OK) { notify(cb, userdata, prepared); return; }
  mgpu::capture_cleanup::writeOutput(writer, output, cb, userdata, family);
}
namespace mgpu::capture_cleanup {
int prepareWriteFamily(MGPUPrivateOutputWriter *writer, MGPUPrivateOutput *output,
                       Family &family) noexcept {
  if (!writer || !output) return MGPU_CONTEXT_LEASE_OK;
#if !defined(_WIN32) || defined(__EMSCRIPTEN__)
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  try { return nativeResult(Family::beginWork(writer->state->pin->sourceDomain(), family)); }
  catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
#endif
}
void writeOutput(MGPUPrivateOutputWriter *writer, MGPUPrivateOutput *output,
    MGPUQueueCompletionCallback cb, void *userdata, const Family &family) {
  if (!cb) return;
  int result = MGPU_CONTEXT_LEASE_INVALID;
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
  bool accepted = false;
  std::shared_ptr<GlobalTaskCustody> global;
  PreparedWork job;
  std::shared_ptr<PreparedWait> wait;
  try {
    if (writer && output) {
      auto w = writer->state; auto o = output->state;
      auto status = [&]() -> int {
        if (!family.matches(w->pin->sourceDomain()) || (family && family.cleanup()))
          return MGPU_CONTEXT_LEASE_INVALID;
        if (w->sealed || o->sealed || w->phase == MGPU_PRIVATE_OUTPUT_QUARANTINED ||
            o->phase == MGPU_PRIVATE_OUTPUT_QUARANTINED) return MGPU_CONTEXT_LEASE_RETIRED;
        if (w->phase != MGPU_PRIVATE_OUTPUT_READY || w->copying || w->creates ||
            o->phase != MGPU_PRIVATE_OUTPUT_EMPTY) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
        if (!w->pin->current() || !o->pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
        if (w->packedState && (!w->packedState->ready.load() || w->packedState->quarantined.load()))
          return MGPU_CONTEXT_LEASE_RETIRED;
        if (!mgpu::samePrivateCreation(w->pin->creationIdentity(), o->pin->creationIdentity()) ||
            w->width != o->width || w->height != o->height) return MGPU_CONTEXT_LEASE_INVALID;
        return MGPU_CONTEXT_LEASE_OK;
      };
      {
        std::lock_guard<std::mutex> writerLock(w->mutex);
        std::lock_guard<std::mutex> outputLock(o->mutex);
        result = status();
      }
      if (result == MGPU_CONTEXT_LEASE_OK) {
        // Covered allocation/free and callable preparation run outside owners.
        auto complete = [w, o, cb, userdata, family](int completed) {
          const auto result = completeCopy(w, o, completed);
          mgpu::capture_output::completeWrite(family, result,
              [cb, userdata](int result) { cb(result, userdata); });
        };
        result = prepareJob(family, job, global);
        if (result == MGPU_CONTEXT_LEASE_OK) {
          result = PreparedWait::create(family, wait);
          if (result == MGPU_CONTEXT_LEASE_OK)
            result = preparePinnedSubmittedWorkWait(w->pin, family, complete, *wait, global);
        }
        if (result == MGPU_CONTEXT_LEASE_OK)
          result = job.configure([w, o, wait] {
            const int submitted = submitCopy(*w, *o);
            if (submitted != MGPU_CONTEXT_LEASE_OK)
              return o->beginAttempted ? Outcome::uncertain() : Outcome::done(submitted);
            // BeginAccess/submission already occurred. Nested enqueue refusal
            // is accepted uncertainty; never restore EMPTY or retry its write.
            return enqueuePinnedSubmittedWorkWait(w->pin, *wait) == MGPU_CONTEXT_LEASE_OK
                ? Outcome::done() : Outcome::uncertain();
          }, [complete](int completed) { if (completed != MGPU_CONTEXT_LEASE_OK) complete(completed); });
        if (result == MGPU_CONTEXT_LEASE_OK) {
          std::lock_guard<std::mutex> writerLock(w->mutex);
          std::lock_guard<std::mutex> outputLock(o->mutex);
          result = status(); // Revalidate after fallible preparation and reentry.
          if (result == MGPU_CONTEXT_LEASE_OK) {
            w->copying = true;
            w->phase = o->phase = MGPU_PRIVATE_OUTPUT_COPYING;
            result = job.enqueue(enqueueOutput, &w->pin->gpu());
            accepted = result == MGPU_CONTEXT_LEASE_OK;
            if (!accepted) {
              w->copying = false; w->phase = MGPU_PRIVATE_OUTPUT_READY;
              o->phase = MGPU_PRIVATE_OUTPUT_EMPTY;
            }
          }
        }
      }
    }
  } catch (...) { result = MGPU_CONTEXT_LEASE_INTERNAL; }
  if (!accepted) notify(cb, userdata, result);
#else
  notify(cb, userdata, writer && output ? MGPU_CONTEXT_LEASE_UNSUPPORTED : result);
#endif
}

} // namespace mgpu::capture_cleanup

namespace {
template<class H>
int seal(H *handle) {
  if (!handle) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    auto s = handle->state;
    std::lock_guard<std::mutex> lock(s->mutex);
    s->sealed = true;
    return s->phase == MGPU_PRIVATE_OUTPUT_QUARANTINED ?
        MGPU_CONTEXT_LEASE_QUARANTINED : MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
template<class H>
int prepareRetire(H *handle, Family &family) noexcept {
  if (!handle) return MGPU_CONTEXT_LEASE_OK;
#if !defined(_WIN32) || defined(__EMSCRIPTEN__)
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  try {
    auto s = handle->state;
    { std::lock_guard<std::mutex> lock(s->mutex); s->sealed = true; }
    return nativeResult(Family::begin(s->pin->sourceDomain(), family));
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
#endif
}
template<class H>
void retire(H *handle, MGPUQueueCompletionCallback cb, void *userdata, const Family &family) {
  if (!cb) return;
  if (!handle) { notify(cb, userdata, MGPU_CONTEXT_LEASE_INVALID); return; }
#if !defined(_WIN32) || defined(__EMSCRIPTEN__)
  notify(cb, userdata, MGPU_CONTEXT_LEASE_UNSUPPORTED);
#else
  int result = MGPU_CONTEXT_LEASE_INTERNAL;
  bool accepted = false;
  std::shared_ptr<GlobalTaskCustody> global;
  PreparedWork job;
  try {
    auto s = handle->state;
    using T = typename decltype(s)::element_type;
    auto status = [&]() -> int {
      if (!family.cleanup() || !family.matches(s->pin->sourceDomain()))
        return MGPU_CONTEXT_LEASE_INVALID;
      s->sealed = true; // Even refused preparation must fence new work.
      if (s->phase == MGPU_PRIVATE_OUTPUT_RETIRED) return MGPU_CONTEXT_LEASE_OK;
      if (s->phase == MGPU_PRIVATE_OUTPUT_QUARANTINED) return MGPU_CONTEXT_LEASE_QUARANTINED;
      bool busy = s->phase == MGPU_PRIVATE_OUTPUT_COPYING || s->phase == MGPU_PRIVATE_OUTPUT_RETIRING;
      if constexpr (std::is_same_v<T, Writer>) busy |= s->copying || s->creates;
      return busy ? MGPU_CONTEXT_LEASE_WOULD_BLOCK : MGPU_CONTEXT_LEASE_OK;
    };
    bool needsWork = false;
    {
      std::lock_guard<std::mutex> lock(s->mutex);
      result = status();
      needsWork = result == MGPU_CONTEXT_LEASE_OK && s->phase != MGPU_PRIVATE_OUTPUT_RETIRED;
    }
    if (needsWork) {
      result = prepareJob(family, job, global);
        if (result == MGPU_CONTEXT_LEASE_OK)
          result = job.configure([s] {
            try {
              if constexpr (std::is_same_v<T, Writer>) releaseWriterNative(*s);
              else releaseOutputNative(*s);
              std::lock_guard<std::mutex> lock(s->mutex);
              s->phase = MGPU_PRIVATE_OUTPUT_RETIRED;
              s->result = MGPU_CONTEXT_LEASE_OK;
              s->proveNativeReleased();
              return Outcome::done();
            } catch (...) {
              std::lock_guard<std::mutex> lock(s->mutex);
              quarantine(*s);
              return Outcome::uncertain();
            }
          }, [cb, userdata](int result) { cb(result, userdata); });
      if (result == MGPU_CONTEXT_LEASE_OK) {
        std::lock_guard<std::mutex> lock(s->mutex);
        result = status();
        if (result == MGPU_CONTEXT_LEASE_OK && s->phase != MGPU_PRIVATE_OUTPUT_RETIRED) {
          const auto previous = s->phase;
          s->phase = MGPU_PRIVATE_OUTPUT_RETIRING;
          result = job.enqueue(enqueueOutput, &s->pin->gpu());
          accepted = result == MGPU_CONTEXT_LEASE_OK;
          if (!accepted) s->phase = previous;
        }
      }
    }
  } catch (...) { result = MGPU_CONTEXT_LEASE_INTERNAL; }
  if (!accepted) notify(cb, userdata, result);
#endif
}

template<class H>
int getInfo(H *handle, MGPUPrivateOutputInfo *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  *out = {};
  if (!handle) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    auto s = handle->state;
    using T = typename decltype(s)::element_type;
    std::lock_guard<std::mutex> lock(s->mutex);
    if constexpr (std::is_same_v<T, Writer>) {
      *out = info(*s, MGPU_PRIVATE_OUTPUT_WRITER);
      out->pending_operations = s->creates + (s->copying ? 1u : 0u);
    } else {
      *out = info(*s, MGPU_PRIVATE_OUTPUT_SNAPSHOT);
      out->pending_operations = s->phase == MGPU_PRIVATE_OUTPUT_COPYING ? 1 : 0;
      out->pending_borrows = s->borrows;
      if (s->previewCreated) out->flags |= MGPU_PRIVATE_OUTPUT_PREVIEW_CREATED;
    }
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
} // namespace
extern "C" int mgpuPrivateOutputWriterSeal(MGPUPrivateOutputWriter *s) { return seal(s); }
extern "C" int mgpuPrivateOutputSeal(MGPUPrivateOutput *s) { return seal(s); }
extern "C" void mgpuPrivateOutputWriterRetireAsync(
    MGPUPrivateOutputWriter *s, MGPUQueueCompletionCallback cb, void *u) {
  if (!cb) return;
  Family family;
  const auto prepared = prepareRetire(s, family);
  if (prepared != MGPU_CONTEXT_LEASE_OK) { notify(cb, u, prepared); return; }
  retire(s, cb, u, family);
}
extern "C" void mgpuPrivateOutputRetireAsync(
    MGPUPrivateOutput *s, MGPUQueueCompletionCallback cb, void *u) {
  if (!cb) return;
  Family family;
  const auto prepared = prepareRetire(s, family);
  if (prepared != MGPU_CONTEXT_LEASE_OK) { notify(cb, u, prepared); return; }
  retire(s, cb, u, family);
}
namespace mgpu::capture_cleanup {
int prepareWriterFamily(MGPUPrivateOutputWriter *s, Family &family) noexcept { return prepareRetire(s, family); }
int prepareOutputFamily(MGPUPrivateOutput *s, Family &family) noexcept { return prepareRetire(s, family); }
void retireWriter(MGPUPrivateOutputWriter *s, MGPUQueueCompletionCallback cb, void *u, const Family &family) {
  retire(s, cb, u, family);
}
void retireOutput(MGPUPrivateOutput *s, MGPUQueueCompletionCallback cb, void *u, const Family &family) {
  retire(s, cb, u, family);
}
} // namespace mgpu::capture_cleanup
extern "C" int mgpuPrivateOutputWriterGetInfo(
    MGPUPrivateOutputWriter *s, MGPUPrivateOutputInfo *out, uint32_t size) {
  return getInfo(s, out, size);
}
extern "C" int mgpuPrivateOutputGetInfo(
    MGPUPrivateOutput *s, MGPUPrivateOutputInfo *out, uint32_t size) {
  return getInfo(s, out, size);
}
extern "C" int mgpuPrivateOutputWriterRelease(MGPUPrivateOutputWriter *handle) {
  if (!handle) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    auto s = handle->state;
    {
      std::lock_guard<std::mutex> lock(s->mutex);
      if (s->phase != MGPU_PRIVATE_OUTPUT_RETIRED || s->copying || s->creates)
        return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      if (!removeWriter(s)) return MGPU_CONTEXT_LEASE_INTERNAL;
    }
    MetadataDelete<MGPUPrivateOutputWriter>{}(handle);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
extern "C" int mgpuPrivateOutputRelease(MGPUPrivateOutput *handle) {
  if (!handle) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    auto s = handle->state;
    {
      std::lock_guard<std::mutex> lock(s->mutex);
      if (s->phase != MGPU_PRIVATE_OUTPUT_RETIRED) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      s->publicOwner = false;
      removeOutputIfUnowned(s);
    }
    MetadataDelete<MGPUPrivateOutput>{}(handle);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
extern "C" int mgpuPrivateOutputGetStats(MGPUPrivateOutputStats *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  *out = {};
  try {
    auto &r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    out->struct_size = sizeof(*out); out->version = 1;
    out->writers = r.writerCount; out->outputs = r.outputCount;
    out->preview_loans = r.loans; out->descriptor_borrows = r.borrows;
    out->pending_jobs = r.jobs;
    out->max_writers = maxWriters; out->max_outputs = maxOutputs;
    out->max_preview_loans = maxLoans; out->max_descriptor_borrows = maxBorrows;
    out->charged_bytes = r.bytes; out->max_bytes = maxBytes;
    out->max_frame_bytes = maxFrameBytes;
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}

extern "C" int mgpuPrivateOutputAcquirePreviewLoan(
    MGPUPrivateOutput *handle, MGPUPrivatePreviewLoan **out) {
  if (out) *out = nullptr;
  if (!handle || !out) return MGPU_CONTEXT_LEASE_INVALID;
#if !defined(_WIN32) || defined(__EMSCRIPTEN__)
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  auto s = handle->state;
  AliasSlot slot{s.get()};
  ResourceCustody role;
  mgpu::capture_output::SlotReservation quota;
  std::shared_ptr<Loan> loan;
  MetadataUnique<MGPUPrivatePreviewLoan> box;
  Loan::Root anchor;
  bool accepted = false;
  try {
    MetadataReservations resourceSlots;
    Result status;
    {
      std::lock_guard<std::mutex> lock(s->mutex);
      if (s->sealed || !s->pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
      if (s->phase != MGPU_PRIVATE_OUTPUT_READY || s->previewCreated ||
          !s->source || !s->sharedHandle || !s->endAttempted)
        return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      status = mgpu::capture_output::validateOutput(s->pin->sourceDomain(),
          s->resourceSource(), s->width, s->height, s->bytes);
      if (status == Result::Ok) status = role.reserveSourceRole(s->pin->sourceDomain(), ResourceKind::PreviewLoan);
      if (status == Result::Ok) status = quota.reserve(s->pin->sourceDomain(), reserveLoanSlot, refundLoanSlot, &slot);
      if (status != Result::Ok) return nativeResult(status);
    }
    status = role.reserveResourceMetadata(resourceSlots);
    if (status != Result::Ok) return nativeResult(status);
    loan = makeMetadataShared<Loan>(resourceSlots, MetadataKind::LoanState, std::move(role));
    loan->output = s; loan->id = slot.id;
    box = makeMetadataUnique<MGPUPrivatePreviewLoan>(resourceSlots, MetadataKind::LoanBox, loan);
    anchor = makeMetadataUnique<Loan::Anchor>(resourceSlots, MetadataKind::LoanAnchor, loan);
    {
      std::lock_guard<std::mutex> lock(s->mutex);
      if (s->sealed || !s->pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
      if (s->phase != MGPU_PRIVATE_OUTPUT_READY || s->previewCreated ||
          !s->source || !s->sharedHandle || !s->endAttempted)
        return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      status = loan->acceptAlias(anchor);
      if (status != Result::Ok) return nativeResult(status);
      quota.commit(); accepted = true; // Actual retention root precedes native refs.
      loan->registration = s->source;
      loan->sharedHandle = s->sharedHandle;
      s->previewCreated = s->previewActive = true;
      *out = box.release();
      return MGPU_CONTEXT_LEASE_OK;
    }
  } catch (...) {
    if (accepted) {
      loan->sealed = loan->quarantined = true;
      loan->quarantineAlias();
      *out = box.release(); return MGPU_CONTEXT_LEASE_QUARANTINED;
    }
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#endif
}
extern "C" int mgpuPrivatePreviewLoanGetInfo(
    MGPUPrivatePreviewLoan *handle, MGPUPrivateOutputInfo *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  *out = {};
  if (!handle) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    auto loan = handle->state;
    auto s = loan->output;
    std::lock_guard<std::mutex> lock(s->mutex);
    *out = info(*s, MGPU_PRIVATE_OUTPUT_PREVIEW);
    out->resource_id = loan->id; out->parent_output_id = s->id;
    out->flags = loan->sealed ? MGPU_PRIVATE_OUTPUT_SEALED : 0;
    out->phase = loan->quarantined ? MGPU_PRIVATE_OUTPUT_QUARANTINED :
                 loan->sealed ? MGPU_PRIVATE_OUTPUT_RETIRING : MGPU_PRIVATE_OUTPUT_READY;
    out->last_result = loan->quarantined ? MGPU_CONTEXT_LEASE_QUARANTINED : MGPU_CONTEXT_LEASE_OK;
    out->pending_borrows = s->borrows;
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
extern "C" int mgpuPrivatePreviewBorrowDescriptor(
    MGPUPrivatePreviewLoan *handle, MGPUPrivatePreviewBorrow **out,
    MGPUPrivatePreviewBorrowInfo *descriptor, uint32_t size) {
  if (out) *out = nullptr;
  if (!descriptor || size < sizeof(*descriptor)) return MGPU_CONTEXT_LEASE_INVALID;
  *descriptor = {};
  if (!handle || !out) return MGPU_CONTEXT_LEASE_INVALID;
#if !defined(_WIN32) || defined(__EMSCRIPTEN__)
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  auto loan = handle->state;
  auto s = loan->output;
  AliasSlot slot{s.get()};
  ResourceCustody role;
  mgpu::capture_output::SlotReservation quota;
  std::shared_ptr<Borrow> borrow;
  MetadataUnique<MGPUPrivatePreviewBorrow> box;
  Borrow::Root anchor;
  bool accepted = false;
  try {
    MetadataReservations resourceSlots;
    Result status;
    {
      std::lock_guard<std::mutex> lock(s->mutex);
      if (loan->sealed || loan->quarantined || loan->uncertainAlias() || !s->pin->current())
        return MGPU_CONTEXT_LEASE_RETIRED;
      if (!s->previewActive || !loan->registration || !loan->sharedHandle)
        return MGPU_CONTEXT_LEASE_INVALID;
      status = mgpu::capture_output::validateOutput(s->pin->sourceDomain(),
          loan->resourceSource(), s->width, s->height, s->bytes);
      if (status == Result::Ok) status = role.reserveSourceRole(s->pin->sourceDomain(), ResourceKind::DescriptorBorrow);
      if (status == Result::Ok) status = quota.reserve(s->pin->sourceDomain(), reserveBorrowSlot, refundBorrowSlot, &slot);
      if (status != Result::Ok) return nativeResult(status);
    }
    // The reserved borrow count protects registration removal while allocation
    // runs without the output gate. Rollback's quota destructor relocks it.
    status = role.reserveResourceMetadata(resourceSlots);
    if (status != Result::Ok) return nativeResult(status);
    borrow = makeMetadataShared<Borrow>(resourceSlots, MetadataKind::BorrowState, std::move(role)); borrow->loan = loan;
    box = makeMetadataUnique<MGPUPrivatePreviewBorrow>(resourceSlots, MetadataKind::BorrowBox, borrow);
    anchor = makeMetadataUnique<Borrow::Anchor>(resourceSlots, MetadataKind::BorrowAnchor, borrow);
    {
      std::lock_guard<std::mutex> lock(s->mutex);
      if (loan->sealed || loan->quarantined || loan->uncertainAlias() || !s->pin->current())
        return MGPU_CONTEXT_LEASE_RETIRED;
      if (!s->previewActive || !loan->registration || !loan->sharedHandle)
        return MGPU_CONTEXT_LEASE_INVALID;
      auto &value = borrow->info;
      value.struct_size = sizeof(value); value.version = 1;
      value.output_id = s->id; value.loan_id = loan->id; value.borrow_id = slot.id;
      value.width = s->width; value.height = s->height; value.pixel_format = 1;
      value.legacy_shared_handle = reinterpret_cast<uintptr_t>(loan->sharedHandle);
      value.device_generation = s->pin->deviceGeneration();
      status = borrow->acceptAlias(anchor);
      if (status != Result::Ok) return nativeResult(status);
      quota.commit(); accepted = true;
      borrow->reference = loan->registration;
      *descriptor = value; *out = box.release();
      return MGPU_CONTEXT_LEASE_OK;
    }
  } catch (...) {
    if (accepted) {
      borrow->quarantineAlias();
      *out = box.release(); return MGPU_CONTEXT_LEASE_QUARANTINED;
    }
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#endif
}
extern "C" int mgpuPrivatePreviewBorrowConsumed(MGPUPrivatePreviewBorrow *handle) {
  if (!handle) return MGPU_CONTEXT_LEASE_INVALID;
  auto borrow = handle->state;
  if (borrow->uncertainAlias()) return MGPU_CONTEXT_LEASE_QUARANTINED;
  try {
    auto loan = borrow->loan;
    auto s = loan->output;
#ifdef _WIN32
    borrow->reference.Reset(); // Actual COM release before any disposition.
#endif
    {
      std::lock_guard<std::mutex> lock(s->mutex);
      auto &r = registry(); std::lock_guard<std::mutex> registryLock(r.mutex);
      if (!s->borrows || !r.borrows) {
        borrow->quarantineAlias(); return MGPU_CONTEXT_LEASE_QUARANTINED;
      }
      --s->borrows; --r.borrows;
    }
    borrow->releasedAlias();
    MetadataDelete<MGPUPrivatePreviewBorrow>{}(handle);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) {
    borrow->quarantineAlias();
    return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
}
extern "C" int mgpuPrivatePreviewLoanSeal(MGPUPrivatePreviewLoan *handle) {
  if (!handle) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    auto loan = handle->state;
    std::lock_guard<std::mutex> lock(loan->output->mutex);
    loan->sealed = true;
    return loan->quarantined ? MGPU_CONTEXT_LEASE_QUARANTINED : MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
extern "C" int mgpuPrivatePreviewLoanQuarantine(MGPUPrivatePreviewLoan *handle) {
  if (!handle) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    auto loan = handle->state;
    std::lock_guard<std::mutex> lock(loan->output->mutex);
    loan->sealed = loan->quarantined = true;
    loan->quarantineAlias();
    return MGPU_CONTEXT_LEASE_QUARANTINED;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
extern "C" int mgpuPrivatePreviewLoanReleaseAfterRemoval(
    MGPUPrivatePreviewLoan *handle, uint64_t expected) {
  if (!handle || !expected) return MGPU_CONTEXT_LEASE_INVALID;
  auto loan = handle->state;
  auto s = loan->output;
  try {
    {
      std::lock_guard<std::mutex> lock(s->mutex);
      if (loan->id != expected) return MGPU_CONTEXT_LEASE_INVALID;
      if (loan->quarantined || loan->uncertainAlias()) return MGPU_CONTEXT_LEASE_QUARANTINED;
      if (!loan->sealed || s->borrows) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      // Caller exclusively owns final release and has independently confirmed
      // actual registration removal. IDs and this gate cannot prove that fact.
    }
#ifdef _WIN32
    loan->registration.Reset(); // Release native reference before refunding.
    loan->sharedHandle = nullptr;
#endif
    {
      std::lock_guard<std::mutex> lock(s->mutex);
      auto &r = registry();
      {
        std::lock_guard<std::mutex> registryLock(r.mutex);
        if (!s->previewActive || !r.loans) {
          loan->quarantineAlias(); return MGPU_CONTEXT_LEASE_QUARANTINED;
        }
        s->previewActive = false; --r.loans;
      }
      removeOutputIfUnowned(s);
    }
    loan->releasedAlias();
    MetadataDelete<MGPUPrivatePreviewLoan>{}(handle);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) {
    loan->quarantineAlias();
    return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
}

namespace mgpu::allocation_layout {
void outputMetadataLayout(MetadataLayoutBuilder &out) noexcept {
  out.shared<private_output::Writer>(MetadataKind::WriterState);
  out.unique<MGPUPrivateOutputWriter>(MetadataKind::WriterBox);
  out.shared<private_output::Output>(MetadataKind::OutputState);
  out.unique<MGPUPrivateOutput>(MetadataKind::OutputBox);
  out.shared<private_output::Loan>(MetadataKind::LoanState);
  out.unique<MGPUPrivatePreviewLoan>(MetadataKind::LoanBox);
  out.unique<private_output::Loan::Anchor>(MetadataKind::LoanAnchor);
  out.shared<private_output::Borrow>(MetadataKind::BorrowState);
  out.unique<MGPUPrivatePreviewBorrow>(MetadataKind::BorrowBox);
  out.unique<private_output::Borrow::Anchor>(MetadataKind::BorrowAnchor);
}
} // namespace mgpu::allocation_layout
