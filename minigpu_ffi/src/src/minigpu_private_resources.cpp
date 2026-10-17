#include "../include/minigpu_private_resources.h"
#include "../include/minigpu_metadata_layout.h"
#include "../include/minigpu_private_resource_owner.h"
#include "../include/minigpu_source_metadata_allocator.h"
#include "../include/minigpu_module_owner.h"
#include "../include/minigpu_inline_result.h"
#include "../include/minigpu_error_scope.h"
#include "../include/compute_shader.h"
#include <array>
#include <cstring>
#include <limits>
#include <mutex>

namespace mgpu::private_resources {
constexpr uint32_t maxShaders = 32, maxBuffers = 128, maxJobs = 32;
constexpr uint64_t maxBufferBytes = 128ull * 1024 * 1024, maxBytes = 512ull * 1024 * 1024;
constexpr uint32_t maxInitialBytes = 65536, maxWgslBytes = 65536, maxBindings = 4, maxSlots = 8;
struct Job;
struct Registry {
  std::mutex gate;
  std::array<std::shared_ptr<PrivateResourceState>, maxShaders + maxBuffers> resources{};
  std::array<std::shared_ptr<Job>, maxJobs> jobs{};
  uint32_t shaders = 0, buffers = 0, liveJobs = 0;
  uint64_t bytes = 0;
};
Registry &registry() { return mgpu::detail::moduleOwner<Registry>(); }
struct Job : capture_recipe::JobCustody {
  Job(source_scope::MetadataAllocation allocation, capture_recipe::JobCustody &&held,
      bool scoped) noexcept(false)
      : JobCustody(std::move(held), std::move(allocation)),
        initial(scoped), code(scoped), preparedBindings(scoped) {}
  std::shared_ptr<ContextPin> pin;
  std::shared_ptr<PrivateResourceState> resource;
  std::array<std::shared_ptr<PrivateResourceState>, maxBindings> dependencies{};
  std::array<MGPUPrivateBufferBindingV1, maxBindings> bindings{};
  capture_payload::Array<uint8_t, capture_recipe_v1::parameterBytes> initial;
  capture_payload::Kernel code;
  capture_payload::Array<BindingEntry, capture_payload::bindingSlots> preparedBindings;
  std::vector<BufferBinding> preparedLegacy;
  capture_work::InlineResult result;
  ComputeShader *shader = nullptr;
  Buffer *buffer = nullptr;
  uint64_t bytes = 0;
  uint32_t usage = 0, count = 0, frameSlot = 0, slot = 0;
  bool committed = false, setup = false;
  InlineCallable<int(Job &)> action;
  task_queue::Entry queue{this, [](void *p) noexcept { static_cast<Job *>(p)->run(); }};
  void run() noexcept;
};
bool refundGlobalJob() noexcept {
  try {
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
    if (!r.liveJobs) return false;
    --r.liveJobs;
    return true;
  } catch (...) { return false; } // Unknown refund remains charged.
}
int admissionResult(source_scope::Result result) noexcept {
  switch (result) {
  case source_scope::Result::Ok: return MGPU_CONTEXT_LEASE_OK;
  case source_scope::Result::Invalid: return MGPU_CONTEXT_LEASE_INVALID;
  case source_scope::Result::WouldBlock: return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  case source_scope::Result::Sealed:
  case source_scope::Result::Joined:
  case source_scope::Result::Poisoned: return MGPU_CONTEXT_LEASE_RETIRED;
  default: return MGPU_CONTEXT_LEASE_INTERNAL;
  }
  // A refused new admission owns no native partial output. QUARANTINED is
  // reserved for this call's uncertain accepted work with its custody marker.
}
bool reserveJob(const std::shared_ptr<Job> &job) {
  auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
  if (r.liveJobs == maxJobs) return false;
  for (uint32_t i = 0; i < maxJobs; ++i) if (!r.jobs[i]) {
    job->slot = i; ++r.liveJobs; r.jobs[i] = job;
    job->setGlobalJobRefund(refundGlobalJob); return true;
  }
  return false;
}
void unrootJob(const std::shared_ptr<Job> &job) {
  auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
  if (r.jobs[job->slot] == job) r.jobs[job->slot].reset();
}
bool reserveResource(const std::shared_ptr<PrivateResourceState> &s) {
  auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
  if ((s->kind == 1 ? r.shaders == maxShaders : r.buffers == maxBuffers) || s->bytes > maxBytes - r.bytes)
    return false;
  for (uint32_t i = 0; i < r.resources.size(); ++i) if (!r.resources[i]) {
    s->slot = i; r.resources[i] = s; r.bytes += s->bytes;
    if (s->kind == 1) ++r.shaders; else ++r.buffers;
    return true;
  }
  return false;
}
void quarantine(Job &job) noexcept {
  job.quarantineSourceJob();
  if (job.resource) {
    job.resource->quarantineSourceRole();
    job.resource->quarantined = true; job.resource->ready = false;
  }
  for (const auto &s : job.dependencies) if (s) {
    s->quarantineSourceRole(); s->quarantined = true; s->ready = false;
  }
}
void quarantine(const std::shared_ptr<Job> &job) noexcept { quarantine(*job); }
int preparedPin(MGPUContextLease *lease, std::shared_ptr<ContextPin> &pin) {
  pin = copyContextLeasePin(lease);
  if (!pin) return MGPU_CONTEXT_LEASE_INVALID;
  if (!pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
  if (std::this_thread::get_id() == pin->gpu().getWebGPUThread().threadId()) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  return MGPU_CONTEXT_LEASE_OK;
}
int scopedSetup(const Job &job, const char *code, size_t length) noexcept {
  if (!job.pin->sourceDomain()) return MGPU_CONTEXT_LEASE_OK;
  if (!job.resource || job.count != 2) return MGPU_CONTEXT_LEASE_INVALID;
  std::array<capture_recipe::Binding, 2> bindings;
  for (size_t i = 0; i < bindings.size(); ++i) {
    const auto &resource = job.dependencies[i];
    if (!resource || !samePrivatePin(*resource, *job.pin) ||
        !resource->ready.load() || resource->quarantined.load())
      return MGPU_CONTEXT_LEASE_INVALID;
    const auto &binding = job.bindings[i];
    bindings[i] = {binding.slot, binding.kind, resource->resourceSource(),
                    resource->sourceRole(), binding.offset, binding.size};
  }
  return admissionResult(capture_recipe::validateSetup(job.pin->sourceDomain(),
      job.resource->resourceSource(), job.resource->sourceRole(), code, length,
      job.frameSlot, bindings));
}
void Job::run() noexcept {
  int completed = MGPU_CONTEXT_LEASE_INTERNAL;
  try { completed = action(*this); }
  catch (...) {
    if (setup && !committed) completed = MGPU_CONTEXT_LEASE_INTERNAL;
    else { quarantine(*this); completed = MGPU_CONTEXT_LEASE_QUARANTINED; }
  }
  // All GPU/submission locks and error scopes have unwound first.
  try {
    if (result.publish(completed)) proveJobFinished();
    else quarantine(*this);
  } catch (...) { quarantine(*this); }
}
template<class Run>
int synchronous(const std::shared_ptr<Job> &job, Run run, bool &accepted) {
  job->action = std::move(run);
  task_queue::Prepared task;
  if (!task.prepare(job->queue, job)) return MGPU_CONTEXT_LEASE_INVALID;
  const auto admitted = job->acceptSourceJob();
  if (admitted != source_scope::Result::Ok) return admissionResult(admitted);
  const auto queued = job->pin->gpu().getWebGPUThread().enqueuePrepared(task);
  if (queued != MGPU_CONTEXT_LEASE_OK) { job->proveEnqueueRejected(); return queued; }
  accepted = true;
  return job->result.wait();
}
} // namespace mgpu::private_resources

namespace mgpu {
ComputeShader::ComputeShader(std::shared_ptr<PrivateResourceState> state)
    : ComputeShader(source_scope::MetadataUniqueRecord{}, std::move(state)) {}
ComputeShader::ComputeShader(source_scope::MetadataUniqueRecord storage,
                             std::shared_ptr<PrivateResourceState> state)
    : MetadataUniqueRecord(std::move(storage)), mgpu(state->pin->gpu()), lifetimePin_(state->pin), privateResource_(state),
      shaderCode(!!state->pin->sourceDomain()), bindings(!!state->pin->sourceDomain()), creationDevice(state->pin->device()) {
  wgpuDeviceAddRef(creationDevice);
}
Buffer::Buffer(std::shared_ptr<PrivateResourceState> state)
    : Buffer(source_scope::MetadataUniqueRecord{}, std::move(state)) {}
Buffer::Buffer(source_scope::MetadataUniqueRecord storage,
               std::shared_ptr<PrivateResourceState> state)
    : MetadataUniqueRecord(std::move(storage)), mgpu(state->pin->gpu()), privateResource_(state), dataType(kUInt32), elementCount(state->bytes / 4) {
  bufferData.size = static_cast<size_t>(state->bytes);
  bufferData.creationDevice = state->pin->device();
  wgpuDeviceAddRef(bufferData.creationDevice);
}
void releasePrivateResource(const std::shared_ptr<PrivateResourceState> &s) noexcept {
  try {
    if (!s->nativeReleased) {
      s->quarantineSourceRole(); s->quarantined = true; s->ready = false; return;
    }
    s->proveNativeReleased();
    auto &r = private_resources::registry(); std::lock_guard<std::mutex> lock(r.gate);
    if (r.resources[s->slot] != s) return;
    r.bytes -= s->bytes;
    if (s->kind == 1) --r.shaders; else --r.buffers;
    r.resources[s->slot].reset();
  } catch (...) {s->quarantineSourceRole(); s->quarantined = true;}
}
struct PrivateResourceAccess {
  static std::shared_ptr<PrivateResourceState> state(ComputeShader &s) {return s.privateResource_;}
  static std::shared_ptr<PrivateResourceState> state(Buffer &b) {return b.privateResource_;}
  static void discardCpu(private_resources::Job &j) {
    // No work was accepted. Constructors retained only an exact device ref.
    if (j.shader) {
      if (j.shader->creationDevice) wgpuDeviceRelease(j.shader->creationDevice);
      j.shader->creationDevice = nullptr;
      j.resource->nativeReleased = true; source_scope::MetadataDelete<ComputeShader>{}(j.shader); j.shader = nullptr;
    } else if (j.buffer) {
      if (j.buffer->bufferData.creationDevice) wgpuDeviceRelease(j.buffer->bufferData.creationDevice);
      j.buffer->bufferData.creationDevice = nullptr;
      j.resource->nativeReleased = true; source_scope::MetadataDelete<Buffer>{}(j.buffer); j.buffer = nullptr;
    } else if (j.resource) {
      j.resource->nativeReleased = true; releasePrivateResource(j.resource);
    }
  }
  static bool discardWorker(private_resources::Job &j) {
    if (j.shader) {
      if (!j.shader->releaseCheckedOnWorker(*j.pin)) return false;
    } else if (j.buffer) {
      if (!j.buffer->releaseCheckedOnWorker(*j.pin)) return false;
    }
    return true;
  }
  static int create(private_resources::Job &j) {
    int result = MGPU_CONTEXT_LEASE_RETIRED;
    if (j.pin->current() && capture_recipe::workStatus(j.pin->sourceDomain()) == source_scope::Result::Ok) {
      result = MGPU_CONTEXT_LEASE_OK;
      if (j.buffer) {
        bool valid = false;
        {
          PinnedErrorScopes errors(j.pin);
          {
            mgpu::lock_guard<mgpu::mutex> lock(j.pin->gpu().getGpuMutex());
            WGPULimits limits{};
            if (wgpuDeviceGetLimits(j.pin->device(), &limits) == WGPUStatus_Success && j.bytes <= limits.maxBufferSize) {
              WGPUBufferDescriptor desc{};
              desc.size = j.bytes; desc.usage = static_cast<WGPUBufferUsage>(j.usage);
              desc.mappedAtCreation = !j.initial.empty();
              j.buffer->bufferData.usage = desc.usage;
              j.buffer->bufferData.buffer = wgpuDeviceCreateBuffer(j.pin->device(), &desc);
              valid = !!j.buffer->bufferData.buffer;
              if (valid && !j.initial.empty()) {
                auto *mapped = wgpuBufferGetMappedRange(j.buffer->bufferData.buffer, 0, j.initial.size());
                valid = mapped != nullptr;
                if (mapped) std::memcpy(mapped, j.initial.data(), j.initial.size());
                wgpuBufferUnmap(j.buffer->bufferData.buffer);
              }
            }
          }
          valid = errors.finish() && valid; // Event pumping outside GPU locks.
        }
        if (!valid) result = MGPU_CONTEXT_LEASE_INTERNAL;
      }
      if (!j.pin->current() || capture_recipe::workStatus(j.pin->sourceDomain()) != source_scope::Result::Ok)
        result = MGPU_CONTEXT_LEASE_RETIRED;
    }
    if (result != MGPU_CONTEXT_LEASE_OK) {
      if (!discardWorker(j)) {j.resource->quarantined = true; return MGPU_CONTEXT_LEASE_QUARANTINED;}
      return result;
    }
    j.resource->ready = true;
    return MGPU_CONTEXT_LEASE_OK;
  }
  static int setup(private_resources::Job &j) {
    using namespace private_resources;
    const auto checked = scopedSetup(j, j.code.data(), j.code.size());
    if (checked != MGPU_CONTEXT_LEASE_OK) return checked;
    auto &s = *j.shader;
    mgpu::lock_guard<mgpu::mutex> gate(s.submissionMutex_);
    if (!s.privateResource_ || !samePrivatePin(*s.privateResource_, *j.pin)) return MGPU_CONTEXT_LEASE_INVALID;
    if (!j.pin->current() || capture_recipe::workStatus(j.pin->sourceDomain()) != source_scope::Result::Ok ||
        s.privateResource_->quarantined || !s.privateResource_->ready ||
        s.destroyRequested_ || s.interference_) return MGPU_CONTEXT_LEASE_RETIRED;
    if (s.exclusiveUse_ || s.privateResource_->configured || !s.shaderCode.empty()) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    if (s.shaderCode.scoped() != j.code.scoped() || s.bindings.scoped() != j.preparedBindings.scoped())
      return MGPU_CONTEXT_LEASE_INVALID;
    mgpu::lock_guard<mgpu::mutex> lock(j.pin->gpu().getGpuMutex());
    WGPULimits limits{};
    if (wgpuDeviceGetLimits(j.pin->device(), &limits) != WGPUStatus_Success) return MGPU_CONTEXT_LEASE_INTERNAL;
    if (j.frameSlot >= limits.maxBindingsPerBindGroup || j.frameSlot >= j.preparedBindings.size()) return MGPU_CONTEXT_LEASE_INVALID;
    uint32_t storage = 0, uniform = 0;
    for (uint32_t i = 0; i < j.count; ++i) {
      const auto &binding = j.bindings[i];
      const auto &held = j.dependencies[i];
      auto &buffer = *reinterpret_cast<Buffer *>(binding.buffer);
      if (!held || !samePrivatePin(*held, *j.pin) || !held->ready || held->quarantined ||
          binding.slot >= limits.maxBindingsPerBindGroup || binding.slot >= j.preparedBindings.size() || !buffer.bufferData.buffer ||
          buffer.bufferData.creationDevice != j.pin->device() || binding.offset > held->bytes ||
          binding.size > held->bytes - binding.offset) return MGPU_CONTEXT_LEASE_INVALID;
      auto &entry = j.preparedBindings[binding.slot];
      const bool isUniform = binding.kind == MGPU_PRIVATE_UNIFORM_BUFFER;
      const auto alignment = isUniform ? limits.minUniformBufferOffsetAlignment : limits.minStorageBufferOffsetAlignment;
      const auto maxSize = isUniform ? limits.maxUniformBufferBindingSize : limits.maxStorageBufferBindingSize;
      const auto usage = isUniform ? WGPUBufferUsage_Uniform : WGPUBufferUsage_Storage;
      if (!alignment || binding.offset % alignment || binding.size > maxSize || !(buffer.bufferData.usage & usage))
        return MGPU_CONTEXT_LEASE_INVALID;
      if (isUniform) {++uniform; entry.kind = BindingKind::kUniformBuffer;}
      else {++storage; entry.kind = binding.kind == MGPU_PRIVATE_STORAGE_BUFFER ? BindingKind::kStorageBuffer : BindingKind::kReadOnlyStorageBuffer;}
      entry.buffer = buffer.bufferData.buffer; entry.size = static_cast<size_t>(binding.size);
      entry.offset = static_cast<size_t>(binding.offset);
    }
    if (storage > limits.maxStorageBuffersPerShaderStage || uniform > limits.maxUniformBuffersPerShaderStage)
      return MGPU_CONTEXT_LEASE_INVALID;
    if (!j.pin->current() || capture_recipe::workStatus(j.pin->sourceDomain()) != source_scope::Result::Ok)
      return MGPU_CONTEXT_LEASE_RETIRED;
    s.shaderCode.swap(j.code); s.bindings.swap(j.preparedBindings); s.buffers.swap(j.preparedLegacy);
    s.pipelineDirty = true; s.bindingsDirty = true;
    if (j.pin->sourceDomain()) s.privateResource_->bindingHolds = j.dependencies;
    j.committed = true; s.privateResource_->configured = true;
    return j.pin->current() ? MGPU_CONTEXT_LEASE_OK : MGPU_CONTEXT_LEASE_QUARANTINED;
  }
};
} // namespace mgpu
using namespace mgpu;
using namespace mgpu::private_resources;

template<class Publish>
static int createPrivate(MGPUContextLease *lease, uint64_t bytes, uint32_t usage,
                         const void *initial, uint32_t initialSize, Publish publish) noexcept {
#if !defined(_WIN32) || defined(__EMSCRIPTEN__)
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  std::shared_ptr<Job> job;
  bool accepted = false;
  try {
    std::shared_ptr<ContextPin> pin;
    const int prepared = preparedPin(lease, pin);
    if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared;
    source_scope::ResourceKind role = source_scope::ResourceKind::Count;
    if (pin->sourceDomain()) {
      const auto checked = capture_recipe::classifyResource(pin->sourceDomain(),
          bytes, usage, initial, initialSize, role);
      if (checked != source_scope::Result::Ok) return admissionResult(checked);
    }
    capture_recipe::JobCustody held;
    const auto admitted = held.prepareSourceJob(pin->sourceDomain());
    if (admitted != source_scope::Result::Ok) return admissionResult(admitted);
    source_scope::MetadataReservations metadataSlots;
    const auto covered = held.reserveJobMetadata(metadataSlots);
    if (covered != source_scope::Result::Ok) return admissionResult(covered);
    job = source_scope::makeMetadataShared<Job>(metadataSlots, source_scope::MetadataKind::PrimaryJob, std::move(held), !!pin->sourceDomain());
    job->pin = std::move(pin);
    if (!reserveJob(job)) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    job->bytes = bytes; job->usage = usage;
    capture_recipe::ResourceCustody resourceRole;
    const auto reserved = resourceRole.reserveSourceRole(job->pin->sourceDomain(), role);
    if (reserved != source_scope::Result::Ok) {unrootJob(job); return admissionResult(reserved);}
    source_scope::MetadataReservations resourceSlots;
    const auto resourceCovered = resourceRole.reserveResourceMetadata(resourceSlots);
    if (resourceCovered != source_scope::Result::Ok) {unrootJob(job); return admissionResult(resourceCovered);}
    job->resource = source_scope::makeMetadataShared<PrivateResourceState>(resourceSlots,
        source_scope::MetadataKind::PrivateResourceState, std::move(resourceRole));
    job->resource->pin = job->pin; job->resource->bytes = bytes; job->resource->kind = bytes ? 2u : 1u;
    if (!reserveResource(job->resource)) {unrootJob(job); return MGPU_CONTEXT_LEASE_WOULD_BLOCK;}
    // Both source and global reservations precede all dynamic payload copies.
    if (initialSize) job->initial.assign(static_cast<const uint8_t *>(initial), static_cast<const uint8_t *>(initial) + initialSize);
    if (job->pin->sourceDomain()) {
      source_scope::ResourceKind copiedRole = source_scope::ResourceKind::Count;
      const auto checked = capture_recipe::classifyResource(job->pin->sourceDomain(),
          bytes, usage, job->initial.empty() ? nullptr : job->initial.data(), initialSize, copiedRole);
      if (checked != source_scope::Result::Ok || copiedRole != role) {
        PrivateResourceAccess::discardCpu(*job); unrootJob(job);
        return checked == source_scope::Result::Ok ? MGPU_CONTEXT_LEASE_INVALID : admissionResult(checked);
      }
    }
    const auto entered = job->resource->acceptSourceRole();
    if (entered != source_scope::Result::Ok) {
      PrivateResourceAccess::discardCpu(*job); unrootJob(job); return admissionResult(entered);
    }
    if (bytes) job->buffer = source_scope::makeMetadataUnique<Buffer>(resourceSlots,
        source_scope::MetadataKind::BufferCpu, job->resource).release();
    else job->shader = source_scope::makeMetadataUnique<ComputeShader>(resourceSlots,
        source_scope::MetadataKind::ShaderCpu, job->resource).release();
    int result = synchronous(job, PrivateResourceAccess::create, accepted);
    if (result == MGPU_CONTEXT_LEASE_OK && !job->pin->current()) result = MGPU_CONTEXT_LEASE_QUARANTINED;
    if (result == MGPU_CONTEXT_LEASE_QUARANTINED) {quarantine(job); publish(*job); return result;}
    if (result == MGPU_CONTEXT_LEASE_OK) publish(*job);
    unrootJob(job);
    if (result != MGPU_CONTEXT_LEASE_OK) PrivateResourceAccess::discardCpu(*job);
    return result;
  } catch (...) {
    if (accepted && job) {quarantine(job); publish(*job); return MGPU_CONTEXT_LEASE_QUARANTINED;}
    if (job) {
      try {PrivateResourceAccess::discardCpu(*job); unrootJob(job);}
      catch (...) {quarantine(job); publish(*job); return MGPU_CONTEXT_LEASE_QUARANTINED;}
    }
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#endif
}
extern "C" int mgpuContextLeaseCreateComputeShaderChecked(MGPUContextLease *lease, MGPUComputeShader **out) {
  if (out) *out = nullptr;
  if (!lease || !out) return MGPU_CONTEXT_LEASE_INVALID;
  return createPrivate(lease, 0, 0, nullptr, 0, [out](Job &j) {*out = reinterpret_cast<MGPUComputeShader *>(j.shader);});
}
extern "C" int mgpuContextLeaseCreateBufferChecked(MGPUContextLease *lease, const MGPUPrivateBufferDescV1 *desc,
    uint32_t size, MGPUBuffer **out) {
  if (out) *out = nullptr;
  if (!lease || !out || !desc || size < sizeof(*desc)) return MGPU_CONTEXT_LEASE_INVALID;
  if (desc->struct_size != sizeof(*desc) || desc->version != 1 || desc->reserved || desc->byte_size < 4 ||
      desc->byte_size > maxBufferBytes || (desc->byte_size & 3) || (desc->usage & ~0xccu) ||
      ((desc->usage & 0xc0u) != 0x40u && (desc->usage & 0xc0u) != 0x80u) ||
      desc->initial_size > maxInitialBytes || desc->initial_size > desc->byte_size || (desc->initial_size & 3) ||
      (desc->initial_size == 0) != (desc->initial_data == nullptr)) return MGPU_CONTEXT_LEASE_INVALID;
  return createPrivate(lease, desc->byte_size, desc->usage, desc->initial_data, desc->initial_size,
      [out](Job &j) {*out = reinterpret_cast<MGPUBuffer *>(j.buffer);});
}
extern "C" int mgpuComputeShaderSetupChecked(MGPUContextLease *lease, MGPUComputeShader *shader,
    const MGPUPrivateShaderSetupV1 *desc, uint32_t size) {
  if (!lease || !shader || !desc || size < sizeof(*desc)) return MGPU_CONTEXT_LEASE_INVALID;
  if (desc->struct_size != sizeof(*desc) || desc->version != 1 || desc->reserved || !desc->wgsl || !desc->wgsl_size ||
      desc->wgsl_size > maxWgslBytes || desc->binding_count > maxBindings ||
      (desc->binding_count == 0) != (desc->bindings == nullptr) || desc->frame_slot >= maxSlots)
    return MGPU_CONTEXT_LEASE_INVALID;
#if !defined(_WIN32) || defined(__EMSCRIPTEN__)
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  std::shared_ptr<Job> job;
  bool accepted = false;
  try {
    std::shared_ptr<ContextPin> pin;
    const int prepared = preparedPin(lease, pin);
    if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared;
    capture_recipe::JobCustody held;
    const auto admitted = held.prepareSourceJob(pin->sourceDomain());
    if (admitted != source_scope::Result::Ok) return admissionResult(admitted);
    source_scope::MetadataReservations metadataSlots;
    const auto covered = held.reserveJobMetadata(metadataSlots);
    if (covered != source_scope::Result::Ok) return admissionResult(covered);
    job = source_scope::makeMetadataShared<Job>(metadataSlots, source_scope::MetadataKind::PrimaryJob, std::move(held), !!pin->sourceDomain());
    job->pin = std::move(pin);
    job->setup = true;
    job->shader = reinterpret_cast<ComputeShader *>(shader);
    job->resource = PrivateResourceAccess::state(*job->shader);
    if (!job->resource || !samePrivatePin(*job->resource, *job->pin) || job->resource->quarantined || !job->resource->ready)
      return MGPU_CONTEXT_LEASE_INVALID;
    job->count = desc->binding_count; job->frameSlot = desc->frame_slot;
    uint32_t slots = 1u << desc->frame_slot;
    for (uint32_t i = 0; i < job->count; ++i) {
      const auto b = desc->bindings[i];
      if (!b.buffer || b.slot >= maxSlots || (slots & (1u << b.slot)) || b.kind < 1 || b.kind > 3 ||
          !b.size || (b.offset & 3) || (b.size & 3)) return MGPU_CONTEXT_LEASE_INVALID;
      slots |= 1u << b.slot; job->bindings[i] = b;
      job->dependencies[i] = PrivateResourceAccess::state(*reinterpret_cast<Buffer *>(b.buffer));
      if (!job->dependencies[i] || !samePrivatePin(*job->dependencies[i], *job->pin)) return MGPU_CONTEXT_LEASE_INVALID;
    }
    const auto checked = scopedSetup(*job, desc->wgsl, desc->wgsl_size);
    if (checked != MGPU_CONTEXT_LEASE_OK) return checked;
    if (!reserveJob(job)) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    if (!job->code.assign(desc->wgsl, desc->wgsl_size)) {unrootJob(job); return MGPU_CONTEXT_LEASE_INVALID;}
    if (job->code.find('\0') != std::string::npos) {unrootJob(job); return MGPU_CONTEXT_LEASE_INVALID;}
    const auto copied = scopedSetup(*job, job->code.data(), job->code.size());
    if (copied != MGPU_CONTEXT_LEASE_OK) {unrootJob(job); return copied;}
    const bool scoped = !!job->pin->sourceDomain();
    job->preparedBindings.resize(scoped ? capture_payload::bindingSlots : maxSlots);
    if (!scoped) job->preparedLegacy.resize(maxSlots);
    const int result = synchronous(job, PrivateResourceAccess::setup, accepted);
    if (result == MGPU_CONTEXT_LEASE_QUARANTINED || (job->committed && !job->pin->current())) {
      quarantine(job); return MGPU_CONTEXT_LEASE_QUARANTINED;
    }
    unrootJob(job); return result;
  } catch (...) {
    if (job && accepted) {quarantine(job); return MGPU_CONTEXT_LEASE_QUARANTINED;}
    if (job) {try {unrootJob(job);} catch (...) {quarantine(job); return MGPU_CONTEXT_LEASE_QUARANTINED;}}
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#endif
}

namespace mgpu::allocation_layout {
void privateResourceMetadataLayout(MetadataLayoutBuilder &out) noexcept {
  out.shared<private_resources::Job>(MetadataKind::PrimaryJob, MGPU_CAPTURE_META_VARIANT_PRIVATE);
  out.shared<PrivateResourceState>(MetadataKind::PrivateResourceState);
  out.unique<Buffer>(MetadataKind::BufferCpu);
  out.unique<ComputeShader>(MetadataKind::ShaderCpu);
}
} // namespace mgpu::allocation_layout
