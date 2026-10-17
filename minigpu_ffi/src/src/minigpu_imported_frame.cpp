#include "../include/minigpu_imported_frame_owner.h"
#include "../include/minigpu_metadata_layout.h"
#include "../include/minigpu_module_owner.h"
#include "../include/minigpu_error_scope.h"
#include "../include/minigpu_capture_work_owner.h"
#include "../include/minigpu_capture_cleanup_adapter.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <dawn/native/D3D11Backend.h>
#endif

struct MGPUImportedFrame : mgpu::source_scope::MetadataUniqueRecord {
  explicit MGPUImportedFrame(mgpu::source_scope::MetadataUniqueRecord storage) noexcept
      : MetadataUniqueRecord(std::move(storage)) {}
  std::shared_ptr<mgpu::ImportedFrameState> state;
};

namespace mgpu {
struct ImportedFrameNative : source_scope::MetadataUniqueRecord {
  explicit ImportedFrameNative(source_scope::MetadataUniqueRecord storage) noexcept
      : MetadataUniqueRecord(std::move(storage)) {}
#ifdef _WIN32
  Microsoft::WRL::ComPtr<ID3D11Texture2D> source;
  WGPUSharedTextureMemory memory = nullptr;
  WGPUTexture texture = nullptr;
  bool beginAttempted = false;
  bool endAttempted = false;
#endif
};
ImportedFrameState::ImportedFrameState(source_scope::MetadataAllocation allocation,
    capture_recipe::ResourceCustody &&role, source_scope::MetadataReservations &slots)
    : capture_recipe::ResourceCustody(std::move(role), std::move(allocation)),
      native(source_scope::makeMetadataUnique<ImportedFrameNative>(slots,
          source_scope::MetadataKind::ImportedFrameNative)) {}
ImportedFrameState::~ImportedFrameState() = default;

namespace {
constexpr uint32_t maxFrames = 256;
constexpr uint64_t maxFrameBytes = 128ull * 1024 * 1024;
constexpr uint64_t maxRetainedBytes = 512ull * 1024 * 1024;
struct FrameRegistry {
  std::mutex mutex;
  std::array<std::shared_ptr<ImportedFrameState>, maxFrames> owners{};
  uint32_t count = 0;
  uint64_t bytes = 0;
};
FrameRegistry &registry() {
  // Process-rooted retention: module/process teardown must not reinterpret a
  // quarantined native owner as successfully retired. Fixed slot count.
  return mgpu::detail::moduleOwner<FrameRegistry>();
}
bool validProfile(const MGPUExternalVideoBuffer &d, uint64_t &bytes) {
  const auto &p = d.planes[0];
  if (d.content_type != MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE ||
      d.pixel_format != MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32 || d.num_planes != 1 ||
      !d.width || !d.height || !p.data_ptr ||
      p.data_ptr == reinterpret_cast<void *>(intptr_t(-1)) ||
      p.width != d.width || p.height != d.height || p.stride_bytes ||
      p.offset_bytes || p.subresource_index ||
      (p.dmabuf_fd != -1 && p.dmabuf_fd != 0) ||
      p.drm_format_modifier != MGPU_D3D11_IMMUTABLE_READY_TAG ||
      (d.fence.sync_fd != 0 && d.fence.sync_fd != -1) ||
      d.fence.d3d11_fence || d.fence.metal_shared_event || d.fence.metal_fence_value)
    return false;
  // Bounds before multiplication prevent overflow even for uint32 dimensions.
  if (d.width > maxFrameBytes / 4 ||
      d.height > maxFrameBytes / (uint64_t(d.width) * 4)) return false;
  bytes = uint64_t(d.width) * d.height * 4;
  return bytes && bytes <= maxFrameBytes;
}
std::shared_ptr<ImportedFrameState> reserveFrame(
    std::shared_ptr<ContextPin> pin, const MGPUExternalVideoBuffer &d,
    uint64_t bytes, capture_recipe::ResourceCustody &&role, source_scope::MetadataReservations &slots) {
  auto state = source_scope::makeMetadataShared<ImportedFrameState>(slots,
      source_scope::MetadataKind::ImportedFrameState, std::move(role), slots);
  state->pin = std::move(pin);
  state->width = d.width;
  state->height = d.height;
  state->bytes = bytes;
  auto &r = registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  if (r.count >= maxFrames || bytes > maxRetainedBytes - r.bytes) return {};
  for (uint32_t slot = 0; slot < maxFrames; ++slot) {
    if (!r.owners[slot]) {
      state->registrySlot = slot;
      r.owners[slot] = state;
      ++r.count;
      r.bytes += bytes;
      return state;
    }
  }
  return {};
}
bool removeFrameCharge(const std::shared_ptr<ImportedFrameState> &state) {
  auto &r = registry();
  std::shared_ptr<ImportedFrameState> retired;
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    if (state->registrySlot >= maxFrames ||
        r.owners[state->registrySlot] != state) return false;
    retired = std::move(r.owners[state->registrySlot]);
    --r.count;
    r.bytes -= state->bytes;
  } // Final State/member/control-block destruction must run outside the gate.
  return true;
}

#ifdef _WIN32
// Dawn can return error objects rather than null. Scope receipts contain no
// foreign callbacks and remain on this worker's stack until every pop answers.
// A stalled backend remains pending with its charged registry owner retained.
using ErrorScopes = mgpu::PinnedErrorScopes;

void releaseNativeReferences(const std::shared_ptr<ImportedFrameState> &s) {
  // Creation worker only, after known no-access failure or checked retirement.
  mgpu::lock_guard<mgpu::mutex> gpu(s->pin->gpu().getGpuMutex());
  if (s->view) wgpuTextureViewRelease(s->view);
  s->view = nullptr;
  if (s->native->texture) wgpuTextureRelease(s->native->texture);
  s->native->texture = nullptr;
  if (s->native->memory) wgpuSharedTextureMemoryRelease(s->native->memory);
  s->native->memory = nullptr;
  s->native->source.Reset();
}

int importOnWorker(const std::shared_ptr<ImportedFrameState> &s,
                   void *sharedHandle) {
  const auto &pin = s->pin;
  if (!pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
  WGPUAdapterInfo adapter{};
  const auto adapterResult = wgpuDeviceGetAdapterInfo(pin->device(), &adapter);
  const bool d3d11 = adapterResult == WGPUStatus_Success &&
                     adapter.backendType == WGPUBackendType_D3D11;
  wgpuAdapterInfoFreeMembers(adapter);
  if (!d3d11 || !wgpuDeviceHasFeature(
      pin->device(), WGPUFeatureName_SharedTextureMemoryD3D11Texture2D))
    return MGPU_CONTEXT_LEASE_UNSUPPORTED;
  WGPULimits limits{};
  if (wgpuDeviceGetLimits(pin->device(), &limits) != WGPUStatus_Success)
    return MGPU_CONTEXT_LEASE_INTERNAL;
  if (s->width > limits.maxTextureDimension2D ||
      s->height > limits.maxTextureDimension2D) return MGPU_CONTEXT_LEASE_INVALID;
  using Microsoft::WRL::ComPtr;
  auto device = dawn::native::d3d11::GetD3D11Device(pin->device());
  ComPtr<ID3D11Device1> device1;
  if (!device || FAILED(device.As(&device1))) return MGPU_CONTEXT_LEASE_UNSUPPORTED;
  if (FAILED(device1->OpenSharedResource1(
      static_cast<HANDLE>(sharedHandle), IID_PPV_ARGS(&s->native->source))) ||
      !s->native->source)
    return MGPU_CONTEXT_LEASE_INVALID;
  D3D11_TEXTURE2D_DESC actual{};
  s->native->source->GetDesc(&actual);
  constexpr UINT sharing = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                            D3D11_RESOURCE_MISC_SHARED;
  ComPtr<ID3D11Device> actualDevice;
  ComPtr<IUnknown> expectedIdentity, actualIdentity;
  s->native->source->GetDevice(&actualDevice);
  const bool sameDevice = actualDevice && SUCCEEDED(device.As(&expectedIdentity)) &&
      SUCCEEDED(actualDevice.As(&actualIdentity)) &&
      expectedIdentity.Get() == actualIdentity.Get();
  if (!sameDevice || actual.Format != DXGI_FORMAT_B8G8R8A8_UNORM ||
      actual.Width != s->width || actual.Height != s->height ||
      actual.ArraySize != 1 || actual.MipLevels != 1 ||
      actual.SampleDesc.Count != 1 || actual.SampleDesc.Quality != 0 ||
      actual.Usage != D3D11_USAGE_DEFAULT || actual.CPUAccessFlags != 0 ||
      actual.BindFlags != D3D11_BIND_SHADER_RESOURCE || actual.MiscFlags != sharing)
    return MGPU_CONTEXT_LEASE_INVALID;

  int result = MGPU_CONTEXT_LEASE_OK;
  ErrorScopes objects(pin);
  {
    mgpu::lock_guard<mgpu::mutex> gpu(pin->gpu().getGpuMutex());
    result = [&] {
      if (!pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
      dawn::native::d3d11::SharedTextureMemoryD3D11Texture2DDescriptor chain;
      chain.texture = s->native->source;
      WGPUSharedTextureMemoryDescriptor descriptor{};
      descriptor.nextInChain = const_cast<WGPUChainedStruct *>(
          reinterpret_cast<const WGPUChainedStruct *>(
              static_cast<const wgpu::ChainedStruct *>(&chain)));
      s->native->memory = wgpuDeviceImportSharedTextureMemory(pin->device(), &descriptor);
      if (!s->native->memory) return MGPU_CONTEXT_LEASE_INTERNAL;
      WGPUSharedTextureMemoryProperties props{};
      if (wgpuSharedTextureMemoryGetProperties(s->native->memory, &props) !=
          WGPUStatus_Success || props.format != WGPUTextureFormat_BGRA8Unorm ||
          props.size.width != s->width || props.size.height != s->height ||
          props.size.depthOrArrayLayers != 1 ||
          !(props.usage & WGPUTextureUsage_TextureBinding))
        return MGPU_CONTEXT_LEASE_INVALID;
      WGPUTextureDescriptor texture{};
      texture.usage = WGPUTextureUsage_TextureBinding;
      texture.dimension = WGPUTextureDimension_2D;
      texture.size = props.size;
      texture.format = props.format;
      texture.mipLevelCount = texture.sampleCount = 1;
      s->native->texture = wgpuSharedTextureMemoryCreateTexture(s->native->memory, &texture);
      if (!s->native->texture) return MGPU_CONTEXT_LEASE_INTERNAL;
      WGPUTextureViewDescriptor view{};
      view.dimension = WGPUTextureViewDimension_2D;
      view.format = WGPUTextureFormat_BGRA8Unorm;
      view.mipLevelCount = view.arrayLayerCount = 1;
      view.aspect = WGPUTextureAspect_All;
      s->view = wgpuTextureCreateView(s->native->texture, &view);
      return s->view ? MGPU_CONTEXT_LEASE_OK : MGPU_CONTEXT_LEASE_INTERNAL;
    }();
  }
  // Pop/pump outside GPU locks. No BeginAccess until every object is verified.
  if (!objects.finish()) result = MGPU_CONTEXT_LEASE_INTERNAL;
  if (result != MGPU_CONTEXT_LEASE_OK) return result;
  if (!pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
  ErrorScopes access(pin);
  WGPUStatus began;
  {
    mgpu::lock_guard<mgpu::mutex> gpu(pin->gpu().getGpuMutex());
    WGPUSharedTextureMemoryBeginAccessDescriptor begin{};
    begin.initialized = true;
    begin.concurrentRead = false;
    s->native->beginAttempted = true; // Error can follow native state changes.
    began = wgpuSharedTextureMemoryBeginAccess(
        s->native->memory, s->native->texture, &begin);
  }
  const bool clean = access.finish();
  if (began != WGPUStatus_Success || !clean || !pin->current()) {
    quarantineImportedFrame(s);
    return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
  return MGPU_CONTEXT_LEASE_OK;
}
#endif
} // namespace

std::shared_ptr<ImportedFrameState> copyImportedFrameState(MGPUImportedFrame *f) {
  return f ? f->state : nullptr;
}
bool sameFrameCreation(const ContextPin &a, const ContextPin &b) {
  return sameRetainedCreation(a.creationIdentity(), b.creationIdentity());
}
void quarantineImportedFrame(const std::shared_ptr<ImportedFrameState> &s) noexcept {
  if (!s) return;
  std::lock_guard<std::mutex> lock(s->mutex);
  s->phase = MGPU_IMPORTED_FRAME_QUARANTINED;
  s->result = MGPU_CONTEXT_LEASE_QUARANTINED;
  s->quarantineSourceRole();
}
int finishImportedFrameRetirement(const std::shared_ptr<ImportedFrameState> &s) {
  if (!s) return MGPU_CONTEXT_LEASE_INVALID;
#ifdef _WIN32
  {
    std::lock_guard<std::mutex> lock(s->mutex);
    if (s->phase != MGPU_IMPORTED_FRAME_RETIRING ||
        s->native->endAttempted || !s->native->beginAttempted ||
        std::this_thread::get_id() != s->pin->gpu().getWebGPUThread().threadId())
      return MGPU_CONTEXT_LEASE_INVALID;
  }
  if (!s->pin->current()) {
    quarantineImportedFrame(s);
    return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
  ErrorScopes errors(s->pin);
  WGPUStatus ended;
  {
    mgpu::lock_guard<mgpu::mutex> gpu(s->pin->gpu().getGpuMutex());
    s->native->endAttempted = true;
    WGPUSharedTextureMemoryEndAccessState end{};
    ended = wgpuSharedTextureMemoryEndAccess(s->native->memory, s->native->texture, &end);
    wgpuSharedTextureMemoryEndAccessStateFreeMembers(end);
  }
  const bool clean = errors.finish();
  if (ended != WGPUStatus_Success || !clean || !s->pin->current()) {
    quarantineImportedFrame(s);
    return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
  releaseNativeReferences(s);
  {
    std::lock_guard<std::mutex> lock(s->mutex);
    s->phase = MGPU_IMPORTED_FRAME_RETIRED;
    s->result = MGPU_CONTEXT_LEASE_OK;
    s->proveNativeReleased();
  }
  return MGPU_CONTEXT_LEASE_OK;
#else
  quarantineImportedFrame(s);
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#endif
}
} // namespace mgpu

extern "C" int mgpuContextLeaseImportImmutableD3D11BgraFrame(
    MGPUContextLease *lease, const MGPUExternalVideoBuffer *descriptor,
    uint32_t size, MGPUImportedFrame **out) {
  if (!out) return MGPU_CONTEXT_LEASE_INVALID;
  *out = nullptr;
  if (!descriptor || size != sizeof(*descriptor)) return MGPU_CONTEXT_LEASE_INVALID;
  const auto copied = *descriptor;
  uint64_t bytes = 0;
  if (!mgpu::validProfile(copied, bytes)) return MGPU_CONTEXT_LEASE_INVALID;
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
  if (!lease) return MGPU_CONTEXT_LEASE_INVALID;
  mgpu::source_scope::MetadataUnique<MGPUImportedFrame> holder;
  std::shared_ptr<mgpu::ImportedFrameState> state;
  bool submitted = false;
  try {
    auto pin = mgpu::copyContextLeasePin(lease);
    if (!pin) return MGPU_CONTEXT_LEASE_INVALID;
    if (!pin->current()) return MGPU_CONTEXT_LEASE_RETIRED;
    const auto geometry = mgpu::capture_work::validateFrame(pin->sourceDomain(),
        copied.width, copied.height, bytes);
    if (geometry != mgpu::source_scope::Result::Ok)
      return mgpu::capture_cleanup::nativeResult(geometry);
    if (std::this_thread::get_id() == pin->gpu().getWebGPUThread().threadId())
      return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    mgpu::capture_recipe::ResourceCustody role;
    auto reserved = role.reserveSourceRole(pin->sourceDomain(),
        mgpu::source_scope::ResourceKind::ImportedFrame);
    if (reserved != mgpu::source_scope::Result::Ok)
      return mgpu::capture_cleanup::nativeResult(reserved);
    mgpu::source_scope::MetadataReservations resourceMetadata;
    reserved = role.reserveResourceMetadata(resourceMetadata);
    if (reserved != mgpu::source_scope::Result::Ok)
      return mgpu::capture_cleanup::nativeResult(reserved);
    mgpu::capture_cleanup::Family family;
    reserved = mgpu::capture_cleanup::Family::beginWork(pin->sourceDomain(), family);
    if (reserved != mgpu::source_scope::Result::Ok)
      return mgpu::capture_cleanup::nativeResult(reserved);
    mgpu::capture_work::PreparedWork job;
    int result = job.reserve(family);
    if (result != MGPU_CONTEXT_LEASE_OK) return result;
    holder = mgpu::source_scope::makeMetadataUnique<MGPUImportedFrame>(resourceMetadata,
        mgpu::source_scope::MetadataKind::ImportedFrameBox);
    state = mgpu::reserveFrame(pin, copied, bytes, std::move(role), resourceMetadata);
    if (!state) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    holder->state = state;
    result = job.configureSynchronous([state, sharedHandle = copied.planes[0].data_ptr] {
      using mgpu::capture_work::Outcome;
      try {
        const int imported = mgpu::importOnWorker(state, sharedHandle);
        if (imported == MGPU_CONTEXT_LEASE_QUARANTINED) return Outcome::uncertain();
        if (imported != MGPU_CONTEXT_LEASE_OK) {
          // Known pre-access failure only; actual native release precedes proof.
          mgpu::releaseNativeReferences(state);
          state->proveNativeReleased();
        }
        return Outcome::done(imported);
      } catch (...) {
        mgpu::quarantineImportedFrame(state);
        return Outcome::uncertain();
      }
    });
    if (result == MGPU_CONTEXT_LEASE_OK)
      result = mgpu::capture_cleanup::nativeResult(state->acceptSourceRole());
    if (result == MGPU_CONTEXT_LEASE_OK)
      result = job.enqueue([](void *context, mgpu::task_queue::Prepared &task) noexcept {
        return static_cast<mgpu::MGPU *>(context)->getWebGPUThread().enqueuePrepared(task);
      }, &pin->gpu());
    if (result != MGPU_CONTEXT_LEASE_OK) {
      state->proveNativeReleased(); // Proven pre-enqueue; no native import ran.
      if (!mgpu::removeFrameCharge(state)) throw std::runtime_error("frame charge mismatch");
      return result;
    }
    submitted = true;
    result = job.wait(); // Job/closure custody outlives result publication.
    if (result == MGPU_CONTEXT_LEASE_OK || result == MGPU_CONTEXT_LEASE_QUARANTINED) {
      *out = holder.release();
      return result;
    }
    if (!mgpu::removeFrameCharge(state)) throw std::runtime_error("frame charge mismatch");
    return result;
  } catch (...) {
    if (state) {
      if (!submitted) {
        // Holder/channel/closure preparation failed before queue acceptance.
        // No native import could run; retain quarantine only on registry doubt.
        try {
          if (mgpu::removeFrameCharge(state)) {
            state->proveNativeReleased();
            return MGPU_CONTEXT_LEASE_INTERNAL;
          }
        } catch (...) {}
      }
      mgpu::quarantineImportedFrame(state);
      *out = holder.release();
      return MGPU_CONTEXT_LEASE_QUARANTINED;
    }
    return MGPU_CONTEXT_LEASE_INTERNAL;
  }
#else
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#endif
}

extern "C" int mgpuImportedFrameGetInfo(MGPUImportedFrame *frame,
    MGPUImportedFrameInfo *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  std::memset(out, 0, sizeof(*out));
  if (!frame || !frame->state) return MGPU_CONTEXT_LEASE_INVALID;
  auto s = frame->state;
  std::lock_guard<std::mutex> lock(s->mutex);
  out->struct_size = sizeof(*out);
  out->version = 1;
  out->phase = s->phase;
  out->last_result = s->result;
  out->context_generation = s->pin->contextGeneration();
  out->device_generation = s->pin->deviceGeneration();
  out->charged_bytes = s->bytes;
  out->width = s->width;
  out->height = s->height;
  return MGPU_CONTEXT_LEASE_OK;
}
extern "C" int mgpuImportedFrameGetLifetimeStats(
    MGPUImportedFrameLifetimeStats *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  std::memset(out, 0, sizeof(*out));
  try {
    auto &r = mgpu::registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    out->struct_size = sizeof(*out);
    out->version = 1;
    out->live_frames = r.count;
    out->max_frames = mgpu::maxFrames;
    out->charged_bytes = r.bytes;
    out->max_bytes = mgpu::maxRetainedBytes;
    out->max_frame_bytes = mgpu::maxFrameBytes;
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
extern "C" void mgpuImportedFrameRetireAsync(MGPUImportedFrame *frame,
    MGPUQueueCompletionCallback callback, void *userdata) {
  if (!callback) return;
  mgpu::capture_cleanup::Family family;
  const auto prepared = mgpu::capture_cleanup::prepareFrameFamily(frame, family);
  if (prepared != MGPU_CONTEXT_LEASE_OK) { callback(prepared, userdata); return; }
  mgpu::capture_cleanup::retireFrame(frame, callback, userdata, family);
}
namespace mgpu::capture_cleanup {
int prepareFrameFamily(MGPUImportedFrame *frame, Family &out) noexcept {
  try {
    const auto state = copyImportedFrameState(frame);
    // Preserve legacy tracked null-owner callback receipts.
    return state ? nativeResult(Family::begin(state->pin->sourceDomain(), out))
                 : MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
void retireFrame(MGPUImportedFrame *frame, MGPUQueueCompletionCallback callback,
                 void *userdata, const Family &family) {
  if (!callback) return;
  if (!frame || !frame->state) {
    callback(MGPU_CONTEXT_LEASE_INVALID, userdata);
    return;
  }
  auto s = frame->state;
  if (!family.cleanup() || !family.matches(s->pin->sourceDomain())) {
    callback(MGPU_CONTEXT_LEASE_INVALID, userdata); return;
  }
  capture_work::PreparedWait wait;
  int prepared;
  try {
    prepared = preparePinnedSubmittedWorkWait(s->pin, family,
        [s, callback, userdata](int result) {
      try {
        if (result == MGPU_CONTEXT_LEASE_OK) result = finishImportedFrameRetirement(s);
        if (result != MGPU_CONTEXT_LEASE_OK) quarantineImportedFrame(s);
      } catch (...) {
        quarantineImportedFrame(s); result = MGPU_CONTEXT_LEASE_QUARANTINED;
      }
      callback(result, userdata);
    }, wait);
  } catch (...) { prepared = MGPU_CONTEXT_LEASE_INTERNAL; }
  if (prepared != MGPU_CONTEXT_LEASE_OK) { callback(prepared, userdata); return; }
  int immediate = MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  bool admitted = false;
  {
    std::lock_guard<std::mutex> lock(s->mutex);
    if (s->phase == MGPU_IMPORTED_FRAME_RETIRED) immediate = MGPU_CONTEXT_LEASE_OK;
    else if (s->phase == MGPU_IMPORTED_FRAME_QUARANTINED)
      immediate = MGPU_CONTEXT_LEASE_QUARANTINED;
    else if (s->phase == MGPU_IMPORTED_FRAME_READY) {
      s->phase = MGPU_IMPORTED_FRAME_RETIRING;
      admitted = true;
    }
  }
  if (!admitted) { callback(immediate, userdata); return; }
  const auto enqueued = enqueuePinnedSubmittedWorkWait(s->pin, wait);
  if (enqueued != MGPU_CONTEXT_LEASE_OK) {
    // No queued/native work was accepted; phase restoration is truthful.
    { std::lock_guard<std::mutex> lock(s->mutex); s->phase = MGPU_IMPORTED_FRAME_READY; }
    callback(enqueued, userdata);
  }
}
} // namespace mgpu::capture_cleanup
extern "C" int mgpuImportedFrameRelease(MGPUImportedFrame *frame) {
  if (!frame || !frame->state) return MGPU_CONTEXT_LEASE_INVALID;
  auto s = frame->state;
  {
    std::lock_guard<std::mutex> lock(s->mutex);
    if (s->phase == MGPU_IMPORTED_FRAME_QUARANTINED)
      return MGPU_CONTEXT_LEASE_QUARANTINED;
    if (s->phase != MGPU_IMPORTED_FRAME_RETIRED) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  }
  if (!mgpu::removeFrameCharge(s)) return MGPU_CONTEXT_LEASE_INTERNAL;
  mgpu::source_scope::MetadataDelete<MGPUImportedFrame>{}(frame);
  return MGPU_CONTEXT_LEASE_OK;
}

namespace mgpu::allocation_layout {
void importedMetadataLayout(MetadataLayoutBuilder &out) noexcept {
  out.shared<ImportedFrameState>(MetadataKind::ImportedFrameState);
  out.unique<ImportedFrameNative>(MetadataKind::ImportedFrameNative);
  out.unique<MGPUImportedFrame>(MetadataKind::ImportedFrameBox);
}
} // namespace mgpu::allocation_layout
