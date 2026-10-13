#include "../include/minigpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

extern "C" mgpu::MGPU minigpu;

struct MGPUTexture {
  std::shared_ptr<mgpu::OwnedTextureState> state;
};

struct MGPUTextureView {
  std::shared_ptr<mgpu::OwnedTextureViewState> state;
};

struct MGPUSampler {
  std::shared_ptr<mgpu::OwnedSamplerState> state;
};

namespace mgpu {

OwnedTextureState::~OwnedTextureState() {
  if (texture) wgpuTextureRelease(texture);
  if (creationDevice) wgpuDeviceRelease(creationDevice);
}

OwnedTextureViewState::~OwnedTextureViewState() {
  if (view) wgpuTextureViewRelease(view);
}

bool OwnedTextureViewState::isAlive() const {
  return alive.load(std::memory_order_acquire) && textureState &&
         textureState->alive.load(std::memory_order_acquire);
}

OwnedSamplerState::~OwnedSamplerState() {
  if (sampler) wgpuSamplerRelease(sampler);
  if (creationDevice) wgpuDeviceRelease(creationDevice);
}

} // namespace mgpu

namespace {

struct FormatInfo {
  WGPUTextureFormat format;
  uint32_t bytesPerTexel;
  bool depth;
  bool filterable;
  bool storage;
};

bool mapFormat(int code, FormatInfo &out) {
  switch (code) {
  case 1:
    out = {WGPUTextureFormat_RGBA8Unorm, 4, false, true, true};
    return true;
  case 2:
    out = {WGPUTextureFormat_BGRA8Unorm, 4, false, true, false};
    return true;
  case 3:
    out = {WGPUTextureFormat_RGBA16Float, 8, false, true, true};
    return true;
  case 4:
    out = {WGPUTextureFormat_R32Float, 4, false, false, true};
    return true;
  case 5:
    out = {WGPUTextureFormat_Depth32Float, 4, true, false, false};
    return true;
  case 6:
    out = {WGPUTextureFormat_RGBA8UnormSrgb, 4, false, true, false};
    return true;
  case 7:
    out = {WGPUTextureFormat_BGRA8UnormSrgb, 4, false, true, false};
    return true;
  default:
    return false;
  }
}

uint32_t maxMipLevels(uint32_t width, uint32_t height) {
  uint32_t extent = std::max(width, height);
  uint32_t levels = 0;
  do {
    ++levels;
    extent >>= 1;
  } while (extent != 0);
  return levels;
}

template <typename F> auto runOnWorker(mgpu::MGPU &owner, F &&operation) {
  using Result = decltype(operation());
  auto &worker = owner.getWebGPUThread();
  if (std::this_thread::get_id() == worker.threadId()) return operation();
  return worker.enqueueSync<Result>(std::forward<F>(operation));
}

std::shared_ptr<mgpu::OwnedTextureState>
createTexture(mgpu::MGPU &owner, int width, int height, int formatCode,
              int usageCode, int layers, int mipLevels) {
  FormatInfo info{};
  if (!mapFormat(formatCode, info) || width < 1 || width > 32768 ||
      height < 1 || height > 32768 || layers < 1 || layers > 32768 ||
      mipLevels < 1 || usageCode <= 0 || (usageCode & ~31) != 0 ||
      static_cast<uint32_t>(mipLevels) >
          maxMipLevels(static_cast<uint32_t>(width),
                       static_cast<uint32_t>(height))) {
    return {};
  }
  if (info.depth && (usageCode & ~(4 | 16)) != 0) return {};
  if ((usageCode & 8) != 0 && !info.storage) return {};

  WGPUDevice device = owner.retainReadyDevice();
  if (!device) return {};
  WGPULimits limits = {};
  if (wgpuDeviceGetLimits(device, &limits) != WGPUStatus_Success ||
      static_cast<uint32_t>(width) > limits.maxTextureDimension2D ||
      static_cast<uint32_t>(height) > limits.maxTextureDimension2D ||
      static_cast<uint32_t>(layers) > limits.maxTextureArrayLayers) {
    wgpuDeviceRelease(device);
    return {};
  }

  WGPUTexture texture = nullptr;
  try {
    texture = runOnWorker(owner, [&]() -> WGPUTexture {
      WGPUDevice current = owner.retainReadyDevice();
      if (!current || current != device) {
        if (current) wgpuDeviceRelease(current);
        return nullptr;
      }
      wgpuDeviceRelease(current);
      mgpu::lock_guard<mgpu::mutex> lock(owner.getGpuMutex());
      WGPUTextureDescriptor descriptor = {};
      descriptor.usage = static_cast<WGPUTextureUsage>(usageCode);
      descriptor.dimension = WGPUTextureDimension_2D;
      descriptor.size = {static_cast<uint32_t>(width),
                         static_cast<uint32_t>(height),
                         static_cast<uint32_t>(layers)};
      descriptor.format = info.format;
      descriptor.mipLevelCount = static_cast<uint32_t>(mipLevels);
      descriptor.sampleCount = 1;
      return wgpuDeviceCreateTexture(device, &descriptor);
    });
    if (!texture) {
      wgpuDeviceRelease(device);
      return {};
    }
    auto state = std::make_shared<mgpu::OwnedTextureState>();
    state->owner = &owner;
    state->creationDevice = device;
    state->texture = texture;
    state->format = info.format;
    state->usage = static_cast<WGPUTextureUsage>(usageCode);
    state->width = static_cast<uint32_t>(width);
    state->height = static_cast<uint32_t>(height);
    state->layers = static_cast<uint32_t>(layers);
    state->mipLevels = static_cast<uint32_t>(mipLevels);
    state->bytesPerTexel = info.bytesPerTexel;
    state->depth = info.depth;
    state->filterable = info.filterable;
    state->storageCapable = info.storage;
    return state;
  } catch (...) {
    if (texture) wgpuTextureRelease(texture);
    wgpuDeviceRelease(device);
    return {};
  }
}

bool mapViewDimension(int code, WGPUTextureViewDimension &dimension) {
  switch (code) {
  case 0:
    dimension = WGPUTextureViewDimension_2D;
    return true;
  case 1:
    dimension = WGPUTextureViewDimension_2DArray;
    return true;
  case 2:
    dimension = WGPUTextureViewDimension_Cube;
    return true;
  case 3:
    dimension = WGPUTextureViewDimension_CubeArray;
    return true;
  default:
    return false;
  }
}

bool mapAddressMode(int code, WGPUAddressMode &mode) {
  switch (code) {
  case 0: mode = WGPUAddressMode_ClampToEdge; return true;
  case 1: mode = WGPUAddressMode_Repeat; return true;
  case 2: mode = WGPUAddressMode_MirrorRepeat; return true;
  default: return false;
  }
}

bool mapFilter(int code, WGPUFilterMode &mode) {
  if (code == 0) { mode = WGPUFilterMode_Nearest; return true; }
  if (code == 1) { mode = WGPUFilterMode_Linear; return true; }
  return false;
}

bool mapMipmapFilter(int code, WGPUMipmapFilterMode &mode) {
  if (code == 0) { mode = WGPUMipmapFilterMode_Nearest; return true; }
  if (code == 1) { mode = WGPUMipmapFilterMode_Linear; return true; }
  return false;
}

bool mapCompare(int code, WGPUCompareFunction &compare) {
  static const WGPUCompareFunction values[] = {
      WGPUCompareFunction_Undefined,    WGPUCompareFunction_Never,
      WGPUCompareFunction_Less,         WGPUCompareFunction_Equal,
      WGPUCompareFunction_LessEqual,    WGPUCompareFunction_Greater,
      WGPUCompareFunction_NotEqual,     WGPUCompareFunction_GreaterEqual,
      WGPUCompareFunction_Always};
  if (code < 0 || code > 8) return false;
  compare = values[code];
  return true;
}

std::shared_ptr<mgpu::OwnedSamplerState>
createSampler(mgpu::MGPU &owner, int addressU, int addressV, int addressW,
              int mag, int min, int mip, float lodMin, float lodMax,
              int compareCode) {
  WGPUAddressMode u, v, w;
  WGPUFilterMode magFilter, minFilter;
  WGPUMipmapFilterMode mipFilter;
  WGPUCompareFunction compare;
  if (!mapAddressMode(addressU, u) || !mapAddressMode(addressV, v) ||
      !mapAddressMode(addressW, w) || !mapFilter(mag, magFilter) ||
      !mapFilter(min, minFilter) || !mapMipmapFilter(mip, mipFilter) ||
      !mapCompare(compareCode, compare) || !std::isfinite(lodMin) ||
      !std::isfinite(lodMax) || lodMin < 0 || lodMin > lodMax || lodMax > 32) {
    return {};
  }

  WGPUDevice device = owner.retainReadyDevice();
  if (!device) return {};
  WGPUSampler sampler = nullptr;
  try {
    sampler = runOnWorker(owner, [&]() -> WGPUSampler {
      WGPUDevice current = owner.retainReadyDevice();
      if (!current || current != device) {
        if (current) wgpuDeviceRelease(current);
        return nullptr;
      }
      wgpuDeviceRelease(current);
      mgpu::lock_guard<mgpu::mutex> lock(owner.getGpuMutex());
      WGPUSamplerDescriptor descriptor = {};
      descriptor.addressModeU = u;
      descriptor.addressModeV = v;
      descriptor.addressModeW = w;
      descriptor.magFilter = magFilter;
      descriptor.minFilter = minFilter;
      descriptor.mipmapFilter = mipFilter;
      descriptor.lodMinClamp = lodMin;
      descriptor.lodMaxClamp = lodMax;
      descriptor.compare = compare;
      descriptor.maxAnisotropy = 1;
      return wgpuDeviceCreateSampler(device, &descriptor);
    });
    if (!sampler) {
      wgpuDeviceRelease(device);
      return {};
    }
    auto state = std::make_shared<mgpu::OwnedSamplerState>();
    state->owner = &owner;
    state->creationDevice = device;
    state->sampler = sampler;
    state->comparison = compareCode != 0;
    state->allNearest = mag == 0 && min == 0 && mip == 0;
    return state;
  } catch (...) {
    if (sampler) wgpuSamplerRelease(sampler);
    wgpuDeviceRelease(device);
    return {};
  }
}

bool currentDeviceMatches(mgpu::MGPU &owner, WGPUDevice creationDevice) {
  WGPUDevice current = owner.retainReadyDevice();
  const bool matches = current && current == creationDevice;
  if (current) wgpuDeviceRelease(current);
  return matches;
}

} // namespace

extern "C" {

uint32_t mgpuTextureResourceVersion(void) { return 1; }

MGPUTexture *mgpuCreateTexture2D(int width, int height, int format, int usage,
                                 int layers, int mipLevels) {
  try {
    auto state = createTexture(minigpu, width, height, format, usage, layers,
                               mipLevels);
    return state ? new MGPUTexture{std::move(state)} : nullptr;
  } catch (...) {
    return nullptr;
  }
}

MGPUTexture *mgpuContextCreateTexture2D(MGPUContextHandle *context, int width,
                                        int height, int format, int usage,
                                        int layers, int mipLevels) {
  if (!context) return nullptr;
  try {
    auto &owner = *reinterpret_cast<mgpu::MGPU *>(context);
    auto state = createTexture(owner, width, height, format, usage, layers,
                               mipLevels);
    return state ? new MGPUTexture{std::move(state)} : nullptr;
  } catch (...) {
    return nullptr;
  }
}

MGPUTextureView *mgpuCreateTextureView(MGPUTexture *texture, int dimensionCode,
                                       int baseMip, int mipCount, int baseLayer,
                                       int layerCount) {
  if (!texture || !texture->state || baseMip < 0 || mipCount <= 0 ||
      baseLayer < 0 || layerCount <= 0) return nullptr;
  try {
    auto textureState = texture->state;
    WGPUTextureViewDimension dimension;
    if (!textureState->alive.load(std::memory_order_acquire) ||
        !mapViewDimension(dimensionCode, dimension)) return nullptr;
    const uint32_t baseMipU = static_cast<uint32_t>(baseMip);
    const uint32_t mipCountU = static_cast<uint32_t>(mipCount);
    const uint32_t baseLayerU = static_cast<uint32_t>(baseLayer);
    const uint32_t layerCountU = static_cast<uint32_t>(layerCount);
    if (baseMipU > textureState->mipLevels ||
        mipCountU > textureState->mipLevels - baseMipU ||
        baseLayerU > textureState->layers ||
        layerCountU > textureState->layers - baseLayerU) return nullptr;
    if ((dimensionCode == 0 && layerCount != 1) ||
        (dimensionCode == 2 && layerCount != 6) ||
        (dimensionCode == 3 && layerCount % 6 != 0) ||
        ((dimensionCode == 2 || dimensionCode == 3) &&
         textureState->width != textureState->height)) return nullptr;
    if (!currentDeviceMatches(*textureState->owner,
                              textureState->creationDevice)) return nullptr;

    auto state = std::make_shared<mgpu::OwnedTextureViewState>();
    state->textureState = textureState;
    state->dimension = dimension;
    state->mipCount = mipCountU;
    state->mipWidth = std::max(1u, textureState->width >> baseMipU);
    state->mipHeight = std::max(1u, textureState->height >> baseMipU);
    state->layerCount = layerCountU;
    WGPUTextureView view = runOnWorker(
        *textureState->owner, [&]() -> WGPUTextureView {
          if (!textureState->alive.load(std::memory_order_acquire) ||
              !currentDeviceMatches(*textureState->owner,
                                    textureState->creationDevice)) return nullptr;
          mgpu::lock_guard<mgpu::mutex> lock(
              textureState->owner->getGpuMutex());
          WGPUTextureViewDescriptor descriptor = {};
          descriptor.format = textureState->format;
          descriptor.dimension = dimension;
          descriptor.baseMipLevel = baseMipU;
          descriptor.mipLevelCount = mipCountU;
          descriptor.baseArrayLayer = baseLayerU;
          descriptor.arrayLayerCount = layerCountU;
          descriptor.aspect = WGPUTextureAspect_All;
          return wgpuTextureCreateView(textureState->texture, &descriptor);
        });
    if (!view) return nullptr;
    state->view = view;
    return new MGPUTextureView{std::move(state)};
  } catch (...) {
    return nullptr;
  }
}

MGPUSampler *mgpuCreateSampler(int addressU, int addressV, int addressW,
                               int mag, int min, int mip, float lodMin,
                               float lodMax, int compare) {
  try {
    auto state = createSampler(minigpu, addressU, addressV, addressW, mag, min,
                               mip, lodMin, lodMax, compare);
    return state ? new MGPUSampler{std::move(state)} : nullptr;
  } catch (...) {
    return nullptr;
  }
}

MGPUSampler *mgpuContextCreateSampler(MGPUContextHandle *context, int addressU,
                                      int addressV, int addressW, int mag,
                                      int min, int mip, float lodMin,
                                      float lodMax, int compare) {
  if (!context) return nullptr;
  try {
    auto &owner = *reinterpret_cast<mgpu::MGPU *>(context);
    auto state = createSampler(owner, addressU, addressV, addressW, mag, min,
                               mip, lodMin, lodMax, compare);
    return state ? new MGPUSampler{std::move(state)} : nullptr;
  } catch (...) {
    return nullptr;
  }
}

uint32_t mgpuTextureRegionUploadVersion(void) { return 1; }

int mgpuWriteTextureRegion2D(MGPUTexture *texture, const uint8_t *bytes,
    int byteCount, int x, int y, int width, int height, int mipLevel,
    int arrayLayer) {
  if (!texture || !texture->state || !bytes || byteCount < 0 || mipLevel < 0 ||
      arrayLayer < 0 || x < 0 || y < 0 || width <= 0 || height <= 0) return 1;
  try {
    auto state = texture->state;
    if (!state->alive.load(std::memory_order_acquire) || state->depth ||
        (state->usage & WGPUTextureUsage_CopyDst) == 0 ||
        static_cast<uint32_t>(mipLevel) >= state->mipLevels ||
        static_cast<uint32_t>(arrayLayer) >= state->layers) return 1;
    const uint32_t mipWidth = std::max(1u, state->width >> mipLevel);
    const uint32_t mipHeight = std::max(1u, state->height >> mipLevel);
    const auto originX = static_cast<uint32_t>(x);
    const auto originY = static_cast<uint32_t>(y);
    const auto regionWidth = static_cast<uint32_t>(width);
    const auto regionHeight = static_cast<uint32_t>(height);
    if (originX >= mipWidth || originY >= mipHeight ||
        regionWidth > mipWidth - originX ||
        regionHeight > mipHeight - originY) return 1;
    const uint64_t expected = static_cast<uint64_t>(regionWidth) * regionHeight *
                              state->bytesPerTexel;
    if (expected > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        static_cast<uint64_t>(byteCount) != expected) return 1;
    if (!currentDeviceMatches(*state->owner, state->creationDevice)) return 2;
    std::vector<uint8_t> owned(bytes, bytes + byteCount);
    state->owner->getWebGPUThread().enqueueAsync(
        [state, owned = std::move(owned), originX, originY,
         regionWidth, regionHeight, mipLevel,
         arrayLayer]() {
          if (!state->alive.load(std::memory_order_acquire) ||
              !currentDeviceMatches(*state->owner, state->creationDevice)) return;
          mgpu::lock_guard<mgpu::mutex> lock(state->owner->getGpuMutex());
          state->owner->flushBatchLocked();
          WGPUQueue queue = state->owner->tryGetQueue();
          if (!queue) return;
          WGPUTexelCopyTextureInfo destination = {};
          destination.texture = state->texture;
          destination.mipLevel = static_cast<uint32_t>(mipLevel);
          destination.origin = {originX, originY, static_cast<uint32_t>(arrayLayer)};
          destination.aspect = WGPUTextureAspect_All;
          WGPUTexelCopyBufferLayout layout = {};
          layout.bytesPerRow = regionWidth * state->bytesPerTexel;
          layout.rowsPerImage = regionHeight;
          WGPUExtent3D extent = {regionWidth, regionHeight, 1};
          wgpuQueueWriteTexture(queue, &destination, owned.data(), owned.size(),
                                &layout, &extent);
        });
    return 0;
  } catch (...) {
    return 3;
  }
}

int mgpuWriteTexture2D(MGPUTexture *texture, const uint8_t *bytes,
                       int byteCount, int mipLevel, int arrayLayer) {
  if (!texture || !texture->state || mipLevel < 0 ||
      static_cast<uint32_t>(mipLevel) >= texture->state->mipLevels) return 1;
  const auto &state = texture->state;
  return mgpuWriteTextureRegion2D(texture, bytes, byteCount, 0, 0,
      static_cast<int>(std::max(1u, state->width >> mipLevel)),
      static_cast<int>(std::max(1u, state->height >> mipLevel)), mipLevel, arrayLayer);
}

int mgpuSetOwnedTexture(MGPUComputeShader *shader, int slot,
                        MGPUTextureView *view, int kind) {
  if (!shader || !view || !view->state || slot < 0 || slot > 31 || kind < 0 ||
      kind > 3) return 1;
  try {
    auto *nativeShader = reinterpret_cast<mgpu::ComputeShader *>(shader);
    auto state = view->state;
    auto texture = state->textureState;
    if (!state->isAlive() || !texture ||
        texture->owner != &nativeShader->getMGPU()) return 1;
    if (nativeShader->getCreationDevice() != texture->creationDevice ||
        !currentDeviceMatches(*texture->owner, texture->creationDevice)) return 2;
    if (kind <= 2) {
      if ((texture->usage & WGPUTextureUsage_TextureBinding) == 0) return 1;
      if ((kind == 0 && (!texture->filterable || texture->depth)) ||
          (kind == 1 && texture->depth) || (kind == 2 && !texture->depth)) return 1;
    } else if ((texture->usage & WGPUTextureUsage_StorageBinding) == 0 ||
               !texture->storageCapable || state->mipCount != 1 ||
               (state->dimension != WGPUTextureViewDimension_2D &&
                state->dimension != WGPUTextureViewDimension_2DArray)) {
      return 1;
    }
    WGPUDevice device = texture->owner->retainReadyDevice();
    if (!device) return 2;
    WGPULimits limits = {};
    const bool limitsOk =
        wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success;
    wgpuDeviceRelease(device);
    if (!limitsOk) return 3;
    if (static_cast<uint32_t>(slot) >= limits.maxBindingsPerBindGroup ||
        (kind == 3 ? limits.maxStorageTexturesPerShaderStage
                   : limits.maxSampledTexturesPerShaderStage) == 0) return 1;
    const mgpu::BindingKind bindingKind =
        kind == 0 ? mgpu::BindingKind::kSampledTextureFloat
        : kind == 1 ? mgpu::BindingKind::kSampledTextureUnfilterable
        : kind == 2 ? mgpu::BindingKind::kDepthTexture
                    : mgpu::BindingKind::kStorageTextureWriteOnly;
    nativeShader->setOwnedTextureQueued(slot, std::move(state), bindingKind);
    return 0;
  } catch (...) {
    return 3;
  }
}

int mgpuSetOwnedSampler(MGPUComputeShader *shader, int slot,
                        MGPUSampler *sampler, int kind) {
  if (!shader || !sampler || !sampler->state || slot < 0 || slot > 31 ||
      kind < 0 || kind > 2) return 1;
  try {
    auto *nativeShader = reinterpret_cast<mgpu::ComputeShader *>(shader);
    auto state = sampler->state;
    if (!state->alive.load(std::memory_order_acquire) ||
        state->owner != &nativeShader->getMGPU()) return 1;
    if (nativeShader->getCreationDevice() != state->creationDevice ||
        !currentDeviceMatches(*state->owner, state->creationDevice)) return 2;
    if ((kind == 2) != state->comparison || (kind == 1 && !state->allNearest))
      return 1;
    WGPUDevice device = state->owner->retainReadyDevice();
    if (!device) return 2;
    WGPULimits limits = {};
    const bool limitsOk =
        wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success;
    wgpuDeviceRelease(device);
    if (!limitsOk) return 3;
    if (static_cast<uint32_t>(slot) >= limits.maxBindingsPerBindGroup ||
        limits.maxSamplersPerShaderStage == 0) return 1;
    const mgpu::BindingKind bindingKind =
        kind == 0 ? mgpu::BindingKind::kFilteringSampler
        : kind == 1 ? mgpu::BindingKind::kNonFilteringSampler
                    : mgpu::BindingKind::kComparisonSampler;
    nativeShader->setOwnedSamplerQueued(slot, std::move(state), bindingKind);
    return 0;
  } catch (...) {
    return 3;
  }
}
int mgpuReleaseTextureBinding(MGPUComputeShader *shader, int slot) {
  if (!shader || slot < 0 || slot > 63) return 1;
  try {
    reinterpret_cast<mgpu::ComputeShader *>(shader)->releaseTextureBindingQueued(slot);
    return 0;
  } catch (...) {
    return 3;
  }
}

void mgpuDestroyTexture(MGPUTexture *texture) {
  if (!texture) return;
  try {
    if (texture->state)
      texture->state->alive.store(false, std::memory_order_release);
    delete texture;
  } catch (...) {
  }
}

void mgpuDestroyTextureView(MGPUTextureView *view) {
  if (!view) return;
  try {
    if (view->state) view->state->alive.store(false, std::memory_order_release);
    delete view;
  } catch (...) {
  }
}

void mgpuDestroySampler(MGPUSampler *sampler) {
  if (!sampler) return;
  try {
    if (sampler->state)
      sampler->state->alive.store(false, std::memory_order_release);
    delete sampler;
  } catch (...) {
  }
}

} // extern "C"

namespace mgpu {

std::shared_ptr<OwnedTextureViewState>
retainOwnedTextureViewState(MGPUTextureView *view) {
  return view ? view->state : std::shared_ptr<OwnedTextureViewState>{};
}

std::shared_ptr<OwnedSamplerState> retainOwnedSamplerState(MGPUSampler *sampler) {
  return sampler ? sampler->state : std::shared_ptr<OwnedSamplerState>{};
}

} // namespace mgpu
