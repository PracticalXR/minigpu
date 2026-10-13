#ifndef MINIGPU_RESOURCES_H
#define MINIGPU_RESOURCES_H

#include "export.h"
#include "stdint.h"

#ifdef __cplusplus
#include "webgpu.h"
#include <atomic>
#include <memory>
extern "C" {
#endif

typedef struct MGPUContextHandle MGPUContextHandle;
typedef struct MGPUComputeShader MGPUComputeShader;
typedef struct MGPUTexture MGPUTexture;
typedef struct MGPUTextureView MGPUTextureView;
typedef struct MGPUSampler MGPUSampler;

EXPORT uint32_t mgpuTextureResourceVersion(void);
EXPORT MGPUTexture *mgpuCreateTexture2D(int width, int height, int format,
                                        int usage, int layers, int mipLevels);
EXPORT MGPUTexture *mgpuContextCreateTexture2D(
    MGPUContextHandle *context, int width, int height, int format, int usage,
    int layers, int mipLevels);
EXPORT MGPUTextureView *mgpuCreateTextureView(
    MGPUTexture *texture, int dimension, int baseMip, int mipCount,
    int baseLayer, int layerCount);
EXPORT MGPUSampler *mgpuCreateSampler(int addressU, int addressV, int addressW,
                                      int mag, int min, int mip, float lodMin,
                                      float lodMax, int compare);
EXPORT MGPUSampler *mgpuContextCreateSampler(
    MGPUContextHandle *context, int addressU, int addressV, int addressW,
    int mag, int min, int mip, float lodMin, float lodMax, int compare);
EXPORT int mgpuWriteTexture2D(MGPUTexture *texture, const uint8_t *bytes,
                              int byteCount, int mipLevel, int arrayLayer);
EXPORT uint32_t mgpuTextureRegionUploadVersion(void);
EXPORT int mgpuWriteTextureRegion2D(MGPUTexture *texture, const uint8_t *bytes,
    int byteCount, int x, int y, int width, int height, int mipLevel,
    int arrayLayer);
EXPORT int mgpuSetOwnedTexture(MGPUComputeShader *shader, int slot,
                               MGPUTextureView *view, int kind);
EXPORT int mgpuSetOwnedSampler(MGPUComputeShader *shader, int slot,
                               MGPUSampler *sampler, int kind);
EXPORT void mgpuDestroyTexture(MGPUTexture *texture);
EXPORT void mgpuDestroyTextureView(MGPUTextureView *view);
EXPORT void mgpuDestroySampler(MGPUSampler *sampler);

#ifdef __cplusplus
}

namespace mgpu {

class MGPU;

struct OwnedTextureState {
  MGPU *owner = nullptr;
  WGPUDevice creationDevice = nullptr;
  WGPUTexture texture = nullptr;
  WGPUTextureFormat format = WGPUTextureFormat_Undefined;
  WGPUTextureUsage usage = WGPUTextureUsage_None;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t layers = 0;
  uint32_t mipLevels = 0;
  uint32_t bytesPerTexel = 0;
  bool depth = false;
  bool filterable = false;
  bool storageCapable = false;
  std::atomic<bool> alive{true};
  ~OwnedTextureState();
};

struct OwnedTextureViewState {
  std::shared_ptr<OwnedTextureState> textureState;
  WGPUTextureView view = nullptr;
  WGPUTextureViewDimension dimension = WGPUTextureViewDimension_Undefined;
  uint32_t mipCount = 0;
  uint32_t mipWidth = 0;
  uint32_t mipHeight = 0;
  uint32_t layerCount = 0;
  std::atomic<bool> alive{true};
  ~OwnedTextureViewState();
  bool isAlive() const;
};

struct OwnedSamplerState {
  MGPU *owner = nullptr;
  WGPUDevice creationDevice = nullptr;
  WGPUSampler sampler = nullptr;
  bool comparison = false;
  bool allNearest = true;
  std::atomic<bool> alive{true};
  ~OwnedSamplerState();
};

std::shared_ptr<OwnedTextureViewState>
retainOwnedTextureViewState(MGPUTextureView *view);
std::shared_ptr<OwnedSamplerState> retainOwnedSamplerState(MGPUSampler *sampler);

} // namespace mgpu
#endif

#endif // MINIGPU_RESOURCES_H
