#include "d3d11_texture_handler.h"

#include <flutter/texture_registrar.h>

namespace minigpu_view {

D3D11TextureHandler::D3D11TextureHandler(flutter::TextureRegistrar* registrar)
    : registrar_(registrar) {}

D3D11TextureHandler::~D3D11TextureHandler() {
  if (texture_id_ >= 0 && registrar_) {
    registrar_->UnregisterTexture(texture_id_);
  }
}

bool D3D11TextureHandler::Initialize(ID3D11Texture2D* texture, int width,
                                     int height) {
  if (!texture || width <= 0 || height <= 0) return false;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    current_texture_ = texture;  // AddRef via ComPtr assignment
    width_ = width;
    height_ = height;
  }

  // Build a GpuSurfaceTexture variant that re-samples the descriptor
  // every frame via our CopyDescriptor callback.
  variant_ = std::make_unique<flutter::TextureVariant>(
      flutter::GpuSurfaceTexture(
          kFlutterDesktopGpuSurfaceTypeD3d11Texture2D,
          [this](size_t w, size_t h)
              -> const FlutterDesktopGpuSurfaceDescriptor* {
            return CopyDescriptor(w, h);
          }));

  texture_id_ = registrar_->RegisterTexture(variant_.get());
  if (texture_id_ < 0) return false;

  // Trigger an initial paint.
  registrar_->MarkTextureFrameAvailable(texture_id_);
  return true;
}

bool D3D11TextureHandler::InitializeFromSharedHandle(void* shared_handle,
                                                     int width, int height) {
  if (!shared_handle || width <= 0 || height <= 0) return false;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    shared_handle_ = shared_handle;
    current_texture_ = nullptr;  // not used in shared-handle path
    width_ = width;
    height_ = height;
  }

  variant_ = std::make_unique<flutter::TextureVariant>(
      flutter::GpuSurfaceTexture(
          kFlutterDesktopGpuSurfaceTypeDxgiSharedHandle,
          [this](size_t w, size_t h)
              -> const FlutterDesktopGpuSurfaceDescriptor* {
            return CopyDescriptor(w, h);
          }));

  texture_id_ = registrar_->RegisterTexture(variant_.get());
  if (texture_id_ < 0) return false;

  registrar_->MarkTextureFrameAvailable(texture_id_);
  return true;
}

void D3D11TextureHandler::Update(ID3D11Texture2D* texture, int width,
                                 int height) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    current_texture_ = texture;
    width_ = width;
    height_ = height;
  }
  if (texture_id_ >= 0 && registrar_) {
    registrar_->MarkTextureFrameAvailable(texture_id_);
  }
}

void D3D11TextureHandler::UpdateFromSharedHandle(void* shared_handle,
                                                 int width, int height) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    shared_handle_ = shared_handle;
    width_ = width;
    height_ = height;
  }
  if (texture_id_ >= 0 && registrar_) {
    registrar_->MarkTextureFrameAvailable(texture_id_);
  }
}

const FlutterDesktopGpuSurfaceDescriptor* D3D11TextureHandler::CopyDescriptor(
    size_t /* width */, size_t /* height */) {
  // Build a snapshot of the current frame on the raster thread.
  std::lock_guard<std::mutex> lock(mutex_);
  // Prefer the shared-handle path when available (cross-device safe).
  if (shared_handle_) {
    descriptor_.struct_size = sizeof(FlutterDesktopGpuSurfaceDescriptor);
    descriptor_.handle = shared_handle_;
    descriptor_.width = static_cast<size_t>(width_);
    descriptor_.height = static_cast<size_t>(height_);
    descriptor_.visible_width = static_cast<size_t>(width_);
    descriptor_.visible_height = static_cast<size_t>(height_);
    descriptor_.format = kFlutterDesktopPixelFormatBGRA8888;
    descriptor_.release_callback = nullptr;
    descriptor_.release_context = nullptr;
    return &descriptor_;
  }
  if (!current_texture_) return nullptr;
  descriptor_.struct_size = sizeof(FlutterDesktopGpuSurfaceDescriptor);
  descriptor_.handle = current_texture_.Get();
  descriptor_.width = static_cast<size_t>(width_);
  descriptor_.height = static_cast<size_t>(height_);
  descriptor_.visible_width = static_cast<size_t>(width_);
  descriptor_.visible_height = static_cast<size_t>(height_);
  descriptor_.format = kFlutterDesktopPixelFormatBGRA8888;
  // NOTE: Flutter Desktop's GpuSurface API only documents BGRA8888 for
  // D3D11. minigpu's SharedOutputTexture is created as RGBA8. If the
  // raster shows red/blue swapped, run a one-pass swizzle on the
  // producer side (or recreate SharedOutputTexture with a BGRA view).
  // No release callback: the producer owns the texture lifetime.
  descriptor_.release_callback = nullptr;
  descriptor_.release_context = nullptr;
  return &descriptor_;
}

}  // namespace minigpu_view
