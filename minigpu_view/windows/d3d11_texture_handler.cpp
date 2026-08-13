#include "d3d11_texture_handler.h"

#include <flutter/texture_registrar.h>

#include <cstdio>

namespace minigpu_view {

// static
bool D3D11TextureHandler::ProbeSharedHandleBindable(void* shared_handle,
                                                    std::string* why) {
  // One probe device + a last-handle cache: the handle is stable per
  // producer texture, so steady-state presents cost one pointer compare.
  static std::mutex probe_mutex;
  static Microsoft::WRL::ComPtr<ID3D11Device> probe_device;
  static bool device_attempted = false;
  static void* last_handle = nullptr;
  static bool last_ok = false;
  static std::string last_why;

  if (!shared_handle) return true;  // nothing to probe

  std::lock_guard<std::mutex> lock(probe_mutex);
  if (shared_handle == last_handle) {
    if (!last_ok && why) *why = last_why;
    return last_ok;
  }

  if (!device_attempted) {
    device_attempted = true;
    // Adapter nullptr = the default adapter = the one driving the primary
    // display = the adapter ANGLE's device is created on. This device only
    // opens handles (no rendering), so no feature levels / flags needed.
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE,
                                   nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                                   probe_device.GetAddressOf(), nullptr,
                                   nullptr);
    if (FAILED(hr)) probe_device.Reset();
  }
  if (!probe_device) return true;  // can't probe -> don't block the present

  Microsoft::WRL::ComPtr<ID3D11Texture2D> opened;
  HRESULT hr = probe_device->OpenSharedResource(
      static_cast<HANDLE>(shared_handle), IID_PPV_ARGS(&opened));
  last_handle = shared_handle;
  last_ok = SUCCEEDED(hr);
  if (!last_ok) {
    // CLASSIFY THE HRESULT. This used to report every failure as "the producer
    // created it on a different GPU", which is only one of the ways
    // OpenSharedResource fails and sends anyone reading the message hunting
    // for a multi-GPU problem they do not have. E_OUTOFMEMORY in particular
    // means resources are exhausted — commonly leaked shared textures, since
    // each one holds a kernel handle — and the usual reaction to a
    // cross-adapter verdict (fall back to a CPU path) makes that WORSE rather
    // than working around it.
    const char* cause;
    switch (hr) {
      case E_INVALIDARG:
      case E_ACCESSDENIED:
        // The codes a genuine adapter mismatch produces; minigpu's own
        // importer uses the same two to decide it must fall back.
        cause = "the producer created it on a different GPU";
        break;
      case E_OUTOFMEMORY:
        cause = "OUT OF RESOURCES on the primary-display adapter -- this is "
                "NOT an adapter mismatch. Shared textures each hold a kernel "
                "handle, so a leak of them exhausts the process long before "
                "VRAM looks full. Falling back to a CPU path will not help";
        break;
      default:
        cause = "reason unclassified -- not a known adapter-mismatch code";
        break;
    }
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "OpenSharedResource on the default (primary display) "
                  "adapter failed with 0x%08lX: %s",
                  static_cast<unsigned long>(hr), cause);
    last_why = buf;
    if (why) *why = last_why;
  }
  return last_ok;
}

D3D11TextureHandler::D3D11TextureHandler(flutter::TextureRegistrar* registrar)
    : registrar_(registrar) {}

D3D11TextureHandler::~D3D11TextureHandler() {
  if (texture_id_ < 0 || !registrar_) return;

  // UNREGISTRATION IS ASYNCHRONOUS. The engine may call the surface callback
  // on the raster thread until it has actually processed this, so neither the
  // TextureVariant it holds a raw pointer to nor the state that callback reads
  // may die here. Hand both to the completion callback and let it own them:
  // whenever the engine is finished, the captures go out of scope and the
  // memory is released — on the raster thread, with nothing still using it.
  //
  // The deprecated single-argument overload gives no completion signal, which
  // is exactly what made this a use-after-free: freed mutex, descriptor
  // written into freed heap, dangling pointer returned to the compositor. The
  // symptom lands far away — corrupted heap surfaces as an unrelated crash,
  // classically the Dart VM aborting on an innocent FFI callback.
  auto variant = variant_;
  auto state = state_;
  registrar_->UnregisterTexture(texture_id_, [variant, state]() mutable {
    // Sole purpose: extend both lifetimes to here.
    variant.reset();
    state.reset();
  });
  texture_id_ = -1;
}

bool D3D11TextureHandler::RegisterWithEngine(FlutterDesktopGpuSurfaceType type) {
  // The callback captures the STATE, not `this` — see SurfaceState.
  auto state = state_;
  variant_ = std::make_shared<flutter::TextureVariant>(flutter::GpuSurfaceTexture(
      type,
      [state](size_t /* w */, size_t /* h */)
          -> const FlutterDesktopGpuSurfaceDescriptor* {
        return state->CopyDescriptor();
      }));

  texture_id_ = registrar_->RegisterTexture(variant_.get());
  if (texture_id_ < 0) {
    variant_.reset();
    return false;
  }

  // Trigger an initial paint.
  registrar_->MarkTextureFrameAvailable(texture_id_);
  return true;
}

bool D3D11TextureHandler::Initialize(ID3D11Texture2D* texture, int width,
                                     int height) {
  if (!texture || width <= 0 || height <= 0) return false;

  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->texture = texture;  // AddRef via ComPtr assignment
    state_->width = width;
    state_->height = height;
  }

  return RegisterWithEngine(kFlutterDesktopGpuSurfaceTypeD3d11Texture2D);
}

bool D3D11TextureHandler::InitializeFromSharedHandle(void* shared_handle,
                                                     int width, int height) {
  if (!shared_handle || width <= 0 || height <= 0) return false;

  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->shared_handle = shared_handle;
    state_->texture = nullptr;  // not used in shared-handle path
    state_->width = width;
    state_->height = height;
  }

  return RegisterWithEngine(kFlutterDesktopGpuSurfaceTypeDxgiSharedHandle);
}

void D3D11TextureHandler::Update(ID3D11Texture2D* texture, int width,
                                 int height) {
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->texture = texture;
    state_->width = width;
    state_->height = height;
  }
  if (texture_id_ >= 0 && registrar_) {
    registrar_->MarkTextureFrameAvailable(texture_id_);
  }
}

void D3D11TextureHandler::UpdateFromSharedHandle(void* shared_handle,
                                                 int width, int height) {
  {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->shared_handle = shared_handle;
    state_->width = width;
    state_->height = height;
  }
  if (texture_id_ >= 0 && registrar_) {
    registrar_->MarkTextureFrameAvailable(texture_id_);
  }
}

const FlutterDesktopGpuSurfaceDescriptor*
D3D11TextureHandler::SurfaceState::CopyDescriptor() {
  // Build a snapshot of the current frame on the raster thread.
  std::lock_guard<std::mutex> lock(mutex);
  // Prefer the shared-handle path when available (cross-device safe).
  if (shared_handle) {
    descriptor.struct_size = sizeof(FlutterDesktopGpuSurfaceDescriptor);
    descriptor.handle = shared_handle;
    descriptor.width = static_cast<size_t>(width);
    descriptor.height = static_cast<size_t>(height);
    descriptor.visible_width = static_cast<size_t>(width);
    descriptor.visible_height = static_cast<size_t>(height);
    descriptor.format = kFlutterDesktopPixelFormatBGRA8888;
    descriptor.release_callback = nullptr;
    descriptor.release_context = nullptr;
    return &descriptor;
  }
  if (!texture) return nullptr;
  descriptor.struct_size = sizeof(FlutterDesktopGpuSurfaceDescriptor);
  descriptor.handle = texture.Get();
  descriptor.width = static_cast<size_t>(width);
  descriptor.height = static_cast<size_t>(height);
  descriptor.visible_width = static_cast<size_t>(width);
  descriptor.visible_height = static_cast<size_t>(height);
  descriptor.format = kFlutterDesktopPixelFormatBGRA8888;
  // NOTE: Flutter Desktop's GpuSurface API only documents BGRA8888 for
  // D3D11. minigpu's SharedOutputTexture is created as RGBA8. If the
  // raster shows red/blue swapped, run a one-pass swizzle on the
  // producer side (or recreate SharedOutputTexture with a BGRA view).
  // No release callback: the producer owns the texture lifetime.
  descriptor.release_callback = nullptr;
  descriptor.release_context = nullptr;
  return &descriptor;
}

}  // namespace minigpu_view
