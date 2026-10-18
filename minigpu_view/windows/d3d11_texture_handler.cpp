#include "d3d11_texture_handler.h"

#include <flutter/texture_registrar.h>

#include <d3d11_1.h>

#include <cstdio>

namespace minigpu_view {

// static
bool D3D11TextureHandler::ProbeSharedHandleBindable(void* shared_handle,
                                                    std::string* why) {
  // One probe device and NO verdict cache. The probe only runs for a handle
  // that has no texture yet, and a cached verdict outlived what it was about:
  // a handle value is reused once closed, and one transient failure (say
  // E_OUTOFMEMORY) refused a stable producer handle until the next resize.
  static std::mutex probe_mutex;
  static Microsoft::WRL::ComPtr<ID3D11Device> probe_device;
  static bool device_attempted = false;

  if (!shared_handle) return true;  // nothing to probe

  std::lock_guard<std::mutex> lock(probe_mutex);
  auto create_device = [] {
    // Adapter nullptr = the default adapter = the one driving the primary
    // display = the adapter ANGLE's device is created on. This device only
    // opens handles (no rendering), so no feature levels / flags needed.
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE,
                                   nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                                   probe_device.GetAddressOf(), nullptr,
                                   nullptr);
    if (FAILED(hr)) probe_device.Reset();
  };
  if (!device_attempted) {
    device_attempted = true;
    create_device();
  }
  if (!probe_device) return true;  // can't probe -> don't block the present

  // Legacy handles (IDXGIResource::GetSharedHandle) open with
  // OpenSharedResource. NT handles (CreateSharedHandle, e.g. the D3D12
  // backend's shared output) get E_INVALIDARG there, the same code an
  // adapter mismatch gives, so they need OpenSharedResource1 before any
  // verdict.
  const char* api = "OpenSharedResource";
  auto open = [&]() -> HRESULT {
    Microsoft::WRL::ComPtr<ID3D11Texture2D> opened;
    api = "OpenSharedResource";
    HRESULT result = probe_device->OpenSharedResource(
        static_cast<HANDLE>(shared_handle), IID_PPV_ARGS(&opened));
    Microsoft::WRL::ComPtr<ID3D11Device1> device1;
    if (result == E_INVALIDARG && SUCCEEDED(probe_device.As(&device1))) {
      api = "OpenSharedResource and OpenSharedResource1";
      result = device1->OpenSharedResource1(static_cast<HANDLE>(shared_handle),
                                            IID_PPV_ARGS(&opened));
    }
    // D3D11 may defer destroying the opened texture to the next flush, and
    // this device submits nothing else, so it would keep every probed
    // producer allocation alive.
    opened.Reset();
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
    probe_device->GetImmediateContext(&context);
    if (context) context->Flush();
    return result;
  };
  HRESULT hr = open();
  // A removed device (TDR, driver update) refuses every handle until it is
  // replaced, so without this no NEW preview would bind again until restart.
  if (FAILED(hr) &&
      (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET ||
       probe_device->GetDeviceRemovedReason() != S_OK)) {
    probe_device.Reset();
    create_device();
    if (!probe_device) return true;  // can't probe -> don't block the present
    hr = open();
  }
  if (FAILED(hr)) {
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
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%s on the default (primary display) "
                  "adapter failed with 0x%08lX: %s",
                  api, static_cast<unsigned long>(hr), cause);
    if (why) *why = buf;
  }
  return SUCCEEDED(hr);
}

D3D11TextureHandler::D3D11TextureHandler(
    flutter::TextureRegistrar* registrar,
    std::shared_ptr<UnregisterTracker> unregistrations)
    : registrar_(registrar), unregistrations_(std::move(unregistrations)) {}

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
  auto unregistrations = unregistrations_;
  const uint64_t ticket = unregistrations ? unregistrations->Begin() : 0;
  registrar_->UnregisterTexture(
      texture_id_, [variant, state, unregistrations, ticket]() mutable {
        // Extend both lifetimes to here, then report the engine let go.
        variant.reset();
        state.reset();
        if (unregistrations) unregistrations->End(ticket);
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
  // The built-in SharedOutputTexture is BGRA8, matching this descriptor.
  // Packed RGBA buffer input is converted by logical WGSL color components;
  // it does not require another channel swap here. Other producers must supply
  // the compatible surface format; this descriptor does not convert formats.
  // State retains the COM reference. No callback acknowledges per-frame GPU
  // sampling; producer writes and final consumer retirement remain separate.
  descriptor.release_callback = nullptr;
  descriptor.release_context = nullptr;
  return &descriptor;
}

}  // namespace minigpu_view
