#ifndef FLUTTER_PLUGIN_minigpu_view_D3D11_TEXTURE_HANDLER_H_
#define FLUTTER_PLUGIN_minigpu_view_D3D11_TEXTURE_HANDLER_H_

#include <flutter/texture_registrar.h>

#include <d3d11.h>
#include <wrl/client.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

namespace minigpu_view {

// Wraps a single producer-side ID3D11Texture2D* and exposes it to
// Flutter via FlutterDesktopGpuSurfaceDescriptor (TextureVariant::
// GpuSurfaceTexture, kFlutterDesktopGpuSurfaceTypeD3d11Texture2D).
//
// The texture is BORROWED — this class does NOT own the lifetime. The
// producer (minigpu's SharedOutputTexture) keeps it alive.
class D3D11TextureHandler {
 public:
  explicit D3D11TextureHandler(flutter::TextureRegistrar* registrar);
  ~D3D11TextureHandler();

  D3D11TextureHandler(const D3D11TextureHandler&) = delete;
  D3D11TextureHandler& operator=(const D3D11TextureHandler&) = delete;

  // Registers the GPU-surface texture with Flutter and stores the
  // initial frame. Returns false on failure.
  bool Initialize(ID3D11Texture2D* texture, int width, int height);

  // Initialize using a legacy DXGI shared HANDLE (from
  // IDXGIResource::GetSharedHandle on a producer-side texture created
  // with D3D11_RESOURCE_MISC_SHARED).  Required when the producer's
  // D3D11 device differs from Flutter's ANGLE D3D11 device — the
  // D3d11Texture2D path requires same-device textures and would fail
  // EGL_BAD_PARAMETER inside ANGLE's CreateSurfaceFromHandle.
  bool InitializeFromSharedHandle(void* shared_handle, int width, int height);

  // Swap in a newer frame and notify Flutter to re-rasterize. Cheap;
  // no GPU work happens here — the actual present is one Skia draw call
  // on the raster thread.
  void Update(ID3D11Texture2D* texture, int width, int height);

  // Update for the shared-handle path. The shared handle stays the same
  // as long as the producer texture stays the same; this just bumps the
  // frame-available marker so Flutter rasterizes the next frame.
  void UpdateFromSharedHandle(void* shared_handle, int width, int height);

  // Probes whether [shared_handle] (a LEGACY DXGI share handle) opens on
  // the DEFAULT adapter — the adapter Flutter's ANGLE device lives on
  // (EnumAdapters(0), the one driving the primary display). Legacy shared
  // handles cannot cross adapters, so a producer texture created on
  // another GPU (e.g. Dawn picked the discrete GPU while the iGPU drives
  // the display) registers fine but fails every raster-time bind
  // ("Binding D3D surface failed." from the engine) with nothing
  // surfaced to Dart. Returning the failure from present() instead lets
  // the app fall back to a CPU preview. Caches the last probed handle;
  // fails OPEN (returns true) when no probe device can be created.
  static bool ProbeSharedHandleBindable(void* shared_handle,
                                        std::string* why);

  int64_t texture_id() const { return texture_id_; }

 private:
  // Called by the Flutter engine on the raster thread when it needs the
  // current GPU-side descriptor to compose the next frame.
  const FlutterDesktopGpuSurfaceDescriptor* CopyDescriptor(size_t width,
                                                           size_t height);

  flutter::TextureRegistrar* registrar_ = nullptr;
  std::unique_ptr<flutter::TextureVariant> variant_;
  int64_t texture_id_ = -1;

  // Protects the swap of current_texture_ between Dart-thread Update()
  // calls and raster-thread CopyDescriptor() calls.
  std::mutex mutex_;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> current_texture_;
  // For the shared-handle path: the legacy DXGI share HANDLE that
  // Flutter's ANGLE will OpenSharedResource() on its own device.
  void* shared_handle_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  FlutterDesktopGpuSurfaceDescriptor descriptor_{};
};

}  // namespace minigpu_view

#endif  // FLUTTER_PLUGIN_minigpu_view_D3D11_TEXTURE_HANDLER_H_
