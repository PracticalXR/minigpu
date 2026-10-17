#ifndef FLUTTER_PLUGIN_minigpu_view_D3D11_TEXTURE_HANDLER_H_
#define FLUTTER_PLUGIN_minigpu_view_D3D11_TEXTURE_HANDLER_H_

#include <flutter/texture_registrar.h>

#include <d3d11.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace minigpu_view {

// Counts texture unregistrations the engine has not finished. UnregisterTexture
// only POSTS the removal to the raster thread, so a reply that promises "this
// texture is gone" has to wait for End.
class UnregisterTracker {
 public:
  virtual ~UnregisterTracker() = default;
  // Platform thread, as a handler unregisters. The ticket goes to End.
  virtual uint64_t Begin() = 0;
  // Any thread, once the engine has removed the texture.
  virtual void End(uint64_t ticket) = 0;
};

// Wraps a single producer-side ID3D11Texture2D* and exposes it to
// Flutter via FlutterDesktopGpuSurfaceDescriptor (TextureVariant::
// GpuSurfaceTexture, kFlutterDesktopGpuSurfaceTypeD3d11Texture2D).
//
// The raw-texture path retains an independent COM reference. The shared-handle
// path borrows the legacy handle and relies on the producer retaining the
// underlying resource. Neither path supplies per-frame GPU-read completion or
// permission to overwrite/release the producer's resource after a present reply.
class D3D11TextureHandler {
 public:
  // [unregistrations], when given, hears when the engine has really dropped
  // this handler's texture after its destruction.
  explicit D3D11TextureHandler(
      flutter::TextureRegistrar* registrar,
      std::shared_ptr<UnregisterTracker> unregistrations = nullptr);
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

  // Probes whether [shared_handle] (a legacy DXGI share handle or an NT
  // handle) opens on the DEFAULT adapter — the adapter Flutter's ANGLE
  // device lives on (EnumAdapters(0), the one driving the primary display).
  // Shared handles cannot cross adapters, so a producer texture created on
  // another GPU (e.g. Dawn picked the discrete GPU while the iGPU drives
  // the display) registers fine but fails every raster-time bind
  // ("Binding D3D surface failed." from the engine) with nothing
  // surfaced to Dart. Returning the failure from present() instead lets
  // the app fall back to a CPU preview. Probes every call (no verdict
  // cache) and replaces a removed probe device once per call; fails OPEN
  // (returns true) when no probe device can be created.
  static bool ProbeSharedHandleBindable(void* shared_handle,
                                        std::string* why);

  int64_t texture_id() const { return texture_id_; }

 private:
  // EVERYTHING THE RASTER THREAD TOUCHES, with a lifetime independent of the
  // handler.
  //
  // `TextureRegistrar::UnregisterTexture` is ASYNCHRONOUS — the engine keeps
  // using the registered TextureVariant, and calling its surface callback,
  // until the unregistration is actually processed on the raster thread. A
  // handler that unregisters and then dies takes this state with it, so the
  // next raster-thread callback locks a destroyed mutex, writes a descriptor
  // into freed heap and hands the engine a dangling pointer. That corruption
  // does not announce itself: it surfaces later, anywhere, as whatever else
  // was living in that heap block — including the Dart VM aborting with
  // "Callback invoked after it has been deleted" on a completely innocent
  // FFI callback.
  //
  // So the callback captures a shared_ptr to this, never `this`, and the
  // handler's destructor keeps both it and the variant alive until the engine
  // confirms the unregistration.
  struct SurfaceState {
    // Protects the swap of the frame between Dart-thread Update() calls and
    // raster-thread CopyDescriptor() calls.
    std::mutex mutex;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    // For the shared-handle path: the legacy DXGI share HANDLE that Flutter's
    // ANGLE will OpenSharedResource() on its own device.
    void* shared_handle = nullptr;
    int width = 0;
    int height = 0;
    FlutterDesktopGpuSurfaceDescriptor descriptor{};

    // Called by the Flutter engine on the raster thread when it needs the
    // current GPU-side descriptor to compose the next frame.
    const FlutterDesktopGpuSurfaceDescriptor* CopyDescriptor();
  };

  // Registers `variant_` and wires the surface callback to `state_`.
  bool RegisterWithEngine(FlutterDesktopGpuSurfaceType type);

  flutter::TextureRegistrar* registrar_ = nullptr;
  std::shared_ptr<UnregisterTracker> unregistrations_;
  std::shared_ptr<flutter::TextureVariant> variant_;
  std::shared_ptr<SurfaceState> state_ = std::make_shared<SurfaceState>();
  int64_t texture_id_ = -1;
};

}  // namespace minigpu_view

#endif  // FLUTTER_PLUGIN_minigpu_view_D3D11_TEXTURE_HANDLER_H_
