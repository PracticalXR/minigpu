#ifndef FLUTTER_PLUGIN_minigpu_view_PLUGIN_IMPL_H_
#define FLUTTER_PLUGIN_minigpu_view_PLUGIN_IMPL_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>

#include "d3d11_texture_handler.h"

namespace minigpu_view {

// Plugin entry point. Owns one D3D11TextureHandler per Dart-side
// MinigpuPreviewController instance AND per source handle.
//
// PER-HANDLE, NOT PER-INSTANCE, and that is a resource-exhaustion fix. A
// producer that multi-buffers (double/triple buffered publish) presents a
// DIFFERENT shared handle each frame. With one handler per instance, every
// present re-pointed the same Flutter texture at a new handle, and the
// engine re-created its ANGLE offscreen swapchain (color + depth surfaces)
// for every content frame — thousands of allocations per minute, ending in
// E_OUTOFMEMORY ("SwapChain11: Could not create depthstencil surface").
// With one texture per handle, a ring of N handles costs N swapchains TOTAL,
// created once and reused; a present of a known handle is just
// MarkTextureFrameAvailable, and the Dart controller already propagates the
// changing textureId to its widget.
//
// kMaxHandlersPerInstance bounds a misbehaving producer (a NEW handle every
// frame would otherwise grow this map like the old leak): least-recently
// presented is evicted. A triple ring uses 3; 4 leaves headroom for one
// resize transition without churn.
class MinigpuViewPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(
      flutter::PluginRegistrarWindows* registrar);

  explicit MinigpuViewPlugin(flutter::PluginRegistrarWindows* registrar);

  virtual ~MinigpuViewPlugin();

  // Disallow copy and assign.
  MinigpuViewPlugin(const MinigpuViewPlugin&) = delete;
  MinigpuViewPlugin& operator=(const MinigpuViewPlugin&) = delete;

 private:
  static constexpr size_t kMaxHandlersPerInstance = 4;

  struct HandlerEntry {
    std::unique_ptr<D3D11TextureHandler> handler;
    uint64_t last_use = 0;
  };

  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue>& method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  flutter::PluginRegistrarWindows* registrar_;
  flutter::TextureRegistrar* textures_;
  // instanceId -> (source key -> handler). The key is the shared handle when
  // present, else the raw texture pointer.
  std::map<int, std::map<int64_t, HandlerEntry>> handlers_;
  uint64_t use_counter_ = 0;
};

}  // namespace minigpu_view

#endif  // FLUTTER_PLUGIN_minigpu_view_PLUGIN_IMPL_H_
