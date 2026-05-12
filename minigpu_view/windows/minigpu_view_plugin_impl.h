#ifndef FLUTTER_PLUGIN_minigpu_view_PLUGIN_IMPL_H_
#define FLUTTER_PLUGIN_minigpu_view_PLUGIN_IMPL_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>

#include <map>
#include <memory>

#include "d3d11_texture_handler.h"

namespace minigpu_view {

// Plugin entry point. Owns one D3D11TextureHandler per Dart-side
// MiniavPreviewController instance (keyed by `instanceId`).
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
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue>& method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

  flutter::PluginRegistrarWindows* registrar_;
  flutter::TextureRegistrar* textures_;
  std::map<int, std::unique_ptr<D3D11TextureHandler>> handlers_;
};

}  // namespace minigpu_view

#endif  // FLUTTER_PLUGIN_minigpu_view_PLUGIN_IMPL_H_
