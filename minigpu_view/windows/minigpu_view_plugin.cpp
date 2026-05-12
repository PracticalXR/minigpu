#include "include/minigpu_view/minigpu_view_plugin.h"
#include "minigpu_view_plugin_impl.h"

#include <flutter/encodable_value.h>
#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>

#include <chrono>
#include <memory>
#include <variant>

namespace minigpu_view {

// static
void MinigpuViewPlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarWindows* registrar) {
  auto channel =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          registrar->messenger(), "minigpu_view",
          &flutter::StandardMethodCodec::GetInstance());

  auto plugin = std::make_unique<MinigpuViewPlugin>(registrar);

  channel->SetMethodCallHandler(
      [plugin_pointer = plugin.get()](const auto& call, auto result) {
        plugin_pointer->HandleMethodCall(call, std::move(result));
      });

  registrar->AddPlugin(std::move(plugin));
}

MinigpuViewPlugin::MinigpuViewPlugin(flutter::PluginRegistrarWindows* registrar)
    : registrar_(registrar), textures_(registrar->texture_registrar()) {}

MinigpuViewPlugin::~MinigpuViewPlugin() = default;

namespace {

// Helper: extract a 64-bit int from an EncodableValue that may be int32 or
// int64. Method channel codec stores small ints as i32 by default.
int64_t GetInt64(const flutter::EncodableValue& v) {
  if (std::holds_alternative<int64_t>(v)) return std::get<int64_t>(v);
  if (std::holds_alternative<int32_t>(v)) return std::get<int32_t>(v);
  return 0;
}

int64_t NowMicros() {
  using namespace std::chrono;
  return duration_cast<microseconds>(
             system_clock::now().time_since_epoch())
      .count();
}

}  // namespace

void MinigpuViewPlugin::HandleMethodCall(
    const flutter::MethodCall<flutter::EncodableValue>& method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  const auto& method = method_call.method_name();
  const auto* args =
      std::get_if<flutter::EncodableMap>(method_call.arguments());
  if (!args) {
    result->Error("invalid_args", "Expected a map");
    return;
  }

  auto get = [&](const std::string& key) -> const flutter::EncodableValue* {
    auto it = args->find(flutter::EncodableValue(key));
    if (it == args->end()) return nullptr;
    return &it->second;
  };

  const auto* instance_v = get("instanceId");
  if (!instance_v) {
    result->Error("invalid_args", "Missing instanceId");
    return;
  }
  const int instance_id = static_cast<int>(GetInt64(*instance_v));

  if (method == "present") {
    const auto* kind_v = get("kind");
    const auto* handle_v = get("handle");
    const auto* shared_handle_v = get("sharedHandle");
    const auto* w_v = get("width");
    const auto* h_v = get("height");
    if (!kind_v || !w_v || !h_v) {
      result->Error("invalid_args", "Missing required keys");
      return;
    }
    const std::string kind = std::get<std::string>(*kind_v);
    if (kind != "nativeSharedTexture") {
      result->Error("unsupported",
                    "Windows plugin only supports nativeSharedTexture");
      return;
    }
    const int64_t handle = handle_v ? GetInt64(*handle_v) : 0;
    const int64_t shared_handle = shared_handle_v ? GetInt64(*shared_handle_v) : 0;
    const int width = static_cast<int>(GetInt64(*w_v));
    const int height = static_cast<int>(GetInt64(*h_v));

    if (width <= 0 || height <= 0) {
      result->Error("invalid_handle", "Bad size");
      return;
    }

    // Prefer the shared-handle path: it works cross-device, which is
    // required because Flutter's ANGLE has its own D3D11 device separate
    // from the producer's (e.g. Dawn's) D3D11 device.  The raw texture
    // pointer path only works if both producer and Flutter run on the
    // same D3D11 device, which is not the case in practice.
    auto it = handlers_.find(instance_id);
    if (shared_handle != 0) {
      void* shared_h = reinterpret_cast<void*>(shared_handle);
      if (it == handlers_.end()) {
        auto handler = std::make_unique<D3D11TextureHandler>(textures_);
        if (!handler->InitializeFromSharedHandle(shared_h, width, height)) {
          result->Error("init_failed",
                        "Failed to initialize D3D11 shared-handle texture handler");
          return;
        }
        it = handlers_.emplace(instance_id, std::move(handler)).first;
      } else {
        it->second->UpdateFromSharedHandle(shared_h, width, height);
      }
    } else {
      auto* tex = reinterpret_cast<ID3D11Texture2D*>(handle);
      if (!tex) {
        result->Error("invalid_handle", "Missing handle and sharedHandle");
        return;
      }
      if (it == handlers_.end()) {
        auto handler = std::make_unique<D3D11TextureHandler>(textures_);
        if (!handler->Initialize(tex, width, height)) {
          result->Error("init_failed",
                        "Failed to initialize D3D11 texture handler");
          return;
        }
        it = handlers_.emplace(instance_id, std::move(handler)).first;
      } else {
        it->second->Update(tex, width, height);
      }
    }

    flutter::EncodableMap reply{
        {flutter::EncodableValue("textureId"),
         flutter::EncodableValue(it->second->texture_id())},
        {flutter::EncodableValue("width"), flutter::EncodableValue(width)},
        {flutter::EncodableValue("height"), flutter::EncodableValue(height)},
        {flutter::EncodableValue("presentedAtUs"),
         flutter::EncodableValue(NowMicros())},
    };
    result->Success(flutter::EncodableValue(reply));
    return;
  }

  if (method == "dispose") {
    handlers_.erase(instance_id);
    result->Success();
    return;
  }

  result->NotImplemented();
}

}  // namespace minigpu_view

void MinigpuViewPluginRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  minigpu_view::MinigpuViewPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
