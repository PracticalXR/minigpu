// Linux stub — DMA-BUF/GBM zero-copy path is not yet implemented.
// All `present` calls fail with `unsupported` so the Dart side can
// surface UnsupportedPreviewException.

#include <flutter_linux/flutter_linux.h>
#include <gtk/gtk.h>

#define minigpu_view_PLUGIN(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), minigpu_view_plugin_get_type(), \
                              MinigpuViewPlugin))

G_DECLARE_FINAL_TYPE(MinigpuViewPlugin, minigpu_view_plugin, minigpu_view,
                     PLUGIN, GObject)

struct _MinigpuViewPlugin {
  GObject parent_instance;
};

G_DEFINE_TYPE(MinigpuViewPlugin, minigpu_view_plugin, G_TYPE_OBJECT)

static void method_call_cb(FlMethodChannel* channel, FlMethodCall* method_call,
                           gpointer user_data) {
  g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(
      fl_method_error_response_new("unsupported",
                                   "Linux GPU preview not yet implemented",
                                   nullptr));
  fl_method_call_respond(method_call, response, nullptr);
}

static void minigpu_view_plugin_class_init(MinigpuViewPluginClass* klass) {}

static void minigpu_view_plugin_init(MinigpuViewPlugin* self) {}

extern "C" __attribute__((visibility("default")))
void minigpu_view_plugin_register_with_registrar(FlPluginRegistrar* registrar) {
  MinigpuViewPlugin* plugin = minigpu_view_PLUGIN(
      g_object_new(minigpu_view_plugin_get_type(), nullptr));
  g_autoptr(FlStandardMethodCodec) codec = fl_standard_method_codec_new();
  g_autoptr(FlMethodChannel) channel = fl_method_channel_new(
      fl_plugin_registrar_get_messenger(registrar), "minigpu_view",
      FL_METHOD_CODEC(codec));
  fl_method_channel_set_method_call_handler(
      channel, method_call_cb, g_object_ref(plugin), g_object_unref);
  g_object_unref(plugin);
}
