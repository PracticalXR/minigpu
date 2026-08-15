/// Zero-copy Flutter rendering for minigpu / miniav GPU resources.
///
/// Provides a [MiniavGpuPreview] widget that displays a [PreviewSource]
/// without any CPU readback on every frame.
///
/// TWO WAYS IN, and the difference is who owns the destination texture.
///
/// **You already have a texture** — present it directly. No copies:
///
/// ```dart
/// final controller = MinigpuPreviewController();
/// final shared = gpu.createSharedOutputTexture(W, H);
/// // ... write to `shared` via minigpu pipeline ...
/// await controller.present(shared.asPreviewSource());
///
/// MiniavGpuPreview(controller: controller)   // in your widget tree
/// ```
///
/// **You have frames in whatever shape a producer gave you** — use a
/// [GpuPane], which owns the destination and takes any of them:
///
/// ```dart
/// final pane = GpuPane(gpu);
/// await pane.showSharedHandle(handle, width: w, height: h);  // capture/decode
/// await pane.showBuffer(buffer, width: w, height: h);        // compute output
/// await pane.showBytes(rgba, width: w, height: h);           // host memory
///
/// MiniavGpuPreview(controller: pane.controller)
/// ```
///
/// A pane costs one GPU-to-GPU blit per displayed frame and buys two things:
/// it accepts every source shape without the caller writing a display path per
/// producer, and — because the caller drives [GpuPane.flush] — it makes the
/// pane's update rate something you control rather than something the producer
/// decides. Host-memory frames are uploaded and composited on the GPU, never
/// decoded into an image on the UI isolate.
library;

export 'src/gpu_frame.dart';
export 'src/gpu_pane.dart';
export 'src/preview_source.dart';
export 'src/preview_controller.dart';
export 'src/preview_widget.dart';
export 'src/adapters/minigpu_adapter.dart';
export 'src/adapters/miniav_adapter.dart';
export 'src/adapters/miniav_frame_adapter.dart';
export 'src/planar_convert.dart';
export 'src/exceptions.dart';
