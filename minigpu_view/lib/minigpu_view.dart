/// Zero-copy Flutter rendering for minigpu / miniav GPU resources.
///
/// Provides a [MiniavGpuPreview] widget that displays a [PreviewSource]
/// without any CPU readback on every frame.
///
/// ```dart
/// final controller = MiniavPreviewController();
/// final shared = gpu.createSharedOutputTexture(W, H);
/// // ... write to `shared` via minigpu pipeline ...
/// await controller.present(shared.asPreviewSource());
///
/// // In your widget tree:
/// MiniavGpuPreview(controller: controller)
/// ```
library;

export 'src/preview_source.dart';
export 'src/preview_controller.dart';
export 'src/preview_widget.dart';
export 'src/adapters/minigpu_adapter.dart';
export 'src/adapters/miniav_adapter.dart';
export 'src/exceptions.dart';
