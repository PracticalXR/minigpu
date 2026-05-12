/// The widget that displays a [MiniavPreviewController]'s current frame.
library;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/widgets.dart';

import 'preview_controller.dart';

/// Renders the current GPU frame from a [MiniavPreviewController].
///
/// Until the first [MiniavPreviewController.present] call completes,
/// shows [placeholder] (or empty space).
class MiniavGpuPreview extends StatelessWidget {
  const MiniavGpuPreview({
    super.key,
    required this.controller,
    this.fit = BoxFit.contain,
    this.alignment = Alignment.center,
    this.filterQuality = FilterQuality.medium,
    this.placeholder,
  });

  final MiniavPreviewController controller;
  final BoxFit fit;
  final Alignment alignment;
  final FilterQuality filterQuality;
  final Widget? placeholder;

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: controller,
      builder: (context, _) {
        final id = controller.textureId;
        final size = controller.size;
        if (id == null || size.isEmpty) {
          return placeholder ?? const SizedBox.shrink();
        }
        return FittedBox(
          fit: fit,
          alignment: alignment,
          child: SizedBox(
            width: size.width,
            height: size.height,
            child: kIsWeb
                ? HtmlElementView(viewType: 'minigpu_view::$id')
                : Texture(
                    textureId: id,
                    filterQuality: filterQuality,
                    freeze: false,
                  ),
          ),
        );
      },
    );
  }
}
