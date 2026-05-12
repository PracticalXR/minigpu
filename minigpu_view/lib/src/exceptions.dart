/// Exceptions thrown by the minigpu_view widget pipeline.
library;

/// Thrown when the current platform cannot present a given source.
///
/// On platforms whose GPU surface API isn't yet wired (Linux pre-GBM,
/// for example), apps should catch this and fall back to a CPU rendering
/// path.
class UnsupportedPreviewException implements Exception {
  final String message;
  const UnsupportedPreviewException(this.message);

  @override
  String toString() => 'UnsupportedPreviewException: $message';
}

/// Thrown when the platform plugin reports an error wiring the GPU
/// resource into Flutter's texture registry (e.g. handle is invalid,
/// device was lost, format mismatch).
class PreviewPresentException implements Exception {
  final String message;
  final Object? cause;
  const PreviewPresentException(this.message, {this.cause});

  @override
  String toString() =>
      'PreviewPresentException: $message${cause != null ? ' ($cause)' : ''}';
}
