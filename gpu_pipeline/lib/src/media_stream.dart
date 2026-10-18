import 'dart:typed_data';
import 'package:gpu_tensor/gpu_tensor.dart';

enum MediaStreamType { audio, video, screen }

/// Kind of media frame payload.
enum MediaFrameKind { gpuTensor, typedData }

/// Unified frame wrapper (either a GPU Tensor or host Float32List).
class MediaFrame {
  final MediaFrameKind kind;
  final BufferDataType dataType;
  final Tensor? tensor;
  final TypedData? hostData;
  final List<int> shape;

  MediaFrame._(
    this.kind,
    this.tensor,
    this.hostData,
    this.dataType,
    this.shape,
  );

  factory MediaFrame.gpu(Tensor t) => MediaFrame._(
    MediaFrameKind.gpuTensor,
    t,
    null,
    BufferDataType.float32,
    t.shape,
  );

  factory MediaFrame.host(TypedData data, List<int> shape) => MediaFrame._(
    MediaFrameKind.typedData,
    null,
    data,
    BufferDataType.float32,
    List<int>.from(shape),
  );

  bool get isGpu => kind == MediaFrameKind.gpuTensor;
  bool get isHost => kind == MediaFrameKind.typedData;
}

/// Generic media stream that can emit GPU or CPU data.
abstract class MediaStream {
  final String id;
  final String deviceId;
  final String name;
  final int color;
  final MediaStreamType type;
  final dynamic context;

  bool _isActive = false;

  // New unified callbacks
  final List<void Function(MediaFrame)> _frameCallbacks = [];

  // Backward-compatible tensor-only callbacks
  final List<void Function(Tensor)> _tensorCallbacks = [];

  MediaStream({
    required this.id,
    required this.deviceId,
    required this.name,
    required this.color,
    required this.type,
    required this.context,
  });

  /// Start the stream
  Future<void> start() async {
    if (_isActive) return;
    await _startCapture();
    _isActive = true;
  }

  /// Stop the stream
  Future<void> stop() async {
    if (!_isActive) return;
    await _stopCapture();
    _isActive = false;
  }

  // New unified subscription
  void onDataUpdate(void Function(MediaFrame) callback) {
    _frameCallbacks.add(callback);
  }

  void removeDataCallback(void Function(MediaFrame) callback) {
    _frameCallbacks.remove(callback);
  }

  // Legacy GPU tensor subscription (kept for compatibility)
  void onTensorUpdate(void Function(Tensor) callback) {
    _tensorCallbacks.add(callback);
  }

  void removeTensorCallback(void Function(Tensor) callback) {
    _tensorCallbacks.remove(callback);
  }

  // Emit GPU tensor
  void notifyTensorUpdate(Tensor tensor) {
    // Legacy callbacks
    for (final cb in _tensorCallbacks) {
      cb(tensor);
    }
    // Unified callbacks
    if (_frameCallbacks.isNotEmpty) {
      final frame = MediaFrame.gpu(tensor);
      for (final cb in _frameCallbacks) {
        cb(frame);
      }
    }
  }

  // Emit host (CPU) data without forcing GPU upload
  void notifyHostData(Float32List data, List<int> shape) {
    if (_frameCallbacks.isEmpty) return;
    final frame = MediaFrame.host(data, shape);
    for (final cb in _frameCallbacks) {
      cb(frame);
    }
  }

  bool get isActive => _isActive;

  /// Override to implement specific capture logic
  Future<void> _startCapture();
  Future<void> _stopCapture();

  void dispose() {
    stop();
    _tensorCallbacks.clear();
    _frameCallbacks.clear();
  }
}
