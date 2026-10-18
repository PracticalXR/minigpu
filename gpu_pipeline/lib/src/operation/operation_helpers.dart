import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

/// Global resource cache for pipeline operations

class CachedShader {
  final ComputeShader shader;
  final String shaderCode;
  final DateTime createdAt;

  CachedShader({
    required this.shader,
    required this.shaderCode,
    required this.createdAt,
  });
}

/// Helper class for shape calculations and tensor management
class TensorShapeHelper {
  /// Calculate output shape based on operation type and inputs
  static List<int> calculateOutputShape(
    String operationType,
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
  ) {
    final firstInput = inputs.values.first;

    switch (operationType) {
      case 'reshape':
        final newShape = parameters['shape'] as List<int>?;
        if (newShape != null) return newShape;
        throw Exception('Reshape operation requires shape parameter');

      case 'transpose':
        final axes = parameters['axes'] as List<int>?;
        if (axes != null) {
          final shape = List<int>.from(firstInput.shape);
          final result = List<int>.filled(shape.length, 0);
          for (int i = 0; i < axes.length; i++) {
            result[i] = shape[axes[i]];
          }
          return result;
        }
        // Default transpose (reverse dimensions)
        return firstInput.shape.reversed.toList();

      case 'conv':
        // Simplified conv shape calculation
        final kernelShape =
            (inputs['kernel'] ?? inputs.values.skip(1).first).shape;
        final strideH = parameters['strideH'] as int? ?? 1;
        final strideW = parameters['strideW'] as int? ?? 1;
        final padH = parameters['padH'] as int? ?? 0;
        final padW = parameters['padW'] as int? ?? 0;

        if (firstInput.shape.length >= 4 && kernelShape.length >= 4) {
          final batch = firstInput.shape[0];
          final channels = kernelShape[0];
          final height =
              ((firstInput.shape[2] + 2 * padH - kernelShape[2]) / strideH)
                  .floor() +
              1;
          final width =
              ((firstInput.shape[3] + 2 * padW - kernelShape[3]) / strideW)
                  .floor() +
              1;
          return [batch, channels, height, width];
        }
        break;

      case 'sum':
      case 'mean':
        final axis = parameters['axis'] as int? ?? -1;
        if (axis == -1) {
          return [1]; // Reduce all dimensions
        } else {
          final shape = List<int>.from(firstInput.shape);
          shape.removeAt(axis);
          return shape.isEmpty ? [1] : shape;
        }

      case 'fft':
        // FFT typically preserves shape but might change last dimension
        return firstInput.shape;

      default:
        // Most operations preserve input shape
        return firstInput.shape;
    }

    return firstInput.shape;
  }

  /// Check if two shapes are compatible for element-wise operations
  static bool areShapesCompatible(List<int> shape1, List<int> shape2) {
    if (shape1.length != shape2.length) return false;
    for (int i = 0; i < shape1.length; i++) {
      if (shape1[i] != shape2[i]) return false;
    }
    return true;
  }

  /// Calculate total number of elements in a shape
  static int calculateElementCount(List<int> shape) {
    return shape.fold(1, (a, b) => a * b);
  }

  /// Calculate workgroup dispatch size for compute shaders
  static DispatchSize calculateDispatchSize(
    List<int> shape, {
    int workgroupSizeX = 256,
    int workgroupSizeY = 1,
    int workgroupSizeZ = 1,
  }) {
    final totalElements = calculateElementCount(shape);

    if (shape.length == 1) {
      // 1D dispatch
      final x = (totalElements + workgroupSizeX - 1) ~/ workgroupSizeX;
      return DispatchSize(x, 1, 1);
    } else if (shape.length == 2) {
      // 2D dispatch
      final x = (shape[1] + workgroupSizeX - 1) ~/ workgroupSizeX;
      final y = (shape[0] + workgroupSizeY - 1) ~/ workgroupSizeY;
      return DispatchSize(x, y, 1);
    } else if (shape.length >= 3) {
      // 3D dispatch
      final x = (shape[2] + workgroupSizeX - 1) ~/ workgroupSizeX;
      final y = (shape[1] + workgroupSizeY - 1) ~/ workgroupSizeY;
      final z = (shape[0] + workgroupSizeZ - 1) ~/ workgroupSizeZ;
      return DispatchSize(x, y, z);
    }

    // Fallback
    final x = (totalElements + workgroupSizeX - 1) ~/ workgroupSizeX;
    return DispatchSize(x, 1, 1);
  }
}

/// Dispatch size for compute shader execution
class DispatchSize {
  final int x, y, z;

  const DispatchSize(this.x, this.y, this.z);

  @override
  String toString() => 'DispatchSize($x, $y, $z)';

  @override
  bool operator ==(Object other) {
    return other is DispatchSize &&
        other.x == x &&
        other.y == y &&
        other.z == z;
  }

  @override
  int get hashCode => Object.hash(x, y, z);
}
