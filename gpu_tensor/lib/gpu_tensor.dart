/// GPU Tensor library.
///
/// This library provides a simple API for performing tensor operations
/// on the GPU.
library;

export 'src/gpu_tensor_base.dart';
export 'src/gpu_activation.dart';
export 'src/gpu_transform.dart';
export 'src/gpu_data.dart';
export 'src/gpu_print.dart';
export 'src/gpu_pooling.dart';
export 'src/gpu_ops.dart';
export 'src/gpu_linear_ops.dart';
// Shader-authoring helpers (cachedShader, dispatchLinear, prepareShader,
// getWGSLType) — public so downstream packages (gpu_ml, gpu_pipeline) can
// write kernels that follow the same conventions.
export 'src/gpu_helpers.dart';
export 'package:minigpu/minigpu.dart' show BufferDataType;
