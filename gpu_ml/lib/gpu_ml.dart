/// GPU-accelerated ML inference on WebGPU.
///
/// Builds on gpu_tensor: GGUF model loading, quantized weights held in VRAM
/// in their original packing with fused dequant kernels, and the
/// transformer/SSM building blocks needed for llama/qwen-family inference.
///
/// This entrypoint is web-safe (no dart:io).  For streaming GGUF loading
/// from local files, import `package:gpu_ml/gpu_ml_io.dart`.
library;

export 'package:gpu_tensor/gpu_tensor.dart';

export 'src/gguf.dart';
export 'src/gpu_quant.dart';
export 'src/gpu_nn.dart';
export 'src/gpu_moe.dart';
export 'src/gpu_attn.dart';
export 'src/gpu_delta_net.dart';
export 'src/bpe_tokenizer.dart';
export 'src/quant_cpu.dart';
