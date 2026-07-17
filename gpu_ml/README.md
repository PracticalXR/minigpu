# gpu_ml

GPU-accelerated ML inference for Dart and Flutter on WebGPU, built on
[gpu_tensor](../gpu_tensor). Where gpu_tensor is math on dense tensors,
gpu_ml is everything that knows what a model, file, or token is:

- **GGUF loading** — pure-Dart parser (web-safe, `package:gpu_ml/gpu_ml.dart`)
  plus a streaming reader (`package:gpu_ml/gpu_ml_io.dart`) that range-reads
  tensors from multi-GB model files without loading them into memory.
- **Quantized weights** — `QuantizedTensor` keeps F16 / Q8_0 / Q4_0 weights in
  VRAM in their original GGML packing; fused dequant matVec kernels unpack in
  registers (no `shader-f16` extension required — works on every WebGPU
  target).
- **Transformer/SSM building blocks** — `rmsNorm`, `rope`, `silu`, `gelu`,
  composing with gpu_tensor's matMul/softmax/reshape/transpose into full
  decoder blocks (see `test/llama_block_test.dart`).

**Web/WASM:** `package:gpu_ml/gpu_ml.dart` is web-safe and compiles under
both dart2js and dart2wasm (gated by `test/web_compile_smoke_test.dart`) —
WebGPU inference runs in the browser via minigpu's `dart:js_interop`
bindings. Load models on web with `fetch` + `GgufFile.parse`; only the
`gpu_ml_io.dart` streaming reader is native-only.

**Status: experimental.** Roadmap: [`../GPU_ML_PLAN.md`](../GPU_ML_PLAN.md) —
target is llama.cpp-comparable local inference (currently bringing up a
Qwen3.6 MoE hybrid), then exposing loaded models as gpu_pipeline stages for
real-time transcription/captioning/vision inside AV pipelines.

```dart
import 'package:gpu_ml/gpu_ml_io.dart';

final model = await GgufStream.open(r'C:\models\model.gguf');
print(model.metadata['general.architecture']);

final w = await model.loadQuantized('blk.0.ffn_gate_shexp.weight');
final y = await w.matVec(x); // fused dequant GEMV, weights stay packed
```

Inspect any GGUF without loading it:

```console
dart run tool/gguf_inspect.dart C:\models\model.gguf
```
