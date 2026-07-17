# gpu_tensor Update Plan

READ FIRST for gpu_tensor sessions. Status legend: [ ] todo, [x] done, [~] partial/deferred.
Origin: 2026-07-16 full-library review (approach + shader audit + GGML/ONNX importer brainstorm).

Test command (from `gpu_tensor/`):
```
dart --enable-experiment=native-assets test --concurrency 1
```

## Phase 0 — Verification harness  [DONE 2026-07-16]

- [x] CPU-reference DFT comparison test for fft1d/fft2d/fft3d — test/fft_reference_test.dart.
      CONFIRMED: fft2d/fft3d omitted bit-reversal (wrong values), mutated input,
      threw on real input. Existing tests only used delta-at-origin/DC inputs,
      which are bit-reversal-invariant and could never catch this.
- [x] fromBytes round-trip test for float32. CONFIRMED: prepareShader rewrote
      `input_bytes: array<u32>` to `array<f32>`; silent validation error -> zeros.

## Phase 1 — Correctness fixes  [DONE 2026-07-16, suite 126/0]

- [x] `prepareShader`: rewrite only canonical `array<f32>`, once, after substitution.
- [x] `modScalar`: exact literal (was toStringAsFixed(1): 0.25 -> 0.3).
- [x] `reshape`/`reshapeView`: new `Tensor.view` (retains parent, no second
      finalizer, destroy() on a view is a no-op for the shared buffer).
- [x] fft2d/fft3d: per-axis `_bitReverseAxis` reorder before each axis's stages;
      verified against CPU reference DFT (shifted delta + random inputs).
- [x] fft2d/fft3d: never write into `this`; two owned scratch tensors ping-pong,
      loser destroyed. Real-input upgrade path fixed (upgradeRealToComplex now
      appends the complex dim for rank>=2, matching its doc). fft(isRealInput:
      true) routes rank-2/3 to fft2d/fft3d.
- [x] `gpu:`/`dataType:` passthrough on all result allocations (activations,
      max/min/avg, pooling, conv, FFT scratch).
- [x] `_copyAllData` guard (compared input.size to itself).
- [x] `avg`/`avgWeighted`: chunked `_weightedSum`, max 7 inputs + output per
      dispatch (8-binding WebGPU default limit), accumulate across passes.
- [x] `getElement`: single-element readback via Buffer.read readOffset.
- [x] BONUS: transpose shader had `array<dtype;` (missing `>`) — never compiled;
      fused notEqualTo/gte/lte into single-dispatch kernels (were 3 dispatches +
      leaked intermediate).

## Phase 2 — Foundations (perf + scale)  [DONE 2026-07-16 except deferred]

- [x] Zero-upload dropped in `Tensor._` (WebGPU zero-initializes). `zeros()` is
      now allocation-only (no dispatch). resize/padTo redundant fill(0) removed.
- [x] Dispatch folding: `dispatchLinear` + num_workgroups-based linear index in
      ALL 1D kernels (source stays dispatch-size-independent -> cache friendly).
      Batched matMul still caps at 65535 batches (gid.z) — flagged inline.
- [x] All steady-state ops on `gpu.cachedShader` (matMul, conv/conv2d,
      activations, pooling, fromBytes, slice, FFT stage + reorder kernels).
      Kept per-call compile ONLY where source varies per call by design
      (random: seed; fill/setElement: value baked).
- [x] matMul: col=gid.x/row=gid.y coalesced mapping (2D and batched).
- [~] fft2d/fft3d: param-buffer per stage (fft1d pattern) instead of per-stage
      source. cachedShader makes steady-state OK; param-buffer still cleaner. Defer.
- [x] GPU-side `slice` + `sliceLinear` (strided gather kernel; old flat-sublist
      readback was also WRONG for non-contiguous inner-dim slices).
- [x] `random`: PCG hash of (seed, i) — old per-element LCG on seed+i was
      correlated across adjacent elements. (Pulled forward from Phase 3.)

## Phase 3 — Kernel quality  [GGML-blocking items DONE 2026-07-16, suite 139/0]

- [x] Tiled shared-memory matmul (16x16 var<workgroup> tiles, 2D + batched) +
      GEMV fast path for M==1 (decode hot path). Register blocking = future.
      Tests: test/phase3_reference_test.dart (non-tile-multiple dims, batched,
      GEMV vs CPU). TRAP: `active` is a reserved WGSL keyword (like `ref`).
- [x] Workgroup-per-row softmax for d >= 64 (strided local reduce + shared-mem
      tree; rows folded over x/y workgroups). Small-d keeps per-element kernel.
      Workgroup-parallel sum/max/min/argmax reductions still TODO.
- [ ] Single-dispatch shared-memory FFT (Stockham) for n <= 4096 — one dispatch
      instead of log2(n) with CPU syncs; the gpu_pipeline spectrogram sizes.
- [x] Broadcasting elementwise add/subtract/multiply/divide (NumPy semantics,
      stride-0 gather; legacy equal-size-different-shape flat path preserved).
      Operator overloads inherit it. mod/min/max/comparisons still TODO.
- [ ] Conv: settle layout on NCHW (doc says HWC, shader indexes planar, kernel is
      HWIO — inconsistent today); im2col+GEMM or tiled direct conv.
- [ ] Pooling: fold max/min duplication into one generator; move load inside mask
      check; ONNX-style begin/end pads (current formula uses single pad, not 2*pad).
- [x] `random`: replace correlated per-element LCG (seed+i) with PCG/pcg3d hash.
      (done 2026-07-16 with Phase 2)
- [ ] minigpu: command-batch API (record N dispatches, one submit, sync at outputs).
      Prereq for graph execution; also fixes FFT per-stage sync cost.

## Phase 4 — GGML/ONNX importer groundwork (design in review notes)

STARTED 2026-07-16 (suite 149/0):
- [x] GGUF v2/v3 parser: lib/src/gguf.dart — pure Dart, bytes-based, web-safe
      (manual u64 combine, no getUint64), metadata KV (all 13 value types incl.
      arrays), tensor directory (ne innermost-first -> .shape reversed),
      general.alignment handling, per-tensor byte views. Plus f16 encode/decode
      helpers (floatToHalfBits/halfBitsToFloat).
- [x] QuantizedTensor: lib/src/gpu_quant.dart — F16/Q8_0/Q4_0 weights stay in
      VRAM in original packing (raw u32 upload, byte-addressed WGSL accessors
      since 34B/18B blocks are unaligned; unpack2x16float for f16 scales — no
      shader-f16 extension needed). dequantize() debug kernel + FUSED matVec
      (workgroup-per-row + shared-mem reduce). GgufFile.loadQuantized/loadF32.
      Tests: test/gguf_quant_test.dart (in-test GGUF writer + CPU quantizers
      using stored f16-rounded scales -> tight agreement).
- [x] Transformer NN kernels: lib/src/gpu_nn.dart — rmsNorm (workgroup-per-row
      reduce), rope (llama "norm"/interleaved-pair convention, positionOffset
      for decode), silu, gelu (tanh approx). Tests: test/gpu_nn_test.dart.
- [x] FULL LLAMA DECODER BLOCK E2E (test/llama_block_test.dart, suite 156/0):
      single-token decode step vs CPU double reference using identical
      effective (Q8_0-dequant) weights — rmsNorm -> Q8_0 fused matVec QKV ->
      rope -> per-head attention as ONE batched pipeline
      (reshape [heads,1,hd] @ transpose-K [heads,hd,seq] -> softmax ->
      @ V [heads,seq,hd] -> reshape concat) -> Wo -> residual ->
      rmsNorm -> SiLU-gated FFN -> residual. Everything on GPU except
      KV-cache concat (readback+re-upload; GPU append op = gap below).

Remaining for real-model parity (see "llama.cpp comparison" below):
- [ ] Streaming/chunked GGUF load (current parser takes whole file in memory)
      + real-model smoke test (needs a small .gguf on disk).
- [ ] K-quants (Q4_K/Q5_K/Q6_K fused kernels) — what current model uploads
      actually ship; Q4_0/Q8_0 covers legacy quants only.
- [ ] KV-cache append op (write-at-row-offset kernel) so decode never round
      trips through the CPU; f16 KV cache for memory.
- [ ] GQA (grouped-query attention): n_kv_heads < n_heads head-index mapping.
- [ ] Tokenizer (SentencePiece/BPE from GGUF metadata) + sampling
      (greedy/top-k/top-p/temperature) — CPU-side, pure Dart.
- [ ] Model runner: parse GGUF metadata (n_layer/n_head/n_embd/rope params),
      build layer loop, embedding gather + final norm + lm_head matVec.
- [ ] Prefill path: matVec is decode-only (one token); prompt ingestion needs
      quantized matMUL (x as [tokens, cols]) or per-token loop (slow but ok).
- [ ] Perf: minigpu command batching (one submit per token instead of ~10
      dispatches x n_layer round trips), GEMM register blocking, RoPE fused
      into QKV, maybe subgroup ops on Dawn.

Data types remaining: bf16 bitcast, int64->int32 downcast policy, ONNX QDQ.

Ops (transformer-first): GEMM semantics (transA/B, alpha/beta, fused bias), RMSNorm/
LayerNorm/GroupNorm fused rows, GELU/SiLU, RoPE, masked/causal softmax, KV-cache
append/views, Gather (embedding), Scatter, Concat/Split/Slice/Expand/Tile/Where/Cast/
Cumsum/TopK, multi-axis reductions + keepdims. Vision tier: NCHW Conv (groups/autopad),
ConvTranspose, AveragePool, Resize, folded BatchNorm.

Runtime: graph executor + liveness-based arena allocator, params-buffer shaders so
dynamic shapes don't recompile, GGUF parser + chunked file->GPU weight streaming,
maxStorageBufferBindingSize handling (128MB default; shard big matrices), per-op
golden tests vs onnxruntime/llama.cpp.

Sequencing: foundations (Ph1+2) -> GEMM/softmax/broadcast (Ph3) -> f16 + GGUF + Q4/Q8
fused kernels + RMSNorm/RoPE/GELU = small llama end-to-end -> ONNX breadth.
