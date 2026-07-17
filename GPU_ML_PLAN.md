# gpu_ml — package plan & path to llama.cpp-comparable inference

READ FIRST for ML-inference sessions. Companion to gpu_tensor/UPDATE_PLAN.md
(Phases 0-4 there are DONE and are the foundation this builds on).
Written 2026-07-16 after inspecting the local target models.

## Target models (inspected with gpu_tensor/tool/gguf_inspect.dart)

`C:\models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-*.gguf`

| file | size | tensor types | new kernels needed |
|------|------|--------------|--------------------|
| Q8_K_P | 40.6 GB | F32(301) F16(77) Q8_0(355) | NONE — all implemented |
| Q5_K_P | 26.1 GB | F32(301) Q8_0(97) Q5_K(285) Q6_K(50) | Q5_K, Q6_K |

Architecture `qwen35moe` (metadata):
- 40 blocks, embed 2048, vocab 248320, context 262144
- HYBRID: `full_attention_interval: 4` — blk.3,7,11,... are full attention
  (attn_q/k/v/output + q_norm/k_norm [256]); all OTHER layers are linear
  attention / Gated-DeltaNet-style SSM (attn_qkv [2048,8192], attn_gate
  [2048,4096], ssm_conv1d k=4, ssm_a/alpha/beta/dt (32 heads?), ssm_norm
  [128], ssm_out [4096,2048], state_size 128, group_count 16)
- Attention: 16 Q heads / 2 KV heads (GQA 8:1), head dim 256 (key_length),
  QK-norm, PARTIAL RoPE (rope.dimension_count 64 of 256, freq_base 1e7,
  MRoPE dimension_sections [11,11,10,0] — text-only path uses section 0..?
  VERIFY against llama.cpp qwen35moe source)
- MoE FFN every layer: router ffn_gate_inp [2048,256] f32 → top-8 of 256
  experts (expert ffn 512) + shared expert (512) with sigmoid gate
  (ffn_gate_inp_shexp); SiLU-gated (gate/up/down)
- Tokenizer: gpt2 byte-level BPE, pre=qwen35, 247587 merges
- Sampling defaults in metadata: temp 1.0, top_k 20, top_p 0.95
- KV cache is TINY (only 10 attention layers x 2 KV heads x 256 x 2(k,v)):
  ~2 KB/token f16 → long context cheap. SSM layers carry fixed-size state
  (128 x head x group) instead — recurrent, O(1) per token.

## VRAM budget (RTX 4090, 24 GB)

Q8 file: MoE expert tensors ≈ 34 GB of the 40.6; everything else ≈ 6.4 GB.
Active experts/token: 8/256 per layer ≈ 27 MB/layer ≈ 1.07 GB/token total.
Plan: non-expert weights VRAM-resident (6.4 GB) + expert LRU cache (~14 GB)
+ upload-on-miss over PCIe (25 GB/s → 43 ms/token worst case, far less with
cache hits + routing skew). Q5 file: experts ≈ 22 GB → ~85% cacheable.
=> Host-RAM residency + VRAM expert cache is REQUIRED, not optional.

## Library boundary — create gpu_ml NOW

Rule: gpu_tensor = math on dense tensors (no file formats, no model
semantics). gpu_ml = anything that knows what a model/file/token is.

Moves from gpu_tensor to gpu_ml (created there in Phase 4 bring-up):
`src/gguf.dart`, `src/gpu_quant.dart`, `src/gpu_nn.dart`,
`tool/gguf_inspect.dart`, tests (gguf_quant/gpu_nn/llama_block).
gpu_tensor keeps: base/ops/linear/activation/pooling/transform/data/print.
Deps: gpu_ml -> gpu_tensor -> minigpu. Why now: gpu_tensor is published and
consumed by AV users (gpu_pipeline); the ML surface is about to 5x
(tokenizer, runner, MoE, SSM, residency manager); different release cadence.

## Pipeline structure (gpu_pipeline) — how it fits

- Do NOT build the model on gpu_pipeline's stage graph: LLM decode needs
  dynamic control flow (router top-k per token, recurrent state, cache
  eviction) that fights a fixed AV stage DAG.
- gpu_ml gets its own `ForwardPlan`: all shaders compiled + buffers bound at
  LOAD time, per token only param writes + dispatch loop (zero setup/token).
- The SHARED need is minigpu-level COMMAND BATCHING (record N dispatches,
  one submit, one await). ~500+ dispatches/token here; at ~0.2-1 ms sync
  each that's 100-500 ms/token of pure overhead — THE tok/s lever. Build it
  once in minigpu; gpu_pipeline gets it for free (per-frame win too).
- Later integration: expose a loaded gpu_ml model AS a gpu_pipeline stage
  (live captioning / ASR / vision inside AV pipelines — livetensor venues).

## Milestones

M0 — gpu_ml bootstrap (mechanical)  [DONE 2026-07-16]
  - [x] minigpu/gpu_ml created; gguf/gpu_quant/gpu_nn srcs + inspector tool +
        3 test suites moved from gpu_tensor; gpu_tensor now exports
        gpu_helpers (cachedShader/dispatchLinear are public conventions for
        downstream kernel authors). Suites: gpu_tensor 139/0, gpu_ml 21/0.
  - [x] Streaming reader: lib/gpu_ml_io.dart `GgufStream` (header-chunk parse
        + per-tensor range reads; loadQuantized/loadF32 disk→VRAM). Web path
        keeps in-memory GgufFile.parse. VERIFIED against the real 40.6 GB
        Qwen3.6 Q8_K_P file (test/real_model_smoke_test.dart, skips when
        C:\models absent): header/arch asserts, f32 norm sanity, Q8_0 GPU
        dequant == CPU dequant of real weights, fused matVec row-sums match.

M1 — kernel set for the Qwen3.6 family
  - [x] Q5_K + Q6_K fused matVec + dequantize (2026-07-16, suite 29/0).
        CPU reference decoders in lib/src/quant_cpu.dart (dequantizeCpu for
        f32/f16/q8_0/q4_0/q5_k/q6_k).  VALIDATED THREE WAYS on real files
        (test/kquant_real_test.dart): GPU==CPU decode exact on real
        token_embd (Q5_K) + output.weight (Q6_K); fused matVec == decode+dot;
        CROSS-FILE Q5_K_P-vs-Q8_K_P rel-RMS small (independent llama.cpp
        encodings — catches layout misreads a shared-bug reference cannot).
  - [x] Expert-indexed matVec: QuantizedTensor supports [experts, rows, cols]
        stacks; expert byte offset travels in a params buffer so ONE cached
        shader serves all experts (no per-expert shader-cache explosion).
        GgufStream.readTensorBytes gained byteOffset/byteLength range reads —
        the expert-streaming primitive (verified: range-read ONE real expert
        out of blk.0.ffn_gate_exps and matVec'd it correctly).
  - [x] MoE combine (2026-07-17, gpu_ml 31/0 + gpu_tensor 145/0): MoeFfn in
        lib/src/gpu_moe.dart — router matVec + softmax + CPU top-k (route()
        exposed for tests/prefetchers), weighted expert FFNs via
        expert-indexed matVec, shared expert with scalar sigmoid gate.
        GgufStream.loadMoeFfn(blk) loads a whole layer.  VERIFIED: synthetic
        vs CPU + REAL blk.0 full-layer forward (855MB of expert stacks in
        VRAM) matches CPU decode reference.
        FOUND+FIXED along the way: minigpu never requested device limits →
        Dawn spec defaults (128MiB maxStorageBufferBindingSize) made
        >128MiB storage bindings fail CreateBindGroup with a SILENT
        uncaptured validation error (zero outputs).  minigpu_ffi buffer.cpp
        now requests the adapter's full limits (4090/D3D12: 2GiB buffer +
        binding).  Per-tensor ceiling is now 2GiB — sharding (M4) only
        needed beyond that or on web.
  - [x] SEMANTICS PINNED (2026-07-17): docs/QWEN35MOE_SEMANTICS.md — exact
        transcription of llama.cpp qwen35moe.cpp + delta-net-base.cpp +
        ggml rope/l2_norm/softplus.  Key facts: attn_q outputs INTERLEAVED
        per-head (q, gate); IMRoPE on text == NEOX partial rope (64 of 256
        dims, base 1e7); DeltaNet decode recurrence = S<-exp(g)S,
        d=beta(v - S^T k), S+=k(x)d, out=S^T q; k-head mapping is TILE
        (h % 16, ggml_repeat); l2_norm eps FLOORS the norm.
  - [x] AttentionLayer (lib/src/gpu_attn.dart): project() + attend() with
        new ropeNeox partial-rope kernel + GQA-grouped batched attention +
        sigmoid output gating.  VALIDATED vs CPU reference on REAL blk.3
        weights (relRms < 2e-3).
  - [x] DeltaNetLayer (lib/src/gpu_delta_net.dart): causal conv kernel
        (per-channel, rolls raw-input history in place) + delta-rule
        recurrence kernel (one workgroup per v-head, thread-per-v-dim,
        decay+update+readout in two row passes, state in place) + gated
        rmsNorm*silu(z) + out proj.  New primitives: ropeNeox, l2NormRows
        (gpu_nn).  VALIDATED vs CPU on REAL blk.0 weights across TWO decode
        steps (conv history + recurrent state evolution), relRms < 2e-3.
        Suite 35/0.
  - [ ] KV-cache append kernel (write-at-row-offset) + f16 KV cache;
        SSM state buffers (persistent, in-place).

M2 — runner + tokenizer  [FIRST GENERATION 2026-07-17 🎉]
  "The capital of France is" -> " Paris, a city renowned for its rich"
  (real 40.6 GB Q8_K_P file, greedy, entirely on WebGPU). Landed:
  - [x] BpeTokenizer (lib/src/bpe_tokenizer.dart, web-safe): gpt2 byte-level
        BPE from GGUF metadata, llama.cpp qwen35 pre-tokenizer regex;
        round-trip verified on the real 248320-token vocab.
  - [x] Qwen35Model runner (lib/src/qwen35_runner.dart): metadata-driven
        config, 40-block hybrid loop, residency v1 (norms + attn/delta +
        routers + shexps + lm_head resident ≈ 3.3 GB; experts disk-streamed
        through byte-budgeted LRU, default 8 GB — hit rate >50% within 8
        tokens), embedding rows range-read + CPU-dequant (2 KB each),
        greedy generate(). Load 3.3 s (warm OS cache).
  - [x] E2E test test/generation_e2e_test.dart (gated RUN_E2E=1): tokenizer
        round-trip + 8-token greedy + "contains Paris" sanity. PASSES.
  - Perf today ~4.5-5 s/token warm (~0.2 tok/s) — entirely M5 territory
        (500+ awaited dispatches/token + per-token KV rebuild + expert
        upload). Correctness first: done.
  Original scope notes below:
  - [ ] Config from GGUF metadata (qwen35moe key set), Model.load with
        residency policy, layer loop (SSM vs attention by interval), final
        norm + lm_head, logits.
  - [ ] gpt2 byte-level BPE tokenizer from metadata (tokens + merges +
        pre=qwen35 regex) — pure Dart, unit-tested against llama.cpp
        tokenization of fixture strings.
  - [ ] Sampling: greedy + temp/top-k/top-p (defaults from metadata);
        chat template application (metadata has the Jinja template — v1:
        hardcode the ChatML-ish equivalent, don't write a Jinja engine).

M3 — correctness gates (BEFORE perf)
  - [ ] Per-layer parity harness: llama.cpp eval-callback dumps layer
        activations for a fixture prompt; compare ours layer by layer.
  - [ ] Golden test: greedy tokens match llama.cpp for 50+ tokens on both
        files. Bring-up order: Q8_K_P FIRST (zero new quant kernels — pure
        architecture work), then Q5_K_P (adds Q5_K/Q6_K validation).
  - [ ] Optional de-risk: tiny dense GGUF (Qwen3-0.6B class) to shake out
        runner/tokenizer before MoE+SSM complexity.

M4 — memory tiering
  - [ ] Residency manager: norms/attn/router/shared-experts VRAM-resident;
        experts host-resident (file range reads or RAM cache) + VRAM LRU
        (~14 GB budget) with upload-on-miss; telemetry (hit rate, MB/token).
  - [ ] Async prefetch: overlap expert upload of layer L+1 with compute of
        layer L (needs minigpu async copy or second queue — investigate).

M5 — perf to "comparable"
  - [ ] minigpu command batching (record/submit/await-once) — do FIRST, it
        dwarfs everything else.
  - [ ] Prefill path: multi-token quantized matmul (x [tokens, cols]) so
        prompts aren't per-token GEMV loops; batched SSM scan for prefill.
  - [ ] Fusions: QK-norm+RoPE into projection epilogue; router+top-k on GPU;
        GEMV vec4/register blocking; subgroup ops (Dawn) where available.
  - [ ] Benchmark harness: tok/s prefill + decode vs llama.cpp same machine,
        tracked in-repo.

M5b — Web/WASM track (CONFIRMED goal; WebGPU on web is first-class)
  - [x] gpu_ml.dart is web-safe (no dart:io) and the full import graph
        (gpu_ml -> gpu_tensor -> minigpu_web dart:js_interop bindings)
        compiles under BOTH dart2js and dart2wasm — gated forever by
        test/web_compile_smoke_test.dart (compiles example/web_smoke.dart
        with both compilers; catches io-leaks / legacy-interop regressions).
  - [ ] Web model loading: ranged-fetch reader (HTTP Range requests)
        mirroring GgufStream + OPFS cache for downloaded weights; small
        models via fetch + in-memory GgufFile.parse work today.
  - [ ] Browser execution smoke: run the quant kernels in real Chrome
        WebGPU (dev_rig-style harness or flutter build web --wasm example);
        browser limits are stricter (128MB default binding size, buffer
        caps) — the limit-raising/sharding items in M4 matter doubly here.
  - [ ] Web-scale model reality: 27-40 GB MoE files are not web targets;
        web targets are small dense models (ASR/vision/small LLMs) — same
        kernels, GGUF via fetch.

M6 — gpu_pipeline integration (CONFIRMED goal: real-time transcription/
captions/CV inside AV pipelines)
  - [ ] `ModelStage` adapter: a loaded gpu_ml model exposed as a gpu_pipeline
        stage — input tensor(s) from upstream stages (audio features, video
        frames already in minigpu buffers = zero-copy), output tensor/text
        downstream. Model executes its own ForwardPlan internally; the stage
        contract only sees buffers.
  - [ ] First target: streaming ASR (whisper-family GGUF encoder-decoder or
        a streaming CTC model) fed by gpu_pipeline audio stages (mel
        spectrogram ALREADY exists in minigpu_av spectrogram stages) →
        captions for livetensor venues.
  - [ ] Vision: frame tensor (miniav capture → minigpu RGBA, zero-copy path
        proven in miniav codecs work) → ViT/CNN GGUF → detections/embeddings.
  - [ ] Cadence control: model stages run at their own rate (e.g. ASR every
        N audio frames), pipeline continues at frame rate — needs an async
        stage contract in gpu_pipeline (investigate existing DynamicStage).

Definition of "comparable": same GGUF in, same tokens out (greedy-match),
decode tok/s within ~2-3x of llama.cpp CUDA on the 4090 (WebGPU won't beat
tuned CUDA; the win is portable Dart/web/embedded + livetensor integration).

## Notes / risks
- MRoPE sections + attn_gate semantics need llama.cpp source verification —
  budget a reading session; wrong guesses here cost days in parity debugging.
- Q8_K_P filename vs content: quant suffix in the NAME doesn't match tensor
  types inside (it's Q8_0/F16/F32) — always trust gguf_inspect, not names.
- imatrix metadata present (quantize.imatrix.*) — irrelevant at inference.
- Everything stays f32 activations; f16 activations are a later perf lever.
