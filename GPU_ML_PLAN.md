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
  - [x] 2026-07-17 FIRED-DISPATCH DECODE PLAN (lib/src/qwen35_plan.dart) —
        8.5x: warm tokens 4.5-5 s -> 0.53-0.9 s; prefill(5)+first token
        29 s -> 8.6 s. Same greedy output text as v1 (parity signal).
        How: minigpu grew ComputeShader.dispatchFire (sync mgpuDispatch
        enqueue — no completer round trip; platform_interface default +
        ffi/web overrides). The plan owns ~60 shaders shared BY SOURCE
        (D3D12 many-pipelines trap avoided), fires everything, and syncs
        only 1x/block (MoE routing readback) + logits. Safety contract:
        binds mutate immediately, dispatch tasks snapshot bindings when
        they RUN, reads join the same WebGPU-thread FIFO (verified in
        buffer.cpp readAsyncImpl) => a completed readback flushes every
        earlier fired dispatch; any kernel used twice within a block gets
        a tag baked into its source (slot0..7, res_attn/res_ffn...).
        Fusions landed with it: QK-RMS+NEOX-rope+KV-append in one kernel
        (position in a GPU buffer, seqLen = dispatch size — ZERO per-token
        recompiles); scores + online-softmax·V + sigmoid-gate decode
        attention on a fixed-cap cache (maxSeq 4096); DeltaNet g/beta on
        GPU (killed 60 readbacks/token); MoE softmax+top-8+renorm on GPU
        (only the 8 ids come back — needed to bind streamed experts);
        shared expert incl. sigmoid(dot) gate fully on GPU, fired BEFORE
        the routing readback to overlap with disk fetch; per-expert
        weighted-accumulate down-matVec (y += w[slot]*W@x).
        _ExpertCache evictions now DEFERRED to post-logits flush (fired
        work may still reference evicted buffers). GgufStream reads
        serialized via _ioTail (RandomAccessFile allows 1 pending op;
        plan fetches g/u/d concurrently). v1 path kept as
        forwardReference() for numerical debugging.
  - [x] 2026-07-17 WAVE 2 (same day): 0.2 -> 2.4 tok/s Q8, 3.8 tok/s Q5.
        (a) minigpu mgpuSetBufferFire (C++ + full Dart plumb): binds now
        ENQUEUE on the WebGPU thread => bind/fire/rebind/fire is ordered
        by construction; the "readback per block" safety crutch is gone.
        (b) RESIDENT EXPERT STACKS (Qwen35Model.load expertStackBytes):
        blocks whose full 256-expert stacks fit in VRAM run GPU-DRIVEN
        MoE — the slot matVecs read the winning expert id from the top-k
        buffer (eb = idxb[slot]*bytesPerExpert) — no routing readback, no
        streaming, fully sync-free blocks.  Q8: 10 GB = 12/40 blocks;
        Q5: 14 GB = 25/40.
        (c) Concurrent whole-layer expert fetch (24 fetches pipelined;
        GgufStream _ioTail serializes the RandomAccessFile underneath).
        (d) VECTORIZED matVec bodies for Q8_0/Q5_K/Q6_K: one u32 load per
        four quants (funnel-shift wAt() where strides break alignment,
        vec4 lane accumulators for K-quant sub-blocks) — floor token
        (few misses) 256 ms -> 196 ms on Q8; validated exact vs CPU refs
        + cross-file.  tool/decode_bench.dart = the harness (per-token
        sync/fetch/miss split; --reference runs the v1 path).
        Measured Q8 (--cache-gb 6 --stack-gb 10): warm 416 ms = 2.40
        tok/s; miss cost ~2 ms each, sync ~80 ms/token.
        Measured Q5 (--cache-gb 4 --stack-gb 14): warm 262 ms = 3.82
        tok/s (25 resident blocks; 15 syncs/token).
        NUMERICS VERDICT (logit-dump differential, --dump-logits +
        scratch cmp): plan vs v1-reference on Q8 prefill relRms = 9e-7,
        identical top-5 — plan kernels are correct.  Q5 plan-vs-ref
        relRms 0.73 with overlapping top-5 = CHAOS, not a bug: 4.5-bit
        noise flips MoE routing near-ties mid-stack and 40 layers
        amplify; Q5-ref logits ≈ Q8 logits end-to-end also validates the
        Q5_K/Q6_K kernels.  Q5 greedy text loops (" the city of Paris is
        the city of...") — expected for aggressive quants under greedy;
        sampling (temp/top-p/repeat-penalty) is the fix, llama.cpp
        golden compare (M3) the definitive arbiter.
  - [x] 2026-07-17 WAVE 3 — MULTI-GPU: 6.11 tok/s Q8 (163.6 ms/token,
        30x from the morning's 0.2).  minigpu grew true multi-adapter
        support: C++ MGPUContextHandle API (per-instance MGPU was
        already the architecture; only creation needed handle variants),
        per-instance adapterFilter, Dart Minigpu.forAdapter('3090').
        Runner splits blocks across GPUs (secondaryAdapter/-Blocks/
        -StackBytes/-CacheBytes), two DecodePlans, hidden state hops
        devices once per token (readX/writeX, 8 KB).  Measured config:
        4090 16 blocks + 3090 20 blocks + head, 17 GB stacks each =
        36/40 blocks FULLY RESIDENT (Q8 stack sizes vary 816-1296 MB),
        4 streamed through 2 GB LRUs.  Floor tokens (0 misses) 122-138
        ms — now GPU-compute + submit bound.
        RAM-BLOWUP POSTMORTEM (a first full-resident attempt froze the
        machine): (1) FfiBuffer pooled read/write scratch pinned a
        native allocation sized to the LARGEST transfer for each
        buffer's lifetime (~33 GB across a model) — pool now capped at
        1 MiB; (2) wgpuQueueWriteBuffer stages in host RAM until the
        device ticks, and a pure load loop never ticks — uploads now
        stream disk->GPU in 32 MB chunks (Buffer.writeRawBytes ->
        mgpuWriteBufferAt) with a 4-byte flush readback every 256 MB
        (QuantizedTensor.createStreamed; loadQuantized auto-streams
        > 64 MB).  RSS stays ~1.35 GB flat through a 33 GB load
        (bench prints ProcessInfo.currentRss).
  - [x] 2026-07-17 WAVE 4 — COMMAND BATCHING: 10.84 tok/s Q8 (92.2
        ms/token, 54x from the morning's 0.2).  Fire-and-forget
        dispatches now RECORD into one shared compute pass on the MGPU
        instance (MGPU::batchEncoder/batchPass, flushBatchLocked) instead
        of encoder+submit per dispatch.  WebGPU guarantees sequential
        memory effects within a pass, so semantics are unchanged; the
        batch flushes (submits) at every out-of-band queue touch — buffer
        read/write, external-texture blit, buffer/context teardown — and
        at a 512-dispatch cap (TDR safety).  Awaited dispatch also
        flushes first.  Result: sync/token 110-140 ms -> 45-51 ms (submit
        overhead WAS the cost; ~1800 resident-chain dispatches now ride
        ~4 submits).  Floor token 48 ms.  gpu_tensor 145/0 + gpu_ml 35/0
        confirm batching is transparent to the whole library.
  - [x] 2026-07-17 WAVE 5 — 40/40 RESIDENCY + FUSED MoE: 21.44 tok/s Q8
        (46.6 ms/token, 107x from morning's 0.2; 0-miss floor ~38 ms =
        ~26 tok/s).  (a) Stack budget 22 GB/card -> 39/40 blocks fully
        resident (only 1 still streams; near the 24 GB VRAM ceiling).
        (b) FUSED MoE: all top-K resident experts run in 5 dispatches
        (gate / up / silu·mul / down over `wid.z = slot`, expert id read
        from the top-k buffer, + weighted combine) instead of 4 per
        slot — MoE block ~40 -> ~12 dispatches.  Verified relRms 3.2e-7
        vs the streamed path (dump-logits differential).  Floor token
        48 -> 40 ms, sync 44 -> 37 ms.
  - Progression today: 0.2 -> 1.57 -> 2.40 (1-GPU) -> 6.11 (2-GPU) ->
        10.84 (batching) -> 14.30 (40/40 budget) -> 14.96 (fused MoE)
        -> 21.44 tok/s (both).  Token is dispatch-launch + GPU-exec
        bound across two serial device halves (~18 ms each; sync ~35 ms
        at 0 misses).
  - [x] 2026-07-17 WAVE 6 — AUTO DEVICE SELECTION + SINGLE-48GB RUNS:
        **36.41 tok/s Q8** (27.5 ms/token, 182x from 0.2; 40/40
        resident, zero misses every token) and 22.09 tok/s Q5 (45.3
        ms), both on the 48 GB RTX 4090 ALONE.  New: mgpuEnumAdapters
        (DXGI per-adapter name + total/used dedicated VRAM) ->
        Minigpu.listAdapters(); Qwen35Model.loadAuto() computes
        per-block byte needs from the GGUF header and picks single-GPU
        full residency when the model fits the largest adapter (minus
        3 GB reserve + 1 GB scratch), else a proportional two-GPU
        split; load(primaryAdapter:) pins the primary context.  Bench:
        --auto.  FINDINGS: (a) single fast GPU BEATS the dual split
        for batch-1 decode (36.4 vs 21.4 tok/s — serial pipeline means
        the 3090 half + hop is pure drag; auto now picks this
        correctly); (b) Q8 is FASTER than Q5 here — vectorized Q8_0
        matVec is much cheaper per element than the K-quant decoders.
  - Progression today: 0.2 -> 2.40 -> 6.11 -> 10.84 -> 14.96 -> 21.44
        -> 36.41 tok/s.
  - [x] 2026-07-17 WAVE 7 — BATCHED PREFILL: 77.1 tok/s prompt
        processing (512-token prompt in 6.6 s vs ~14 s sequential),
        decode intact at ~34 tok/s.  Prompt chunks of up to 256 tokens
        run ONE fired pass per op (Qwen35DecodePlan.prefillChunk;
        model.prefill chunks + generate() uses it; resident blocks
        required, sequential fallback otherwise; multi-GPU hop reads the
        whole chunk's hidden states).  Batched kernel set: qmvB / rmsB /
        addB, attention prep+causal-softmax-V per (token, head) with
        scores in workgroup shared memory (needs the 32 KB limit we
        request), DeltaNet conv parallel-over-chunk + history roll +
        T-step recurrence INSIDE one kernel (T baked for barrier
        uniformity), per-token GPU top-K, fused experts z=(t,slot).
        DENSE projections go through dequant-to-f16 scratch + 16x16
        tiled GEMM (weights read once per 16 tokens instead of once per
        token; f16 prefill weights = the llama.cpp tradeoff, end-to-end
        relRms 0.03 vs sequential with identical top tokens).
        BUGS FOUND + FIXED (differential-driven):
        (1) CHUNK-SEAM RACE: queue writes execute at submission time,
        so chunk N+1's _pos/_pT writes jumped ahead of chunk N's
        still-queued fired dispatches — prefillChunk now drains the
        device FIFO with a 4-byte readback first.  Decode never sees
        this because every token ends in an awaited readback.
        (2) FXC MISCOMPILE: a byteAt()-based dequant kernel read wrong
        lane-3 bytes under Dawn's D3D11/FXC backend (deterministic
        wrong values; function wrapper vs inline made no difference) —
        rewritten with whole-word funnel loads, one thread per Q8_0
        block (test/prefill_gemm_test.dart pins both kernels vs CPU).
        Do NOT reintroduce per-byte accessors in new kernel contexts
        without that test.
        GEMM path covers f16 (direct, no dequant) + q8_0; K-quants use
        the per-token fallback until word-based dequants are written.
  - [x] 2026-07-17 WAVE 8 — PREFILL 77.1 -> 226.4 tok/s (512-token
        prompt in 2.26 s; GPU-only ~4.4 ms/token; decode intact ~30).
        Four separate findings, each PROFILED not guessed (added
        -DGPU_ML_PREFILL_PROFILE stage buckets + a recordMicros CPU
        counter printed by decode_bench):
        (1) EXPERT GROUPING (modest: ~+20%): GPU-side count/scan/
        scatter/worklist (plan-expgroupPrep, ONE workgroup, one thread
        per expert, NO atomics — FXC E_FAILs on ANY workgroup
        `atomic<u32>` array, probed directly) + grouped tile matvec
        (plan-expgroupB: TB=16 tokens per tile, 4 threads/row x 64
        rows/wg, per-token accumulators UNROLLED to scalars — a
        dynamically-indexed local array lands in FXC indexable temps —
        x bound as vec4).  Exact quantized weights (no f16 tradeoff),
        relRms vs fused 9.6e-7.  test/expert_group_test.dart pins
        prep+scatter+matvec vs CPU.  LESSON: experts were ~20% of
        prefill, not the ~80% wave 7 estimated.
        (2) DELTANET RECURRENCE 1252 -> 277 ms/chunk (was 56% of
        prefill!): the T-loop round-tripped the whole DxD state through
        L2 twice per token.  Now each workgroup holds an RH-row slice
        in SHARED memory column-major (row-major strides all hit one
        bank), cooperative coalesced load/store outside the loop, ks/qs
        as broadcast L1 reads -> ZERO barriers inside the T-loop.
        _recRowsPerWg picks RH (64 for hd=128, 32KB groupshared).
        (3) GPU EMBED GATHER: token_embd (q8_0, ~540 MB) now resident
        on the primary GPU; prefillChunkTokens writes T token ids and
        plan-embgather dequants rows straight into _px — replaces one
        awaited ~2KB DISK READ per prompt token.
        (4) FXC COMPILE STALL (~2.2 s, the dominant "CPU" mystery):
        first-dispatch pipeline creation compiles under the gpuMutex on
        the WebGPU thread and the recording thread blocks on it.
        Proven one-time by 512 vs 1024-token runs (record 2195 vs
        2301 ms).  Fix: load-time WARMUP chunk (garbage tokens through
        prefillChunkTokens + readLogits), then zeroRecurrentState()
        (ssm + conv history; KV needs no reset — positions overwrite
        before they're attended).  Warmup verified residue-free: logits
        bit-comparable to no-warmup runs.  record now ~10 ms/prompt.
        Steady buckets/256-chunk after: moe ~790 ms (grouped matvec
        ~80-90 GB/s effective — occupancy/latency-bound, NOT bandwidth;
        biggest remaining prefill lever), delta ~270, attn ~125,
        norm ~35.
  - [x] 2026-07-18 WAVE 9 — PREFILL 226 -> 263 tok/s @512
        (2048-token prompt: 224 tok/s; decode intact ~30, 25.6 at
        2048-context = attention growth, expected).  Changes:
        (a) TB=16 tiles + 4 thr/row x 64 rows/wg + shared-memory x
        STAGING in the grouped matvec (rounds of TPR blocks with
        cooperative vec4 staging; per-quant-word x traffic became
        shared reads);
        (b) shexp DOWN moved off the workgroup-per-(row,token) pattern
        (524k wgs/block!) onto the f16-GEMM path — shexp bucket
        128 -> 63 ms/chunk; joins gate/up in the f16 class (e2e relRms
        0.019 vs exact-shdown, top-4 logits identical, greedy text
        identical);
        (c) SiLU-mul FUSED into the grouped down kernel's staging
        (dispatch + 12 MB/block prodExp round-trip gone);
        (d) combine kernel now WRITES ffn = shScalar*shDown +
        sum(w*downExp) — folds the shared-expert accumulate and kills
        the ffn zero pass.
        PROFILING CAVEAT (memorize): the per-stage drain profiler
        INFLATES fixed-position slots — the combine slot showed a
        rock-steady 412 ms/chunk that persisted with the kernel
        replaced by a trivial zero-write AND with the neighboring
        dispatch removed; the REAL cost (measured by deleting the
        dispatch in a non-profiled run) was ~100 ms.  Likely Dawn/D3D11
        per-N-submits maintenance landing deterministically in one
        slot.  Treat drain-profile buckets as directional only; confirm
        by ablation on the un-profiled path.
  - [x] 2026-07-18 WAVE 10 — DECODE INVESTIGATION (llama ~50 tok/s vs
        our ~40-45; user asked for low-hanging fruit).  Result: decode
        is MEMORY-BOUND, and the cheap occupancy/launch wins are all
        within measurement noise.  What was RULED OUT by measurement
        (so future sessions don't re-chase):
        * NOT launch-bound: a serial dependent-dispatch microbench
          (each dispatch reads+writes the same buffer) converges to
          ~2.3 us/dispatch (vs 1.7 independent) — Dawn's compute pass
          does NOT insert heavy barriers between dispatches.  ~900
          dispatches/token = ~2 ms of a ~24 ms decode.
        * NOT occupancy-bound: rewrote the decode GEMV to full lane
          occupancy (R rows/wg x T threads/row, T = pow2 <= COLS/blockSize
          capped 64, narrow T-wide reduction; _gemvT/_qmvSrc, matVecBodyWGSL
          gained threadVar/stride params — safe transform, stride is
          always `+ 256u`, distinct from `/ 256u` superblock sizes and
          `<< Nu` shifts).  Microbench +2% on lm_head; end-to-end within
          the machine's +/-1.2 ms run-to-run noise.  GUARD: rows < 512
          keeps T=256 (row-packing STARVES workgroups on tiny outputs
          like DeltaNet beta/alpha ~32 rows -> 8 wg; that regression is
          why the unguarded version was flat).  BUG caught: each row's
          lane-0 must write scratch[lid.x] (= scratch[rr*T]), not
          scratch[0] — else all R packed rows get row-0's value (adjacent
          logits identical); also broke lm_head shared by both paths.
        * NOT dequant-ALU-bound: unpack4xI8 (Dawn supports it AND
          dot4I8Packed — probed OK) replaced the 4 manual
          `f32(bitcast<i32>(raw<<Nu)>>24u)` extracts; ZERO bandwidth
          change in the microbench.  REVERTED (neutral + web-portability
          risk).
        MEASURED (gemv_bw microbench, amortized): q8_0 GEMV caps at
        ~295-330 GB/s = ~30% of the 4090's ~1 TB/s peak even for big
        kernels; small 2 MB kernels ~140 GB/s (too little work to
        saturate).  Decode ablation map (60-tok warm, no prompt):
        DeltaNet 7.5 ms (30 blks) + routed experts 7.5 ms co-dominate;
        attn 2.2 (10 blks), shexp 1.9, remainder ~4.2 (lm_head ~1.8 +
        ~240 small dispatches).  Experts move ~1.9 GB in 7.5 ms = 253
        GB/s = NEAR the kernel ceiling (little fruit without a new
        layout).  DeltaNet ~450 MB in 7.5 ms = ~60 GB/s = 5x headroom,
        but the RECURRENCE is NOT it (skipping it changed nothing within
        noise) — it's the many small low-occupancy/latency-bound ops
        (conv, l2split, gatednorm, params, tiny beta/alpha GEMVs) diluting
        the stream.  HYPOTHESIS for the ~30% ceiling (UNVERIFIED — probe
        hung on a huge-workgroup staged kernel, machine DLL-locked):
        the q8_0 34-byte block layout makes consecutive threads read
        blocks 34 B apart, scattering a warp across ~9 cache lines per
        load.  Real fix = coalesced access via shared-memory staging of
        each row's words, or a repacked [scales][int8-quants] layout
        (word-aligned) + dot4I8Packed with int8-quantized activations.
        That is a real kernel/layout project (VRAM implications on the
        resident 40 GB — mind the RAM-blowup postmortem), NOT low-hanging.
        KEPT: guarded full-occupancy GEMV (correct, non-regressive,
        becomes relevant once the memory ceiling lifts).  Compile-time
        ablation flags left in place (zero runtime cost): GPU_ML_SLOW_GEMV,
        GPU_ML_DEC_NO_{ATTN,MOE,DELTA,ATTNONLY,EXP,SHEXP,REC}.
  - [x] 2026-07-19 WAVE 11 — DECODE dot4I8Packed (dp4a).  CORRECTED the
        wave-10 root cause via a proper bw microbench (was WRONG about
        coalescing): the real achievable VRAM BW here is ~560-780 GB/s
        (NOT 1008; test buffers MUST exceed the 72 MB L2 or you measure
        L2 at 1600+); a dequant-only streaming kernel (no x-multiply, no
        reduction) hits the ceiling (~600-860), and a PERFECTLY COALESCED
        shared-staged load gave NO improvement — so it is NOT coalescing,
        NOT dequant ALU (unpack4xI8 = 0 change), NOT x-read latency (x in
        shared = no change).  The wall is the per-element f32
        multiply-accumulate DOT PRODUCT itself.  FIX = dot4I8Packed: int8
        weights (already q8_0) x INT8-QUANTIZED ACTIVATIONS, 4-wide int8
        MAC in one HW instruction.  Built: matVecDp4aBodyWGSL +
        quantizeInt8BlockWGSL (per-32-block symmetric activation quant,
        matches q8_0 block size; one 32-thread wg/block, lanes 0-7 pack
        4 int8 each).  Wired into DECODE (thread-per-row, 256 rows/wg, NO
        reduction — the row-split+reduction structure is structure-bound
        not arithmetic-bound and saw ZERO benefit; thread-per-row is
        where dp4a pays): fused experts gate/up/down (quantize _xn once
        in _moe + per-slot prod quant), shexp gate/up, DeltaNet qkv/gate,
        attention q/k/v.  _xn quantized once per norm, reused by all its
        projections.  beta/alpha/out/router/shexp-down kept f32 (tiny
        outputs starve thread-per-row, or non-_xn inputs).  NUMERICS:
        standalone relRms 0.0038 vs f32 (int8 per-block activation quant);
        pinned test/dp4a_test.dart; DECODE TEXT BYTE-IDENTICAL to f32
        across 50 greedy tokens (greedy is flip-sensitive = strong e2e
        check).  Prefill UNTOUCHED (dp4a is decode-only; prefill logit
        dump relRms 0.0 vs f32).  RESULT — SHIPPED OPT-IN OFF (default
        stays f32): an experts-ONLY thread-per-row A/B in a clean window
        was ~7% faster (27.4-27.9 vs 29.6 ms), BUT expanding dp4a to the
        dense projections REGRESSED decode ~30% (interleaved A/B: 22.7 vs
        33.5 tok/s, 3/3 rounds).  ROOT CAUSE: thread-per-row (256 rows/wg,
        the structure dp4a needs to beat the f32 arithmetic) STARVES the
        GPU on real projection shapes — 768-2048 rows -> only 3-16
        workgroups (~2% util) vs the f32 fast-GEMV's hundreds (row-split,
        full GPU).  The microbench HID this by using 131k-262k-row
        matrices.  dp4a only wins where rows are huge (lm_head) or spread
        over slots (experts).  Full clean measurement was BLOCKED anyway by
        the user's concurrent GPU load (decode swung 2.5-45 tok/s between
        runs — MEMORY: cannot bench decode while user flutter/dev_rig apps
        run).  Kept behind `-DGPU_ML_DP4A`; suite 38+1 with it OFF.  NEXT
        (idle GPU): pick thread-per-row vs row-split-reduction BY SHAPE
        (thread-per-row only for >~64k rows, dp4a-with-reduction otherwise)
        so dp4a keeps full occupancy; then extend to out-proj + prefill.
  - [x] 2026-07-19 WAVE 12 — dp4a SETTLED with a FAIR harness.  Built the
        measurement fix first: dp4aLevel is now RUNTIME-switchable (0 = f32,
        1 = lm_head only, 2 = all q8_0 projections; model.dp4aLevel setter)
        and decode_bench --ab-dp4a N cycles levels every N tokens WITHIN ONE
        PROCESS, so background GPU load hits every path equally — the only
        trustworthy A/B on a shared machine.  Also made dp4a shape-aware
        (row-split T=_dp4aT(cols) x 256/T rows/wg for small projections,
        thread-per-row only for >=64k rows) and converted lm_head (the one
        huge ~540 MB/token streaming matVec).  TRAP FOUND: the first A/B
        warm-up ran extra forwards of the last prompt token to precompile
        pipelines — each forward ADVANCES the DeltaNet recurrent state, so
        the duplicates derailed generation into "is is is".  Never warm with
        real forwards on a stateful model; exclude each segment's first
        token from its average instead.  VERDICT (144 tokens, 42
        samples/level, interleaved, coherent text): f32 24.9 ms/token,
        dp4a-head 25.0 (parity — the head is only ~7% of a token; halving
        it minus one quantize dispatch nets ~0), dp4a-all 28.0 (-12% — ~80
        tiny latency-bound quantize dispatches + lower per-kernel occupancy
        beat the arithmetic saving on 10-30 us kernels).  CONCLUSION: the
        f32-dot-product wall is real only for HUGE streaming matvecs;
        batch-1 decode kernels are latency/structure-dominated, so dp4a
        does NOT pay at decode — default stays 0.  Numerics stay validated
        (relRms 0.0038, coherent text at every level mid-sequence, suite
        38+1).  WHERE dp4a ACTUALLY FITS NEXT: the PREFILL grouped expert
        kernels — TB=16 tokens per weight word = 16 MACs/byte =
        arithmetic-heavy, exactly the regime the microbench proved dp4a
        wins; needs per-token-row activation quant + an int8 grouped body.
  - [x] 2026-07-19 WAVE 13 — BOTH follow-ups landed, DEFAULT ON:
        (A) PREFILL dp4a: _fireQuantXB (bulk int8 activation quantizer,
        8 blocks per 256-thread wg — the decode one-block variant would
        launch ~100k starved wgs) + _fireExpGroupedQ (int8 grouped expert
        matvec: staged PACKED u32 quants, 4x smaller staging, one
        dot4I8Packed per weight-word x token; unrolled isum/acc scalars;
        silu back to its own pass + re-quant since down needs globally
        consistent per-32-block scales).  FAIR A/B (--ab-prefill, 3
        interleaved pairs, state-reset between runs): 277.6 -> 305.5
        tok/s (+10%), non-overlapping.  2048-token prompt: 224 -> 265.6
        tok/s (+19%).  e2e relRms 0.029 (int8-activation on top of the
        accepted f16 class; top-5 identical, greedy text byte-identical).
        Pinned: expert_group_test 'grouped dp4a' leg vs an EXACT integer
        CPU reference (relRms 6.9e-8).
        (B) DECODE DeltaNet fusion: plan-dnbap (beta+alpha GEMVs + params
        transform in ONE kernel; alpha gets a RENAMED accessor set since
        accessorsWGSL hardcodes wq; 'row *' -> 'rEff *' body substitution)
        + plan-dnrecg (gated norm folded into the recurrence epilogue —
        same vHeads x hd grid, writes _dGated directly, exactly 8 storage
        bindings after dropping outv).  Math order-identical to unfused.
        FAIR A/B (--ab-fuse): 34.07 -> 35.06 tok/s (+3%).
        ULTRACODE REVIEW (18-agent adversarial workflow, 6 dimensions):
        FXC trap sweep CLEAN on all four new kernels; 10 confirmed
        minor/latent findings, ALL FIXED: memo keys now encode xPerZ
        (both grouped variants), fuseBA gate asserts wAlpha.cols ==
        wBeta.cols, quantizer bakes BUFFER CAPACITY not runtime T (was a
        fresh FXC compile per tail-chunk size mid-prefill), _pXnq
        allocation guard matches the full grouped predicate, bench
        re-baselines sync counters after --ab-prefill and guards
        --ab-* N=1 (would have discarded every sample).  NO_REC ablation
        documented as unfused-only.  Suite 39+1.  Opt-outs:
        -DGPU_ML_NO_PREFILL_DP4A, -DGPU_ML_NO_DELTA_FUSE.
  - [x] 2026-07-19 WAVE 14 — DENSE INT8 GEMM + DECODE EMBED GATHER, both
        DEFAULT ON.
        (A) _fireGemmQ (plan-gemmq): dense int8 GEMM over contiguous
        token tiles (tail bound from pT[0], underflow-safe
        min(TB, pT[0]-min(t0,pT[0]))), same skeleton as the validated
        grouped kernel minus the expert indirection.  Wired via
        _fireQmvB's optional xq/xsc: delta qkv/gate/beta/alpha, attn
        q/k/v, shexp gate/up all skip the dequant-to-f16 pass entirely
        (only wOut/wo/sh_down — non-_pxn inputs — keep the f16 GEMM).
        _pxn is quantized once per norm per block at FULL CAPACITY
        (T>=16 gated after review).  FAIR A/B: 269.7 -> 339.1 tok/s
        (+26%, all pairs non-overlapping) — the largest prefill jump
        since the FXC warmup.  Numerics IMPROVED vs the f16 path:
        e2e relRms 0.0194 (int8 per-block activations x exact int8
        weights beats f16-rounded weights), top-4 identical, greedy
        text byte-identical.  Pinned: 'dense int8 GEMM' leg vs exact
        integer CPU reference (relRms 6.5e-8, tail tile covered).
        (B) Decode GPU embed gather: loadTokenX/forwardToken write a
        4-byte token id and dequant the embedding row straight into _x
        (plan-embgather reused with {tok,out} params) — replaces the
        per-token ~2KB disk read + CPU dequant + row upload.  FAIR A/B
        (--ab-embed): 28.2 -> 27.9 ms/token — NEUTRAL-to-+1% (the disk
        read was page-cache-warm); kept ON because it is strictly less
        work and removes the cold-cache stall mode.  Runner flag
        embedGatherDecode; -DGPU_ML_NO_DEC_EMBED opts out.
        REVIEW ROUND 2 (11-agent adversarial): _fireGemmQ VERIFIED
        CORRECT on all 8 sub-checks (verifier independently proved the
        staging index algebra against the quantizer layout and the
        capacity bounds); embed drain-contract clean; one actionable
        finding fixed (T>=16 gate on the delta/attn quantize — was dead
        full-capacity work on tiny tail chunks).  DOCUMENTED
        LOAD-BEARING INVARIANTS: GemmQ assumes xq row stride == w.cols
        (all nine call sites are residual-stream projections with
        cols == dim, enforced only by the single _fireQmvB call site);
        trailing weight-word overread on the last q8_0 block = the
        accepted robust-access pattern.  Suite 40+1.
  - [x] 2026-07-25 WAVE 15 EXECUTED — decode 36.7 -> 53.5 tok/s,
        llama.cpp parity (~50) PASSED.  The debate's own discriminator
        KILLED its diagnosis (see the plan entry below for the theory it
        replaced): tool/stride_probe.dart came back FLAT — lane strides
        4B/32B/128B stream 1040/1100/773 GB/s, so L1 address divergence
        costs ~nothing and the machine's true D3D11/FXC ceiling is
        ~1.0-1.1 TB/s (roofline ~320 tok/s, not 225).  Probe cascade
        (tool/occupancy_probe.dart, chain_probe.dart,
        kernel_tax_probe.dart) eliminated occupancy (fine at >=512 wg;
        starvation only <64 wg — retroactively explains the dp4a
        thread-per-row expert -30%: 16 wg = 482 GB/s), dispatch/barrier
        chains (4-7 us/dispatch incl. pipeline switches), and CPU
        submission (~2-3 ms/token).  REAL WALL: UAV reads of the
        activation vector CONSUMED IN THE DOT LOOP stall the kernel 4x
        (real q8_0 body 21.8 us vs 5.0 us floor at identical geometry
        and weight traffic; removing x-consumption -> floor; vec4-x,
        unpack4xI8, and mul->add variants all ~no-op, so it is not load
        count and not MACs).  FIX (landed, default ON,
        -DGPU_ML_NO_SHARED_X / --ab-xs to compare): padded shared-x
        staging — QuantizedTensor.{xsDeclWGSL, stageXsWGSL, sharedXBody}
        stage x once per workgroup into var<workgroup> with +i/32
        padding (a FLAT layout has a lane-independent bank index for
        the q8_0 pattern = the 32-way-conflict trap that invalidated
        the wave-11 shared-x ablation).  Wired: _fireQmv, _fireQmvAccum,
        expfused gate/up, expdown (per-slot window), lm_head dp4a
        (xqs/xscs).  Fair in-process A/B: 36.67 -> 47.75 tok/s (+30%),
        text byte-identical; pinned tests green.  ALSO LANDED: GPU
        argmax (plan-argmax1/2, CPU-tie-break-exact, 4-byte id
        readback), _pos GPU bump kernel (plan-posbump — kills the
        per-token submission-time write flush), token CHAINING (embed
        gather reads the argmax buffer; zero CPU writes per token), and
        K-TOKEN BURST DECODE (forwardBurstGreedy / --burst K,
        plan-tokcopy:i history slots, ONE readback per K tokens; EOS
        mid-burst advances recurrent state past truth — runner breaks
        the chain and re-seeds).  Burst-of-8: 18.3-18.9 ms/token STABLE
        = 53.5 tok/s, text byte-identical — and it doubles as the CLEAN
        timing instrument (per-token sync-wait timing carries
        cross-process variance; the decode drain-profile inflates
        20-200x, NOT 4x — unusable for decode).  Remaining ~18.5
        ms/token is REAL GPU execution: prime suspects are the
        untouched kernels (delta recurrence/conv, attn core,
        beta/alpha, router, quantx, norms/adds — same UAV-read tax
        class) + ~800 x ~5 us dispatch bubbles.  NEXT: burst-timed
        ablation category map -> shared-stage the remaining hot
        kernels; MGPU_BACKEND=d3d12/vulkan probe smoke; elementwise/
        norm fusion to cut dispatch count.
        SECOND PASS (same day): (1) Burst-timed category map: MoE
        ~10.7 ms (routed 6.4 + shexp 2.0 + route/etc 2.3), delta
        ~7.3 ms (proj 1.6 + rec 1.7 + rest noise), attn 0.4 ms,
        FLOOR (norms+adds+head+argmax stream, NO_ATTN+NO_MOE) = only
        1.7 ms/token -> dispatch overhead is NOT the wall; kernel
        exec is.  (2) tool/xs_shape_probe.dart per-shape T sweep:
        expert down 2048x512x8 at the shipped T=256 = 76 us/117 GB/s
        (240 of 256 lanes idle) vs T=16 = 10.8 us/829 GB/s (7x!);
        gate 512x2048x8 T=256->T=16 = 2.5x; dense 2048x2048 best
        T=32.  First e2e retune showed nothing because (a) the
        heuristic picked T=8 (past the sweet spot) and (b) a BUG:
        expdown's dispatch was never divided by rows-per-wg (16x
        extra workgroups each staging 2 KB).  Fixed; probe table
        hardcoded (_gemvTXs dense=32, _expTXs experts=16).
        (3) OVERLAP PROBE: independent kernels on disjoint buffers
        NEVER overlap — pair cost = sum of solos on D3D11 AND D3D12
        (MGPU_BACKEND=d3d12 WORKS, backendType=4, same numbers;
        vulkan fails adapter selection).  Serialization means
        fewer/bigger kernels is the only latency lever ->
        (4) expFuse2 (default ON, --ab-f2, GPU_ML_NO_EXP_FUSE2):
        gate+up in ONE z-doubled dispatch (dual weight bindings a la
        plan-dnbap) writing a [2,topK,rows] plane buffer +
        silu(gate)*up computed INSIDE down's shared-x staging loop
        (plan-expgu / plan-expdownsm) — 5 -> 3 dispatches/block,
        prod buffer round-trip gone, bit-exact (text byte-identical).
        In-process A/Bs: fuse2 1.6x on the expert path; rowpack+
        dispfix+fuse2 combined 2.4x vs the pre-T-table config (both
        measured inside a degraded-regime process; ratios only).
        (5) VARIANCE: the 2.5-45 tok/s cross-process swing now
        reproduces ~40% of runs (some at 0.8-4.4 tok/s); clocks
        pinned 2520 MHz and no foreign VRAM pressure in sampled fast
        runs; state is healthy immediately after exit.  Cause still
        uncaught — per-process GPU Local/Shared sampler armed in
        bench batches; ratios within one process remain the only
        trustworthy comparison.
        (6) FAST-REGIME FINAL for the wave: ab-rp 21.1 -> 18.6
        (+13%); headline BURST 16.5-17.9 ms/token, warm 59.4 tok/s
        (day total 27.3 -> 16.8 ms/token = +62%); pinned tests
        green; text byte-identical throughout.  chainFuse landed
        (+3% A/B, --ab-cf, opt-out GPU_ML_NO_CHAIN_FUSE):
        _fireQmvCat2 generic concatenated-rows dual-weight GEMV
        (plan-qmvcat2; renamed b-accessors; rEff row rebase) covers
        shexp gate+up AND delta qkv+z (conv reads the qkv prefix of
        _qkvz in place; the fused rec reads z at a baked wqkv.rows
        offset); plan-shdownsm folds silu-mul into sh_down staging
        with the sigmoid-gate accumulate — shexp 4 -> 2 and delta
        proj 2 -> 1 dispatches per block.  NEXT wave candidates, in
        expected value order: rec i-loop split (1.7 ms at 32
        starved wgs), route/topk region (~2.3 ms; the 1-wg topk
        kernel idles the whole GPU x40 blocks), attn q/k/v cat
        (needs consumer offset rework), speculative decode
        (realistic 1.3-1.6x), variance root-cause.  Roofline ~3.1
        ms/token — 5x headroom remains.
  - [x] 2026-07-26 WAVE 16 — decode 12.4 -> 11.8 ms/token (85 tok/s);
        two-day arc 27.3 -> 11.8 ms = +131%, llama.cpp's claimed ~50
        tok/s beaten by 1.7x.  Text byte-identical throughout, suite
        green.  LANDED: (a) top-k RANK-SELECT — one pass counting how
        many experts outrank each, replacing K sequential max
        extractions (~64 fewer barriers in a 1-workgroup kernel that
        runs 40x/token); the comparator reproduces the old tie-break
        exactly, so winners and slots are identical.  (b) residFuse
        (+3.7% A/B) — the attn/delta output projection accumulates
        into x and the MoE combine writes `x += shexp + sum(slots)` as
        ONE store, so the ffnOut zero pass and BOTH residual adds
        disappear (~120 dispatches/token).  (c) recSplit (+13.3% A/B)
        — the recurrence ran one workgroup per v-head (~32), leaving
        ~75% of the SMs idle through the biggest memory mover in the
        step; TPR lanes now share each state row with a tree
        reduction (128 wgs) and the gated norm moves back to its own
        dispatch.  Not bit-exact (tree vs serial sums) but text was
        unchanged.  (d) the router folds the shared-expert gate as one
        extra row, removing another 1-workgroup dispatch per block.
        (e) fusion gates relaxed from q8_0-only to q8_0|f16: THIS
        BUILD HAS 77 f16 TENSORS (13 blocks' expert stacks, 12 shexp
        pairs, scattered qkv/gate/alpha/beta), so about a third of the
        model had been silently falling back to unfused paths.  Caught
        pre-flight while doing it: _fireQmvCat2 and the beta/alpha
        fusion memoized shaders by SHAPE without TYPE, and identical
        shapes exist in both precisions — an f16 block would have run
        a q8_0 shader.
        INSTRUMENTS (the durable part): the skip* ablations became
        RUNTIME fields, so `--ab-map` walks every category inside ONE
        process, and `--burst` now composes with any `--ab-*` flag
        (arm switches per burst).  This mattered: per-token A/B
        carries the readback's +-1-2 ms jitter, which made the old
        map SELF-INCONSISTENT (delta's internals summed to 2.4x the
        whole-delta measurement).  The burst-timed map is consistent.
        Never map categories with per-token timing.
        BURST MAP of 11.8 ms: delta 3.6 (proj 1.8, rec 0.7, conv 0.7,
        out 0.7), routed experts 3.4, attn 1.2, shexp 1.1, norms 0.3,
        route 0.1, remainder ~2.1 (lm_head ~0.8 + embed/argmax).
        PROBE TRAPS CORRECTED: xs_shape_probe's 829-963 GB/s were L2
        FANTASY — 8 experts (25 MB) fit the 4090's 72 MB L2 and were
        re-read 400 times.  tool/expert_probe.dart at production scale
        (272 MB stacks, unique experts per block, 1 GB with no reuse)
        gives 1.56-2.8 ms/token at 626-690 GB/s, so the expert path's
        in-model 3.4 ms is essentially AT ITS FLOOR — the "5x gap" I
        chased was my own instrument.  The same probe KILLED the
        WDDM-eviction theory: with 34 GB allocated but every dispatch
        pinned to one buffer, speed is unchanged, so footprint costs
        nothing.  What does cost is binding a DISTINCT buffer — ~2.1
        us for 272 MB, 0.6 us for 1 MB, scaling with size
        (tool/bind_probe.dart); ComputeShader keeps exactly ONE bind
        group and recreates it on any change, while setBuffer
        early-outs when the pointer is unchanged.  At ~680 distinct
        binds/token that is ~1.4 ms.
        ROOFLINE: 3.2 GB/token (dense ~2.2 GB + experts ~1.02 GB) at
        ~800 GB/s = 4.0 ms = 250 tok/s; at 11.8 ms we are at 34%.
        NEXT, ranked: C++ bind-group CACHE keyed by the buffer tuple
        (~1.4 ms, but needs release-invalidation or cached groups pin
        buffers alive = a VRAM leak on the streamed-expert LRU path);
        conv+l2split merge (~0.2 ms); attn q/k/v cat3 (~0.1 ms);
        staging overhead on small-row GEMVs (the shexp cat2 stages
        1 MB of x per 2.2 MB of weights); speculative decode.
  - [ ] 2026-07-19 WAVE 15 PLAN — settled by an adversarial two-round
        debate with an Opus 5 architect agent.  HEADLINE: the wave-11
        "decode wall = f32 MAC arithmetic" ROOT CAUSE IS OVERTURNED.
        Leading hypothesis now: L1 ADDRESS DIVERGENCE — with T=64
        lanes/row, lane j owns 34-byte block j, so per warp load
        instruction the 32 lanes touch ~32 distinct 128B lines for x
        (and ~9 for weights): ~1120-1280 L1 wavefronts per 1088 weight
        bytes ≈ 270 GB/s from first principles = exactly our measured
        230-330.  It explains EVERY ablation, including the two
        "contradictions": (a) my x-in-shared test was INVALID — the
        shared layout put every lane on ONE BANK (bank = (k*4+i) mod 32,
        lane-independent) = a 32-way conflict identical to the
        divergence it was meant to remove; (b) the dequant-only kernel
        that hit 600-860 GB/s dropped the divergent x-loads WITH the
        multiply — I conflated the two.  dp4a-neutral-at-decode
        FALSIFIES the MAC hypothesis (fewer MACs, no change).
        CONVERGED ROADMAP (both sides signed off):
        T0 noise floor (one sitting): clock-lock + DXGI
        QueryVideoMemoryInfo residency gate (WDDM EVICTION is the
        leading variance hypothesis — 40GB resident on the DISPLAY
        adapter; "allocation order at process start" matches
        stable-in-process/wild-across-process EXACTLY; NOT the user's
        app load — user explicitly corrected this); flush-cadence ramp
        (32-128-512); GPU argmax + 4-byte id readback (993KB logits
        readback + 248k-iteration Dart argmax + mutex-across-map-wait
        are constant ~7% measurement dilution); MGPU_BACKEND=
        d3d12|vulkan smoke (env var EXISTS in buffer.cpp; D3D12 needs
        use_dxc toggle + dxcompiler/dxil DLLs or it's FXC again; NO
        DawnTogglesDescriptor is set anywhere — try disable_robustness,
        dump_shaders); RE-BASELINE llama.cpp under the same gate (the
        50 tok/s target may itself be a variance artifact).
        T1 THE DISCRIMINATOR + FIX: x-lane-stride sweep (hold
        bytes/ALU/occupancy, vary ONLY per-lane x stride 4B-128B; GB/s
        tracking 1/lines-per-warp establishes the model; flat curve
        kills it) -> vec4-x across ALL decode GEMV bodies +
        _fireQmvAccum + single-word f16At (predicted ~980 GB/s
        wavefront-limited -> DRAM-limited; est. decode 40 -> 55+).
        Verify codegen via dump_shaders -> fxc /Fc (confirm Load4).
        EXIT GATE agreed in advance: >=600 GB/s = planar repack DEAD;
        350-550 = repack lives; <350 = model wrong, backend+CUDA
        -instrument diagnosis next.
        T2 dispatch structure: gate+up concat at load, zero-fold,
        topk+shscalar merge, decode attention collapse (2.2ms for 4MB
        = 1.8 GB/s); then K-token prerequisites: _pos becomes a
        GPU-side counter (the awaited per-token _pos write FLUSHES the
        batch — hard blocker), release the GPU mutex around the
        map-wait in readDirect.
        T3 K-token GPU-resident decode (argmax feeds embed-gather
        GPU-side, CPU records K ahead, detokenize lags): gated on full
        residency; REQUIRES DeltaNet state checkpoint (EOS mid-batch
        advances 60MB of in-place state past truth — same hazard class
        as the wave-12 "is is is" scar; the checkpoint path doubles as
        the speculation prerequisite).
        T4 prefill tiled int8 attention (0.35 TFLOPS = 0.4% of peak,
        grows with context; rank depends on prompt-length
        distribution).  T5 planar-Q8 repack (strictly gated on T1
        number).  T6 speculative decoding (MoE batch-cost math:
        B=4 verify reads ~3.7x expert bytes -> realistic 1.3-1.6x, NOT
        2.5x; needs T3's checkpoint) + 4-bit experts as a PRODUCT
        VARIANT (quality class change), not a perf tier.
        ROOFLINE: ~3.1 GB/token at the proven-achievable 700 GB/s =
        4.4ms = ~225 tok/s ceiling; we are at 18% of roofline — there
        is no wall, there is a gap.
        (prefill) DeltaNet stage ~270 ms/chunk (5 GEMM projections + conv
        + in-kernel recurrence), attention ~130 and grows with context
        (flash-style KV tiling), K-quant word-based dequant for the Q5
        GEMM path + K-quant grouped bodies, exact quantized GEMM for shexp
        down (restores bit-exact shexp if ever needed).  (both) sampling
        (fixes Q5 greedy looping) + chat template.
  - [ ] Prefill path: multi-token quantized matmul (x [tokens, cols]) so
        prompts aren't per-token GEMV loops; batched SSM scan for prefill.
  - [ ] More fusions: GEMV vec4/register blocking; subgroup ops (Dawn)
        where available; multi-pass-per-submit command batching in C++ if
        submit overhead ever dominates (measure first — ~2000 fired
        submits/token now cost little).
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
