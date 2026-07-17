# qwen35moe — exact inference semantics (from llama.cpp source)

Extracted 2026-07-17 from ggml-org/llama.cpp master:
`src/models/qwen35moe.cpp`, `src/models/delta-net-base.cpp`,
`ggml/src/ggml-cpu/ops.cpp` (rope/l2_norm/softplus), `src/llama-model.cpp`
(rope type). This is the reference for gpu_ml's kernels; per-layer parity
tests validate against llama.cpp behavior, not guesses.

Model dims (our files): n_embd=2048, n_layer=40, full attention every 4th
layer (blk.3,7,...,39; others are Gated DeltaNet), n_head=16, n_head_kv=2,
head_dim=256, rope dims=64, rope base=1e7, rms_eps=1e-6, experts=256 top-8,
expert_ff=512, shared_ff=512, vocab=248320. NOTE: some qwen3.6 GGUFs carry
extra MTP/NextN draft layers (`*.nextn.*` tensors, n_layer_nextn>0) — ours
do not; ignore MTP.

## Block structure (both layer kinds)

```
x   -> attn_norm (RMS) -> [attention OR delta-net] -> + x        (residual)
    -> post_attention_norm (RMS) -> MoE FFN         -> + (pre-norm residual)
```
Final: output_norm (RMS) -> output.weight (lm_head).
RMS norm eps = 1e-6 everywhere.

## Full-attention layer (blk.3, 7, ..., every 4th)

1. `attn_q.weight` [2048 -> 8192] outputs INTERLEAVED per-head (query, gate):
   for head h (0..15): q_h = out[h*512 .. h*512+256), gate_h = out[h*512+256
   .. h*512+512).  NOT q-block-then-gate-block.
2. Per-head RMS norm on q with `attn_q_norm.weight` [256] (BEFORE rope).
3. k = `attn_k.weight` [2048 -> 512] -> reshape [2, 256] -> per-head RMS norm
   with `attn_k_norm.weight`.  v = `attn_v.weight` [2048 -> 512], no norm.
4. IMRoPE (rope_type = LLAMA_ROPE_TYPE_IMROPE) on q and k, n_rot=64,
   sections [11,11,10,0], freq_base=1e7:
   - NEOX pairing within the FIRST 64 dims of each 256-dim head: pair index
     ic in 0..31 rotates elements (x[ic], x[ic+32]).
   - theta_ic = position_channel * base^(-2*ic/64).
   - The imrope section interleave (ic%3 -> t/h/w channel) ONLY selects which
     of the 4 position channels supplies `position`.  FOR TEXT all channels
     carry the same token position => IMRoPE on text == plain NEOX partial
     rope over the first 64 dims.  Dims 64..255 pass through unrotated.
5. Standard causal GQA attention, kq_scale = 1/sqrt(256); q-head h uses
   kv-head h/8 (contiguous groups: heads 0..7 -> kv0, 8..15 -> kv1).
6. Output gating BEFORE the output projection:
   attn_out = attn_out * sigmoid(gate)   (elementwise, per head, [4096])
7. `attn_output.weight` [4096 -> 2048].

## Gated DeltaNet layer (all non-attention layers)

Dims: head_k_dim = head_v_dim = 128 (= ssm.state_size), n_k_heads = 16
(= ssm.group_count), n_v_heads = 32 (= ssm.time_step_rank), key_dim = 2048,
value_dim = 4096, conv channels = 2*key_dim + value_dim = 8192, conv
kernel = 4.

Projections from the normed input x:
- `attn_qkv.weight` [2048 -> 8192]: channels [q(2048) | k(2048) | v(4096)].
- `attn_gate.weight` [2048 -> 4096]: z (output gate), viewed [32, 128].
- `ssm_beta.weight` [2048 -> 32]:  beta = sigmoid(.) per v-head.
- `ssm_alpha.weight` [2048 -> 32]: g = ssm_a[h] * softplus(alpha[h] +
  ssm_dt.bias[h]) per v-head.  ssm_a [32] is stored NEGATIVE
  ("-A_log.exp()"), so g < 0 and exp(g) in (0,1) is the state decay.
  softplus = x > 20 ? x : log1p(exp(x)).

Causal depthwise conv (per channel, kernel 4, NO bias):
- conv_state holds the previous 3 raw qkv_mixed vectors (pre-conv).
- out[c] = sum_{t=0..3} w[t,c] * concat(state, current)[t,c]; then SiLU.
- state update: shift left, append current raw qkv_mixed.
- `ssm_conv1d.weight` ne=[4, 8192]: w[t,c] = data[c*4 + t] (t innermost).

Split conv output -> q [16,128], k [16,128], v [32,128]; then
- L2-normalize q and k per head: x / max(||x||_2, eps), eps = 1e-6
  (ggml_l2_norm: eps floors the NORM, it is not added under the sqrt).
- GQA repeat is ggml_repeat = TILE semantics: v-head h uses k-head h % 16
  (NOT h/2 interleave).  torch `.repeat` in the HF reference tiles the same
  way. [VERIFY in per-layer parity if outputs ever disagree.]
- q scaled by 1/sqrt(128).

Decode recurrence (n_tokens == 1), per v-head h, state S [128 x 128]
(S[i][j]: i = k-dim, j = v-dim):
```
S      <- exp(g_h) * S                    (decay whole matrix)
sk[j]   = sum_i S[i][j] * k[i]            (S^T k)
d[j]    = (v[j] - sk[j]) * beta_h         (delta rule)
S[i][j] += k[i] * d[j]                    (outer-product update)
out[j]  = sum_i S[i][j] * q[i]            (S^T q, q pre-scaled)
```
(Prefill uses a chunked formulation — build_delta_net_chunking — same math;
port later for prompt ingestion. Decode-only is enough for token loops.)

Output path:
- gated norm: rmsnorm(out; `ssm_norm.weight` [128], eps) * silu(z_h),
  per head, elementwise.
- concat heads [4096] -> `ssm_out.weight` [4096 -> 2048].

State per layer per sequence: conv 3*8192 floats + S 32*128*128 floats
(~2.1 MB f32).  10 attention layers keep KV cache instead (2 kv-heads x 256
x 2 = 1 KB/token f16).

## MoE FFN (every layer) — MATCHES gpu_ml MoeFfn as implemented

- router `ffn_gate_inp.weight` [2048 -> 256] (f32) -> softmax -> top-8 ->
  RENORMALIZE selected weights to sum 1 (norm_topk=true in build_moe_ffn).
- expert e: down_e( silu(gate_e(x)) * up_e(x) ), weighted sum.
- shared expert: same SiLU FFN shape, output scaled by
  sigmoid(`ffn_gate_inp_shexp.weight` . x) (scalar per token), ADDED to the
  routed sum.

## Verification anchors

- Attention/DeltaNet formulas above are transcriptions of
  build_layer_attn / build_layer_attn_linear / build_delta_net_autoregressive
  — not reconstructions from papers.
- Remaining uncertainty: none identified for text-only decode. The k-head
  tile-vs-interleave mapping is the one place a silent swap is conceivable;
  flag it first if per-layer parity vs llama.cpp diverges on DeltaNet layers.
