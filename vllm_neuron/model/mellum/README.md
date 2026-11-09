# Mellum

BF16 implementation of Mellum (`MellumForCausalLM`) for the Neuron backend. Tested with
[`JetBrains/Mellum2.1-12B-A2.5B-Thinking`](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking).
Every shape comes from the config.

Run recipe and accuracy gate: [`tools/mellum/README.md`](../../../tools/mellum/README.md).

## Architecture

| Parameter | Mellum2.1-12B-A2.5B |
|---|---|
| Parameters | 12.1B total, 2.5B active |
| hidden_size / num_hidden_layers / vocab_size | 2304 / 28 / 98304 |
| num_attention_heads / num_key_value_heads / head_dim | 32 / 4 / 128 |
| Attention schedule (`layer_types`) | 3 sliding-window (1024) : 1 full, 21 + 7 layers |
| RoPE (`rope_parameters`, nested by layer type) | sliding: default, theta 5e5. full: YaRN, theta 5e5, factor 16, original 8192, attention_factor 1.2773 |
| QK norm | per-head RMSNorm on head_dim, before RoPE |
| MoE (every layer) | 64 experts, top-8, softmax router in fp32, renormalized top-k, SiLU GLU, moe_intermediate_size 896, no shared expert |
| Normalization | RMSNorm, eps 1e-6, `w * norm(x)` |
| tie_word_embeddings | False |

## Implementation

| Part | Source |
|---|---|
| Attention (TP head sharding, QK norm, per-layer sliding window in prefill and in the fused decode kernel) | `Gemma3Attention` from [`model/gemma3`](../gemma3/model.py) with Mellum's norms; head_dim 128 lets the decode kernel fold the output projection |
| Decode masks (sliding + full), SWA position rebase, deferred and coalesced KV writes, KV spec | `model/gemma3` |
| RoPE | `MellumRotaryEmbedding`: default on sliding layers, YaRN (transformers' `_compute_yarn_parameters`) on full layers, cos/sin times `attention_factor` |
| MoE | `MellumSparseMoeBlock`: router + routed experts, TP on the expert intermediate size, optional EP |
| Config | `MellumConfig` parses the nested `rope_parameters` and asserts the schedule at load time |

vLLM 0.29 ships a `MellumConfig` that subclasses `Qwen3MoeConfig`. It adds a flat `rope_theta` next to the
nested RoPE dicts, and `vllm serve` then fails in `validate_rope` with
`AttributeError: 'float' object has no attribute 'get'`. The plugin replaces it with the transformers
`MellumConfig` (`vllm_neuron/vllm/patches/mellum_config_patch.py`).

## MoE paths

| Path | Default | When |
|---|---|---|
| Decode: routed-expert gather (`MELLUM_MOE_DECODE=auto`) | T x 8 <= local experts | fp32 router; reads only the T x top-8 routed experts (`index_select`, chunks of 64) |
| Decode: dense (`MELLUM_MOE_DECODE=auto`) | T x 8 > local experts | fp32 router; every local expert once, weighted by the affinity |
| Decode: fused `moe_block_tkg` (`MELLUM_MOE_DECODE=kernel`) | no | Needs I/rank % 128 == 0 (EP=4). Its router runs in bf16 and fails the fp32 gate on some prompts |
| Prefill with EP: `shard_on_i_hybrid` kernel (`MELLUM_MOE_PREFILL_IMPL=auto`) | yes | Static loop over the expected non-empty blocks per rank, then a dynamic loop over the rest; padding tokens get no routes. 2.6x faster than `shard_on_block` at 4k prompts (534 -> 206 ms), which computes every worst-case block (about 70 % empty at EP4) and spills SBUF at I = 896 |
| Prefill without EP: blockwise `moe_cte` (`shard_on_block`) kernel | yes | The experts' H is zero-padded to 2560 inside the MoE block (the kernel tiles H by 512). Without EP the kernel's hidden-state I/O is fp32 (`MELLUM_MOE_CTE_IO=auto`): its bf16 output accumulation over the 8 experts of a token moved a near-tie under segmented prefill |
| Prefill: dense torch MoE | `MELLUM_MOE_PREFILL=torch` | Slower (TTFT about 2x at 4k prompts without EP) |

## Parallelism

One Trn2 device, LNC=2: `--tensor-parallel-size 4`, with or without `--enable-expert-parallel`
(EP=4: 16 experts per rank). EP is recommended. 4 KV heads give 1 KV head per rank at TP=4.

## Feature status

| Feature | Status | Notes |
|---|---|---|
| TP=4 (EP=1, EP=4) | ✅ | fp32 first-divergence gate PASS on 7 prompts (up to 3.5k input, 1k output) |
| Sliding window attention | ✅ | Requires `VLLM_NEURON_SWA_ALIGN_TOKENS=1024` (or the ring) at LNC=2: the trimmed window table must be a multiple of 256 tokens |
| SWA ring / real admission / early free | ✅ | `VLLM_NEURON_SWA_RING=1` etc. The ring refuses segmented prefill and prefix caching |
| Segmented prefill / prefix caching | ✅ | With the ring off. Gate PASS at EP=4 with APC on, three repeats |
| Reasoning parser (`qwen3`), tool calls (`hermes`) | ✅ | `vllm serve` |
| Packed prefill (several prompts per prefill step) | ✅ | `VLLM_NEURON_MAX_PREFILLS_PER_BATCH=8`. Whole-prompt prefill only (`--max-num-batched-tokens` >= `--max-model-len`, no prefix caching, no speculation, no DP); the model refuses other configurations |
| MTP speculative decoding | ❌ | The checkpoint has no MTP head |
| FP8 weights / KV | ❌ | Not implemented |

## Model-specific environment variables

| Variable | Default | |
|---|---|---|
| `MELLUM_MOE_DECODE` | `auto` | `gather` / `dense` force a path; `kernel`: fused moe_block_tkg |
| `MELLUM_MOE_PREFILL` | `kernel` | `torch`: dense torch MoE in prefill |
| `MELLUM_MOE_PREFILL_IMPL` | `auto` | `hybrid` (EP default) / `block` (no-EP default) / `shard_on_i` |
| `MELLUM_MOE_PREFILL_NSB` | expected blocks | Static block count of the hybrid kernel |
| `MELLUM_MOE_MASK_PAD_ROUTE` | `auto` | Padding tokens get no expert routes (on with the hybrid kernel) |
| `MELLUM_MOE_CTE_IO` | `auto` | Prefill kernel hidden-state dtype: fp32 without EP, bf16 with EP. `fp32` / `bf16` force |
| `MELLUM_MOE_MAP_SHARD` | `0` | `1`: split the token-to-block mapping across the MoE TP ranks (same result) |
| `MELLUM_MOE_ROUTER` | `kernel` | `torch`: fp32 torch router in prefill |
| `MELLUM_MOE_PAD_H` | `1` | `0`: no H padding (prefill must then use `torch`) |
| `MELLUM_MOE_DECODE_HSLICE` | `auto` | The dense decode path reads only the real 2,304 H rows of the padded experts. `1` / `0` force it for both paths |
| `MELLUM_MOE_PAD_I` | `0` | `1`: zero-pad the expert width per rank to a multiple of 128 |
| `MELLUM_MOE_PAD_MASK` | `1` | Padding rows take no expert blocks in prefill |
| `VLLM_NEURON_MAX_PREFILLS_PER_BATCH` | `1` | `N > 1`: pack up to N waiting prompts into one prefill step (per-request attention bounds and MoE padding mask) |
| `MELLUM_MOE_FP8_ROUTED` | `0` | `1` (no EP only): FP8 e4m3 (128x128 block scales) copy of the experts and the fused routed-expert SwiGLU decode kernel (`swiglu_moe_fp8.py`) for decode batches up to `MELLUM_MOE_FP8_ROUTED_MAX_T` (24). Needs `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION=0.2` (the FP8 copy is +3.3 GB per core) |
| `MELLUM_DECODE_ATTN` | `stock` | `torch`: torch reference of the decode attention kernel (CPU mode at TP=1) |
