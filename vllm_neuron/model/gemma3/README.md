# Gemma3 (text decoder)

BF16 implementation of the Gemma3 text decoder for the Neuron backend. Tested with
[`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) and
[`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) (and finetunes). Text-only: the
SigLIP vision tower in the checkpoint is dropped at weight-load time. Every shape comes from the config.

Recommended run recipe (Trn2, one device, TP=4) and the measured throughput: [`tools/gemma3/README.md`](../../../tools/gemma3/README.md).

## Architecture

| Parameter | gemma-3-12b-it | gemma-3-4b-it |
|---|---|---|
| hidden_size / intermediate_size | 3840 / 15360 | 2560 / 10240 |
| num_hidden_layers | 48 (40 sliding + 8 global) | 34 (29 sliding + 5 global) |
| num_attention_heads / num_key_value_heads | 16 / 8 | 8 / 4 |
| head_dim | **256** (explicit, not hidden / heads) | **256** |
| vocab_size | 262208 | 262208 |
| Attention schedule | 5 sliding (window 1024) : 1 global | same |

Common to both:

| Parameter | Value |
|---|---|
| Normalization | RMSNorm, eps 1e-6, **`(1 + gamma)` form**, 4 per layer ("sandwich": input / post-attn / pre-FFN / post-FFN, post-norms before the residual add) |
| RoPE | **Dual** rotate_half: sliding θ=1e4 (unscaled), global θ=1e6 with linear ×8 (applied to `inv_freq`) |
| QK norm | per-head RMSNorm on `head_dim`, applied **before** RoPE |
| Softmax scale | **`query_pre_attn_scalar ** -0.5`**, not `head_dim ** -0.5` |
| Activation | `gelu_pytorch_tanh`, gated (`down(gelu(gate) * up)`) |
| Bias / soft-capping | none / none (rejected if configured) |
| tie_word_embeddings | **True**; the checkpoint has no `lm_head.weight` |
| Embedding scale | **bf16(`sqrt(hidden_size)`)** |

TP must divide the KV heads or be a multiple of them: TP=4 is the recommended layout on one Trn2 device
(2 KV heads per rank for the 12B model, 1 for the 4B model).

## Key Differences from Reference

Primary reference: [`model/qwen3/model.py`](../qwen3/model.py) (dense GQA with
per-head Q/K RMSNorm before RoPE). Per-layer sliding-window machinery — the
two-mask decode precompute and the per-layer `sliding_window_size` in
`get_kv_spec()` — is taken from
[`model/gpt_oss/model_bf16.py`](../gpt_oss/model_bf16.py).

1. **4-norm sandwich** instead of Qwen3's 2 norms, and the post-norms are
   applied to the sublayer output *before* the residual add. This is a
   structural difference, not just "two extra norms".
2. **`(1 + gamma)` RMSNorm.** HF's `Gemma3RMSNorm` initializes gamma to zeros
   and computes `norm(x) * (1 + gamma)` in fp32 before the downcast. This port
   folds the `+1` into the loaded gamma so every consumer uses the plain `w * x`
   form. **This fold is mandatory, not stylistic**: the decode mega-kernel's
   `rmsnorm_QK_pre_rope_W_Q`/`_W_K` arguments have no `(1 + w)` form, so without
   the fold decode would silently disagree with prefill.
3. **Dual RoPE tables**, computed once per step in the backbone and selected
   per layer by the sliding/global schedule. The linear ×8 scaling is applied to
   `inv_freq` (`inv_freq /= factor`), matching HF's
   `_compute_linear_scaling_rope_parameters`, not to the position ids — they are
   algebraically equal but not bit-identical.
4. **Per-layer hybrid attention** with a 5:1 schedule. Note gpt_oss uses
   `layer_idx % 2 == 0`; Gemma3's rule is `(layer_idx + 1) % pattern != 0`,
   resolved by `Gemma3Config.is_sliding()` (which prefers an explicit
   `layer_types` list when the config carries one).
5. **`query_pre_attn_scalar` softmax scale.** For Gemma3-4B this happens to
   equal `head_dim ** -0.5`, but the config fields are independent.
6. **`gelu_pytorch_tanh` written out by hand** rather than via `NF.mlp`. The torch fallback of the nkilib
   MLP calls `F.gelu(approximate="tanh")`, which breaks a `fullgraph=True` trace.
7. **`sqrt(hidden_size)` embedding scale, rounded to bf16.** HF downcasts the
   scale before multiplying (transformers#29402); using the fp32 value instead
   introduces a small systematic error in every embedding.
8. **Tied LM head** with no `lm_head.weight` in the checkpoint. Both parameters
   map to the same checkpoint key and load independently — they are deliberately
   *not* aliased in python before loading, which would leave one pointing at a
   stale meta tensor.
9. **Nested-config + vision drop.** `Gemma3Config.from_configs` reads the nested
   `text_config` and pops `vision_config`; `load_weights` detects the
   `language_model.model.` key prefix and never references `vision_tower.*` /
   `multi_modal_projector.*`.
10. **Registered under two architecture strings** (`Gemma3ForConditionalGeneration`
    and `Gemma3ForCausalLM`) because the text-only checkpoint declares the
    multimodal wrapper name.
11. **`head_dim = 256` changes which kernel arguments are legal.** The stock decode
    mega-kernel cannot fold the output projection or the KV cache update above head_dim 128.
    With the stock NKI Library this port passes `W_out=None` + `update_cache=False` and runs
    `NF.o_proj` and a deferred KV scatter outside the kernel. With the NKI Library branch
    [`gemma3-trn2-perf-2.33`](https://github.com/jimburtoft/nki-library/tree/gemma3-trn2-perf-2.33)
    (markers `SUPPORTS_D256_OUT_PROJ`, `SUPPORTS_D256_BLOCK_KV_UPDATE`) both run in the kernel.

## Feature Status

| Feature | Status | Notes |
|---|---|---|
| TP (head sharding) | ✅ | Q/KV heads + MLP intermediate + vocab sharded. TP=4 recommended. |
| SP (sequence parallel) | ✅ | Prefill all-gather before attention/MLP, reduce-scatter after. |
| DP (data parallel) | ✅ | No model-code changes needed. |
| Eagle3 spec decode | ❌ | No public Gemma3 Eagle3 draft. |
| FP8 KV cache | ❌ | `bind_kv_cache` raises on an fp8 cache. |
| Sliding window attention | ✅ | Per-layer window in prefill and decode. With `VLLM_NEURON_SWA_RING=1` each sliding layer keeps a fixed ring of window blocks per request (`vllm_neuron/vllm/core/swa_ring.py`). |
| SWA-aware KV admission | ✅ | `VLLM_NEURON_SWA_REAL_ADMISSION=1` + `VLLM_NEURON_SWA_EARLY_FREE=1` (`vllm_neuron/vllm/core/kv_admission.py`): admits by the real per-request peak instead of the worst case. |
| Single-shot prefill | ✅ | `max_num_batched_tokens` >= prompt length. |
| Segmented / chunked prefill, prefix caching | ⚠️ | The code path exists, but it is not validated at head_dim 256 and is refused with the SWA ring. Keep prefix caching off. |
| On-device sampling | ✅ | Greedy (`all_greedy`) measured. |
| Quantization (FP8/MX) | ❌ | BF16 only. |
| Vision tower | ❌ | Text-only. |

## Model-specific environment variables

All are read at trace time; the defaults are the recommended settings unless noted.

| Variable | Default | Effect |
|---|---|---|
| `GEMMA3_PACKED_PREFILL` | `0` | `1`: per-request KV bounds in packed prefill. The recipe sets it. |
| `GEMMA3_STOCK_DECODE` | `auto` | Fused decode kernel at batch >= 2, torch fallback at batch 1. `1` / `0` force one path. |
| `GEMMA3_KERNEL_KV_UPDATE` | `auto` | In-kernel KV write when the NKI Library supports it and the modular-flow partitioner is off (`VLLM_NEURON_MODULAR_FLOW_MAC_THRESHOLD=1e15`). |
| `GEMMA3_FUSED_OUT_PROJ` | `auto` | In-kernel output projection when the NKI Library supports it. `0` forces `NF.o_proj`. |
| `GEMMA3_KV_COALESCE`, `GEMMA3_PREFILL_KV_COALESCE`, `GEMMA3_KV_PTR_GROUP` | `1` | Deferred and coalesced KV write-back (one scatter per physical cache tensor per step). |
| `GEMMA3_FUSED_MLP` | `0` | Fused nkilib decode MLP. Slower; leave off. |

## Module Structure

```text
vllm_neuron/model/gemma3/
├── __init__.py     # exports the FACTORY's Gemma3ForCausalLM
├── README.md       # this file
├── config.py       # nested text_config extraction, dual-RoPE + schedule normalization
├── factory.py      # from_configs / _select_implementation / _validate_config
└── model.py        # 7 sections: RMSNorm, dual RoPE, attention, MLP, decoder
                    # layer, backbone, LM head + weight loading
```
