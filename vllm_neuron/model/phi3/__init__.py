# SPDX-License-Identifier: Apache-2.0
"""Phi-3 family (Phi-4, Phi-4-mini) on the Llama 3 implementation.

``Phi3ForCausalLM`` is a Llama-style dense decoder (GQA, RMSNorm, SiLU-gated
MLP, rotate-half RoPE, no biases). It differs from Llama in four ways, all
handled here without touching the Llama 3 kernels:

1. Fused checkpoint weights. ``self_attn.qkv_proj.weight`` holds [Q; K; V]
   rows and ``mlp.gate_up_proj.weight`` holds [gate; up] rows. A checkpoint
   wrapper exposes them under the Llama names (``q_proj`` ... ``up_proj``),
   so the Llama 3 loaders and their TP sharding are reused unchanged.
2. Partial rotary (``partial_rotary_factor`` < 1, Phi-4-mini: 96 of 128 dims).
   HF rotates dims [0, r) with rotate-half pairs (i, i + r/2) and passes the
   rest through. Llama 3 rotates the full head with pairs (i, i + D/2). The
   wrapper permutes the Q and K rows of every head (the same permutation on
   both, so q.k is unchanged) so that HF's rotary pairs land on Llama's pairs,
   and the pass-through dims land on pairs whose frequency is 0 (cos 1, sin 0).
3. LongRoPE (``rope_scaling.type: longrope``). The rescale factors divide the
   inverse frequencies and the attention factor scales cos/sin of the rotary
   dims. As in upstream vLLM (``Phi3LongRoPEScaledRotaryEmbedding``), the
   factor set is chosen once from ``max_model_len``: short factors when
   ``max_model_len <= original_max_position_embeddings``, long factors for
   every position otherwise (so the KV cache never mixes the two).
4. KV heads that do not divide TP (Phi-4: 10 KV heads at TP4). KV heads are
   replicated at load (10 -> 20) so every rank holds whole KV heads; each
   replica serves the same Q heads as the original.
"""

import math

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.llama3 import model as llama_model
from vllm_neuron.model.llama3.config import LlamaConfig
from vllm_neuron.model.llama3.factory import LlamaForCausalLM as _LlamaFactory
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint


def _cfg(hf_config, key, default=None):
    if isinstance(hf_config, dict):
        return hf_config.get(key, default)
    return getattr(hf_config, key, default)


def _rope_info(hf_config) -> dict:
    """Return the Phi-3 RoPE settings from either transformers 4.x or 5.x configs."""
    rp = _cfg(hf_config, "rope_parameters", None) or {}
    rs = _cfg(hf_config, "rope_scaling", None) or {}
    src = dict(rs)
    src.update({k: v for k, v in rp.items() if v is not None})
    theta = src.get("rope_theta", _cfg(hf_config, "rope_theta", 10000.0))
    rtype = src.get("rope_type", src.get("type", "default")) or "default"
    partial = src.get(
        "partial_rotary_factor", _cfg(hf_config, "partial_rotary_factor", 1.0)
    )
    if partial is None:
        partial = 1.0
    return {
        "rope_theta": float(theta),
        "rope_type": rtype,
        "partial_rotary_factor": float(partial),
        "short_factor": src.get("short_factor"),
        "long_factor": src.get("long_factor"),
        "attention_factor": src.get("attention_factor"),
        "original_max_position_embeddings": src.get(
            "original_max_position_embeddings",
            _cfg(hf_config, "original_max_position_embeddings", None),
        ),
        "max_position_embeddings": _cfg(hf_config, "max_position_embeddings"),
    }


def _kv_replication(num_heads: int, num_kv: int, tp: int) -> int:
    """Smallest r so that num_kv * r divides evenly over TP and into the Q heads."""
    if num_kv >= tp and num_kv % tp == 0:
        return 1
    if num_kv < tp and tp % num_kv == 0:
        return 1  # Llama 3 already replicates KV heads across ranks here.
    for r in range(2, num_heads // num_kv + 1):
        n = num_kv * r
        if num_heads % n == 0 and (n % tp == 0 or (n < tp and tp % n == 0)):
            return r
    raise NotImplementedError(
        f"Phi3: {num_kv} KV heads / {num_heads} Q heads cannot be laid out at TP={tp}"
    )


def _rotary_row_perm(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """new slot -> old dim, mapping HF partial rotate-half onto full rotate-half."""
    half, rh = head_dim // 2, rotary_dim // 2
    npass = half - rh
    perm = (
        list(range(0, rh))                              # slots [0, rh)        <- rot first half
        + list(range(rotary_dim, rotary_dim + npass))   # slots [rh, half)     <- pass first half
        + list(range(rh, rotary_dim))                   # slots [half, half+rh) <- rot second half
        + list(range(rotary_dim + npass, head_dim))     # slots [half+rh, D)   <- pass second half
    )
    assert sorted(perm) == list(range(head_dim))
    return perm  # plain list: built under a meta-device context; indexed on CPU at load


class _TensorSlice:
    """Minimal PySafeSlice stand-in over an in-memory tensor."""

    def __init__(self, t: torch.Tensor):
        self._t = t

    def get_shape(self):
        return list(self._t.shape)

    def get_dtype(self):
        return str(self._t.dtype).removeprefix("torch.")

    def __getitem__(self, idx):
        return self._t[idx]


class _VirtualIndex(dict):
    """tensor-name -> file index that also answers the virtual Llama names."""

    def __init__(self, owner: "_Phi3Checkpoint"):
        super().__init__()
        self._owner = owner

    def _src(self, key):
        return self._owner._virtual_source(key)

    def __contains__(self, key):
        if dict.__contains__(self, key):
            return True
        src = self._src(key)
        return src is not None and dict.__contains__(self, src[0])

    def __getitem__(self, key):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        src = self._src(key)
        if src is not None:
            return dict.__getitem__(self, src[0])
        raise KeyError(key)


class _Phi3Checkpoint(SafetensorsCheckpoint):
    """Exposes fused Phi-3 tensors under Llama 3 names (see module docstring)."""

    layout: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tensor_name_to_file = _VirtualIndex(self)

    def _virtual_source(self, key: str):
        for suffix in (
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
        ):
            if key.endswith(suffix):
                return (key[: -len(suffix)] + ".self_attn.qkv_proj.weight", suffix[11])
        for suffix in (".mlp.gate_proj.weight", ".mlp.up_proj.weight"):
            if key.endswith(suffix):
                part = "g" if "gate" in suffix else "u"
                return (key[: -len(suffix)] + ".mlp.gate_up_proj.weight", part)
        return None

    def _get_slice(self, name: str):
        if dict.__contains__(self._tensor_name_to_file, name):
            return super()._get_slice(name)
        src = self._virtual_source(name)
        if src is None:
            return super()._get_slice(name)
        fused_name, part = src
        fused = super()._get_slice(fused_name)
        L = self.layout
        D = L["head_dim"]
        if part in ("g", "u"):
            inter = L["intermediate_size"]
            lo = 0 if part == "g" else inter
            return _TensorSlice(fused[lo : lo + inter])
        q_rows = L["num_heads"] * D
        kv_rows = L["num_kv_heads"] * D
        if part == "q":
            t = fused[0:q_rows]
        elif part == "k":
            t = fused[q_rows : q_rows + kv_rows]
        else:
            t = fused[q_rows + kv_rows : q_rows + 2 * kv_rows]
        perm = L.get("row_perm")
        if perm is not None and part in ("q", "k"):
            idx = torch.tensor(perm, dtype=torch.long, device="cpu")
            t = t.view(-1, D, t.shape[-1]).index_select(1, idx).reshape(-1, t.shape[-1])
        r = L.get("kv_rep", 1)
        if r > 1 and part in ("k", "v"):
            t = (
                t.view(L["num_kv_heads"], 1, D, t.shape[-1])
                .expand(L["num_kv_heads"], r, D, t.shape[-1])
                .reshape(-1, t.shape[-1])
            )
        return _TensorSlice(t.contiguous())


class Phi3RotaryEmbedding(torch.nn.Module):
    """cos/sin in the Llama 3 rotate-half layout for Phi-3 partial/LongRoPE."""

    def __init__(self, head_dim: int, info: dict, use_long: bool = False):
        super().__init__()
        self.use_long = use_long
        self.head_dim = head_dim
        half = head_dim // 2
        rotary_dim = int(head_dim * info["partial_rotary_factor"])
        assert rotary_dim % 2 == 0 and rotary_dim <= head_dim
        rh = rotary_dim // 2
        # Pure Python (the model is constructed under a meta-device context), rounded
        # to fp32 at each step like HF / vLLM so long positions match their tables.
        import struct

        def f32(x):
            return struct.unpack("f", struct.pack("f", x))[0]

        pw = [f32(f32(info["rope_theta"]) ** f32(i / rotary_dim)) for i in range(0, rotary_dim, 2)]
        inv = [f32(1.0 / p) for p in pw]
        self.attention_scaling = 1.0
        if info["rope_type"] == "longrope":
            fac = [float(x) for x in info["long_factor" if use_long else "short_factor"]]
            assert len(fac) == rh, (len(fac), rh)
            inv = [f32(1.0 / f32(f32(b) * p)) for b, p in zip(fac, pw)]  # HF: 1 / (factor * base**x)
            af = info.get("attention_factor")
            orig = info["original_max_position_embeddings"]
            if af is None:
                factor = info["max_position_embeddings"] / orig
                af = 1.0 if factor <= 1.0 else math.sqrt(1 + math.log(factor) / math.log(orig))
            self.attention_scaling = float(af)
        elif info["rope_type"] not in ("default", None):
            raise NotImplementedError(f"Phi3: rope_type {info['rope_type']!r}")
        # Plain Python lists, not buffers: the model may be built on meta (CPU
        # compile) and buffers would not be re-materialized. Tensors are built
        # on CPU in forward and moved, as the Llama 3 rotary does.
        full = [0.0] * half
        full[:rh] = inv
        scale = [1.0] * half
        scale[:rh] = [self.attention_scaling] * rh
        self.inv_freq_list = full
        self.rot_scale_list = scale + scale

    def forward(self, position_ids, device, dtype):
        inv = torch.tensor(self.inv_freq_list, dtype=torch.float32).to(device)
        s = torch.tensor(self.rot_scale_list, dtype=torch.float32).to(device)
        freqs = position_ids[:, None].float() * inv[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = (emb.cos() * s).to(dtype=dtype), (emb.sin() * s).to(dtype=dtype)
        import os
        if os.environ.get("PHI3_DEBUG_ROPE") == "1":
            print("PHI3ROPE", tuple(position_ids.shape), position_ids.dtype, tuple(cos.shape),
                  bool(torch.isnan(cos).any()), position_ids.flatten()[:4].tolist(), flush=True)
        return cos, sin


class _Phi3Model(llama_model.LlamaForCausalLM):
    """Llama 3 model with Phi-3 weight layout, rotary and KV replication."""

    phi3_layout: dict = {}

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        from vllm.distributed import get_tp_group

        info = _rope_info(hf_config)
        config = LlamaConfig.from_configs(hf_config, neuron_config)
        n_h = _cfg(hf_config, "num_attention_heads")
        n_kv = _cfg(hf_config, "num_key_value_heads")
        H = _cfg(hf_config, "hidden_size")
        head_dim = _cfg(hf_config, "head_dim", None) or H // n_h
        tp = get_tp_group().world_size
        rep = _kv_replication(n_h, n_kv, tp)

        # Silent-config-drop guards: every field the Llama 3 code reads.
        config.head_dim = head_dim
        config.num_key_value_heads = n_kv * rep
        config.rope_parameters = {"rope_type": "default", "rope_theta": info["rope_theta"]}
        config.tie_word_embeddings = bool(_cfg(hf_config, "tie_word_embeddings", False))
        # The compute dtype is vLLM's resolved --dtype, not the checkpoint's.
        try:
            from vllm.config import get_current_vllm_config

            config.torch_dtype = get_current_vllm_config().model_config.dtype
        except Exception:
            pass
        assert config.hidden_size == H
        assert config.num_attention_heads == n_h
        assert config.intermediate_size == _cfg(hf_config, "intermediate_size")
        assert config.num_hidden_layers == _cfg(hf_config, "num_hidden_layers")
        assert config.vocab_size == _cfg(hf_config, "vocab_size")
        assert abs(config.rms_norm_eps - _cfg(hf_config, "rms_norm_eps")) < 1e-12
        kv_rank = config.num_key_value_heads
        assert (kv_rank % tp == 0) or (tp % kv_rank == 0), (kv_rank, tp)

        rotary_dim = int(head_dim * info["partial_rotary_factor"])
        layout = {
            "head_dim": head_dim,
            "num_heads": n_h,
            "num_kv_heads": n_kv,
            "kv_rep": rep,
            "intermediate_size": config.intermediate_size,
            "row_perm": None if rotary_dim == head_dim else _rotary_row_perm(head_dim, rotary_dim),
            "rope": info,
        }
        model = cls(config)
        model.phi3_layout = layout
        use_long = False
        if info["rope_type"] == "longrope":
            try:
                from vllm.config import get_current_vllm_config

                mml = get_current_vllm_config().model_config.max_model_len
            except Exception:  # no vLLM config (unit tests)
                mml = 0
            use_long = mml > info["original_max_position_embeddings"]
        model.model.rotary_emb = Phi3RotaryEmbedding(head_dim, info, use_long)
        import os

        if os.environ.get("PHI3_DEBUG_ROPE") == "1":
            def _hook(name):
                def h(mod, args, kwargs, out):
                    o = out[0] if isinstance(out, tuple) else out
                    am = kwargs.get("attn_metadata", "absent")
                    kc = getattr(mod, "k_cache", None)
                    print("PHI3NAN", name, tuple(o.shape), "out_nan", bool(torch.isnan(o).any()),
                          "absmax", float(o.float().abs().max()), "am_none", am is None,
                          "kcache_nan", None if kc is None else bool(torch.isnan(kc).any()), flush=True)
                return h
            for i, layer in enumerate(model.model.layers):
                layer.self_attn.register_forward_hook(_hook(f"L{i}.attn"), with_kwargs=True)
        import logging

        logging.getLogger(__name__).info(
            "Phi3 on llama3: heads %d/%d (kv x%d -> %d), head_dim %d, rotary_dim %d, "
            "rope %s%s theta %.0f attn_scale %.6f, tie %s, tp %d, dtype %s",
            n_h, n_kv, rep, n_kv * rep, head_dim, rotary_dim, info["rope_type"],
            " (long factors)" if use_long else "",
            info["rope_theta"], model.model.rotary_emb.attention_scaling,
            config.tie_word_embeddings, tp, config.torch_dtype,
        )
        return model

    def _with_phi3_checkpoint(self, fn, *args):
        cls = type("_BoundPhi3Checkpoint", (_Phi3Checkpoint,), {"layout": self.phi3_layout})
        orig = llama_model.SafetensorsCheckpoint
        llama_model.SafetensorsCheckpoint = cls
        try:
            return fn(*args)
        finally:
            llama_model.SafetensorsCheckpoint = orig

    def load_weights(self, checkpoint_path, device, cache_dir):
        r = self._with_phi3_checkpoint(
            super().load_weights, checkpoint_path, device, cache_dir
        )
        import os

        if os.environ.get("PHI3_DEBUG_ROPE") == "1":
            for n, p in self.named_parameters():
                if "layers.0." in n or "layers" not in n:
                    t = p.detach().float().cpu()
                    print("PHI3W", n, tuple(t.shape), "absmax", float(t.abs().max()),
                          "nan", bool(torch.isnan(t).any()), flush=True)
        return r

    def load_weights_lite(self, checkpoint_path, device, cache_dir):
        return self._with_phi3_checkpoint(
            super().load_weights_lite, checkpoint_path, device, cache_dir
        )


class Phi3ForCausalLM(_LlamaFactory):
    """Factory: validate the Phi-3 config, then build the Llama 3-based model."""

    @classmethod
    def _select_implementation(cls, hf_config, neuron_config):
        cls._validate_config(hf_config, neuron_config)
        return _Phi3Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(cls, hf_config, neuron_config):
        sw = _cfg(hf_config, "sliding_window", None)
        max_pos = _cfg(hf_config, "max_position_embeddings", None)
        if sw is not None and (max_pos is None or sw < max_pos):
            raise NotImplementedError(
                f"Phi3ForCausalLM with sliding_window={sw} < max_position_embeddings "
                f"is not supported (the Llama 3 path is full attention only)."
            )
        if _cfg(hf_config, "attention_bias", False) or _cfg(hf_config, "mlp_bias", False):
            raise NotImplementedError("Phi3: biases are not supported")
        super()._validate_config(hf_config, neuron_config)
