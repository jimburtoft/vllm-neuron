# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE MTP draft head for vLLM-Neuron 0.29 (EAGLE-style proposer flow).

Checkpoint keys ``mtp.*`` (one full-attention layer + 256-expert MoE):
  h_in  = fc(cat[pre_fc_norm_embedding(embed(tok_{t+1})), pre_fc_norm_hidden(h_t)])
  h_out = norm(layer(h_in))           logits = lm_head(h_out)
where h_t is the target's final normed hidden state at position t. embed_tokens and
lm_head are the target's (loaded from the same checkpoint). The draft is run k times
(the same layer), feeding its own h_out back as h_t.

Runs on the decode path only (S=1 per request per pass); the initial pass covers the
target's T verify/prefill tokens so the draft KV for every position is written.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.llama3.eagle3_model import extract_accepted_tokens


def compute_slot_mapping(positions, block_table_tensor, block_size):
    """slot = block_table[pos // bs] * bs + pos % bs, with the block column clamped
    to the table (padded rows can carry out-of-range positions) and -1 for unused
    blocks."""
    pos = positions.long().clamp(min=0)
    col = (pos // block_size).clamp(max=block_table_tensor.shape[1] - 1)
    bid = block_table_tensor.gather(1, col.view(-1, 1)).view(-1).long()
    slot = bid * block_size + pos % block_size
    return torch.where(bid < 0, torch.full_like(slot, -1), slot)
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import Qwen3_5MoeTextConfig
from .model import PartialRotary, Qwen3_5Attention, Qwen3_5SparseMoe, RMSNorm, _plus_one_loader

logger = logging.getLogger(__name__)


class _MTPMoe(Qwen3_5SparseMoe):
    """MTP MoE: experts are stored per expert (experts.N.{gate,up,down}_proj)."""

    def __init__(self, config):
        super().__init__(config)
        I, tpd = self.I, self.tp_degree

        def gate_up_t(s, r):  # 2*E slices: gate_0, up_0, gate_1, up_1, ...
            rr = r % tpd
            out = []
            for e in range(len(s) // 2):
                g = s[2 * e][rr * I : (rr + 1) * I, :]
                u = s[2 * e + 1][rr * I : (rr + 1) * I, :]
                out.append(torch.cat([g, u], 0).t())
            return torch.stack(out, 0).contiguous()

        def down_t(s, r):
            rr = r % tpd
            return torch.stack([x[:, rr * I : (rr + 1) * I].t() for x in s], 0).contiguous()

        set_weight_loader(self.gate_up_proj_weight, SafetensorsWeightLoader(transform=gate_up_t))
        set_weight_loader(self.down_proj_weight, SafetensorsWeightLoader(transform=down_t))


class _MTPLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        eps = config.rms_norm_eps
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps, torch.float32)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps, torch.float32)
        set_weight_loader(self.input_layernorm.weight, _plus_one_loader())
        set_weight_loader(self.post_attention_layernorm.weight, _plus_one_loader())
        self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = _MTPMoe(config)

    def forward(self, h, positions, cos, sin, attn_metadata):
        md = attn_metadata[self.self_attn.layer_name]
        is_decode = md["max_query_len"] <= md["decode_token_threshold"]
        r = h
        xn = self.input_layernorm(h)
        if is_decode:
            x = self.self_attn.forward_decode(xn, positions, cos, sin, md)
        else:  # tokens are NOT sequence-sharded in the draft
            x = self.self_attn.forward_prefill(xn, positions, cos, sin, md, sp=False)
        h = r + x
        hn = self.post_attention_layernorm(h)
        y = self.mlp.forward_decode(hn) if is_decode else self.mlp.forward_prefill(hn, positions, sp=False)
        return h + y


class _Only(nn.Module):
    """View of a module exposing only the parameters named in a mapping (so the
    pipelined loader does not try to read the target-shared embed/lm_head)."""

    def __init__(self, mod, names):
        super().__init__()
        self._m = [mod]
        self._names = set(names)

    def named_parameters(self, *a, **k):
        return [(n, p) for n, p in self._m[0].named_parameters() if n in self._names]

    def parameters(self, *a, **k):
        return [p for _, p in self.named_parameters()]


class Qwen3_5MoeMTPForCausalLM(nn.Module):
    def __init__(self, config: Qwen3_5MoeTextConfig, start_layer_idx: int):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        dt = config.torch_dtype
        H = config.hidden_size
        self.layer_idx = start_layer_idx
        self.num_speculative_tokens = 1
        self.kv_fill_only = False
        self.target_mlp_dp_size = 1

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size, embed_dim=H, dtype=dt, tp_group=self.tp_group.device_group
        )
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0, shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size, pad_shard=True,
            ),
        )
        eps = config.rms_norm_eps
        self.pre_fc_norm_embedding = RMSNorm(H, eps, torch.float32)
        self.pre_fc_norm_hidden = RMSNorm(H, eps, torch.float32)
        self.norm = RMSNorm(H, eps, torch.float32)
        for m in (self.pre_fc_norm_embedding, self.pre_fc_norm_hidden, self.norm):
            set_weight_loader(m.weight, _plus_one_loader())
        # fc [2H -> H], replicated (small: 2048 x 4096)
        self.fc_weight = nn.Parameter(torch.empty(2 * H, H, dtype=dt))
        set_weight_loader(self.fc_weight, SafetensorsWeightLoader(transform=lambda s, r: s[0][:].t().contiguous()))
        self.layer = _MTPLayer(config, start_layer_idx)
        self.rotary = PartialRotary(config)
        self.lm_head = neuron_nn.ColumnParallelLinear(
            H, config.vocab_size, bias=False, dtype=dt, gather_output=False,
            tp_group=self.tp_group.device_group,
        )
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader(shard_dim=0, shard_size=config.vocab_size // self.world_size,
                                   num_shards=self.world_size),
        )
        nc = config.neuron_config
        self.on_device_sampling = nc is not None and nc.on_device_sampling_config is not None
        if self.on_device_sampling:
            self.sampler = Sampler(nc.on_device_sampling_config, process_group=self.tp_group.device_group)
        self.attn_name = f"layers.{start_layer_idx}.self_attn"

    @classmethod
    def from_configs(cls, config, start_layer_idx, neuron_config=None):
        cfg = Qwen3_5MoeTextConfig.from_configs(config, neuron_config)
        return cls(cfg, start_layer_idx)

    # one MTP pass: tokens (next ids) + previous hidden -> new hidden, logits-sampled token
    def _pass(self, ids, h_prev, positions, attn_metadata, rank):
        e = self.embed_tokens(ids, scatter_tokens=False, rank=rank)
        x = torch.cat([self.pre_fc_norm_embedding(e), self.pre_fc_norm_hidden(h_prev.to(e.dtype))], -1)
        x = x @ self.fc_weight
        cos, sin = self.rotary(positions)
        h = self.norm(self.layer(x, positions, cos, sin, attn_metadata))
        return h

    def _sample(self, h, rank):
        logits = self.lm_head(h)
        if self.on_device_sampling:
            return self.sampler(logits, tp_rank=rank).to(torch.int32)
        return torch.argmax(logits, -1).to(torch.int32)

    @torch.no_grad()
    def forward(
        self, input_ids, positions, initial_target_hidden_states, attn_metadata=None,
        sampling_positions=None, rank=None, raw_sampled_token_ids=None,
        prev_sampled_token_ids=None, prev_num_draft_tokens=None, req_indices_per_token=None,
    ):
        import vllm_neuron.functional as NF

        if prev_sampled_token_ids is not None and prev_num_draft_tokens is not None \
                and req_indices_per_token is not None:
            positions, attn_metadata = NF.correct_spec_decode_positions_and_slot_mapping(
                positions, attn_metadata, prev_sampled_token_ids, prev_num_draft_tokens,
                req_indices_per_token, self.config.vocab_size,
            )
        positions = positions.to(torch.int32)
        bonus = None
        if raw_sampled_token_ids is not None:
            input_ids, sampling_positions, bonus = extract_accepted_tokens(
                input_ids, sampling_positions, raw_sampled_token_ids,
                self.config.vocab_size, self.num_speculative_tokens,
            )
        md = attn_metadata[self.attn_name]
        # initial pass over all target tokens (writes draft KV at each position)
        h = self._pass(input_ids, initial_target_hidden_states, positions,
                       {self.attn_name: md}, rank)
        h = h.index_select(0, sampling_positions.long())
        pos = positions.index_select(0, sampling_positions.long())
        tok = self._sample(h, rank)
        toks = ([bonus, tok] if bonus is not None else [tok])
        bt, bs, mb = md["block_table_tensor"], md["block_size"], md["max_blocks_per_seq"]
        for _ in range(self.num_speculative_tokens - 1):
            pos = pos + 1
            step_md = {self.attn_name: {
                "block_table_tensor": bt, "slot_mapping": compute_slot_mapping(pos.long(), bt, bs),
                "max_query_len": 1, "block_size": bs, "max_blocks_per_seq": mb,
                "decode_token_threshold": 1,
            }}
            h = self._pass(tok, h, pos, step_md, rank)
            tok = self._sample(h, rank)
            toks.append(tok)
        stacked = torch.stack(toks, 1)
        drafts = torch.stack(toks[1:] if bonus is not None else toks, 1)
        return stacked, drafts, None

    def get_kv_spec(self):
        a = self.layer.self_attn
        return KVSpec(layers=[LayerSpec(name=self.attn_name, num_kv_heads=a.nkh,
                                        head_size=a.head_dim, dtype=a.dtype)])

    def bind_kv_cache(self, kv_caches):
        kv = kv_caches[self.attn_name]
        self.layer.self_attn.k_cache, self.layer.self_attn.v_cache = kv[0], kv[1]

    def load_weights(self, checkpoint_path, device, cache_dir=None):
        E = self.config.num_experts
        m = {
            "pre_fc_norm_embedding.weight": "mtp.pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight": "mtp.pre_fc_norm_hidden.weight",
            "norm.weight": "mtp.norm.weight",
            "fc_weight": "mtp.fc.weight",
        }
        p, hp = "layer", "mtp.layers.0"
        m[f"{p}.input_layernorm.weight"] = f"{hp}.input_layernorm.weight"
        m[f"{p}.post_attention_layernorm.weight"] = f"{hp}.post_attention_layernorm.weight"
        a = f"{p}.self_attn"
        m[f"{a}.qkv_proj_weight"] = [f"{hp}.self_attn.{x}_proj.weight" for x in ("q", "k", "v")]
        m[f"{a}.o_proj_weight"] = f"{hp}.self_attn.o_proj.weight"
        m[f"{a}.q_norm.weight"] = f"{hp}.self_attn.q_norm.weight"
        m[f"{a}.k_norm.weight"] = f"{hp}.self_attn.k_norm.weight"
        m[f"{p}.mlp.router_weight"] = f"{hp}.mlp.gate.weight"
        m[f"{p}.mlp.gate_up_proj_weight"] = [
            f"{hp}.mlp.experts.{e}.{w}_proj.weight" for e in range(E) for w in ("gate", "up")
        ]
        m[f"{p}.mlp.down_proj_weight"] = [f"{hp}.mlp.experts.{e}.down_proj.weight" for e in range(E)]
        m[f"{p}.mlp.shared_gate_up"] = [f"{hp}.mlp.shared_expert.gate_proj.weight",
                                        f"{hp}.mlp.shared_expert.up_proj.weight"]
        m[f"{p}.mlp.shared_down"] = f"{hp}.mlp.shared_expert.down_proj.weight"
        m[f"{p}.mlp.shared_gate_w"] = f"{hp}.mlp.shared_expert_gate.weight"
        ckpt = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank = self.tp_group.rank_in_group
        sd = ckpt.load_sharded_pipelined(
            rank, self.world_size, _Only(self, m), m, device
        ).state_dict
        pdt = {n: q.dtype for n, q in self.named_parameters()}
        for n, t in sd.items():
            if n in pdt and t.dtype != pdt[n]:
                sd[n] = t.to(pdt[n])
        shared = ("embed_tokens.weight", "lm_head.weight")  # aliased to the target
        for n in shared:
            sd.pop(n, None)
        missing = [n for n in pdt if n not in sd and n not in shared]
        assert not missing, f"MTP unloaded params: {missing[:6]}"
        self.load_state_dict(sd, strict=False, assign=True)
        # Placeholders until MTPProposer.alias_target_weights shares the target's
        # tensors (keeps .to(device) from touching meta storage).
        self.embed_tokens.weight = nn.Parameter(torch.empty(0, dtype=self.config.torch_dtype, device=device),
                                                requires_grad=False)
        self.lm_head.weight = nn.Parameter(torch.empty(0, dtype=self.config.torch_dtype, device=device),
                                           requires_grad=False)
