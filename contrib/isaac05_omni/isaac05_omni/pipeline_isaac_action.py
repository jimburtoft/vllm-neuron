# SPDX-License-Identifier: Apache-2.0
"""IsaacActionPipeline: Isaac-0.5 robot policy as a single vLLM-Omni DIFFUSION stage on Neuron (public 0.29 plugin).

One OpenPI observation (``sampling_params.extra_args["robot_obs"]``) -> a normalized + unnormalized action chunk:

1. host: the LeRobot ``PerceptronIsaacPolicy`` preprocessor renders the mHarmony TensorStream; the checkpoint's own
   VLA embedding modules (text table, Qwen3.5 vision tower, proprio vector MLP; CPU) produce ``inputs_embeds`` and the
   3-D mRoPE positions (``Qwen35VLAModel.forward`` semantics, NTP truncation, non-spatial tokens collapsed).
2. device, TP over the stage: the vllm-neuron Isaac decoder (``Qwen3_5MoeForCausalLM`` layers + final norm, null-expert
   MoE, nkilib GDN / MoE CTE) -> post-norm final hidden states (``action.vlm_prefill_hidden``), one compiled graph per
   prompt bucket.
3. device, rank-local: the checkpoint's DiT action expert, 10-step Euler with the dim mask (``rtc.integrate_rtc_euler``
   semantics, flow_time = i / steps), one compiled graph per (bucket, horizon).
4. host: the policy postprocessor unnormalizes; ``DiffusionOutput(output={"actions": ...})`` (OpenPI reads
   ``multimodal_output["actions"]``).

Deploy: ``contrib/isaac05_omni/examples/isaac05_action_tp4.yaml`` (single requests) or ``isaac05_action_tp4_batch.yaml``
(request batching).
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import types

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

PIPELINE_REGISTRY = [{"model_arch": "IsaacActionPipeline", "class_name": "IsaacActionPipeline"}]
PREFILL_BUCKETS = (512, 1024, 2048)
BLOCK = 256


def _ckpt_module(ckpt: str, name: str):
    pkg = "isaac05_ckpt"
    if pkg not in sys.modules:
        p = types.ModuleType(pkg)
        p.__path__ = [ckpt]
        sys.modules[pkg] = p
    return importlib.import_module(f"{pkg}.{name}")


def _all_tp_ranks(group):
    """The full TP partition of the world (every replica's ranks), as NeuronGroupCoordinator needs the whole mesh."""
    world = torch.distributed.get_world_size()
    tp = group.world_size
    return [list(range(i, i + tp)) for i in range(0, world, tp)]


def _patch_png_level0():
    """mHarmony's render takes images as PNG bytes, so LeRobot PNG-encodes every frame and mHarmony decodes it again.
    PNG is lossless at any zlib level: level 0 (stored) gives the same pixels after decode and roughly halves
    encode time. The tokens and pixels are unchanged."""
    from PIL import Image as PILImage

    from lerobot.policies.perceptron_isaac import mharmony_native as mn

    if getattr(mn, "_isaac05_png0", False):
        return
    import base64
    import io

    def _image_to_png_b64(image):
        if isinstance(image, bytes | bytearray):
            with PILImage.open(io.BytesIO(bytes(image))) as pil:
                arr = np.asarray(pil.convert("RGB"))
        else:
            arr = mn.coerce_rgb_uint8(image, contiguous=True,
                                      shape_error="Unsupported Isaac image shape for native mharmony: {shape}.")
        buf = io.BytesIO()
        PILImage.fromarray(arr, mode="RGB").save(buf, format="PNG", compress_level=0)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    mn._image_to_png_b64 = _image_to_png_b64
    mn._isaac05_png0 = True


_RAW_IMAGES: dict = {}


def _patch_raw_images():
    """skip the PNG round trip entirely. LeRobot's mHarmony render PNG-encodes every frame, and mHarmony
    (Rust) base64-decodes it and hands the bytes to the python qwen35 image processor, which PIL-decodes them. The
    frames are uint8 RGB already, so pass a short handle instead and resolve it to the array in the processor. The
    pixels are identical (PNG is lossless), so the patches and tokens are identical."""
    import base64

    from mharmony.processors import qwen35 as q

    from lerobot.policies.perceptron_isaac import mharmony_native as mn

    if getattr(mn, "_isaac05_raw", False):
        return

    def _image_to_png_b64(image):
        if isinstance(image, bytes | bytearray):
            from PIL import Image as PILImage
            import io

            with PILImage.open(io.BytesIO(bytes(image))) as pil:
                arr = np.asarray(pil.convert("RGB"))
        else:
            arr = mn.coerce_rgb_uint8(image, contiguous=True,
                                      shape_error="Unsupported Isaac image shape for native mharmony: {shape}.")
        key = b"ISAACRAW%d" % len(_RAW_IMAGES)
        _RAW_IMAGES[key] = np.ascontiguousarray(arr)
        return base64.b64encode(key).decode("ascii")

    orig = q.process_qwen35_image_bytes

    class _ArrImage:  # minimal PIL.Image stand-in for process_qwen35_image_bytes' decode block
        def __init__(self, arr):
            self._a = arr
            self.height, self.width = int(arr.shape[0]), int(arr.shape[1])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def convert(self, mode):
            return self

        def __array__(self, dtype=None, copy=None):
            return self._a

    def process(image_bytes, **kw):
        arr = _RAW_IMAGES.pop(bytes(image_bytes), None)
        if arr is None:
            return orig(image_bytes, **kw)
        import PIL.Image as PILImage

        real_open = PILImage.open
        PILImage.open = lambda *_a, **_k: _ArrImage(arr)
        exif = q._apply_exif_orientation
        q._apply_exif_orientation = lambda im: im
        try:
            return orig(image_bytes, **kw)
        finally:
            PILImage.open = real_open
            q._apply_exif_orientation = exif

    q.process_qwen35_image_bytes = process
    mn._image_to_png_b64 = _image_to_png_b64
    mn._isaac05_raw = True


def _rmsnorm_noamp(self, x):  # ActionExpertRMSNorm without the autocast guard (no AMP on neuron); same math
    xf = x.to(torch.float32)
    out = (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)
    return out * self.weight if self.weight is not None else out


class IsaacActionPipeline(nn.Module):
    weights_sources: list = []
    vae = None
    # vllm-omni request batching (stage max_num_seqs > 1): forward() receives a DiffusionRequestBatch of up
    # to MAXB observations; they run as ONE packed VLM prefill (B x 512 tokens) and ONE batched DiT plan.
    supports_request_batch = True

    def __init__(self, *, od_config, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        mc = dict(getattr(od_config, "model_config", None) or {})
        self.ckpt = od_config.model
        self.policy_dir = mc.get("policy_dir", os.path.join(self.ckpt, "lerobot_policy"))
        self.num_steps = int(mc.get("num_inference_steps", 10))
        self.policy_server_config = mc.get("policy_server_config", {})
        self._device = torch.device("cpu")
        self._built = False
        self._vision_dp = os.environ.get("ISAAC05_VISION_DP", "0") == "1"  # DP vision 63 vs 40 ms (patch broadcast + gather), off

    # ---------------------------------------------------------------- build / load
    def _build(self):
        from vllm.config import CacheConfig, SchedulerConfig, VllmConfig, set_current_vllm_config

        from vllm_neuron.model.qwen3_5_moe.config import Qwen3_5MoeTextConfig
        from vllm_neuron.model.qwen3_5_moe.model import Qwen3_5MoeForCausalLM

        # The text model reads scheduler / cache config at construction (request slots, prefix-caching guard).
        import copy

        from vllm.config import get_current_vllm_config

        vc = copy.copy(get_current_vllm_config())  # the diffusion worker's config (parallel config already set)
        self._maxb = int(os.environ.get("ISAAC05_MAXB", "1"))
        vc.scheduler_config = SchedulerConfig(max_num_seqs=self._maxb, max_num_batched_tokens=max(PREFILL_BUCKETS),
                                              max_model_len=max(PREFILL_BUCKETS), is_encoder_decoder=False)
        vc.cache_config = CacheConfig(block_size=BLOCK, enable_prefix_caching=False)
        os.environ.setdefault("QWEN35_DECODE_MK", "0")
        # one T=512 prefill fills ~5 % of the 256-token MoE blocks; 128-token blocks cut the per-block
        # memset / DMA work (VLM 225 -> 200 ms on trn2, action cos unchanged). EP=4 (ISAAC05_EP=1) measured no gain.
        os.environ.setdefault("ISAAC_MOE_BLOCK", "128")
        # job-list MoE prefill kernel (kernels/moe_prefill_jobs.py): VLM 200 -> ~140 ms
        os.environ.setdefault("ISAAC_MOE_PREFILL_IMPL", "jobs")
        # The omni diffusion worker's TP group is vllm-omni's GroupCoordinator (no reduce_scatter / stamped rank tensor);
        # the vllm-neuron decoder needs a NeuronGroupCoordinator over the same ranks (same mesh as the omni group).
        import vllm.distributed.parallel_state as vps

        from vllm_neuron.parallel.neuron_group_coordinator import NeuronGroupCoordinator

        omni_tp = vps.get_tp_group()
        ntp = NeuronGroupCoordinator(group_ranks=[list(r) for r in _all_tp_ranks(omni_tp)], local_rank=omni_tp.local_rank,
                                     torch_distributed_backend=torch.distributed.get_backend(omni_tp.cpu_group),
                                     use_device_communicator=True, group_name="isaac_tp")
        self._neuron_tp = ntp
        if os.environ.get("ISAAC05_EP", "0") == "1" and ntp.world_size > 1:
            # EP = TP over the group: each rank holds 1/tp of the experts at full intermediate (EP path). The
            # action path is one prefill with no decode, which removes the EP decode penalty that ruled EP out for serving.
            import vllm_neuron.parallel.neuron_parallel_state as nps

            tp = ntp.world_size
            world = torch.distributed.get_world_size()
            be = torch.distributed.get_backend(omni_tp.cpu_group)
            nps._NEURON_EP_TP = NeuronGroupCoordinator(group_ranks=[[r] for r in range(world)], local_rank=omni_tp.local_rank,
                                                       torch_distributed_backend=be, use_device_communicator=True,
                                                       group_name="isaac_ep_tp")
            nps._NEURON_EP = NeuronGroupCoordinator(group_ranks=[list(r) for r in _all_tp_ranks(omni_tp)],
                                                    local_rank=omni_tp.local_rank, torch_distributed_backend=be,
                                                    use_device_communicator=True, group_name="isaac_ep")
            vc.parallel_config = copy.copy(vc.parallel_config)
            vc.parallel_config.enable_expert_parallel = True
            self._ep_groups = (nps._NEURON_EP_TP, nps._NEURON_EP)
            # vllm-neuron's get_world_group() is not initialized inside the omni diffusion worker; EP degree = TP here.
            nps.get_neuron_ep_degree = lambda: tp
        saved_tp = vps._TP
        vps._TP = ntp
        try:
            self._build_models(vc, set_current_vllm_config, Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM)
        finally:
            vps._TP = saved_tp
        self._built = True

    def _build_models(self, vc, set_current_vllm_config, Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM):
        with set_current_vllm_config(vc):
            cfg = Qwen3_5MoeTextConfig.from_configs(os.path.join(self.ckpt, "config.json"))
            with torch.device("meta"):
                self.lm = Qwen3_5MoeForCausalLM(cfg)
        V = _ckpt_module(self.ckpt, "modeling_qwen35_vla")
        V.ActionExpertRMSNorm.forward = _rmsnorm_noamp
        full = json.load(open(os.path.join(self.ckpt, "config.json")))
        self.expert = V.build_action_expert_head(full["action_expert"], vlm_dim=full["text_config"]["hidden_size"])

    def load_weights(self, weights=None):
        if not self._built:
            self._build()
        from safetensors import safe_open

        dev = self._device
        self.lm.load_weights(self.ckpt, dev, None)
        self._bind_scratch_kv(dev)
        idx = json.load(open(os.path.join(self.ckpt, "model.safetensors.index.json")))["weight_map"]
        pre = "model.action_expert."
        sd = {}
        for k, f in idx.items():
            if k.startswith(pre):
                with safe_open(os.path.join(self.ckpt, f), "pt") as h:
                    sd[k[len(pre):]] = h.get_tensor(k)
        self.expert.load_state_dict(sd, strict=True)
        self.expert = self.expert.to(torch.bfloat16).to(dev).eval()
        self._load_host_side()
        return set()

    def _bind_scratch_kv(self, dev):
        """GQA layers write their K/V into a scratch paged cache (one observation; nothing kept between requests)."""
        nblk = max(PREFILL_BUCKETS) // BLOCK + 1
        kv = {}
        for layer in self.lm.layers:
            if layer.is_full:
                a = layer.self_attn
                shape = (nblk, a.nkh, BLOCK, a.head_dim)
                kv[a.layer_name] = (torch.zeros(shape, dtype=a.dtype, device=dev), torch.zeros(shape, dtype=a.dtype, device=dev))
        self.lm.bind_kv_cache(kv)

    def _load_host_side(self):
        """LeRobot policy (render + normalization) and the checkpoint's VLA embedding modules on CPU."""
        from lerobot.configs import PreTrainedConfig
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.perceptron_isaac import PerceptronIsaacPolicy

        pc = PreTrainedConfig.from_pretrained(self.policy_dir)
        h, w = pc.image_size
        pc.input_features = {
            **{f"observation.images.{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w)) for c in pc.camera_order},
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(pc.proprio_dim,)),
        }
        pc.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(pc.action_dim,))}
        pc.device = "cpu"
        if os.environ.get("ISAAC05_RAW_IMAGES", "1") == "1":
            _patch_raw_images()
        elif os.environ.get("ISAAC05_PNG_LEVEL0", "1") == "1":
            _patch_png_level0()
        self.policy = PerceptronIsaacPolicy.from_pretrained(self.policy_dir, config=pc)
        self.pcfg = self.policy.config
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        acfg = AutoConfig.from_pretrained(self.ckpt, trust_remote_code=True, local_files_only=True)
        top = get_class_from_dynamic_module(acfg.auto_map["AutoModelForCausalLM"], self.ckpt)
        vmod = sys.modules[top.__module__.rsplit(".", 1)[0] + ".modeling_isaac05_vla"]
        acfg.text_config.num_hidden_layers = 1
        acfg.text_config.layer_types = acfg.text_config.layer_types[:1]
        with torch.device("meta"):
            emb = vmod.Isaac05Qwen36VLAModel(acfg)
        from safetensors import safe_open

        idx = json.load(open(os.path.join(self.ckpt, "model.safetensors.index.json")))["weight_map"]
        sd = {}
        for k, f in idx.items():
            if k.startswith(("model.visual.", "model.vector_embedding.", "model.language_model.embed_tokens.")):
                with safe_open(os.path.join(self.ckpt, f), "pt") as hh:
                    sd[k[len("model."):]] = hh.get_tensor(k).float()
        emb.to_empty(device="cpu")
        emb.load_state_dict(sd, strict=False)
        vmod.reset_isaac05_qwen36_nonpersistent_buffers(emb)
        self.embedder = emb.float().eval()
        # Vision tower on the device (replicated per rank; uniform-grid image batches, one graph per (N, H, W)).
        # ISAAC05_VISION=cpu keeps the HF CPU path (reference).
        if os.environ.get("ISAAC05_VISION", "device") == "device":
            from .vision import IsaacVisionTower

            vt = IsaacVisionTower(self.embedder.visual)
            self._vision_tower = vt
            self._vision_graphs = {}
            cpu_embed_vision = self.embedder._embed_vision
            pipe = self

            def _embed_vision(patches, grid_thw):
                g = grid_thw.tolist()
                if pipe._device.type == "cpu" or any(tuple(r) != tuple(g[0]) or r[0] != 1 for r in g):
                    return cpu_embed_vision(patches, grid_thw)
                n_all, (_, h, w) = len(g), g[0]
                dev = pipe._device
                pos, cos, sin = vt.tables(h, w, "cpu")
                grp = pipe._neuron_tp
                dp = pipe._vision_dp and grp.world_size > 1
                if dp:
                    import time as _tv

                    t0 = _tv.time()
                    # patches go as ONE bf16 tensor broadcast (exactly what the device graph consumes);
                    # pickling the fp32 patches in broadcast_object_list cost ~0.5 s per 96 images
                    pb = patches.to(torch.bfloat16).contiguous()
                    # static per-rank slice: pad the image count to the batch bucket's (one graph per bucket)
                    nob = max(1, getattr(pipe, "_vis_nobs", 1))
                    n_pad = max(n_all, getattr(pipe, "_vis_bucket", nob) * (n_all // nob))
                    nsl = -(-n_pad // grp.world_size)
                    torch.distributed.broadcast_object_list([("vision", (tuple(pb.shape), n_all, h, w, nsl))],
                                                            src=grp.ranks[0], group=pipe._cpu_pg())
                    torch.distributed.broadcast(pb, src=grp.ranks[0], group=pipe._cpu_pg())
                    out = pipe._vision_slice(pb, n_all, h, w, nsl)
                    pipe._tim_vision = _tv.time() - t0
                    return out
                n = n_all
                key = (n, h, w)
                if key not in pipe._vision_graphs:
                    pipe._vision_graphs[key] = (torch.compile(vt.forward, fullgraph=True, dynamic=False,
                                                              **pipe._compile_kwargs),
                                                (pos.to(torch.bfloat16).to(dev), cos.to(torch.bfloat16).to(dev),
                                                 sin.to(torch.bfloat16).to(dev)))
                fn, tabs = pipe._vision_graphs[key]
                import time as _tv

                t0 = _tv.time()
                x = patches.view(n, h * w, -1).to(torch.bfloat16).to(dev)
                out = fn(x, *tabs).cpu().float()
                pipe._tim_vision = _tv.time() - t0
                return out

            self.embedder._embed_vision = _embed_vision
        self._stream_tools = (_ckpt_module(self.ckpt, "tensor_stream_utils"), _ckpt_module(self.ckpt, "tensor_stream"))
        stub = self

        class _Capture:  # the policy's model: records the rendered stream instead of sampling
            def sample_action(self, stream, **kw):
                stub._stream = stream
                return torch.zeros(1, kw.get("num_action_steps") or stub.pcfg.chunk_size, 64)

            def parameters(self):
                yield torch.zeros(1, dtype=torch.bfloat16)

        self.policy._isaac_model = _Capture()
        self.policy._load_backbone = lambda *a, **k: None
        self.policy._model_dtype = lambda: torch.bfloat16
        self.policy._resolve_device = lambda: "cpu"

    def _cpu_pg(self):
        """CPU (gloo) group for host-side collectives: the pipeline thread's private group while it runs a host part,
        else the TP group's cpu group."""
        pg = getattr(self, "_host_pg", None)
        return pg if pg is not None else self._neuron_tp.cpu_group

    def _recv_patches(self, data):
        shape, n_all, h, w, nsl = data
        grp = self._neuron_tp
        pb = torch.empty(shape, dtype=torch.bfloat16)
        torch.distributed.broadcast(pb, src=grp.ranks[0], group=self._cpu_pg())
        return pb, n_all, h, w, nsl

    def _vision_slice(self, patches, n_all, h, w, n=None):
        """Data-parallel vision: images [r*n, (r+1)*n) on group rank r (tail zero-padded, one graph for all ranks),
        then all-gather of the merged features over gloo. Returns all images' features on every rank."""
        grp, dev, vt = self._neuron_tp, self._device, self._vision_tower
        tp, r = grp.world_size, grp.rank_in_group
        if n is None:
            n = -(-n_all // tp)
        pv = patches.view(n_all, h * w, -1)
        sl = pv[min(r * n, n_all):min((r + 1) * n, n_all)]
        if sl.shape[0] < n:
            sl = torch.cat([sl, sl.new_zeros(n - sl.shape[0], *sl.shape[1:])], 0)
        key = (n, h, w)
        if key not in self._vision_graphs:
            pos, cos, sin = vt.tables(h, w, "cpu")
            self._vision_graphs[key] = (torch.compile(vt.forward, fullgraph=True, dynamic=False, **self._compile_kwargs),
                                        (pos.to(torch.bfloat16).to(dev), cos.to(torch.bfloat16).to(dev),
                                         sin.to(torch.bfloat16).to(dev)))
        fn, tabs = self._vision_graphs[key]
        out = fn(sl.to(torch.bfloat16).to(dev), *tabs).cpu()  # bf16 graph output: gather it as bf16 (exact)
        parts = [torch.empty_like(out) for _ in range(tp)]
        torch.distributed.all_gather(parts, out, group=self._cpu_pg())
        m = out.shape[0] // n  # merged tokens per image
        return torch.cat(parts, 0)[: n_all * m].float()  # rank r's rows [r*n, (r+1)*n): concat order = image order

    def to(self, *args, **kwargs):
        """The runner loads on CPU, then moves the pipeline: move the decoder (params + model-owned buffers + the
        GDN mask pack), the DiT expert and re-bind the scratch KV on the device. Host-side modules stay on CPU."""
        dev = args[0] if args else kwargs.get("device")
        if isinstance(dev, (str, torch.device)):
            dev = torch.device(dev)
            if dev != self._device and self._built:
                self.lm.to(dev)
                self.lm.gdn_mask_pack = self.lm.gdn_mask_pack.to(dev)
                if getattr(self.lm, "vocab_padded", False):
                    self.lm.vocab_pad_bias = self.lm.vocab_pad_bias.to(dev)
                self.expert = self.expert.to(dev)
                if getattr(self, "_vision_tower", None) is not None:
                    self.embedder.visual.to(torch.bfloat16).to(dev)
                self._bind_scratch_kv(dev)
                # SPMD: the group rank as an on-device graph input (one NEFF for all ranks)
                self._neuron_tp.rank_in_group_tensor = torch.tensor([self._neuron_tp.rank_in_group], dtype=torch.int32,
                                                                    device=dev)
                for g in getattr(self, "_ep_groups", ()):
                    g.rank_in_group_tensor = torch.tensor([g.rank_in_group], dtype=torch.int32, device=dev)
            self._device = dev
        return self

    # ---------------------------------------------------------------- compile
    def compile(self, *args, **kwargs):
        self._compile_kwargs = {k: v for k, v in kwargs.items() if k in ("backend", "options")}
        self._vlm_graphs = {}
        self._dit_graphs = {}
        return self

    def _vlm_fn(self, T, n_real):
        key = (T, n_real)
        if key not in self._vlm_graphs:
            from vllm_neuron.model.qwen3_5_moe.action import vlm_prefill_hidden

            nblk = T // BLOCK
            dev = self._device
            md = {"block_size": BLOCK, "block_table_tensor": (torch.arange(1, nblk + 1, dtype=torch.int32) .view(1, -1)).to(dev),
                  "slot_mapping": torch.cat([torch.arange(BLOCK, BLOCK + n_real, dtype=torch.int64),
                                             torch.zeros(T - n_real, dtype=torch.int64)]).to(dev),
                  "max_query_len": T, "decode_token_threshold": 1,
                  # one fresh request (no cached prefix), no decode follows: attend over the live keys and
                  # skip the KV-cache write (ISAAC05_FRESH_ATTN=0 restores the serving path)
                  "fresh": os.environ.get("ISAAC05_FRESH_ATTN", "1") == "1",
                  "skip_kv_write": os.environ.get("ISAAC05_FRESH_ATTN", "1") == "1"}
            slots = torch.full((1,), self.lm.num_request_slots - 1, dtype=torch.int32).to(dev)
            fn = lambda e, p: vlm_prefill_hidden(self.lm, e, p, md, slots, n_real=n_real)  # noqa: E731
            self._vlm_graphs[key] = torch.compile(fn, fullgraph=True, dynamic=False, **self._compile_kwargs)
        return self._vlm_graphs[key]

    def _vlm_packed_fn(self, B, seg):
        key = ("packed", B, seg)
        if key not in self._vlm_graphs:
            from vllm_neuron.model.qwen3_5_moe.action import vlm_prefill_hidden_packed

            dev = self._device
            md = {"block_size": BLOCK, "max_query_len": B * seg, "decode_token_threshold": 1, "skip_kv_write": True,
                  "seg": seg if os.environ.get("ISAAC05_BATCH_ATTN", "seg") == "seg" else 0,
                  "slot_mapping": torch.zeros(B * seg, dtype=torch.int64).to(dev)}
            slots = torch.arange(B, dtype=torch.int32).to(dev)  # GDN state rows 0..B-1 (scratch, nothing kept)
            fn = lambda e, p, nr: vlm_prefill_hidden_packed(self.lm, e, p, md, slots, seg, nr)  # noqa: E731
            self._vlm_graphs[key] = torch.compile(fn, fullgraph=True, dynamic=False, **self._compile_kwargs)
        return self._vlm_graphs[key]

    def _dit_modulation(self):
        """the AdaLN modulation of every block depends only on the flow time, and the Euler times are fixed
        (i / steps). Compute all of it once on the device (same ops as forward_with_context) and keep it resident,
        so the 10-step graph does not re-read the 36 modulation linears (1/3 of the DiT weight bytes) every step.
        Layout [steps, 1 + 9 * n_blocks + 2, H]: conditioning, 9 chunks per block, 2 final chunks."""
        if getattr(self, "_dit_mod", None) is None:
            ae = self.expert.action_expert
            steps = self.num_steps
            dt = next(ae.parameters()).dtype

            def mods(t_all):
                rows = []
                for i in range(steps):
                    cond = ae._time_conditioning(t_all[i:i + 1])
                    parts = [cond]
                    for blk in ae.blocks:
                        parts.extend(blk.modulation(cond).chunk(9, dim=-1))
                    parts.extend(ae.final_layer.modulation(cond).chunk(2, dim=-1))
                    rows.append(torch.stack([p.reshape(-1) for p in parts], 0))
                return torch.stack(rows, 0)

            t_all = torch.tensor([i / steps for i in range(steps)], dtype=dt).to(self._device)
            fn = torch.compile(mods, fullgraph=True, dynamic=False, **self._compile_kwargs)
            self._dit_mod = fn(t_all)
        return self._dit_mod

    def _dit_fn(self, L, horizon, out_dim, B=1):
        key = (L, horizon, out_dim, B)
        if key not in self._dit_graphs:
            ex = self.expert
            steps = self.num_steps
            premod = os.environ.get("ISAAC05_DIT_PREMOD", "1") == "1"
            SM = _ckpt_module(self.ckpt, "rtc").ActionExpertStepModulation
            nb = len(ex.action_expert.blocks)

            def plan(ctx, mask, noise, mod):
                context = ex._build_single_context(ctx, mask, None, seq_len=horizon, batch_size=ctx.shape[0],
                                                   device=ctx.device, dtype=ctx.dtype)
                dm = (torch.arange(noise.shape[-1], device=noise.device) < out_dim).to(torch.float32).view(1, 1, -1)
                x = noise.float() * dm
                for i in range(steps):  # rtc.integrate_rtc_euler (no RTC prefix), flow_time = i / steps
                    t = torch.full((1,), i / steps, dtype=ctx.dtype, device=ctx.device)
                    sm = None
                    if premod:
                        r = mod[i].unsqueeze(1)  # [rows, 1, H]
                        sm = SM(conditioning=r[0], block_modulations=[tuple(r[1 + 9 * b + j] for j in range(9))
                                                                       for b in range(nb)],
                                final_modulation=(r[1 + 9 * nb], r[2 + 9 * nb]))
                    v = ex.action_expert.forward_with_context(x.to(ctx.dtype), t, context=context,
                                                              modulation=sm).float() * dm
                    x = (x + v / steps) * dm
                return x

            g = torch.compile(plan, fullgraph=True, dynamic=False, **self._compile_kwargs)
            self._dit_graphs[key] = (lambda c, m, n: g(c, m, n, self._dit_modulation())) if premod else \
                (lambda c, m, n: g(c, m, n, None))
        return self._dit_graphs[key]

    # ---------------------------------------------------------------- request
    def _embed(self, robot_obs):
        """OpenPI observation dict (LeRobot keys) -> (inputs_embeds [L, H] bf16, mRoPE [3, L], context mask [L])."""
        return self._embed_stream(self._render(robot_obs))

    def _render(self, robot_obs):
        """OpenPI observation dict -> rendered TensorStream (LeRobot + mHarmony, host)."""
        obs = {}
        for k, v in robot_obs.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if torch.is_tensor(v) and k.startswith("observation.") and v.dim() in (1, 3):
                v = v[None]
            obs[k] = v
        if isinstance(obs.get("task"), str):
            obs["task"] = [obs["task"]]
        import time as _t

        t0 = _t.time()
        self.policy.reset()
        self.policy.predict_action_chunk(obs)
        self._tim_render = _t.time() - t0
        return self._stream.to(device="cpu", dtype=torch.float32)

    def _embed_streams(self, streams):
        """B rendered streams of EQUAL length -> one batched embed_stream (one device vision call over all
        B x 6 images). Returns a list of (e [L, H], pos [3, L], cmask [L]) per observation."""
        tsu, ts = self._stream_tools
        mod_ts = sys.modules[type(streams[0]).__module__]
        cat = mod_ts.TensorStream([s_.shallow_copy() for t_ in streams for s_ in t_.streams])
        Bn = len(streams)
        with torch.no_grad():
            e = self.embedder.embed_stream(cat)[:, :-1]
        pos = tsu.compute_mrope_pos_tensor(cat)
        mod = tsu.modality_mask(cat)
        nsp = ~((mod == ts.VisionType.I.value) | (mod == ts.VisionType.P.value))
        pos = pos.clone()
        pos[nsp] = pos[nsp][..., 0:1].expand(-1, pos.shape[-1])
        L = e.shape[1]
        start = tsu.first_event_start_indices(cat, ts.FLOW_ACTION_TEXT_TYPE, fallback_start=int(L))
        cmask, _ = tsu.build_action_context_mask(cat, action_batch_indices=list(range(Bn)), action_start_indices=start,
                                                 l_model=L, device=torch.device("cpu"))
        out = []
        for b in range(Bn):
            pb = pos[b:b + 1].permute(2, 0, 1)[:, 0, :-1].contiguous()  # [3, L]
            out.append((e[b].to(torch.bfloat16), pb, cmask[b]))
        return out

    def _embed_many(self, obs_list):
        """Render each observation (host), then embed them in one batch when the streams have equal length."""
        import time as _t

        t0 = _t.time()
        streams = [self._render(o) for o in obs_list]
        t1 = _t.time()
        lens = {tuple(s_.shape) for s_ in streams}
        self._tim_vision = 0.0
        if len(streams) > 1 and len(lens) == 1 and os.environ.get("ISAAC05_BATCH_EMBED", "1") == "1":
            out = self._embed_streams(streams)
        else:
            out = [self._embed_stream(s_) for s_ in streams]
        self._tim_batch_host = {"render": round((t1 - t0) * 1e3, 1), "embed": round((_t.time() - t1) * 1e3, 1),
                                "vision": round(self._tim_vision * 1e3, 1)}
        return out

    def _embed_stream(self, stream):
        tsu, ts = self._stream_tools
        with torch.no_grad():
            e = self.embedder.embed_stream(stream)[:, :-1]
        pos = tsu.compute_mrope_pos_tensor(stream)
        mod = tsu.modality_mask(stream)
        nsp = ~((mod == ts.VisionType.I.value) | (mod == ts.VisionType.P.value))
        pos = pos.clone()
        pos[nsp] = pos[nsp][..., 0:1].expand(-1, pos.shape[-1])
        pos = pos.permute(2, 0, 1)[:, 0, :-1].contiguous()  # [3, L]
        L = e.shape[1]
        start = tsu.first_event_start_indices(stream, ts.FLOW_ACTION_TEXT_TYPE, fallback_start=int(L))
        cmask, _ = tsu.build_action_context_mask(stream, action_batch_indices=[0], action_start_indices=start, l_model=L,
                                                 device=torch.device("cpu"))
        return e[0].to(torch.bfloat16), pos, cmask[0]

    @torch.no_grad()
    def forward(self, req, **kwargs):
        from vllm_omni.diffusion.data import DiffusionOutput

        reqs = getattr(req, "requests", None)
        if reqs is not None:  # DiffusionRequestBatch (request-batch mode)
            if len(reqs) == 1:
                return [self._forward_one(reqs[0])]
            return self._forward_batch(reqs)
        return self._forward_one(req)

    def _forward_batch(self, reqs):
        """a wave of R requests. With ISAAC05_PIPE=1 and R >= 2 * ISAAC05_PIPE_MIN, the wave is split into
        sub-batches that run as a 2-stage pipeline: the host part (render + embed + vision, rank 0 + DP vision) of
        sub-batch i+1 runs in a worker thread while the device part (packed VLM prefill + DiT) of sub-batch i runs.
        Device executes release the GIL (measured: 222 ms device || 221 ms Python -> 272 ms), so the two overlap."""
        n = len(reqs)
        k = int(os.environ.get("ISAAC05_PIPE_SPLIT", "2"))
        # Measured result: OFF by default. The "host" half is mostly the DP vision tower, which runs on the
        # same NeuronCores as the VLM/DiT (vision 108 -> 260 ms when overlapped), and two B=8 sub-batches cost more
        # device time than one B=16 (~1100 vs 881 ms): a 16-request wave took ~1.45 s pipelined vs ~1.34 s plain.
        if os.environ.get("ISAAC05_PIPE", "0") != "1" or n < 2 * int(os.environ.get("ISAAC05_PIPE_MIN", "4")) or k < 2:
            return self._forward_batch_one(reqs)
        import concurrent.futures as cf

        step = -(-n // k)
        parts = [reqs[i:i + step] for i in range(0, n, step)]
        outs = []
        ex = getattr(self, "_pipe_pool", None)
        if ex is None:
            ex = self._pipe_pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="isaac-host")
        # All 4 ranks run this same schedule (same request list), so every rank's host thread issues the same
        # host-side collectives (gloo broadcasts / gathers on the cpu group) in the same order, and every rank's main
        # thread issues the same device graphs in the same order. The host thread uses its own gloo group
        # (self._pipe_pg) so its collectives never interleave with the main thread's on one communicator.
        self._ensure_pipe_group()
        fut = ex.submit(self._batch_host_pg, parts[0])
        for i in range(len(parts)):
            hp = fut.result()
            if i + 1 < len(parts):
                fut = ex.submit(self._batch_host_pg, parts[i + 1])  # host work of the next sub-batch ...
            outs.extend(self._batch_device(hp))  # ... overlaps this sub-batch's device work
        return outs

    def _ensure_pipe_group(self):
        if getattr(self, "_pipe_pg", None) is None:
            grp = self._neuron_tp
            # every rank creates the same new group (collective call, main thread, same order on all ranks)
            self._pipe_pg = torch.distributed.new_group(ranks=list(grp.ranks), backend="gloo")

    def _batch_host_pg(self, reqs):
        self._host_pg = self._pipe_pg
        try:
            with torch.inference_mode():  # grad mode is thread-local: the pool thread starts with grad enabled
                return self._batch_host(reqs)
        finally:
            self._host_pg = None

    def _forward_batch_one(self, reqs):
        return self._batch_device(self._batch_host(reqs))

    def _batch_host(self, reqs):
        """Host half of a batch: parse, render, embed (vision on the device, DP over the ranks), bucket padding."""
        from vllm_omni.diffusion.data import DiffusionOutput
        import time as _t

        chunk, adim = self.pcfg.chunk_size, self.pcfg.action_dim
        t0 = _t.time()
        obs_l, noises, seeds = [], [], []
        for r in reqs:
            extra = getattr(r.sampling_params, "extra_args", None) or {}
            o = dict(extra.get("robot_obs") or {})
            noises.append(o.pop("noise", None))
            seeds.append(getattr(r.sampling_params, "seed", None))
            obs_l.append(o)
        real = [o for o in obs_l if o]
        NR = len(reqs)
        # static graph shapes: pad the batch to the next bucket (ISAAC05_BATCH_BUCKETS, default powers of 2 <= MAXB)
        bks = [int(x) for x in os.environ.get("ISAAC05_BATCH_BUCKETS", "").split(",") if x] or \
            [b for b in (1, 2, 4, 8, 16, 32) if b <= max(self._maxb, 1)]
        B = next((b for b in sorted(bks) if b >= NR), NR)
        obs_l = obs_l + [{}] * (B - NR)
        noises = noises + [None] * (B - NR)
        seeds = seeds + [0] * (B - NR)
        if not real:  # warmup / dummy batch
            return {"dummy": [DiffusionOutput(output={"actions": np.zeros((chunk, adim), dtype=np.float32)})
                              for _ in range(NR)]}
        grp = self._neuron_tp
        def _emb_all():
            idx = [i for i, o in enumerate(obs_l) if o]
            got = self._embed_many([obs_l[i] for i in idx])
            res = [None] * len(obs_l)
            for i, g_ in zip(idx, got):
                res[i] = g_
            return res

        vdp = grp.world_size > 1 and len(real) >= int(os.environ.get("ISAAC05_BATCH_VISION_DP_MIN", "2"))
        self._vis_nobs, self._vis_bucket = len(real), B
        if grp.world_size > 1 and vdp:
            # data-parallel vision for batches (B x 6 images split over the 4 ranks; rank 0 broadcasts the
            # patches, all ranks encode their slice, gather). Other ranks serve vision slices until the payload.
            if grp.rank_in_group == 0:
                self._vision_dp = True
                try:
                    emb = _emb_all()
                finally:
                    self._vision_dp = False
                torch.distributed.broadcast_object_list([("payload", emb)], src=grp.ranks[0], group=self._cpu_pg())
            else:
                while True:
                    msg = [None]
                    torch.distributed.broadcast_object_list(msg, src=grp.ranks[0], group=self._cpu_pg())
                    kind, data = msg[0]
                    if kind == "payload":
                        emb = data
                        break
                    self._vision_slice(*self._recv_patches(data))
        elif grp.world_size > 1:
            payload = [_emb_all() if grp.rank_in_group == 0 else None]
            torch.distributed.broadcast_object_list(payload, src=grp.ranks[0], group=self._cpu_pg())
            emb = payload[0]
        else:
            emb = _emb_all()
        th = _t.time() - t0
        return {"emb": emb, "B": B, "NR": NR, "noises": noises, "seeds": seeds, "th": th,
                "host": dict(getattr(self, "_tim_batch_host", {}))}

    def _batch_device(self, hp):
        """Device half of a batch: packed VLM prefill + batched DiT (+ unnormalize)."""
        from vllm_omni.diffusion.data import DiffusionOutput
        import time as _t

        if "dummy" in hp:
            return hp["dummy"]
        chunk, adim = self.pcfg.chunk_size, self.pcfg.action_dim
        emb, B, NR, noises, seeds, th = hp["emb"], hp["B"], hp["NR"], hp["noises"], hp["seeds"], hp["th"]
        fill = next(x for x in emb if x is not None)
        emb = [x if x is not None else fill for x in emb]  # dummy rows reuse a real observation (outputs dropped)
        SEG = PREFILL_BUCKETS[0]
        Ls = [x[0].shape[0] for x in emb]
        assert max(Ls) <= SEG, f"batched action path supports L <= {SEG} per observation, got {max(Ls)}"
        Hd = emb[0][0].shape[1]
        ep = torch.zeros(B * SEG, Hd, dtype=torch.bfloat16)
        pp = torch.zeros(3, B * SEG, dtype=torch.int64)
        for b, (e, pos, _) in enumerate(emb):
            L = e.shape[0]
            ep[b * SEG:b * SEG + L] = e
            pp[:, b * SEG:b * SEG + SEG] = pos[:, -1:]
            pp[:, b * SEG:b * SEG + L] = pos
        dev = self._device
        nr = torch.tensor(Ls, dtype=torch.int32)
        t1 = _t.time()
        h = self._vlm_packed_fn(B, SEG)(ep.to(dev), pp.to(torch.int32).to(dev), nr.to(dev))
        Lm = max(Ls)
        # contiguous [B, Lm, H] context: device tensors cannot be sliced non-contiguously; the row gather stays on
        # the device (one index_select)
        ridx = (torch.arange(B).view(B, 1) * SEG + torch.arange(Lm).view(1, Lm)).reshape(-1).to(dev)
        ctx = torch.index_select(h, 0, ridx).view(B, Lm, Hd)
        cm = torch.zeros(B, Lm, dtype=torch.bfloat16)
        for b, (_, _, m) in enumerate(emb):
            cm[b, :m.shape[0]] = m.to(torch.bfloat16)
        nz = []
        for b in range(B):
            if noises[b] is not None:
                nz.append(torch.as_tensor(np.asarray(noises[b]), dtype=torch.float32).view(chunk, -1))
            else:
                g = torch.Generator().manual_seed(int(seeds[b])) if seeds[b] is not None else None
                nz.append(torch.randn(chunk, self.expert.action_dim, generator=g))
        noise = torch.stack(nz, 0)
        a = self._dit_fn(Lm, chunk, adim, B)(ctx, cm.to(dev), noise.to(dev)).cpu().float()
        td = _t.time() - t1
        outs = []
        for b in range(NR):
            norm = a[b, :, :adim]
            if os.environ.get("ISAAC05_RETURN_NORMALIZED", "0") == "1":
                out = norm.numpy()
            else:
                out = np.asarray(self._unnormalize(norm), dtype=np.float32)
            outs.append(DiffusionOutput(output={"actions": out}))
        self._last_timing = {"B": B, "n_req": NR, "host_embed": round(th * 1e3, 1), "device": round(td * 1e3, 1),
                             **hp["host"]}
        logger.info("IsaacActionPipeline: batch B=%d Lmax=%d timing_ms %s", B, Lm, self._last_timing)
        return outs

    def _forward_one(self, req):
        from vllm_omni.diffusion.data import DiffusionOutput

        extra = getattr(req.sampling_params, "extra_args", None) or {}
        obs = extra.get("robot_obs")
        chunk, adim = self.pcfg.chunk_size, self.pcfg.action_dim
        if obs is None:  # engine warmup / dummy request
            return DiffusionOutput(output={"actions": np.zeros((chunk, adim), dtype=np.float32)})
        import time as _t

        tim = {}
        t0 = _t.time()
        t_enter = t0
        obs = dict(obs)
        noise = obs.pop("noise", None)  # optional client-supplied initial noise [chunk, 64] (accuracy gates)
        grp = self._neuron_tp
        if grp.world_size > 1 and self._vision_dp:
            # rank 0 renders + embeds (host); the device vision tower runs data-parallel: rank 0 broadcasts the
            # patches, every rank encodes its slice of images, rank 0 gathers the features. Other ranks loop on the
            # broadcasts until the final ("payload", embeds) message.
            if grp.rank_in_group == 0:
                e, pos, cmask = self._embed(obs)
                torch.distributed.broadcast_object_list([("payload", (e, pos, cmask))], src=grp.ranks[0],
                                                        group=self._cpu_pg())
            else:
                while True:
                    msg = [None]
                    torch.distributed.broadcast_object_list(msg, src=grp.ranks[0], group=self._cpu_pg())
                    kind, data = msg[0]
                    if kind == "payload":
                        e, pos, cmask = data
                        break
                    self._vision_slice(*self._recv_patches(data))
        elif grp.world_size > 1 and os.environ.get("ISAAC05_RENDER_RANK0", "1") == "1":
            # Render + embed + vision once (group rank 0) and broadcast: every rank receives the same request, and
            # four identical host renders on one host contend for the CPUs (3-4x slower each) and skew the ranks.
            payload = [self._embed(obs) if grp.rank_in_group == 0 else None]
            torch.distributed.broadcast_object_list(payload, src=grp.ranks[0], group=self._cpu_pg())
            e, pos, cmask = payload[0]
        else:
            e, pos, cmask = self._embed(obs)
        tim["host_embed"] = _t.time() - t0
        L = e.shape[0]
        T = next(b for b in PREFILL_BUCKETS if b >= L)
        ep = torch.zeros(T, e.shape[1], dtype=torch.bfloat16)
        ep[:L] = e
        pp = pos[:, -1:].repeat(1, T)
        pp[:, :L] = pos
        dev = self._device
        t0 = _t.time()
        h = self._vlm_fn(T, L)(ep.to(dev), pp.to(torch.int32).to(dev))
        if os.environ.get("ISAAC05_TIMING", "0") == "1":
            h.cpu()
            tim["vlm"] = _t.time() - t0
        ctx = h[:L].unsqueeze(0)
        seed = getattr(req.sampling_params, "seed", None)
        g = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        noise = torch.as_tensor(np.asarray(noise), dtype=torch.float32).view(1, chunk, -1) if noise is not None else \
            torch.randn(1, chunk, self.expert.action_dim, generator=g)
        a = self._dit_fn(L, chunk, adim)(ctx, cmask.to(torch.bfloat16).view(1, L).to(dev), noise.to(dev)).cpu().float()
        tim["device"] = _t.time() - t0
        norm = a[0, :, :adim]
        if os.environ.get("ISAAC05_RETURN_NORMALIZED", "0") == "1":
            out = norm.numpy()
        else:
            out = np.asarray(self._unnormalize(norm), dtype=np.float32)
        tim["render"] = getattr(self, "_tim_render", 0.0)
        tim["vision"] = getattr(self, "_tim_vision", 0.0)
        self._last_timing = {k: round(v * 1e3, 1) for k, v in tim.items()}
        self._last_timing["t_enter"] = round(t_enter, 4)
        self._last_timing["t_exit"] = round(_t.time(), 4)
        logger.info("IsaacActionPipeline: L=%d bucket=%d timing_ms %s", L, T, self._last_timing)
        return DiffusionOutput(output={"actions": out})

    def _unnormalize(self, norm):
        post = getattr(self.policy, "postprocessor", None)
        if post is None:
            return norm.numpy()
        try:
            return post(norm.unsqueeze(0))[0].detach().cpu().numpy()
        except Exception:  # noqa: BLE001 - postprocessors differ across LeRobot versions; fall back to normalized
            logger.warning("IsaacActionPipeline: postprocessor failed; returning normalized actions", exc_info=True)
            return norm.numpy()
