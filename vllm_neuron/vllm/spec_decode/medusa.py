# SPDX-License-Identifier: Apache-2.0
"""
MedusaProposer -- the thin spec-decode drafter for native-backend Whisper.

Task 020 Milestone 2. Mirrors ``EagleProposer``'s RUNNER CONTRACT (so the
method-agnostic accept/reject machinery -- ``RejectionSampler``,
``_parse_rejection_sampling_output``, ``SpecDecodeMetadata`` -- is reused
UNCHANGED) but is MUCH thinner (M0 spec Task 020 §1c):

  * NO separate draft model -- the Medusa heads live INSIDE the target model
    (constructed in ``WhisperForConditionalGeneration.__init__``, loaded in
    ``load_weights``; Task 020 M1). ``load_model`` is therefore a near no-op.
  * NO draft KV cache -- Medusa heads are stateless MLPs on the target
    decoder's LAST hidden state. The runner's draft-KV alloc/bind
    (``neuron_model_runner.py`` gated on ``speculative_config.use_eagle()``)
    auto-skips because ``use_eagle()`` is False for ``method=="medusa"``.
  * NO shifted input_ids, NO aux_hidden_states, NO async-correction kwargs --
    the proposer reads the target decoder's LAST hidden state at the anchor
    position and runs N independent heads -> N greedy draft token ids.

Contract (M0 spec §1c): ``propose(...)`` returns ``drafts_only [bs, K]`` int32
(BS=1 -> ``[1, K]``), exactly the tensor ``EagleProposer.propose`` returns in
sync mode (``neuron_model_runner.py`` returns ``drafts_only.cpu()``). The
scheduler then places those drafts into the next step's ``input_ids`` and the
``SpecDecodeMetadata`` builder re-reads them, exactly like EAGLE.

The proposal itself is a thin gather+argmax over head logits: the target NEFF
computes the last hidden state; ``propose`` calls the target model's
``medusa_propose(last_hidden)`` (Task 020 M1) which runs the heads + the tied
sharded lm_head + argmax and returns ``[K]`` int32 draft ids. Because the heads
are children of the target model they compile INTO the target NEFF -- no
separate draft NEFF, no cross-NEFF handshake.
"""

import logging

import torch
from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


class MedusaProposer:
    """Thin Medusa drafter. See module docstring for the runner contract."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        on_device_sampling: bool = True,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None

        self.method = self.speculative_config.method
        assert self.method == "medusa", (
            f"MedusaProposer requires method=='medusa', got {self.method!r}"
        )

        self.device = device
        self.on_device_sampling = on_device_sampling
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        # The Medusa heads live in the TARGET model; the runner sets this
        # reference in ``load_model`` (below). No separate draft model.
        self.model = None
        # EagleProposer exposes ``attn_layer_names`` for draft-KV binding;
        # Medusa has no draft KV, so this stays empty and the runner's
        # ``use_eagle()``-gated draft-KV alloc auto-skips.
        self.attn_layer_names: list[str] = []
        # EagleProposer exposes a capture_backend_model for graph extraction;
        # the Medusa heads compile into the target NEFF, so there is no
        # separate draft graph to capture.
        self.capture_backend_model = None

    def load_model(self, target_model=None, **kwargs) -> None:
        """No-op (heads live in the target model, loaded in M1).

        The runner calls ``drafter.load_model(...)`` after loading the target
        model. For EAGLE this compiles + loads the separate draft model; for
        Medusa the heads are already constructed + loaded inside the target
        model (Task 020 M1), so we only stash a reference to the target model
        so ``propose`` can call its ``medusa_propose``.
        """
        if target_model is not None:
            self.model = target_model
        logger.info(
            "MedusaProposer.load_model: no separate draft model "
            "(heads live in the target; num_speculative_tokens=%d).",
            self.num_speculative_tokens,
        )

    def set_target_model(self, target_model) -> None:
        """Bind the target model (whose ``medusa_propose`` the proposer calls).

        Called by the runner once ``self.model`` is loaded, since Medusa has no
        separate draft model to load.
        """
        self.model = target_model

    def warmup(self, num_tokens=None, num_reqs=None, attn_metadata=None, **kwargs) -> None:
        """No-op: the Medusa heads compile INTO the target NEFF, so they are
        warmed by the target's decode/verify-bucket warmup (the runner warms
        ``decode_b{num_reqs}_s{tokens_per_req}`` when a drafter is present).
        There is no separate draft NEFF to warm."""
        logger.debug("MedusaProposer.warmup: no-op (heads warm with the target NEFF).")

    def graph_extract(self, num_tokens=None, num_reqs=None, attn_metadata=None, **kwargs) -> None:
        """No-op: no separate draft graph to capture (heads compile into the
        target NEFF)."""
        logger.debug("MedusaProposer.graph_extract: no-op (no separate draft graph).")

    def propose(
        self,
        last_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        raw_sampled_token_ids: torch.Tensor | None = None,
        is_warmup: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Return the K greedy draft token ids per request.

        Args (M0 spec §1c):
            last_hidden_states: ``[num_tokens, d_model]`` the target decoder's
                LAST hidden state (post final LayerNorm) -- the SAME tensor the
                lm_head consumes. Medusa reads the anchor row(s).
            last_token_indices: ``[bs]`` the row index (into
                ``last_hidden_states``) of the anchor position to draft from.
            raw_sampled_token_ids: the target's just-sampled bonus token
                (unused by the Medusa heads -- they condition only on the
                hidden state; accepted for signature parity with EAGLE).

        Returns:
            drafts_only: ``[bs, K]`` int32. BS=1 -> ``[1, K]``.
        """
        assert self.model is not None, (
            "MedusaProposer.propose called before the target model was bound "
            "(runner must call set_target_model / load_model first)."
        )
        num_reqs = last_token_indices.shape[0]
        K = self.num_speculative_tokens

        if num_reqs == 0:
            return torch.empty((0, K), dtype=torch.int32, device=last_hidden_states.device)

        # Gather the anchor hidden state per request. BS=1 is the target
        # regime (Task 020 non-goal: BS>1), but the gather generalizes.
        anchor_hidden = torch.index_select(
            last_hidden_states, dim=0, index=last_token_indices.to(torch.long)
        )  # [bs, d_model]

        # Run the heads + tied sharded lm_head + argmax -> [K] per request.
        # medusa_propose handles a single anchor row; loop over the (BS=1)
        # requests to build [bs, K]. In-graph read of the hidden state (heads
        # are target children) -> no separate NEFF.
        drafts_rows = []
        for r in range(num_reqs):
            row = anchor_hidden[r : r + 1]  # [1, d_model]
            drafts_r = self.model.medusa_propose(row)  # [K] int32
            drafts_rows.append(drafts_r.view(1, K))
        drafts_only = torch.cat(drafts_rows, dim=0).to(torch.int32)  # [bs, K]
        return drafts_only
