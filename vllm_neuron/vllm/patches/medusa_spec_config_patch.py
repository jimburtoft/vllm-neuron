# SPDX-License-Identifier: Apache-2.0
"""
Accept ``method="medusa"`` in ``vllm serve --speculative-config`` WITHOUT a
separate draft model.

Why this patch exists
---------------------
For most speculative methods vLLM-core's ``SpeculativeConfig`` expects the
``model`` field to point at a *separate* draft-model repo (EAGLE head repo, a
Medusa-typed ``config.json`` with ``model_type: "medusa"``, an MTP module, ...).
It then builds a second ``ModelConfig`` (``draft_model_config``) for that repo
and loads it alongside the target.

The native-backend Whisper Medusa framework (Task 020) is different: the Medusa
heads live *inside the target model* (constructed in
``WhisperForConditionalGeneration.__init__`` and loaded in ``load_weights``;
they read the decoder's last hidden state in-graph, and compile INTO the target
verify-K NEFF). There is **no** separate draft model and **no** draft KV cache.

Without this patch, ``SpeculativeConfig.__post_init__`` fails one of two ways
when the customer runs ``--speculative-config '{"method":"medusa",
"num_speculative_tokens":5}'`` (no ``model`` field):

  1. ``model is None`` -> no ``elif`` matches ``medusa`` in the model-inference
     block -> ``raise ValueError("num_speculative_tokens was provided but
     without speculative model.")``.
  2. If we naively set ``model = <target>`` it would fall into the generic
     ``else`` branch, build a draft ``ModelConfig`` for the *whisper* target,
     detect ``model_type == "whisper"`` (not ``"medusa"``), and
     ``raise NotImplementedError("Unsupported speculative method: 'medusa'")``.

The fix
-------
Patch ``SpeculativeConfig.__post_init__`` to special-case ``method == "medusa"``
and give it the exact treatment ngram/mtp already get for "the draft config IS
the target config": set ``model``/``draft_model_config``/``draft_parallel_config``
to the target's, zero the ngram lookup fields, and skip the separate-draft-model
construction path entirely. The rest of vLLM-core is already medusa-safe:
``use_eagle()`` returns False for medusa (so the runner's draft-KV alloc/bind
auto-skips), ``_verify_args``'s aux-hidden-state model allow-list only gates
eagle3/extract/dflash, and ``verify_equal_vocab_size_if_draft_model`` trivially
passes because the draft config *is* the target. (Patching ``__post_init__`` on
the pydantic dataclass works: pydantic invokes it via the validator on every
construction.)

The circular-import timing problem (and how this avoids it)
-----------------------------------------------------------
``vllm_neuron`` is imported by vLLM's plugin discovery *while ``import vllm`` is
still running*, so at plugin-import time neither ``vllm.config.speculative`` nor
``vllm.engine.arg_utils`` is fully initialized -- a direct
``from vllm.config.speculative import SpeculativeConfig`` raises a circular
``ImportError``. To sidestep this we do NOT import by name. Instead
``apply_medusa_spec_config_patch`` is idempotent and is called from BOTH plugin
import time and the platform's ``register()``; each call tries to fetch the
ALREADY-LOADED ``vllm.config.speculative`` module from ``sys.modules`` (no
re-import, no circular trigger) and patch the class there. By the time an
engine is actually built and ``SpeculativeConfig(...)`` runs, the module is in
``sys.modules`` and the patch is in place.

Config surface
--------------
K (number of Medusa lookahead heads, default 5) comes straight from
``--speculative-config`` ``num_speculative_tokens`` -- no separate config repo.
The heads SOURCE (``path`` | ``zero`` | ``random``) is supplied via
``--additional-config '{"medusa_config": {...}}'`` (read by the Whisper model's
``from_configs``), NOT via ``--speculative-config``, because the heads are part
of the target model, not a draft model.
"""

import sys

from vllm.logger import init_logger

logger = init_logger(__name__)

_applied = False
_original_post_init = None
_hook_installed = False


def _medusa_aware_post_init(self):
    """Route ``method=="medusa"`` around the separate-draft-model machinery."""
    if getattr(self, "method", None) == "medusa":
        # Mirror the ngram / mtp "draft config == target config" treatment.
        # The Medusa heads live in the target; there is no separate draft repo.
        if self.target_model_config is None:
            raise ValueError(
                "target_model_config must be present for method='medusa' "
                "(the Medusa heads live inside the target model)."
            )
        if self.num_speculative_tokens is None:
            raise ValueError(
                "method='medusa' requires num_speculative_tokens (K, the number "
                "of Medusa lookahead heads; default 5). Pass it in "
                "--speculative-config, e.g. "
                "'{\"method\":\"medusa\",\"num_speculative_tokens\":5}'."
            )
        if self.num_speculative_tokens <= 0:
            raise ValueError(
                "num_speculative_tokens must be > 0 for method='medusa'; got "
                f"{self.num_speculative_tokens}."
            )

        # Point every "draft" field at the target itself -- no second model is
        # loaded. This is exactly what ngram does in the stock __post_init__
        # (self.draft_model_config = self.target_model_config).
        self.model = self.target_model_config.model
        self.draft_model_config = self.target_model_config
        self.draft_parallel_config = self.target_parallel_config
        # ngram-style: no prompt-lookup window applies to medusa.
        self.prompt_lookup_max = 0
        self.prompt_lookup_min = 0

        logger.info(
            "Medusa spec-config shim: method='medusa', "
            "num_speculative_tokens(K)=%d, draft config = target (heads live in "
            "target model, no separate draft model loaded).",
            self.num_speculative_tokens,
        )
        return self

    return _original_post_init(self)


def apply_medusa_spec_config_patch():
    """Patch SpeculativeConfig.__post_init__ to accept method='medusa'.

    Idempotent + circular-import-safe. Uses the ALREADY-LOADED
    ``vllm.config.speculative`` module from ``sys.modules`` (never triggers a
    fresh import), so it can be called from plugin-import time (when the module
    may not yet be present -> installs a deferred hook) and from register()
    (when it is present -> applied). Returns True once the patch is in place.

    If the module is not loaded yet at call time, installs a one-shot
    ``sys.meta_path`` finder that applies the patch the instant
    ``vllm.config.speculative`` is imported -- which happens (in every process
    that builds a SpeculativeConfig, including the API-server frontend) as a
    transitive import of ``vllm.engine.arg_utils`` BEFORE
    ``create_speculative_config`` runs. This is what makes ``vllm serve
    --speculative-config method=medusa`` work even though the neuron platform's
    ``register()`` is not called in the frontend process.
    """
    global _applied

    if _applied:
        return True

    if _try_patch_now():
        return True

    _install_deferred_hook()
    return False


def _try_patch_now():
    """Patch if vllm.config.speculative is already loaded; else False."""
    global _applied, _original_post_init

    if _applied:
        return True

    mod = sys.modules.get("vllm.config.speculative")
    if mod is None:
        return False

    spec_cfg_cls = getattr(mod, "SpeculativeConfig", None)
    if spec_cfg_cls is None:
        return False

    # Guard against double-patching (e.g. hook + explicit call).
    if getattr(spec_cfg_cls.__post_init__, "_is_medusa_shim", False):
        _applied = True
        return True

    _original_post_init = spec_cfg_cls.__post_init__
    _medusa_aware_post_init._is_medusa_shim = True
    spec_cfg_cls.__post_init__ = _medusa_aware_post_init
    _applied = True
    logger.info(
        "Medusa spec-config shim applied "
        "(SpeculativeConfig.__post_init__ patched; method='medusa' accepted)"
    )
    return True


def _install_deferred_hook():
    """Install a one-shot meta_path finder that patches on first import of
    vllm.config.speculative. Harmless + removes itself once fired."""
    global _hook_installed
    if _hook_installed:
        return

    class _MedusaSpecImportHook:
        """A no-op finder whose find_spec side-effect is to try the patch.

        Python calls find_spec on every meta_path entry for every import; we
        watch for vllm.config.speculative appearing in sys.modules and patch it,
        then uninstall ourselves. Returning None lets the real finders handle
        the import.
        """

        def find_spec(self, fullname, path=None, target=None):
            if _applied:
                _uninstall_hook()
                return None
            # After (or during the tail of) the target module's import, the
            # module object is registered in sys.modules; try to patch it.
            if fullname == "vllm.config.speculative" or (
                "vllm.config.speculative" in sys.modules
            ):
                if _try_patch_now():
                    _uninstall_hook()
            return None

    hook = _MedusaSpecImportHook()
    sys.meta_path.insert(0, hook)
    _hook_installed = True
    logger.debug(
        "Medusa spec-config shim: deferred import hook installed "
        "(vllm.config.speculative not loaded yet)."
    )


def _uninstall_hook():
    global _hook_installed
    if not _hook_installed:
        return
    sys.meta_path[:] = [
        h for h in sys.meta_path if type(h).__name__ != "_MedusaSpecImportHook"
    ]
    _hook_installed = False
