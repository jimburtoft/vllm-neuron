# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Qwen3-ASR encoder warmup inputs.

Warmup compiles the audio tower with synthetic inputs. Serving calls the tower
with the inputs that embed_multimodal builds. A compiled graph is reused only
when the shapes and dtypes are the same, so the tests compare the two.

Run with:
    pytest test/vllm_neuron/model/qwen3_asr/bf16/modules/test_encoder_warmup.py -v
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.interfaces import SupportsEncoderWarmup
from vllm_neuron.model.qwen3_asr.config import Qwen3ASRAudioConfig
from vllm_neuron.model.qwen3_asr.model_bf16 import Qwen3ASRForConditionalGeneration
from vllm_neuron.model.qwen3_asr.utils.audio_preprocessing import plan_audio_blocks

pytestmark = [pytest.mark.fast]

_BLOCK_SIZE = 128


def _synthetic_inputs(bucket: int, dp_size: int) -> dict[str, torch.Tensor]:
    model = SimpleNamespace(config=SimpleNamespace(audio_config=Qwen3ASRAudioConfig()))
    encoder_neuron_config = SimpleNamespace(
        encoder_attention_block_size=_BLOCK_SIZE, dp_size=dp_size
    )
    return Qwen3ASRForConditionalGeneration.build_encoder_synthetic_inputs(
        model, bucket, encoder_neuron_config, torch.device("cpu")
    )


def _serving_inputs(
    feature_lens: list[int], bucket: int, dp_size: int
) -> dict[str, torch.Tensor]:
    """Build the audio tower inputs the way embed_multimodal does."""
    audio_config = Qwen3ASRAudioConfig()
    plan = plan_audio_blocks(
        feature_lens,
        block_size=_BLOCK_SIZE,
        window_tokens=audio_config.window_tokens,
        n_window=audio_config.n_window,
        n_window_infer=audio_config.n_window_infer,
        buckets=[bucket],
        one_item_per_block=True,
        dp_size=dp_size,
    )
    bound_min, bound_max = plan.block_bounds()
    num_frames = plan.num_slots * 2 * audio_config.n_window
    return {
        "input_features": torch.zeros(
            1, audio_config.num_mel_bins, num_frames, dtype=audio_config.torch_dtype
        ),
        "bound_min": bound_min,
        "bound_max": bound_max,
        "chunk_slots": plan.chunk_slots(),
        "write_block_ids": torch.zeros(plan.num_blocks, dtype=torch.int64),
    }


def test_model_implements_encoder_warmup():
    assert issubclass(Qwen3ASRForConditionalGeneration, SupportsEncoderWarmup)


@pytest.mark.parametrize("bucket", [4, 8])
@pytest.mark.parametrize("dp_size", [1, 2, 3])
@pytest.mark.parametrize("feature_lens", [[1000], [3000], [400, 1500]])
def test_synthetic_inputs_match_serving(bucket, dp_size, feature_lens):
    """Each input has the shape and dtype that the serving call gives."""
    synthetic = _synthetic_inputs(bucket, dp_size)
    serving = _serving_inputs(feature_lens, bucket, dp_size)

    assert synthetic.keys() == serving.keys()
    for name, tensor in serving.items():
        assert synthetic[name].shape == tensor.shape, name
        assert synthetic[name].dtype == tensor.dtype, name


@pytest.mark.parametrize("bucket, dp_size, num_blocks", [(4, 1, 4), (4, 3, 6)])
def test_write_block_ids_count_blocks(bucket, dp_size, num_blocks):
    """An audio bucket is a block count, padded to a multiple of dp_size."""
    synthetic = _synthetic_inputs(bucket, dp_size)

    assert synthetic["write_block_ids"].shape == (num_blocks,)
    assert synthetic["bound_min"].shape == (num_blocks, _BLOCK_SIZE, 1)
