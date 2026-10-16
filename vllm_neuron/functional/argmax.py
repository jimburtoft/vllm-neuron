# SPDX-License-Identifier: Apache-2.0
"""
Distributed argmax kernel for tensor-parallel inference.

This module provides a distributed argmax operation that works across
sharded tensors in a tensor-parallel setting.
"""

import logging
import math
from typing import Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed._functional_collectives import all_gather_tensor
import nki
from nkilib.core.max.cascaded_max import cascaded_max
from vllm_neuron import envs
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

cascaded_max_jit = nki.jit()(cascaded_max)

logger = logging.getLogger(__name__)


def argmax(
    tensor: Tensor,
    dim: int,
    gather_dim: int,
    keepdim: bool = False,
    process_group: Optional[ProcessGroup] = None,
) -> Tensor:
    """
    Performs distributed argmax on sharded tensors using 2-step algorithm.

    This function implements a distributed argmax operation for tensor-parallel
    inference where tensors are sharded across multiple devices. It uses a
    2-step approach:

    1. **Local argmax**: Each rank computes argmax on its local shard using
       NKI cascaded_max kernel (when conditions are met) or torch.max fallback
    2. **Global argmax**: Results are gathered and global argmax is computed

    **Sharding Layout**:
    The input tensor is assumed to be uniformly sharded along `gather_dim` across
    all ranks in the process group. For example, with TP=2 and vocab_size=256:
    - Rank 0: logits[:, 0:128]   (first 128 vocab tokens)
    - Rank 1: logits[:, 128:256] (last 128 vocab tokens)

    **When to Use**:

    This function is intended for distributed argmax. It will fall back to torch.argmax when process_group is not provided.

    **Kernel**:
    In step 1 we compute the local maxs using either a nki kernel or torch.max.
    NKI cascaded_max kernel is used when: dim=-1, 2D/3D tensor, size>=128, and
    not in CPU mode. Otherwise falls back to torch.max.

    Args:
        tensor: Input tensor to perform argmax on. Must be uniformly sharded
            along `gather_dim` across all ranks in `process_group`.
        dim: Dimension along which to find argmax.
        gather_dim: Dimension the tensor is sharded on. When `dim == gather_dim`,
            indices are corrected for sharding offset.
        keepdim: Whether to keep the reduced dimension. Defaults to False.
        process_group: Process group for distributed operations.
            Must be provided for distributed execution.

    Returns:
        Tensor with global argmax indices across all shards. All ranks return
        the same result.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from vllm_neuron.functional.argmax import argmax
        >>>
        >>> # Tensor-parallel inference with TP=2, vocab_size=256
        >>> # Each rank has logits of shape (batch=1, vocab_per_rank=128)
        >>> logits = torch.randn(1, 128)  # Local shard on this rank
        >>> pg = dist.new_group([0, 1])
        >>>
        >>> # Compute global argmax across both ranks
        >>> token_id = argmax(logits, dim=1, gather_dim=1, process_group=pg)
        >>> # Returns shape (1,) with global token index in range [0, 256)
        >>> # Rank 0 indices are in [0, 128), Rank 1 indices are in [128, 256)
    """
    if process_group is None:
        raise ValueError("process_group must be provided for distributed argmax")

    tp_degree = torch.distributed.get_world_size(group=process_group)

    # Fast path for single device
    if tp_degree == 1:
        return torch.argmax(tensor, dim=dim, keepdim=keepdim)

    # TODO: Remove this upcast - NKI kernel and functional tests work with bfloat16, but E2E models show accuracy regression without it
    tensor = tensor.to(torch.float32)
    # Find local max values and indices
    local_value, local_index = _compute_local_max(tensor, dim)

    # Gather results from all ranks
    global_values = all_gather_tensor(local_value, gather_dim, group=process_group)
    global_indices = all_gather_tensor(local_index, gather_dim, group=process_group)

    # Correct indices for sharding offset when gather_dim == dim
    if gather_dim == dim:
        global_indices = _apply_sharding_offset(
            global_indices, dim, tensor.shape[gather_dim], tp_degree
        )

    # Find global argmax and extract final indices
    global_argmax = torch.argmax(global_values, dim=dim, keepdim=True)
    final_indices = torch.gather(global_indices, dim, global_argmax)

    if not keepdim:
        return final_indices.squeeze(dim)
    return final_indices


def _compute_local_max(tensor: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """Compute local max using NKI kernel when possible, otherwise torch.max."""
    if _can_use_nki_max(tensor, dim):
        is_3d = len(tensor.shape) == 3
        input_tensor = tensor.squeeze(0) if is_3d else tensor

        cascaded_max_nki = wrap_nki(cascaded_max_jit)
        value, index = cascaded_max_nki[2](input_tensor)
        # NKI kernel returns uint32 indices; gloo all_gather doesn't support
        # uint32, so cast to int64 (matching torch.max) in CPU mode.
        if envs.VLLM_NEURON_CPU_MODE:
            index = index.to(torch.int64)
        # Restore dimension if squeezed
        if is_3d:
            value = value.unsqueeze(0)
            index = index.unsqueeze(0)
        return value, index
    else:
        return torch.max(tensor, dim=dim, keepdim=True)


def _can_use_nki_max(tensor: Tensor, dim: int) -> bool:
    """
    Check if we can use the NKI max kernel.

    Requirements:
    - NKI kernels can run (hardware or CPU simulator)
    - dim must be the last dimension (NKI kernel only supports this)
    - Tensor must be 2D or 3D with shape[0] == 1
    - Reduction dimension must have at least 128 elements
    """
    if not can_run_kernel(tensor):
        return False

    shape = tensor.shape
    num_dims = len(shape)

    # NKI kernel only works on last dimension
    if dim != num_dims - 1 and dim != -1:
        return False

    # Check tensor dimensionality
    if not (num_dims == 2 or (num_dims == 3 and shape[0] == 1)):
        return False

    # Check minimum reduction size
    if shape[dim] < 128:
        return False

    # The nkilib cascaded_max kernel splits each row across up to 128 partitions
    # (rows are split across the 2 cores at LNC=2), and its nc_find_index8 step
    # takes at most 16384 elements per partition. Large batches over a large
    # per-rank vocabulary (for example batch >= 96 for GPT-OSS at TP=4, 50272
    # per rank) exceed that and fail to compile, so use torch.max for them.
    rows = math.prod(shape[:-1])
    rows_per_core = math.ceil(rows / 2) if rows > 1 else 1
    stages = min(128 // rows_per_core, shape[dim] // 128)
    return stages >= 1 and math.ceil(shape[dim] / stages) <= 16384


def _apply_sharding_offset(
    indices: Tensor, dim: int, shard_size: int, tp_degree: int
) -> Tensor:
    """Apply offset to indices to account for tensor sharding."""
    offset_shape = [1] * len(indices.shape)
    offset_shape[dim] = tp_degree

    offset = torch.arange(0, shard_size * tp_degree, shard_size, device=indices.device)
    offset = offset.view(offset_shape)

    return indices + offset
