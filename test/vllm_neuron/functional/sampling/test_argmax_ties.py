# SPDX-License-Identifier: Apache-2.0
"""Distributed / NKI argmax must break exact ties like torch.argmax (lowest index), independent of the
number of rows, so greedy decoding (1 row) and speculative verification (1 + K rows) agree."""

import pytest
import torch

from vllm_neuron.functional.argmax import _lowest_index_of

pytestmark = [pytest.mark.fast]


@pytest.mark.parametrize("rows", [1, 2, 16, 64])
def test_lowest_index_of_matches_torch_argmax_on_ties(rows):
    torch.manual_seed(0)
    x = torch.randn(rows, 37984).to(torch.bfloat16).to(torch.float32)
    # Force exact ties at the max: copy each row's max to two later columns.
    m, i = x.max(-1, keepdim=True)
    x[:, -1] = m[:, 0]
    x[torch.arange(rows), (i[:, 0] + 7) % x.shape[1]] = m[:, 0]
    got = _lowest_index_of(x, m, -1)
    assert torch.equal(got, torch.argmax(x, -1, keepdim=True))


def test_lowest_index_of_3d():
    x = torch.zeros(1, 4, 256)
    x[0, :, 10] = 1.0
    x[0, :, 3] = 1.0
    got = _lowest_index_of(x, x.amax(-1, keepdim=True), -1)
    assert got.shape == (1, 4, 1) and bool((got == 3).all())
