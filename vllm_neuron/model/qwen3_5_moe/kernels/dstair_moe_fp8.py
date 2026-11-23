# Adapted from dstair/qwen3.5-moe-fp8-trainium2 (Apache-2.0, see NOTICE.dstair): experts LNC-sharded;
# output is [n_prgs*B, H] per-program partials (caller sums).
"""Fused all-local-expert FP8 MoE decode kernels for trn2.

The router and shared expert stay in the surrounding PyTorch graph. These
kernels fuse the local routed experts:

    gate/up GEMM -> SwiGLU -> down GEMM -> affinity scale -> expert sum

without materializing `[experts, batch, hidden]`. For each 128-token tile they
keep one expert's gate/up/activated intermediate and the routed accumulator in
SBUF.

Weights are trn2 legacy-E4M3 bytes stored as `nl.int8` and viewed as
`nl.float8_e4m3` in SBUF, so the HLO operand stays int8 (neuronx-cc rejects
`torch.float8_e4m3fn` on trn2). TensorE consumes them natively as the moving
operand. `moe_w8.py` prepares them from the official E4M3FN checkpoint.

Entry points:

* `nki_moe_fused_w8_fp8_block_coalesced` -- one scale per 128x128 block. Pair
  with `moe_w8` ``fp8_impl="block_pow2_coalesced"`` (default, lossless).
* `nki_moe_fused_w8_fp8_block_coalesced_ob` -- one scale per 128 output rows,
  PSUM-accumulated and scaled once. Faster at B=128 only; pair with
  ``fp8_impl="block_ob_coalesced"``.

Shapes: hidden [B, H] bf16 with B in {32, 64, 128}; gate_up [E, H, 2, I] and
down [E, I, H] int8 with I == 512 and H % 512 == 0; coalesced scale tables
[E, 128, N] bf16; affinities [B, E] f32. Output [B, H] f32 (pre all-reduce).
"""

import nki
import nki.isa as nisa
import nki.language as nl


TILE = 128
COALESCED_WIDTH = 512


def _load_coalesced_scale_table(source, expert):
    """Load and convert one expert's repeated BF16 scale table once."""
    table_size = source.shape[2]
    stored = nl.ndarray(
        (TILE, table_size), dtype=source.dtype, buffer=nl.sbuf
    )
    nisa.dma_copy(dst=stored, src=source[expert, :, :])
    converted = nl.ndarray(
        (TILE, table_size), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.activation(
        dst=converted, op=nl.copy, data=stored, scale=1.0
    )
    return converted


def _moe_fused_w8_block_coalesced(
    hidden,
    gate_up,
    down,
    gate_up_scales,
    down_scales,
    affinities,
    coarse_scale=False,
):
    """Fused native-E4M3 MoE over every local expert, with 128x128 block scales.

    The ring-buffer and TensorE column-packing schedule is adapted from
    nki-library commit 7a5b6f9 (`moe_tkg/mlp_tkg_*_projection.py`). Weight
    slabs are wider than scale blocks, so each 128-column PSUM partial is scaled
    in SBUF before contraction-block accumulation.

    ``coarse_scale`` (the `_ob` entry point) applies only at tokens == 128
    (column_packing == 1). It requires the scale table to be independent of the
    input block -- one scale per 128 output rows, repeated across input tiles,
    as produced by `moe_w8.requantize_official_fp8_output_block` -- and then
    accumulates the whole contraction in PSUM and post-scales each output block
    once, instead of one Vector scale-add per 128x128 block. It is ONLY correct
    on such weights; `moe_w8.assert_output_block_scale_grid` enforces that at
    load. At tokens 32 / 64 both entry points run the same per-block path.
    """
    tokens, hidden_size = hidden.shape
    experts, hidden_size_w, projections, intermediate = gate_up.shape
    experts_d, intermediate_d, hidden_size_d = down.shape
    assert tokens in (32, 64, 128)
    assert projections == 2
    assert hidden_size == hidden_size_w == hidden_size_d
    assert experts == experts_d
    assert intermediate == intermediate_d
    assert hidden_size % COALESCED_WIDTH == 0
    assert intermediate == COALESCED_WIDTH
    assert affinities.shape == (tokens, experts)

    hidden_tiles = hidden_size // TILE
    intermediate_tiles = intermediate // TILE
    assert gate_up_scales.shape == (
        experts,
        TILE,
        2 * hidden_tiles * intermediate_tiles,
    )
    assert down_scales.shape == (
        experts,
        TILE,
        intermediate_tiles * hidden_tiles,
    )

    if tokens == 32:
        column_packing = 4
    elif tokens == 64:
        column_packing = 2
    else:
        column_packing = 1
    assert hidden_tiles % column_packing == 0
    assert intermediate_tiles % column_packing == 0

    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0) if n_prgs > 1 else 0
    assert experts % n_prgs == 0
    experts_local = experts // n_prgs
    output = nl.ndarray(
        (n_prgs * tokens, hidden_size), dtype=nl.float32, buffer=nl.shared_hbm
    )

    hidden_sb = nl.ndarray(
        (TILE, hidden_tiles, tokens),
        dtype=hidden.dtype,
        buffer=nl.sbuf,
    )
    for hidden_tile in nl.affine_range(hidden_tiles):
        hidden_start = hidden_tile * TILE
        hidden_row = nl.ndarray(
            (TILE, TILE), dtype=hidden.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=hidden_row[:tokens, :],
            src=hidden[:, hidden_start : hidden_start + TILE],
        )
        hidden_transposed_psum = nl.ndarray(
            (TILE, tokens), dtype=hidden.dtype, buffer=nl.psum
        )
        nisa.nc_transpose(
            dst=hidden_transposed_psum,
            data=hidden_row[:tokens, :],
        )
        nisa.tensor_copy(
            dst=hidden_sb[:, hidden_tile, :],
            src=hidden_transposed_psum,
        )

    routed = nl.ndarray(
        (tokens, hidden_size), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.memset(dst=routed, value=0.0)
    expert_output = nl.ndarray(
        (tokens, hidden_size), dtype=nl.float32, buffer=nl.sbuf
    )

    # Two rotating slots break load/use anti-dependencies while keeping the
    # expert scratch bounded. Gate and up need independent PSUM streams.
    weight_slots = [
        nl.ndarray(
            (TILE, 2, COALESCED_WIDTH),
            dtype=nl.int8,
            buffer=nl.sbuf,
        ),
        nl.ndarray(
            (TILE, 2, COALESCED_WIDTH),
            dtype=nl.int8,
            buffer=nl.sbuf,
        ),
    ]
    # PSUMs are allocated FRESH per input_group (in the loops below) instead of
    # reused rotating slots: a matmul into a fresh nl.psum ndarray overwrites
    # (starts the accumulation group), so no per-iteration memset is needed.
    # (Reused slots require zeroing because the matmul continues-accumulates
    # onto stale data; confirmed by iso test.)

    for expert_l in nl.sequential_range(experts_local):
        expert = prg_id * experts_local + expert_l
        gate_scale_table = _load_coalesced_scale_table(
            gate_up_scales, expert
        )
        down_scale_table = _load_coalesced_scale_table(
            down_scales, expert
        )
        nisa.memset(dst=expert_output, value=0.0)

        affinity = nl.ndarray(
            (tokens, 1), dtype=affinities.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=affinity,
            src=affinities[:, expert : expert + 1],
        )
        affinity_f32 = nl.ndarray(
            (tokens, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.activation(
            dst=affinity_f32,
            op=nl.copy,
            data=affinity,
            scale=1.0,
        )

        gate = nl.ndarray(
            (tokens, intermediate), dtype=nl.float32, buffer=nl.sbuf
        )
        up = nl.ndarray(
            (tokens, intermediate), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=gate, value=0.0)
        nisa.memset(dst=up, value=0.0)
        partial = nl.ndarray(
            (tokens, COALESCED_WIDTH),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )

        gate_groups = hidden_tiles // column_packing
        if column_packing == 1 and coarse_scale:
            # Coarse output-block scale is input-independent (s[i,o] == s[o]),
            # so sum_i(partial_i * s[o]) == s[o] * sum_i partial_i. Accumulate
            # the whole hidden contraction natively in PSUM (a matmul into a
            # fresh nl.psum starts the accumulation group; subsequent matmuls
            # accumulate), then post-scale each output block ONCE -- replacing
            # hidden_tiles*intermediate_tiles narrow Vector scale-adds with
            # intermediate_tiles wide post-scales per projection.
            gate_psum = nl.ndarray(
                (TILE, COALESCED_WIDTH), dtype=nl.float32, buffer=nl.psum
            )
            up_psum = nl.ndarray(
                (TILE, COALESCED_WIDTH), dtype=nl.float32, buffer=nl.psum
            )
            for input_tile in range(hidden_tiles):
                input_start = input_tile * TILE
                weight_slot = input_tile % 2
                nisa.dma_copy(
                    dst=weight_slots[weight_slot],
                    src=gate_up[
                        expert,
                        nl.ds(input_start, TILE),
                        :,
                        :,
                    ],
                )
                native_weight = weight_slots[weight_slot].view(
                    nl.float8_e4m3
                )
                nisa.nc_matmul(
                    dst=gate_psum,
                    stationary=hidden_sb[:, input_tile, :],
                    moving=native_weight[:, 0, :],
                )
                nisa.nc_matmul(
                    dst=up_psum,
                    stationary=hidden_sb[:, input_tile, :],
                    moving=native_weight[:, 1, :],
                )
            for output_block in range(intermediate_tiles):
                output_start = output_block * TILE
                gate_scale_index = output_block
                up_scale_index = (
                    hidden_tiles * intermediate_tiles + output_block
                )
                nisa.tensor_scalar(
                    dst=gate[:, nl.ds(output_start, TILE)],
                    data=gate_psum[:tokens, nl.ds(output_start, TILE)],
                    op0=nl.multiply,
                    operand0=gate_scale_table[
                        :tokens,
                        gate_scale_index : gate_scale_index + 1,
                    ],
                )
                nisa.tensor_scalar(
                    dst=up[:, nl.ds(output_start, TILE)],
                    data=up_psum[:tokens, nl.ds(output_start, TILE)],
                    op0=nl.multiply,
                    operand0=gate_scale_table[
                        :tokens,
                        up_scale_index : up_scale_index + 1,
                    ],
                )
        else:
            for input_group in range(gate_groups):
                gate_psum = nl.ndarray(
                    (TILE, COALESCED_WIDTH), dtype=nl.float32, buffer=nl.psum
                )
                up_psum = nl.ndarray(
                    (TILE, COALESCED_WIDTH), dtype=nl.float32, buffer=nl.psum
                )
                for packed_index in range(column_packing):
                    input_tile = (
                        input_group * column_packing + packed_index
                    )
                    input_start = input_tile * TILE
                    weight_slot = packed_index % 2
                    nisa.dma_copy(
                        dst=weight_slots[weight_slot],
                        src=gate_up[
                            expert,
                            nl.ds(input_start, TILE),
                            :,
                            :,
                        ],
                    )
                    packed_row = packed_index * tokens
                    native_weight = weight_slots[weight_slot].view(
                        nl.float8_e4m3
                    )
                    nisa.nc_matmul(
                        dst=gate_psum[
                            nl.ds(packed_row, tokens), :
                        ],
                        stationary=hidden_sb[:, input_tile, :],
                        moving=native_weight[:, 0, :],
                        tile_position=(0, packed_row),
                        tile_size=(TILE, tokens),
                    )
                    nisa.nc_matmul(
                        dst=up_psum[
                            nl.ds(packed_row, tokens), :
                        ],
                        stationary=hidden_sb[:, input_tile, :],
                        moving=native_weight[:, 1, :],
                        tile_position=(0, packed_row),
                        tile_size=(TILE, tokens),
                    )

                for packed_index in range(column_packing):
                    input_tile = (
                        input_group * column_packing + packed_index
                    )
                    packed_row = packed_index * tokens
                    nisa.activation(
                        dst=partial,
                        op=nl.copy,
                        data=gate_psum[
                            nl.ds(packed_row, tokens), :
                        ],
                        scale=1.0,
                    )
                    for output_block in range(intermediate_tiles):
                        output_start = output_block * TILE
                        scale_index = (
                            input_tile * intermediate_tiles + output_block
                        )
                        partial_block = partial[
                            :, nl.ds(output_start, TILE)
                        ]
                        gate_block = gate[
                            :, nl.ds(output_start, TILE)
                        ]
                        nisa.scalar_tensor_tensor(
                            data=partial_block,
                            op0=nl.multiply,
                            operand0=gate_scale_table[
                                :tokens,
                                scale_index : scale_index + 1,
                            ],
                            op1=nl.add,
                            operand1=gate_block,
                            dst=gate_block,
                        )

                    nisa.activation(
                        dst=partial,
                        op=nl.copy,
                        data=up_psum[
                            nl.ds(packed_row, tokens), :
                        ],
                        scale=1.0,
                    )
                    for output_block in range(intermediate_tiles):
                        output_start = output_block * TILE
                        scale_index = (
                            hidden_tiles * intermediate_tiles
                            + input_tile * intermediate_tiles
                            + output_block
                        )
                        partial_block = partial[
                            :, nl.ds(output_start, TILE)
                        ]
                        up_block = up[
                            :, nl.ds(output_start, TILE)
                        ]
                        nisa.scalar_tensor_tensor(
                            data=partial_block,
                            op0=nl.multiply,
                            operand0=gate_scale_table[
                                :tokens,
                                scale_index : scale_index + 1,
                            ],
                            op1=nl.add,
                            operand1=up_block,
                            dst=up_block,
                        )

        zero_bias = nl.ndarray(
            (tokens, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.memset(dst=zero_bias, value=0.0)
        silu_gate = nl.ndarray(
            (tokens, intermediate), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.activation(
            dst=silu_gate,
            op=nl.silu,
            data=gate,
            bias=zero_bias,
            scale=1.0,
        )
        product = nl.ndarray(
            (tokens, intermediate), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=product,
            data1=silu_gate,
            data2=up,
            op=nl.multiply,
        )
        activated_row = nl.ndarray(
            (tokens, intermediate),
            dtype=hidden.dtype,
            buffer=nl.sbuf,
        )
        nisa.activation(
            dst=activated_row,
            op=nl.copy,
            data=product,
            scale=1.0,
        )
        activated = nl.ndarray(
            (TILE, intermediate_tiles, tokens),
            dtype=hidden.dtype,
            buffer=nl.sbuf,
        )
        for intermediate_tile in nl.affine_range(intermediate_tiles):
            intermediate_start = intermediate_tile * TILE
            activated_psum = nl.ndarray(
                (TILE, tokens), dtype=hidden.dtype, buffer=nl.psum
            )
            nisa.nc_transpose(
                dst=activated_psum,
                data=activated_row[
                    :, nl.ds(intermediate_start, TILE)
                ],
            )
            nisa.tensor_copy(
                dst=activated[:, intermediate_tile, :],
                src=activated_psum,
            )

        hidden_slabs = hidden_size // COALESCED_WIDTH
        down_groups = intermediate_tiles // column_packing
        for hidden_slab in nl.affine_range(hidden_slabs):
            hidden_start = hidden_slab * COALESCED_WIDTH
            if column_packing == 1 and coarse_scale:
                # Coarse scale (see gate/up): accumulate the intermediate
                # contraction for this hidden slab natively in PSUM, then
                # post-scale each 128-wide output block ONCE with s[o].
                down_psum = nl.ndarray(
                    (TILE, COALESCED_WIDTH), dtype=nl.float32, buffer=nl.psum
                )
                for intermediate_tile in range(intermediate_tiles):
                    intermediate_start = intermediate_tile * TILE
                    weight_slot = intermediate_tile % 2
                    nisa.dma_copy(
                        dst=weight_slots[weight_slot][:, 0, :],
                        src=down[
                            expert,
                            nl.ds(intermediate_start, TILE),
                            nl.ds(hidden_start, COALESCED_WIDTH),
                        ],
                    )
                    native_weight = weight_slots[weight_slot][
                        :, 0, :
                    ].view(nl.float8_e4m3)
                    nisa.nc_matmul(
                        dst=down_psum,
                        stationary=activated[:, intermediate_tile, :],
                        moving=native_weight,
                    )
                for slab_block in range(COALESCED_WIDTH // TILE):
                    output_block = (
                        hidden_slab * (COALESCED_WIDTH // TILE) + slab_block
                    )
                    output_start = slab_block * TILE
                    scale_index = output_block
                    expert_start = hidden_start + output_start
                    nisa.tensor_scalar(
                        dst=expert_output[:, nl.ds(expert_start, TILE)],
                        data=down_psum[:tokens, nl.ds(output_start, TILE)],
                        op0=nl.multiply,
                        operand0=down_scale_table[
                            :tokens,
                            scale_index : scale_index + 1,
                        ],
                    )
                # column_packing==1 is a compile-time constant, so this branch
                # is taken uniformly for every slab (safe inside affine_range).
                continue
            for input_group in range(down_groups):
                down_psum = nl.ndarray(
                    (TILE, COALESCED_WIDTH), dtype=nl.float32, buffer=nl.psum
                )
                for packed_index in range(column_packing):
                    intermediate_tile = (
                        input_group * column_packing + packed_index
                    )
                    intermediate_start = intermediate_tile * TILE
                    weight_slot = packed_index % 2
                    nisa.dma_copy(
                        dst=weight_slots[weight_slot][:, 0, :],
                        src=down[
                            expert,
                            nl.ds(intermediate_start, TILE),
                            nl.ds(hidden_start, COALESCED_WIDTH),
                        ],
                    )
                    packed_row = packed_index * tokens
                    native_weight = weight_slots[weight_slot][
                        :, 0, :
                    ].view(nl.float8_e4m3)
                    nisa.nc_matmul(
                        dst=down_psum[
                            nl.ds(packed_row, tokens), :
                        ],
                        stationary=activated[
                            :, intermediate_tile, :
                        ],
                        moving=native_weight,
                        tile_position=(0, packed_row),
                        tile_size=(TILE, tokens),
                    )

                for packed_index in range(column_packing):
                    intermediate_tile = (
                        input_group * column_packing + packed_index
                    )
                    packed_row = packed_index * tokens
                    nisa.activation(
                        dst=partial,
                        op=nl.copy,
                        data=down_psum[
                            nl.ds(packed_row, tokens), :
                        ],
                        scale=1.0,
                    )
                    for slab_block in range(
                        COALESCED_WIDTH // TILE
                    ):
                        output_block = (
                            hidden_slab * (COALESCED_WIDTH // TILE)
                            + slab_block
                        )
                        output_start = slab_block * TILE
                        scale_index = (
                            intermediate_tile * hidden_tiles
                            + output_block
                        )
                        partial_block = partial[
                            :, nl.ds(output_start, TILE)
                        ]
                        expert_start = hidden_start + output_start
                        expert_block = expert_output[
                            :, nl.ds(expert_start, TILE)
                        ]
                        nisa.scalar_tensor_tensor(
                            dst=expert_block,
                            data=partial_block,
                            op0=nl.multiply,
                            operand0=down_scale_table[
                                :tokens,
                                scale_index : scale_index + 1,
                            ],
                            op1=nl.add,
                            operand1=expert_block,
                        )

        for hidden_tile in nl.affine_range(hidden_tiles):
            hidden_start = hidden_tile * TILE
            routed_block = routed[:, nl.ds(hidden_start, TILE)]
            nisa.scalar_tensor_tensor(
                data=expert_output[:, nl.ds(hidden_start, TILE)],
                op0=nl.multiply,
                operand0=affinity_f32[:, 0],
                op1=nl.add,
                operand1=routed_block,
                dst=routed_block,
            )

    nisa.dma_copy(dst=output[nl.ds(prg_id * tokens, tokens), :], src=routed)
    return output


@nki.jit
def nki_moe_fused_w8_fp8_block_coalesced(
    hidden,
    gate_up,
    down,
    gate_up_scales,
    down_scales,
    affinities,
):
    """Per-128x128-block scales. Default; pair with block_pow2_coalesced weights."""
    return _moe_fused_w8_block_coalesced(
        hidden,
        gate_up,
        down,
        gate_up_scales,
        down_scales,
        affinities,
    )


@nki.jit
def nki_moe_fused_w8_fp8_block_coalesced_ob(
    hidden,
    gate_up,
    down,
    gate_up_scales,
    down_scales,
    affinities,
):
    """Per-128-output-row scales, PSUM-accumulated at B=128. Pair with block_ob_coalesced."""
    return _moe_fused_w8_block_coalesced(
        hidden,
        gate_up,
        down,
        gate_up_scales,
        down_scales,
        affinities,
        coarse_scale=True,
    )
