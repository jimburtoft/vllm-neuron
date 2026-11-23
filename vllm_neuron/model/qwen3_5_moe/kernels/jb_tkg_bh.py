# Adapted from qwen35-gdn-deltanet-neuron-kernels v3.3+ (Apache-2.0): items LNC-sharded.
"""NKI kernel for single-token DeltaNet decode, batched over ALL (request, head) items.

This is the multi-request generalization of `deltanet_tkg_batched`. Where
`deltanet_tkg_batched` processes H heads of ONE request per launch (leading dim H),
this kernel processes all B*H (request, head) items of a serving batch in a SINGLE
launch (leading dim B*H, a flat work list).

Motivation: the layer wrapper previously looped the batch dimension in
Python -- `for b in range(B): deltanet_tkg_batched(item[b])` -- paying the fixed
per-launch dispatch/prologue overhead B times per decode step. Flattening to one
launch amortizes that overhead to a single occurrence.

Measured (trn2.3xlarge SDK 2.31 NKI 0.5.0, torch-xla lazy-tensor path (~50 ms per launch),
H=16, vs the Python B-loop; the gain scales with per-launch overhead):
  B=1  : 0.99x (neutral -- one launch either way)
  B=4  : 3.96x
  B=16 : 16.0x
  B=32 : 31.8x
Bit-identical output vs the per-request loop (abs_max = 0.0). On-core per-item cost
is flat (~3.1 us/item); the entire win is launch amortization. Benefit is
unconditional at B>1 and never a regression at B=1.

Per-item body is the v3.1 tkg body (F1/F2/F3 + SBUF preload). The v3.2 key-fold is
deliberately NOT used here: it was measured to regress on single-token decode
(0.94x at H=16) because there is no sequence loop to amortize its added transposes.

Inputs (HBM):
  q, k, v, g, beta: (BH, 128, 1)   where BH = B*H, items flattened (b-major:
                                    item index = b*H + h)
  state_in:         (BH, 128, 128)
Returns:
  o:         (BH, 128, 1)
  state_out: (BH, 128, 128)
"""

import nki
import nki.isa as nisa
import nki.language as nl

P_MAX = 128
_BROADCAST_MASK = [0] * 32


@nki.jit
def deltanet_tkg_batched_bh(q, k, v, g, beta, state_in,
                            z_in=None, norm_weight=None, norm_eps=1e-6):
    """Single-token DeltaNet decode for all B*H (request, head) items in one launch.

    Optional folded RMSNormGated tail. Pass BOTH to enable:
      z_in:        (BH, 128, 1) -- gate input z (pre-silu), same layout as v
      norm_weight: (128, 1)     -- RMSNorm weight as a COLUMN (see layout note below)
      norm_eps:    float

    When enabled the kernel applies
        out = (o * rsqrt(mean_D(o^2) + eps)) * norm_weight * silu(z)
    while o is still SBUF-resident, removing ~9 PyTorch launches per layer per token.
    Decode runs once per generated token, so this is where fold savings actually
    accumulate (prefill fires once per request -- measured neutral there).

    LAYOUT WARNING -- this kernel is the MIRROR IMAGE of the prefill kernel:
      prefill: [token=partition, D=free]  -> reduce over D is a FREE-axis reduce, so
               `nisa.activation_reduce` works and norm_weight is a (1, D) ROW.
      decode:  [D=partition, 1=free]      -> reduce over D is a PARTITION-axis reduce, so
               activation_reduce does NOT apply; use nc_matmul(o_sq, ones_col) on the
               Tensor engine, and norm_weight must be a (D, 1) COLUMN.
    Copying the prefill fold here verbatim would produce silently wrong numbers rather
    than an error.
    """
    num_items = q.shape[0]
    dim = P_MAX
    fold_tail = (z_in is not None) and (norm_weight is not None)

    o_hbm = nl.ndarray((num_items, P_MAX, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    state_out_hbm = nl.ndarray((num_items, P_MAX, dim), dtype=nl.float32, buffer=nl.shared_hbm)

    # Bulk preload all items' vectors as (128, BH) in SBUF (v3.1 preload pattern).
    q_all = nl.ndarray((P_MAX, num_items), dtype=q.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_all, src=q.ap(pattern=[[1, P_MAX], [P_MAX, num_items]], offset=0))
    k_all = nl.ndarray((P_MAX, num_items), dtype=k.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_all, src=k.ap(pattern=[[1, P_MAX], [P_MAX, num_items]], offset=0))
    v_all = nl.ndarray((P_MAX, num_items), dtype=v.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v_all, src=v.ap(pattern=[[1, P_MAX], [P_MAX, num_items]], offset=0))
    g_all = nl.ndarray((P_MAX, num_items), dtype=g.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_all, src=g.ap(pattern=[[1, P_MAX], [P_MAX, num_items]], offset=0))
    beta_all = nl.ndarray((P_MAX, num_items), dtype=beta.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=beta_all, src=beta.ap(pattern=[[1, P_MAX], [P_MAX, num_items]], offset=0))

    if fold_tail:
        z_all = nl.ndarray((P_MAX, num_items), dtype=z_in.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=z_all, src=z_in.ap(pattern=[[1, P_MAX], [P_MAX, num_items]], offset=0))
        # norm_weight arrives as (128, 1): D on the partition axis, matching o's layout.
        nw_col = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=nw_col, src=norm_weight)
        # Ones column for the partition-axis sum-of-squares matmul.
        ones_col = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_col, value=1.0)

    o_all = nl.ndarray((P_MAX, num_items), dtype=nl.float32, buffer=nl.sbuf)

    n_prgs = nl.num_programs(0)
    prg_id = nl.program_id(0) if n_prgs > 1 else 0
    n_loc = num_items // n_prgs
    for it_l in nl.affine_range(n_loc):
        it = prg_id * n_loc + it_l
        state = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=state, src=state_in[it])

        q_sb = q_all[0:P_MAX, it:it+1]
        k_sb = k_all[0:P_MAX, it:it+1]
        v_sb = v_all[0:P_MAX, it:it+1]
        g_sb = g_all[0:P_MAX, it:it+1]
        beta_sb = beta_all[0:P_MAX, it:it+1]

        # F1: state *= exp(g) IN-PLACE
        exp_g = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g, op=nl.exp, data=g_sb, bias=None, scale=1.0)
        nisa.tensor_scalar(dst=state, data=state, op0=nl.multiply, operand0=exp_g, engine=nisa.vector_engine)

        # F2: kv_mem = state^T @ k, PSUM read fuses into next tensor_tensor
        kv_mem_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=kv_mem_psum, stationary=state, moving=k_sb)

        v_sub = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=v_sub, data1=v_sb, data2=kv_mem_psum, op=nl.subtract)
        delta = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=delta, data=v_sub, op0=nl.multiply, operand0=beta_sb, engine=nisa.vector_engine)

        # outer(k, delta) via 4-way nc_stream_shuffle (v3.1 body)
        delta_row_psum = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=delta_row_psum, data=delta)
        delta_row_sb = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=delta_row_sb, src=delta_row_psum)
        delta_broadcast = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        for i_shuf in nl.static_range(P_MAX // 32):
            nisa.nc_stream_shuffle(
                src=delta_row_sb[0:1, 0:P_MAX],
                dst=delta_broadcast[i_shuf * 32 : i_shuf * 32 + 32, 0:P_MAX],
                shuffle_mask=_BROADCAST_MASK,
            )
        outer_prod = nl.ndarray((P_MAX, dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=outer_prod, data=delta_broadcast, op0=nl.multiply, operand0=k_sb, engine=nisa.vector_engine)
        # F3: state += outer_prod IN-PLACE
        nisa.tensor_tensor(dst=state, data1=state, data2=outer_prod, op=nl.add)

        # o = state^T @ q
        o_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=o_psum, stationary=state, moving=q_sb)

        if fold_tail:
            o_sb = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o_sb, src=o_psum)

            # sum_D(o^2): D is on the PARTITION axis, so this is a contraction over
            # partitions -> Tensor engine matmul against a ones column, giving (1, 1).
            o_sq = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o_sq, data1=o_sb, data2=o_sb, op=nl.multiply)
            ss_psum = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=ss_psum, stationary=o_sq, moving=ones_col)

            # rsqrt(mean + eps): activation computes op(data * scale + bias).
            rs = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rs, op=nl.rsqrt, data=ss_psum,
                            scale=1.0 / dim, bias=norm_eps)

            # The (1,1) scalar must be broadcast across all 128 partitions before it can
            # scale o. tensor_scalar broadcasts along FREE, not across partitions.
            rs_b = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for i_shuf in nl.static_range(P_MAX // 32):
                nisa.nc_stream_shuffle(
                    src=rs[0:1, 0:1],
                    dst=rs_b[i_shuf * 32 : i_shuf * 32 + 32, 0:1],
                    shuffle_mask=_BROADCAST_MASK,
                )

            # out = o * rsqrt * norm_weight * silu(z)
            nisa.tensor_tensor(dst=o_sb, data1=o_sb, data2=rs_b, op=nl.multiply)
            nisa.tensor_tensor(dst=o_sb, data1=o_sb, data2=nw_col, op=nl.multiply)
            sz = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sz, op=nl.silu, data=z_all[0:P_MAX, it:it+1],
                            bias=None, scale=1.0)
            nisa.tensor_tensor(dst=o_all[0:P_MAX, it:it+1], data1=o_sb, data2=sz,
                               op=nl.multiply)
        else:
            nisa.tensor_copy(dst=o_all[0:P_MAX, it:it+1], src=o_psum)

        nisa.dma_copy(dst=state_out_hbm[it], src=state)

    if n_prgs > 1:
        nisa.dma_copy(dst=o_hbm.ap(pattern=[[1, P_MAX], [P_MAX, n_loc]], offset=prg_id * n_loc * P_MAX),
                      src=o_all[0:P_MAX, prg_id * n_loc:(prg_id + 1) * n_loc])
    else:
        nisa.dma_copy(dst=o_hbm.ap(pattern=[[1, P_MAX], [P_MAX, num_items]], offset=0), src=o_all)
    return o_hbm, state_out_hbm
