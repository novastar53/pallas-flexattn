"""Pallas Flash Attention implementation.

Input shapes: Q, K, V are (B, H, T, D) where:
    B = batch size
    H = number of heads
    T = sequence length
    D = head dimension

Standard attention computes:
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V

This keeps memory usage at O(N) instead of O(N²) since you never store the full attention matrix.
"""
from functools import partial
from typing import Optional, Tuple
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from pallas_flexattn.mask_mod import MaskMod, causal_mask
from pallas_flexattn.block_mask import BlockMask

# Default kernel configuration
DEFAULT_BLOCK_R = 64
DEFAULT_BLOCK_C = 64
DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 3


@partial(jax.jit, static_argnums=(3,))
def mha_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask_mod: Optional[MaskMod] = None,
) -> jax.Array:
    """Reference multi-head attention (materializes N×N matrix).

    Computes: softmax(Q @ K^T / sqrt(d)) @ V
    Input shape: (B, H, T, D) - batch, heads, sequence, head_dim
    """
    d = q.shape[-1]
    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(d)
    T = q.shape[2]

    if mask_mod is not None:
        q_idx = jnp.arange(T)[:, None]
        kv_idx = jnp.arange(T)[None, :]
        mask = mask_mod(q_idx, kv_idx)
        logits = jnp.where(mask, logits, -jnp.inf)

    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum('bhqk,bhkd->bhqd', probs, v)


def flash_attention_fwd_kernel(
    q_ref, k_ref, v_ref, o_ref, logsumexp_ref, *,
    qk_scale: float,
    num_k_blocks: int,
    block_r: int,
    block_c: int,
    mask_mod: Optional[MaskMod],
):
    """Flash attention forward kernel."""
    q_reg = plgpu.load(q_ref.at[0, :, :])
    o_reg = jnp.zeros(q_reg.shape, jnp.float32)
    max_reg = jnp.full((block_r,), -jnp.inf, dtype=jnp.float32)
    l_reg = jnp.zeros((block_r,), dtype=jnp.float32)

    blk_idx = pl.program_id(1)
    q_idx = block_r * blk_idx + jnp.arange(block_r)

    def body(t, args):
        max_reg, l_reg, o_reg = args
        idx = pl.dslice(t * block_c, block_c)
        k_blk = plgpu.load(k_ref.at[0, idx, :])
        v_blk = plgpu.load(v_ref.at[0, idx, :])

        def compute_block(_):
            s_blk = pl.dot(q_reg, k_blk, trans_b=True, precision='float32') * qk_scale

            if mask_mod is not None:
                kv_idx = block_c * t + jnp.arange(block_c)
                mask = mask_mod(q_idx[:, None], kv_idx[None, :])
                s_blk = jnp.where(mask, s_blk, -jnp.inf)

            max_blk = jnp.maximum(max_reg, jnp.max(s_blk, axis=-1))
            p_blk = jnp.exp(s_blk - max_blk[:, None])
            p_blk = jnp.where(jnp.isnan(p_blk), 0.0, p_blk)
            l_blk = jnp.sum(p_blk, axis=-1)
            o_blk = pl.dot(p_blk.astype(v_blk.dtype), v_blk)
            alpha = jnp.exp(max_reg - max_blk)
            alpha = jnp.where(jnp.isnan(alpha), 0.0, alpha)
            return (max_blk,
                    l_reg * alpha + l_blk,
                    o_reg * alpha[:, None] + o_blk)

        # TODO: Add block skipping optimization for causal masks using BlockMask
        return compute_block(None)

    max_reg, l_reg, o_reg = jax.lax.fori_loop(0, num_k_blocks, body, (max_reg, l_reg, o_reg))
    logsumexp_reg = max_reg + jnp.log(l_reg)
    o_reg = o_reg / l_reg[:, None]
    plgpu.store(o_ref.at[0, :, :], o_reg.astype(o_ref.dtype))
    plgpu.store(logsumexp_ref.at[0, :], logsumexp_reg.astype(logsumexp_ref.dtype))


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def flash_attention_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask_mod: Optional[MaskMod] = None,
    block_r: int = DEFAULT_BLOCK_R,
    block_c: int = DEFAULT_BLOCK_C,
    num_warps: int = DEFAULT_NUM_WARPS,
    num_stages: int = DEFAULT_NUM_STAGES,
    interpret: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """Flash attention forward pass."""
    B, H, T, C = q.shape
    B_flat = B * H
    q_flat = q.reshape(-1, T, C)
    k_flat = k.reshape(-1, T, C)
    v_flat = v.reshape(-1, T, C)
    qk_scale = 1.0 / math.sqrt(C)
    num_k_blocks = pl.cdiv(T, block_c)
    grid = (B_flat, pl.cdiv(T, block_r))

    out_flat, logsumexp = pl.pallas_call(
        partial(
            flash_attention_fwd_kernel,
            qk_scale=qk_scale,
            num_k_blocks=num_k_blocks,
            block_r=block_r,
            block_c=block_c,
            mask_mod=mask_mod,
        ),
        out_shape=[
            jax.ShapeDtypeStruct(q_flat.shape, q_flat.dtype),
            jax.ShapeDtypeStruct((B * H, T), q_flat.dtype)
        ],
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, block_r, C), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0))
        ],
        out_specs=[
            pl.BlockSpec((1, block_r, C), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, block_r), lambda b, t: (b, t))
        ],
        interpret=interpret,
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps,
            num_stages=num_stages
        )
    )(q_flat, k_flat, v_flat)

    out = out_flat.reshape(q.shape)
    logsumexp = logsumexp.reshape(B, H, T)
    return out, logsumexp


def flash_attention_bwd_preprocess_kernel(o_ref, do_ref, d_ref):
    """Compute D = rowsum(O ⊙ dO) for backward pass."""
    o_reg = plgpu.load(o_ref)
    do_reg = plgpu.load(do_ref)
    d_reg = jnp.sum((o_reg * do_reg).astype(jnp.float32), axis=-1)
    plgpu.store(d_ref, d_reg.astype(d_ref.dtype))


def flash_attention_bwd_preprocess(
    o_flat: jax.Array,
    do_flat: jax.Array,
    block_r: int = DEFAULT_BLOCK_R,
    num_warps: int = DEFAULT_NUM_WARPS,
    num_stages: int = DEFAULT_NUM_STAGES,
    interpret: bool = False,
) -> jax.Array:
    """Preprocess for backward: compute D = rowsum(O ⊙ dO)."""
    B_flat, T, C = o_flat.shape
    grid = (B_flat, pl.cdiv(T, block_r))

    d_flat = pl.pallas_call(
        partial(flash_attention_bwd_preprocess_kernel),
        out_shape=jax.ShapeDtypeStruct((B_flat, T), o_flat.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, block_r, C), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, block_r, C), lambda b, t: (b, t, 0)),
        ],
        out_specs=pl.BlockSpec((1, block_r), lambda b, t: (b, t)),
        interpret=interpret,
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps,
            num_stages=num_stages
        )
    )(o_flat, do_flat)
    return d_flat


def flash_attention_bwd_dkv_kernel(
    q_ref, k_ref, v_ref, do_ref, logsumexp_ref, d_ref,
    dk_ref, dv_ref, *,
    qk_scale: float,
    softmax_scale: float,
    num_q_blocks: int,
    block_r: int,
    block_c: int,
    mask_mod: Optional[MaskMod],
):
    """Compute dK and dV gradients."""
    k_reg = plgpu.load(k_ref.at[0, :, :])
    v_reg = plgpu.load(v_ref.at[0, :, :])
    kv_blk_idx = pl.program_id(1)
    kv_idx = block_c * kv_blk_idx + jnp.arange(block_c)

    dk_acc = jnp.zeros(dk_ref.shape, dtype=jnp.float32)
    dv_acc = jnp.zeros(dv_ref.shape, dtype=jnp.float32)

    def body(t, carry):
        dk_acc, dv_acc = carry
        idx = pl.dslice(t * block_r, block_r)
        q_blk = plgpu.load(q_ref.at[0, idx, :])
        do_blk = plgpu.load(do_ref.at[0, idx, :])
        logsumexp_blk = plgpu.load(logsumexp_ref.at[0, idx])
        d_blk = plgpu.load(d_ref.at[0, idx])

        def compute_block(_):
            s_blk = pl.dot(q_blk, k_reg, trans_b=True, precision='float32') * qk_scale

            if mask_mod is not None:
                q_idx = block_r * t + jnp.arange(block_r)
                mask = mask_mod(q_idx[:, None], kv_idx[None, :])
                s_blk = jnp.where(mask, s_blk, -jnp.inf)

            p_blk = jnp.exp(s_blk - logsumexp_blk[..., None])
            dp_blk = pl.dot(do_blk, v_reg, trans_b=True)
            ds_blk = p_blk * (dp_blk - d_blk[..., None]) * softmax_scale
            dv_new = dv_acc + pl.dot(p_blk.astype(do_blk.dtype), do_blk, trans_a=True)
            dk_new = dk_acc + pl.dot(ds_blk.astype(q_blk.dtype), q_blk, trans_a=True)
            return (dk_new, dv_new)

        # TODO: Add block skipping optimization using BlockMask
        return compute_block(None)

    dk_acc, dv_acc = jax.lax.fori_loop(0, num_q_blocks, body, (dk_acc, dv_acc))
    plgpu.store(dk_ref, dk_acc.astype(dk_ref.dtype))
    plgpu.store(dv_ref, dv_acc.astype(dv_ref.dtype))


def flash_attention_bwd_dkv(
    q_flat: jax.Array,
    k_flat: jax.Array,
    v_flat: jax.Array,
    do_flat: jax.Array,
    logsumexp_flat: jax.Array,
    d_flat: jax.Array,
    qk_scale: float,
    softmax_scale: float,
    block_r: int = DEFAULT_BLOCK_R,
    block_c: int = DEFAULT_BLOCK_C,
    num_warps: int = DEFAULT_NUM_WARPS,
    num_stages: int = DEFAULT_NUM_STAGES,
    mask_mod: Optional[MaskMod] = None,
    interpret: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """Compute dK and dV using pallas_call."""
    B_flat, T, C = q_flat.shape
    num_q_blocks = pl.cdiv(T, block_r)
    grid = (B_flat, pl.cdiv(T, block_c))

    dk_flat, dv_flat = pl.pallas_call(
        partial(
            flash_attention_bwd_dkv_kernel,
            qk_scale=qk_scale,
            softmax_scale=softmax_scale,
            num_q_blocks=num_q_blocks,
            block_r=block_r,
            block_c=block_c,
            mask_mod=mask_mod,
        ),
        out_shape=[
            jax.ShapeDtypeStruct(k_flat.shape, k_flat.dtype),
            jax.ShapeDtypeStruct(v_flat.shape, v_flat.dtype),
        ],
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),
            pl.BlockSpec((1, block_c, C), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, block_c, C), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),
            pl.BlockSpec((1, T), lambda b, _: (b, 0)),
            pl.BlockSpec((1, T), lambda b, _: (b, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, block_c, C), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, block_c, C), lambda b, t: (b, t, 0)),
        ],
        interpret=interpret,
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps,
            num_stages=num_stages
        )
    )(q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat)
    return dk_flat, dv_flat


def flash_attention_bwd_dq_kernel(
    q_ref, k_ref, v_ref, do_ref, logsumexp_ref, d_ref,
    dq_ref, *,
    qk_scale: float,
    softmax_scale: float,
    num_kv_blocks: int,
    block_r: int,
    block_c: int,
    mask_mod: Optional[MaskMod],
):
    """Compute dQ gradient."""
    q_reg = plgpu.load(q_ref.at[0, :, :])
    do_reg = plgpu.load(do_ref.at[0, :, :])
    logsumexp_reg = plgpu.load(logsumexp_ref.at[0, :])
    d_reg = plgpu.load(d_ref.at[0, :])
    q_blk_idx = pl.program_id(1)
    q_idx = block_r * q_blk_idx + jnp.arange(block_r)
    dq_acc = jnp.zeros(dq_ref.shape, dtype=jnp.float32)

    def body(t, carry):
        dq_acc = carry
        idx = pl.dslice(t * block_c, block_c)
        k_blk = plgpu.load(k_ref.at[0, idx, :])
        v_blk = plgpu.load(v_ref.at[0, idx, :])

        def compute_block(_):
            s_blk = pl.dot(q_reg, k_blk, trans_b=True, precision='float32') * qk_scale

            if mask_mod is not None:
                kv_idx = block_c * t + jnp.arange(block_c)
                mask = mask_mod(q_idx[:, None], kv_idx[None, :])
                s_blk = jnp.where(mask, s_blk, -jnp.inf)

            p_blk = jnp.exp(s_blk - logsumexp_reg[..., None])
            dp_blk = pl.dot(do_reg, v_blk, trans_b=True)
            ds_blk = p_blk * (dp_blk - d_reg[..., None]) * softmax_scale
            dq_new = dq_acc + pl.dot(ds_blk.astype(k_blk.dtype), k_blk)
            return dq_new

        # TODO: Add block skipping optimization using BlockMask
        return compute_block(None)

    dq_acc = jax.lax.fori_loop(0, num_kv_blocks, body, dq_acc)
    plgpu.store(dq_ref, dq_acc.astype(dq_ref.dtype))


def flash_attention_bwd_dq(
    q_flat: jax.Array,
    k_flat: jax.Array,
    v_flat: jax.Array,
    do_flat: jax.Array,
    logsumexp_flat: jax.Array,
    d_flat: jax.Array,
    qk_scale: float,
    softmax_scale: float,
    block_r: int = DEFAULT_BLOCK_R,
    block_c: int = DEFAULT_BLOCK_C,
    num_warps: int = DEFAULT_NUM_WARPS,
    num_stages: int = DEFAULT_NUM_STAGES,
    mask_mod: Optional[MaskMod] = None,
    interpret: bool = False,
) -> jax.Array:
    """Compute dQ using pallas_call."""
    B_flat, T, C = q_flat.shape
    num_kv_blocks = pl.cdiv(T, block_c)
    grid = (B_flat, pl.cdiv(T, block_r))

    dq_flat = pl.pallas_call(
        partial(
            flash_attention_bwd_dq_kernel,
            qk_scale=qk_scale,
            softmax_scale=softmax_scale,
            num_kv_blocks=num_kv_blocks,
            block_r=block_r,
            block_c=block_c,
            mask_mod=mask_mod,
        ),
        out_shape=jax.ShapeDtypeStruct(q_flat.shape, q_flat.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec((1, block_r, C), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),
            pl.BlockSpec((1, T, C), lambda b, _: (b, 0, 0)),
            pl.BlockSpec((1, block_r, C), lambda b, t: (b, t, 0)),
            pl.BlockSpec((1, block_r), lambda b, t: (b, t)),
            pl.BlockSpec((1, block_r), lambda b, t: (b, t)),
        ],
        out_specs=pl.BlockSpec((1, block_r, C), lambda b, t: (b, t, 0)),
        interpret=interpret,
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps,
            num_stages=num_stages
        )
    )(q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat)
    return dq_flat


def flash_attention_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    o: jax.Array,
    logsumexp: jax.Array,
    do: jax.Array,
    mask_mod: Optional[MaskMod] = None,
    block_r: int = DEFAULT_BLOCK_R,
    block_c: int = DEFAULT_BLOCK_C,
    num_warps: int = DEFAULT_NUM_WARPS,
    num_stages: int = DEFAULT_NUM_STAGES,
    interpret: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Flash attention backward pass using 3 separate kernels."""
    B, H, T, C = q.shape
    qk_scale = 1.0 / math.sqrt(C)
    softmax_scale = 1.0 / math.sqrt(C)

    # Flatten batch and head dimensions
    q_flat = q.reshape(-1, T, C)
    k_flat = k.reshape(-1, T, C)
    v_flat = v.reshape(-1, T, C)
    o_flat = o.reshape(-1, T, C)
    do_flat = do.reshape(-1, T, C)
    logsumexp_flat = logsumexp.reshape(-1, T)

    # Kernel 1: Preprocess - compute D = rowsum(O ⊙ dO)
    d_flat = flash_attention_bwd_preprocess(
        o_flat, do_flat, block_r, num_warps, num_stages, interpret
    )

    # Kernel 2: Compute dK, dV
    dk_flat, dv_flat = flash_attention_bwd_dkv(
        q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat,
        qk_scale, softmax_scale, block_r, block_c, num_warps, num_stages,
        mask_mod, interpret
    )

    # Kernel 3: Compute dQ
    dq_flat = flash_attention_bwd_dq(
        q_flat, k_flat, v_flat, do_flat, logsumexp_flat, d_flat,
        qk_scale, softmax_scale, block_r, block_c, num_warps, num_stages,
        mask_mod, interpret
    )

    return (
        dq_flat.reshape(q.shape),
        dk_flat.reshape(k.shape),
        dv_flat.reshape(v.shape),
    )


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask_mod: Optional[MaskMod] = None,
    block_r: int = DEFAULT_BLOCK_R,
    block_c: int = DEFAULT_BLOCK_C,
    num_warps: int = DEFAULT_NUM_WARPS,
    num_stages: int = DEFAULT_NUM_STAGES,
    interpret: bool = False,
) -> jax.Array:
    """Flash attention with custom backward pass."""
    o, _ = flash_attention_fwd(q, k, v, mask_mod, block_r, block_c, num_warps, num_stages, interpret)
    return o


def flash_attention_fwd_rule(
    q, k, v, mask_mod, block_r, block_c, num_warps, num_stages, interpret
):
    """Forward rule for custom_vjp."""
    o, logsumexp = flash_attention_fwd(q, k, v, mask_mod, block_r, block_c, num_warps, num_stages, interpret)
    # Only store differentiable args and outputs in residuals (not nondiff args)
    return o, (q, k, v, o, logsumexp)


def flash_attention_bwd_rule(mask_mod, block_r, block_c, num_warps, num_stages, interpret, res, do):
    """Backward rule for custom_vjp.

    With nondiff_argnums=(3,4,5,6,7,8), bwd receives each nondiff arg separately.
    """
    q, k, v, o, logsumexp = res
    dq, dk, dv = flash_attention_bwd(
        q, k, v, o, logsumexp, do, mask_mod,
        block_r, block_c, num_warps, num_stages, interpret
    )
    # Return gradients for differentiable inputs only
    return (dq, dk, dv)


flash_attention.defvjp(flash_attention_fwd_rule, flash_attention_bwd_rule)
