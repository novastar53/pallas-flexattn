"""Mask modification functions for FlexAttention.

Mask mods are boolean functions that determine which query-key pairs can attend.
They take query and key indices and return True where attention is allowed.
"""
from typing import Callable

import jax
import jax.numpy as jnp

# Type alias for mask modification functions
MaskMod = Callable[[jax.Array, jax.Array], jax.Array]


def causal_mask(q_idx: jax.Array, kv_idx: jax.Array) -> jax.Array:
    """Causal mask - query can only attend to keys at same or earlier positions."""
    return q_idx >= kv_idx


def bidirectional_mask(q_idx: jax.Array, kv_idx: jax.Array) -> jax.Array:
    """Full bidirectional attention - all positions can attend to all positions."""
    return jnp.ones_like(q_idx, dtype=jnp.bool_)


def sliding_window_mask(left: int, right: int) -> MaskMod:
    """Create a sliding window mask function.

    Args:
        left: Number of positions to look back (-1 for unlimited)
        right: Number of positions to look forward (-1 for unlimited)

    Returns:
        MaskMod function that checks if |q_idx - kv_idx| is within window
    """
    def mask(q_idx: jax.Array, kv_idx: jax.Array) -> jax.Array:
        delta = q_idx - kv_idx
        mask = jnp.ones_like(delta, dtype=jnp.bool_)
        if left >= 0:
            mask = mask & (delta <= left)
        if right >= 0:
            mask = mask & (-delta <= right)
        return mask
    return mask


def document_mask(seq_lengths: jax.Array, block_size: int = 128) -> MaskMod:
    """Create a document boundary mask for variable-length sequences.

    Prevents attention across document boundaries in packed sequences.

    Args:
        seq_lengths: Array of sequence lengths for each document in batch
        block_size: Block size used for tiling

    Returns:
        MaskMod function that blocks cross-document attention
    """
    # TODO: Implement once BlockMask data structure is in place
    raise NotImplementedError("Document mask requires BlockMask support")
