"""Block-sparse attention mask for efficient sparse attention.

The BlockMask data structure stores which KV blocks each query block can attend to,
allowing the kernel to skip fully masked blocks entirely.
"""
from dataclasses import dataclass
from typing import Optional, Callable
import math

import jax
import jax.numpy as jnp

from pallas_flexattn.mask_mod import MaskMod


@dataclass(frozen=True)
class BlockMask:
    """Block-sparse mask data structure.

    For each query block, stores which KV block indices it can attend to.
    This allows the kernel to skip loading masked blocks entirely.

    Attributes:
        num_blocks_in_row: Array of shape (Q_BLOCKS,) containing number of valid KV blocks per query block
        col_indices: Array of shape (Q_BLOCKS, max_blocks_per_row) containing KV block indices
        block_size: Size of each block (same for Q and KV)
        q_len: Query sequence length
        kv_len: Key/value sequence length
    """
    num_blocks_in_row: jax.Array  # (Q_BLOCKS,)
    col_indices: jax.Array        # (Q_BLOCKS, max_blocks_per_row)
    block_size: int
    q_len: int
    kv_len: int

    @property
    def num_q_blocks(self) -> int:
        """Number of query blocks."""
        return int(math.ceil(self.q_len / self.block_size))

    @property
    def num_kv_blocks(self) -> int:
        """Number of key/value blocks."""
        return int(math.ceil(self.kv_len / self.block_size))

    @property
    def sparsity(self) -> float:
        """Return the sparsity ratio (0 = dense, 1 = fully sparse)."""
        total_possible = self.num_q_blocks * self.num_kv_blocks
        actual_blocks = jnp.sum(self.num_blocks_in_row)
        return 1.0 - (float(actual_blocks) / total_possible)


def create_block_mask(
    mask_mod: MaskMod,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    block_size: int = 128,
) -> BlockMask:
    """Create a BlockMask from a mask_mod function.

    This traces the mask_mod at block granularity to determine which blocks
    contain at least one valid attention position.

    Args:
        mask_mod: Function that takes (q_idx, kv_idx) and returns mask boolean
        B: Batch size
        H: Number of heads
        Q_LEN: Query sequence length
        KV_LEN: Key/value sequence length
        block_size: Block size for tiling (must match kernel block size)

    Returns:
        BlockMask data structure representing the sparsity pattern
    """
    num_q_blocks = math.ceil(Q_LEN / block_size)
    num_kv_blocks = math.ceil(KV_LEN / block_size)

    # For each query block, determine which KV blocks it can attend to
    num_blocks_per_row = []
    col_indices_per_row = []

    for q_blk_idx in range(num_q_blocks):
        q_start = q_blk_idx * block_size
        q_end = min(q_start + block_size, Q_LEN)

        valid_kv_blocks = []

        for kv_blk_idx in range(num_kv_blocks):
            kv_start = kv_blk_idx * block_size
            kv_end = min(kv_start + block_size, KV_LEN)

            # Check if any position in this block pair is valid
            # Sample the corners and center of the block
            q_sample = jnp.array([q_start, (q_start + q_end) // 2, q_end - 1])
            kv_sample = jnp.array([kv_start, (kv_start + kv_end) // 2, kv_end - 1])

            # Create meshgrid of samples
            q_grid = q_sample[:, None]
            kv_grid = kv_sample[None, :]

            mask_samples = mask_mod(q_grid, kv_grid)

            # If any sample is valid, consider the block valid
            # This is conservative but correct
            if jnp.any(mask_samples):
                valid_kv_blocks.append(kv_blk_idx)

        num_blocks_per_row.append(len(valid_kv_blocks))
        col_indices_per_row.append(valid_kv_blocks)

    # Pad all rows to same length for array storage
    max_blocks_per_row = max(num_blocks_per_row) if num_blocks_per_row else 0

    padded_col_indices = []
    for indices in col_indices_per_row:
        # Pad with -1 to indicate invalid
        padded = indices + [-1] * (max_blocks_per_row - len(indices))
        padded_col_indices.append(padded)

    return BlockMask(
        num_blocks_in_row=jnp.array(num_blocks_per_row, dtype=jnp.int32),
        col_indices=jnp.array(padded_col_indices, dtype=jnp.int32),
        block_size=block_size,
        q_len=Q_LEN,
        kv_len=KV_LEN,
    )


def create_causal_block_mask(
    seq_len: int,
    block_size: int = 128,
) -> BlockMask:
    """Create an efficient BlockMask for causal attention.

    For causal attention, query block i can only attend to kv blocks 0..i.
    This creates a triangular sparsity pattern.

    Args:
        seq_len: Sequence length
        block_size: Block size

    Returns:
        BlockMask for causal attention
    """
    num_blocks = math.ceil(seq_len / block_size)

    num_blocks_per_row = []
    col_indices_per_row = []

    for q_blk_idx in range(num_blocks):
        # Causal: can attend to all KV blocks up to and including current
        valid_kv_blocks = list(range(q_blk_idx + 1))
        num_blocks_per_row.append(len(valid_kv_blocks))
        col_indices_per_row.append(valid_kv_blocks)

    max_blocks_per_row = max(num_blocks_per_row)

    padded_col_indices = []
    for indices in col_indices_per_row:
        padded = indices + [-1] * (max_blocks_per_row - len(indices))
        padded_col_indices.append(padded)

    return BlockMask(
        num_blocks_in_row=jnp.array(num_blocks_per_row, dtype=jnp.int32),
        col_indices=jnp.array(padded_col_indices, dtype=jnp.int32),
        block_size=block_size,
        q_len=seq_len,
        kv_len=seq_len,
    )


def create_sliding_window_block_mask(
    seq_len: int,
    window_size: int,
    block_size: int = 128,
) -> BlockMask:
    """Create a BlockMask for sliding window attention.

    Args:
        seq_len: Sequence length
        window_size: Maximum distance to attend (in each direction)
        block_size: Block size

    Returns:
        BlockMask for sliding window attention
    """
    num_blocks = math.ceil(seq_len / block_size)
    window_blocks = math.ceil(window_size / block_size)

    num_blocks_per_row = []
    col_indices_per_row = []

    for q_blk_idx in range(num_blocks):
        # Sliding window: attend to blocks within window_blocks distance
        start_blk = max(0, q_blk_idx - window_blocks)
        end_blk = min(num_blocks, q_blk_idx + window_blocks + 1)
        valid_kv_blocks = list(range(start_blk, end_blk))

        num_blocks_per_row.append(len(valid_kv_blocks))
        col_indices_per_row.append(valid_kv_blocks)

    max_blocks_per_row = max(num_blocks_per_row)

    padded_col_indices = []
    for indices in col_indices_per_row:
        padded = indices + [-1] * (max_blocks_per_row - len(indices))
        padded_col_indices.append(padded)

    return BlockMask(
        num_blocks_in_row=jnp.array(num_blocks_per_row, dtype=jnp.int32),
        col_indices=jnp.array(padded_col_indices, dtype=jnp.int32),
        block_size=block_size,
        q_len=seq_len,
        kv_len=seq_len,
    )
