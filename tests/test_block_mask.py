"""Tests for block-sparse attention masks."""

import jax.numpy as jnp
import pytest

from pallas_flexattn.block_mask import (
    BlockMask,
    create_block_mask,
    create_causal_block_mask,
    create_sliding_window_block_mask,
)
from pallas_flexattn.mask_mod import causal_mask, bidirectional_mask, sliding_window_mask


class TestBlockMaskCreation:
    """Test BlockMask creation functions."""

    def test_create_causal_block_mask(self):
        """Test creating a causal block mask."""
        seq_len = 256
        block_size = 64

        block_mask = create_causal_block_mask(seq_len, block_size)

        assert block_mask.block_size == block_size
        assert block_mask.q_len == seq_len
        assert block_mask.kv_len == seq_len
        assert block_mask.num_q_blocks == 4  # 256 / 64

        # For causal mask, block i can attend to blocks 0..i
        # So row i should have i+1 valid blocks
        assert block_mask.num_blocks_in_row[0] == 1  # First block only attends to itself
        assert block_mask.num_blocks_in_row[1] == 2
        assert block_mask.num_blocks_in_row[2] == 3
        assert block_mask.num_blocks_in_row[3] == 4

    def test_create_sliding_window_block_mask(self):
        """Test creating a sliding window block mask."""
        seq_len = 256
        window_size = 64
        block_size = 64

        block_mask = create_sliding_window_block_mask(seq_len, window_size, block_size)

        assert block_mask.block_size == block_size
        assert block_mask.q_len == seq_len

        # Middle blocks should have 3 valid blocks (prev, current, next)
        # due to window_size = block_size
        assert block_mask.num_blocks_in_row[1] == 3
        assert block_mask.num_blocks_in_row[2] == 3

    def test_create_block_mask_from_mask_mod(self):
        """Test creating BlockMask from a mask_mod function."""
        B, H, T = 2, 4, 256
        block_size = 64

        block_mask = create_block_mask(
            mask_mod=causal_mask,
            B=B,
            H=H,
            Q_LEN=T,
            KV_LEN=T,
            block_size=block_size,
        )

        assert block_mask.block_size == block_size
        assert block_mask.q_len == T
        assert block_mask.num_q_blocks == 4

        # Check sparsity - causal mask should have triangular pattern
        # For 4 blocks: (4*3/2) / 16 = 6/16 = 0.375, approaching 0.5 for large N
        assert 0.3 < block_mask.sparsity < 0.5

    def test_block_mask_sparsity_dense(self):
        """Test sparsity calculation for dense mask."""
        T = 256
        block_size = 64

        # Bidirectional attention has 0% sparsity (fully dense)
        block_mask = create_block_mask(
            mask_mod=bidirectional_mask,
            B=1,
            H=1,
            Q_LEN=T,
            KV_LEN=T,
            block_size=block_size,
        )

        assert block_mask.sparsity == 0.0

    def test_block_mask_sparsity_causal(self):
        """Test sparsity calculation for causal mask."""
        T = 1024
        block_size = 128

        block_mask = create_causal_block_mask(T, block_size)

        # For causal: sparsity = 1 - (num_triangular / num_total)
        # num_triangular = n*(n+1)/2 where n = num_blocks
        num_blocks = T // block_size
        triangular = num_blocks * (num_blocks + 1) / 2
        total = num_blocks * num_blocks
        expected_sparsity = 1.0 - (triangular / total)

        assert abs(block_mask.sparsity - expected_sparsity) < 0.01


class TestBlockMaskStructure:
    """Test BlockMask internal structure."""

    def test_col_indices_shape(self):
        """Test that col_indices has correct shape."""
        T = 256
        block_size = 64

        block_mask = create_causal_block_mask(T, block_size)

        # Should be (num_q_blocks, max_blocks_per_row)
        assert block_mask.col_indices.shape[0] == 4
        assert block_mask.col_indices.shape[1] == 4  # max for causal

    def test_col_indices_values_causal(self):
        """Test col_indices values for causal mask."""
        T = 256  # 4 blocks
        block_size = 64

        block_mask = create_causal_block_mask(T, block_size)

        # Convert to numpy for easier testing
        col_indices = block_mask.col_indices

        # Row 0: can attend to block 0 only
        assert col_indices[0, 0] == 0
        assert col_indices[0, 1] == -1  # padded

        # Row 2: can attend to blocks 0, 1, 2
        assert col_indices[2, 0] == 0
        assert col_indices[2, 1] == 1
        assert col_indices[2, 2] == 2
        assert col_indices[2, 3] == -1  # padded

    def test_num_blocks_in_row_dtype(self):
        """Test that num_blocks_in_row has correct dtype."""
        T = 128
        block_size = 64

        block_mask = create_causal_block_mask(T, block_size)

        assert block_mask.num_blocks_in_row.dtype == jnp.int32
        assert block_mask.col_indices.dtype == jnp.int32


class TestBlockMaskEdgeCases:
    """Test edge cases for BlockMask."""

    def test_non_power_of_two_seq_len(self):
        """Test with sequence length that's not a multiple of block_size."""
        seq_len = 300  # Not a multiple of 64
        block_size = 64

        block_mask = create_causal_block_mask(seq_len, block_size)

        # Should round up: ceil(300/64) = 5 blocks
        assert block_mask.num_q_blocks == 5

    def test_very_small_block_size(self):
        """Test with very small block size."""
        seq_len = 64
        block_size = 16

        block_mask = create_causal_block_mask(seq_len, block_size)

        assert block_mask.num_q_blocks == 4

    def test_single_block(self):
        """Test when entire sequence fits in one block."""
        seq_len = 64
        block_size = 128

        block_mask = create_causal_block_mask(seq_len, block_size)

        assert block_mask.num_q_blocks == 1
        assert block_mask.num_blocks_in_row[0] == 1
