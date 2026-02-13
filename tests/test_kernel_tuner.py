"""Tests for kernel tuner."""

import pytest

from pallas_flexattn import get_optimal_params, get_kernel_config


class TestKernelTuner:
    """Test kernel parameter tuning."""

    def test_get_optimal_params_short_sequence(self):
        """Test optimal params for short sequences."""
        block_r, block_c, num_warps, num_stages = get_optimal_params(256, 64)

        assert block_r <= 64
        assert block_c <= 64
        assert num_warps >= 4

    def test_get_optimal_params_long_sequence(self):
        """Test optimal params for long sequences."""
        block_r, block_c, num_warps, num_stages = get_optimal_params(4096, 64)

        assert block_r >= 128
        assert block_c >= 64
        assert num_warps >= 8

    def test_get_optimal_params_large_head_dim(self):
        """Test params are adjusted for large head dimensions."""
        block_r, block_c, num_warps, num_stages = get_optimal_params(2048, 256)

        # Large head dims need smaller blocks
        assert block_r <= 64
        assert block_c <= 64

    def test_get_optimal_params_cached(self):
        """Test that results are cached."""
        # Call twice with same params
        result1 = get_optimal_params(1024, 64)
        result2 = get_optimal_params(1024, 64)

        assert result1 == result2

    def test_get_kernel_config(self):
        """Test kernel config dictionary."""
        config = get_kernel_config(2, 8, 2048, 64, 'causal')

        assert config['batch_size'] == 2
        assert config['num_heads'] == 8
        assert config['seq_len'] == 2048
        assert config['head_dim'] == 64
        assert 'block_r' in config
        assert 'block_c' in config
        assert 'num_warps' in config
        assert 'num_stages' in config

    def test_sparsity_optimization(self):
        """Test that block-sparse masks get optimized blocks."""
        # Without sparsity
        config_dense = get_kernel_config(2, 8, 1024, 64, None)

        # With causal (sparse) mask
        config_sparse = get_kernel_config(2, 8, 1024, 64, 'causal')

        # Sparse configs should have smaller blocks for better granularity
        assert config_sparse['block_r'] <= config_dense['block_r']
