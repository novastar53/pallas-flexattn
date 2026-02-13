"""Unit tests for Pallas Flash Attention."""

import jax
import jax.numpy as jnp
import pytest

from pallas_flexattn import (
    flash_attention,
    flash_attention_fwd,
    mha_reference,
    causal_mask,
    bidirectional_mask,
    sliding_window_mask,
)


@pytest.fixture
def rng_key():
    """Provide a random key for tests."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def small_attn_inputs(rng_key):
    """Create small attention inputs for testing."""
    B, H, T, D = 2, 4, 128, 64
    keys = jax.random.split(rng_key, 4)
    q = jax.random.normal(keys[0], (B, H, T, D), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (B, H, T, D), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (B, H, T, D), dtype=jnp.float32)
    return q, k, v


class TestFlashAttentionForward:
    """Test flash attention forward pass."""

    def test_output_shape(self, small_attn_inputs):
        """Test that output has correct shape."""
        q, k, v = small_attn_inputs
        out = flash_attention(q, k, v, interpret=True)
        assert out.shape == q.shape

    def test_causal_matches_reference(self, small_attn_inputs):
        """Test causal flash attention matches reference implementation."""
        q, k, v = small_attn_inputs

        out_flash, _ = flash_attention_fwd(q, k, v, mask_mod=causal_mask, interpret=True)
        out_ref = mha_reference(q, k, v, mask_mod=causal_mask)

        assert jnp.allclose(out_flash, out_ref, atol=1e-2, rtol=1e-2)

    def test_bidirectional_matches_reference(self, small_attn_inputs):
        """Test bidirectional flash attention matches reference."""
        q, k, v = small_attn_inputs

        out_flash, _ = flash_attention_fwd(q, k, v, mask_mod=bidirectional_mask, interpret=True)
        out_ref = mha_reference(q, k, v, mask_mod=bidirectional_mask)

        assert jnp.allclose(out_flash, out_ref, atol=1e-2, rtol=1e-2)

    def test_sliding_window(self, small_attn_inputs):
        """Test sliding window attention."""
        q, k, v = small_attn_inputs
        window_mask = sliding_window_mask(32, 32)

        out_flash, _ = flash_attention_fwd(
            q, k, v, mask_mod=window_mask, interpret=True
        )
        out_ref = mha_reference(q, k, v, mask_mod=window_mask)

        assert jnp.allclose(out_flash, out_ref, atol=1e-2, rtol=1e-2)

    def test_different_batch_sizes(self, rng_key):
        """Test with different batch sizes."""
        for B in [1, 2, 4]:
            keys = jax.random.split(rng_key, 4)
            q = jax.random.normal(keys[0], (B, 4, 64, 32), dtype=jnp.float32)
            k = jax.random.normal(keys[1], (B, 4, 64, 32), dtype=jnp.float32)
            v = jax.random.normal(keys[2], (B, 4, 64, 32), dtype=jnp.float32)

            out_flash, _ = flash_attention_fwd(q, k, v, interpret=True)
            out_ref = mha_reference(q, k, v)

            assert jnp.allclose(out_flash, out_ref, atol=1e-2, rtol=1e-2)

    def test_different_head_dims(self, rng_key):
        """Test with different head dimensions."""
        for D in [32, 64, 128]:
            keys = jax.random.split(rng_key, 4)
            q = jax.random.normal(keys[0], (2, 4, 64, D), dtype=jnp.float32)
            k = jax.random.normal(keys[1], (2, 4, 64, D), dtype=jnp.float32)
            v = jax.random.normal(keys[2], (2, 4, 64, D), dtype=jnp.float32)

            out_flash, _ = flash_attention_fwd(q, k, v, interpret=True)
            out_ref = mha_reference(q, k, v)

            assert jnp.allclose(out_flash, out_ref, atol=1e-2, rtol=1e-2)


class TestFlashAttentionBackward:
    """Test flash attention backward pass."""

    def test_gradient_shapes(self, small_attn_inputs):
        """Test that gradients have correct shapes."""
        q, k, v = small_attn_inputs

        def loss(q, k, v):
            out = flash_attention(q, k, v, interpret=True)
            return jnp.sum(out ** 2)

        grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)

        assert grads[0].shape == q.shape  # dQ
        assert grads[1].shape == k.shape  # dK
        assert grads[2].shape == v.shape  # dV

    def test_gradients_match_reference(self, small_attn_inputs):
        """Test that gradients match reference implementation."""
        q, k, v = small_attn_inputs
        do = jax.random.normal(jax.random.PRNGKey(1), q.shape, dtype=jnp.float32)

        def loss_ref(q, k, v):
            out = mha_reference(q, k, v, mask_mod=causal_mask)
            return jnp.sum(out * do)

        def loss_flash(q, k, v):
            out = flash_attention(q, k, v, mask_mod=causal_mask, interpret=True)
            return jnp.sum(out * do)

        dq_ref, dk_ref, dv_ref = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
        dq_flash, dk_flash, dv_flash = jax.grad(loss_flash, argnums=(0, 1, 2))(q, k, v)

        assert jnp.allclose(dq_flash, dq_ref, atol=1e-1, rtol=1e-2)
        assert jnp.allclose(dk_flash, dk_ref, atol=1e-1, rtol=1e-2)
        assert jnp.allclose(dv_flash, dv_ref, atol=1e-1, rtol=1e-2)

    def test_causal_backward(self, small_attn_inputs):
        """Test backward pass with causal masking."""
        q, k, v = small_attn_inputs

        def loss(q, k, v):
            out = flash_attention(q, k, v, mask_mod=causal_mask, interpret=True)
            return jnp.sum(out ** 2)

        # Should not raise any errors
        grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
        assert all(g is not None for g in grads)

    def test_sliding_window_backward(self, small_attn_inputs):
        """Test backward pass with sliding window."""
        q, k, v = small_attn_inputs
        window_mask = sliding_window_mask(32, 32)

        def loss(q, k, v):
            out = flash_attention(
                q, k, v, mask_mod=window_mask, interpret=True
            )
            return jnp.sum(out ** 2)

        # Should not raise any errors
        grads = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
        assert all(g is not None for g in grads)


class TestFlashAttentionNumericStability:
    """Test numeric stability of flash attention."""

    def test_no_nans_in_output(self, small_attn_inputs):
        """Test that output doesn't contain NaNs."""
        q, k, v = small_attn_inputs
        out = flash_attention(q, k, v, interpret=True)
        assert not jnp.any(jnp.isnan(out))

    def test_no_infs_in_output(self, small_attn_inputs):
        """Test that output doesn't contain infinities."""
        q, k, v = small_attn_inputs
        out = flash_attention(q, k, v, interpret=True)
        assert not jnp.any(jnp.isinf(out))

    def test_large_sequence_length(self, rng_key):
        """Test with larger sequence length."""
        B, H, T, D = 1, 2, 512, 64
        keys = jax.random.split(rng_key, 4)
        q = jax.random.normal(keys[0], (B, H, T, D), dtype=jnp.float32)
        k = jax.random.normal(keys[1], (B, H, T, D), dtype=jnp.float32)
        v = jax.random.normal(keys[2], (B, H, T, D), dtype=jnp.float32)

        out_flash, _ = flash_attention_fwd(q, k, v, interpret=True)
        out_ref = mha_reference(q, k, v)

        assert jnp.allclose(out_flash, out_ref, atol=1e-2, rtol=1e-2)
