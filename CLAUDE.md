# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pallas FlexAttention is a JAX/Pallas implementation of FlexAttention (inspired by PyTorch's FlexAttention). It implements Flash Attention with custom score modification functions, block-sparse attention masks, and kernel autotuning.

## Development Commands

Use `uv` for all Python operations:

```bash
# Install dependencies
uv sync

# With GPU support (CUDA 12)
uv sync --extra gpu

# With development tools (pytest, ruff, mypy)
uv sync --extra dev

# Run all tests (uses interpret mode for CPU execution)
uv run pytest

# Run specific test file
uv run pytest tests/test_flash_attn.py -v

# Run single test
uv run pytest tests/test_flash_attn.py::TestFlashAttentionForward::test_causal_matches_reference -v

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy src/
```

## Architecture

### Core Kernel Structure

The Flash Attention implementation in `src/pallas_flexattn/flash_attn.py` uses a three-kernel backward pass design:

1. **Forward kernel** (`flash_attention_fwd_kernel`): Computes attention output using online softmax. Iterates over KV blocks in the outer loop, applying causal or sliding window masks per-block.

2. **Backward pass** (split into 3 kernels for memory efficiency):
   - **Preprocess** (`flash_attention_bwd_preprocess_kernel`): Computes D = rowsum(O ⊙ dO)
   - **dK/dV kernel** (`flash_attention_bwd_dkv_kernel`): Computes gradients for K and V
   - **dQ kernel** (`flash_attention_bwd_dq_kernel`): Computes gradient for Q

The `flash_attention` function is wrapped with `@jax.custom_vjp` to enable the custom backward pass.

### Pallas Kernel Grid Layout

The kernels use a 2D grid `(B*H, pl.cdiv(T, BLOCK_R))` where:
- Axis 0: Flattened batch × heads dimension
- Axis 1: Query sequence blocks

KV sequence is iterated over in the kernel body using `jax.lax.fori_loop`.

### Block Masking and Sparsity

The current implementation supports:
- **Causal masking**: Skips blocks where `kv_idx > q_idx` using `jax.lax.cond`
- **Sliding window**: Configurable left/right window sizes; skips blocks entirely outside the window

The roadmap includes a `BlockMask` data structure for efficient block-sparse attention (see Roadmap.md Phase 1.2).

### Testing on CPU

All tests use `interpret=True` mode which runs Pallas kernels on CPU via the JAX interpreter. This allows testing without GPU:

```python
out = flash_attention(q, k, v, interpret=True)
```

### Kernel Configuration

Kernel tuning parameters are exposed at the API level:
- `block_r` (BLOCK_M): Query block size (default 64)
- `block_c` (BLOCK_N): KV block size (default 64)
- `num_warps`: GPU warps per block (default 4)
- `num_stages`: Pipeline stages for async loading (default 3)

Tests validate against `mha_reference()` which implements standard attention materializing the full N×N matrix.

## Code Organization

```
src/pallas_flexattn/
├── flash_attn.py      # Core kernels and public API
└── __init__.py        # Exports flash_attention, flash_attention_fwd, etc.

tests/
├── test_flash_attn.py # Forward/backward correctness tests
└── test_block_mask.py # Placeholder tests for BlockMask
```

## Key Implementation Details

1. **Numerical stability**: Uses online softmax with running max and sum tracking (standard Flash Attention approach)

2. **Block skipping**: For causal and sliding window attention, blocks entirely outside the valid region are skipped with `jax.lax.cond` rather than masked, reducing computation

3. **Precision**: Matrix multiplications use `precision='float32'` internally even for lower-precision inputs

4. **Reference tolerance**: Tests allow `atol=1e-2` for forward pass and `atol=1e-1` for gradients due to numerical differences between Flash Attention and materialized attention

## Roadmap

See `Roadmap.md` for detailed implementation phases. Current status is Phase 1 (core refactoring) with planned work on:
- Modular score modification system
- BlockMask data structure for block-sparse attention
- Autotuning infrastructure
- GQA (Grouped Query Attention) optimizations
- Decode-optimized kernel for inference
