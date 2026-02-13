# Pallas FlexAttention

A FlexAttention implementation in JAX/Pallas, inspired by PyTorch's FlexAttention.

## Overview

This repository contains a from-scratch implementation of Flash Attention with FlexAttention features:
- Custom score modification functions
- Block-sparse attention masks
- Autotuning for optimal kernel configurations
- Support for causal, sliding window, and arbitrary attention patterns

## Installation

```bash
# CPU only
uv sync

# With GPU support
uv sync --extra gpu

# Development dependencies
uv sync --extra dev
```

## Running Tests

```bash
uv run pytest
```

## Structure

```
src/pallas_flexattn/
├── flash_attn.py      # Core flash attention kernel
├── block_mask.py      # Block-sparse mask utilities
├── score_mod.py       # Score modification functions
└── autotune.py        # Kernel autotuning

tests/
├── test_flash_attn.py     # Unit tests for flash attention
└── test_block_mask.py     # Tests for block-sparse masks
```

## Roadmap

See [Roadmap.md](Roadmap.md) for the step-by-step plan to transform this into a full FlexAttention implementation.
