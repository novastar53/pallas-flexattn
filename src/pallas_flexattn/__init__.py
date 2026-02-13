"""Pallas FlexAttention - A JAX/Pallas implementation of FlexAttention."""

from pallas_flexattn.flash_attn import (
    flash_attention,
    flash_attention_fwd,
    flash_attention_bwd,
    mha_reference,
)
from pallas_flexattn.mask_mod import (
    MaskMod,
    causal_mask,
    bidirectional_mask,
    sliding_window_mask,
)
from pallas_flexattn.block_mask import (
    BlockMask,
    create_block_mask,
    create_causal_block_mask,
    create_sliding_window_block_mask,
)
from pallas_flexattn.kernel_tuner import (
    get_optimal_params,
    get_kernel_config,
    print_kernel_config,
)

# Benchmark functions (available but not imported by default to avoid heavy dependencies)
# from pallas_flexattn.benchmark import run_benchmark, benchmark

__all__ = [
    "flash_attention",
    "flash_attention_fwd",
    "flash_attention_bwd",
    "mha_reference",
    "MaskMod",
    "causal_mask",
    "bidirectional_mask",
    "sliding_window_mask",
    "BlockMask",
    "create_block_mask",
    "create_causal_block_mask",
    "create_sliding_window_block_mask",
    "get_optimal_params",
    "get_kernel_config",
    "print_kernel_config",
]
