"""Kernel tuner for Pallas FlexAttention.

Provides optimal kernel hyperparameters for given sequence lengths and head dimensions.
"""
from typing import Tuple, Dict, Optional
import functools

from pallas_flexattn.flash_attn import DEFAULT_BLOCK_R, DEFAULT_BLOCK_C, DEFAULT_NUM_WARPS, DEFAULT_NUM_STAGES


@functools.lru_cache(maxsize=128)
def get_optimal_params(
    seq_len: int,
    head_dim: int = 64,
    use_block_sparsity: bool = False,
) -> Tuple[int, int, int, int]:
    """Get optimal kernel parameters for sequence length.

    Args:
        seq_len: Sequence length (T dimension)
        head_dim: Head dimension (D)
        use_block_sparsity: Whether using block-sparse attention (affects block size selection)

    Returns:
        tuple: (block_r, block_c, num_warps, num_stages)
    """
    # Base configuration
    if seq_len <= 256:
        # Very short sequences: small blocks
        block_r, block_c = 32, 32
        num_warps, num_stages = 4, 2
    elif seq_len <= 512:
        # Short sequences: default small blocks
        block_r, block_c = 64, 64
        num_warps, num_stages = 4, 3
    elif seq_len <= 2048:
        # Medium sequences: taller blocks for fewer kernel launches
        block_r, block_c = 128, 64
        num_warps, num_stages = 4, 3
    elif seq_len <= 4096:
        # Long sequences: max blocks, more warps for utilization
        block_r, block_c = 128, 128
        num_warps, num_stages = 8, 4
    else:
        # Very long sequences: max blocks, extra pipeline stage
        block_r, block_c = 128, 128
        num_warps, num_stages = 8, 5

    # Adjust for head dimension
    if head_dim > 128:
        # Larger head dims need smaller blocks to fit in SMEM
        block_r = min(block_r, 64)
        block_c = min(block_c, 64)
        num_stages = min(num_stages, 3)
    elif head_dim <= 32:
        # Small head dims can use larger blocks
        num_stages = max(num_stages, 4)

    # Block sparsity adjustments
    if use_block_sparsity:
        # For sparse attention, smaller blocks give better sparsity granularity
        block_r = min(block_r, 64)
        block_c = min(block_c, 64)

    return block_r, block_c, num_warps, num_stages


def get_kernel_config(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    mask_type: Optional[str] = None,
) -> Dict[str, int]:
    """Get complete kernel configuration dictionary.

    Args:
        batch_size: Batch size (B)
        num_heads: Number of heads (H)
        seq_len: Sequence length (T)
        head_dim: Head dimension (D)
        mask_type: Type of mask ('causal', 'sliding_window', 'bidirectional', None)

    Returns:
        Dictionary with kernel parameters
    """
    use_block_sparsity = mask_type in ('causal', 'sliding_window')
    block_r, block_c, num_warps, num_stages = get_optimal_params(
        seq_len, head_dim, use_block_sparsity
    )

    return {
        'batch_size': batch_size,
        'num_heads': num_heads,
        'seq_len': seq_len,
        'head_dim': head_dim,
        'block_r': block_r,
        'block_c': block_c,
        'num_warps': num_warps,
        'num_stages': num_stages,
    }


def print_kernel_config(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    mask_type: Optional[str] = None,
):
    """Print optimal kernel configuration.

    Args:
        batch_size: Batch size (B)
        num_heads: Number of heads (H)
        seq_len: Sequence length (T)
        head_dim: Head dimension (D)
        mask_type: Type of mask ('causal', 'sliding_window', 'bidirectional', None)
    """
    config = get_kernel_config(batch_size, num_heads, seq_len, head_dim, mask_type)

    print("=" * 70)
    print("Pallas FlexAttention Kernel Configuration")
    print("=" * 70)
    print(f"Input Configuration:")
    print(f"  Batch size (B):   {config['batch_size']}")
    print(f"  Num heads (H):    {config['num_heads']}")
    print(f"  Seq length (T):   {config['seq_len']}")
    print(f"  Head dim (D):     {config['head_dim']}")
    if mask_type:
        print(f"  Mask type:        {mask_type}")
    print()
    print(f"Kernel Parameters:")
    print(f"  BLOCK_R (block_r):      {config['block_r']}")
    print(f"  BLOCK_C (block_c):      {config['block_c']}")
    print(f"  NUM_WARPS:              {config['num_warps']}")
    print(f"  NUM_STAGES:             {config['num_stages']}")
    print()
    print(f"Usage:")
    print(f"  from pallas_flexattn import flash_attention")
    print(f"  out = flash_attention(")
    print(f"      q, k, v,")
    print(f"      block_r={config['block_r']},")
    print(f"      block_c={config['block_c']},")
    print(f"      num_warps={config['num_warps']},")
    print(f"      num_stages={config['num_stages']},")
    print(f"  )")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Get optimal kernel parameters for Pallas FlexAttention"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (B)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of heads (H)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length (T)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=64,
        help="Head dimension (D)",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default=None,
        choices=['causal', 'sliding_window', 'bidirectional'],
        help="Type of attention mask",
    )

    args = parser.parse_args()

    print_kernel_config(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        mask_type=args.mask_type,
    )
