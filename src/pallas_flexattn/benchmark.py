"""Performance benchmark for Pallas FlexAttention.

Compares performance against reference implementations:
- mha_reference: Materialized attention (O(N²) memory)
- cudnn_attention: JAX's cuDNN flash attention
- flash_attn_jax: C++ CUDA implementation (optional)
- Our flash_attention: Pallas implementation
"""
import time
import warnings
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from pallas_flexattn import (
    flash_attention,
    flash_attention_fwd,
    mha_reference,
    causal_mask,
)
from pallas_flexattn.kernel_tuner import get_optimal_params


# Optional: flash_attn_jax
try:
    from flash_attn_jax import flash_mha
    HAS_FLASH_ATTN_JAX = True
except ImportError:
    HAS_FLASH_ATTN_JAX = False
    flash_mha = None


def benchmark(
    fn: Callable,
    *args,
    warmup: int = 3,
    iters: int = 20,
) -> float:
    """Benchmark a function with warmup.

    Args:
        fn: Function to benchmark
        *args: Arguments to pass to function
        warmup: Number of warmup iterations
        iters: Number of benchmark iterations

    Returns:
        Median execution time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    return np.median(times) * 1000  # Convert to ms


def cudnn_attention_wrapper(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    is_causal: bool = True,
) -> Optional[jax.Array]:
    """JAX cuDNN flash attention wrapper.

    Args:
        q: Query tensor (B, H, T, D)
        k: Key tensor (B, H, T, D)
        v: Value tensor (B, H, T, D)
        is_causal: Whether to use causal masking

    Returns:
        Output tensor (B, H, T, D) or None if cuDNN not available
    """
    try:
        # Transpose from (B, H, T, D) to (B, T, H, D) for jax.nn.dot_product_attention
        q_t = jnp.transpose(q, (0, 2, 1, 3))
        k_t = jnp.transpose(k, (0, 2, 1, 3))
        v_t = jnp.transpose(v, (0, 2, 1, 3))

        out = jax.nn.dot_product_attention(
            q_t, k_t, v_t,
            is_causal=is_causal,
            implementation="cudnn",
        )
        return jnp.transpose(out, (0, 2, 1, 3))  # Back to (B, H, T, D)
    except RuntimeError:
        # cuDNN not available
        return None


def flash_attn_jax_wrapper(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    is_causal: bool = True,
) -> Optional[jax.Array]:
    """flash_attn_jax wrapper.

    Args:
        q: Query tensor (B, H, T, D)
        k: Key tensor (B, H, T, D)
        v: Value tensor (B, H, T, D)
        is_causal: Whether to use causal masking

    Returns:
        Output tensor (B, H, T, D) or None if not available
    """
    if not HAS_FLASH_ATTN_JAX or flash_mha is None:
        return None

    # flash_attn_jax expects (n, l, h, d) format
    # Our format is (B, H, T, D), so transpose axes 1 and 2
    q_t = jnp.transpose(q, (0, 2, 1, 3))
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))

    out = flash_mha(q_t, k_t, v_t, is_causal=is_causal)
    return jnp.transpose(out, (0, 2, 1, 3))


def run_benchmark(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 2048,
    head_dim: int = 64,
    warmup: int = 3,
    iters: int = 20,
    skip_reference: bool = False,
    interpret: bool = False,
) -> dict:
    """Run comprehensive benchmark.

    Args:
        batch_size: Batch size (B)
        num_heads: Number of heads (H)
        seq_len: Sequence length (T)
        head_dim: Head dimension (D)
        warmup: Number of warmup iterations
        iters: Number of benchmark iterations
        skip_reference: Skip reference (materialized) benchmark for long sequences

    Returns:
        Dictionary with benchmark results
    """
    B, H, T, D = batch_size, num_heads, seq_len, head_dim
    dtype = jnp.float32

    print("=" * 70)
    print("Pallas FlexAttention Performance Benchmark")
    print("=" * 70)
    print(f"Configuration: B={B}, H={H}, T={T}, D={D}, dtype={dtype}")
    print()

    # Generate test data
    key = jax.random.key(0)
    keys = jax.random.split(key, 5)

    q = jax.random.normal(keys[0], (B, H, T, D), dtype=dtype)
    k = jax.random.normal(keys[1], (B, H, T, D), dtype=dtype)
    v = jax.random.normal(keys[2], (B, H, T, D), dtype=dtype)
    do = jax.random.normal(keys[3], (B, H, T, D), dtype=dtype)

    results = {
        "config": {"B": B, "H": H, "T": T, "D": D, "dtype": str(dtype)},
        "forward": {},
        "backward": {},
    }

    # Get optimal params for our implementation
    block_r, block_c, num_warps, num_stages = get_optimal_params(T, D)
    print(f"Optimal kernel params: block_r={block_r}, block_c={block_c}, "
          f"num_warps={num_warps}, num_stages={num_stages}")
    print()

    # ===================== Forward Pass =====================
    print("Forward Pass:")
    print("-" * 40)

    # Reference (materialized) - skip for very long sequences
    if not skip_reference:
        ref_fwd = jax.jit(lambda q, k, v: mha_reference(q, k, v, causal_mask))
        _ = ref_fwd(q, k, v).block_until_ready()
        t_ref_fwd = benchmark(ref_fwd, q, k, v, warmup=warmup, iters=iters)
        results["forward"]["reference"] = t_ref_fwd
        print(f"  Reference (materialized):  {t_ref_fwd:8.3f} ms")
    else:
        print(f"  Reference (materialized):  skipped (T={T} too large)")

    # cuDNN attention
    cudnn_fwd = jax.jit(lambda q, k, v: cudnn_attention_wrapper(q, k, v, is_causal=True))
    cudnn_result = cudnn_fwd(q, k, v)
    if cudnn_result is not None:
        cudnn_result.block_until_ready()
        t_cudnn_fwd = benchmark(cudnn_fwd, q, k, v, warmup=warmup, iters=iters)
        results["forward"]["cudnn"] = t_cudnn_fwd
        print(f"  JAX cuDNN attention:       {t_cudnn_fwd:8.3f} ms")
    else:
        print(f"  JAX cuDNN attention:       not available (no GPU/CUDA)")

    # flash_attn_jax (optional)
    if HAS_FLASH_ATTN_JAX:
        flash_jax_fwd = jax.jit(lambda q, k, v: flash_attn_jax_wrapper(q, k, v, is_causal=True))
        _ = flash_jax_fwd(q, k, v).block_until_ready()
        t_flash_jax_fwd = benchmark(flash_jax_fwd, q, k, v, warmup=warmup, iters=iters)
        results["forward"]["flash_attn_jax"] = t_flash_jax_fwd
        print(f"  flash_attn_jax (C++):      {t_flash_jax_fwd:8.3f} ms")
    else:
        print(f"  flash_attn_jax (C++):      not installed")

    # Our Pallas implementation
    our_fwd = jax.jit(lambda q, k, v: flash_attention(
        q, k, v,
        mask_mod=causal_mask,
        block_r=block_r,
        block_c=block_c,
        num_warps=num_warps,
        num_stages=num_stages,
        interpret=interpret,
    ))
    _ = our_fwd(q, k, v).block_until_ready()
    t_our_fwd = benchmark(our_fwd, q, k, v, warmup=warmup, iters=iters)
    results["forward"]["pallas_flexattn"] = t_our_fwd
    print(f"  Pallas FlexAttention:      {t_our_fwd:8.3f} ms")
    print()

    # ===================== Backward Pass =====================
    print("Backward Pass:")
    print("-" * 40)

    # Reference backward
    if not skip_reference:
        def loss_ref(q, k, v):
            out = mha_reference(q, k, v, causal_mask)
            return jnp.sum(out * do)

        grad_fn_ref = jax.grad(loss_ref, argnums=(0, 1, 2))
        _ = grad_fn_ref(q, k, v)
        t_ref_bwd = benchmark(grad_fn_ref, q, k, v, warmup=warmup, iters=iters)
        results["backward"]["reference"] = t_ref_bwd
        print(f"  Reference (materialized):  {t_ref_bwd:8.3f} ms")
    else:
        print(f"  Reference (materialized):  skipped (T={T} too large)")

    # cuDNN backward
    cudnn_result = cudnn_fwd(q, k, v)
    if cudnn_result is not None:
        def loss_cudnn(q, k, v):
            out = cudnn_attention_wrapper(q, k, v, is_causal=True)
            return jnp.sum(out * do)

        grad_fn_cudnn = jax.grad(loss_cudnn, argnums=(0, 1, 2))
        _ = grad_fn_cudnn(q, k, v)
        t_cudnn_bwd = benchmark(grad_fn_cudnn, q, k, v, warmup=warmup, iters=iters)
        results["backward"]["cudnn"] = t_cudnn_bwd
        print(f"  JAX cuDNN attention:       {t_cudnn_bwd:8.3f} ms")
    else:
        print(f"  JAX cuDNN attention:       not available (no GPU/CUDA)")

    # flash_attn_jax backward
    if HAS_FLASH_ATTN_JAX:
        def loss_flash_jax(q, k, v):
            out = flash_attn_jax_wrapper(q, k, v, is_causal=True)
            return jnp.sum(out * do)

        grad_fn_flash_jax = jax.grad(loss_flash_jax, argnums=(0, 1, 2))
        _ = grad_fn_flash_jax(q, k, v)
        t_flash_jax_bwd = benchmark(grad_fn_flash_jax, q, k, v, warmup=warmup, iters=iters)
        results["backward"]["flash_attn_jax"] = t_flash_jax_bwd
        print(f"  flash_attn_jax (C++):      {t_flash_jax_bwd:8.3f} ms")
    else:
        print(f"  flash_attn_jax (C++):      not installed")

    # Our backward
    def loss_our(q, k, v):
        out = flash_attention(
            q, k, v,
            mask_mod=causal_mask,
            block_r=block_r,
            block_c=block_c,
            num_warps=num_warps,
            num_stages=num_stages,
            interpret=interpret,
        )
        return jnp.sum(out * do)

    grad_fn_our = jax.grad(loss_our, argnums=(0, 1, 2))
    _ = grad_fn_our(q, k, v)
    t_our_bwd = benchmark(grad_fn_our, q, k, v, warmup=warmup, iters=iters)
    results["backward"]["pallas_flexattn"] = t_our_bwd
    print(f"  Pallas FlexAttention:      {t_our_bwd:8.3f} ms")
    print()

    # ===================== Summary =====================
    print("=" * 70)
    print("Summary (Forward + Backward)")
    print("=" * 70)

    if "cudnn" in results["forward"] and "pallas_flexattn" in results["forward"]:
        fwd_speedup = results["forward"]["cudnn"] / results["forward"]["pallas_flexattn"]
        bwd_speedup = results["backward"]["cudnn"] / results["backward"]["pallas_flexattn"]
        total_cudnn = results["forward"]["cudnn"] + results["backward"]["cudnn"]
        total_our = results["forward"]["pallas_flexattn"] + results["backward"]["pallas_flexattn"]
        total_speedup = total_cudnn / total_our

        print(f"cuDNN:     {total_cudnn:8.3f} ms (fwd: {results['forward']['cudnn']:.3f}, bwd: {results['backward']['cudnn']:.3f})")
        print(f"Pallas:    {total_our:8.3f} ms (fwd: {results['forward']['pallas_flexattn']:.3f}, bwd: {results['backward']['pallas_flexattn']:.3f})")
        print(f"Speedup:   {total_speedup:8.2f}x (fwd: {fwd_speedup:.2f}x, bwd: {bwd_speedup:.2f}x)")

    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Pallas FlexAttention performance"
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
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip reference (materialized) benchmark for long sequences",
    )
    parser.add_argument(
        "--interpret",
        action="store_true",
        help="Use Pallas interpret mode (required for CPU)",
    )

    args = parser.parse_args()

    # Warn if using CPU
    devices = jax.devices()
    if devices[0].platform == "cpu":
        warnings.warn(
            "Benchmarking on CPU. For meaningful performance results, "
            "run on GPU with JAX CUDA support.",
            UserWarning,
        )

    # Auto-enable interpret mode on CPU
    devices = jax.devices()
    interpret = args.interpret or (devices[0].platform == "cpu")

    run_benchmark(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        warmup=args.warmup,
        iters=args.iters,
        skip_reference=args.skip_reference or args.seq_len > 1024,
        interpret=interpret,
    )
