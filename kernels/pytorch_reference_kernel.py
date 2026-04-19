"""
PyTorch reference attention -- the baseline the Triton agent must beat.

This is a plain PyTorch implementation of scaled dot-product attention
using explicit matmul + softmax. It runs correctly on CUDA but is not
memory-fused: it materialises the full N x N attention matrix in HBM and
makes three separate passes (QK matmul, softmax, AV matmul), operating far
below the A100 memory bandwidth ceiling.

The agent's task: write a fused Triton kernel (FlashAttention-style) that
beats this reference in TFLOPS by processing Q/K/V in tiles and keeping
the running softmax statistics in SRAM instead of spilling them to HBM.
"""
import math
import torch


def attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention in pure PyTorch (unfused reference baseline).

    q, k, v : (B, H, N, D) float16 on CUDA
    returns  : (B, H, N, D) float16

    Performance characteristics:
    - Three separate HBM passes (QK^T matmul, softmax, AV matmul)
    - Full N x N attention matrix materialised in HBM
    - Memory-bandwidth limited at all practical sequence lengths
    - Typical throughput: 1-5 TFLOPS on A100 for standard transformer shapes

    This is the correctness oracle AND the performance floor. Any Triton
    kernel that passes the correctness check and exceeds these TFLOPS wins.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # Accumulate in float32 for numerical stability; return in input dtype
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    if is_causal:
        N = q.shape[-2]
        # Keep lower triangle (token i attends to positions 0..i only)
        keep_mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~keep_mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v.float())
    return out.to(q.dtype)
