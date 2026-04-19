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
    Scaled dot-product attention in pure PyTorch (baseline).

    q, k, v : (B, H, N, D) float16 on CUDA
    returns  : (B, H, N, D) float16

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
