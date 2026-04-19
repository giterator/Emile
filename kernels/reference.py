"""
PyTorch reference attention implementation.
This is the correctness oracle — every Triton kernel is validated against this.
Uses full float32 accumulation to avoid fp16 rounding errors in the reference.
"""
import math
import torch


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention in pure PyTorch.
    q, k, v: (B, H, N, D) float16
    returns:  (B, H, N, D) float16
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # Accumulate in float32 for numerical stability
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    if is_causal:
        N = q.shape[-2]
        # tril = positions we KEEP (token i attends to 0..i); mask everything above
        keep_mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~keep_mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v.float())
    return out.to(q.dtype)
