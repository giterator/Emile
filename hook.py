"""
SDPA monkey-patch — hooks the optimized Triton kernel into any HuggingFace model
that uses torch.nn.functional.scaled_dot_product_attention internally.

Handles GQA transparently: Qwen3-4B uses 32 query heads / 8 KV heads (4:1 ratio).
The optimized Triton kernel is pure MHA; GQA expansion happens here in the hook.

Usage:
    from hook import patch_attention, unpatch_attention, load_best_kernel

    load_best_kernel(kernel_code_str)   # call once after agent finishes
    patch_attention()                    # replaces F.scaled_dot_product_attention
    model.generate(...)
    unpatch_attention()                  # restore original
"""
import math
import torch
import torch.nn.functional as F

_original_sdpa = F.scaled_dot_product_attention
_optimized_kernel = None   # set by load_best_kernel()


def load_best_kernel(kernel_code: str) -> None:
    """
    Exec the agent's best-found kernel code and store the callable.
    kernel_code must define attention_kernel(q, k, v, is_causal, scale).
    """
    global _optimized_kernel

    import triton
    import triton.language as tl

    exec_globals = {
        "torch":  torch,
        "triton": triton,
        "tl":     tl,
        "math":   math,
        "__builtins__": __builtins__,
    }
    exec(kernel_code, exec_globals)
    fn = exec_globals.get("attention_kernel")
    if fn is None:
        raise ValueError("kernel_code must define attention_kernel()")
    _optimized_kernel = fn
    print("[hook] Optimized kernel loaded successfully.")


def _is_causal_mask(mask, q_len: int) -> bool:
    """
    Returns True if mask is a standard additive causal mask
    (0 in lower triangle, large-negative in upper triangle).
    transformers 4.51+ passes this instead of is_causal=True.
    """
    if mask is None or q_len < 2:
        return False
    try:
        return bool((mask[..., 0, -1] < -1e4).all())
    except Exception:
        return False


def _triton_sdpa(
    query:    torch.Tensor,
    key:      torch.Tensor,
    value:    torch.Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool  = False,
    scale:     float = None,
) -> torch.Tensor:
    """
    Drop-in replacement for F.scaled_dot_product_attention.
    Falls back to the original SDPA for unsupported shapes.
    """
    if _optimized_kernel is None:
        return _original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)

    B, H_q, N, D = query.shape

    # transformers passes an explicit additive causal mask instead of is_causal=True.
    # Detect and convert it so the Triton kernel can handle causality natively.
    if _is_causal_mask(attn_mask, N):
        is_causal = True
        attn_mask = None

    # Only handle prefill (N >= 64): decode steps (N=1) fall back to PyTorch SDPA.
    # Also restrict to fp16, supported head dims, no non-causal mask, no dropout.
    if (
        query.dtype != torch.float16
        or D not in (64, 128)
        or dropout_p > 0.0
        or attn_mask is not None
        or N < 64
        or N > 8192
    ):
        return _original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)

    # transformers already expands KV heads before calling SDPA, so no GQA
    # expansion is needed here — H_q == H_kv by the time we see the tensors.
    try:
        return _optimized_kernel(query, key, value, is_causal=is_causal, scale=scale)
    except Exception:
        # Never crash the model — silently fall back
        return _original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)


def patch_attention() -> None:
    """Replace F.scaled_dot_product_attention with the Triton kernel."""
    F.scaled_dot_product_attention = _triton_sdpa
    print("[hook] Attention patched → Triton kernel active.")


def unpatch_attention() -> None:
    """Restore the original PyTorch SDPA."""
    F.scaled_dot_product_attention = _original_sdpa
    print("[hook] Attention restored → PyTorch SDPA active.")


def is_patched() -> bool:
    return F.scaled_dot_product_attention is _triton_sdpa
