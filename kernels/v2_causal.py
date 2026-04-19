"""
V2: Optimized Triton attention kernel with causal masking support.

Improvements over V1:
  - BLOCK_M=128, BLOCK_N=64  : larger tiles, better SRAM reuse
  - num_warps=8               : fills SM warp scheduler
  - num_stages=2              : software pipelining (overlaps HBM loads with compute)
  - IS_CAUSAL constexpr       : skips upper-triangle K tiles, masks partial tile
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, Out,
    stride_qbh, stride_qm, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_om, stride_od,
    N_CTX,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh  = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    Q_base = Q + off_bh * stride_qbh
    K_base = K + off_bh * stride_kbh
    V_base = V + off_bh * stride_vbh
    O_base = Out + off_bh * stride_obh

    q = tl.load(
        Q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < N_CTX, other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # For causal: only iterate over K tiles that are not fully above the diagonal
    n_end = N_CTX
    if IS_CAUSAL:
        n_end = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)

    for start_n in range(0, n_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k = tl.load(
            K_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn,
            mask=offs_n[None, :] < N_CTX, other=0.0,
        )

        qk = tl.dot(q, k) * scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        m_new  = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha  = tl.exp(m_i - m_new)
        p      = tl.exp(qk - m_new[:, None])
        l_i    = l_i * alpha + tl.sum(p, axis=1)
        acc    = acc * alpha[:, None]
        m_i    = m_new

        v = tl.load(
            V_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=offs_n[:, None] < N_CTX, other=0.0,
        )
        acc = acc + tl.dot(p.to(tl.float16), v)

    acc = acc / l_i[:, None]
    tl.store(
        O_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(tl.float16),
        mask=offs_m[:, None] < N_CTX,
    )


def attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float = None,
) -> torch.Tensor:
    B, H, N, D = q.shape
    assert D in (64, 128), f"Head dim {D} not supported"
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q_f = q.reshape(B * H, N, D).contiguous()
    k_f = k.reshape(B * H, N, D).contiguous()
    v_f = v.reshape(B * H, N, D).contiguous()
    out = torch.empty_like(q_f)

    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _attn_fwd[grid](
        q_f, k_f, v_f, out,
        q_f.stride(0), q_f.stride(1), q_f.stride(2),
        k_f.stride(0), k_f.stride(1), k_f.stride(2),
        v_f.stride(0), v_f.stride(1), v_f.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
        N, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=D,
        IS_CAUSAL=is_causal,
        num_warps=8,
        num_stages=2,
    )
    return out.reshape(B, H, N, D)
