"""
V1: Naive Triton attention kernel - the agent's starting point.

Intentionally suboptimal:
  - BLOCK_M=32, BLOCK_N=32  -> small tiles, poor SRAM reuse
  - num_warps=2              -> under-utilises the SM warp scheduler
  - No software pipelining   -> HBM loads are not overlapped with compute

Expected profile on A100 (seq=2048, d=128):
  ~2-3 TFLOPS  (~0.7% of 312 TFLOPS peak)
  Highly memory-bound (low arithmetic intensity)

The agent will read these metrics and evolve toward a tiled, pipelined kernel.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _naive_attn_fwd(
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
):
    # Each program handles one BLOCK_M strip of queries for one (batch, head)
    start_m = tl.program_id(0)
    off_bh  = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    Q_base = Q + off_bh * stride_qbh
    K_base = K + off_bh * stride_kbh
    V_base = V + off_bh * stride_vbh
    O_base = Out + off_bh * stride_obh

    # Load Q tile - stays in SRAM for all K/V iterations
    q = tl.load(
        Q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )  # (BLOCK_M, BLOCK_D)

    # Online softmax state
    m_i  = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i  = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K transposed -> (BLOCK_D, BLOCK_N) so tl.dot gives (BLOCK_M, BLOCK_N)
        k = tl.load(
            K_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn,
            mask=offs_n[None, :] < N_CTX,
            other=0.0,
        )  # (BLOCK_D, BLOCK_N)

        qk = tl.dot(q, k) * scale  # (BLOCK_M, BLOCK_N)

        # Online softmax: update running max and normalizer
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha  = tl.exp(m_i - m_new)
        p      = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        m_i = m_new

        # Load V -> (BLOCK_N, BLOCK_D) and accumulate
        v = tl.load(
            V_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=offs_n[:, None] < N_CTX,
            other=0.0,
        )  # (BLOCK_N, BLOCK_D)

        acc = acc + tl.dot(p.to(tl.float16), v)

    # Final normalisation and store
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
    """
    Naive Triton attention - suboptimal block sizes, minimal warps.
    q, k, v: (B, H, N, D) float16
    """
    B, H, N, D = q.shape
    assert D in (64, 128), f"Head dim {D} not supported (need 64 or 128)"
    assert not is_causal, "Causal masking not implemented in V1 - agent task"

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Flatten batch and head dims for kernel dispatch
    q_f = q.reshape(B * H, N, D).contiguous()
    k_f = k.reshape(B * H, N, D).contiguous()
    v_f = v.reshape(B * H, N, D).contiguous()
    out = torch.empty_like(q_f)

    # Suboptimal knobs - the agent is expected to improve these
    BLOCK_M = 32
    BLOCK_N = 32

    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _naive_attn_fwd[grid](
        q_f, k_f, v_f, out,
        q_f.stride(0), q_f.stride(1), q_f.stride(2),
        k_f.stride(0), k_f.stride(1), k_f.stride(2),
        v_f.stride(0), v_f.stride(1), v_f.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
        N, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=D,
        num_warps=2,   # too few - leaves SM warp slots idle
    )

    return out.reshape(B, H, N, D)
