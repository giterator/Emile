"""
FlashAttention-2 for A100 (Ampere, SM80).

Adapted from the official Triton tutorial:
  https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
  Credits: OpenAI kernel team, Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Changes vs the tutorial source:
  - TensorDescriptor removed  (requires SM90+, A100 is SM80)
  - warp_specialize removed   (Blackwell only)
  - FP8 path removed          (tutorial benchmark feature only)
  - Backward pass removed     (inference-only use case)
  - Wrapped in attention_kernel() to match the agent/hook interface

Everything else -- two-stage causal masking, exp2 scaling, autotune,
online-softmax with l_i initialised to 1.0 -- is preserved as-is.
"""
import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configs (mirrors the tutorial, filtered for A100)
# ---------------------------------------------------------------------------
_configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for s in [2, 3, 4]
    for w in [4, 8]
    if BM >= BN  # tutorial prune rule for causal configs
]


# ---------------------------------------------------------------------------
# Inner loop -- mirrors _attn_fwd_inner from the tutorial (pointer version)
# STAGE=1: off-band tiles (no causal mask)
# STAGE=2: on-band tiles  (causal mask applied)
# STAGE=3: all tiles      (non-causal)
# ---------------------------------------------------------------------------
@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_base, V_base,
    stride_kn, stride_kd, stride_vn, stride_vd,
    start_m, qk_scale,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:  # STAGE == 3: non-causal, all tiles
        lo, hi = 0, N_CTX

    offs_d = tl.arange(0, HEAD_DIM)

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cur_n = start_n + offs_n

        # Load K transposed for tl.dot: (HEAD_DIM, BLOCK_N)
        k = tl.load(
            K_base + cur_n[None, :] * stride_kn + offs_d[:, None] * stride_kd,
            mask=cur_n[None, :] < N_CTX, other=0.0,
        )
        qk = tl.dot(q, k)  # (BLOCK_M, BLOCK_N)

        if STAGE == 2:
            # On-band tile: apply causal mask then scale
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            # Off-band / non-causal: scale first, track running max
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)          # exp2 maps to fast PTX ex2.approx
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        m_i = m_ij

        v = tl.load(
            V_base + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=cur_n[:, None] < N_CTX, other=0.0,
        )
        acc = tl.dot(p.to(tl.float16), v, acc)

    return acc, l_i, m_i


# ---------------------------------------------------------------------------
# Main forward kernel -- mirrors _attn_fwd from the tutorial (pointer path)
# ---------------------------------------------------------------------------
@triton.autotune(configs=_configs, key=["N_CTX", "HEAD_DIM", "STAGE"])
@triton.jit
def _attn_fwd(
    Q, K, V, Out, sm_scale,
    stride_qbh, stride_qm, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_om, stride_od,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,         # 3 = causal, 1 = non-causal (tutorial convention)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh  = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    Q_base = Q + off_bh * stride_qbh
    K_base = K + off_bh * stride_kbh
    V_base = V + off_bh * stride_vbh
    O_base = Out + off_bh * stride_obh

    q = tl.load(
        Q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < N_CTX, other=0.0,
    )

    # Tutorial initialises l_i to 1.0 (not 0.0) -- part of the FA2 formulation
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc  = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Convert scale to log2 domain for exp2 (tutorial trick: avoids ln->exp roundtrip)
    qk_scale = sm_scale * 1.44269504  # sm_scale / ln(2)

    # Stage 1: off-band tiles (only for causal, STAGE=3 -> inner sees STAGE=1)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_base, V_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            4 - STAGE, offs_m, offs_n, N_CTX,
        )

    # Stage 2: on-band / diagonal tiles (causal masking)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_base, V_base,
            stride_kn, stride_kd, stride_vn, stride_vd,
            start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            2, offs_m, offs_n, N_CTX,
        )

    acc = acc / l_i[:, None]
    tl.store(
        O_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        acc.to(tl.float16),
        mask=offs_m[:, None] < N_CTX,
    )


# ---------------------------------------------------------------------------
# Public interface expected by the agent / SDPA hook
# ---------------------------------------------------------------------------
def attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float = None,
) -> torch.Tensor:
    B, H, N, D = q.shape
    assert D in (64, 128), f"Head dim {D} not supported (need 64 or 128)"
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q_f = q.reshape(B * H, N, D).contiguous()
    k_f = k.reshape(B * H, N, D).contiguous()
    v_f = v.reshape(B * H, N, D).contiguous()
    out = torch.empty_like(q_f)

    # Tutorial convention: STAGE=3 for causal, STAGE=1 for non-causal
    stage = 3 if is_causal else 1

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)
    _attn_fwd[grid](
        q_f, k_f, v_f, out, scale,
        q_f.stride(0), q_f.stride(1), q_f.stride(2),
        k_f.stride(0), k_f.stride(1), k_f.stride(2),
        v_f.stride(0), v_f.stride(1), v_f.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
        N_CTX=N, HEAD_DIM=D, STAGE=stage,
    )
    return out.reshape(B, H, N, D)
