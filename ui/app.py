"""
Streamlit UI — two-tab interface:
  Tab 1 · Agent  : live optimization loop with thought stream + metrics chart
  Tab 2 · Demo   : side-by-side Qwen3-4B inference race (baseline vs Triton)

Run with:
    streamlit run ui/app.py
"""
import sys
from pathlib import Path

import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GPU Kernel Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Minimal custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .metric-box {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .thought-bubble {
        background: #1e1e2e;
        border-left: 3px solid #cba6f7;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.9em;
        color: #cdd6f4;
    }
    .tool-call-box {
        background: #181825;
        border-left: 3px solid #89b4fa;
        padding: 8px 14px;
        margin: 4px 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85em;
        color: #89dceb;
        font-family: monospace;
    }
    .speedup-badge {
        font-size: 2.2em;
        font-weight: 700;
        color: #a6e3a1;
    }
    .token-stream {
        font-family: 'Menlo', monospace;
        font-size: 0.92em;
        line-height: 1.6;
        min-height: 280px;
        background: #11111b;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 14px;
        color: #cdd6f4;
        white-space: pre-wrap;
        overflow-y: auto;
    }
    .speed-counter {
        font-size: 1.6em;
        font-weight: 600;
    }
    .baseline-speed  { color: #f38ba8; }
    .triton-speed    { color: #a6e3a1; }
    div[data-testid="stTabs"] button { font-size: 1.05em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "opt_running":    False,
        "opt_done":       False,
        "opt_events":     [],
        "best_kernel":    None,
        "best_metrics":   None,
        "baseline_metrics": None,
        "demo_running":   False,
        "triton_text":    "",
        "triton_tps":     0.0,
        "race_event":     None,
        "demo_done":      False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Load starting kernel from file
# ---------------------------------------------------------------------------

@st.cache_data
def _load_reference_kernel() -> str:
    p = Path(__file__).parent.parent / "kernels" / "pytorch_reference_kernel.py"
    return p.read_text()

V1_KERNEL = _load_reference_kernel()

# ---------------------------------------------------------------------------
# Minimal Triton kernel used for demo tab when the agent hasn't run yet.
# Implements the basic FlashAttention tiling pattern — correct but untuned.
# ---------------------------------------------------------------------------

_DEBUG_TRITON_KERNEL = """\
import math

@triton.jit
def _attn_fwd(
    Q, K, V, Out, sm_scale,
    stride_qbh, stride_qm, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_om, stride_od,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
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

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504

    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cur_n   = start_n + offs_n

        k = tl.load(
            K_base + cur_n[None, :] * stride_kn + offs_d[:, None] * stride_kd,
            mask=cur_n[None, :] < N_CTX, other=0.0,
        )
        qk = tl.dot(q, k) * qk_scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= cur_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p    = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i  = l_i * alpha + tl.sum(p, 1)
        acc  = acc * alpha[:, None]
        m_i  = m_ij

        v = tl.load(
            V_base + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=cur_n[:, None] < N_CTX, other=0.0,
        )
        acc = tl.dot(p.to(tl.float16), v, acc)

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

    BLOCK_M, BLOCK_N = 64, 32
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _attn_fwd[grid](
        q_f, k_f, v_f, out, scale,
        q_f.stride(0), q_f.stride(1), q_f.stride(2),
        k_f.stride(0), k_f.stride(1), k_f.stride(2),
        v_f.stride(0), v_f.stride(1), v_f.stride(2),
        out.stride(0),  out.stride(1),  out.stride(2),
        N_CTX=N, HEAD_DIM=D, IS_CAUSAL=is_causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return out.reshape(B, H, N, D)
"""

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    "## ⚡ GPU Kernel Optimization Agent  "
    "<span style='font-size:0.55em;color:#6c7086;font-weight:400'>"
    "Qwen3-4B · A100-80GB · PyTorch reference baseline + LLM agent</span>",
    unsafe_allow_html=True,
)
st.divider()

tab_agent, tab_demo = st.tabs(["🤖  Agent Optimizer", "🚀  Live Inference Demo"])


# ============================================================================
# TAB 1 — AGENT OPTIMIZER
# ============================================================================

with tab_agent:
    col_controls, col_spacer, col_config = st.columns([3, 0.3, 2])

    with col_controls:
        st.markdown(
            "#### Reference Baseline  "
            "<span style='font-size:0.78em;color:#6c7086;font-weight:400'>"
            "PyTorch reference implementation — agent writes a Triton kernel to beat this</span>",
            unsafe_allow_html=True,
        )
        kernel_editor = st.text_area(
            label="kernel_code",
            value=V1_KERNEL,
            height=340,
            label_visibility="collapsed",
        )

    with col_config:
        st.markdown("#### Benchmark Config")
        seq_len = st.selectbox("Sequence length", [512, 1024, 2048, 4096], index=2)
        d_head  = st.selectbox("Head dimension",  [64, 128], index=1)
        n_heads = st.selectbox("Query heads",     [8, 16, 32], index=2)
        batch   = st.slider("Batch size", 1, 8, 1)

        target_pct = st.slider(
            "Efficiency target (%)", 1.0, 100.0, 70.0, 0.5,
            help="Stop when the Triton kernel's compute efficiency exceeds this % of A100 peak (312 TFLOPS).",
        )
        max_iter = st.slider("Max iterations", 2, 8, 5)

        st.markdown("---")
        run_btn = st.button(
            "▶  Run Optimization Agent",
            type="primary",
            disabled=st.session_state.opt_running,
            use_container_width=True,
        )

    # ── Thought stream + metrics ──────────────────────────────────────────

    st.markdown("---")
    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### Agent Thought Stream")
        thought_area = st.container()

    with right:
        st.markdown("#### Performance Progression")
        chart_placeholder = st.empty()
        summary_placeholder = st.empty()

    # ── Trigger the agent ────────────────────────────────────────────────

    if run_btn:
        st.session_state.opt_running    = True
        st.session_state.opt_done       = False
        st.session_state.opt_events     = []
        st.session_state.best_kernel    = None
        st.session_state.best_metrics   = None
        st.session_state.baseline_metrics = None

        from agent import run_optimization_agent

        config = {
            "seq_len": seq_len,
            "d_head":  d_head,
            "n_heads": n_heads,
            "batch":   batch,
        }

        tflops_history = []   # (iteration, tflops) for chart

        for event in run_optimization_agent(
            kernel_code=kernel_editor,
            config=config,
            max_iterations=max_iter,
        ):
            st.session_state.opt_events.append(event)
            etype = event["type"]

            # Render event immediately
            with thought_area:
                if etype == "thought":
                    st.markdown(
                        f"<div class='thought-bubble'>💭 {event['text']}</div>",
                        unsafe_allow_html=True,
                    )
                elif etype == "tool_call":
                    name = event["name"]
                    cfg  = {k: v for k, v in event["input"].items() if k != "kernel_code"}
                    st.markdown(
                        f"<div class='tool-call-box'>🔧 {name}({cfg})</div>",
                        unsafe_allow_html=True,
                    )
                elif etype == "error":
                    st.error(f"⚠ {event['text']}")
                    with st.expander("Full error detail"):
                        st.code(event["text"])
                elif etype == "metrics":
                    d   = event["data"]
                    itr = event["iteration"]
                    tflops_history.append((itr, d["tflops"]))

                    if itr == 0:
                        st.session_state.baseline_metrics = d

                    label = "Baseline" if itr == 0 else f"V{itr}"
                    st.markdown(
                        f"<div class='thought-bubble'>"
                        f"📊 <b>{label}</b>: {d['tflops']:.2f} TFLOPS | "
                        f"{d['bandwidth_gbs']:.0f} GB/s | "
                        f"{d['efficiency_pct']:.1f}% efficiency | "
                        f"<i>{d['bound']}-bound</i>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Update chart
                    if len(tflops_history) > 1:
                        import pandas as pd
                        df = pd.DataFrame(tflops_history, columns=["Iteration", "TFLOPS"])
                        chart_placeholder.line_chart(df.set_index("Iteration"))

                elif etype == "done":
                    st.session_state.best_kernel  = event["best_code"]
                    st.session_state.best_metrics = event["best_metrics"]
                    st.session_state.opt_done     = True

                    base = event["baseline"]
                    best = event["best_metrics"]
                    speedup = event["speedup"]

                    summary_placeholder.markdown(
                        f"<div class='metric-box'>"
                        f"<b>Optimization complete</b><br>"
                        f"Baseline : {base['tflops']:.2f} TFLOPS ({base['efficiency_pct']:.1f}%)<br>"
                        f"Best     : {best['tflops']:.2f} TFLOPS ({best['efficiency_pct']:.1f}%)<br>"
                        f"<span class='speedup-badge'>{speedup}×</span> kernel speedup"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='thought-bubble'>✅ Agent finished in "
                        f"{event['iterations']} iterations — "
                        f"{speedup}× faster than baseline.</div>",
                        unsafe_allow_html=True,
                    )

        st.session_state.opt_running = False

    # ── Show best kernel if optimization is done ──────────────────────────

    if st.session_state.opt_done and st.session_state.best_kernel:
        st.markdown("---")
        st.markdown("#### Best Kernel Generated by Agent")
        st.code(st.session_state.best_kernel, language="python")
        st.download_button(
            "⬇  Download optimized kernel",
            data=st.session_state.best_kernel,
            file_name="optimized_attention.py",
            mime="text/plain",
        )


# ============================================================================
# TAB 2 — LIVE INFERENCE DEMO
# ============================================================================

with tab_demo:
    st.markdown("#### Side-by-side Qwen3-4B Inference")
    st.caption(
        "Baseline uses PyTorch SDPA · Optimized uses the Triton kernel below · "
        "Same prompt, same model weights, same A100."
    )

    # Determine which kernel to use: agent's best if available, else the debug kernel
    _demo_kernel_default = (
        st.session_state.best_kernel
        if st.session_state.opt_done and st.session_state.best_kernel
        else _DEBUG_TRITON_KERNEL
    )

    demo_col_left, demo_col_right = st.columns([3, 2])
    with demo_col_left:
        prompt = st.text_area(
            "Prompt",
            height=100,
            value=(
                "Explain in detail how the transformer attention mechanism works, "
                "including the mathematical formulation of scaled dot-product attention, "
                "why we scale by the square root of the head dimension, "
                "and how multi-head attention differs from single-head attention. "
                "Also discuss the computational complexity and memory requirements."
            ),
        )
    with demo_col_right:
        max_tokens = st.slider("Max new tokens", 50, 300, 150)
        context_tokens = st.slider(
            "Context length (tokens)",
            128, 4096, 1024, 128,
            help=(
                "Pad the prompt to this many tokens before generation. "
                "Larger values make prefill dominate and surface the real "
                "Triton vs PyTorch-reference speedup (matches the benchmark shape)."
            ),
        )
        if not st.session_state.opt_done:
            st.info(
                "Using built-in debug kernel. Run the Agent Optimizer to test "
                "an agent-generated kernel here.",
                icon="ℹ️",
            )
        else:
            st.success("Using agent's best kernel.", icon="✅")

    with st.expander("Triton kernel used for inference (editable)", expanded=False):
        demo_kernel_code = st.text_area(
            label="demo_kernel",
            value=_demo_kernel_default,
            height=300,
            label_visibility="collapsed",
        )

    demo_btn = st.button(
        "▶  Run Inference Race",
        type="primary",
        disabled=st.session_state.demo_running,
        use_container_width=False,
    )

    st.divider()

    # ── Layout: kernel race banner + model output ──────────────────────
    race_ph   = st.empty()   # kernel benchmark result card
    st.markdown(
        "### Triton Model Output  <span style='color:#a6e3a1;font-size:0.7em'>⚡</span>",
        unsafe_allow_html=True,
    )
    triton_speed_ph  = st.empty()
    triton_ttft_ph   = st.empty()
    triton_output_ph = st.empty()

    summary_row = st.empty()

    # ── Run the demo ──────────────────────────────────────────────────

    if demo_btn:
        st.session_state.demo_running   = True
        st.session_state.triton_text    = ""
        st.session_state.demo_done      = False
        st.session_state.race_event     = None

        import modal

        inference_fn = modal.Function.from_name(
            "qwen3-kernel-optimizer", "run_inference_comparison"
        )

        race_ph.markdown(
            "<div class='metric-box' style='text-align:center'>🔄 Running kernel race...</div>",
            unsafe_allow_html=True,
        )
        triton_speed_ph.markdown(
            "<span class='speed-counter triton-speed'>⏳ waiting for kernel race...</span>",
            unsafe_allow_html=True,
        )

        triton_tps_final = 0.0
        triton_ttft_final = 0.0

        for event in inference_fn.remote_gen(prompt, demo_kernel_code, max_tokens, context_tokens):
            phase = event.get("phase")

            if phase == "loading":
                race_ph.markdown(
                    "<div class='metric-box' style='text-align:center'>🔄 Loading model...</div>",
                    unsafe_allow_html=True,
                )

            elif phase == "kernel_race_done":
                st.session_state.race_event = event
                ref_ms        = event["ref_ms"]
                triton_ms     = event.get("triton_ms")
                ref_tflops    = event["ref_tflops"]
                triton_tflops = event["triton_tflops"]
                speedup       = event["speedup"]
                n_tok         = event["n_tokens"]

                triton_cell = (
                    f"<b>{triton_ms:.2f} ms</b> &nbsp;·&nbsp; {triton_tflops:.1f} TFLOPS"
                    if triton_ms is not None
                    else "<i>not provided</i>"
                )
                race_ph.markdown(
                    f"""
                    <div class='metric-box' style='text-align:center;padding:1em 0'>
                    <div style='font-size:0.85em;color:#6c7086;margin-bottom:0.5em'>
                    Direct kernel race &nbsp;·&nbsp; B=1 H=32 N={n_tok} D=128 fp16 is_causal=True
                    </div>
                    <table style='width:100%;border-collapse:collapse;font-size:1em'>
                    <tr>
                      <th style='text-align:left;color:#f38ba8;padding:0.2em 0.6em'>PyTorch Reference</th>
                      <th style='text-align:left;color:#a6e3a1;padding:0.2em 0.6em'>Triton Kernel</th>
                      <th style='text-align:left;color:#f9e2af;padding:0.2em 0.6em'>Speedup</th>
                    </tr>
                    <tr>
                      <td style='padding:0.2em 0.6em'><b>{ref_ms:.2f} ms</b> &nbsp;·&nbsp; {ref_tflops:.1f} TFLOPS</td>
                      <td style='padding:0.2em 0.6em'>{triton_cell}</td>
                      <td style='padding:0.2em 0.6em'><span class='speedup-badge'>{speedup:.2f}×</span></td>
                    </tr>
                    </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                triton_speed_ph.markdown(
                    "<span class='speed-counter triton-speed'>🔄 generating...</span>",
                    unsafe_allow_html=True,
                )

            elif phase == "triton_token":
                st.session_state.triton_text += event["token"]
                tps = event["tokens_per_sec"]
                triton_speed_ph.markdown(
                    f"<span class='speed-counter triton-speed'>{tps:.1f} tok/s ▌</span>",
                    unsafe_allow_html=True,
                )
                triton_output_ph.markdown(
                    f"<div class='token-stream'>{st.session_state.triton_text}▌</div>",
                    unsafe_allow_html=True,
                )

            elif phase == "triton_done":
                triton_tps_final  = event["tokens_per_sec"]
                triton_ttft_final = event.get("ttft_ms", 0.0)
                st.session_state.triton_tps  = triton_tps_final
                st.session_state.triton_ttft = triton_ttft_final

                triton_speed_ph.markdown(
                    f"<span class='speed-counter triton-speed'>{triton_tps_final:.1f} tok/s</span>",
                    unsafe_allow_html=True,
                )
                triton_ttft_ph.markdown(
                    f"<div style='color:#a6e3a1;font-size:0.85em;margin-top:-0.3em'>"
                    f"TTFT: <b>{triton_ttft_final:.0f} ms</b></div>",
                    unsafe_allow_html=True,
                )
                triton_output_ph.markdown(
                    f"<div class='token-stream'>{st.session_state.triton_text}</div>",
                    unsafe_allow_html=True,
                )
                st.session_state.demo_done = True

            elif phase == "error":
                st.error(f"Demo error: {event['message']}")
                break

        st.session_state.demo_running = False

    # Restore completed state on re-render
    elif st.session_state.demo_done:
        _race = st.session_state.get("race_event")
        if _race:
            _ref_ms    = _race["ref_ms"]
            _tri_ms    = _race.get("triton_ms")
            _ref_tf    = _race["ref_tflops"]
            _tri_tf    = _race["triton_tflops"]
            _speedup   = _race["speedup"]
            _n         = _race["n_tokens"]
            _tri_cell  = (
                f"<b>{_tri_ms:.2f} ms</b> &nbsp;·&nbsp; {_tri_tf:.1f} TFLOPS"
                if _tri_ms is not None else "<i>not provided</i>"
            )
            race_ph.markdown(
                f"""
                <div class='metric-box' style='text-align:center;padding:1em 0'>
                <div style='font-size:0.85em;color:#6c7086;margin-bottom:0.5em'>
                Direct kernel race &nbsp;·&nbsp; B=1 H=32 N={_n} D=128 fp16 is_causal=True
                </div>
                <table style='width:100%;border-collapse:collapse;font-size:1em'>
                <tr>
                  <th style='text-align:left;color:#f38ba8;padding:0.2em 0.6em'>PyTorch Reference</th>
                  <th style='text-align:left;color:#a6e3a1;padding:0.2em 0.6em'>Triton Kernel</th>
                  <th style='text-align:left;color:#f9e2af;padding:0.2em 0.6em'>Speedup</th>
                </tr>
                <tr>
                  <td style='padding:0.2em 0.6em'><b>{_ref_ms:.2f} ms</b> &nbsp;·&nbsp; {_ref_tf:.1f} TFLOPS</td>
                  <td style='padding:0.2em 0.6em'>{_tri_cell}</td>
                  <td style='padding:0.2em 0.6em'><span class='speedup-badge'>{_speedup:.2f}×</span></td>
                </tr>
                </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        _t_tps  = st.session_state.triton_tps
        _t_ttft = st.session_state.get("triton_ttft", 0.0)
        triton_speed_ph.markdown(
            f"<span class='speed-counter triton-speed'>{_t_tps:.1f} tok/s</span>",
            unsafe_allow_html=True,
        )
        triton_ttft_ph.markdown(
            f"<div style='color:#a6e3a1;font-size:0.85em;margin-top:-0.3em'>"
            f"TTFT: <b>{_t_ttft:.0f} ms</b></div>",
            unsafe_allow_html=True,
        )
        triton_output_ph.markdown(
            f"<div class='token-stream'>{st.session_state.triton_text}</div>",
            unsafe_allow_html=True,
        )
        summary_row.markdown(
            f"""
            <div class='metric-box' style='margin-top:1em;text-align:center'>
            <span style='font-size:0.85em;color:#6c7086'>
            Kernel race complete &nbsp;·&nbsp; model output generated with Triton kernel active
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
