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
        "baseline_text":  "",
        "triton_tps":     0.0,
        "race_replay":    None,
        "baseline_rec":   None,
        "triton_rec":     None,
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
# Render helpers for the demo tab race table and TTFT/hit-rate strip
# ---------------------------------------------------------------------------

def _render_metrics(rec: dict, color: str) -> str:
    """Metrics strip: throughput, TTFT, total execution time, total tokens generated."""
    tps      = rec.get("tps", 0.0)
    ttft     = rec.get("ttft_ms", 0.0)
    total_ms = rec.get("total_ms", 0.0)
    count    = rec.get("count", 0)
    return (
        f"<div style='color:{color};font-size:0.85em;margin-top:-0.3em;line-height:1.6em'>"
        f"throughput: <b>{tps:.1f} tok/s</b> &nbsp;·&nbsp; "
        f"TTFT: <b>{ttft:.0f} ms</b> &nbsp;·&nbsp; "
        f"total time: <b>{total_ms:.0f} ms</b> &nbsp;·&nbsp; "
        f"tokens generated: <b>{count}</b>"
        f"</div>"
    )


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
        "Baseline uses naive PyTorch · Optimized uses the Triton kernel below · "
        "Same prompt, same model weights, same A100."
    )
    st.info(
        "**Note:** Triton kernel optimizes **prefill attention**, speeding up TTFT especially for long prompts.",
        icon="💡",
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
            height=380,
            value=(
                "I'm preparing for a FAANG coding interview and I want you to act "
                "as my interviewer. Please walk me through the two LeetCode Hard "
                "problems below with the same rigor an interviewer would expect: "
                "clarify assumptions, propose multiple approaches from naive to "
                "optimal, prove correctness informally, analyse time and space "
                "complexity, identify tricky edge cases, and provide clean, "
                "idiomatic Python code that would pass on LeetCode. After each "
                "problem, briefly reflect on what the interviewer is really "
                "testing and what a strong candidate should emphasise.\n\n"
                "====================================================\n"
                "Problem 1: Median of Two Sorted Arrays  (LeetCode 4, Hard)\n"
                "====================================================\n\n"
                "Given two sorted arrays `nums1` and `nums2` of size `m` and `n` "
                "respectively, return the median of the two sorted arrays. The "
                "overall run time complexity should be O(log (m + n)).\n\n"
                "Example 1:\n"
                "  Input:  nums1 = [1, 3], nums2 = [2]\n"
                "  Output: 2.00000\n"
                "  Explanation: merged array = [1, 2, 3] and median is 2.\n\n"
                "Example 2:\n"
                "  Input:  nums1 = [1, 2], nums2 = [3, 4]\n"
                "  Output: 2.50000\n"
                "  Explanation: merged array = [1, 2, 3, 4] and median is "
                "(2 + 3) / 2 = 2.5.\n\n"
                "Example 3:\n"
                "  Input:  nums1 = [], nums2 = [1]\n"
                "  Output: 1.00000\n\n"
                "Example 4:\n"
                "  Input:  nums1 = [0, 0], nums2 = [0, 0]\n"
                "  Output: 0.00000\n\n"
                "Constraints:\n"
                "  - nums1.length == m\n"
                "  - nums2.length == n\n"
                "  - 0 <= m <= 1000\n"
                "  - 0 <= n <= 1000\n"
                "  - 1 <= m + n <= 2000\n"
                "  - -10^6 <= nums1[i], nums2[i] <= 10^6\n"
                "  - Both arrays are sorted in non-decreasing order.\n\n"
                "Walk me through three approaches with increasing sophistication: "
                "(1) the naive O((m+n) log(m+n)) concat-then-sort solution, "
                "(2) the O(m+n) two-pointer merge that stops at the median index, "
                "and (3) the optimal O(log(min(m, n))) binary-search-on-partitions "
                "solution. For the optimal approach, explain the key insight about "
                "partitioning both arrays so every element on the left side is "
                "<= every element on the right side, the invariants that must hold "
                "at each step, how to handle the empty-partition and boundary edge "
                "cases using +/- infinity sentinels, and why binary searching only "
                "the shorter array is both correct and gives the log(min(m, n)) "
                "bound. Include clean Python code for all three approaches.\n\n"
                "====================================================\n"
                "Problem 2: Regular Expression Matching  (LeetCode 10, Hard)\n"
                "====================================================\n\n"
                "Given an input string `s` and a pattern `p`, implement regular "
                "expression matching with support for '.' and '*' where:\n"
                "  - '.'  matches any single character\n"
                "  - '*'  matches zero or more of the PRECEDING element\n"
                "The matching should cover the entire input string (not partial).\n\n"
                "Example 1:\n"
                "  Input:  s = \"aa\",   p = \"a\"\n"
                "  Output: false\n"
                "  Explanation: \"a\" does not match the entire string \"aa\".\n\n"
                "Example 2:\n"
                "  Input:  s = \"aa\",   p = \"a*\"\n"
                "  Output: true\n"
                "  Explanation: '*' means zero or more of the preceding element, "
                "'a'. Therefore, by repeating 'a' once, it becomes \"aa\".\n\n"
                "Example 3:\n"
                "  Input:  s = \"ab\",   p = \".*\"\n"
                "  Output: true\n"
                "  Explanation: \".*\" means zero or more of any character.\n\n"
                "Example 4:\n"
                "  Input:  s = \"mississippi\", p = \"mis*is*p*.\"\n"
                "  Output: false\n\n"
                "Constraints:\n"
                "  - 1 <= s.length <= 20\n"
                "  - 1 <= p.length <= 20\n"
                "  - `s` contains only lowercase English letters.\n"
                "  - `p` contains only lowercase letters, '.', and '*'.\n"
                "  - It is guaranteed for each appearance of '*', there will be "
                "a previous valid character to match.\n\n"
                "Walk me through: (1) the recursive brute-force solution and why "
                "its worst-case complexity is exponential, (2) the top-down "
                "memoised version with complexity O(m*n), and (3) the bottom-up "
                "dynamic-programming formulation that fills a (m+1) x (n+1) table. "
                "For the DP version, clearly define dp[i][j], state the recurrence "
                "for the three cases ('*' matches zero copies, '*' matches one or "
                "more copies, regular character or '.' match), and explain the "
                "base cases -- especially why `dp[0][j]` for patterns like \"a*b*c*\" "
                "needs careful handling. Discuss edge cases such as leading '*', "
                "consecutive '*' (forbidden by constraints but worth noting), and "
                "empty string / empty pattern. Provide clean Python code and "
                "argue the O(m*n) time and space bound.\n\n"
                "Finally, reflect on what makes these two problems classic Hard "
                "interview questions: what invariant-reasoning, binary-search, and "
                "DP skills they test, and what a strong candidate should emphasise "
                "versus common pitfalls weaker candidates fall into."
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

    # ── Layout: status + side-by-side model output ──────────────────────────
    status_ph = st.empty()   # "Recording baseline / triton / replaying..." status

    col_base, col_sep, col_triton = st.columns([10, 0.4, 10])
    with col_base:
        st.markdown(
            "### Naive PyTorch  <span style='color:#f38ba8;font-size:0.7em'>(baseline)</span>",
            unsafe_allow_html=True,
        )
        base_speed_ph  = st.empty()
        base_ttft_ph   = st.empty()
        base_output_ph = st.empty()
    with col_sep:
        st.markdown(
            "<div style='border-left:2px solid #313244;height:400px;margin-top:2em'></div>",
            unsafe_allow_html=True,
        )
    with col_triton:
        st.markdown(
            "### Triton Kernel Integration  <span style='color:#a6e3a1;font-size:0.7em'>⚡ optimized</span>",
            unsafe_allow_html=True,
        )
        triton_speed_ph  = st.empty()
        triton_ttft_ph   = st.empty()
        triton_output_ph = st.empty()

    summary_row = st.empty()

    # ── Run the demo ──────────────────────────────────────────────────
    if demo_btn:
        import time
        import modal

        st.session_state.demo_running    = True
        st.session_state.demo_done       = False
        st.session_state.baseline_text   = ""
        st.session_state.triton_text     = ""
        st.session_state.baseline_rec    = None
        st.session_state.triton_rec      = None

        inference_fn = modal.Function.from_name(
            "qwen3-kernel-optimizer", "run_inference_comparison"
        )

        status_ph.markdown(
            "<div class='metric-box' style='text-align:center'>🔄 Loading Qwen3-4B...</div>",
            unsafe_allow_html=True,
        )
        base_speed_ph.markdown(
            "<span class='speed-counter baseline-speed'>⏳ waiting...</span>",
            unsafe_allow_html=True,
        )
        triton_speed_ph.markdown(
            "<span class='speed-counter triton-speed'>⏳ waiting...</span>",
            unsafe_allow_html=True,
        )

        for event in inference_fn.remote_gen(prompt, demo_kernel_code, max_tokens, context_tokens):
            phase = event.get("phase")

            if phase == "loading":
                status_ph.markdown(
                    "<div class='metric-box' style='text-align:center'>🔄 Loading Qwen3-4B...</div>",
                    unsafe_allow_html=True,
                )

            elif phase == "recording_start":
                side = event.get("side", "?")
                if side == "baseline":
                    status_ph.markdown(
                        "<div class='metric-box' style='text-align:center;color:#f38ba8'>"
                        "🔴 Recording baseline (naive PyTorch) generation...</div>",
                        unsafe_allow_html=True,
                    )
                    base_speed_ph.markdown(
                        "<span class='speed-counter baseline-speed'>🔴 recording...</span>",
                        unsafe_allow_html=True,
                    )
                elif side == "triton":
                    status_ph.markdown(
                        "<div class='metric-box' style='text-align:center;color:#a6e3a1'>"
                        "🔴 Recording Triton generation...</div>",
                        unsafe_allow_html=True,
                    )
                    triton_speed_ph.markdown(
                        "<span class='speed-counter triton-speed'>🔴 recording...</span>",
                        unsafe_allow_html=True,
                    )

            elif phase == "recording_done":
                side = event.get("side", "?")
                if side == "baseline":
                    st.session_state.baseline_rec = event
                    base_speed_ph.markdown(
                        "<span class='speed-counter baseline-speed'>✓ recorded</span>",
                        unsafe_allow_html=True,
                    )
                elif side == "triton":
                    st.session_state.triton_rec = event
                    triton_speed_ph.markdown(
                        "<span class='speed-counter triton-speed'>✓ recorded</span>",
                        unsafe_allow_html=True,
                    )

            elif phase == "race_replay":
                # ── Animate both recordings simultaneously using the recorded timestamps ──
                baseline = event["baseline"]
                triton   = event["triton"]
                b_tokens = baseline.get("tokens", [])
                t_tokens = triton.get("tokens", [])

                # Small speedup so the demo doesn't take the full wall-clock duration.
                # Using 1.0 = real time; set to e.g. 2.0 to replay at 2x speed. Keep 1.0 for honesty.
                REPLAY_SPEED = 1.0

                status_ph.markdown(
                    "<div class='metric-box' style='text-align:center;color:#cba6f7'>"
                    "🏁 Replaying both generations at recorded speeds (watch the race)...</div>",
                    unsafe_allow_html=True,
                )

                b_idx = t_idx = 0
                b_text = t_text = ""
                b_finished = t_finished = False
                start_wall = time.perf_counter()
                max_elapsed = max(
                    b_tokens[-1]["elapsed_ms"] if b_tokens else 0,
                    t_tokens[-1]["elapsed_ms"] if t_tokens else 0,
                )

                while b_idx < len(b_tokens) or t_idx < len(t_tokens):
                    elapsed_real_ms = (time.perf_counter() - start_wall) * 1000 * REPLAY_SPEED
                    updated = False

                    while b_idx < len(b_tokens) and b_tokens[b_idx]["elapsed_ms"] <= elapsed_real_ms:
                        b_text += b_tokens[b_idx]["text"]
                        b_idx += 1
                        updated = True
                    while t_idx < len(t_tokens) and t_tokens[t_idx]["elapsed_ms"] <= elapsed_real_ms:
                        t_text += t_tokens[t_idx]["text"]
                        t_idx += 1
                        updated = True

                    if updated:
                        b_tps = (b_idx / (elapsed_real_ms / 1000)) if elapsed_real_ms > 0 else 0
                        t_tps = (t_idx / (elapsed_real_ms / 1000)) if elapsed_real_ms > 0 else 0
                        b_cursor = "▌" if b_idx < len(b_tokens) else ""
                        t_cursor = "▌" if t_idx < len(t_tokens) else ""
                        base_speed_ph.markdown(
                            f"<span class='speed-counter baseline-speed'>{b_tps:.1f} tok/s {b_cursor}</span>",
                            unsafe_allow_html=True,
                        )
                        triton_speed_ph.markdown(
                            f"<span class='speed-counter triton-speed'>{t_tps:.1f} tok/s {t_cursor}</span>",
                            unsafe_allow_html=True,
                        )
                        base_output_ph.markdown(
                            f"<div class='token-stream'>{b_text}{b_cursor}</div>",
                            unsafe_allow_html=True,
                        )
                        triton_output_ph.markdown(
                            f"<div class='token-stream'>{t_text}{t_cursor}</div>",
                            unsafe_allow_html=True,
                        )

                    # Flag finish events as they cross the line (first time we reach their end)
                    if not b_finished and b_idx >= len(b_tokens):
                        b_finished = True
                    if not t_finished and t_idx >= len(t_tokens):
                        t_finished = True

                    time.sleep(0.02)

                # Final render with full text and measured stats
                base_speed_ph.markdown(
                    f"<span class='speed-counter baseline-speed'>{baseline['tps']:.1f} tok/s</span>",
                    unsafe_allow_html=True,
                )
                triton_speed_ph.markdown(
                    f"<span class='speed-counter triton-speed'>{triton['tps']:.1f} tok/s</span>",
                    unsafe_allow_html=True,
                )
                base_ttft_ph.markdown(
                    _render_metrics(baseline, "#f38ba8"),
                    unsafe_allow_html=True,
                )
                triton_ttft_ph.markdown(
                    _render_metrics(triton, "#a6e3a1"),
                    unsafe_allow_html=True,
                )
                base_output_ph.markdown(
                    f"<div class='token-stream'>{b_text}</div>", unsafe_allow_html=True,
                )
                triton_output_ph.markdown(
                    f"<div class='token-stream'>{t_text}</div>", unsafe_allow_html=True,
                )

                # Result banner -- honest about wins vs regressions
                sp_ttft = event.get("speedup_ttft")
                sp_tps  = event.get("speedup_tps")

                def _fmt(sp, label):
                    if not sp:
                        return ""
                    if sp >= 1.05:
                        return f"<b>{sp:.2f}×</b> faster {label}"
                    if sp <= 0.95:
                        return f"<b>{1/sp:.2f}×</b> slower {label}"
                    return f"<b>~1.0×</b> {label} (neutral)"

                ttft_badge = _fmt(sp_ttft, "TTFT")
                tps_badge  = _fmt(sp_tps,  "throughput")
                sep = " &nbsp;·&nbsp; " if ttft_badge and tps_badge else ""

                triton_wins = (sp_ttft or 0) >= 1.05 or (sp_tps or 0) >= 1.05
                triton_loses = (sp_ttft or 1) <= 0.95 and (sp_tps or 1) <= 0.95
                if triton_wins:
                    header = "🏆 Triton wins:"
                elif triton_loses:
                    header = "⚠ Triton slower:"
                else:
                    header = "≈ Tie:"

                status_ph.markdown(
                    f"<div class='metric-box' style='text-align:center'>"
                    f"{header} {ttft_badge}{sep}{tps_badge}</div>",
                    unsafe_allow_html=True,
                )

                st.session_state.baseline_text = b_text
                st.session_state.triton_text   = t_text
                st.session_state.race_replay   = event
                st.session_state.demo_done     = True

            elif phase == "error":
                st.error(f"Demo error: {event['message']}")
                break

        st.session_state.demo_running = False

    # Restore completed state on re-render
    elif st.session_state.demo_done:
        _replay = st.session_state.get("race_replay") or {}
        baseline = _replay.get("baseline", {})
        triton   = _replay.get("triton", {})
        if baseline:
            base_speed_ph.markdown(
                f"<span class='speed-counter baseline-speed'>{baseline.get('tps', 0):.1f} tok/s</span>",
                unsafe_allow_html=True,
            )
            base_ttft_ph.markdown(
                _render_metrics(baseline, "#f38ba8"),
                unsafe_allow_html=True,
            )
            base_output_ph.markdown(
                f"<div class='token-stream'>{st.session_state.get('baseline_text', '')}</div>",
                unsafe_allow_html=True,
            )
        if triton:
            triton_speed_ph.markdown(
                f"<span class='speed-counter triton-speed'>{triton.get('tps', 0):.1f} tok/s</span>",
                unsafe_allow_html=True,
            )
            triton_ttft_ph.markdown(
                _render_metrics(triton, "#a6e3a1"),
                unsafe_allow_html=True,
            )
            triton_output_ph.markdown(
                f"<div class='token-stream'>{st.session_state.get('triton_text', '')}</div>",
                unsafe_allow_html=True,
            )
