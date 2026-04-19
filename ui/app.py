"""
Emile — Streamlit host for the Ratatouille-themed UI.

Two tabs:
  Tab 1 · Kitchen       : live optimization loop (timeline + stream + dashboard)
  Tab 2 · Tasting Room  : side-by-side Qwen3-4B inference race (baseline vs Triton)

Run with:
    streamlit run ui/app.py
"""
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.theme import (
    get_streamlit_overrides,
    COPPER,
    SAFFRON,
    BASIL,
    BORDEAUX,
    PARCHMENT,
    PARCHMENT_DIM,
)

import base64

@st.cache_data
def _logo_data_uri() -> str:
    p = Path(__file__).parent / "assets" / "logo.webp"
    return "data:image/webp;base64," + base64.b64encode(p.read_bytes()).decode()
from ui.components.shell import render as render_shell

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Emile · Kernel Kitchen",
    page_icon="🐀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(get_streamlit_overrides(), unsafe_allow_html=True)

# Host-only brandbar styling (lives in the Streamlit shell, not the iframe)
st.markdown(
    """
    <style>
    /* Pull the main container flush to the top */
    div.stMainBlockContainer,
    .stMainBlockContainer.block-container,
    [data-testid="stAppViewContainer"] .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }

    .em-brandbar {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.25rem 0 0.9rem 0;
        margin: 0 0 0.4rem 0;
        border-bottom: 1px solid rgba(201,123,74,0.22);
        position: relative;
    }
    .em-brandbar::after {
        content: "";
        position: absolute;
        left: 0;
        bottom: -1px;
        width: 3.6rem;
        height: 2px;
        background: linear-gradient(90deg, #C97B4A, transparent);
    }
    .em-brand-logo {
        width: 56px;
        height: 56px;
        flex-shrink: 0;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        overflow: hidden;
        background: radial-gradient(circle at 40% 30%, rgba(232,177,74,0.12), transparent 70%);
    }
    .em-brand-logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
    }
    .em-brand-text {
        display: flex;
        flex-direction: column;
        gap: 0.05rem;
        line-height: 1.05;
    }
    .em-wordmark-big {
        font-family: 'Fraunces', ui-serif, Georgia, serif;
        font-weight: 600;
        font-variation-settings: "opsz" 72;
        font-size: 2.4rem;
        letter-spacing: -0.025em;
        color: #F4ECD8;
        line-height: 1;
    }
    .em-wordmark-big .em-accent { color: #C97B4A; }
    .em-wordmark-big .em-hat {
        display: inline-block;
        width: 0.22em;
        height: 0.22em;
        background: #C97B4A;
        border-radius: 50%;
        margin-left: 0.05em;
        vertical-align: 0.18em;
        box-shadow: 0 0 0 3px rgba(201,123,74,0.18);
    }
    .em-tagline {
        font-family: 'Fraunces', ui-serif, Georgia, serif !important;
        font-style: italic;
        font-size: 0.95rem;
        color: #F4ECD8 !important;
        opacity: 0.82;
        margin-top: 0.15rem;
    }
    .em-brand-meta {
        margin-left: auto;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 0.2rem;
        text-align: right;
    }
    .em-meta-chip {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        color: #C9BFA8;
        text-transform: uppercase;
        padding: 0.25rem 0.65rem;
        border: 1px solid #3A2E28;
        border-radius: 999px;
        background: #0E0706;
    }
    .em-meta-chip .em-dot {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #5E8A5A;
        margin-right: 0.4rem;
        vertical-align: 1px;
        box-shadow: 0 0 6px #5E8A5A;
    }
    .em-meta-sub {
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-size: 0.7rem;
        color: #6B6258;
        letter-spacing: 0.02em;
    }
    @media (max-width: 720px) {
        .em-brand-meta { display: none; }
        .em-wordmark-big { font-size: 1.9rem; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "opt_running":      False,
        "opt_done":         False,
        "opt_events":       [],
        "best_kernel":      None,
        "best_metrics":     None,
        "baseline_metrics": None,
        "tflops_history":   [],
        "demo_running":     False,
        "baseline_rec":     None,
        "triton_rec":       None,
        "race_replay":      None,
        "demo_done":        False,
        "demo_phase":       "idle",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def _load_reference_kernel() -> str:
    p = Path(__file__).parent.parent / "kernels" / "pytorch_reference_kernel.py"
    return p.read_text()


V1_KERNEL = _load_reference_kernel()

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


def _build_state(page: str, config: dict) -> dict:
    """Assemble the shell state dict from st.session_state."""
    return {
        "page": page,
        "events":           list(st.session_state.opt_events),
        "running":          st.session_state.opt_running,
        "done":             st.session_state.opt_done,
        "config":           config,
        "baseline_metrics": st.session_state.baseline_metrics,
        "best_metrics":     st.session_state.best_metrics,
        "tflops_history":   list(st.session_state.tflops_history),
        "baseline_rec":     st.session_state.baseline_rec,
        "triton_rec":       st.session_state.triton_rec,
        "race_replay":      st.session_state.race_replay,
        "demo_phase":       st.session_state.demo_phase,
        "demo_done":        st.session_state.demo_done,
    }


def _render_shell_in(slot, state: dict, height: int):
    """Render a full shell doc into a Streamlit components slot."""
    with slot:
        components.html(render_shell(state), height=height, scrolling=True)


# ---------------------------------------------------------------------------
# Brand bar
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="em-brandbar">
      <div class="em-brand-logo"><img src="{_logo_data_uri()}" alt="Emile" /></div>
      <div class="em-brand-text">
        <span class="em-wordmark-big">em<span class="em-accent">i</span>le</span>
        <span class="em-tagline">Anyone can cook. Anyone can write expert kernels.</span>
      </div>
      <div class="em-brand-meta">
        <span class="em-meta-chip"><span class="em-dot"></span>A100 · 80GB</span>
        <span class="em-meta-sub">qwen3-4b · triton · pytorch</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_agent, tab_demo = st.tabs(["🧑‍🍳  Kitchen", "🔥  Tasting Room"])

# ============================================================================
# TAB 1 — KITCHEN (Agent Optimizer)
# ============================================================================

with tab_agent:
    col_kernel, col_gap, col_config = st.columns([3, 0.2, 2])

    with col_kernel:
        st.markdown("##### Reference recipe")
        st.caption("The PyTorch baseline — Emile writes a Triton kernel that beats it.")
        kernel_editor = st.text_area(
            label="kernel_code",
            value=V1_KERNEL,
            height=720,
            label_visibility="collapsed",
        )

    with col_config:
        st.markdown("##### Service parameters")
        seq_len = st.selectbox("Sequence length", [512, 1024, 2048, 4096], index=2)
        d_head  = st.selectbox("Head dimension",  [64, 128], index=1)
        n_heads = st.selectbox("Query heads",     [8, 16, 32], index=2)
        batch   = st.slider("Batch size", 1, 8, 1)
        target_pct = st.slider(
            "Efficiency target (%)", 1.0, 100.0, 70.0, 0.5,
            help="Stop when the Triton kernel's compute efficiency exceeds this % of A100 peak (312 TFLOPS).",
        )
        max_iter = st.slider("Max iterations", 2, 8, 5)

        run_btn = st.button(
            "▶  Start cooking",
            type="primary",
            disabled=st.session_state.opt_running,
            use_container_width=True,
        )

    st.markdown("")

    config = {
        "seq_len":    seq_len,
        "d_head":     d_head,
        "n_heads":    n_heads,
        "batch":      batch,
        "target_pct": target_pct,
        "max_iter":   max_iter,
    }

    # Live surface — one iframe re-rendered on each event
    agent_slot = st.empty()
    _render_shell_in(agent_slot, _build_state("agent", config), height=1500)

    if run_btn:
        st.session_state.opt_running      = True
        st.session_state.opt_done         = False
        st.session_state.opt_events       = []
        st.session_state.tflops_history   = []
        st.session_state.best_kernel      = None
        st.session_state.best_metrics     = None
        st.session_state.baseline_metrics = None

        from agent import run_optimization_agent

        agent_config = {
            "seq_len": seq_len,
            "d_head":  d_head,
            "n_heads": n_heads,
            "batch":   batch,
        }

        _render_shell_in(agent_slot, _build_state("agent", config), height=1500)

        for event in run_optimization_agent(
            kernel_code=kernel_editor,
            config=agent_config,
            max_iterations=max_iter,
        ):
            st.session_state.opt_events.append(event)
            etype = event["type"]

            if etype == "metrics":
                d   = event["data"]
                itr = event["iteration"]
                st.session_state.tflops_history.append((itr, d["tflops"]))
                if itr == 0:
                    st.session_state.baseline_metrics = d
                if (st.session_state.best_metrics is None
                        or d["tflops"] > st.session_state.best_metrics["tflops"]):
                    st.session_state.best_metrics = d

            elif etype == "done":
                st.session_state.best_kernel  = event.get("best_code")
                st.session_state.best_metrics = event.get("best_metrics") or st.session_state.best_metrics
                st.session_state.opt_done     = True

            # Re-render iframe with updated state
            _render_shell_in(agent_slot, _build_state("agent", config), height=1500)

        st.session_state.opt_running = False
        _render_shell_in(agent_slot, _build_state("agent", config), height=1500)

    if st.session_state.opt_done and st.session_state.best_kernel:
        st.markdown("##### The winning kernel")
        st.code(st.session_state.best_kernel, language="python")
        st.download_button(
            "⬇  Download optimized kernel",
            data=st.session_state.best_kernel,
            file_name="optimized_attention.py",
            mime="text/plain",
        )


# ============================================================================
# TAB 2 — TASTING ROOM (Live Inference Demo)
# ============================================================================

with tab_demo:
    st.markdown("##### Side-by-side Qwen3-4B inference")
    st.caption(
        "PyTorch SDPA (reference) vs the Triton kernel below · "
        "same prompt, same weights, same A100."
    )

    _demo_kernel_default = (
        st.session_state.best_kernel
        if st.session_state.opt_done and st.session_state.best_kernel
        else _DEBUG_TRITON_KERNEL
    )

    demo_col_left, demo_col_right = st.columns([3, 2])
    with demo_col_left:
        prompt = st.text_area(
            "Prompt",
            height=260,
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
            "Context length (tokens)", 128, 4096, 1024, 128,
            help=(
                "Pad the prompt to this many tokens before generation. "
                "Larger values make prefill dominate and surface the real "
                "Triton vs PyTorch-reference speedup."
            ),
        )
        if not st.session_state.opt_done:
            st.info("Using built-in debug kernel. Run the Kitchen tab to bring a better one.", icon="ℹ️")
        else:
            st.success("Using Emile's best kernel.", icon="✨")

    with st.expander("Triton kernel used for inference (editable)", expanded=False):
        demo_kernel_code = st.text_area(
            label="demo_kernel",
            value=_demo_kernel_default,
            height=300,
            label_visibility="collapsed",
        )

    demo_btn = st.button(
        "▶  Run inference race",
        type="primary",
        disabled=st.session_state.demo_running,
        use_container_width=False,
    )

    demo_slot = st.empty()
    _render_shell_in(demo_slot, _build_state("demo", config if 'config' in dir() else {}), height=900)

    if demo_btn:
        st.session_state.demo_running = True
        st.session_state.baseline_rec = None
        st.session_state.triton_rec   = None
        st.session_state.race_replay  = None
        st.session_state.demo_done    = False
        st.session_state.demo_phase   = "loading"

        import modal

        inference_fn = modal.Function.from_name(
            "qwen3-kernel-optimizer", "run_inference_comparison"
        )

        _render_shell_in(demo_slot, _build_state("demo", {}), height=900)

        for event in inference_fn.remote_gen(prompt, demo_kernel_code, max_tokens, context_tokens):
            phase = event.get("phase")

            if phase == "loading":
                st.session_state.demo_phase = "loading"

            elif phase == "recording_start":
                side = event.get("side", "")
                st.session_state.demo_phase = f"recording_{side}"

            elif phase == "recording_done":
                side = event.get("side", "")
                rec = {k: v for k, v in event.items() if k not in ("phase", "side")}
                if side == "baseline":
                    st.session_state.baseline_rec = rec
                elif side == "triton":
                    st.session_state.triton_rec = rec

            elif phase == "race_replay":
                st.session_state.race_replay = {
                    "baseline":     event.get("baseline"),
                    "triton":       event.get("triton"),
                    "speedup_ttft": event.get("speedup_ttft"),
                    "speedup_tps":  event.get("speedup_tps"),
                }
                st.session_state.demo_phase = "complete"
                st.session_state.demo_done  = True

            elif phase == "error":
                st.session_state.demo_phase = "error"
                st.error(f"Demo error: {event.get('message', 'unknown')}")
                break

            _render_shell_in(demo_slot, _build_state("demo", {}), height=900)

        st.session_state.demo_running = False
        _render_shell_in(demo_slot, _build_state("demo", {}), height=900)
