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
        "baseline_text":  "",
        "triton_text":    "",
        "baseline_tps":   0.0,
        "triton_tps":     0.0,
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
        "Baseline uses PyTorch SDPA · Optimized uses the Triton kernel found by the AI agent · "
        "Same prompt, same model weights, same A100."
    )

    if not st.session_state.opt_done:
        st.info(
            "⬅  Run the Agent Optimizer first to generate an optimized kernel, "
            "then come back here to see the inference speedup.",
            icon="ℹ️",
        )
    else:
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
        max_tokens = st.slider("Max new tokens", 50, 300, 150)

        demo_btn = st.button(
            "▶  Run Inference Race",
            type="primary",
            disabled=st.session_state.demo_running,
            use_container_width=False,
        )

        st.divider()

        # Side-by-side columns
        col_base, col_sep, col_triton = st.columns([10, 0.4, 10])

        with col_base:
            st.markdown(
                "### PyTorch SDPA  <span style='color:#f38ba8;font-size:0.7em'>(baseline)</span>",
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
                "### Triton Optimized  <span style='color:#a6e3a1;font-size:0.7em'>⚡</span>",
                unsafe_allow_html=True,
            )
            triton_speed_ph  = st.empty()
            triton_ttft_ph   = st.empty()
            triton_output_ph = st.empty()

        # Summary row (shown after both finish)
        summary_row = st.empty()

        # ── Run the demo ──────────────────────────────────────────────────

        if demo_btn:
            st.session_state.demo_running = True
            st.session_state.baseline_text = ""
            st.session_state.triton_text   = ""
            st.session_state.demo_done     = False

            import modal

            # Kernel code is passed directly to the Modal container —
            # no filesystem mounting needed, exec'd inside run_inference_comparison.
            inference_fn = modal.Function.from_name(
                "qwen3-kernel-optimizer", "run_inference_comparison"
            )

            baseline_tps_final  = 0.0
            triton_tps_final    = 0.0
            baseline_time_final = 0.0
            triton_time_final   = 0.0
            baseline_ttft_final = 0.0
            triton_ttft_final   = 0.0

            base_speed_ph.markdown(
                "<span class='speed-counter baseline-speed'>⏳ loading...</span>",
                unsafe_allow_html=True,
            )
            triton_speed_ph.markdown(
                "<span class='speed-counter triton-speed'>⏳ waiting...</span>",
                unsafe_allow_html=True,
            )

            for event in inference_fn.remote_gen(prompt, st.session_state.best_kernel, max_tokens):
                phase = event.get("phase")

                if phase == "loading":
                    base_speed_ph.markdown(
                        "<span class='speed-counter baseline-speed'>🔄 loading model...</span>",
                        unsafe_allow_html=True,
                    )

                elif phase == "baseline_start":
                    base_speed_ph.markdown(
                        "<span class='speed-counter baseline-speed'>🔄 generating...</span>",
                        unsafe_allow_html=True,
                    )

                elif phase == "baseline_done":
                    tps  = event["tokens_per_sec"]
                    text = event["text"]
                    baseline_tps_final  = tps
                    baseline_time_final = event["time_ms"]
                    baseline_ttft_final = event.get("ttft_ms", 0.0)
                    st.session_state.baseline_text = text
                    st.session_state.baseline_tps  = tps
                    st.session_state.baseline_ttft = baseline_ttft_final

                    base_speed_ph.markdown(
                        f"<span class='speed-counter baseline-speed'>{tps:.1f} tok/s</span>",
                        unsafe_allow_html=True,
                    )
                    base_ttft_ph.markdown(
                        f"<div style='color:#cdd6f4;font-size:0.85em;margin-top:-0.3em'>"
                        f"TTFT: <b>{baseline_ttft_final:.0f} ms</b></div>",
                        unsafe_allow_html=True,
                    )
                    base_output_ph.markdown(
                        f"<div class='token-stream'>{text}</div>",
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
                    # Streaming cursor effect
                    triton_output_ph.markdown(
                        f"<div class='token-stream'>{st.session_state.triton_text}▌</div>",
                        unsafe_allow_html=True,
                    )

                elif phase == "triton_done":
                    triton_tps_final  = event["tokens_per_sec"]
                    triton_time_final = event["time_ms"]
                    triton_ttft_final = event.get("ttft_ms", 0.0)
                    speedup           = event["speedup"]
                    ttft_speedup      = event.get("ttft_speedup", 1.0)
                    st.session_state.triton_tps  = triton_tps_final
                    st.session_state.triton_ttft = triton_ttft_final

                    triton_speed_ph.markdown(
                        f"<span class='speed-counter triton-speed'>{triton_tps_final:.1f} tok/s</span>",
                        unsafe_allow_html=True,
                    )
                    triton_ttft_ph.markdown(
                        f"<div style='color:#a6e3a1;font-size:0.85em;margin-top:-0.3em'>"
                        f"TTFT: <b>{triton_ttft_final:.0f} ms</b>"
                        f"&nbsp;<span style='color:#f9e2af'>({ttft_speedup:.1f}× faster)</span></div>",
                        unsafe_allow_html=True,
                    )
                    triton_output_ph.markdown(
                        f"<div class='token-stream'>{st.session_state.triton_text}</div>",
                        unsafe_allow_html=True,
                    )

                    summary_row.markdown(
                        f"""
                        <div class='metric-box' style='margin-top:1em;text-align:center'>
                        <span style='font-size:1.1em'>
                        Throughput: <b>{baseline_tps_final:.1f}</b> → <b>{triton_tps_final:.1f} tok/s</b>
                        &nbsp;<span class='speedup-badge'>{speedup}×</span>&nbsp;
                        &nbsp;|&nbsp;
                        TTFT: <b>{baseline_ttft_final:.0f}</b> → <b>{triton_ttft_final:.0f} ms</b>
                        &nbsp;<span class='speedup-badge'>{ttft_speedup:.1f}×</span>
                        </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.session_state.demo_done = True

                elif phase == "error":
                    st.error(f"Demo error: {event['message']}")
                    break

            st.session_state.demo_running = False

        # Restore completed state on re-render
        elif st.session_state.demo_done:
            _b_tps  = st.session_state.baseline_tps
            _t_tps  = st.session_state.triton_tps
            _b_ttft = st.session_state.get("baseline_ttft", 0.0)
            _t_ttft = st.session_state.get("triton_ttft", 0.0)
            _tput_x = round(_t_tps / _b_tps, 2) if _b_tps > 0 else 1.0
            _ttft_x = round(_b_ttft / _t_ttft, 2) if _t_ttft > 0 else 1.0

            base_speed_ph.markdown(
                f"<span class='speed-counter baseline-speed'>{_b_tps:.1f} tok/s</span>",
                unsafe_allow_html=True,
            )
            base_ttft_ph.markdown(
                f"<div style='color:#cdd6f4;font-size:0.85em;margin-top:-0.3em'>"
                f"TTFT: <b>{_b_ttft:.0f} ms</b></div>",
                unsafe_allow_html=True,
            )
            base_output_ph.markdown(
                f"<div class='token-stream'>{st.session_state.baseline_text}</div>",
                unsafe_allow_html=True,
            )
            triton_speed_ph.markdown(
                f"<span class='speed-counter triton-speed'>{_t_tps:.1f} tok/s</span>",
                unsafe_allow_html=True,
            )
            triton_ttft_ph.markdown(
                f"<div style='color:#a6e3a1;font-size:0.85em;margin-top:-0.3em'>"
                f"TTFT: <b>{_t_ttft:.0f} ms</b>"
                f"&nbsp;<span style='color:#f9e2af'>({_ttft_x:.1f}× faster)</span></div>",
                unsafe_allow_html=True,
            )
            triton_output_ph.markdown(
                f"<div class='token-stream'>{st.session_state.triton_text}</div>",
                unsafe_allow_html=True,
            )
            summary_row.markdown(
                f"""
                <div class='metric-box' style='margin-top:1em;text-align:center'>
                <span style='font-size:1.1em'>
                Throughput: <b>{_b_tps:.1f}</b> → <b>{_t_tps:.1f} tok/s</b>
                &nbsp;<span class='speedup-badge'>{_tput_x}×</span>&nbsp;
                &nbsp;|&nbsp;
                TTFT: <b>{_b_ttft:.0f}</b> → <b>{_t_ttft:.0f} ms</b>
                &nbsp;<span class='speedup-badge'>{_ttft_x:.1f}×</span>
                </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
