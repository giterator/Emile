"""Race component — kernel head-to-head + token stream for the inference demo.

Entry point: render(state: dict) -> str

Layout:
  * Top: "race panel" — two plates (PyTorch reference vs Triton kernel)
    flanking a center speedup chip.
  * Bottom: token stream card showing streamed Triton output with tok/s,
    TTFT chip, blinking cursor during streaming, and phase indicator.

All visuals are inline HTML/CSS + hand-rolled SVG (flame icon). JS is
only used to auto-scroll the token stream body to the bottom on render.
"""

from __future__ import annotations

import html
import math
from typing import Optional

# Color constants — mirrors theme.py
C_ESPRESSO = "#1A0F0B"
C_STEEL = "#2A2420"
C_STEEL_2 = "#3A2E28"
C_PARCHMENT = "#F4ECD8"
C_PARCHMENT_DIM = "#C9BFA8"
C_COPPER = "#C97B4A"
C_COPPER_GLOW = "#E09968"
C_SAFFRON = "#E8B14A"
C_BASIL = "#5E8A5A"
C_BORDEAUX = "#7A1F2B"
C_SLATE = "#6B6258"


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def _esc(v) -> str:
    return html.escape(str(v), quote=True)


def _fmt(x: Optional[float], digits: int = 2, dash: str = "—") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return dash
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return dash


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ----------------------------------------------------------------------------
# SVG glyphs
# ----------------------------------------------------------------------------

_FLAME_SVG = (
    '<svg class="rc-flame" viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">'
    '<defs><linearGradient id="rcFlameGrad" x1="0" x2="0" y1="1" y2="0">'
    f'<stop offset="0%" stop-color="{C_BORDEAUX}"/>'
    f'<stop offset="55%" stop-color="{C_COPPER}"/>'
    f'<stop offset="100%" stop-color="{C_SAFFRON}"/>'
    '</linearGradient></defs>'
    '<path d="M12 2c.9 3.2-.6 4.8-2 6.5C8.5 10.3 7 12 7 14.5 7 18 9.5 21 12.5 21S18 18.5 18 15c0-2.2-1.2-3.6-2-5 1.2 0 2 .8 2.5 1.8C18.7 9 16.3 6 14 4.5c-.2 1.2-.6 2.2-2 3 .6-2 .3-4-1-5z" '
    'fill="url(#rcFlameGrad)"/>'
    '</svg>'
)

_FLAME_SVG_SMALL = (
    '<svg class="rc-flame-sm" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">'
    f'<path d="M12 2c.9 3.2-.6 4.8-2 6.5C8.5 10.3 7 12 7 14.5 7 18 9.5 21 12.5 21S18 18.5 18 15c0-2.2-1.2-3.6-2-5 1.2 0 2 .8 2.5 1.8C18.7 9 16.3 6 14 4.5c-.2 1.2-.6 2.2-2 3 .6-2 .3-4-1-5z" '
    f'fill="{C_SAFFRON}"/></svg>'
)


# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------

_CSS = """
<style>
.rc-root { width:100%; color: var(--parchment, #F4ECD8); font-family: Inter, system-ui, sans-serif; }

/* ---------------- Race panel ---------------- */
.rc-race {
  background: var(--steel, #2A2420);
  border: 1px solid var(--steel-2, #3A2E28);
  border-radius: 12px;
  padding: 18px 18px 14px 18px;
  margin-bottom: 14px;
  position: relative;
  overflow: hidden;
}
.rc-race::before {
  content:""; position:absolute; inset:0; pointer-events:none;
  background: radial-gradient(140% 100% at 50% 0%, rgba(232,177,74,0.06), transparent 60%);
}
.rc-race-grid {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 16px;
  align-items: stretch;
}
@media (max-width: 720px) {
  .rc-race-grid { grid-template-columns: 1fr; }
}
.rc-plate {
  padding: 10px 12px;
  border-radius: 10px;
  background: linear-gradient(180deg, rgba(0,0,0,0.12), rgba(0,0,0,0));
  display: flex; flex-direction: column; justify-content: space-between; min-height: 140px;
}
.rc-plate.ref { border-left: 3px solid var(--bordeaux, #7A1F2B); }
.rc-plate.trt { border-left: 3px solid var(--basil, #5E8A5A); }
.rc-plate .rc-label {
  font-family: Fraunces, serif; font-weight: 500; font-size: 1.05rem;
  letter-spacing: .01em;
}
.rc-plate.ref .rc-label { color: var(--bordeaux, #7A1F2B); }
.rc-plate.trt .rc-label { color: var(--basil, #5E8A5A); }
.rc-big {
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 2.4rem; font-weight: 500;
  color: var(--parchment, #F4ECD8); line-height: 1;
  margin-top: 6px;
}
.rc-big .rc-unit { font-size: 1rem; color: var(--parchment-dim, #C9BFA8); margin-left: 4px; font-weight:400; }
.rc-sec {
  margin-top: 4px; font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: .82rem; color: var(--parchment-dim, #C9BFA8);
}
.rc-bar {
  margin-top: 10px; height: 8px; background: rgba(0,0,0,0.25);
  border-radius: 4px; overflow: hidden;
}
.rc-bar > div {
  height: 100%; border-radius: 4px;
  transition: width 420ms ease-out;
}
.rc-bar.ref > div { background: linear-gradient(90deg, #5A1621, var(--bordeaux, #7A1F2B)); }
.rc-bar.trt > div { background: linear-gradient(90deg, #47683F, var(--basil, #5E8A5A)); }

.rc-center {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  min-width: 140px; padding: 0 6px;
}
.rc-speedup {
  font-family: Fraunces, serif; font-weight: 600; font-size: 3rem; line-height: 1;
  color: var(--saffron, #E8B14A);
  text-shadow: 0 0 24px rgba(232,177,74,0.22);
  display: flex; align-items: center; gap: 6px;
}
.rc-speedup.slow { color: var(--bordeaux, #7A1F2B); text-shadow: none; }
.rc-speedup-sub {
  margin-top: 6px;
  font-family: Inter, sans-serif;
  text-transform: uppercase; letter-spacing: .18em;
  font-size: 11px; color: var(--parchment-dim, #C9BFA8);
}
.rc-flame { filter: drop-shadow(0 0 6px rgba(232,177,74,0.35)); }

.rc-race-sub {
  margin-top: 12px;
  font-family: Inter, sans-serif; font-size: 12px;
  color: var(--parchment-dim, #C9BFA8);
  text-align: center;
  letter-spacing: .02em;
}

.rc-race-empty {
  display: flex; align-items: center; justify-content: center;
  min-height: 140px;
  font-family: Inter, sans-serif; font-size: 14px;
  color: var(--copper, #C97B4A);
  letter-spacing: .04em;
}
.rc-race-empty .dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 999px;
  background: var(--copper, #C97B4A); margin: 0 4px;
  animation: rcPulse 1200ms ease-in-out infinite;
}
.rc-race-empty .dot:nth-child(2) { animation-delay: 150ms; }
.rc-race-empty .dot:nth-child(3) { animation-delay: 300ms; }
@keyframes rcPulse {
  0%,100% { opacity:.25; transform: scale(0.9); }
  50%     { opacity:1;   transform: scale(1.15); }
}

/* ---------------- Token stream ---------------- */
.rc-stream {
  background: var(--steel, #2A2420);
  border: 1px solid var(--steel-2, #3A2E28);
  border-radius: 12px;
  padding: 14px 16px;
  position: relative;
}
.rc-stream-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 12px; margin-bottom: 10px;
}
.rc-stream-title {
  display: flex; align-items: center; gap: 8px;
  font-family: Fraunces, serif; font-size: 1.05rem; font-weight: 500;
  color: var(--parchment, #F4ECD8);
}
.rc-tps {
  font-family: Fraunces, serif; font-size: 1.6rem; font-weight: 600;
  color: var(--basil, #5E8A5A); line-height: 1;
}
.rc-tps .u { font-family: Inter, sans-serif; font-size: 0.75rem;
  color: var(--parchment-dim, #C9BFA8); margin-left: 4px; font-weight:400; }
.rc-ttft {
  display:inline-block; margin-left: 8px;
  padding: 3px 9px; border-radius: 999px; border:1px solid var(--steel-2, #3A2E28);
  background: rgba(201,123,74,0.08);
  color: var(--copper, #C97B4A);
  font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 11px;
}
.rc-stream-body {
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 13px; line-height: 1.55;
  color: var(--parchment, #F4ECD8);
  background: rgba(0,0,0,0.25);
  border-radius: 8px;
  padding: 12px 14px;
  white-space: pre-wrap;
  word-break: break-word;
  min-height: 280px; max-height: 440px;
  overflow: auto;
}
.rc-cursor {
  display: inline-block; width: 0.55em; height: 1.1em;
  background: var(--copper-glow, #E09968);
  vertical-align: -0.18em; margin-left: 2px;
  animation: rcBlink 900ms steps(1, end) infinite;
}
@keyframes rcBlink { 50% { opacity: 0; } }

.rc-stream-foot {
  display: flex; align-items: center; justify-content: space-between;
  margin-top: 10px; gap: 10px; flex-wrap: wrap;
}
.rc-phase {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 999px;
  font-family: Inter, sans-serif; font-size: 11px;
  text-transform: uppercase; letter-spacing: .12em; font-weight: 600;
  border: 1px solid transparent;
}
.rc-phase .pd { width:6px; height:6px; border-radius: 999px; background: currentColor; }
.rc-phase.idle       { color: var(--parchment-dim, #C9BFA8); border-color: rgba(201,191,168,.25); background: rgba(201,191,168,.05); }
.rc-phase.loading    { color: var(--copper, #C97B4A);       border-color: rgba(201,123,74,.4);  background: rgba(201,123,74,.08); }
.rc-phase.racing     { color: var(--saffron, #E8B14A);      border-color: rgba(232,177,74,.4);  background: rgba(232,177,74,.08); }
.rc-phase.generating { color: var(--basil, #5E8A5A);        border-color: rgba(94,138,90,.4);   background: rgba(94,138,90,.08); }
.rc-phase.complete   { color: var(--basil, #5E8A5A);        border-color: rgba(94,138,90,.55);  background: rgba(94,138,90,.14); }
.rc-phase.error      { color: var(--bordeaux, #7A1F2B);     border-color: rgba(122,31,43,.5);   background: rgba(122,31,43,.12); }
.rc-phase.loading .pd, .rc-phase.racing .pd, .rc-phase.generating .pd { animation: rcPulse 1200ms ease-in-out infinite; }

.em-fade-in { animation: emFadeIn 380ms ease-out both; }
@keyframes emFadeIn { from { opacity:0; transform: translateY(4px); } to { opacity:1; transform:none; } }
</style>
"""


# ----------------------------------------------------------------------------
# Race panel
# ----------------------------------------------------------------------------

def _race_panel(state: dict) -> str:
    ev = state.get("race_event")
    cfg = state.get("config") or {}
    n_heads = cfg.get("n_heads", 32)
    d_head = cfg.get("d_head", 128)
    batch = cfg.get("batch", 1)

    if not ev or not isinstance(ev, dict):
        return f"""
        <div class="rc-race em-fade-in">
          <div class="rc-race-empty">
            <span>awaiting kernel race</span>
            <span class="dot"></span><span class="dot"></span><span class="dot"></span>
          </div>
          <div class="rc-race-sub">Direct kernel race · B={_esc(batch)} H={_esc(n_heads)} N=— D={_esc(d_head)} fp16 is_causal=True</div>
        </div>
        """

    ref_ms = float(ev.get("ref_ms") or 0.0)
    trt_ms = float(ev.get("triton_ms") or 0.0)
    ref_tf = ev.get("ref_tflops")
    trt_tf = ev.get("triton_tflops")
    speedup = ev.get("speedup")
    n_tokens = ev.get("n_tokens", "—")

    # Bars normalized so the slower (longer time) kernel gets 100%.
    max_ms = max(ref_ms, trt_ms, 1e-6)
    ref_pct = _clamp(ref_ms / max_ms * 100.0, 0.0, 100.0)
    trt_pct = _clamp(trt_ms / max_ms * 100.0, 0.0, 100.0)

    # Speedup computation if missing
    if speedup is None and trt_ms > 0:
        speedup = ref_ms / trt_ms
    try:
        sp_val = float(speedup) if speedup is not None else None
    except Exception:
        sp_val = None

    if sp_val is None:
        sp_html = '<div class="rc-speedup">—</div>'
    else:
        slow_class = " slow" if sp_val < 1.0 else ""
        sp_html = (
            f'<div class="rc-speedup{slow_class}">'
            f'{_FLAME_SVG}'
            f'<span>×{_fmt(sp_val,2)}</span>'
            f'</div>'
        )

    subtitle = (
        f'Direct kernel race · B={_esc(batch)} H={_esc(n_heads)} '
        f'N={_esc(n_tokens)} D={_esc(d_head)} fp16 is_causal=True'
    )

    return f"""
    <div class="rc-race em-fade-in">
      <div class="rc-race-grid">
        <!-- PyTorch Reference -->
        <div class="rc-plate ref">
          <div>
            <div class="rc-label">PyTorch Reference</div>
            <div class="rc-big">{_fmt(ref_ms,2)}<span class="rc-unit">ms</span></div>
            <div class="rc-sec">{_fmt(ref_tf,1)} TFLOPS</div>
          </div>
          <div class="rc-bar ref"><div style="width:{ref_pct:.1f}%"></div></div>
        </div>

        <!-- Speedup chip -->
        <div class="rc-center">
          {sp_html}
          <div class="rc-speedup-sub">{'slower' if (sp_val is not None and sp_val < 1.0) else 'faster'}</div>
        </div>

        <!-- Triton Kernel -->
        <div class="rc-plate trt">
          <div>
            <div class="rc-label">Triton Kernel</div>
            <div class="rc-big">{_fmt(trt_ms,2)}<span class="rc-unit">ms</span></div>
            <div class="rc-sec">{_fmt(trt_tf,1)} TFLOPS</div>
          </div>
          <div class="rc-bar trt"><div style="width:{trt_pct:.1f}%"></div></div>
        </div>
      </div>
      <div class="rc-race-sub">{subtitle}</div>
    </div>
    """


# ----------------------------------------------------------------------------
# Token stream
# ----------------------------------------------------------------------------

_PHASES = {"idle", "loading", "racing", "generating", "complete", "error"}


def _token_stream(state: dict) -> str:
    text = state.get("triton_text") or ""
    tps = state.get("triton_tps")
    ttft = state.get("triton_ttft")
    phase = str(state.get("demo_phase") or "idle").lower()
    if phase not in _PHASES:
        phase = "idle"

    tps_html = (
        f'<div class="rc-tps">{_fmt(tps,1)}<span class="u">tok/s</span></div>'
        if (tps is not None) else
        '<div class="rc-tps">—<span class="u">tok/s</span></div>'
    )
    ttft_html = ""
    try:
        if ttft is not None and float(ttft) > 0:
            ttft_html = f'<span class="rc-ttft">TTFT {_fmt(ttft,0)} ms</span>'
    except Exception:
        ttft_html = ""

    show_cursor = phase in ("generating", "racing", "loading")
    cursor_html = '<span class="rc-cursor"></span>' if show_cursor else ''

    body_text = _esc(text) if text else '<span style="color:var(--parchment-dim);">— no output yet —</span>' if not show_cursor else ''

    phase_label = {
        "idle": "idle",
        "loading": "loading model",
        "racing": "racing kernels",
        "generating": "generating",
        "complete": "complete",
        "error": "error",
    }.get(phase, phase)

    # Auto-scroll script: target body by unique id per render.
    # Streamlit may re-render this HTML fragment, so using a stable id is fine.
    stream_id = "rc-stream-body"

    js = f"""
    <script>
      (function(){{
        try {{
          var el = document.getElementById("{stream_id}");
          if (el) {{ el.scrollTop = el.scrollHeight; }}
        }} catch (e) {{}}
      }})();
    </script>
    """

    return f"""
    <div class="rc-stream em-fade-in">
      <div class="rc-stream-head">
        <div class="rc-stream-title">{_FLAME_SVG_SMALL} Triton Kernel Output</div>
        <div style="display:flex;align-items:center;gap:8px;">
          {tps_html}
          {ttft_html}
        </div>
      </div>
      <div class="rc-stream-body" id="{stream_id}">{body_text}{cursor_html}</div>
      <div class="rc-stream-foot">
        <span class="rc-phase {phase}"><span class="pd"></span>{_esc(phase_label)}</span>
        <span style="font-family:Inter,sans-serif;font-size:11px;color:var(--parchment-dim,#C9BFA8);letter-spacing:.04em;">
          {_esc(len(text))} chars streamed
        </span>
      </div>
      {js}
    </div>
    """


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def render(state: dict) -> str:
    """Return HTML fragment for the inference demo: race panel + token stream."""
    if not isinstance(state, dict):
        state = {}

    return "\n".join([
        _CSS,
        '<div class="rc-root">',
        _race_panel(state),
        _token_stream(state),
        '</div>',
    ])
