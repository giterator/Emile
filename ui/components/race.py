"""Race component — side-by-side baseline vs Triton inference recordings.

Consumes the run_inference_comparison event schema:
  state["baseline_rec"]  — recording dict (or None while still running)
  state["triton_rec"]    — recording dict (or None)
  state["race_replay"]   — final payload with speedup_ttft / speedup_tps
  state["demo_phase"]    — "idle" | "loading" | "recording_baseline" |
                          "recording_triton" | "complete" | "error"

Recording dict shape:
  {
    "tokens":  [{"text": str, "elapsed_ms": float}, ...],
    "ttft_ms": float, "total_ms": float, "tps": float, "count": int,
  }

The Triton side additionally includes hook_total / hook_triton / hook_sdpa.

When both recordings are present, a small vanilla-JS replay animates both
streams simultaneously using their recorded timestamps — so the visual race
is honest (Triton side finishes first by the measured ratio).
"""

from __future__ import annotations

import html
import json
import math
from typing import Optional

C_BORDEAUX = "#7A1F2B"
C_BASIL = "#5E8A5A"
C_COPPER = "#C97B4A"
C_SAFFRON = "#E8B14A"
C_PARCHMENT_DIM = "#C9BFA8"


def _esc(v) -> str:
    return html.escape(str(v), quote=True)


def _fmt(x: Optional[float], digits: int = 2, dash: str = "—") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return dash
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return dash


_FLAME_SVG = (
    '<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true" '
    'style="filter:drop-shadow(0 0 6px rgba(232,177,74,0.35));">'
    '<defs><linearGradient id="rcFlameGrad" x1="0" x2="0" y1="1" y2="0">'
    f'<stop offset="0%" stop-color="{C_BORDEAUX}"/>'
    f'<stop offset="55%" stop-color="{C_COPPER}"/>'
    f'<stop offset="100%" stop-color="{C_SAFFRON}"/>'
    '</linearGradient></defs>'
    '<path d="M12 2c.9 3.2-.6 4.8-2 6.5C8.5 10.3 7 12 7 14.5 7 18 9.5 21 12.5 21S18 18.5 18 15c0-2.2-1.2-3.6-2-5 1.2 0 2 .8 2.5 1.8C18.7 9 16.3 6 14 4.5c-.2 1.2-.6 2.2-2 3 .6-2 .3-4-1-5z" '
    'fill="url(#rcFlameGrad)"/></svg>'
)


_CSS = """
<style>
.rc-root { width:100%; color: var(--parchment, #F4ECD8); font-family: Inter, system-ui, sans-serif; }

.rc-headline {
  display: flex; align-items: center; justify-content: space-between;
  gap: 16px; margin-bottom: 14px;
  background: var(--steel, #2A2420);
  border: 1px solid var(--steel-2, #3A2E28);
  border-radius: 12px; padding: 14px 18px;
  position: relative; overflow: hidden;
}
.rc-headline::before {
  content:""; position:absolute; inset:0; pointer-events:none;
  background: radial-gradient(120% 100% at 50% 0%, rgba(232,177,74,0.05), transparent 60%);
}
.rc-headline-left {
  font-family: Fraunces, serif; font-size: 1.1rem; font-weight: 500;
  color: var(--parchment, #F4ECD8);
  display:flex; align-items:center; gap:10px;
}
.rc-speedup {
  display: inline-flex; align-items: center; gap: 8px;
  font-family: Fraunces, serif; font-weight: 600; font-size: 2.2rem; line-height: 1;
  color: var(--saffron, #E8B14A);
  text-shadow: 0 0 24px rgba(232,177,74,0.22);
}
.rc-speedup.slow { color: var(--bordeaux, #7A1F2B); text-shadow: none; }
.rc-speedup-lbl {
  font-family: Inter, sans-serif;
  text-transform: uppercase; letter-spacing: .16em;
  font-size: 10px; color: var(--parchment-dim, #C9BFA8);
  margin-top: 3px;
}

.rc-phase {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 999px;
  font-family: Inter, sans-serif; font-size: 11px;
  text-transform: uppercase; letter-spacing: .12em; font-weight: 600;
  border: 1px solid transparent;
}
.rc-phase .pd { width:6px; height:6px; border-radius:999px; background: currentColor; }
.rc-phase.idle       { color: var(--parchment-dim,#C9BFA8); border-color: rgba(201,191,168,.25); background: rgba(201,191,168,.05); }
.rc-phase.loading    { color: var(--copper,#C97B4A);       border-color: rgba(201,123,74,.4);  background: rgba(201,123,74,.08); }
.rc-phase.recording  { color: var(--saffron,#E8B14A);      border-color: rgba(232,177,74,.4);  background: rgba(232,177,74,.08); }
.rc-phase.complete   { color: var(--basil,#5E8A5A);        border-color: rgba(94,138,90,.55);  background: rgba(94,138,90,.14); }
.rc-phase.error      { color: var(--bordeaux,#7A1F2B);     border-color: rgba(122,31,43,.5);   background: rgba(122,31,43,.12); }
.rc-phase.loading .pd, .rc-phase.recording .pd { animation: rcPulse 1200ms ease-in-out infinite; }
@keyframes rcPulse { 0%,100% { opacity:.35; transform:scale(.9); } 50% { opacity:1; transform:scale(1.15); } }

.rc-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}
@media (max-width: 900px) { .rc-grid { grid-template-columns: 1fr; } }

.rc-panel {
  background: var(--steel, #2A2420);
  border: 1px solid var(--steel-2, #3A2E28);
  border-radius: 12px; padding: 14px 16px;
  display: flex; flex-direction: column; min-height: 440px;
  position: relative;
}
.rc-panel.ref { border-top: 3px solid var(--bordeaux, #7A1F2B); }
.rc-panel.trt { border-top: 3px solid var(--basil, #5E8A5A); }

.rc-phead {
  display: flex; align-items: baseline; justify-content: space-between;
  margin-bottom: 8px; gap: 8px;
}
.rc-plabel {
  font-family: Fraunces, serif; font-weight: 500; font-size: 1.05rem;
  letter-spacing: .01em;
}
.rc-panel.ref .rc-plabel { color: var(--bordeaux, #7A1F2B); }
.rc-panel.trt .rc-plabel { color: var(--basil, #5E8A5A); }
.rc-psub {
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: .78rem; color: var(--parchment-dim, #C9BFA8);
}

.rc-stats {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
  margin-bottom: 10px;
}
.rc-stat {
  background: rgba(0,0,0,0.20); border-radius: 8px;
  padding: 8px 10px;
}
.rc-stat .k {
  font-family: Inter, sans-serif; font-size: 10px;
  color: var(--parchment-dim,#C9BFA8); text-transform: uppercase;
  letter-spacing: .12em;
}
.rc-stat .v {
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 1.05rem; color: var(--parchment,#F4ECD8);
  margin-top: 2px;
}
.rc-stat .u { font-size: .72rem; color: var(--parchment-dim,#C9BFA8); margin-left:3px; }

.rc-body {
  flex: 1;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 13px; line-height: 1.55;
  color: var(--parchment, #F4ECD8);
  background: rgba(0,0,0,0.25);
  border-radius: 8px;
  padding: 12px 14px;
  white-space: pre-wrap;
  word-break: break-word;
  min-height: 260px; max-height: 380px;
  overflow: auto;
}
.rc-cursor {
  display: inline-block; width: 0.55em; height: 1.1em;
  background: var(--copper-glow, #E09968);
  vertical-align: -0.18em; margin-left: 2px;
  animation: rcBlink 900ms steps(1, end) infinite;
}
@keyframes rcBlink { 50% { opacity: 0; } }

.rc-placeholder {
  color: var(--parchment-dim,#C9BFA8); font-style: italic; font-size: 12.5px;
}
.rc-dot {
  display: inline-block; width:6px; height:6px; border-radius:999px;
  background: var(--copper,#C97B4A); margin: 0 2px;
  animation: rcPulse 1200ms ease-in-out infinite;
}
.rc-dot:nth-child(2) { animation-delay: 150ms; }
.rc-dot:nth-child(3) { animation-delay: 300ms; }

.rc-replaybar {
  margin-top: 10px; height: 6px; background: rgba(0,0,0,0.25);
  border-radius: 4px; overflow: hidden;
}
.rc-replaybar > div {
  height: 100%; border-radius: 4px;
  transition: width 160ms linear;
}
.rc-panel.ref .rc-replaybar > div { background: linear-gradient(90deg, #5A1621, var(--bordeaux,#7A1F2B)); }
.rc-panel.trt .rc-replaybar > div { background: linear-gradient(90deg, #47683F, var(--basil,#5E8A5A)); }

.em-fade-in { animation: emFadeIn 380ms ease-out both; }
@keyframes emFadeIn { from { opacity:0; transform: translateY(4px); } to { opacity:1; transform:none; } }
</style>
"""


_PHASE_LABELS = {
    "idle":                "idle",
    "loading":             "loading model",
    "recording_baseline":  "recording baseline",
    "recording_triton":    "recording triton",
    "complete":            "complete",
    "error":               "error",
}

_PHASE_CLASSES = {
    "idle":                "idle",
    "loading":             "loading",
    "recording_baseline":  "recording",
    "recording_triton":    "recording",
    "complete":            "complete",
    "error":               "error",
}


def _stats_block(rec: Optional[dict]) -> str:
    if not rec:
        return (
            '<div class="rc-stats">'
            '<div class="rc-stat"><div class="k">TTFT</div><div class="v">—</div></div>'
            '<div class="rc-stat"><div class="k">Total</div><div class="v">—</div></div>'
            '<div class="rc-stat"><div class="k">Throughput</div><div class="v">—</div></div>'
            '</div>'
        )
    return (
        '<div class="rc-stats">'
        f'<div class="rc-stat"><div class="k">TTFT</div>'
        f'<div class="v">{_fmt(rec.get("ttft_ms"),0)}<span class="u">ms</span></div></div>'
        f'<div class="rc-stat"><div class="k">Total</div>'
        f'<div class="v">{_fmt(rec.get("total_ms"),0)}<span class="u">ms</span></div></div>'
        f'<div class="rc-stat"><div class="k">Throughput</div>'
        f'<div class="v">{_fmt(rec.get("tps"),1)}<span class="u">tok/s</span></div></div>'
        '</div>'
    )


def _panel(
    side: str,         # "ref" or "trt"
    label: str,
    sub: str,
    rec: Optional[dict],
    phase: str,
    recording_now: bool,
    replay_id: str,
) -> str:
    full_text = "".join(t.get("text", "") for t in (rec.get("tokens") if rec else []) or [])
    body_id = f"rc-body-{side}"
    bar_id = f"rc-bar-{side}"

    if rec is None:
        if recording_now:
            body_html = (
                '<span class="rc-placeholder">recording generation '
                '<span class="rc-dot"></span><span class="rc-dot"></span><span class="rc-dot"></span></span>'
            )
        elif phase == "loading":
            body_html = '<span class="rc-placeholder">waiting for model weights…</span>'
        elif phase == "complete":
            body_html = '<span class="rc-placeholder">— no output —</span>'
        else:
            body_html = '<span class="rc-placeholder">queued</span>'
    else:
        # Recording complete. If the replay script is active we start empty and
        # let JS fill it; otherwise (fallback, no-JS, or before replay primes)
        # show the full text directly.
        body_html = _esc(full_text)

    return f"""
    <div class="rc-panel {side} em-fade-in">
      <div class="rc-phead">
        <div>
          <div class="rc-plabel">{_esc(label)}</div>
          <div class="rc-psub">{_esc(sub)}</div>
        </div>
      </div>
      {_stats_block(rec)}
      <div class="rc-body" id="{body_id}">{body_html}</div>
      <div class="rc-replaybar"><div id="{bar_id}" style="width:0%"></div></div>
    </div>
    """


def _replay_script(
    baseline_tokens: list,
    triton_tokens: list,
    base_total_ms: float,
    trt_total_ms: float,
) -> str:
    """Inject a tiny replay that writes tokens into both bodies using the
    recorded per-token `elapsed_ms`. The longer-running side sets the timebase
    so the visual race respects the actual measured ratio."""
    payload = {
        "base":       baseline_tokens or [],
        "trt":        triton_tokens or [],
        "base_total": float(base_total_ms or 0.0),
        "trt_total":  float(trt_total_ms or 0.0),
    }
    data_json = json.dumps(payload)
    return f"""
    <script>
    (function() {{
      var data;
      try {{ data = {data_json}; }} catch(e) {{ return; }}
      var bRef = document.getElementById('rc-body-ref');
      var bTrt = document.getElementById('rc-body-trt');
      var barRef = document.getElementById('rc-bar-ref');
      var barTrt = document.getElementById('rc-bar-trt');
      if (!bRef || !bTrt) return;

      // Target a ~6s max visual replay but respect the real ratio.
      var realMax = Math.max(data.base_total, data.trt_total, 1);
      var visMax  = Math.min(6000, Math.max(realMax, 1500));
      var scale   = visMax / realMax;

      bRef.textContent = '';
      bTrt.textContent = '';

      function schedule(tokens, el, bar, totalMs) {{
        var i = 0;
        function step() {{
          if (i >= tokens.length) {{
            if (bar) bar.style.width = '100%';
            return;
          }}
          var tok = tokens[i];
          el.textContent += tok.text;
          el.scrollTop = el.scrollHeight;
          if (bar && totalMs > 0) {{
            bar.style.width = Math.min(100, (tok.elapsed_ms / totalMs) * 100).toFixed(1) + '%';
          }}
          i++;
          var nextElapsed = (i < tokens.length) ? tokens[i].elapsed_ms : totalMs;
          var dtReal = Math.max(0, nextElapsed - tok.elapsed_ms);
          setTimeout(step, Math.max(10, dtReal * scale));
        }}
        if (tokens.length > 0) {{
          setTimeout(step, Math.max(0, tokens[0].elapsed_ms * scale));
        }} else if (bar) {{
          bar.style.width = '100%';
        }}
      }}

      schedule(data.base, bRef, barRef, data.base_total);
      schedule(data.trt,  bTrt, barTrt, data.trt_total);
    }})();
    </script>
    """


def render(state: dict) -> str:
    if not isinstance(state, dict):
        state = {}

    phase = str(state.get("demo_phase") or "idle").lower()
    phase_class = _PHASE_CLASSES.get(phase, "idle")
    phase_label = _PHASE_LABELS.get(phase, phase)

    base_rec = state.get("baseline_rec")
    trt_rec = state.get("triton_rec")
    replay = state.get("race_replay") or {}

    # Prefer race_replay payload once present (it freezes the canonical numbers)
    final_base = (replay.get("baseline") if replay else None) or base_rec
    final_trt = (replay.get("triton") if replay else None) or trt_rec

    speedup_ttft = replay.get("speedup_ttft") if replay else None
    speedup_tps = replay.get("speedup_tps") if replay else None

    # Headline speedup: prefer end-to-end total_ms ratio (what users feel)
    overall = None
    if final_base and final_trt:
        try:
            bt = float(final_base.get("total_ms") or 0.0)
            tt = float(final_trt.get("total_ms") or 0.0)
            if tt > 0:
                overall = bt / tt
        except Exception:
            overall = None

    if overall is None:
        headline_html = (
            '<div class="rc-speedup" style="font-size:1.3rem;color:var(--parchment-dim);">'
            'awaiting race</div>'
        )
    else:
        slow = " slow" if overall < 1.0 else ""
        headline_html = (
            f'<div>'
            f'<div class="rc-speedup{slow}">{_FLAME_SVG}<span>×{_fmt(overall,2)}</span></div>'
            f'<div class="rc-speedup-lbl" style="text-align:right;">'
            f'{"slower" if overall < 1.0 else "faster end-to-end"}</div>'
            f'</div>'
        )

    sub_chips = ""
    if speedup_ttft is not None:
        sub_chips += (
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'color:var(--saffron,#E8B14A);margin-right:10px;">'
            f'TTFT ×{_fmt(speedup_ttft,2)}</span>'
        )
    if speedup_tps is not None:
        sub_chips += (
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'color:var(--basil,#5E8A5A);">tok/s ×{_fmt(speedup_tps,2)}</span>'
        )

    recording_now = phase.startswith("recording_")
    is_recording_base = phase == "recording_baseline"
    is_recording_trt = phase == "recording_triton"

    ref_panel = _panel(
        "ref", "PyTorch Reference", "eager · matmul + softmax + matmul",
        final_base, phase, is_recording_base, "ref",
    )
    trt_panel = _panel(
        "trt", "Triton Kernel", "emile · fused prefill",
        final_trt, phase, is_recording_trt, "trt",
    )

    replay_js = ""
    if replay and final_base and final_trt:
        replay_js = _replay_script(
            final_base.get("tokens") or [],
            final_trt.get("tokens") or [],
            final_base.get("total_ms") or 0.0,
            final_trt.get("total_ms") or 0.0,
        )

    return "\n".join([
        _CSS,
        '<div class="rc-root">',
        '<div class="rc-headline em-fade-in">',
        '<div class="rc-headline-left">',
        f'<span class="rc-phase {phase_class}"><span class="pd"></span>{_esc(phase_label)}</span>',
        '<span style="color:var(--parchment-dim);font-size:12px;">side-by-side Qwen3-4B · same prompt · same weights</span>',
        '</div>',
        f'<div style="display:flex;align-items:center;gap:12px;">{sub_chips}{headline_html}</div>',
        '</div>',
        '<div class="rc-grid">',
        ref_panel,
        trt_panel,
        '</div>',
        '</div>',
        replay_js,
    ])
