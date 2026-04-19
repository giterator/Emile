"""Emile shell: composes the full iframe HTML for a given app state.

The shell is the orchestrator — it owns page chrome (brand bar, layout grid,
reveal banner) and delegates timeline / stream / dashboard / race to sibling
component modules, which are imported lazily so missing modules during dev do
not break the UI.
"""

from __future__ import annotations

from html import escape
from typing import Any

from ui.theme import get_css, get_logo_svg, get_wordmark_html


# ---------------------------------------------------------------------------
# Lazy sibling imports (timeline / stream / dashboard / race)
# ---------------------------------------------------------------------------

def _safe_render(module_name: str, state: dict) -> str:
    """Import `ui.components.<module_name>` and call its `render(state)`.

    Returns a friendly placeholder on any ImportError / AttributeError / runtime
    error so the shell still renders during parallel development.
    """
    try:
        mod = __import__(f"ui.components.{module_name}", fromlist=["render"])
        render = getattr(mod, "render")
        out = render(state)
        return out if isinstance(out, str) else str(out)
    except Exception as exc:  # noqa: BLE001 - intentional broad catch during dev
        label = module_name.capitalize()
        return (
            f'<div class="em-card"><div class="em-placeholder">'
            f"{label} component loading… "
            f"<span class=\"em-mono\">({escape(type(exc).__name__)})</span>"
            f"</div></div>"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stage_label(event_type: str, iteration: int | None = None, success: bool | None = None) -> str:
    """Return a short human label for a stage / event."""
    et = (event_type or "").lower()
    if et in ("baseline", "baseline_metrics"):
        return "Baseline"
    if et in ("done", "finished", "complete"):
        return "Done" if success is not False else "Halted"
    if et == "error":
        return f"Iter {iteration} · error" if iteration is not None else "Error"
    if et in ("profile", "profiling"):
        return f"Iter {iteration} · profile" if iteration is not None else "Profile"
    if et in ("kernel", "kernel_gen", "generate"):
        return f"Iter {iteration} · kernel" if iteration is not None else "Kernel"
    if et in ("metrics", "measure"):
        return f"Iter {iteration} · measure" if iteration is not None else "Measure"
    if et in ("thought", "thinking"):
        return f"Iter {iteration} · think" if iteration is not None else "Think"
    if et == "tool_call":
        return f"Iter {iteration} · tool" if iteration is not None else "Tool"
    if iteration is not None:
        return f"Iter {iteration} · {et}"
    return et.replace("_", " ").title() or "Stage"


# ---------------------------------------------------------------------------
# Page fragments
# ---------------------------------------------------------------------------

def _brand_bar(page: str, state: dict) -> str:
    # The host page (ui/app.py) already renders the Emile wordmark + logo,
    # so the iframe only needs a small state-chip strip.
    tab = "Kitchen" if page == "agent" else "Tasting Room"
    running = bool(state.get("running"))
    done = bool(state.get("done"))
    if running:
        status_chip = '<span class="em-chip copper"><span class="em-pulse" style="width:8px;height:8px;background:var(--copper);display:inline-block;border-radius:50%;"></span> Running</span>'
    elif done:
        status_chip = '<span class="em-chip saffron">Done</span>'
    else:
        status_chip = '<span class="em-chip slate">Idle</span>'

    return f"""
<div class="em-statusbar" style="display:flex;align-items:center;justify-content:flex-end;gap:0.5rem;margin-bottom:0.75rem;">
  <span class="em-chip">{escape(tab)}</span>
  {status_chip}
</div>
""".strip()


def _reveal_banner(state: dict) -> str:
    best = state.get("best_metrics") or {}
    base = state.get("baseline_metrics") or {}
    speedup = None
    try:
        if best.get("latency_ms") and base.get("latency_ms"):
            speedup = float(base["latency_ms"]) / float(best["latency_ms"])
        elif best.get("tflops") and base.get("tflops"):
            speedup = float(best["tflops"]) / float(base["tflops"])
    except Exception:  # noqa: BLE001
        speedup = None

    tflops = best.get("tflops")
    latency = best.get("latency_ms")
    eff = best.get("efficiency_pct") or best.get("efficiency")

    def _fmt(v: Any, unit: str = "") -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v):.2f}{unit}"
        except Exception:  # noqa: BLE001
            return escape(str(v))

    speedup_html = (
        f'<div class="em-reveal-num em-saffron-glow">{speedup:.2f}×</div>'
        if speedup is not None
        else '<div class="em-reveal-num em-saffron-glow">Voilà</div>'
    )

    return f"""
<section class="em-reveal em-fade-in" aria-label="Result">
  <div class="em-reveal-label em-serif">Emile has tasted the kernel.</div>
  {speedup_html}
  <div class="em-reveal-label">speedup over baseline</div>
  <div style="display:flex;gap:0.6rem;justify-content:center;margin-top:1rem;flex-wrap:wrap;">
    <span class="em-chip saffron">{_fmt(tflops, ' TFLOP/s')}</span>
    <span class="em-chip copper">{_fmt(latency, ' ms')}</span>
    <span class="em-chip basil">{_fmt(eff, '%')} efficiency</span>
  </div>
</section>
""".strip()


# ---------------------------------------------------------------------------
# Page assemblers
# ---------------------------------------------------------------------------

def _agent_page(state: dict) -> str:
    timeline_html = _safe_render("timeline", state)
    stream_html = _safe_render("stream", state)
    dashboard_html = _safe_render("dashboard", state)
    reveal_html = _reveal_banner(state) if state.get("done") else ""

    return f"""
{_brand_bar('agent', state)}
<section style="margin-bottom:1rem;">{timeline_html}</section>
<section class="em-grid-2">
  <div>{stream_html}</div>
  <div>{dashboard_html}</div>
</section>
{reveal_html}
""".strip()


def _demo_page(state: dict) -> str:
    race_html = _safe_render("race", state)
    triton_text = escape(state.get("triton_text") or "")
    tps = state.get("triton_tps") or 0.0
    ttft = state.get("triton_ttft") or 0.0

    try:
        tps_fmt = f"{float(tps):.1f}"
    except Exception:  # noqa: BLE001
        tps_fmt = str(tps)
    try:
        ttft_fmt = f"{float(ttft)*1000:.0f} ms" if float(ttft) < 10 else f"{float(ttft):.0f} ms"
    except Exception:  # noqa: BLE001
        ttft_fmt = str(ttft)

    return f"""
{_brand_bar('demo', state)}
<section style="margin-bottom:1rem;">{race_html}</section>
<section class="em-card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;gap:0.5rem;flex-wrap:wrap;">
    <h3 style="margin:0;">Triton stream</h3>
    <div style="display:flex;gap:0.4rem;">
      <span class="em-chip saffron">{tps_fmt} tok/s</span>
      <span class="em-chip copper">TTFT {ttft_fmt}</span>
    </div>
  </div>
  <pre class="em-mono" style="margin:0;padding:1rem;background:var(--ink);border:1px solid var(--steel-2);border-radius:10px;max-height:360px;overflow:auto;white-space:pre-wrap;color:var(--parchment);">{triton_text}</pre>
</section>
""".strip()


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def render(state: dict) -> str:
    """Return the full standalone HTML document for the iframe."""
    page = (state or {}).get("page", "agent")
    body = _agent_page(state) if page != "demo" else _demo_page(state)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Emile</title>
  {get_css()}
</head>
<body>
  <div class="em-shell em-fade-in">
    {body}
  </div>
  <script>
    // Auto-post scrollHeight so the Streamlit iframe host can resize.
    (function() {{
      function post() {{
        try {{
          const h = document.documentElement.scrollHeight;
          window.parent.postMessage({{ type: 'emile:height', height: h }}, '*');
        }} catch (e) {{}}
      }}
      window.addEventListener('load', post);
      new ResizeObserver(post).observe(document.body);
    }})();
  </script>
</body>
</html>"""
