"""Dashboard component — live metrics visualizations for the agent page.

Four-panel dashboard plus a three-card summary strip. All charts are
hand-written inline SVG (no external JS libraries) so the fragment can
render inside a self-contained st.components.v1.html iframe.

Entry point: render(state: dict) -> str
"""

from __future__ import annotations

import html
import math
from typing import Iterable, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------------
# Constants (mirror theme.py)
# ----------------------------------------------------------------------------

A100_PEAK_TFLOPS = 312.0
A100_PEAK_BANDWIDTH = 2000.0  # GB/s
RIDGE_POINT = 156.0           # FLOP/byte

# Colors (kept in sync with theme.py). These are only used for SVG fills/strokes
# where we can't rely on CSS vars rendering inside <svg> attributes reliably.
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


def _fmt(x: Optional[float], digits: int = 1, dash: str = "—") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return dash
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return dash


def _scale_linear(val: float, src_min: float, src_max: float,
                  dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    t = (val - src_min) / (src_max - src_min)
    return dst_min + t * (dst_max - dst_min)


def _scale_log(val: float, src_min: float, src_max: float,
               dst_min: float, dst_max: float) -> float:
    # val/src_min/src_max all > 0
    lv = math.log10(max(val, 1e-12))
    lo = math.log10(max(src_min, 1e-12))
    hi = math.log10(max(src_max, 1e-12))
    if hi == lo:
        return (dst_min + dst_max) / 2.0
    t = (lv - lo) / (hi - lo)
    return dst_min + t * (dst_max - dst_min)


def _points_str(points: Sequence[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    """Human-friendly linear tick picker."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, n - 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    start = math.floor(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.001:
        if v >= lo - step * 0.001:
            ticks.append(round(v, 10))
        v += step
    return ticks


# ----------------------------------------------------------------------------
# State extraction
# ----------------------------------------------------------------------------

def _metrics_events(state: dict) -> List[dict]:
    out = []
    for ev in state.get("events") or []:
        if isinstance(ev, dict) and ev.get("type") == "metrics":
            out.append(ev)
    return out


def _latest_metrics(state: dict) -> Optional[dict]:
    evs = _metrics_events(state)
    if not evs:
        return None
    return evs[-1].get("data")


def _best_iter_index(events: Sequence[dict]) -> Optional[int]:
    """Index (in events list) of the highest TFLOPS metrics event."""
    best_i = None
    best_v = -math.inf
    for i, ev in enumerate(events):
        data = ev.get("data") or {}
        t = data.get("tflops")
        if t is None:
            continue
        if t > best_v:
            best_v = t
            best_i = i
    return best_i


# ============================================================================
# CSS
# ============================================================================

_CSS = """
<style>
.db-root { width:100%; color: var(--parchment, #F4ECD8); font-family: Inter, system-ui, sans-serif; }
.db-summary {
  display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; margin-bottom:14px;
}
.db-sumcard {
  background: var(--steel, #2A2420); border:1px solid var(--steel-2, #3A2E28);
  border-radius: 10px; padding: 14px 16px; position:relative; overflow:hidden;
}
.db-sumcard::before {
  content:""; position:absolute; inset:0; pointer-events:none;
  background: radial-gradient(120% 80% at 0% 0%, rgba(201,123,74,0.08), transparent 60%);
}
.db-sumlabel {
  font-family: Inter, sans-serif; font-size: 11px; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--parchment-dim, #C9BFA8);
}
.db-sumval {
  font-family: Fraunces, serif; font-weight: 600; font-size: 2rem;
  line-height: 1.1; margin-top: 6px;
}
.db-sumval.copper  { color: var(--copper, #C97B4A); }
.db-sumval.saffron { color: var(--saffron, #E8B14A); }
.db-sumval.parch   { color: var(--parchment, #F4ECD8); }
.db-sumsub {
  margin-top: 6px; font-family: Inter, sans-serif; font-size: 12px;
  color: var(--parchment-dim, #C9BFA8);
}
.db-boundchip {
  display:inline-block; padding: 4px 10px; border-radius: 999px;
  font-family: Inter, sans-serif; font-size: 11px; letter-spacing:.08em;
  text-transform: uppercase; font-weight:600;
  border: 1px solid transparent;
}
.db-boundchip.memory  { color: var(--copper, #C97B4A); border-color: rgba(201,123,74,.5); background: rgba(201,123,74,.08); }
.db-boundchip.compute { color: var(--basil, #5E8A5A);  border-color: rgba(94,138,90,.5); background: rgba(94,138,90,.08); }

.db-grid {
  display:grid; grid-template-columns: 1fr; gap:14px;
}
@media (min-width: 900px) {
  .db-grid { grid-template-columns: 1fr 1fr; }
}
.db-panel {
  background: var(--steel, #2A2420); border:1px solid var(--steel-2, #3A2E28);
  border-radius: 10px; padding: 14px 16px; position:relative;
}
.db-panel h4 {
  margin:0 0 10px 0; font-family: Fraunces, serif; font-weight: 500;
  font-size: 1.05rem; color: var(--parchment, #F4ECD8); letter-spacing:.01em;
}
.db-panel .db-sub {
  font-family: Inter, sans-serif; font-size: 11px; color: var(--parchment-dim, #C9BFA8);
  margin-bottom: 6px;
}
.db-empty {
  border: 1px dashed var(--steel-2, #3A2E28); border-radius: 8px;
  padding: 28px 12px; text-align:center;
  color: var(--parchment-dim, #C9BFA8); font-family: Inter, sans-serif;
  font-size: 12px; letter-spacing:.06em; text-transform: uppercase;
}
.db-chart { position: relative; width: 100%; }
.db-chart.tflops   { height: 320px; }
.db-chart.roofline { height: 360px; }
.db-chart.spark    { height: 160px; }
.db-svg { width:100%; height:100%; display:block; }

/* HTML labels overlayed on the stretched SVG so text never distorts. */
.db-lbl {
  position: absolute;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 11px;
  color: var(--parchment-dim, #C9BFA8);
  line-height: 1;
  pointer-events: none;
  white-space: nowrap;
}
.db-lbl.axis-y { text-align: right; transform: translate(-100%, -50%); padding-right: 6px; }
.db-lbl.axis-x { transform: translate(-50%, 0); padding-top: 4px; }
.db-lbl.axis-y2 { text-align: left; transform: translate(0, -50%); padding-left: 6px; }
.db-lbl.inline-r { text-align: right; transform: translate(-100%, -120%); padding-right: 6px; }
.db-lbl.inline-l { text-align: left;  transform: translate(0, -120%); padding-left: 6px; }
.db-lbl.font-sans { font-family: Inter, sans-serif; }
.db-lbl.saffron { color: var(--saffron, #E8B14A); }
.db-lbl.copper  { color: var(--copper,  #C97B4A); }
.db-lbl.basil   { color: var(--basil,   #5E8A5A); }
.db-lbl.bordeaux{ color: var(--bordeaux,#7A1F2B); }
.db-lbl.parch-dim { color: var(--parchment-dim,#C9BFA8); opacity: 0.85; }
.db-occ-wrap {
  display:flex; align-items:center; gap:14px;
}
.db-occ-svg { width: 180px; height: 120px; flex: 0 0 auto; }
.db-occ-center { display:flex; flex-direction: column; align-items:center; }
.db-occ-big {
  font-family: Fraunces, serif; font-size: 2.4rem; line-height:1; color: var(--parchment, #F4ECD8);
}
.db-occ-pct { font-family: Inter, sans-serif; font-size: 0.9rem; color: var(--parchment-dim, #C9BFA8); margin-left:2px;}
.db-occ-meta {
  flex:1; display:flex; flex-direction: column; gap:6px;
  font-family: Inter, sans-serif; font-size: 12px; color: var(--parchment-dim, #C9BFA8);
}
.db-occ-limiter { font-family: Inter, sans-serif; font-size: 13px; }
.db-occ-limiter.registers { color: var(--copper, #C97B4A); }
.db-occ-limiter.shared    { color: var(--saffron, #E8B14A); }
.db-occ-limiter.threads   { color: var(--basil, #5E8A5A); }
.db-occ-limiter.warps     { color: var(--basil, #5E8A5A); }
.db-occ-nums { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 12px; color: var(--parchment, #F4ECD8); }

.db-spark-legend {
  font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 12px;
  color: var(--parchment-dim, #C9BFA8); margin-top: 4px;
}
.db-spark-legend .r { color: var(--copper, #C97B4A); }
.db-spark-legend .s { color: var(--bordeaux, #7A1F2B); }

.em-fade-in { animation: emFadeIn 380ms ease-out both; }
@keyframes emFadeIn { from { opacity:0; transform: translateY(4px); } to { opacity:1; transform:none; } }
</style>
"""


# ============================================================================
# Summary strip
# ============================================================================

def _summary_strip(state: dict) -> str:
    latest = _latest_metrics(state)
    baseline = state.get("baseline_metrics")
    best = state.get("best_metrics")

    # Card 1 — Current TFLOPS
    if latest and latest.get("tflops") is not None:
        tf = float(latest["tflops"])
        pct = tf / A100_PEAK_TFLOPS * 100.0
        card1_val = f'<div class="db-sumval copper">{_fmt(tf,1)}<span style="font-family:Inter;font-size:0.9rem;color:var(--parchment-dim);margin-left:6px;">TFLOPS</span></div>'
        card1_sub = f'<div class="db-sumsub">{_fmt(pct,1)}% of A100 peak</div>'
    else:
        card1_val = '<div class="db-sumval copper">—</div>'
        card1_sub = '<div class="db-sumsub">awaiting first profile</div>'

    # Card 2 — Speedup
    if (best and best.get("tflops") is not None
            and baseline and baseline.get("tflops")):
        sp = float(best["tflops"]) / max(float(baseline["tflops"]), 1e-9)
        card2_val = f'<div class="db-sumval saffron">×{_fmt(sp,2)}</div>'
        card2_sub = '<div class="db-sumsub">vs PyTorch reference</div>'
    else:
        card2_val = '<div class="db-sumval saffron">—</div>'
        card2_sub = '<div class="db-sumsub">vs PyTorch reference</div>'

    # Card 3 — Bound
    if latest and latest.get("bound"):
        bound = str(latest["bound"]).lower()
        klass = "compute" if bound == "compute" else "memory"
        ai = latest.get("arithmetic_intensity")
        chip = f'<span class="db-boundchip {klass}">{_esc(bound)}</span>'
        sub = f'<div class="db-sumsub">arithmetic intensity: {_fmt(ai,2)} FLOP/byte</div>'
        card3_val = f'<div class="db-sumval parch" style="font-size:1.4rem;padding-top:4px;">{chip}</div>'
        card3_sub = sub
    else:
        card3_val = '<div class="db-sumval parch">—</div>'
        card3_sub = '<div class="db-sumsub">no profile yet</div>'

    return f"""
    <div class="db-summary em-fade-in">
      <div class="db-sumcard">
        <div class="db-sumlabel">Current TFLOPS</div>
        {card1_val}
        {card1_sub}
      </div>
      <div class="db-sumcard">
        <div class="db-sumlabel">Speedup</div>
        {card2_val}
        {card2_sub}
      </div>
      <div class="db-sumcard">
        <div class="db-sumlabel">Bound</div>
        {card3_val}
        {card3_sub}
      </div>
    </div>
    """


# ============================================================================
# Panel 1 — TFLOPS progression
# ============================================================================

def _panel_tflops(state: dict) -> str:
    hist = list(state.get("tflops_history") or [])
    cfg = state.get("config") or {}
    target_pct = float(cfg.get("target_pct", 70.0))
    target_tf = target_pct / 100.0 * A100_PEAK_TFLOPS

    if not hist:
        body = '<div class="db-empty">awaiting first profile</div>'
        return f"""
        <div class="db-panel em-fade-in">
          <h4>TFLOPS Progression</h4>
          {body}
        </div>
        """

    # Canvas
    W, H = 640, 300
    MX_L, MX_R, MY_T, MY_B = 44, 12, 14, 28
    x0, x1 = MX_L, W - MX_R
    y0, y1 = MY_T, H - MY_B

    iters = [int(h[0]) for h in hist]
    vals = [float(h[1]) for h in hist]
    xi_min = min(iters)
    xi_max = max(iters)
    if xi_max == xi_min:
        xi_max = xi_min + 1

    # Y range: include 0, peak target, a bit above max
    y_hi = max(max(vals), target_tf, A100_PEAK_TFLOPS * 0.35) * 1.1
    y_lo = 0.0
    yticks = _nice_ticks(y_lo, y_hi, 5)
    y_hi = max(y_hi, yticks[-1])

    def sx(i):
        return _scale_linear(i, xi_min, xi_max, x0, x1)

    def sy(v):
        return _scale_linear(v, y_lo, y_hi, y1, y0)

    # Baseline band (0..baseline tflops) in bordeaux 20%
    baseline = state.get("baseline_metrics")
    band_svg = ""
    if baseline and baseline.get("tflops") is not None:
        b = float(baseline["tflops"])
        if b > 0:
            by = sy(b)
            band_svg = (
                f'<rect x="{x0}" y="{by:.2f}" width="{x1-x0}" '
                f'height="{y1-by:.2f}" fill="{C_BORDEAUX}" opacity="0.20" />'
            )

    # Axes / grid (lines only — labels become HTML overlays)
    grid = []
    html_labels: List[str] = []
    def _pct(coord: float, dim: float) -> str:
        return f"{coord / dim * 100:.3f}%"

    for yt in yticks:
        gy = sy(yt)
        grid.append(f'<line x1="{x0}" y1="{gy:.2f}" x2="{x1}" y2="{gy:.2f}" '
                    f'stroke="{C_STEEL_2}" stroke-width="1" />')
        html_labels.append(
            f'<div class="db-lbl axis-y" style="left:{_pct(x0, W)};'
            f'top:{_pct(gy, H)};">{_fmt(yt,0)}</div>'
        )

    # X ticks — one per iteration (cap to reasonable number)
    unique_iters = sorted(set(iters))
    if len(unique_iters) > 12:
        step = max(1, len(unique_iters) // 10)
        unique_iters = unique_iters[::step] + [unique_iters[-1]]
    xticks_svg = []
    for xi in unique_iters:
        gx = sx(xi)
        xticks_svg.append(
            f'<line x1="{gx:.2f}" y1="{y1}" x2="{gx:.2f}" y2="{y1+4}" '
            f'stroke="{C_SLATE}" stroke-width="1" />'
        )
        html_labels.append(
            f'<div class="db-lbl axis-x" style="left:{_pct(gx, W)};'
            f'top:{_pct(y1+6, H)};">{xi}</div>'
        )

    # Target line (saffron dashed) + HTML label
    target_line = ""
    if target_tf >= y_lo and target_tf <= y_hi:
        ty = sy(target_tf)
        target_line = (
            f'<line x1="{x0}" y1="{ty:.2f}" x2="{x1}" y2="{ty:.2f}" '
            f'stroke="{C_SAFFRON}" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.85" />'
        )
        html_labels.append(
            f'<div class="db-lbl inline-r font-sans saffron" '
            f'style="left:{_pct(x1, W)};top:{_pct(ty, H)};">'
            f'target {_fmt(target_pct,0)}% · {_fmt(target_tf,0)} TFLOPS</div>'
        )

    # Peak line (parchment dim dashed) + HTML label
    peak_line = ""
    if A100_PEAK_TFLOPS <= y_hi:
        py = sy(A100_PEAK_TFLOPS)
        peak_line = (
            f'<line x1="{x0}" y1="{py:.2f}" x2="{x1}" y2="{py:.2f}" '
            f'stroke="{C_PARCHMENT_DIM}" stroke-width="1" stroke-dasharray="2 4" opacity="0.4" />'
        )
        html_labels.append(
            f'<div class="db-lbl inline-r font-sans parch-dim" '
            f'style="left:{_pct(x1, W)};top:{_pct(py, H)};">A100 peak · 312</div>'
        )

    # Build line / area
    pts: List[Tuple[float, float]] = [(sx(i), sy(v)) for i, v in zip(iters, vals)]
    area_pts = [(pts[0][0], y1)] + pts + [(pts[-1][0], y1)]
    area = (f'<polygon points="{_points_str(area_pts)}" fill="{C_COPPER}" '
            f'opacity="0.25" />')
    line = (f'<polyline points="{_points_str(pts)}" fill="none" '
            f'stroke="{C_COPPER}" stroke-width="2" stroke-linejoin="round" '
            f'stroke-linecap="round" />')

    # Best point index
    best_i = max(range(len(vals)), key=lambda i: vals[i])

    dots = []
    for idx, ((px, py), it, v) in enumerate(zip(pts, iters, vals)):
        is_best = (idx == best_i)
        fill = C_SAFFRON if is_best else C_COPPER
        r = 4.5 if is_best else 3.2
        dots.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{r}" fill="{fill}" '
            f'stroke="{C_ESPRESSO}" stroke-width="1">'
            f'<title>iter {it}: {_fmt(v,2)} TFLOPS</title></circle>'
        )

    # Axis frame (subtle)
    frame = (f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" '
             f'stroke="{C_STEEL_2}" stroke-width="1.2"/>'
             f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" '
             f'stroke="{C_STEEL_2}" stroke-width="1.2"/>')

    svg = f"""
    <div class="db-chart tflops">
      <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" class="db-svg"
           role="img" aria-label="TFLOPS progression chart">
        {band_svg}
        {''.join(grid)}
        {frame}
        {area}
        {line}
        {''.join(dots)}
        {target_line}
        {peak_line}
        {''.join(xticks_svg)}
      </svg>
      {''.join(html_labels)}
    </div>
    """

    return f"""
    <div class="db-panel em-fade-in">
      <h4>TFLOPS Progression</h4>
      <div class="db-sub">iteration vs. measured throughput (A100)</div>
      {svg}
    </div>
    """


# ============================================================================
# Panel 2 — Roofline plot
# ============================================================================

def _panel_roofline(state: dict) -> str:
    events = _metrics_events(state)

    W, H = 640, 360
    MX_L, MX_R, MY_T, MY_B = 54, 14, 16, 36
    x0, x1 = MX_L, W - MX_R
    y0, y1 = MY_T, H - MY_B

    X_MIN, X_MAX = 0.1, 10_000.0
    Y_MIN, Y_MAX = 0.1, 400.0

    def sx(ai):
        return _scale_log(ai, X_MIN, X_MAX, x0, x1)

    def sy(tf):
        return _scale_log(tf, Y_MIN, Y_MAX, y1, y0)

    # Background region shading
    ridge_x = sx(RIDGE_POINT)
    mem_shade = (
        f'<rect x="{x0}" y="{y0}" width="{ridge_x-x0:.2f}" height="{y1-y0:.2f}" '
        f'fill="{C_COPPER}" opacity="0.08" />'
    )
    cmp_shade = (
        f'<rect x="{ridge_x:.2f}" y="{y0}" width="{x1-ridge_x:.2f}" height="{y1-y0:.2f}" '
        f'fill="{C_BASIL}" opacity="0.08" />'
    )

    # Grid (powers of 10) — lines only, labels are HTML overlays
    grid_svg = []
    html_labels: List[str] = []
    def _pct(coord: float, dim: float) -> str:
        return f"{coord / dim * 100:.3f}%"

    for p in range(-1, 5):
        v = 10 ** p
        if v < X_MIN or v > X_MAX:
            continue
        gx = sx(v)
        grid_svg.append(
            f'<line x1="{gx:.2f}" y1="{y0}" x2="{gx:.2f}" y2="{y1}" '
            f'stroke="{C_STEEL_2}" stroke-width="1" opacity="0.7"/>'
        )
        if p < 0 or p > 3:
            label = f"10<sup>{p}</sup>"
        else:
            label = f"{int(v)}"
        html_labels.append(
            f'<div class="db-lbl axis-x" style="left:{_pct(gx, W)};'
            f'top:{_pct(y1+6, H)};">{label}</div>'
        )
    for p in range(-1, 3):
        v = 10 ** p
        if v < Y_MIN or v > Y_MAX:
            continue
        gy = sy(v)
        grid_svg.append(
            f'<line x1="{x0}" y1="{gy:.2f}" x2="{x1}" y2="{gy:.2f}" '
            f'stroke="{C_STEEL_2}" stroke-width="1" opacity="0.7"/>'
        )
        if p < 0 or p > 2:
            label = f"10<sup>{p}</sup>"
        else:
            label = f"{int(v)}"
        html_labels.append(
            f'<div class="db-lbl axis-y" style="left:{_pct(x0, W)};'
            f'top:{_pct(gy, H)};">{label}</div>'
        )

    # Roofline — sample many x in log space, y = min(312, 2x)
    samples = []
    N = 120
    for k in range(N + 1):
        t = k / N
        lx = math.log10(X_MIN) + t * (math.log10(X_MAX) - math.log10(X_MIN))
        xv = 10 ** lx
        yv = min(A100_PEAK_TFLOPS, 2.0 * xv)
        yv = max(yv, Y_MIN)
        samples.append((sx(xv), sy(yv)))
    roof_line = (f'<polyline points="{_points_str(samples)}" fill="none" '
                 f'stroke="{C_PARCHMENT}" stroke-width="1.8" opacity="0.9" />')

    # Ridge vertical dashed + HTML label
    ridge_top = sy(A100_PEAK_TFLOPS)
    ridge_svg = (
        f'<line x1="{ridge_x:.2f}" y1="{y0}" x2="{ridge_x:.2f}" y2="{y1}" '
        f'stroke="{C_PARCHMENT_DIM}" stroke-width="1" stroke-dasharray="3 4" opacity="0.6"/>'
    )
    html_labels.append(
        f'<div class="db-lbl font-sans parch-dim" style="left:{_pct(ridge_x+6, W)};'
        f'top:{_pct(ridge_top+10, H)};transform:none;">ridge point (156 FLOP/byte)</div>'
    )
    # Axis titles (absolute-positioned inside chart wrap)
    html_labels.append(
        f'<div class="db-lbl font-sans parch-dim" style="left:50%;bottom:2px;top:auto;'
        f'transform:translateX(-50%);font-size:12px;">Arithmetic Intensity (FLOP/byte)</div>'
    )
    html_labels.append(
        f'<div class="db-lbl font-sans parch-dim" style="left:0;top:50%;'
        f'transform:rotate(-90deg) translateX(50%);transform-origin:left center;'
        f'font-size:12px;">TFLOPS</div>'
    )
    axis_labels = ""  # moved to HTML overlay

    # Points
    if not events:
        body_svg = f"""
        <div class="db-chart roofline">
          <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" class="db-svg"
               role="img" aria-label="Empty roofline chart">
            {mem_shade}{cmp_shade}
            {''.join(grid_svg)}
            {roof_line}
            {ridge_svg}
          </svg>
          {''.join(html_labels)}
        </div>
        <div class="db-empty" style="margin-top:8px;">awaiting first profile</div>
        """
        return f"""
        <div class="db-panel em-fade-in">
          <h4>Roofline — A100</h4>
          <div class="db-sub">memory-bound · ridge 156 FLOP/byte · compute-bound</div>
          {body_svg}
        </div>
        """

    # Best-iteration index among events
    best_i = _best_iter_index(events)

    # Build scatter + trajectory
    scatter_pts: List[Tuple[float, float]] = []
    scatter_svg: List[str] = []
    n = len(events)
    for i, ev in enumerate(events):
        d = ev.get("data") or {}
        ai = d.get("arithmetic_intensity")
        tf = d.get("tflops")
        if ai is None or tf is None or ai <= 0 or tf <= 0:
            continue
        ai_c = min(max(float(ai), X_MIN), X_MAX)
        tf_c = min(max(float(tf), Y_MIN), Y_MAX)
        cx = sx(ai_c)
        cy = sy(tf_c)
        scatter_pts.append((cx, cy))

        # Color gradient copper -> saffron
        t = i / max(1, n - 1)
        color = _mix_hex(C_COPPER, C_SAFFRON, t)
        is_best = (i == best_i)
        title = (f'iter {ev.get("iteration","?")}: '
                 f'{_fmt(tf,1)} TFLOPS @ {_fmt(ai,2)} FLOP/byte')
        if is_best:
            # star
            scatter_svg.append(_svg_star(cx, cy, 7.5, C_SAFFRON, title))
        else:
            scatter_svg.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4" fill="{color}" '
                f'stroke="{C_ESPRESSO}" stroke-width="1"><title>{_esc(title)}</title></circle>'
            )

    traj = ""
    if len(scatter_pts) >= 2:
        traj = (f'<polyline points="{_points_str(scatter_pts)}" fill="none" '
                f'stroke="{C_SLATE}" stroke-width="1.2" opacity="0.8" '
                f'stroke-dasharray="2 3" />')

    svg = f"""
    <div class="db-chart roofline">
      <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" class="db-svg"
           role="img" aria-label="Roofline chart">
        {mem_shade}{cmp_shade}
        {''.join(grid_svg)}
        {roof_line}
        {ridge_svg}
        {traj}
        {''.join(scatter_svg)}
      </svg>
      {''.join(html_labels)}
    </div>
    """

    return f"""
    <div class="db-panel em-fade-in">
      <h4>Roofline — A100</h4>
      <div class="db-sub">memory-bound · ridge 156 FLOP/byte · compute-bound</div>
      {svg}
    </div>
    """


def _mix_hex(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    r = round(ar + (br - ar) * t)
    g = round(ag + (bg - ag) * t)
    bv = round(ab + (bb - ab) * t)
    return f"#{r:02X}{g:02X}{bv:02X}"


def _svg_star(cx: float, cy: float, r: float, fill: str, title: str) -> str:
    # 5-point star
    pts = []
    for k in range(10):
        ang = -math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.45
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    return (f'<polygon points="{_points_str(pts)}" fill="{fill}" '
            f'stroke="{C_ESPRESSO}" stroke-width="1">'
            f'<title>{_esc(title)}</title></polygon>')


# ============================================================================
# Panel 3 — Occupancy meter
# ============================================================================

def _panel_occupancy(state: dict) -> str:
    latest = _latest_metrics(state)
    occ = (latest or {}).get("occupancy") if latest else None

    if not occ:
        body = '<div class="db-empty">awaiting first profile</div>'
        return f"""
        <div class="db-panel em-fade-in">
          <h4>Occupancy</h4>
          {body}
        </div>
        """

    pct = float(occ.get("occupancy_pct") or 0.0)
    pct = max(0.0, min(100.0, pct))
    limiter = str(occ.get("limiter") or "").lower() or "—"
    blocks = occ.get("blocks_per_sm")
    warps = occ.get("warps_per_sm")

    # Gauge — semicircle from 180° (left) to 360°/0° (right) going clockwise.
    W, H = 220, 130
    cx, cy = W / 2, H - 20
    rad = 82
    stroke_w = 14

    def pt(angle_deg):
        a = math.radians(angle_deg)
        return (cx + rad * math.cos(a), cy + rad * math.sin(a))

    # Arc from 180° (left) clockwise to 180 + 180*pct/100
    # But SVG y grows downward; using standard math.cos/sin, 180° → (-r,0)
    # and 0° → (r,0), 270° → (0,-r)... Actually math.sin positive goes
    # downward in SVG because y axis is flipped. We want the arc above cy.
    # For an "upper" half semicircle going clockwise from left (180°) to
    # right (0°), angles decrease from 180° to 0° in math convention
    # *but* since SVG flips Y, we go the other way. Easiest: use angle
    # going from 180° back to 360° via the TOP. In math convention on a
    # screen (Y-down), 180°→left, 270°→bottom, 360°→right. So the top
    # half is 180° → 90° (top) → 0°. So we parameterize with theta from
    # 180° to 360° if we want the BOTTOM half. For TOP half with Y-down
    # SVG: angle goes 180° → 270°? No — let's just manually build.
    #
    # We will go from "left" to "right" through the TOP. In SVG
    # coordinates: left = (cx-r, cy), top = (cx, cy-r), right = (cx+r, cy).
    # Parametrize with u in [0,1]: angle_svg = pi + u*pi (radians) BUT using
    # standard sin/cos (SVG Y-down) that sweeps the bottom half. So we
    # negate the y offset to flip to top half.
    def arc_pt(u):
        ang = math.pi + u * math.pi  # π -> 2π
        return (cx + rad * math.cos(ang), cy - rad * math.sin(ang) * -1)
        # Hmm, let's just directly compute: for u in [0,1],
        #   x = cx - rad*cos(u*pi)   (starts at -r, ends at +r)
        #   y = cy - rad*sin(u*pi)   (starts at 0, up to -r, back to 0) → top half

    def arc_pt2(u):
        x = cx - rad * math.cos(u * math.pi)
        y = cy - rad * math.sin(u * math.pi)
        return (x, y)

    start = arc_pt2(0.0)
    full_end = arc_pt2(1.0)

    # Background arc (full)
    bg_arc = (f'<path d="M {start[0]:.2f} {start[1]:.2f} '
              f'A {rad} {rad} 0 0 1 {full_end[0]:.2f} {full_end[1]:.2f}" '
              f'fill="none" stroke="{C_STEEL_2}" stroke-width="{stroke_w}" '
              f'stroke-linecap="round"/>')

    # Foreground arc (pct filled)
    u = pct / 100.0
    end = arc_pt2(u)
    # Color by pct: bordeaux -> copper -> basil
    if pct < 40:
        color = _mix_hex(C_BORDEAUX, C_COPPER, pct / 40.0)
    elif pct < 80:
        color = _mix_hex(C_COPPER, C_BASIL, (pct - 40) / 40.0)
    else:
        color = C_BASIL

    fg_arc = ""
    if u > 0:
        # Large arc flag: 0 since <=180°. Sweep 1 (clockwise via our
        # parameterization through the top).
        fg_arc = (f'<path d="M {start[0]:.2f} {start[1]:.2f} '
                  f'A {rad} {rad} 0 0 1 {end[0]:.2f} {end[1]:.2f}" '
                  f'fill="none" stroke="{color}" stroke-width="{stroke_w}" '
                  f'stroke-linecap="round"/>')

    gauge_svg = f"""
    <svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" class="db-occ-svg"
         role="img" aria-label="Occupancy gauge">
      {bg_arc}
      {fg_arc}
      <text x="{cx}" y="{cy-8}" text-anchor="middle" fill="{C_PARCHMENT}"
            font-family="Fraunces,serif" font-size="28" font-weight="600">
        {_fmt(pct,0)}<tspan font-size="14" fill="{C_PARCHMENT_DIM}" dx="2">%</tspan>
      </text>
    </svg>
    """

    lim_class = {
        "registers": "registers",
        "shared": "shared", "shared_memory": "shared",
        "threads": "threads",
        "warps": "warps",
    }.get(limiter, "")

    meta = f"""
      <div class="db-occ-meta">
        <div class="db-occ-limiter {lim_class}">limiter · <span style="font-weight:600">{_esc(limiter)}</span></div>
        <div class="db-occ-nums">blocks/SM: {_esc(blocks if blocks is not None else '—')} · warps/SM: {_esc(warps if warps is not None else '—')}</div>
      </div>
    """

    return f"""
    <div class="db-panel em-fade-in">
      <h4>Occupancy</h4>
      <div class="db-sub">latest kernel profile</div>
      <div class="db-occ-wrap">
        {gauge_svg}
        {meta}
      </div>
    </div>
    """


# ============================================================================
# Panel 4 — Registers & Spills trend
# ============================================================================

def _panel_registers(state: dict) -> str:
    events = _metrics_events(state)
    # Collect (iter, registers, spills) from kernel_metadata[0]
    series: List[Tuple[int, Optional[int], Optional[int]]] = []
    for ev in events:
        it = ev.get("iteration")
        d = ev.get("data") or {}
        km = d.get("kernel_metadata") or []
        if not km or not isinstance(km, list):
            continue
        # pick the hottest kernel variant (highest reg count) so we track the
        # real bottleneck, matching how _estimate_occupancy chooses its "hot" one.
        def _reg_of(k):
            if not isinstance(k, dict):
                return -1
            return (k.get("n_regs") or k.get("registers") or 0)
        hot = max((k for k in km if isinstance(k, dict)), key=_reg_of, default=None)
        if not hot:
            continue
        regs = hot.get("n_regs", hot.get("registers"))
        spills = hot.get("n_spills", hot.get("spills"))
        series.append((int(it) if it is not None else 0, regs, spills))

    if not series or all(r is None for _, r, _ in series):
        body = '<div class="db-empty">awaiting first profile</div>'
        return f"""
        <div class="db-panel em-fade-in">
          <h4>Registers &amp; Spills</h4>
          {body}
        </div>
        """

    W, H = 640, 160
    MX_L, MX_R, MY_T, MY_B = 34, 48, 10, 20
    x0, x1 = MX_L, W - MX_R
    y0, y1 = MY_T, H - MY_B

    iters = [s[0] for s in series]
    regs_vals = [s[1] for s in series if s[1] is not None]
    spills_all = [s[2] for s in series]
    has_spills = any((sp or 0) > 0 for sp in spills_all)

    xi_min = min(iters)
    xi_max = max(iters)
    if xi_max == xi_min:
        xi_max = xi_min + 1

    r_lo = 0
    r_hi = max(max(regs_vals), 32)
    r_hi = max(r_hi * 1.15, r_hi + 4)

    if has_spills:
        s_vals = [sp if sp is not None else 0 for sp in spills_all]
        s_lo = 0
        s_hi = max(max(s_vals), 1) * 1.2
    else:
        s_lo, s_hi = 0, 1

    def sx(i):
        return _scale_linear(i, xi_min, xi_max, x0, x1)

    def sy_r(v):
        return _scale_linear(v, r_lo, r_hi, y1, y0)

    def sy_s(v):
        return _scale_linear(v, s_lo, s_hi, y1, y0)

    # Axis frame
    frame = (f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" '
             f'stroke="{C_STEEL_2}" stroke-width="1"/>'
             f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" '
             f'stroke="{C_STEEL_2}" stroke-width="1"/>')
    if has_spills:
        frame += (f'<line x1="{x1}" y1="{y0}" x2="{x1}" y2="{y1}" '
                  f'stroke="{C_STEEL_2}" stroke-width="1"/>')

    # Registers line
    reg_pts = [(sx(it), sy_r(r)) for it, r, _ in series if r is not None]
    reg_line = ""
    if len(reg_pts) >= 2:
        reg_line = (f'<polyline points="{_points_str(reg_pts)}" fill="none" '
                    f'stroke="{C_COPPER}" stroke-width="2" stroke-linejoin="round"/>')
    reg_dots = "".join(
        f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.5" fill="{C_COPPER}"/>'
        for px, py in reg_pts
    )

    # Spills line
    spill_line = ""
    spill_dots = ""
    if has_spills:
        sp_pts = [(sx(it), sy_s(sp if sp is not None else 0))
                  for it, _, sp in series]
        if len(sp_pts) >= 2:
            spill_line = (f'<polyline points="{_points_str(sp_pts)}" fill="none" '
                          f'stroke="{C_BORDEAUX}" stroke-width="2" stroke-linejoin="round"/>')
        spill_dots = "".join(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.5" fill="{C_BORDEAUX}"/>'
            for px, py in sp_pts
        )

    # Y-axis labels — HTML overlays so they don't distort with stretched SVG
    def _pct(c: float, dim: float) -> str:
        return f"{c / dim * 100:.3f}%"
    html_labels: List[str] = [
        f'<div class="db-lbl axis-y copper" style="left:{_pct(x0, W)};top:{_pct(sy_r(r_hi), H)};">{_fmt(r_hi,0)}</div>',
        f'<div class="db-lbl axis-y copper" style="left:{_pct(x0, W)};top:{_pct(sy_r(r_lo), H)};">0</div>',
    ]
    if has_spills:
        html_labels += [
            f'<div class="db-lbl axis-y2 bordeaux" style="left:{_pct(x1, W)};top:{_pct(sy_s(s_hi), H)};">{_fmt(s_hi,0)}</div>',
            f'<div class="db-lbl axis-y2 bordeaux" style="left:{_pct(x1, W)};top:{_pct(sy_s(s_lo), H)};">0</div>',
        ]

    svg = f"""
    <div class="db-chart spark">
      <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" class="db-svg"
           role="img" aria-label="Registers and spills trend">
        {frame}
        {reg_line}{reg_dots}
        {spill_line}{spill_dots}
      </svg>
      {''.join(html_labels)}
    </div>
    """

    latest_reg = next((r for _, r, _ in reversed(series) if r is not None), None)
    latest_spill = next((sp for _, _, sp in reversed(series)), None)
    legend = (f'<div class="db-spark-legend">'
              f'<span class="r">R: {_esc(latest_reg if latest_reg is not None else "—")}</span>'
              f'  ·  '
              f'<span class="s">S: {_esc(latest_spill if latest_spill is not None else 0)}</span>'
              f'</div>')

    return f"""
    <div class="db-panel em-fade-in">
      <h4>Registers &amp; Spills</h4>
      <div class="db-sub">first kernel variant · per-thread</div>
      {svg}
      {legend}
    </div>
    """


# ============================================================================
# Public API
# ============================================================================

def render(state: dict) -> str:
    """Return HTML fragment with dashboard visualizations."""
    if not isinstance(state, dict):
        state = {}

    parts = [
        _CSS,
        '<div class="db-root">',
        _summary_strip(state),
        '<div class="db-grid">',
        _panel_tflops(state),
        _panel_roofline(state),
        _panel_occupancy(state),
        _panel_registers(state),
        '</div>',
        '</div>',
    ]
    return "\n".join(parts)
