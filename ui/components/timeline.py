"""Timeline component for Emile UI.

Renders a horizontal stepper visualizing each agent iteration as a node
(Baseline, Iter 1, Iter 2, ..., Done). Each node shows a kitchen-themed
SVG icon and status (pending / running / success / failed / done).
"""

from __future__ import annotations

import html
from typing import Any

# --- Ratatouille / kitchen SVG icons (20x20, currentColor) ---------------

ICON_SALT = (
    '<svg viewBox="0 0 20 20" width="20" height="20" fill="none" '
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<path d="M6 6h8l-1 10H7z"/>'
    '<path d="M7 6V4h6v2"/>'
    '<circle cx="9" cy="5" r="0.4" fill="currentColor"/>'
    '<circle cx="11" cy="5" r="0.4" fill="currentColor"/>'
    '<circle cx="10" cy="4.2" r="0.4" fill="currentColor"/>'
    '</svg>'
)

ICON_CHEF_HAT = (
    '<svg viewBox="0 0 20 20" width="20" height="20" fill="none" '
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<path d="M5 12c-1.5 0-2.5-1-2.5-2.5C2.5 8 4 7 5 7.2 5.2 5.3 7 4 9 4.2 '
    '10 3 11.5 3 13 4c1.5-.3 3 .8 3 2.8 0 1.8-1.3 2.5-2.5 2.5"/>'
    '<path d="M5 12h10v4H5z"/>'
    '</svg>'
)

ICON_NOSE = (
    '<svg viewBox="0 0 20 20" width="20" height="20" fill="none" '
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<path d="M10 3c-.5 3-2 5-3 7 0 2 1.5 3 3 3s3-1 3-3c-1-2-2.5-4-3-7z"/>'
    '<path d="M8.5 12c.5.5 2 .5 3 0"/>'
    '</svg>'
)

ICON_WHISK = (
    '<svg viewBox="0 0 20 20" width="20" height="20" fill="none" '
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<path d="M10 3v8"/>'
    '<path d="M7 11c.5 2 2 3 3 3s2.5-1 3-3"/>'
    '<path d="M8 4c-.3 3 .3 5 2 7"/>'
    '<path d="M12 4c.3 3-.3 5-2 7"/>'
    '<path d="M10 14v3"/>'
    '</svg>'
)

ICON_SPOON = (
    '<svg viewBox="0 0 20 20" width="20" height="20" fill="none" '
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<ellipse cx="10" cy="6" rx="3" ry="4"/>'
    '<path d="M10 10v7"/>'
    '</svg>'
)

ICON_FIRE_EXT = (
    '<svg viewBox="0 0 20 20" width="20" height="20" fill="none" '
    'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<path d="M6 4l2 2"/>'
    '<path d="M6 8h5v9H6z"/>'
    '<path d="M7 8V6h3v2"/>'
    '<path d="M14 5l-3 2"/>'
    '</svg>'
)

ICON_SPARKLE = (
    '<svg viewBox="0 0 20 20" width="20" height="20" fill="currentColor">'
    '<path d="M10 2l1.4 4.6L16 8l-4.6 1.4L10 14l-1.4-4.6L4 8l4.6-1.4z"/>'
    '<circle cx="15" cy="15" r="1"/>'
    '<circle cx="4" cy="15" r="0.8"/>'
    '</svg>'
)


# --- CSS ------------------------------------------------------------------

_CSS = """
<style>
.tl-wrap{
  width:100%;
  overflow-x:auto;
  padding:18px 8px 22px;
  font-family:"Inter",system-ui,sans-serif;
  color:var(--parchment,#F4ECD8);
}
.tl-track{
  display:flex;
  align-items:flex-start;
  gap:0;
  min-width:max-content;
  padding:0 12px;
}
.tl-node{
  display:flex;
  flex-direction:column;
  align-items:center;
  min-width:110px;
  position:relative;
  animation: tl-fade .35s ease both;
}
.tl-badge{
  width:44px;height:44px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  background:var(--steel,#2A2420);
  border:3px solid var(--slate,#6B6258);
  color:var(--parchment-dim,#C9BFA8);
  position:relative;
  transition: all .3s ease;
  z-index:2;
}
.tl-node.pending .tl-badge{
  border-color:var(--slate,#6B6258);
  color:var(--slate,#6B6258);
  border-style:dashed;
}
.tl-node.running .tl-badge{
  border-color:var(--copper,#C97B4A);
  color:var(--copper-glow,#E09968);
  box-shadow:0 0 14px rgba(201,123,74,.55);
}
.tl-node.running .tl-badge::before{
  content:"";position:absolute;inset:-6px;border-radius:50%;
  border:2px solid var(--copper,#C97B4A);
  animation: tl-pulse 1.2s ease-in-out infinite;
  pointer-events:none;
}
.tl-node.success .tl-badge{
  border-color:var(--basil,#5E8A5A);
  color:var(--basil,#5E8A5A);
}
.tl-node.failed .tl-badge{
  border-color:var(--bordeaux,#7A1F2B);
  color:var(--bordeaux,#7A1F2B);
}
.tl-node.done .tl-badge{
  border-color:var(--saffron,#E8B14A);
  color:var(--saffron,#E8B14A);
  box-shadow:0 0 16px rgba(232,177,74,.45);
}
.tl-node.ready .tl-badge{
  border-color:var(--copper,#C97B4A);
  border-style:dashed;
  color:var(--copper,#C97B4A);
}
.tl-connector{
  flex:0 0 60px;
  height:3px;
  margin-top:22px;
  background:var(--steel-2,#3A2E28);
  position:relative;
  z-index:1;
}
.tl-connector.filled{
  background:linear-gradient(90deg,var(--copper,#C97B4A),var(--copper-glow,#E09968));
}
.tl-label{
  margin-top:10px;
  font-family:"Fraunces",serif;
  font-size:12px;
  letter-spacing:.14em;
  text-transform:uppercase;
  color:var(--parchment,#F4ECD8);
  font-weight:600;
}
.tl-metrics{
  margin-top:4px;
  font-family:"JetBrains Mono",monospace;
  font-size:11px;
  color:var(--parchment-dim,#C9BFA8);
  white-space:nowrap;
}
.tl-delta{
  margin-top:6px;
  display:inline-block;
  font-family:"JetBrains Mono",monospace;
  font-size:10.5px;
  padding:2px 7px;
  border-radius:10px;
  border:1px solid var(--steel-2,#3A2E28);
}
.tl-delta.up{ color:var(--saffron,#E8B14A); border-color:rgba(232,177,74,.45);}
.tl-delta.flat{ color:var(--slate,#6B6258);}
.tl-delta.down{ color:var(--bordeaux,#7A1F2B); border-color:rgba(122,31,43,.55);}
.tl-speedup{
  margin-top:6px;
  display:inline-block;
  font-family:"JetBrains Mono",monospace;
  font-size:11px;
  padding:3px 9px;
  border-radius:10px;
  background:rgba(232,177,74,.15);
  color:var(--saffron,#E8B14A);
  border:1px solid rgba(232,177,74,.5);
}
@keyframes tl-pulse{
  0%  { transform:scale(.92); opacity:.9;}
  70% { transform:scale(1.15); opacity:0;}
  100%{ transform:scale(1.15); opacity:0;}
}
@keyframes tl-fade{
  from{opacity:0; transform:translateY(6px);}
  to  {opacity:1; transform:translateY(0);}
}
</style>
"""


# --- Helpers --------------------------------------------------------------

def _iter_label(i: int) -> str:
    return "Baseline" if i == 0 else f"Iter {i}"


def _icon_for_iter(i: int, status: str) -> str:
    if status == "failed":
        return ICON_FIRE_EXT
    if i == 0:
        return ICON_SALT
    # Alternate kitchen icons so the timeline feels lively.
    cycle = [ICON_CHEF_HAT, ICON_WHISK, ICON_NOSE, ICON_SPOON]
    return cycle[(i - 1) % len(cycle)]


def _build_iterations(events: list[dict]) -> tuple[dict[int, dict], bool, dict | None]:
    """Group events into per-iteration buckets.

    Returns (iters_by_index, has_done, done_event).
    """
    iters: dict[int, dict] = {}
    has_done = False
    done_evt: dict | None = None
    current_iter = 0  # events before any explicit iter belong to baseline/0
    max_iter_seen = 0

    for ev in events:
        et = ev.get("type")
        if et == "metrics":
            i = int(ev.get("iteration", current_iter))
            current_iter = i
            max_iter_seen = max(max_iter_seen, i)
            bucket = iters.setdefault(i, {"metrics": None, "error": False, "events": []})
            bucket["metrics"] = ev.get("data") or {}
            bucket["events"].append(ev)
        elif et == "kernel":
            i = int(ev.get("iteration", current_iter))
            max_iter_seen = max(max_iter_seen, i)
            bucket = iters.setdefault(i, {"metrics": None, "error": False, "events": []})
            bucket["events"].append(ev)
        elif et == "error":
            bucket = iters.setdefault(current_iter, {"metrics": None, "error": False, "events": []})
            bucket["error"] = True
            bucket["events"].append(ev)
        elif et == "done":
            has_done = True
            done_evt = ev
        else:
            bucket = iters.setdefault(current_iter, {"metrics": None, "error": False, "events": []})
            bucket["events"].append(ev)

    # Ensure we always have a baseline bucket if any events exist.
    if events and 0 not in iters:
        iters[0] = {"metrics": None, "error": False, "events": []}

    return iters, has_done, done_evt


def _node_status(
    i: int,
    bucket: dict | None,
    next_bucket_has_metrics: bool,
    running: bool,
    is_latest: bool,
) -> str:
    if bucket is None:
        return "pending"
    if bucket.get("error"):
        return "failed"
    if bucket.get("metrics") is not None:
        return "success"
    # No metrics yet for this iteration.
    if running and is_latest:
        return "running"
    return "pending"


def _fmt_metrics_line(m: dict | None) -> str:
    if not m:
        return "&middot; &middot;"
    tf = m.get("tflops")
    eff = m.get("efficiency_pct")
    tf_s = f"{tf:.2f} TF" if isinstance(tf, (int, float)) else "— TF"
    eff_s = f"{eff:.1f}%" if isinstance(eff, (int, float)) else "—%"
    return f"{tf_s} &middot; {eff_s}"


def _delta_badge(curr: dict | None, prev: dict | None) -> str:
    if not curr or not prev:
        return ""
    a = curr.get("tflops")
    b = prev.get("tflops")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return ""
    d = a - b
    if abs(d) < 0.01:
        return '<span class="tl-delta flat">&#9644;</span>'
    if d > 0:
        return f'<span class="tl-delta up">+{d:.2f}</span>'
    return f'<span class="tl-delta down">{d:.2f}</span>'


# --- Public API -----------------------------------------------------------

def render(state: dict) -> str:
    """Return HTML fragment for the agent stage timeline."""
    events: list[dict] = state.get("events") or []
    running: bool = bool(state.get("running"))

    # Empty / ready state.
    if not events:
        frag = (
            f'{_CSS}'
            '<div class="tl-wrap"><div class="tl-track">'
            '<div class="tl-node ready">'
            f'<div class="tl-badge">{ICON_SALT}</div>'
            '<div class="tl-label">Ready to cook</div>'
            '<div class="tl-metrics">awaiting first taste</div>'
            '</div></div></div>'
        )
        return frag

    iters, has_done, done_evt = _build_iterations(events)
    indices = sorted(iters.keys())
    latest_idx = max(indices) if indices else 0

    nodes_html: list[str] = []
    prev_metrics: dict | None = None

    for pos, i in enumerate(indices):
        bucket = iters.get(i)
        is_latest = (i == latest_idx) and not has_done
        status = _node_status(i, bucket, False, running, is_latest)

        icon = _icon_for_iter(i, status)
        label = _iter_label(i)
        metrics = bucket.get("metrics") if bucket else None
        metrics_line = _fmt_metrics_line(metrics)
        delta = _delta_badge(metrics, prev_metrics) if i > 0 else ""

        node = (
            f'<div class="tl-node {status}">'
            f'<div class="tl-badge">{icon}</div>'
            f'<div class="tl-label">{html.escape(label)}</div>'
            f'<div class="tl-metrics">{metrics_line}</div>'
            f'{delta}'
            f'</div>'
        )

        # Connector before this node (except first).
        if pos > 0:
            prev_i = indices[pos - 1]
            prev_bucket = iters.get(prev_i)
            filled = bool(prev_bucket and prev_bucket.get("metrics") is not None
                          and not prev_bucket.get("error"))
            cls = "tl-connector filled" if filled else "tl-connector"
            nodes_html.append(f'<div class="{cls}"></div>')

        nodes_html.append(node)
        if metrics is not None:
            prev_metrics = metrics

    # Terminal done node.
    if has_done:
        speedup = 0.0
        iters_count = 0
        if done_evt:
            speedup = float(done_evt.get("speedup") or 0.0)
            iters_count = int(done_evt.get("iterations") or 0)
        nodes_html.append('<div class="tl-connector filled"></div>')
        nodes_html.append(
            f'<div class="tl-node done">'
            f'<div class="tl-badge">{ICON_SPARKLE}</div>'
            f'<div class="tl-label">Done</div>'
            f'<div class="tl-metrics">{iters_count} iters</div>'
            f'<span class="tl-speedup">{speedup:.2f}&times; speedup</span>'
            f'</div>'
        )

    return (
        f'{_CSS}'
        f'<div class="tl-wrap"><div class="tl-track">{"".join(nodes_html)}</div></div>'
    )


def status_counts(state: dict) -> dict[str, Any]:
    """Return counts of node states plus whether a terminal done exists."""
    events: list[dict] = state.get("events") or []
    running: bool = bool(state.get("running"))
    counts = {"pending": 0, "running": 0, "success": 0, "failed": 0, "done": False}

    if not events:
        counts["pending"] = 1
        return counts

    iters, has_done, _ = _build_iterations(events)
    counts["done"] = has_done
    indices = sorted(iters.keys())
    latest = max(indices) if indices else 0

    for i in indices:
        bucket = iters[i]
        is_latest = (i == latest) and not has_done
        s = _node_status(i, bucket, False, running, is_latest)
        if s in counts and isinstance(counts[s], int):
            counts[s] += 1
    return counts
