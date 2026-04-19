"""Stream component for Emile UI.

Renders a chronological stream of the agent's thoughts, tool calls,
metrics, kernel changes, errors, and the final done banner. Each event
type has its own card style with Ratatouille-themed iconography.
"""

from __future__ import annotations

import html
import re
from typing import Any

# --- Inline SVG icons -----------------------------------------------------

ICON_WHISK = (
    '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" '
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<path d="M10 3v8"/>'
    '<path d="M7 11c.5 2 2 3 3 3s2.5-1 3-3"/>'
    '<path d="M8 4c-.3 3 .3 5 2 7"/>'
    '<path d="M12 4c.3 3-.3 5-2 7"/>'
    '<path d="M10 14v3"/>'
    '</svg>'
)

ICON_CROSS = (
    '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round">'
    '<path d="M5 5l10 10M15 5L5 15"/>'
    '</svg>'
)

ICON_NOSE = (
    '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" '
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<path d="M10 3c-.5 3-2 5-3 7 0 2 1.5 3 3 3s3-1 3-3c-1-2-2.5-4-3-7z"/>'
    '<path d="M8.5 12c.5.5 2 .5 3 0"/>'
    '</svg>'
)

ICON_SPARKLE = (
    '<svg viewBox="0 0 20 20" width="20" height="20" fill="currentColor">'
    '<path d="M10 2l1.4 4.6L16 8l-4.6 1.4L10 14l-1.4-4.6L4 8l4.6-1.4z"/>'
    '</svg>'
)


# --- Diagnosis tag system -------------------------------------------------

DIAGNOSIS_TAGS: dict[str, str] = {
    "roofline": "copper",
    "compile": "steel",
    "occupancy": "basil",
    "torch.profiler": "slate",
    "proton": "saffron",
    "critical": "bordeaux",
}

# Match [tag] where tag is one of the known diagnosis tags (case-insensitive).
_TAG_RE = re.compile(
    r"\[(" + "|".join(re.escape(t) for t in DIAGNOSIS_TAGS) + r")\]",
    re.IGNORECASE,
)


def _extract_diagnosis_chips(text: str) -> tuple[str, list[str]]:
    """Strip known [tag] markers from text and return them as a list."""
    if not text:
        return "", []
    found: list[str] = []

    def _collect(m: re.Match) -> str:
        found.append(m.group(1).lower())
        return ""

    stripped = _TAG_RE.sub(_collect, text)
    # Deduplicate preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for t in found:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    # Normalize whitespace left by removed tags.
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped, uniq


def _render_chips(tags: list[str]) -> str:
    if not tags:
        return ""
    parts = []
    for t in tags:
        color = DIAGNOSIS_TAGS.get(t, "slate")
        pulse = " st-chip-pulse" if t == "critical" else ""
        parts.append(
            f'<span class="st-chip st-chip-{color}{pulse}">{html.escape(t)}</span>'
        )
    return f'<div class="st-chips">{"".join(parts)}</div>'


# --- CSS ------------------------------------------------------------------

_CSS = """
<style>
.st-scroll{
  max-height: 70vh;
  overflow-y:auto;
  padding:12px 14px 20px;
  font-family:"Inter",system-ui,sans-serif;
  color:var(--parchment,#F4ECD8);
  scroll-behavior:smooth;
}
.st-scroll::-webkit-scrollbar{width:8px;}
.st-scroll::-webkit-scrollbar-thumb{
  background:var(--steel-2,#3A2E28); border-radius:4px;
}
.st-card{
  background:var(--steel,#2A2420);
  border:1px solid var(--steel-2,#3A2E28);
  border-radius:10px;
  padding:12px 14px;
  margin:10px 0;
  animation: st-enter .3s ease both;
  position:relative;
}
.st-card.st-thought{
  display:flex; gap:12px;
}
.st-quote{
  font-family:"Fraunces",serif;
  font-size:48px; line-height:.8;
  color:var(--copper,#C97B4A);
  flex:0 0 auto;
  margin-top:-4px;
}
.st-body{
  flex:1 1 auto;
  min-width:0;
}
.st-thought-text{
  max-height:200px;
  overflow:auto;
  position:relative;
  padding-right:4px;
  font-size:14px;
  line-height:1.55;
  color:var(--parchment,#F4ECD8);
  white-space:pre-wrap;
  word-break:break-word;
}
.st-tool{
  display:flex; gap:12px; align-items:flex-start;
}
.st-tool-icon{
  color:var(--copper-glow,#E09968);
  margin-top:2px;
  flex:0 0 auto;
}
.st-tool-name{
  font-family:"Fraunces",serif;
  font-weight:700;
  font-size:15px;
  color:var(--parchment,#F4ECD8);
}
.st-tool-args{
  margin-top:6px;
  font-family:"JetBrains Mono",monospace;
  font-size:12px;
  color:var(--parchment-dim,#C9BFA8);
  background:var(--ink,#0E0706);
  border:1px solid var(--steel-2,#3A2E28);
  border-radius:6px;
  padding:6px 9px;
  white-space:pre-wrap;
  word-break:break-word;
  max-height:160px; overflow:auto;
}
.st-error{
  border-color:var(--bordeaux,#7A1F2B);
  background: linear-gradient(180deg, rgba(122,31,43,.18), var(--steel,#2A2420));
}
.st-error-head{
  display:flex; gap:10px; align-items:center;
  color:var(--bordeaux,#7A1F2B);
  font-weight:700;
}
.st-error-first{
  margin-top:4px;
  color:var(--parchment,#F4ECD8);
  font-weight:600;
}
.st-error details{margin-top:8px;}
.st-error summary{
  cursor:pointer;
  color:var(--parchment-dim,#C9BFA8);
  font-size:12px;
}
.st-error pre{
  margin-top:6px;
  background:var(--ink,#0E0706);
  border:1px solid var(--steel-2,#3A2E28);
  border-radius:6px;
  padding:8px;
  font-family:"JetBrains Mono",monospace;
  font-size:11.5px;
  color:var(--parchment-dim,#C9BFA8);
  white-space:pre-wrap; word-break:break-word;
  max-height:260px; overflow:auto;
}
.st-metrics{
  display:flex; flex-wrap:wrap; gap:8px; align-items:center;
}
.st-iter-label{
  font-family:"Fraunces",serif;
  font-size:12px;
  letter-spacing:.14em;
  text-transform:uppercase;
  color:var(--parchment-dim,#C9BFA8);
  padding-right:6px;
}
.st-metric{
  font-family:"JetBrains Mono",monospace;
  font-size:12px;
  padding:3px 9px;
  border-radius:12px;
  border:1px solid var(--steel-2,#3A2E28);
  background:var(--ink,#0E0706);
  color:var(--parchment,#F4ECD8);
}
.st-metric.bound-memory{ color:var(--copper,#C97B4A); border-color:rgba(201,123,74,.5);}
.st-metric.bound-compute{ color:var(--basil,#5E8A5A); border-color:rgba(94,138,90,.5);}
.st-kernel{
  display:inline-flex; align-items:center; gap:6px;
  background:var(--ink,#0E0706);
  border:1px dashed var(--steel-2,#3A2E28);
  color:var(--parchment-dim,#C9BFA8);
  border-radius:14px;
  padding:4px 11px;
  font-family:"JetBrains Mono",monospace;
  font-size:11.5px;
  margin:8px 0;
  animation: st-enter .3s ease both;
}
.st-chips{
  display:flex; flex-wrap:wrap; gap:5px; margin-top:8px;
}
.st-chip{
  font-family:"JetBrains Mono",monospace;
  font-size:10.5px;
  padding:2px 8px;
  border-radius:10px;
  border:1px solid transparent;
  letter-spacing:.04em;
  text-transform:lowercase;
}
.st-chip-copper{ color:var(--copper,#C97B4A); border-color:rgba(201,123,74,.55); background:rgba(201,123,74,.1);}
.st-chip-steel{ color:var(--parchment-dim,#C9BFA8); border-color:var(--steel-2,#3A2E28); background:var(--ink,#0E0706);}
.st-chip-basil{ color:var(--basil,#5E8A5A); border-color:rgba(94,138,90,.55); background:rgba(94,138,90,.1);}
.st-chip-slate{ color:var(--parchment-dim,#C9BFA8); border-color:var(--slate,#6B6258); background:rgba(107,98,88,.15);}
.st-chip-saffron{ color:var(--saffron,#E8B14A); border-color:rgba(232,177,74,.55); background:rgba(232,177,74,.1);}
.st-chip-bordeaux{ color:var(--parchment,#F4ECD8); border-color:var(--bordeaux,#7A1F2B); background:rgba(122,31,43,.35);}
.st-chip-pulse{ animation: st-chip-pulse 1.6s ease-in-out infinite;}
.st-done{
  background:linear-gradient(120deg, rgba(232,177,74,.18), rgba(201,123,74,.08));
  border:1px solid rgba(232,177,74,.55);
  padding:18px 20px;
  border-radius:14px;
  margin:14px 0 8px;
  animation: st-enter .4s ease both;
}
.st-done-head{
  display:flex; align-items:center; gap:10px;
  color:var(--saffron,#E8B14A);
}
.st-done-title{
  font-family:"Fraunces",serif;
  font-size:24px;
  font-weight:700;
}
.st-done-meta{
  margin-top:8px;
  font-family:"JetBrains Mono",monospace;
  font-size:13px;
  color:var(--parchment,#F4ECD8);
}
.st-speedup{
  display:inline-block;
  padding:3px 10px;
  border-radius:12px;
  background:rgba(232,177,74,.2);
  color:var(--saffron,#E8B14A);
  border:1px solid rgba(232,177,74,.6);
  font-family:"JetBrains Mono",monospace;
  margin-right:10px;
}
.st-bonjour{
  margin-top:10px;
  font-family:"Fraunces",serif;
  font-style:italic;
  color:var(--parchment-dim,#C9BFA8);
  font-size:13px;
}
.st-empty{
  text-align:center;
  color:var(--parchment-dim,#C9BFA8);
  padding:40px 20px;
  font-family:"Fraunces",serif;
  font-style:italic;
}
@keyframes st-enter{
  from{opacity:0; transform:translateY(6px);}
  to  {opacity:1; transform:translateY(0);}
}
@keyframes st-chip-pulse{
  0%,100%{ box-shadow:0 0 0 0 rgba(122,31,43,.6);}
  50%    { box-shadow:0 0 0 6px rgba(122,31,43,0);}
}
</style>
"""

_JS_AUTOSCROLL = (
    "<script>"
    "(function(){const s=document.querySelector('.st-scroll');"
    "if(s){s.scrollTop=s.scrollHeight;}})();"
    "</script>"
)


# --- Event renderers ------------------------------------------------------

def _render_thought(ev: dict) -> str:
    text = ev.get("text") or ""
    stripped, tags = _extract_diagnosis_chips(text)
    chips = _render_chips(tags)
    safe = html.escape(stripped) if stripped else "<em>…</em>"
    return (
        '<div class="st-card st-thought">'
        '<div class="st-quote">&ldquo;</div>'
        '<div class="st-body">'
        f'<div class="st-thought-text">{safe}</div>'
        f'{chips}'
        '</div></div>'
    )


def _fmt_arg_value(v: Any) -> str:
    if isinstance(v, str):
        if len(v) > 80:
            return repr(v[:77] + "...")
        return repr(v)
    if isinstance(v, (int, float, bool)) or v is None:
        return repr(v)
    if isinstance(v, dict):
        inner = ", ".join(f"{k}={_fmt_arg_value(val)}" for k, val in v.items())
        return "{" + inner + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_arg_value(x) for x in v) + "]"
    return repr(v)


def _render_tool_call(ev: dict) -> str:
    name = ev.get("name") or "tool"
    args: dict = ev.get("input") or {}
    # Drop noisy keys.
    filtered = {k: v for k, v in args.items() if k != "kernel_code"}
    arg_lines = []
    for k, v in filtered.items():
        arg_lines.append(f"{html.escape(str(k))} = {html.escape(_fmt_arg_value(v))}")
    arg_body = "\n".join(arg_lines) if arg_lines else "(no args)"
    return (
        '<div class="st-card st-tool">'
        f'<div class="st-tool-icon">{ICON_WHISK}</div>'
        '<div class="st-body">'
        f'<div class="st-tool-name">{html.escape(name)}</div>'
        f'<div class="st-tool-args">{arg_body}</div>'
        '</div></div>'
    )


def _render_error(ev: dict) -> str:
    text = ev.get("text") or "Unknown error"
    first = text.splitlines()[0] if text else "Unknown error"
    return (
        '<div class="st-card st-error">'
        f'<div class="st-error-head">{ICON_CROSS}<span>Error</span></div>'
        f'<div class="st-error-first">{html.escape(first)}</div>'
        '<details><summary>Full traceback</summary>'
        f'<pre>{html.escape(text)}</pre>'
        '</details>'
        '</div>'
    )


def _render_metrics(ev: dict) -> str:
    data = ev.get("data") or {}
    iter_n = ev.get("iteration", 0)
    label = "Baseline" if iter_n == 0 else f"Iter {iter_n}"

    tf = data.get("tflops")
    bw = data.get("bandwidth_gbs")
    eff = data.get("efficiency_pct")
    bound = (data.get("bound") or "").lower()

    chips: list[str] = []
    if isinstance(tf, (int, float)):
        chips.append(f'<span class="st-metric">{tf:.2f} TF</span>')
    if isinstance(bw, (int, float)):
        chips.append(f'<span class="st-metric">{bw:.0f} GB/s</span>')
    if isinstance(eff, (int, float)):
        chips.append(f'<span class="st-metric">{eff:.1f}% eff</span>')
    if bound:
        cls = "bound-memory" if "mem" in bound else ("bound-compute" if "comp" in bound else "")
        chips.append(
            f'<span class="st-metric {cls}">{html.escape(bound)}-bound</span>'
        )

    return (
        '<div class="st-card">'
        '<div class="st-metrics">'
        f'<span class="st-iter-label">{html.escape(label)}</span>'
        f'{"".join(chips)}'
        '</div></div>'
    )


def _render_kernel(ev: dict) -> str:
    code = ev.get("code") or ""
    lines = code.count("\n") + (1 if code else 0)
    iter_n = ev.get("iteration", 0)
    return (
        f'<div class="st-kernel">&#128221; new kernel '
        f'(iter {int(iter_n)}, ~{lines} lines)</div>'
    )


def _render_done(ev: dict) -> str:
    speedup = float(ev.get("speedup") or 0.0)
    iters = int(ev.get("iterations") or 0)
    best = ev.get("best_metrics") or {}
    tf = best.get("tflops")
    eff = best.get("efficiency_pct")
    tf_s = f"{tf:.2f} TF" if isinstance(tf, (int, float)) else "—"
    eff_s = f"{eff:.1f}%" if isinstance(eff, (int, float)) else "—"
    return (
        '<div class="st-done">'
        '<div class="st-done-head">'
        f'{ICON_SPARKLE}<div class="st-done-title">Service complete</div>'
        '</div>'
        '<div class="st-done-meta">'
        f'<span class="st-speedup">{speedup:.2f}&times;</span>'
        f'best {html.escape(tf_s)} &middot; {html.escape(eff_s)} '
        f'over {iters} iteration{"s" if iters != 1 else ""}'
        '</div>'
        '<div class="st-bonjour">Bonjour &mdash; anyone can cook. *wink*</div>'
        '</div>'
    )


# --- Public API -----------------------------------------------------------

def render(state: dict) -> str:
    """Return HTML fragment for the live event stream."""
    events: list[dict] = state.get("events") or []

    if not events:
        body = (
            '<div class="st-empty">'
            "Mise en place&hellip; the kitchen awaits your command."
            '</div>'
        )
        return f'{_CSS}<div class="st-scroll">{body}</div>{_JS_AUTOSCROLL}'

    pieces: list[str] = []
    for ev in events:
        et = ev.get("type")
        try:
            if et == "thought":
                pieces.append(_render_thought(ev))
            elif et == "tool_call":
                pieces.append(_render_tool_call(ev))
            elif et == "error":
                pieces.append(_render_error(ev))
            elif et == "metrics":
                pieces.append(_render_metrics(ev))
            elif et == "kernel":
                pieces.append(_render_kernel(ev))
            elif et == "done":
                pieces.append(_render_done(ev))
            else:
                # Unknown event: render as plain thought-ish card.
                txt = html.escape(str(ev))
                pieces.append(f'<div class="st-card"><div class="st-body">{txt}</div></div>')
        except Exception as e:  # pragma: no cover - defensive
            pieces.append(
                f'<div class="st-card st-error">'
                f'<div class="st-error-head">{ICON_CROSS}<span>Render error</span></div>'
                f'<div class="st-error-first">{html.escape(str(e))}</div>'
                f'</div>'
            )

    body = "".join(pieces)
    return f'{_CSS}<div class="st-scroll">{body}</div>{_JS_AUTOSCROLL}'
