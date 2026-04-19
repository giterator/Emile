"""Emile design system: palette, typography, CSS, wordmark, and logo mark.

"Emile is the taster."  —  Anyone can cook. Anyone can write expert kernels.

This module is the single source of truth for colors, fonts, and base component
styles used across the Emile UI. It exposes:

- PALETTE + per-color Python constants
- A100 reference peaks (used by dashboard roofline)
- get_css()              -> full <style> block injected into iframe
- get_streamlit_overrides() -> <style> block injected at the Streamlit page level
- get_wordmark_html(size)   -> inline HTML for the emile wordmark
- get_logo_svg()            -> inline SVG for the chef-rat mark
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

PALETTE: dict[str, str] = {
    "espresso": "#1A0F0B",      # page background
    "ink": "#0E0706",           # deepest surfaces
    "steel": "#2A2420",         # card surfaces
    "steel_2": "#3A2E28",       # elevated surfaces / borders
    "parchment": "#F4ECD8",     # primary text on dark
    "parchment_dim": "#C9BFA8", # secondary text
    "copper": "#C97B4A",        # primary accent
    "copper_glow": "#E09968",   # hover / glow
    "saffron": "#E8B14A",       # secondary accent, success glow
    "basil": "#5E8A5A",         # success / on-target
    "bordeaux": "#7A1F2B",      # error / compute-bound warn
    "slate": "#6B6258",         # muted / pending
}

# Python-side constants (so callers can reference colors without dict lookups).
ESPRESSO = PALETTE["espresso"]
INK = PALETTE["ink"]
STEEL = PALETTE["steel"]
STEEL_2 = PALETTE["steel_2"]
PARCHMENT = PALETTE["parchment"]
PARCHMENT_DIM = PALETTE["parchment_dim"]
COPPER = PALETTE["copper"]
COPPER_GLOW = PALETTE["copper_glow"]
SAFFRON = PALETTE["saffron"]
BASIL = PALETTE["basil"]
BORDEAUX = PALETTE["bordeaux"]
SLATE = PALETTE["slate"]

# A100 reference peaks (SXM4 40GB) used by dashboard / roofline.
A100_PEAK_TFLOPS = 312.0
A100_PEAK_BANDWIDTH = 2000.0  # GB/s


# ---------------------------------------------------------------------------
# CSS — injected into the iframe body via st.components.v1.html
# ---------------------------------------------------------------------------

def get_css() -> str:
    """Return the full <style> block used inside the iframe."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --espresso:      #1A0F0B;
  --ink:           #0E0706;
  --steel:         #2A2420;
  --steel-2:       #3A2E28;
  --parchment:     #F4ECD8;
  --parchment-dim: #C9BFA8;
  --copper:        #C97B4A;
  --copper-glow:   #E09968;
  --saffron:       #E8B14A;
  --basil:         #5E8A5A;
  --bordeaux:      #7A1F2B;
  --slate:         #6B6258;

  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --hairline: 1px;
  --ease: cubic-bezier(0.2, 0.7, 0.2, 1);
  --dur: 0.2s;

  --serif: 'Fraunces', ui-serif, Georgia, serif;
  --sans:  'Inter', ui-sans-serif, system-ui, sans-serif;
  --mono:  'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  background: var(--espresso);
  color: var(--parchment);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

/* Iframe body is transparent; the host Streamlit page owns the texture
   so it reads as one continuous surface across the whole UI. */
body { background: transparent; }

h1, h2, h3 {
  font-family: var(--serif);
  font-weight: 600;
  color: var(--parchment);
  letter-spacing: -0.01em;
  margin: 0 0 0.5rem 0;
}
h1 { font-size: 2rem;   line-height: 1.15; }
h2 { font-size: 1.4rem; line-height: 1.2; }
h3 { font-size: 1.1rem; line-height: 1.25; }

p { margin: 0 0 0.75rem 0; color: var(--parchment-dim); }
a { color: var(--copper-glow); text-decoration: none; }
a:hover { color: var(--saffron); }

code, pre, .em-mono {
  font-family: var(--mono);
  font-feature-settings: "liga" 0;
}
code {
  background: var(--ink);
  border: 1px solid var(--steel-2);
  padding: 0.05rem 0.35rem;
  border-radius: var(--radius-sm);
  font-size: 0.85em;
  color: var(--parchment);
}

.em-serif { font-family: var(--serif); }
.em-mono  { font-family: var(--mono); }

/* -------- shell -------- */
.em-shell {
  max-width: 1280px;
  margin: 0 auto;
  padding: 1.25rem 1.5rem 3rem 1.5rem;
}

/* -------- cards -------- */
.em-card {
  background: linear-gradient(180deg, var(--steel) 0%, #241E1A 100%);
  border: 1px solid var(--steel-2);
  border-radius: var(--radius-lg);
  padding: 1.1rem 1.25rem;
  box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset,
              0 6px 24px rgba(0,0,0,0.35);
  transition: transform var(--dur) var(--ease), border-color var(--dur) var(--ease);
}
.em-card:hover {
  transform: translateY(-1px);
  border-color: #4a3a32;
}
.em-card h3 { margin-bottom: 0.35rem; }
.em-card .em-card-sub {
  color: var(--parchment-dim);
  font-size: 0.85rem;
  margin-bottom: 0.75rem;
}

/* -------- chips -------- */
.em-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 500;
  background: var(--ink);
  color: var(--parchment-dim);
  border: 1px solid var(--steel-2);
  line-height: 1.4;
  white-space: nowrap;
}
.em-chip.copper  { color: var(--copper-glow); border-color: rgba(201,123,74,0.35); background: rgba(201,123,74,0.08); }
.em-chip.saffron { color: var(--saffron);     border-color: rgba(232,177,74,0.35); background: rgba(232,177,74,0.08); }
.em-chip.basil   { color: #B7D5B2;            border-color: rgba(94,138,90,0.40);  background: rgba(94,138,90,0.10); }
.em-chip.bordeaux{ color: #E4A9B0;            border-color: rgba(122,31,43,0.50);  background: rgba(122,31,43,0.15); }
.em-chip.slate   { color: var(--parchment-dim); }

/* -------- buttons -------- */
.em-btn, .em-btn-ghost {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.55rem 0.95rem;
  font-family: var(--sans);
  font-weight: 500;
  font-size: 0.9rem;
  border-radius: var(--radius-md);
  border: 1px solid transparent;
  cursor: pointer;
  transition: transform var(--dur) var(--ease),
              box-shadow var(--dur) var(--ease),
              background var(--dur) var(--ease),
              border-color var(--dur) var(--ease);
}
.em-btn {
  background: linear-gradient(180deg, var(--copper) 0%, #B56A3E 100%);
  color: #1A0F0B;
  box-shadow: 0 1px 0 rgba(255,255,255,0.15) inset,
              0 4px 14px rgba(201,123,74,0.25);
}
.em-btn:hover {
  transform: translateY(-1px);
  background: linear-gradient(180deg, var(--copper-glow) 0%, var(--copper) 100%);
  box-shadow: 0 0 0 3px rgba(224,153,104,0.20),
              0 6px 18px rgba(201,123,74,0.35);
}
.em-btn:active { transform: translateY(0); }

.em-btn-ghost {
  background: transparent;
  color: var(--parchment);
  border-color: var(--steel-2);
}
.em-btn-ghost:hover {
  border-color: var(--copper);
  color: var(--copper-glow);
}

/* -------- keyboard chip -------- */
.em-kbd {
  font-family: var(--mono);
  font-size: 0.72rem;
  padding: 0.1rem 0.4rem;
  border-radius: var(--radius-sm);
  background: var(--ink);
  border: 1px solid var(--steel-2);
  color: var(--parchment-dim);
  box-shadow: 0 1px 0 var(--steel-2);
}

/* -------- hairline divider (under wordmark etc) -------- */
.em-divider-hairline {
  height: 1px;
  width: 100%;
  background: linear-gradient(90deg, transparent, var(--copper) 20%, var(--copper) 80%, transparent);
  opacity: 0.6;
  margin: 0.35rem 0 0 0;
}

/* -------- animations -------- */
@keyframes em-pulse {
  0%   { box-shadow: 0 0 0 0 rgba(201,123,74,0.55); }
  70%  { box-shadow: 0 0 0 10px rgba(201,123,74,0); }
  100% { box-shadow: 0 0 0 0 rgba(201,123,74,0); }
}
.em-pulse {
  border-radius: 999px;
  animation: em-pulse 1.6s var(--ease) infinite;
}

@keyframes em-fade-in {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0); }
}
.em-fade-in { animation: em-fade-in 0.3s var(--ease) both; }

@keyframes em-saffron-glow {
  0%, 100% { text-shadow: 0 0 18px rgba(232,177,74,0.25); }
  50%      { text-shadow: 0 0 32px rgba(232,177,74,0.55); }
}
.em-saffron-glow { animation: em-saffron-glow 2.4s var(--ease) infinite; color: var(--saffron); }

/* -------- brand bar -------- */
.em-brandbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  padding-bottom: 1rem;
  margin-bottom: 1.25rem;
  border-bottom: 1px solid var(--steel-2);
}
.em-brandbar .em-tagline {
  font-family: var(--serif);
  font-style: italic;
  color: var(--parchment-dim);
  font-size: 0.95rem;
}

/* -------- wordmark -------- */
.em-wordmark {
  display: inline-flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 0.15rem;
  color: var(--parchment);
  user-select: none;
}
.em-wordmark .em-wordmark-row {
  display: inline-flex;
  align-items: baseline;
  gap: 0.55rem;
}
.em-wordmark .em-logo {
  width: 1.1em;
  height: 1.1em;
  color: var(--copper);
  margin-right: 0.15em;
  align-self: center;
}
.em-wordmark .em-word {
  font-family: var(--serif);
  font-weight: 700;
  font-variation-settings: "opsz" 72;
  letter-spacing: -0.03em;
  line-height: 1;
}
.em-wordmark.sm .em-word { font-size: 22px; }
.em-wordmark.md .em-word { font-size: 36px; }
.em-wordmark.lg .em-word { font-size: 56px; }
.em-wordmark .em-word .em-accent-i {
  color: var(--copper);
  font-style: italic;
  padding: 0 0.02em;
}
.em-wordmark .em-word .em-dot {
  display: inline-block;
  position: relative;
  width: 0.18em;
  height: 0.18em;
  margin: 0 0.02em;
  border-radius: 50%;
  background: var(--copper);
  vertical-align: 0.82em;
}
.em-wordmark .em-word .em-dot::before {
  content: "";
  position: absolute;
  left: -0.14em;
  top: -0.28em;
  width: 0.46em;
  height: 0.28em;
  border-radius: 0.2em 0.2em 0 0;
  background: var(--copper);
  opacity: 0.9;
}
.em-wordmark .em-divider-hairline { width: 3.2em; }

/* -------- grids -------- */
.em-grid-2 {
  display: grid;
  grid-template-columns: 1.15fr 1fr;
  gap: 1rem;
}
@media (max-width: 960px) {
  .em-grid-2 { grid-template-columns: 1fr; }
}

/* -------- reveal banner -------- */
.em-reveal {
  text-align: center;
  padding: 2rem 1.5rem;
  border-radius: var(--radius-lg);
  background:
    radial-gradient(600px 200px at 50% 0%, rgba(232,177,74,0.18), transparent 70%),
    linear-gradient(180deg, var(--steel) 0%, var(--ink) 100%);
  border: 1px solid rgba(232,177,74,0.35);
  margin-top: 1rem;
}
.em-reveal .em-reveal-num {
  font-family: var(--serif);
  font-weight: 700;
  font-size: 4.5rem;
  line-height: 1;
  color: var(--saffron);
}
.em-reveal .em-reveal-label {
  font-family: var(--serif);
  font-style: italic;
  color: var(--parchment-dim);
  margin-top: 0.5rem;
}

/* -------- scrollbars -------- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--ink); }
::-webkit-scrollbar-thumb {
  background: var(--steel-2);
  border-radius: 999px;
  border: 2px solid var(--ink);
}
::-webkit-scrollbar-thumb:hover { background: var(--copper); }

/* -------- focus -------- */
:focus-visible {
  outline: 2px solid var(--copper-glow);
  outline-offset: 2px;
  border-radius: var(--radius-sm);
}

/* -------- placeholder fallback when sibling components are missing -------- */
.em-placeholder {
  font-family: var(--mono);
  color: var(--slate);
  font-size: 0.8rem;
  padding: 0.75rem;
  border: 1px dashed var(--steel-2);
  border-radius: var(--radius-md);
  text-align: center;
}
</style>
""".strip()


# ---------------------------------------------------------------------------
# Streamlit-level overrides — injected with st.markdown(unsafe_allow_html=True)
# ---------------------------------------------------------------------------

def get_streamlit_overrides() -> str:
    """Return a <style> block to inject at the top of the Streamlit page."""
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* Hide Streamlit chrome */
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }

/* App background — Parisian bistro tile texture on the host so the whole
   UI (header + iframes) shares one continuous surface. */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background-color: #1A0F0B !important;
  color: #F4ECD8 !important;
  font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
}
[data-testid="stAppViewContainer"] {
  background-image:
    repeating-linear-gradient( 45deg, rgba(244,236,216,0.035) 0 1px, transparent 1px 28px),
    repeating-linear-gradient(-45deg, rgba(244,236,216,0.035) 0 1px, transparent 1px 28px),
    radial-gradient(1200px 600px at 20% -10%, rgba(201,123,74,0.08), transparent 60%),
    radial-gradient(900px  500px at 110% 110%, rgba(232,177,74,0.05), transparent 60%) !important;
  background-attachment: fixed !important;
}
/* Streamlit iframes get a transparent background so the host texture shows through */
iframe[title="streamlit_app.components.v1.html.html"],
iframe[title^="st."],
.stCustomComponent iframe { background: transparent !important; }
[data-testid="stSidebar"] {
  background: #0E0706 !important;
  border-right: 1px solid #3A2E28 !important;
}
[data-testid="stSidebar"] * { color: #F4ECD8 !important; }

/* Headings */
h1, h2, h3, h4 {
  font-family: 'Fraunces', ui-serif, Georgia, serif !important;
  color: #F4ECD8 !important;
  letter-spacing: -0.01em;
}

/* Paragraph / markdown */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  color: #C9BFA8 !important;
  font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
}

/* Labels */
label, .stSlider label, .stTextInput label, .stSelectbox label, .stNumberInput label {
  font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
  color: #C9BFA8 !important;
  font-weight: 500 !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
  background: #2A2420 !important;
  color: #F4ECD8 !important;
  border: 1px solid #3A2E28 !important;
  border-radius: 10px !important;
  font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
  border-color: #C97B4A !important;
  box-shadow: 0 0 0 3px rgba(201,123,74,0.25) !important;
}

/* Primary button */
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(180deg, #C97B4A 0%, #B56A3E 100%) !important;
  color: #1A0F0B !important;
  border: 1px solid transparent !important;
  border-radius: 10px !important;
  font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
  font-weight: 600 !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px);
  background: linear-gradient(180deg, #E09968 0%, #C97B4A 100%) !important;
  box-shadow: 0 0 0 3px rgba(224,153,104,0.20), 0 6px 18px rgba(201,123,74,0.30) !important;
}
.stButton > button[kind="secondary"] {
  background: transparent !important;
  color: #F4ECD8 !important;
  border-color: #3A2E28 !important;
}

/* Slider track */
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: #C97B4A !important;
  border-color: #E09968 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 1.25rem;
  border-bottom: 1px solid #3A2E28;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Fraunces', ui-serif, Georgia, serif !important;
  font-size: 1.1rem !important;
  color: #C9BFA8 !important;
  background: transparent !important;
  padding: 0.5rem 0.25rem !important;
}
.stTabs [aria-selected="true"] {
  color: #F4ECD8 !important;
  border-bottom: 2px solid #C97B4A !important;
}

/* Code blocks */
code, pre, .stCode {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
}
pre, .stCode {
  background: #0E0706 !important;
  border: 1px solid #3A2E28 !important;
  border-radius: 10px !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #0E0706; }
::-webkit-scrollbar-thumb { background: #3A2E28; border-radius: 999px; border: 2px solid #0E0706; }
::-webkit-scrollbar-thumb:hover { background: #C97B4A; }
</style>
""".strip()


# ---------------------------------------------------------------------------
# Wordmark + logo mark (inline SVG, no file I/O)
# ---------------------------------------------------------------------------

def get_wordmark_html(size: str = "lg") -> str:
    """Return inline HTML for the 'emile' wordmark.

    Size tokens: 'sm' ~22px, 'md' ~36px, 'lg' ~56px.
    """
    size = size if size in ("sm", "md", "lg") else "lg"
    return f"""
<div class="em-wordmark {size}">
  <div class="em-wordmark-row">
    <span class="em-word">em<span class="em-accent-i">i</span>le</span>
  </div>
  <div class="em-divider-hairline" aria-hidden="true"></div>
</div>
""".strip()


def get_logo_svg() -> str:
    """Return the Emile chef-rat mark as an inline SVG string.

    Uses fill="currentColor" so color is inherited from the surrounding text.
    Keep in sync with ui/assets/emile_mark.svg.
    """
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" '
        'fill="currentColor" class="em-logo" aria-label="Emile">'
        '<path d="M17 8.5c-2.6 0-4.6 2-4.6 4.4 0 .6.1 1.2.3 1.7-1.2.4-2 1.5-2 2.8 '
        '0 1.5 1.1 2.7 2.6 2.9v2.2h13.4v-2.2c1.5-.2 2.6-1.4 2.6-2.9 0-1.3-.8-2.4-2-2.8 '
        '.2-.5.3-1.1.3-1.7 0-2.4-2-4.4-4.6-4.4-1 0-1.9.3-2.7.8-.8-.5-1.7-.8-2.7-.8z"/>'
        '<circle cx="20" cy="28" r="8.2"/>'
        '<path d="M12.8 22.4c-1.6-.6-3.4.1-4 1.6s.3 3.3 1.9 3.9c.4-2.2 1.2-4.1 2.1-5.5z'
        'M27.2 22.4c1.6-.6 3.4.1 4 1.6s-.3 3.3-1.9 3.9c-.4-2.2-1.2-4.1-2.1-5.5z"/>'
        '<path d="M4 28l6 2-6 2" fill="none" stroke="currentColor" stroke-width="1.3" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        '<path d="M36 28l-6 2 6 2" fill="none" stroke="currentColor" stroke-width="1.3" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="20" cy="30.2" r="1.1" fill="#1A0F0B"/>'
        '<circle cx="22.2" cy="26" r="1" fill="#1A0F0B"/>'
        "</svg>"
    )
