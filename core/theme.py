"""
core/theme.py — Aesthetic theme: CSS injection, page-header helper, Plotly template.

Direction: "Modern Minimal" — slate neutrals, indigo accent, system sans + system mono.
Fully offline — no webfont fetches, no external assets.

Public API
----------
inject_theme()                        Idempotent. Call once near the top of every page,
                                      after st.set_page_config(...) and init_state().
page_header(eyebrow, title, lede)     Replaces bare st.title(...) on each page.
pill(label, muted=False)              Inline HTML chip — caller must allow HTML.
PALETTE                               Categorical colour list for charts.
"""
from __future__ import annotations

import html

import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go


# ── Palette ────────────────────────────────────────────────────────────────
PRIMARY        = "#4f46e5"  # indigo-600
PRIMARY_HOVER  = "#4338ca"  # indigo-700
PRIMARY_LIGHT  = "#eef2ff"  # indigo-50
PRIMARY_BORDER = "#c7d2fe"  # indigo-200

INK            = "#0f172a"  # slate-900
INK_MUTED      = "#475569"  # slate-600
INK_FADED      = "#64748b"  # slate-500
SURFACE        = "#ffffff"
SURFACE_ALT    = "#f8fafc"  # slate-50
BORDER         = "#e2e8f0"  # slate-200
BORDER_STRONG  = "#cbd5e1"  # slate-300

PALETTE = [
    "#4f46e5",  # indigo
    "#0ea5e9",  # sky
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#a855f7",  # purple
    "#14b8a6",  # teal
    "#64748b",  # slate
]

FONT_SANS = ('-apple-system, BlinkMacSystemFont, "Segoe UI", '
             '"Inter", Roboto, "Helvetica Neue", Arial, sans-serif')
FONT_MONO = ('ui-monospace, SFMono-Regular, "SF Mono", '
             'Menlo, Consolas, "Liberation Mono", monospace')


# ── Plotly template ────────────────────────────────────────────────────────
def _register_plotly_template() -> None:
    t = go.layout.Template()
    t.layout.colorway      = PALETTE
    t.layout.font          = dict(family=FONT_SANS, color=INK, size=12)
    t.layout.plot_bgcolor  = SURFACE
    t.layout.paper_bgcolor = SURFACE
    t.layout.title         = dict(font=dict(size=14, color=INK))
    axis = dict(
        gridcolor=BORDER,
        zerolinecolor=BORDER_STRONG,
        linecolor=BORDER_STRONG,
        ticks="outside",
        tickcolor=BORDER_STRONG,
        tickfont=dict(color=INK_MUTED),
    )
    t.layout.xaxis  = axis
    t.layout.yaxis  = axis
    t.layout.legend = dict(
        bgcolor="rgba(255,255,255,0)",
        font=dict(color=INK_MUTED, size=11),
    )
    pio.templates["eda"] = t
    pio.templates.default = "simple_white+eda"


# ── CSS ────────────────────────────────────────────────────────────────────
_CSS = f"""
<style>
/* Typography ----------------------------------------------------------- */
/* Set the font on the root only, so it cascades by inheritance.
   Streamlit's own icon spans (Material Symbols) have direct font-family
   rules that will override inheritance — leave them alone. */
html, body, .stApp {{
    font-family: {FONT_SANS};
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}}
code, pre, kbd, samp, .mono,
[data-testid="stMetricValue"] {{ font-family: {FONT_MONO}; }}

/* Headings ------------------------------------------------------------- */
h1, h2, h3, h4 {{ letter-spacing: -0.02em; color: {INK}; font-weight: 600; }}
h1 {{ font-size: 1.875rem; line-height: 1.2; }}
h2 {{ font-size: 1.375rem; line-height: 1.3; }}
h3 {{ font-size: 1.125rem; line-height: 1.4; }}

/* App canvas ----------------------------------------------------------- */
.stApp {{ background-color: {SURFACE}; }}

/* Sidebar -------------------------------------------------------------- */
[data-testid="stSidebar"] {{
    background-color: {SURFACE_ALT};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {{
    border-color: {BORDER}; margin: .75rem 0;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ font-size: .95rem; }}

/* Metric cards --------------------------------------------------------- */
[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}}
[data-testid="stMetricLabel"] {{
    color: {INK_FADED} !important;
    font-size: .75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: .05em;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.625rem !important;
    font-weight: 600 !important;
    letter-spacing: -.02em;
    color: {INK} !important;
}}

/* Bordered containers -------------------------------------------------- */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 12px !important;
    border-color: {BORDER} !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, .03);
}}

/* Expanders ------------------------------------------------------------ */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    background: {SURFACE};
}}
[data-testid="stExpander"] summary {{ font-weight: 500; color: {INK_MUTED}; }}
[data-testid="stExpander"] summary:hover {{ color: {PRIMARY}; }}

/* Buttons -------------------------------------------------------------- */
.stButton > button {{
    border-radius: 8px;
    font-weight: 500;
    transition: all 120ms ease;
    border-color: {BORDER};
}}
.stButton > button:hover {{ border-color: {PRIMARY_BORDER}; color: {PRIMARY}; }}
.stButton > button[kind="primary"] {{
    background: {PRIMARY}; border-color: {PRIMARY}; color: white;
}}
.stButton > button[kind="primary"]:hover {{
    background: {PRIMARY_HOVER}; border-color: {PRIMARY_HOVER};
    box-shadow: 0 4px 10px rgba(79, 70, 229, .25);
    color: white;
}}

/* DataFrame ------------------------------------------------------------ */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow: hidden;
}}

/* Tabs ----------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
    font-weight: 500; color: {INK_FADED};
    padding: .5rem 1rem; border-radius: 8px 8px 0 0;
}}
.stTabs [aria-selected="true"] {{ color: {PRIMARY} !important; }}

/* Soft dividers -------------------------------------------------------- */
[data-testid="stMarkdownContainer"] hr {{
    border-color: {BORDER}; margin: 1.5rem 0;
}}

/* Page-header helper classes ------------------------------------------ */
.eyebrow {{
    color: {PRIMARY};
    font-size: .75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: .25rem;
}}
.lede {{
    color: {INK_MUTED};
    font-size: 1.0625rem;
    line-height: 1.6;
    max-width: 64ch;
    margin: .25rem 0 1.5rem 0;
}}

/* Feature card content (inside st.container(border=True)) ------------- */
.feature-num {{
    color: {PRIMARY};
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
}}
.feature-title {{
    margin: .25rem 0 .375rem 0;
    font-size: 1.0625rem;
    font-weight: 600;
    color: {INK};
}}
.feature-body {{
    color: {INK_FADED};
    font-size: .875rem;
    margin: 0 0 .75rem 0;
    line-height: 1.5;
}}

/* Pill / chip ---------------------------------------------------------- */
.pill {{
    display: inline-block;
    background: {PRIMARY_LIGHT};
    color: {PRIMARY_HOVER};
    padding: .125rem .625rem;
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 500;
    margin-right: .25rem;
}}
.pill-muted {{
    background: {SURFACE_ALT};
    color: {INK_FADED};
    border: 1px solid {BORDER};
}}

/* Scrollbar polish (WebKit) ------------------------------------------- */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: {BORDER_STRONG}; border-radius: 999px; }}
::-webkit-scrollbar-thumb:hover {{ background: {INK_FADED}; }}
</style>
"""


# ── Public API ────────────────────────────────────────────────────────────
def inject_theme() -> None:
    """Inject CSS and register the Plotly template. Idempotent — safe per page."""
    st.markdown(_CSS, unsafe_allow_html=True)
    _register_plotly_template()


def page_header(eyebrow: str, title: str, lede: str | None = None) -> None:
    """Eyebrow tag + h1 + optional lede paragraph."""
    parts = [
        f'<div class="eyebrow">{html.escape(eyebrow)}</div>',
        f'<h1 style="margin:0 0 .25rem 0">{html.escape(title)}</h1>',
    ]
    if lede:
        parts.append(f'<p class="lede">{html.escape(lede)}</p>')
    st.markdown("\n".join(parts), unsafe_allow_html=True)


def pill(label: str, muted: bool = False) -> str:
    """Inline HTML chip. Caller must render with unsafe_allow_html=True."""
    cls = "pill pill-muted" if muted else "pill"
    return f'<span class="{cls}">{html.escape(str(label))}</span>'
