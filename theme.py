"""
theme.py
--------
Centralized theme management and responsive CSS injection for the AI News Summarizer application.
Provides seamless Light Mode and Dark Mode support with high contrast, elegant typography,
and comprehensive styling across all Streamlit components, BaseWeb controls, and Plotly charts.

Key Features:
- ☀️ Light Mode: Crisp white surfaces (#FFFFFF), soft background (#F8FAFC), slate text (#111827),
  subtle borders (#D1D5DB), and vivid blue accents (#2563EB).
- 🌙 Dark Mode: Modern deep background (#0E1117), dark slate sidebar (#111827), card containers (#161B22),
  input surfaces (#1C2330), borders (#30363D), bright text (#F5F7FA), and royal blue accents (#3B82F6).
- Non-destructive: Theme changes are purely visual (stored in Streamlit session state) and never
  affect active articles, filters, search queries, database records, or AI states.
- Dynamic Plotly chart template synchronization ('plotly_white' vs 'plotly_dark').
- Comprehensive Streamlit 1.x / BaseWeb selector coverage for selectboxes, multiselects,
  popovers, dropdown menus, text/number/date inputs, buttons, metrics, tabs, and expanders.
"""

from typing import Any
import streamlit as st

# ---------------------------------------------------------------------------
# Centralized Theme Palette Tokens
# ---------------------------------------------------------------------------

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "app_bg": "#F8FAFC",
        "surface_bg": "#F1F5F9",
        "card_bg": "#FFFFFF",
        "card_secondary": "#F8FAFC",
        "input_bg": "#FFFFFF",
        "text_primary": "#111827",
        "text_secondary": "#4B5563",
        "text_muted": "#6B7280",
        "border_color": "#D1D5DB",
        "border_hover": "#9CA3AF",
        "accent_primary": "#2563EB",
        "accent_hover": "#1D4ED8",
        "accent_light": "#EFF6FF",
        "sidebar_bg": "#FFFFFF",
        "sidebar_border": "#E5E7EB",
        "sidebar_text": "#111827",
        "sidebar_header": "#4B5563",
        "success_color": "#16A34A",
        "warning_color": "#D97706",
        "error_color": "#DC2626",
        "kp_card_bg": "#F8FAFC",
        "kp_border": "#2563EB",
        "kp_badge_bg": "#DBEAFE",
        "kp_badge_text": "#1D4ED8",
        "keyword_chip_bg": "#EFF6FF",
        "keyword_chip_text": "#1D4ED8",
        "keyword_chip_border": "#BFDBFE",
        "category_pill_bg": "#ECFDF5",
        "category_pill_text": "#047857",
        "category_pill_border": "#A7F3D0",
        "pos_sentiment_bg": "#ECFDF5",
        "pos_sentiment_text": "#047857",
        "pos_sentiment_border": "#A7F3D0",
        "neu_sentiment_bg": "#F1F5F9",
        "neu_sentiment_text": "#475569",
        "neu_sentiment_border": "#CBD5E1",
        "neg_sentiment_bg": "#FFF1F2",
        "neg_sentiment_text": "#E11D48",
        "neg_sentiment_border": "#FECDD3",
        "ai_result_bg": "#F8FAFC",
        "ai_result_border": "#E2E8F0",
        "reader_bg": "#FFFFFF",
        "reader_text": "#111827",
        "reader_border": "#E2E8F0",
        "meta_pill_bg": "#F1F5F9",
        "meta_pill_text": "#4B5563",
        "status_pill_bg": "rgba(34, 197, 94, 0.12)",
        "status_pill_text": "#16A34A",
        "status_pill_border": "rgba(34, 197, 94, 0.25)",
        "plotly_template": "plotly_white",
        "chart_bg": "#FFFFFF",
        "chart_paper_bg": "#FFFFFF",
        "chart_font_color": "#111827",
    },
    "dark": {
        "app_bg": "#0E1117",
        "surface_bg": "#1C2330",
        "card_bg": "#161B22",
        "card_secondary": "#1C2330",
        "input_bg": "#1C2330",
        "text_primary": "#F5F7FA",
        "text_secondary": "#B8C0CC",
        "text_muted": "#8B949E",
        "border_color": "#30363D",
        "border_hover": "#484F58",
        "accent_primary": "#3B82F6",
        "accent_hover": "#2563EB",
        "accent_light": "rgba(59, 130, 246, 0.18)",
        "sidebar_bg": "#111827",
        "sidebar_border": "#1F2937",
        "sidebar_text": "#F5F7FA",
        "sidebar_header": "#9CA3AF",
        "success_color": "#22C55E",
        "warning_color": "#F59E0B",
        "error_color": "#EF4444",
        "kp_card_bg": "#161B22",
        "kp_border": "#3B82F6",
        "kp_badge_bg": "#1E3A8A",
        "kp_badge_text": "#93C5FD",
        "keyword_chip_bg": "#1E293B",
        "keyword_chip_text": "#93C5FD",
        "keyword_chip_border": "#334155",
        "category_pill_bg": "#064E3B",
        "category_pill_text": "#6EE7B7",
        "category_pill_border": "#047857",
        "pos_sentiment_bg": "#064E3B",
        "pos_sentiment_text": "#6EE7B7",
        "pos_sentiment_border": "#047857",
        "neu_sentiment_bg": "#1E293B",
        "neu_sentiment_text": "#94A3B8",
        "neu_sentiment_border": "#334155",
        "neg_sentiment_bg": "#4C0519",
        "neg_sentiment_text": "#FDA4AF",
        "neg_sentiment_border": "#9F1239",
        "ai_result_bg": "#161B22",
        "ai_result_border": "#30363D",
        "reader_bg": "#111827",
        "reader_text": "#F5F7FA",
        "reader_border": "#30363D",
        "meta_pill_bg": "#1C2330",
        "meta_pill_text": "#B8C0CC",
        "status_pill_bg": "rgba(34, 197, 94, 0.18)",
        "status_pill_text": "#4ADE80",
        "status_pill_border": "rgba(74, 222, 128, 0.35)",
        "plotly_template": "plotly_dark",
        "chart_bg": "#161B22",
        "chart_paper_bg": "#161B22",
        "chart_font_color": "#F5F7FA",
    },
}


# ---------------------------------------------------------------------------
# Theme Helper Functions
# ---------------------------------------------------------------------------

def get_current_theme() -> str:
    """
    Returns the currently active theme name ('light' or 'dark').
    Defaults to 'dark' if not explicitly initialized in session state.
    """
    return st.session_state.get("theme", "dark")


def get_theme_tokens(theme_name: str | None = None) -> dict[str, str]:
    """
    Returns the dictionary of color tokens for the specified or active theme.
    """
    th = theme_name or get_current_theme()
    return THEMES.get(th, THEMES["dark"])


def get_theme_colors(theme_name: str | None = None) -> dict[str, str]:
    """
    Alias for get_theme_tokens(). Returns color tokens for active theme.
    """
    return get_theme_tokens(theme_name)


def get_plotly_template(theme_name: str | None = None) -> str:
    """
    Returns the appropriate Plotly chart template ('plotly_white' or 'plotly_dark')
    matching the current theme.
    """
    tokens = get_theme_tokens(theme_name)
    return tokens.get("plotly_template", "plotly_dark")


def get_chart_template(theme_name: str | None = None) -> str:
    """
    Alias for get_plotly_template(). Returns plotly chart template name.
    """
    return get_plotly_template(theme_name)


def apply_theme(theme_name: str | None = None) -> None:
    """
    Injects custom CSS matching the current theme into the Streamlit application.
    Controls global app background, typography, borders, cards, pills, sidebar, inputs,
    popovers, dropdown menus, buttons, metrics, tabs, and expanders.
    """
    t = get_theme_tokens(theme_name)
    is_dark = (get_current_theme() == "dark") if theme_name is None else (theme_name == "dark")

    css = f"""
    <style>
    /* ===================================================================
       Root Theme CSS Variables ({'Dark Mode' if is_dark else 'Light Mode'})
       =================================================================== */
    :root {{
        --bg: {t['app_bg']};
        --surface: {t['surface_bg']};
        --card: {t['card_bg']};
        --card-secondary: {t['card_secondary']};
        --input-bg: {t['input_bg']};
        --text-primary: {t['text_primary']};
        --text-secondary: {t['text_secondary']};
        --text-muted: {t['text_muted']};
        --border: {t['border_color']};
        --border-hover: {t['border_hover']};
        --accent: {t['accent_primary']};
        --accent-hover: {t['accent_hover']};
        --accent-light: {t['accent_light']};
        --sidebar-bg: {t['sidebar_bg']};
        --sidebar-border: {t['sidebar_border']};
        --sidebar-text: {t['sidebar_text']};
        --sidebar-header: {t['sidebar_header']};
        --success: {t['success_color']};
        --warning: {t['warning_color']};
        --error: {t['error_color']};
    }}

    /* ===================================================================
       1. Global App & Header Layout
       =================================================================== */
    .stApp {{
        background-color: var(--bg) !important;
        background: var(--bg) !important;
        color: var(--text-primary) !important;
        transition: background-color 0.2s ease, color 0.2s ease;
    }}

    header[data-testid="stHeader"] {{
        background-color: var(--bg) !important;
        background: var(--bg) !important;
    }}

    [data-testid="stToolbar"] {{
        color: var(--text-secondary) !important;
    }}

    /* Sidebar container */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarHeader"],
    [data-testid="stSidebarNav"] {{
        background-color: var(--sidebar-bg) !important;
        background: var(--sidebar-bg) !important;
        color: var(--sidebar-text) !important;
    }}

    section[data-testid="stSidebar"],
    [data-testid="stSidebar"] {{
        border-right: 1px solid var(--sidebar-border) !important;
    }}

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] small {{
        color: var(--sidebar-text);
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] strong {{
        color: var(--sidebar-header) !important;
    }}

    /* Global Typography */
    h1, h2, h3, h4, h5, h6, strong, b {{
        color: var(--text-primary) !important;
    }}

    p, span, label, div {{
        color: var(--text-secondary);
    }}

    .stCaption, small {{
        color: var(--text-muted) !important;
    }}

    hr, [data-testid="stDivider"] {{
        border-color: var(--border) !important;
    }}

    /* ===================================================================
       2. Streamlit BaseWeb Selectboxes & Multiselects
       =================================================================== */
    /* Selectbox wrapper & container */
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stMultiSelect"] div[data-baseweb="select"],
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div {{
        background-color: var(--input-bg) !important;
        background: var(--input-bg) !important;
        border: 1px solid var(--border) !important;
        border-color: var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }}

    div[data-baseweb="select"]:hover > div,
    div[data-baseweb="select"]:focus-within > div {{
        border-color: var(--accent) !important;
    }}

    /* Clear all inner backgrounds in selectbox control to avoid split colors */
    div[data-baseweb="select"] > div * {{
        background-color: transparent !important;
        background: transparent !important;
    }}

    /* Selectbox selected value and input text */
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] p,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p {{
        color: var(--text-primary) !important;
        background-color: transparent !important;
        background: transparent !important;
    }}

    /* Selectbox dropdown arrow & icons */
    div[data-baseweb="select"] svg,
    div[data-baseweb="select"] svg path {{
        fill: var(--text-primary) !important;
        stroke: var(--text-primary) !important;
        color: var(--text-primary) !important;
    }}

    /* Multiselect tags */
    div[data-baseweb="tag"] {{
        background-color: var(--surface) !important;
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }}

    div[data-baseweb="tag"] span {{
        color: var(--text-primary) !important;
    }}

    div[data-baseweb="tag"] svg {{
        fill: var(--text-muted) !important;
    }}

    /* Selectbox / Multiselect Label */
    [data-testid="stSelectbox"] label,
    [data-testid="stMultiSelect"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stTextArea"] label,
    [data-testid="stNumberInput"] label,
    [data-testid="stDateInput"] label,
    [data-testid="stRadio"] label {{
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }}

    /* ===================================================================
       3. Popovers & Dropdown Menus (Open Selectbox Options)
       =================================================================== */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] > ul,
    ul[role="listbox"],
    ul[data-testid="stSelectboxVirtualDropdown"],
    div[data-testid="stSelectboxVirtualDropdown"] {{
        background-color: var(--card) !important;
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, {'0.6' if is_dark else '0.15'}) !important;
    }}

    /* Dropdown Options */
    li[role="option"],
    ul[role="listbox"] li,
    div[data-baseweb="menu"] li,
    [data-testid="stSelectboxVirtualDropdown"] li {{
        background-color: var(--card) !important;
        background: var(--card) !important;
        color: var(--text-primary) !important;
        padding: 8px 14px !important;
        font-size: 0.9rem !important;
        transition: background-color 0.15s ease, color 0.15s ease;
        border-bottom: 1px solid var(--border) !important;
    }}

    li[role="option"]:last-child,
    ul[role="listbox"] li:last-child {{
        border-bottom: none !important;
    }}

    li[role="option"] *,
    ul[role="listbox"] li * {{
        color: var(--text-primary) !important;
        background-color: transparent !important;
        background: transparent !important;
    }}

    /* Hovered & Selected Option */
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"],
    li[role="option"]:focus,
    ul[role="listbox"] li:hover,
    ul[role="listbox"] li[aria-selected="true"] {{
        background-color: var(--surface) !important;
        background: var(--surface) !important;
        color: var(--accent) !important;
    }}

    li[role="option"]:hover *,
    li[role="option"][aria-selected="true"] *,
    li[role="option"]:focus *,
    ul[role="listbox"] li:hover *,
    ul[role="listbox"] li[aria-selected="true"] * {{
        color: var(--accent) !important;
        background-color: transparent !important;
        background: transparent !important;
    }}

    /* Dropdown Scrollbars */
    div[data-baseweb="popover"] ::-webkit-scrollbar,
    ul[role="listbox"]::-webkit-scrollbar,
    div[data-baseweb="menu"]::-webkit-scrollbar,
    [data-testid="stSelectboxVirtualDropdown"]::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    div[data-baseweb="popover"] ::-webkit-scrollbar-track,
    ul[role="listbox"]::-webkit-scrollbar-track,
    div[data-baseweb="menu"]::-webkit-scrollbar-track,
    [data-testid="stSelectboxVirtualDropdown"]::-webkit-scrollbar-track {{
        background: var(--card);
    }}
    div[data-baseweb="popover"] ::-webkit-scrollbar-thumb,
    ul[role="listbox"]::-webkit-scrollbar-thumb,
    div[data-baseweb="menu"]::-webkit-scrollbar-thumb,
    [data-testid="stSelectboxVirtualDropdown"]::-webkit-scrollbar-thumb {{
        background: var(--border);
        border-radius: 4px;
    }}

    /* ===================================================================
       4. Text Inputs, Number Inputs, Date Inputs, Textareas
       =================================================================== */
    div[data-baseweb="input"],
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="textarea"],
    div[data-baseweb="textarea"] > textarea,
    input[type="text"],
    input[type="number"],
    textarea {{
        background-color: var(--input-bg) !important;
        background: var(--input-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-color: var(--border) !important;
        border-radius: 8px !important;
    }}

    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input {{
        background-color: transparent !important;
        background: transparent !important;
        color: var(--text-primary) !important;
        border: none !important;
    }}

    input::placeholder,
    textarea::placeholder {{
        color: var(--text-muted) !important;
        opacity: 0.8;
    }}

    /* Number input step up/down buttons */
    div[data-testid="stNumberInputStepDown"],
    div[data-testid="stNumberInputStepUp"],
    div[data-baseweb="input"] button {{
        background-color: var(--surface) !important;
        background: var(--surface) !important;
        color: var(--text-primary) !important;
        border-color: var(--border) !important;
    }}

    div[data-testid="stNumberInputStepDown"] svg,
    div[data-testid="stNumberInputStepUp"] svg,
    div[data-baseweb="input"] button svg {{
        fill: var(--text-primary) !important;
        stroke: var(--text-primary) !important;
        color: var(--text-primary) !important;
    }}

    /* Date input calendar picker */
    div[data-baseweb="calendar"],
    div[data-baseweb="calendar"] * {{
        background-color: var(--card) !important;
        color: var(--text-primary) !important;
    }}
    div[data-baseweb="calendar"] [aria-selected="true"] {{
        background-color: var(--accent) !important;
        color: #FFFFFF !important;
    }}

    /* ===================================================================
       5. Buttons (Primary, Secondary, Link)
       =================================================================== */
    /* Primary Button */
    div[data-testid="stButton"] > button[kind="primary"],
    button[data-testid="baseButton-primary"],
    button[kind="primary"] {{
        background-color: var(--accent) !important;
        background: var(--accent) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--accent) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }}

    div[data-testid="stButton"] > button[kind="primary"] *,
    button[data-testid="baseButton-primary"] *,
    button[kind="primary"] * {{
        color: #FFFFFF !important;
        background-color: transparent !important;
        background: transparent !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }}

    div[data-testid="stButton"] > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[kind="primary"]:hover {{
        background-color: var(--accent-hover) !important;
        background: var(--accent-hover) !important;
        border-color: var(--accent-hover) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }}

    div[data-testid="stButton"] > button[kind="primary"]:hover *,
    button[data-testid="baseButton-primary"]:hover *,
    button[kind="primary"]:hover * {{
        color: #FFFFFF !important;
    }}

    /* Secondary Button */
    div[data-testid="stButton"] > button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    button[kind="secondary"],
    div[data-testid="stButton"] > button:not([kind="primary"]) {{
        background-color: var(--surface) !important;
        background: var(--surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }}

    div[data-testid="stButton"] > button[kind="secondary"] *,
    button[data-testid="baseButton-secondary"] *,
    button[kind="secondary"] *,
    div[data-testid="stButton"] > button:not([kind="primary"]) * {{
        color: var(--text-primary) !important;
        background-color: transparent !important;
        background: transparent !important;
        fill: var(--text-primary) !important;
    }}

    div[data-testid="stButton"] > button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    button[kind="secondary"]:hover,
    div[data-testid="stButton"] > button:not([kind="primary"]):hover {{
        background-color: var(--card) !important;
        border-color: var(--border-hover) !important;
        color: var(--accent) !important;
    }}

    div[data-testid="stButton"] > button[kind="secondary"]:hover *,
    button[data-testid="baseButton-secondary"]:hover *,
    button[kind="secondary"]:hover *,
    div[data-testid="stButton"] > button:not([kind="primary"]):hover * {{
        color: var(--accent) !important;
    }}

    /* Link Button */
    div[data-testid="stLinkButton"] > a,
    a[data-testid="baseButton-secondary"] {{
        background-color: var(--surface) !important;
        background: var(--surface) !important;
        color: var(--accent) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        text-decoration: none !important;
        transition: all 0.2s ease;
    }}

    div[data-testid="stLinkButton"] > a:hover,
    a[data-testid="baseButton-secondary"]:hover {{
        background-color: var(--card) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }}

    /* ===================================================================
       6. Containers & Cards (st.container)
       =================================================================== */
    div[data-testid="stVerticalBlockBorderWrapper"] > div,
    div[data-testid="stContainer"][data-border="true"],
    div[data-testid="stContainer"] {{
        background-color: var(--card) !important;
        background: var(--card) !important;
        border-color: var(--border) !important;
        border-radius: 10px;
    }}

    /* ===================================================================
       7. Metrics (st.metric)
       =================================================================== */
    div[data-testid="stMetric"] {{
        background-color: var(--card) !important;
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px;
        padding: 12px 16px;
    }}

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {{
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {{
        color: var(--text-primary) !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }}

    /* ===================================================================
       8. Tabs (st.tabs)
       =================================================================== */
    div[data-testid="stTabs"] {{
        background-color: transparent !important;
    }}

    div[data-baseweb="tab-list"] {{
        background-color: transparent !important;
        border-bottom: 2px solid var(--border) !important;
        gap: 8px;
    }}

    button[data-baseweb="tab"] {{
        background-color: transparent !important;
        color: var(--text-muted) !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 10px 16px !important;
        border-radius: 6px 6px 0 0 !important;
        transition: color 0.2s ease, background-color 0.2s ease;
    }}

    button[data-baseweb="tab"]:hover {{
        color: var(--text-primary) !important;
        background-color: var(--surface) !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--accent) !important;
        font-weight: 700 !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] * {{
        color: var(--accent) !important;
    }}

    div[data-baseweb="tab-highlight"] {{
        background-color: var(--accent) !important;
    }}

    div[data-baseweb="tab-border"] {{
        background-color: var(--border) !important;
    }}

    /* ===================================================================
       9. Radio Buttons & Checkboxes
       =================================================================== */
    div[data-testid="stRadio"] label,
    div[data-testid="stCheckbox"] label {{
        color: var(--text-primary) !important;
    }}

    div[data-testid="stRadio"] span,
    div[data-testid="stRadio"] p,
    div[data-testid="stCheckbox"] span,
    div[data-testid="stCheckbox"] p {{
        color: var(--text-primary) !important;
    }}

    div[data-testid="stRadio"] div[role="radiogroup"] {{
        gap: 8px;
    }}

    /* ===================================================================
       10. Expanders (st.expander)
       =================================================================== */
    div[data-testid="stExpander"] {{
        background-color: var(--card) !important;
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }}

    div[data-testid="stExpander"] summary {{
        background-color: var(--card) !important;
        background: var(--card) !important;
        color: var(--text-primary) !important;
        padding: 10px 14px !important;
    }}

    div[data-testid="stExpander"] summary:hover {{
        color: var(--accent) !important;
    }}

    div[data-testid="stExpander"] summary * {{
        color: var(--text-primary) !important;
    }}

    div[data-testid="stExpanderDetails"] {{
        background-color: var(--card) !important;
        background: var(--card) !important;
        border-top: 1px solid var(--border) !important;
        color: var(--text-secondary) !important;
        padding: 14px !important;
    }}

    /* ===================================================================
       11. Alerts (st.info, st.success, st.warning, st.error)
       =================================================================== */
    div[data-testid="stAlert"] {{
        background-color: var(--surface) !important;
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }}

    div[data-testid="stAlert"] * {{
        color: var(--text-primary) !important;
    }}

    /* ===================================================================
       12. Chat Messages & Chat Input (Feature 15)
       =================================================================== */
    div[data-testid="stChatMessage"] {{
        background-color: var(--card) !important;
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        margin-bottom: 10px !important;
    }}

    div[data-testid="stChatMessage"] * {{
        color: var(--text-primary) !important;
    }}

    div[data-testid="stChatInput"] {{
        border-color: var(--border) !important;
    }}

    div[data-testid="stChatInput"] textarea {{
        background-color: var(--input-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }}

    div[data-testid="stChatInput"] button {{
        background-color: var(--accent) !important;
        color: #FFFFFF !important;
    }}

    /* ===================================================================
       13. DataFrames & Tables (st.dataframe, st.table)
       =================================================================== */
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {{
        background-color: var(--card) !important;
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }}

    div[data-testid="stDataFrame"] * {{
        color: var(--text-primary) !important;
    }}

    /* ===================================================================
       14. Custom Modular UI Component Classes
       =================================================================== */
    /* Header Titles */
    .main-header-title {{
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.2rem !important;
        letter-spacing: -0.02em;
        color: var(--text-primary) !important;
    }}

    .main-header-subtitle {{
        font-size: 1.05rem !important;
        color: var(--text-muted) !important;
        margin-bottom: 0.8rem !important;
    }}

    /* Online Status Indicator Pill */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: {t['status_pill_bg']};
        color: {t['status_pill_text']};
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid {t['status_pill_border']};
        margin-bottom: 1.2rem;
    }}

    .status-dot {{
        width: 8px;
        height: 8px;
        background-color: {t['status_pill_text']};
        border-radius: 50%;
        display: inline-block;
    }}

    /* Sidebar Section Titles & Features */
    .sidebar-section-title {{
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--sidebar-header) !important;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }}

    .sidebar-feature-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        padding: 3px 0;
        color: var(--sidebar-text) !important;
    }}

    .sidebar-features-list {{
        font-size: 0.88rem;
        line-height: 1.8;
        color: var(--sidebar-text);
    }}

    /* Database Status Card */
    .db-status-card {{
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 4px;
    }}

    .db-status-text {{
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }}

    .db-status-text strong {{
        color: var(--text-primary);
    }}

    /* Feed Information Banner */
    .feed-info-banner {{
        background-color: var(--surface);
        border: 1px solid var(--border);
        padding: 10px 16px;
        border-radius: 8px;
        margin-bottom: 18px;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }}

    .feed-info-banner strong {{
        color: var(--text-primary);
    }}

    /* Meta Information Pills */
    .meta-pill {{
        background-color: var(--surface);
        color: var(--text-secondary) !important;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid var(--border);
    }}

    .meta-pill-provider {{
        background-color: var(--accent-light);
        color: var(--accent) !important;
        font-weight: 600;
        border: 1px solid var(--accent)33;
    }}

    /* Image Placeholder Box */
    .image-placeholder {{
        width: 100%;
        height: 140px;
        background: linear-gradient(135deg, var(--surface) 0%, var(--card) 100%);
        border: 1px dashed var(--border);
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: var(--text-muted);
        font-size: 0.82rem;
        font-weight: 500;
        text-align: center;
        padding: 12px;
    }}

    /* AI Key Points Display */
    .kp-card {{
        background-color: {t['kp_card_bg']};
        border-left: 4px solid {t['kp_border']};
        border: 1px solid var(--border);
        border-left-width: 4px;
        border-radius: 0 8px 8px 0;
        padding: 10px 16px;
        margin-bottom: 10px;
        font-size: 0.95rem;
        line-height: 1.5;
        color: var(--text-primary);
    }}

    .kp-badge {{
        display: inline-block;
        background-color: {t['kp_badge_bg']};
        color: {t['kp_badge_text']} !important;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 8px;
    }}

    /* Keyword Tag Chips */
    .keyword-chip {{
        display: inline-block;
        background-color: {t['keyword_chip_bg']};
        color: {t['keyword_chip_text']} !important;
        padding: 5px 12px;
        margin: 4px 6px 4px 0;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid {t['keyword_chip_border']};
    }}

    /* AI Category Pill */
    .category-pill {{
        display: inline-block;
        background-color: {t['category_pill_bg']};
        color: {t['category_pill_text']} !important;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        border: 1px solid {t['category_pill_border']};
        margin-bottom: 10px;
    }}

    /* AI Sentiment Pills */
    .sentiment-pill-positive {{
        display: inline-block;
        background-color: {t['pos_sentiment_bg']};
        color: {t['pos_sentiment_text']} !important;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        border: 1px solid {t['pos_sentiment_border']};
        margin-bottom: 10px;
    }}

    .sentiment-pill-neutral {{
        display: inline-block;
        background-color: {t['neu_sentiment_bg']};
        color: {t['neu_sentiment_text']} !important;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        border: 1px solid {t['neu_sentiment_border']};
        margin-bottom: 10px;
    }}

    .sentiment-pill-negative {{
        display: inline-block;
        background-color: {t['neg_sentiment_bg']};
        color: {t['neg_sentiment_text']} !important;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 1.15rem;
        font-weight: 700;
        border: 1px solid {t['neg_sentiment_border']};
        margin-bottom: 10px;
    }}

    /* AI Output & Result Box */
    .ai-result-box {{
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 18px;
        margin-top: 12px;
        margin-bottom: 16px;
        line-height: 1.6;
        color: var(--text-primary);
    }}

    .ai-result-meta {{
        font-size: 0.85rem;
        color: var(--text-muted);
        border-top: 1px solid var(--border);
        padding-top: 8px;
    }}

    /* Article Reader Box */
    .article-reader-box {{
        background-color: {t['reader_bg']};
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 18px 22px;
        font-size: 0.96rem;
        line-height: 1.7;
        color: {t['reader_text']} !important;
        max-height: 380px;
        overflow-y: auto;
        margin-top: 10px;
        margin-bottom: 16px;
    }}

    /* Related News & Comparison Cards */
    .related-card {{
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: border-color 0.2s ease;
    }}

    .related-card:hover {{
        border-color: var(--border-hover);
    }}

    .badge-duplicate {{
        display: inline-block;
        background-color: {t['pos_sentiment_bg']};
        color: {t['pos_sentiment_text']} !important;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid {t['pos_sentiment_border']};
    }}

    .badge-related {{
        display: inline-block;
        background-color: var(--accent-light);
        color: var(--accent) !important;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid var(--accent)44;
    }}

    .similarity-score-pill {{
        display: inline-block;
        background-color: var(--surface);
        color: var(--text-primary) !important;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid var(--border);
    }}

    /* Multi-Source Comparison Differences Card */
    .diff-card {{
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px;
        height: 100%;
    }}

    .diff-card-source {{
        color: var(--accent);
        font-weight: 700;
    }}

    .diff-card-text {{
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 6px;
        line-height: 1.5;
    }}

    /* AI Chat Banner */
    .chat-banner {{
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 0.88rem;
        color: var(--text-secondary);
    }}

    .chat-banner strong {{
        color: var(--text-primary);
    }}

    /* Trending Topics Cards in Dashboard */
    .trend-card {{
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: border-color 0.2s ease;
    }}

    .trend-card:hover {{
        border-color: var(--border-hover);
    }}

    .trend-card-title {{
        color: var(--text-primary);
        font-size: 0.95rem;
        font-weight: 700;
    }}

    .trend-card-meta {{
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-top: 2px;
    }}

    .trend-card-meta strong {{
        color: var(--text-secondary);
    }}

    /* Analytics Top Keyword Tags */
    .keyword-tag-pill {{
        display: inline-block;
        background-color: var(--surface);
        border: 1px solid var(--border);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.83rem;
        margin: 3px 2px;
        color: var(--text-secondary);
    }}

    /* AI Progress Cards in Dashboard */
    .ai-progress-card {{
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }}

    .ai-progress-label {{
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--text-primary);
    }}

    .ai-progress-meta {{
        font-size: 0.9rem;
        color: var(--text-muted);
    }}

    .ai-progress-track {{
        background-color: var(--border);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin-top: 8px;
    }}

    /* Footer */
    .app-footer {{
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid var(--border);
        text-align: center;
        font-size: 0.85rem;
        color: var(--text-muted);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
