"""
Theme specification, color tokens, and UI helpers for NLP Visual Lab.
Enforces light theme aesthetics as defined in PRD Section 4.
"""
import html

# --- Core Color Palette ---
COLOR_BG = "#FFFFFF"               # Main page background
COLOR_SECONDARY_BG = "#F5F7FB"     # Cards, containers, tabs
COLOR_PRIMARY = "#4F46E5"          # Primary indigo accent
COLOR_TEXT = "#1F2937"             # Deep charcoal body text
COLOR_MUTED = "#6B7280"            # Muted captions & secondary text
COLOR_BORDER = "#E5E7EB"           # Card and divider outlines

# --- Pastel Highlight Colors (for tokens, POS tags, and entities) ---
PASTEL_BLUE = "#DBEAFE"
PASTEL_GREEN = "#DCFCE7"
PASTEL_YELLOW = "#FEF9C3"
PASTEL_PINK = "#FCE7F3"
PASTEL_PURPLE = "#EDE9FE"
PASTEL_ORANGE = "#FFEDD5"

PASTEL_COLORS = [
    PASTEL_BLUE,
    PASTEL_GREEN,
    PASTEL_YELLOW,
    PASTEL_PINK,
    PASTEL_PURPLE,
    PASTEL_ORANGE,
]

# POS to Pastel Color mapping
POS_COLOR_MAP = {
    "NOUN": PASTEL_BLUE,
    "PROPN": PASTEL_BLUE,
    "VERB": PASTEL_GREEN,
    "AUX": PASTEL_GREEN,
    "ADJ": PASTEL_YELLOW,
    "ADV": PASTEL_ORANGE,
    "PRON": PASTEL_PURPLE,
    "DET": PASTEL_PINK,
    "ADP": PASTEL_PINK,
    "CCONJ": PASTEL_PINK,
    "SCONJ": PASTEL_PINK,
    "PUNCT": "#F3F4F6",
    "NUM": PASTEL_ORANGE,
}

# NER Entity Label to Pastel Color mapping
ENTITY_COLOR_MAP = {
    "PERSON": PASTEL_BLUE,
    "ORG": PASTEL_PURPLE,
    "GPE": PASTEL_GREEN,
    "LOC": PASTEL_GREEN,
    "DATE": PASTEL_YELLOW,
    "TIME": PASTEL_YELLOW,
    "MONEY": PASTEL_ORANGE,
    "PERCENT": PASTEL_ORANGE,
    "CARDINAL": PASTEL_PINK,
    "ORDINAL": PASTEL_PINK,
    "EVENT": PASTEL_PURPLE,
    "WORK_OF_ART": PASTEL_BLUE,
}

# --- Custom Global CSS for Streamlit ---
CUSTOM_CSS = f"""
<style>
    /* Global light theme container aesthetics */
    .stApp {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    
    /* Sleek card container */
    .nlp-card {{
        background-color: {COLOR_SECONDARY_BG};
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}
    
    .nlp-card-title {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {COLOR_TEXT};
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .nlp-card-desc {{
        font-size: 0.92rem;
        color: {COLOR_MUTED};
        line-height: 1.5;
        margin-bottom: 12px;
    }}
    
    /* Chip flex container */
    .chip-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin-top: 10px;
        margin-bottom: 14px;
        line-height: 1.8;
    }}
    
    /* Interactive pastel chip */
    .nlp-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.95rem;
        color: {COLOR_TEXT} !important;
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    
    .nlp-chip:hover {{
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    }}
    
    .nlp-chip-sub {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.75;
        padding: 1px 4px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.6);
        margin-left: 2px;
    }}
    
    /* Streamlit tabs refinement */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {COLOR_SECONDARY_BG};
        padding: 6px;
        border-radius: 10px;
        border: 1px solid {COLOR_BORDER};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 6px;
        padding: 8px 16px;
        color: {COLOR_TEXT};
        font-weight: 500;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLOR_BG} !important;
        color: {COLOR_PRIMARY} !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }}
</style>
"""


def chip(text: str, color: str = PASTEL_BLUE, subtitle: str = None) -> str:
    """
    Renders a colored word chip with guaranteed dark text (#1F2937) for readability.
    
    Args:
        text: The token or word text to display.
        color: Hex color string for the chip background (defaults to pastel blue).
        subtitle: Optional sub-label (e.g. POS tag or Entity label).
    
    Returns:
        HTML string representing the chip.
    """
    safe_text = html.escape(str(text))
    sub_html = ""
    if subtitle:
        safe_sub = html.escape(str(subtitle))
        sub_html = f'<span class="nlp-chip-sub">{safe_sub}</span>'
    
    return (
        f'<span class="nlp-chip" style="background-color: {color}; color: {COLOR_TEXT};">'
        f'{safe_text}{sub_html}'
        f'</span>'
    )


def card(title: str, description: str = "", content_html: str = "") -> str:
    """
    Generates a clean, rounded, light-themed card wrapper.
    """
    safe_title = html.escape(title)
    desc_html = f'<div class="nlp-card-desc">{html.escape(description)}</div>' if description else ""
    return (
        f'<div class="nlp-card">'
        f'<div class="nlp-card-title">{safe_title}</div>'
        f'{desc_html}'
        f'{content_html}'
        f'</div>'
    )
