"""Theme CSS and small HTML helpers. Visual output is intentionally unchanged."""

from __future__ import annotations

import streamlit as st

THEME_CSS = """
    <style>
    /* Hide Streamlit default UI */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}

    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Playfair Display', serif !important; letter-spacing: -0.01em; }

    .stApp { background: #fdf0f3; color: #2d1a20; }

    /* Main content top padding */
    .block-container { padding-top: 2rem !important; }

    /* Metric tiles */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
    .metric-tile {
        flex: 1; min-width: 120px;
        background: #fff5f7;
        border: 1px solid #f5b8cb;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(233,100,139,0.07);
    }
    .metric-tile .val {
        font-family: 'Playfair Display', serif;
        font-size: 2rem; font-weight: 800;
        color: #e9648b; line-height: 1;
    }
    .metric-tile .lbl {
        font-size: 0.68rem; color: #b07a8a;
        text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem;
    }

    /* Input form card */
    .form-card {
        background: #fff5f7;
        border: 1px solid #f5b8cb;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 16px rgba(233,100,139,0.08);
        position: relative;
        overflow: hidden;
    }
    .form-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #e9648b, #f48fb1, #ce93d8);
    }

    /* Plan card */
    .plan-card {
        background: #fff5f7;
        border: 1px solid #f5b8cb;
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 16px rgba(233,100,139,0.08);
        position: relative;
        overflow: hidden;
        line-height: 1.9;
        font-size: 0.95rem;
        color: #2d1a20;
    }
    .plan-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #e9648b, #f48fb1, #ce93d8);
    }

    /* Buttons */
    .stButton > button {
        background: #e9648b !important;
        color: #fff !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 2px 10px rgba(233,100,139,0.25) !important;
        transition: background 0.15s !important;
        width: 100%;
    }
    .stButton > button:hover { background: #d4476d !important; }

    /* Inputs */
    .stTextInput input, .stNumberInput input {
        background: #fff5f7 !important;
        border-color: #f5b8cb !important;
        border-radius: 8px !important;
        color: #2d1a20 !important;
        font-size: 0.95rem !important;
    }
    .stTextInput label, .stNumberInput label, .stSlider label {
        color: #2d1a20 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    /* Slider */
    [data-testid="stSlider"] { padding: 0.25rem 0; }

    /* Alerts */
    .stSuccess {
        background: #fce4ec !important;
        border-color: #f48fb1 !important;
        color: #880e4f !important;
        border-radius: 10px !important;
    }
    .stWarning { background: #fff8e1 !important; border-radius: 10px !important; }
    .stError { border-radius: 10px !important; }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Playfair Display', serif !important;
        font-weight: 600 !important;
        color: #2d1a20 !important;
        background: #fff5f7 !important;
        border: 1px solid #f5b8cb !important;
        border-radius: 10px !important;
    }
    .streamlit-expanderContent {
        background: #fff5f7 !important;
        border: 1px solid #f5b8cb !important;
        border-top: none !important;
    }

    hr { border-color: #f5b8cb !important; }

    .stDownloadButton > button {
        background: transparent !important;
        color: #e9648b !important;
        border: 1.5px solid #e9648b !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }
    .stDownloadButton > button:hover { background: #fce4ec !important; }

    .stCaption { color: #b07a8a !important; font-size: 0.75rem !important; }
    .stDataFrame { border: 1px solid #f5b8cb !important; border-radius: 10px !important; }
    p, li { color: #2d1a20; }

    /* Section labels */
    .section-label {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #2d1a20;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
    }
    </style>
    """


def inject_css() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def metric_tile(value: str, label: str) -> str:
    return (
        f'<div class="metric-tile"><div class="val">{value}</div>'
        f'<div class="lbl">{label}</div></div>'
    )
