"""
UnderScore — Sports Analytics | Streamlit App
Déploiement du modèle XGBoost de prédiction de matchs PSL.
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy.stats import poisson

# ─── Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="UnderScore — Sports Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_pkl")

# ─── Custom CSS ───────────────────────────────────────────────
STADIUM_BG = "https://images.unsplash.com/photo-1522778119026-d647f0596c20?w=1920&q=80"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="st-"] {{ font-family: 'Inter', sans-serif; }}

    /* ── Stadium Background ── */
    .stApp {{
        background: linear-gradient(180deg, rgba(15,17,23,0.92) 0%, rgba(15,17,23,0.85) 50%, rgba(15,17,23,0.95) 100%),
                    url('{STADIUM_BG}') center/cover no-repeat fixed;
    }}

    /* ── Floating Football Particles ── */
    .football-particles {{
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none; z-index: 0; overflow: hidden;
    }}
    .football-particles span {{
        position: absolute; font-size: 1.5rem; opacity: 0.06;
        animation: floatUp linear infinite;
    }}
    .football-particles span:nth-child(1) {{ left: 5%;  animation-duration: 18s; animation-delay: 0s; }}
    .football-particles span:nth-child(2) {{ left: 15%; animation-duration: 22s; animation-delay: 3s; font-size: 2rem; }}
    .football-particles span:nth-child(3) {{ left: 30%; animation-duration: 16s; animation-delay: 7s; }}
    .football-particles span:nth-child(4) {{ left: 50%; animation-duration: 25s; animation-delay: 2s; font-size: 1.2rem; }}
    .football-particles span:nth-child(5) {{ left: 65%; animation-duration: 20s; animation-delay: 5s; }}
    .football-particles span:nth-child(6) {{ left: 80%; animation-duration: 19s; animation-delay: 8s; font-size: 1.8rem; }}
    .football-particles span:nth-child(7) {{ left: 90%; animation-duration: 23s; animation-delay: 1s; }}
    .football-particles span:nth-child(8) {{ left: 42%; animation-duration: 17s; animation-delay: 6s; font-size: 2.2rem; }}
    @keyframes floatUp {{
        0%   {{ transform: translateY(110vh) rotate(0deg); opacity: 0; }}
        10%  {{ opacity: 0.07; }}
        90%  {{ opacity: 0.05; }}
        100% {{ transform: translateY(-10vh) rotate(720deg); opacity: 0; }}
    }}

    /* ── Hide Sidebar completely ── */
    [data-testid="stSidebar"] {{ display: none; }}
    button[kind="header"] {{ display: none; }}
    [data-testid="collapsedControl"] {{ display: none; }}

    /* ── Top Navigation Bar ── */
    .topnav {{
        position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
        background: rgba(13,17,23,0.92);
        backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(59,130,246,0.12);
        padding: 0 2rem;
        display: flex; align-items: center; justify-content: space-between;
        height: 64px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }}
    .topnav-logo {{
        font-size: 1.6rem; font-weight: 800; letter-spacing: -1px;
        display: flex; align-items: center; gap: 0; flex-shrink: 0;
    }}
    .topnav-logo .logo-white {{ color: #fff; }}
    .topnav-logo .logo-blue  {{ color: #3b82f6; text-shadow: 0 0 20px rgba(59,130,246,0.4); }}
    .topnav-links {{
        display: flex; align-items: center; gap: 0.25rem;
    }}
    .topnav-links a {{
        color: #8892a4; text-decoration: none;
        font-size: 0.88rem; font-weight: 600;
        padding: 0.5rem 1rem; border-radius: 8px;
        transition: all 0.25s ease;
        white-space: nowrap;
    }}
    .topnav-links a:hover {{
        color: #e8eaf0; background: rgba(59,130,246,0.1);
    }}
    .topnav-links a.active {{
        color: #3b82f6; background: rgba(59,130,246,0.15);
        box-shadow: 0 0 12px rgba(59,130,246,0.1);
    }}
    .topnav-accent {{
        position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, #3b82f6, #8b5cf6, #06b6d4, transparent);
    }}
    /* Push content below fixed topbar */
    .main .block-container {{ padding-top: 5rem !important; }}

    /* ── Desktop nav links in topbar ── */
    .topnav-links {{
        display: flex; align-items: center; gap: 0.25rem;
    }}
    .topnav-links a {{
        color: #8892a4; text-decoration: none;
        font-size: 0.85rem; font-weight: 600;
        padding: 0.5rem 1rem; border-radius: 8px;
        transition: all 0.25s ease; white-space: nowrap;
        display: flex; align-items: center; gap: 0.4rem;
    }}
    .topnav-links a:hover {{ color: #e8eaf0; background: rgba(59,130,246,0.1); }}
    .topnav-links a.active {{
        color: #3b82f6; background: rgba(59,130,246,0.15);
        box-shadow: 0 0 12px rgba(59,130,246,0.08);
    }}
    .topnav-links svg {{ width: 16px; height: 16px; }}

    /* ── Mobile: hide topbar, add bottom padding ── */
    @media (max-width: 768px) {{
        .topnav {{ display: none !important; }}
        .main .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 5.5rem !important;
        }}
    }}

    /* ── Glassmorphism Cards ── */
    .metric-card {{
        background: rgba(26,29,39,0.75);
        backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(59,130,246,0.12);
        border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }}
    .metric-card:hover {{
        border-color: rgba(59,130,246,0.35);
        box-shadow: 0 8px 32px rgba(59,130,246,0.1);
        transform: translateY(-2px);
    }}
    .metric-card-label {{
        font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; color: #8892a4; margin-bottom: 0.5rem;
    }}
    .metric-card-value {{ font-size: 1.5rem; font-weight: 700; color: #fff; }}
    .metric-card-value.green {{ color: #10b981; text-shadow: 0 0 12px rgba(16,185,129,0.3); }}
    .metric-card-sub {{ font-size: 0.8rem; color: #8892a4; margin-top: 0.35rem; }}

    /* ── Result Box ── */
    .result-box {{
        background: linear-gradient(135deg, rgba(30,58,95,0.9), rgba(26,45,74,0.9));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 16px; padding: 2rem; text-align: center; margin: 1rem 0;
        box-shadow: 0 0 30px rgba(59,130,246,0.08);
    }}
    .result-label {{
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.07em; color: #93c5fd; margin-bottom: 0.5rem;
    }}
    .result-value {{
        font-size: 2rem; font-weight: 800; color: #fff;
        text-shadow: 0 0 20px rgba(59,130,246,0.3);
    }}

    /* ── Premium Box ── */
    .premium-box {{
        background: linear-gradient(135deg, rgba(30,58,95,0.9), rgba(26,45,74,0.9));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
        box-shadow: 0 0 30px rgba(59,130,246,0.08);
    }}
    .premium-label {{
        font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: #93c5fd; margin-bottom: 0.5rem;
    }}
    .premium-value {{
        font-size: 2.4rem; font-weight: 800; color: #fff;
        text-shadow: 0 0 20px rgba(59,130,246,0.3);
    }}
    .premium-sub {{ font-size: 0.8rem; color: #93c5fd; margin-top: 0.2rem; }}

    /* ── Interpretation ── */
    .interpretation-box {{
        background: rgba(59,130,246,0.05);
        backdrop-filter: blur(8px);
        border: 1px dashed rgba(59,130,246,0.3);
        padding: 1.25rem; border-radius: 12px; margin-top: 1rem;
    }}
    .interpretation-title {{
        font-size: 0.85rem; margin-bottom: 0.75rem; color: #3b82f6;
        font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .interpretation-text {{ font-size: 0.88rem; color: #8892a4; line-height: 1.6; }}

    /* ── Badge ── */
    .badge-red {{
        background: rgba(239,68,68,0.15); color: #f87171;
        border: 1px solid rgba(239,68,68,0.3);
        padding: 0.25rem 0.75rem; border-radius: 999px;
        font-weight: 700; font-size: 0.9rem; display: inline-block;
    }}

    /* ── Profit Row ── */
    .profit-row {{
        background: rgba(16,185,129,0.08); padding: 0.75rem;
        border-radius: 8px; border: 1px solid rgba(16,185,129,0.2);
    }}
    .profit-value {{ color: #10b981; font-weight: 700; font-size: 1.1rem; }}

    /* ── Hide Streamlit Chrome ── */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ── Table ── */
    .stDataFrame {{ border-radius: 12px; overflow: hidden; }}

    /* ── File uploader fix ── */
    [data-testid="stFileUploader"] {{ overflow: hidden; }}
    [data-testid="stFileUploader"] section {{ padding: 0; }}
    [data-testid="stFileUploader"] button {{ font-size: 0.82rem; }}

    /* ── Divider ── */
    .section-divider {{ height: 1px; background: linear-gradient(90deg, transparent, #2a2f45, transparent); margin: 1.5rem 0; }}

    /* ── Streamlit elements polish ── */
    .stSelectbox > div > div {{ background: rgba(26,29,39,0.8); border-color: #2a2f45; }}
    .stNumberInput > div > div > input {{ background: rgba(26,29,39,0.8); }}
    button[kind="primary"] {{
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
        transition: all 0.3s ease !important;
    }}
    button[kind="primary"]:hover {{
        box-shadow: 0 6px 25px rgba(59,130,246,0.5) !important;
        transform: translateY(-1px) !important;
    }}

    /* ── Mobile responsive content ── */
    @media (max-width: 768px) {{
        /* Stack Streamlit columns vertically */
        [data-testid="stHorizontalBlock"] {{
            flex-direction: column !important;
        }}
        [data-testid="stHorizontalBlock"] > div {{
            width: 100% !important; flex: 1 1 100% !important;
        }}
        /* Smaller text and padding */
        .metric-card {{ padding: 1rem; border-radius: 12px; }}
        .metric-card-value {{ font-size: 1.2rem; }}
        .result-value {{ font-size: 1.4rem; }}
        .premium-value {{ font-size: 1.6rem; }}
        .premium-box, .result-box {{ padding: 1rem; border-radius: 12px; }}
        .interpretation-box {{ padding: 1rem; }}
        .interpretation-text {{ font-size: 0.82rem; }}
        /* Table horizontal scroll */
        .stDataFrame {{ overflow-x: auto; }}
        /* General padding */
        .main .block-container {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
    }}
</style>

<!-- Floating football particles -->
<div class="football-particles">
    <span>⚽</span><span>⚽</span><span>⚽</span><span>⚽</span>
    <span>⚽</span><span>⚽</span><span>⚽</span><span>⚽</span>
</div>
""", unsafe_allow_html=True)


# ─── Model Loading (cached) ──────────────────────────────────
@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "calibrated_model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_features():
    with open(os.path.join(MODEL_DIR, "features_list.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_elo():
    with open(os.path.join(MODEL_DIR, "latest_elo.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_rolling_stats():
    with open(os.path.join(MODEL_DIR, "latest_rolling_stats.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_teams():
    with open(os.path.join(MODEL_DIR, "teams.pkl"), "rb") as f:
        return sorted(pickle.load(f))

@st.cache_data
def load_shap():
    path = os.path.join(MODEL_DIR, "shap_data.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


# ─── Prediction Logic ────────────────────────────────────────
def predict_match(home_team: str, away_team: str) -> dict:
    """Reproduce ml_pipeline.predict_single_match with .pkl files."""
    model = load_model()
    features_list = load_features()
    elo_dict = load_elo()
    rolling_stats = load_rolling_stats()

    home_elo = elo_dict.get(home_team, 1500)
    away_elo = elo_dict.get(away_team, 1500)

    h_stats = rolling_stats.get(home_team, {})
    a_stats = rolling_stats.get(away_team, {})

    row = {}
    row["Home_Elo"] = home_elo
    row["Away_Elo"] = away_elo
    row["Elo_Difference"] = home_elo - away_elo

    for k, v in h_stats.items():
        row[f"Home_{k}"] = v
    for k, v in a_stats.items():
        row[f"Away_{k}"] = v

    # Poisson probabilities
    h_proj = (row.get("Home_Roll_xG_For", 1) + row.get("Away_Roll_xG_Against", 1)) / 2
    a_proj = (row.get("Away_Roll_xG_For", 1) + row.get("Home_Roll_xG_Against", 1)) / 2

    def poisson_probs(h_lambda, a_lambda):
        h_probs = [poisson.pmf(i, max(0.1, h_lambda)) for i in range(7)]
        a_probs = [poisson.pmf(i, max(0.1, a_lambda)) for i in range(7)]
        matrix = np.outer(h_probs, a_probs)
        return np.sum(np.tril(matrix, -1)), np.sum(np.diag(matrix)), np.sum(np.triu(matrix, 1))

    h_win, draw, a_win = poisson_probs(h_proj, a_proj)
    row["Poisson_HomeWin"] = h_win
    row["Poisson_Draw"] = draw
    row["Poisson_AwayWin"] = a_win

    input_vector = [row.get(f, 0) for f in features_list]
    X_pred = pd.DataFrame([input_vector], columns=features_list)
    probs = model.predict_proba(X_pred)[0]

    classes = ["Away Win", "Draw", "Home Win"]
    pred_idx = int(np.argmax(probs))

    return {
        "home_win_prob": round(probs[2] * 100, 2),
        "draw_prob": round(probs[1] * 100, 2),
        "away_win_prob": round(probs[0] * 100, 2),
        "prediction": classes[pred_idx],
    }


def fmt(n):
    """Format as South African Rand."""
    return f"R {round(n):,}".replace(",", " ")


# ─── Plotly chart helper ──────────────────────────────────────
def probability_chart(home_prob, draw_prob, away_prob):
    colors = ["#3b82f6", "#8b5cf6", "#ef4444"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Home Win", "Draw", "Away Win"],
                y=[home_prob, draw_prob, away_prob],
                marker_color=colors,
                marker_line_width=0,
                text=[f"{v}%" for v in [home_prob, draw_prob, away_prob]],
                textposition="outside",
                textfont=dict(color="#e8eaf0", size=14, family="Inter"),
            )
        ]
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892a4", family="Inter"),
        xaxis=dict(showgrid=False, tickfont=dict(size=13)),
        yaxis=dict(visible=False, range=[0, max(home_prob, draw_prob, away_prob) * 1.25]),
        margin=dict(l=10, r=10, t=20, b=40),
        height=280,
        showlegend=False,
        bargap=0.35,
    )
    return fig


# ─── Navigation ───────────────────────────────────────────────
PAGE_KEYS  = ["dashboard", "predict", "insurance", "batch"]
PAGE_NAMES = ["Dashboard", "Match Predictor", "Insurance Pricing", "Batch Predict"]
# SVG paths for clean icons (no emojis)
PAGE_SVGS = [
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>',
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L4 6v6c0 5.5 3.5 10.7 8 12 4.5-1.3 8-6.5 8-12V6z"/></svg>',
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>',
]

# Read current page from query params
qp = st.query_params
current_key = qp.get("p", "dashboard")
if current_key not in PAGE_KEYS:
    current_key = "dashboard"
current_idx = PAGE_KEYS.index(current_key)

# ─── Desktop: topbar with logo + nav links ───
def _desktop_link(idx):
    active = ' active' if idx == current_idx else ''
    return (f'<a href="?p={PAGE_KEYS[idx]}" class="{active}" '
            f'target="_self">{PAGE_SVGS[idx]} {PAGE_NAMES[idx]}</a>')

st.markdown(
    '<div class="topnav">'
    '  <div class="topnav-logo"><span class="logo-white">Under</span><span class="logo-blue">Score</span></div>'
    '  <div class="topnav-links">' + ''.join(_desktop_link(i) for i in range(len(PAGE_KEYS))) + '</div>'
    '  <div class="topnav-accent"></div>'
    '</div>',
    unsafe_allow_html=True,
)

# ─── Mobile: bottom bar via components.html (real JS, clickable) ───
def _mob_item_html(idx):
    active = ' active' if idx == current_idx else ''
    return (f'<div class="item{active}" onclick="go(\'{PAGE_KEYS[idx]}\')">' 
            f'<div class="icon">{PAGE_SVGS[idx]}</div>'
            f'<div class="label">{PAGE_NAMES[idx].split()[0]}</div></div>')

mobile_html = f"""
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background: transparent; }}
  .bar {{
    position:fixed; bottom:0; left:0; right:0;
    background: rgba(13,17,23,0.97);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border-top: 1px solid rgba(59,130,246,0.12);
    box-shadow: 0 -4px 30px rgba(0,0,0,0.5);
    display:flex; justify-content:space-around; align-items:center;
    padding: 6px 4px 10px 4px;
    font-family: 'Inter', -apple-system, sans-serif;
  }}
  .item {{
    display:flex; flex-direction:column; align-items:center;
    gap:2px; padding:6px 8px; border-radius:10px;
    color:#6b7280; font-size:10px; font-weight:600;
    cursor:pointer; transition: all 0.2s; flex:1;
    text-align:center;
  }}
  .item:active {{ transform: scale(0.92); }}
  .item.active {{ color:#3b82f6; background:rgba(59,130,246,0.1); }}
  .item .icon {{ width:22px; height:22px; }}
  .item .icon svg {{ width:100%; height:100%; }}
  .item.active .icon svg {{ filter: drop-shadow(0 0 4px rgba(59,130,246,0.5)); }}
  .item .label {{ margin-top:1px; }}
  @media (min-width: 769px) {{ .bar {{ display:none; }} }}
</style>
<div class="bar">
  {''.join(_mob_item_html(i) for i in range(len(PAGE_KEYS)))}
</div>
<script>
  function go(pageKey) {{
    window.parent.location.href = window.parent.location.pathname + '?p=' + pageKey;
  }}
</script>
"""
components.html(mobile_html, height=0)

# Map to page display name
page = PAGE_NAMES[current_idx]


# ═══════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown("## Overview")
    st.markdown('<p style="color:#8892a4; margin-top:-0.5rem;">Status of the AI prediction engine.</p>', unsafe_allow_html=True)

    # Status cards
    col1, col2, col3 = st.columns(3)

    model_exists = os.path.exists(os.path.join(MODEL_DIR, "calibrated_model.pkl"))

    with col1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-card-label">Pipeline Status</div>'
            f'<div class="metric-card-value green">{"● Active" if model_exists else "● Inactive"}</div>'
            f'<div class="metric-card-sub">{"Ready for predictions" if model_exists else "Model not found"}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-card-label">Model</div>'
            '<div class="metric-card-value">XGBoost</div>'
            '<div class="metric-card-sub">Calibrated + SHAP</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col3:
        teams = load_teams() if model_exists else []
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-card-label">Teams</div>'
            f'<div class="metric-card-value">{len(teams)}</div>'
            '<div class="metric-card-sub">PSL 2024/25 & 2025/26</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ELO Rankings & SHAP side by side
    col_elo, col_shap = st.columns([3, 2])

    with col_elo:
        st.markdown("### 🏆 ELO Rankings")
        if model_exists:
            elo = load_elo()
            elo_sorted = sorted(elo.items(), key=lambda x: -x[1])
            elo_df = pd.DataFrame(elo_sorted, columns=["Team", "ELO Rating"])
            elo_df.index = range(1, len(elo_df) + 1)
            elo_df.index.name = "Rank"
            elo_df["ELO Rating"] = elo_df["ELO Rating"].round(1)
            st.dataframe(elo_df, use_container_width=True, height=500)
        else:
            st.warning("Model not trained yet.")

    with col_shap:
        st.markdown("### 🔍 Feature Importance (SHAP)")
        shap_data = load_shap()
        if shap_data:
            features = list(shap_data.keys())
            values = list(shap_data.values())
            fig = go.Figure(
                go.Bar(
                    x=values,
                    y=features,
                    orientation="h",
                    marker_color="#3b82f6",
                    marker_line_width=0,
                )
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8892a4", family="Inter"),
                xaxis=dict(showgrid=False, title="Mean |SHAP value|"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=10, b=40),
                height=250,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SHAP data available.")


# ═══════════════════════════════════════════════════════════════
# PAGE 2: MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════
elif page == "Match Predictor":
    st.markdown("## ⚡ Match Predictor")
    st.markdown(
        '<p style="color:#8892a4; margin-top:-0.5rem;">Select two teams to get AI-driven win probabilities.</p>',
        unsafe_allow_html=True,
    )

    teams = load_teams()

    col_form, col_result = st.columns(2)

    with col_form:
        home_team = st.selectbox("Home Team", [""] + teams, index=0, key="pred_home")
        away_team = st.selectbox("Away Team", [""] + teams, index=0, key="pred_away")
        predict_btn = st.button("Generate Prediction", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            if not home_team or not away_team:
                st.error("Please select both teams.")
            elif home_team == away_team:
                st.error("Home and Away teams must be different.")
            else:
                with st.spinner("Calculating probabilities..."):
                    try:
                        result = predict_match(home_team, away_team)

                        st.markdown(
                            '<div class="result-box">'
                            '<div class="result-label">Most Likely Outcome</div>'
                            f'<div class="result-value">{result["prediction"]}</div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                        fig = probability_chart(
                            result["home_win_prob"],
                            result["draw_prob"],
                            result["away_win_prob"],
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Details
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Home Win", f'{result["home_win_prob"]}%')
                        m2.metric("Draw", f'{result["draw_prob"]}%')
                        m3.metric("Away Win", f'{result["away_win_prob"]}%')
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
        else:
            st.markdown(
                '<div style="text-align:center; padding:3rem; color:#8892a4;">'
                '<div style="font-size:2.5rem; opacity:0.4; margin-bottom:0.75rem;">⚡</div>'
                '<p>Run a prediction to see probabilities.</p>'
                '</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════
# PAGE 3: INSURANCE PRICING
# ═══════════════════════════════════════════════════════════════
elif page == "Insurance Pricing":
    st.markdown("## 🛡 Prize Indemnity Underwriting")
    st.markdown(
        '<p style="color:#8892a4; margin-top:-0.5rem;">Turn match probabilities into actuarially priced insurance products.</p>',
        unsafe_allow_html=True,
    )

    teams = load_teams()

    col_params, col_result = st.columns([2, 3])

    with col_params:
        st.markdown("#### Scenario Parameters")
        prize = st.number_input("Corporate Prize Amount (R)", min_value=1000, value=5000000, step=100000)
        ins_home = st.selectbox("Home Team", [""] + teams, index=0, key="ins_home")
        ins_away = st.selectbox("Away Team", [""] + teams, index=0, key="ins_away")
        condition = st.selectbox(
            "Payout Condition",
            ["home", "draw", "away"],
            format_func=lambda x: {"home": "Fan predicts Home Win", "draw": "Fan predicts Draw", "away": "Fan predicts Away Win"}[x],
        )
        margin = st.number_input("Insurer Profit Margin (%)", min_value=0, max_value=1000, value=30)
        calc_btn = st.button("Calculate Premium", type="primary", use_container_width=True)

    with col_result:
        if calc_btn:
            if not ins_home or not ins_away:
                st.error("Please select both teams.")
            elif ins_home == ins_away:
                st.error("Home and Away teams must be different.")
            else:
                with st.spinner("Calculating premium..."):
                    try:
                        result = predict_match(ins_home, ins_away)
                        prob = (
                            result["home_win_prob"] if condition == "home"
                            else result["draw_prob"] if condition == "draw"
                            else result["away_win_prob"]
                        )
                        prob_dec = prob / 100
                        expected_loss = prob_dec * prize
                        premium = expected_loss * (1 + margin / 100)
                        profit = premium - expected_loss

                        # Premium header
                        st.markdown(
                            '<div class="premium-box">'
                            '<div class="premium-label">Underwriting Recommendation</div>'
                            f'<div class="premium-value">{fmt(premium)}</div>'
                            '<div class="premium-sub">Upfront Premium Required</div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                        # Detail rows
                        r1, r2 = st.columns(2)
                        r1.markdown(f"**Event Probability**")
                        r2.markdown(f'<span class="badge-red">{prob:.2f}%</span>', unsafe_allow_html=True)

                        r1, r2 = st.columns(2)
                        r1.markdown("**Total Liability (Prize)**")
                        r2.markdown(f"{fmt(prize)}")

                        r1, r2 = st.columns(2)
                        r1.markdown("**Expected Loss (Base Cost)**")
                        r2.markdown(f"**{fmt(expected_loss)}**")

                        r1, r2 = st.columns(2)
                        r1.markdown("**Profit Margin**")
                        r2.markdown(f"{margin}%")

                        st.markdown(
                            f'<div class="profit-row" style="display:flex; justify-content:space-between; align-items:center; margin-top:0.5rem;">'
                            f'<span><strong>Projected Long-Term Profit</strong></span>'
                            f'<span class="profit-value">+{fmt(profit)}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Interpretation
                        condition_label = {"home": "Home Win", "draw": "Draw", "away": "Away Win"}[condition]
                        st.markdown(
                            f'<div class="interpretation-box">'
                            f'<div class="interpretation-title">Underwriter\'s Interpretation</div>'
                            f'<div class="interpretation-text">'
                            f'The XGBoost algorithm determined a <strong style="color:#e8eaf0">{prob:.2f}%</strong> probability for {condition_label}. '
                            f'To cover a risk of <strong style="color:#e8eaf0">{fmt(prize)}</strong>, the actuarial "pure" cost is '
                            f'<strong style="color:#e8eaf0">{fmt(expected_loss)}</strong>. '
                            f'By applying a <strong style="color:#e8eaf0">{margin}%</strong> margin, the premium of '
                            f'<strong style="color:#e8eaf0">{fmt(premium)}</strong> generates a statistical profit of '
                            f'<strong style="color:#e8eaf0">+{fmt(profit)}</strong>. '
                            f'This pricing secures the financial risk by relying on objective historical data rather than sporting intuition.'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.error(f"Calculation failed: {e}")
        else:
            st.markdown(
                '<div style="text-align:center; padding:3rem; color:#8892a4;">'
                '<div style="font-size:2.5rem; opacity:0.4; margin-bottom:0.75rem;">🛡</div>'
                '<p>Configure scenario and calculate premium.</p>'
                '</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════
# PAGE 4: BATCH PREDICT
# ═══════════════════════════════════════════════════════════════
elif page == "Batch Predict":
    st.markdown("## 📁 Batch Predictor")
    st.markdown(
        '<p style="color:#8892a4; margin-top:-0.5rem;">Upload a CSV with <code>HomeTeam</code> and <code>AwayTeam</code> columns.</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
            st.error("CSV must contain **HomeTeam** and **AwayTeam** columns.")
        else:
            with st.spinner(f"Predicting {len(df)} matches..."):
                results = []
                progress = st.progress(0)
                for i, (_, row) in enumerate(df.iterrows()):
                    try:
                        pred = predict_match(row["HomeTeam"], row["AwayTeam"])
                        results.append({
                            "Home Team": row["HomeTeam"],
                            "Away Team": row["AwayTeam"],
                            "Home Win %": pred["home_win_prob"],
                            "Draw %": pred["draw_prob"],
                            "Away Win %": pred["away_win_prob"],
                            "Prediction": pred["prediction"],
                        })
                    except Exception:
                        results.append({
                            "Home Team": row["HomeTeam"],
                            "Away Team": row["AwayTeam"],
                            "Home Win %": "—",
                            "Draw %": "—",
                            "Away Win %": "—",
                            "Prediction": "❌ Error",
                        })
                    progress.progress((i + 1) / len(df))

                progress.empty()
                res_df = pd.DataFrame(results)

                st.markdown(f"### Results — {len(res_df)} matches")
                st.dataframe(
                    res_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Home Win %": st.column_config.NumberColumn(format="%.2f%%"),
                        "Draw %": st.column_config.NumberColumn(format="%.2f%%"),
                        "Away Win %": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                )

                # Download button
                csv = res_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="underscore_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.markdown(
            '<div style="text-align:center; padding:3rem; color:#8892a4; border:2px dashed #2a2f45; border-radius:12px;">'
            '<div style="font-size:2.5rem; opacity:0.4; margin-bottom:0.75rem;">📂</div>'
            '<p>Drag and drop or click to select CSV</p>'
            '</div>',
            unsafe_allow_html=True,
        )
