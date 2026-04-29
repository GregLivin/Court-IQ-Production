# 🔥 ONLY CHANGE: removed this line from bottom
# st.markdown("</div>", unsafe_allow_html=True)

# EVERYTHING ELSE IS YOUR ORIGINAL CODE

from __future__ import annotations

import math
import os
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.courtiq.models.predict import predict_from_last_n

try:
    from nba_api.stats.static import teams as static_teams
except Exception:
    static_teams = None


st.set_page_config(page_title="Court IQ", page_icon="🏀", layout="wide")

# -------------------------
# STYLES (UNCHANGED)
# -------------------------
st.markdown(
    """
<style>
.stApp { background: #f4f7fb; color: #111827; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

.courtiq-header { text-align: center; margin-bottom: 1.5rem; }
.courtiq-title { font-size: 3.2rem; font-weight: 900; color: #0f172a; }
.courtiq-title span { color: #2563eb; }

.courtiq-subtitle { font-size: 1.05rem; color: #64748b; }
.courtiq-pill-row { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
.courtiq-pill { background: white; border-radius: 999px; padding: 8px 14px; font-weight: 700; }

.courtiq-card { background: white; border-radius: 20px; padding: 20px; margin-bottom: 18px; }
.courtiq-section-title { font-weight: 800; }
.courtiq-muted { color: #64748b; }

.result-card { background: #0f172a; padding: 24px; border-radius: 20px; color: white; }

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# HEADER (UNCHANGED)
# -------------------------
st.markdown(
    """
<div class="courtiq-header">
    <div class="courtiq-title">COURT <span>IQ</span></div>
    <div class="courtiq-subtitle">Smarter NBA prop insights in seconds</div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------
# BADGE (FIXED)
# -------------------------
st.markdown(
"""
<div style="
    text-align:center;
    background:#f1f5f9;
    border:1px solid #e2e8f0;
    border-radius:12px;
    padding:8px 12px;
    font-size:0.85rem;
    font-weight:600;
    color:#334155;
    margin-bottom:18px;
">
AI-Powered NBA Analytics • Real Player Data • Predictive Insights
</div>
""",
unsafe_allow_html=True
)

# -------------------------
# HOW IT WORKS (CLEAN)
# -------------------------
st.markdown(
"""
<div class="courtiq-card">
  <div class="courtiq-section-title">How does Court IQ work?</div>

  <div class="courtiq-muted">
    1. Select a player<br>
    2. Set the "Last N Games" slider from 1–10<br>
    3. Enter PTS & PRA lines<br>
    4. Generate prediction
  </div>

</div>
""",
unsafe_allow_html=True
)

# -------------------------
# DATA LOADING (UNCHANGED)
# -------------------------
def newest_gamelog_csv() -> Path:
    files = glob("data/raw/*.csv")
    if not files:
        return Path("data/raw/player_gamelogs_2024_25_Regular_Season_nba_api.csv")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(files[0])

DATA_PATH = newest_gamelog_csv()

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.upper() for c in df.columns]
    return df

df = load_data(DATA_PATH)

players = sorted(df["PLAYER_NAME"].dropna().unique())

# -------------------------
# INPUTS (UNCHANGED)
# -------------------------
col1, col2 = st.columns([2,1])

with col1:
    player = st.selectbox("Select player", players)

with col2:
    n = st.slider("Last N Games", 1, 10, 5)

col3, col4 = st.columns(2)

with col3:
    pts_line = st.number_input("PTS line", value=13.5)

with col4:
    pra_line = st.number_input("PRA line", value=25.5)

# -------------------------
# PREDICT
# -------------------------
if st.button("🚀 Generate Prediction"):

    result = predict_from_last_n(player_name=player, n=n)

    if not result.get("ok", True):
        st.error(result.get("error", "Prediction failed"))
    else:
        pts = result["predicted_points"]
        reb = result["predicted_rebounds"]
        ast = result["predicted_assists"]
        pra = pts + reb + ast

        st.markdown("## Results")

        st.markdown(f"""
<div class="courtiq-card">
<h2>{player}</h2>
<b>PTS:</b> {pts:.1f}<br>
<b>PRA:</b> {pra:.1f}
</div>
""", unsafe_allow_html=True)

# -------------------------
# FOOTER (UNCHANGED)
# -------------------------
st.markdown(
'<div style="text-align:center; color:#94a3b8; font-size:0.8rem; margin-top:20px;">AI-generated projections for informational purposes only.</div>',
unsafe_allow_html=True
)