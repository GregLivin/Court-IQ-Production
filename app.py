from __future__ import annotations
import math
import os
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ✅ KEEP YOUR REAL MODEL
from src.courtiq.models.predict import predict_from_last_n


# -------------------------
# PAGE SETUP
# -------------------------
st.set_page_config(page_title="Court IQ", page_icon="🏀", layout="wide")


# -------------------------
# UI STYLES
# -------------------------
st.markdown(
    """
<style>
.courtiq-hero {
    background:#f8fafc;
    padding:24px;
    border-radius:18px;
    border:1px solid #e2e8f0;
    margin-bottom:20px;
}
.courtiq-card {
    background:#ffffff;
    padding:20px;
    border-radius:18px;
    border:1px solid #e2e8f0;
    margin-bottom:20px;
}
.courtiq-section-title {
    font-size:1.4rem;
    font-weight:800;
    margin-bottom:10px;
}
.courtiq-muted {
    color:#475569;
    font-size:0.95rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------
# HERO
# -------------------------
st.markdown(
    """
<div class="courtiq-hero">
  <div class="courtiq-section-title">Smarter NBA prop insights in seconds</div>

  <div class="courtiq-muted" style="margin-bottom:10px;">
    Transform raw player data into actionable insights using AI-powered projections,
    matchup intelligence, and probability modeling.
  </div>

  <div class="courtiq-muted">
    Updated with the latest NBA player performance data
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Clean badge
st.markdown(
    """
<div style="
    background:#ecfdf5;
    border:1px solid #bbf7d0;
    color:#166534;
    font-weight:800;
    padding:12px 16px;
    border-radius:14px;
    margin-bottom:18px;
">
  Live AI-powered NBA player prediction system
</div>
""",
    unsafe_allow_html=True,
)


# -------------------------
# HOW IT WORKS (FIXED HTML)
# -------------------------
st.markdown(
    """
<div class="courtiq-card">
  <div class="courtiq-section-title">How does Court IQ work?</div>

  <div class="courtiq-muted">
    Court IQ uses machine learning models trained on real NBA player data 
    to generate intelligent projections and performance insights.

    <br/><br/>

    <b>What the model analyzes:</b>
    <ul>
      <li>Recent player performance trends</li>
      <li>Matchup-specific historical data vs opponents</li>
      <li>Player consistency and statistical variance</li>
    </ul>

    <br/>

    <b>What you get:</b>
    <ul>
      <li>Projected player stats (PTS, REB, AST, PRA)</li>
      <li>Over/Under probability estimates</li>
      <li>Confidence score based on consistency</li>
    </ul>

    <br/>

    <b>Demo Flow:</b><br/>
    Select a player → choose recent games → enter prop lines → generate prediction
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# -------------------------
# LOAD DATA
# -------------------------
DATA_PATH = Path("data/raw/player_gamelogs_2025_26_Regular_Season_nba_api.csv")

if not DATA_PATH.exists():
    st.error("Dataset not found.")
    st.stop()

df = pd.read_csv(DATA_PATH)


# -------------------------
# INPUTS
# -------------------------
st.markdown("### Single Player Prediction")

player = st.selectbox("Select Player", sorted(df["PLAYER_NAME"].dropna().unique()))
n = st.slider("Last N Games", 1, 10, 5)

pts_line = st.number_input("PTS Line", value=20.0)
pra_line = st.number_input("PRA Line", value=30.0)


# -------------------------
# PREDICTION
# -------------------------
if st.button("🚀 Generate Prediction"):

    try:
        result = predict_from_last_n(player_name=player, n=n)

        pts = result.get("predicted_points", 0)
        reb = result.get("predicted_rebounds", 0)
        ast = result.get("predicted_assists", 0)

        pra = pts + reb + ast

        st.markdown("## Results")

        st.metric("Projected PTS", f"{pts:.2f}")
        st.metric("Projected PRA", f"{pra:.2f}")

        st.metric("PTS Over", "YES" if pts > pts_line else "NO")
        st.metric("PRA Over", "YES" if pra > pra_line else "NO")

    except Exception as e:
        st.error(f"Prediction error: {e}")


# -------------------------
# DISCLAIMER
# -------------------------
st.markdown(
    """
<div class="courtiq-card">
  <div style="color:#dc2626; font-weight:800; font-size:1rem;">
    ⚠️ Disclaimer: Court IQ is an AI-powered analytics tool designed to provide
    data-driven projections based on historical NBA performance trends.

    <br/><br/>

    All outputs are probabilistic estimates — not guarantees — and are intended
    strictly for educational, research, and demonstration purposes.

    <br/><br/>

    Users should not rely on these predictions for financial decisions,
    wagering, or real-world risk-based activities.
  </div>
</div>
""",
    unsafe_allow_html=True,
)