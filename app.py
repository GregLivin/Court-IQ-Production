from __future__ import annotations

import math
import os
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.courtiq.models.predict import predict_from_last_n


# -------------------------
# PAGE SETUP
# -------------------------
st.set_page_config(page_title="CourtIQ", layout="wide")

st.title("CourtIQ — Player Predictions")


# -------------------------
# DATA FILE SELECTOR (FIXED)
# -------------------------
def newest_gamelog_csv() -> Path:
    # PRIORITY: always use 2025–26 if it exists
    preferred = Path("data/raw/player_gamelogs_2025_26_Regular_Season_nba_api.csv")
    if preferred.exists():
        return preferred

    # fallback to newest file
    files = glob("data/raw/player_gamelogs_*_Regular_Season*_nba_api.csv")

    if not files:
        return Path("data/raw/player_gamelogs_2024_25_Regular_Season_nba_api.csv")

    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(files[0])


DATA_PATH = newest_gamelog_csv()


# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_gamelogs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df.columns = [col.strip().upper() for col in df.columns]

    # normalize key columns
    if "PLAYER_NAME" not in df.columns and "PLAYER" in df.columns:
        df.rename(columns={"PLAYER": "PLAYER_NAME"}, inplace=True)

    if "TEAM_ABBREVIATION" not in df.columns and "TEAM" in df.columns:
        df.rename(columns={"TEAM": "TEAM_ABBREVIATION"}, inplace=True)

    # build opponent from MATCHUP if needed
    if "MATCHUP" in df.columns and "OPP_TEAM_ABBREVIATION" not in df.columns:
        df["OPP_TEAM_ABBREVIATION"] = df["MATCHUP"].astype(str).str.split().str[-1]

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    return df


if DATA_PATH.exists():
    df_logs = load_gamelogs(DATA_PATH)
else:
    st.error("No gamelog file found.")
    st.stop()


# -------------------------
# UI HEADER
# -------------------------
st.markdown(f"""
### Smarter NBA prop insights in seconds

Using dataset: **{DATA_PATH.name}**
""")

st.markdown("""
### How to Use
1. Enter player name  
2. Select last N games  
3. (Optional) choose opponent  
4. Enter lines  
5. Click Predict  
""")


# -------------------------
# INPUT SECTION
# -------------------------
st.markdown("## Step 1: Choose Player")

col1, col2 = st.columns(2)

with col1:
    player = st.text_input("Player", value="Kevin Durant")

with col2:
    n = st.slider("Last N games", 1, 10, 5)


teams = sorted(df_logs["TEAM_ABBREVIATION"].dropna().unique())

opp = st.selectbox("Opponent (optional)", ["—"] + teams)

col1, col2 = st.columns(2)

with col1:
    pts_line = st.number_input("PTS line", value=13.5)

with col2:
    pra_line = st.number_input("PRA line", value=25.5)


# -------------------------
# PREDICT
# -------------------------
if st.button("Predict"):

    with st.spinner("Analyzing..."):

        result = predict_from_last_n(player_name=player, n=n)

        base_pts = result["predicted_points"]
        base_reb = result["predicted_rebounds"]
        base_ast = result["predicted_assists"]

        pra = base_pts + base_reb + base_ast

    st.markdown("## Step 2: Results")

    c1, c2, c3 = st.columns(3)

    c1.metric("PTS", f"{base_pts:.2f}")
    c2.metric("PRA", f"{pra:.2f}")
    c3.metric("REB + AST", f"{(base_reb+base_ast):.2f}")

    st.markdown("## Step 3: Quick Insight")

    if base_pts > pts_line:
        st.success("Lean: OVER")
    else:
        st.warning("Lean: UNDER")


# -------------------------
# PICK BUILDER
# -------------------------
st.markdown("## Step 4: Pick Builder")

team = st.selectbox("Select Team", teams)

players = df_logs[df_logs["TEAM_ABBREVIATION"] == team]["PLAYER_NAME"].dropna().unique()
player_pick = st.selectbox("Select Player", players)

if "pool" not in st.session_state:
    st.session_state.pool = []

if st.button("Add to Pool"):
    if player_pick not in st.session_state.pool:
        st.session_state.pool.append(player_pick)

if st.button("Clear Pool"):
    st.session_state.pool = []

st.write("Current Pool:", st.session_state.pool)

num = st.slider("Number of picks", 2, 8, 5)

if st.button("Generate Picks"):

    picks = st.session_state.pool[:num]

    st.success("Generated Picks")

    for p in picks:
        st.write(p)


# -------------------------
# FOOTER
# -------------------------
st.caption("CourtIQ is for research and entertainment purposes.")