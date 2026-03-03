"""
CourtIQ Streamlit App (Clean UI + Matchup Layer + Probabilities + Logs + Charts)

What’s in this app:
- Clean white UI (black text)
- Single Player Prediction:
  - Base projection (our ML model: predict_from_last_n)
  - Optional matchup-adjusted PTS (lightweight layer)
  - Matchup history (H2H) section
  - Over/Under probabilities for BOTH PTS and PRA using a Normal distribution fit
  - Sigma fallback logic:
      1) Use H2H sigma if enough H2H games
      2) Else use last N sigma
      3) Else use a safe fallback sigma
  - Last 5 games log table with DATE + OPP + PTS/REB/AST/PRA/MIN
  - Simple trend charts for PTS/REB/AST/PRA (last games)

Notes to team:
- “Confidence” is a consistency score from last N volatility (not guaranteed).
- Probabilities here are simple normal-fit estimates (not sportsbook odds).
- Matchup-adjusted PTS is not a new model yet; it’s a UI layer on top of base PTS.
"""

from __future__ import annotations

import math
import os
from glob import glob
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.courtiq.models.predict import predict_from_last_n

# Optional: nba_api static teams list (TEAM_ABBREVIATION -> TEAM_ID for NBA CDN logos)
try:
    from nba_api.stats.static import teams as static_teams
except Exception:
    static_teams = None


# ============================================================
# UI CONFIG
# ============================================================
st.set_page_config(page_title="CourtIQ", layout="wide")

st.markdown(
    """
<style>
.stApp { background: #ffffff; color: #000000; }
h1,h2,h3,h4 { color: #000000; }

.courtiq-card {
  background: #f6f7f9;
  border: 1px solid #d9dde3;
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 14px;
}

.courtiq-muted { color: #4b5563; font-size: 0.92rem; }
.courtiq-divider { border-top: 1px solid #d9dde3; margin: 12px 0; }

.stButton button { border-radius: 10px !important; font-weight: 650 !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("CourtIQ — Player Predictions")


# ============================================================
# HELPERS: DATA PATH + LOADING
# ============================================================
def newest_gamelog_csv() -> Path:
    """
    Auto-pick the newest nba_api Regular Season file in data/raw.
    This makes the app use your 2025-26 file automatically once it's created.
    """
    files = glob("data/raw/player_gamelogs_*_Regular_Season*_nba_api.csv")
    if not files:
        return Path("data/raw/player_gamelogs_2024_25_Regular_Season_nba_api.csv")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(files[0])


DATA_PATH = newest_gamelog_csv()


@st.cache_data
def load_gamelogs(csv_path: Path) -> pd.DataFrame:
    """
    Load and normalize gamelogs for UI features:
    - player list
    - team list
    - matchup history
    - last games table
    """
    df = pd.read_csv(csv_path)

    # Normalize text columns
    for col in ["PLAYER_NAME", "TEAM_ABBREVIATION", "MATCHUP", "OPP_TEAM_ABBREVIATION"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Normalize date
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    # Normalize stats
    for col in ["PTS", "REB", "AST", "MIN"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data
def build_team_id_map() -> dict[str, int]:
    """
    TEAM_ABBREVIATION -> TEAM_ID for logos (NBA CDN).
    """
    if static_teams is None:
        return {}
    all_teams = static_teams.get_teams()
    return {t["abbreviation"]: int(t["id"]) for t in all_teams if "abbreviation" in t and "id" in t}


def logo_url_from_team_id(team_id: int) -> str:
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.png"


def logo_url_from_abbr(team_abbr: str, team_id_map: dict[str, int]) -> str | None:
    """
    Prefer NBA CDN via TEAM_ID; if missing, fall back to ESPN logo URL.
    """
    if not team_abbr:
        return None
    abbr = team_abbr.upper().strip()

    if abbr in team_id_map:
        return logo_url_from_team_id(team_id_map[abbr])

    # ESPN fallback
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{abbr.lower()}.png"


def get_player_team_abbr(df: pd.DataFrame, player_name: str) -> str | None:
    """
    Infer player team by most recent row in the dataset.
    """
    if "PLAYER_NAME" not in df.columns or "TEAM_ABBREVIATION" not in df.columns:
        return None

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()
    if sub.empty:
        return None

    if "GAME_DATE" in sub.columns:
        sub = sub.sort_values("GAME_DATE")
    return str(sub.iloc[-1]["TEAM_ABBREVIATION"])


# ============================================================
# HELPERS: MATCHUP + CONFIDENCE + NORMAL PROBABILITIES
# ============================================================
def compute_confidence_from_last_n(df: pd.DataFrame, player_name: str, n: int) -> int | None:
    """
    Consistency score from last N PTS volatility:
    - lower std => higher confidence
    """
    if "PLAYER_NAME" not in df.columns or "PTS" not in df.columns:
        return None

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()
    if sub.empty:
        return None

    if "GAME_DATE" in sub.columns:
        sub = sub.sort_values("GAME_DATE")

    last_n = sub.tail(n)
    pts = pd.to_numeric(last_n.get("PTS"), errors="coerce").dropna()
    if pts.empty:
        return None

    pts_std = float(pts.std(ddof=0)) if len(pts) > 1 else 0.0

    # Simple mapping we can tune later
    conf = 95 - (pts_std * 4.5)
    conf = max(10, min(95, conf))
    return int(round(conf))


def parse_opponent_from_matchup(matchup: str, team_abbr: str) -> str | None:
    """
    Many datasets include MATCHUP like:
      "LAL vs DEN" or "LAL @ DEN"
    Opponent is typically the last token.
    """
    if not matchup or not team_abbr:
        return None

    parts = str(matchup).strip().split()
    if len(parts) < 3:
        return None

    opp = parts[-1].strip().upper()
    team_abbr = str(team_abbr).strip().upper()
    if opp == team_abbr:
        return None
    return opp


def get_matchup_history(df: pd.DataFrame, player_name: str, opp_abbr: str, last_k: int = 10) -> pd.DataFrame:
    """
    Pull last_k games for a player vs opp_abbr.
    Uses OPP_TEAM_ABBREVIATION if present; otherwise parses MATCHUP.
    """
    if "PLAYER_NAME" not in df.columns:
        return pd.DataFrame()

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()
    if sub.empty:
        return pd.DataFrame()

    if "OPP_TEAM_ABBREVIATION" in sub.columns:
        sub["OPP"] = sub["OPP_TEAM_ABBREVIATION"].astype(str).str.upper()
    elif "MATCHUP" in sub.columns and "TEAM_ABBREVIATION" in sub.columns:
        sub["OPP"] = sub.apply(
            lambda r: parse_opponent_from_matchup(r.get("MATCHUP"), r.get("TEAM_ABBREVIATION")),
            axis=1,
        )
    else:
        return pd.DataFrame()

    opp_abbr = str(opp_abbr).upper().strip()
    sub = sub[sub["OPP"] == opp_abbr].copy()
    if sub.empty:
        return pd.DataFrame()

    if "GAME_DATE" in sub.columns:
        sub = sub.sort_values("GAME_DATE")

    cols = []
    for c in ["GAME_DATE", "MATCHUP", "PTS", "REB", "AST", "MIN"]:
        if c in sub.columns:
            cols.append(c)

    out = sub[cols].tail(last_k).copy()
    if "GAME_DATE" in out.columns:
        out["GAME_DATE"] = out["GAME_DATE"].dt.date

    # Add PRA if possible
    if all(c in out.columns for c in ["PTS", "REB", "AST"]):
        out["PRA"] = pd.to_numeric(out["PTS"], errors="coerce") + pd.to_numeric(out["REB"], errors="coerce") + pd.to_numeric(out["AST"], errors="coerce")

    return out


def normal_cdf(z: float) -> float:
    """
    Standard normal CDF using erf (no scipy needed).
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_prob_over(mu: float, sigma: float, line: float) -> float:
    """
    P(X > line) for X ~ N(mu, sigma^2).
    """
    if sigma <= 1e-9:
        return 1.0 if mu > line else 0.0
    z = (line - mu) / sigma
    p_under = normal_cdf(z)
    return float(1.0 - p_under)


def last_n_series(df: pd.DataFrame, player_name: str, col: str, n: int) -> pd.Series:
    """
    Return last N values for a stat col for a player (sorted by date if available).
    """
    if "PLAYER_NAME" not in df.columns or col not in df.columns:
        return pd.Series(dtype=float)

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()
    if sub.empty:
        return pd.Series(dtype=float)

    if "GAME_DATE" in sub.columns:
        sub = sub.sort_values("GAME_DATE")

    s = pd.to_numeric(sub[col], errors="coerce").dropna().tail(n)
    return s


def last_n_pra_series(df: pd.DataFrame, player_name: str, n: int) -> pd.Series:
    """
    PRA = PTS + REB + AST for last N games.
    """
    pts = last_n_series(df, player_name, "PTS", n)
    reb = last_n_series(df, player_name, "REB", n)
    ast = last_n_series(df, player_name, "AST", n)
    if pts.empty or reb.empty or ast.empty:
        return pd.Series(dtype=float)

    # Align last N lengths safely
    m = min(len(pts), len(reb), len(ast))
    pts = pts.tail(m).reset_index(drop=True)
    reb = reb.tail(m).reset_index(drop=True)
    ast = ast.tail(m).reset_index(drop=True)
    return (pts + reb + ast).dropna()


def sigma_with_fallback(
    df: pd.DataFrame,
    player_name: str,
    n: int,
    stat: str,
    opp_abbr_for_sigma: str | None,
    h2h_sigma_min_games: int = 4,
    safe_fallback_sigma: float = 6.0,
) -> tuple[float, str]:
    """
    Sigma fallback logic:
    1) If opponent provided and we have >= min H2H games, use H2H sigma
    2) Else use last N sigma
    3) Else safe fallback sigma
    Returns (sigma, explanation)
    """
    # 1) H2H sigma attempt
    if opp_abbr_for_sigma:
        hist = get_matchup_history(df, player_name, opp_abbr_for_sigma, last_k=50)
        if not hist.empty:
            if stat == "PRA" and "PRA" in hist.columns:
                vals = pd.to_numeric(hist["PRA"], errors="coerce").dropna()
            else:
                vals = pd.to_numeric(hist.get(stat), errors="coerce").dropna()

            if len(vals) >= h2h_sigma_min_games:
                sigma = float(vals.std(ddof=0)) if len(vals) > 1 else safe_fallback_sigma
                sigma = max(1.0, sigma)
                return sigma, f"Sigma used H2H vs {opp_abbr_for_sigma} (n={len(vals)})"

    # 2) last N sigma
    if stat == "PRA":
        s = last_n_pra_series(df, player_name, n)
    else:
        s = last_n_series(df, player_name, stat, n)

    if len(s) >= 2:
        sigma = float(s.std(ddof=0))
        sigma = max(1.0, sigma)
        return sigma, f"Sigma used last {n} games (n={len(s)})"

    # 3) safe fallback
    return safe_fallback_sigma, "Sigma used safe fallback (not enough game data)"


def matchup_adjusted_pts(df: pd.DataFrame, player_name: str, opp_abbr: str, base_pts: float) -> tuple[float, str]:
    """
    Lightweight matchup adjustment:
    - If player has H2H games vs opponent, compute delta = (H2H avg PTS - lastN avg PTS)
    - Apply a capped shift to base_pts

    This keeps the adjustment conservative so we don't overreact to small sample sizes.
    """
    opp_abbr = str(opp_abbr).upper().strip()
    hist = get_matchup_history(df, player_name, opp_abbr, last_k=20)

    if hist.empty or "PTS" not in hist.columns:
        return float(base_pts), "No matchup rows found. Using base PTS."

    h2h_pts = pd.to_numeric(hist["PTS"], errors="coerce").dropna()
    if h2h_pts.empty:
        return float(base_pts), "No valid H2H PTS found. Using base PTS."

    # Last N baseline from raw logs
    last_pts = last_n_series(df, player_name, "PTS", n=10)
    if last_pts.empty:
        return float(base_pts), "No recent PTS series found. Using base PTS."

    h2h_mean = float(h2h_pts.mean())
    recent_mean = float(last_pts.mean())
    delta = h2h_mean - recent_mean

    # Cap the adjustment so one matchup doesn't swing too hard
    capped_delta = max(-4.0, min(4.0, delta))
    adj = float(base_pts + capped_delta)

    explanation = (
        f"H2H avg PTS ({h2h_mean:.1f}) vs recent avg PTS ({recent_mean:.1f}) "
        f"-> delta {delta:+.1f}, capped to {capped_delta:+.1f}. "
        f"Shifted mu to adjusted PTS."
    )
    return adj, explanation


def over_under_probabilities(
    df: pd.DataFrame,
    player_name: str,
    n: int,
    line: float,
    stat: str,
    mu_override: float | None,
    opp_abbr_for_sigma: str | None,
    h2h_sigma_min_games: int = 4,
) -> dict[str, Any]:
    """
    Compute Over/Under using a normal distribution:
      X ~ N(mu, sigma^2)

    - mu is last N mean by default, unless mu_override is passed (e.g., adjusted PTS)
    - sigma uses sigma_with_fallback (H2H sigma -> lastN sigma -> safe fallback)
    """
    try:
        line = float(line)
    except Exception:
        return {"ok": False, "error": "Line must be a number."}

    # mu from series unless overridden
    if mu_override is not None:
        mu = float(mu_override)
        mu_source = "mu_override"
    else:
        if stat == "PRA":
            s = last_n_pra_series(df, player_name, n)
        else:
            s = last_n_series(df, player_name, stat, n)
        if s.empty:
            return {"ok": False, "error": f"No {stat} data found for {player_name}."}
        mu = float(s.mean())
        mu_source = f"last_{n}_mean"

    sigma, sigma_note = sigma_with_fallback(
        df=df,
        player_name=player_name,
        n=n,
        stat=stat,
        opp_abbr_for_sigma=opp_abbr_for_sigma,
        h2h_sigma_min_games=h2h_sigma_min_games,
        safe_fallback_sigma=6.0 if stat == "PTS" else 8.0,
    )

    p_over = normal_prob_over(mu=mu, sigma=sigma, line=line)
    p_under = 1.0 - p_over

    return {
        "ok": True,
        "stat": stat,
        "line": float(line),
        "mu": mu,
        "sigma": sigma,
        "mu_source": mu_source,
        "sigma_note": sigma_note,
        "prob_over": float(p_over),
        "prob_under": float(p_under),
    }


def last_games_table(df: pd.DataFrame, player_name: str, k: int = 5) -> pd.DataFrame:
    """
    Build last k games table with:
    DATE, OPP, PTS, REB, AST, PRA, MIN
    """
    if "PLAYER_NAME" not in df.columns:
        return pd.DataFrame()

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()
    if sub.empty:
        return pd.DataFrame()

    if "GAME_DATE" in sub.columns:
        sub = sub.sort_values("GAME_DATE")
    sub = sub.tail(k).copy()

    # opponent column
    if "OPP_TEAM_ABBREVIATION" in sub.columns:
        sub["OPP"] = sub["OPP_TEAM_ABBREVIATION"].astype(str).str.upper()
    elif "MATCHUP" in sub.columns and "TEAM_ABBREVIATION" in sub.columns:
        sub["OPP"] = sub.apply(
            lambda r: parse_opponent_from_matchup(r.get("MATCHUP"), r.get("TEAM_ABBREVIATION")),
            axis=1,
        )
    else:
        sub["OPP"] = ""

    out_cols = []
    if "GAME_DATE" in sub.columns:
        out_cols.append("GAME_DATE")
    if "MATCHUP" in sub.columns:
        out_cols.append("MATCHUP")
    if "OPP" in sub.columns:
        out_cols.append("OPP")

    for c in ["PTS", "REB", "AST", "MIN"]:
        if c in sub.columns:
            out_cols.append(c)

    out = sub[out_cols].copy()

    # PRA
    if all(c in out.columns for c in ["PTS", "REB", "AST"]):
        out["PRA"] = pd.to_numeric(out["PTS"], errors="coerce") + pd.to_numeric(out["REB"], errors="coerce") + pd.to_numeric(out["AST"], errors="coerce")

    if "GAME_DATE" in out.columns:
        out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.date

    # nice ordering
    preferred = ["GAME_DATE", "MATCHUP", "OPP", "PTS", "REB", "AST", "PRA", "MIN"]
    final_cols = [c for c in preferred if c in out.columns]
    return out[final_cols].sort_values("GAME_DATE", ascending=False)


def last_games_chart_df(df: pd.DataFrame, player_name: str, k: int = 10) -> pd.DataFrame:
    """
    Small dataframe for charts: GAME_DATE + PTS/REB/AST/PRA
    """
    if "PLAYER_NAME" not in df.columns:
        return pd.DataFrame()

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()
    if sub.empty:
        return pd.DataFrame()

    if "GAME_DATE" in sub.columns:
        sub = sub.sort_values("GAME_DATE")
    sub = sub.tail(k).copy()

    out = pd.DataFrame()
    out["GAME_DATE"] = pd.to_datetime(sub.get("GAME_DATE"), errors="coerce").dt.date

    for c in ["PTS", "REB", "AST"]:
        if c in sub.columns:
            out[c] = pd.to_numeric(sub[c], errors="coerce")

    if all(c in out.columns for c in ["PTS", "REB", "AST"]):
        out["PRA"] = out["PTS"] + out["REB"] + out["AST"]

    out = out.dropna(subset=["GAME_DATE"])
    return out


# ============================================================
# LOAD DATA ONCE
# ============================================================
df_logs = None
team_id_map: dict[str, int] = {}

if DATA_PATH.exists():
    df_logs = load_gamelogs(DATA_PATH)
    team_id_map = build_team_id_map()
else:
    st.warning(f"Could not find gamelog file: {DATA_PATH}")


# ============================================================
# TOP: NEAT USER GUIDE (NO SIDEBAR)
# ============================================================
st.markdown(
    f"""
<div class="courtiq-card">
  <div style="font-weight:700; font-size:1.05rem;">How to use</div>
  <div class="courtiq-muted" style="margin-top:6px;">
    • Enter a player name and choose Last N games<br/>
    • Optional: choose an opponent to apply matchup-adjusted PTS + show matchup history<br/>
    • Enter PTS and PRA lines for Over/Under probabilities<br/>
    <span style="font-size:0.88rem;">Using file: <b>{DATA_PATH.name}</b></span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SINGLE PLAYER SECTION
# ============================================================
st.markdown('<div class="courtiq-card">', unsafe_allow_html=True)
st.markdown("## Single Player Prediction")
st.markdown(
    '<div class="courtiq-muted">Base projection + optional matchup-adjusted PTS + matchup-aware Over/Under probability.</div>',
    unsafe_allow_html=True,
)

# Inputs
player = st.text_input("Player name", value="Kevin Durant")
n = st.slider("Last N games", min_value=1, max_value=10, value=5)

opp_abbr = None
teams_list: list[str] = []
if df_logs is not None and "TEAM_ABBREVIATION" in df_logs.columns:
    teams_list = sorted(df_logs["TEAM_ABBREVIATION"].dropna().unique().tolist())
    opp_abbr = st.selectbox(
        "Opponent (optional, for matchup history + adjusted PTS)",
        options=["—"] + teams_list,
    )

c_line1, c_line2 = st.columns(2)
with c_line1:
    pts_line = st.number_input("PTS line (for Over/Under probability)", min_value=0.0, value=13.5, step=0.5)
with c_line2:
    pra_line = st.number_input("PRA line (for Over/Under probability)", min_value=0.0, value=25.5, step=0.5)

# Predict button ABOVE logs (user friendly)
st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)

note_col, btn_col = st.columns([3, 1])
with note_col:
    st.caption("Tip: If you don’t see recent games, make sure you fetched the newest season CSV (2025–26).")
with btn_col:
    do_predict = st.button("Predict", use_container_width=True)

# Output
if do_predict:
    if df_logs is None:
        st.error("Gamelog dataset is missing. Add a CSV into data/raw and re-run the app.")
    else:
        # Base model projection
        result = predict_from_last_n(player_name=player, n=n)

        base_pts = float(result.get("predicted_points", 0.0))
        base_reb = float(result.get("predicted_rebounds", 0.0))
        base_ast = float(result.get("predicted_assists", 0.0))

        # Matchup-adjusted PTS (optional)
        adj_pts = base_pts
        adj_note = "No opponent selected. Using base PTS."
        opp_for_sigma = None

        if opp_abbr and opp_abbr != "—":
            adj_pts, adj_note = matchup_adjusted_pts(df_logs, player, opp_abbr, base_pts)
            opp_for_sigma = opp_abbr

        adj_pra = float(adj_pts + base_reb + base_ast)

        # Logos
        p_team = get_player_team_abbr(df_logs, player)
        p_logo = logo_url_from_abbr(p_team, team_id_map) if p_team else None
        opp_logo = logo_url_from_abbr(opp_abbr, team_id_map) if (opp_abbr and opp_abbr != "—") else None

        # Confidence
        conf = compute_confidence_from_last_n(df_logs, player, n)

        # Probabilities (PTS + PRA)
        pts_ou = over_under_probabilities(
            df=df_logs,
            player_name=player,
            n=n,
            line=float(pts_line),
            stat="PTS",
            mu_override=float(adj_pts),  # Shift mu to adjusted PTS
            opp_abbr_for_sigma=opp_for_sigma,
            h2h_sigma_min_games=4,
        )

        pra_ou = over_under_probabilities(
            df=df_logs,
            player_name=player,
            n=n,
            line=float(pra_line),
            stat="PRA",
            mu_override=float(adj_pra),  # Shift mu to adjusted PRA
            opp_abbr_for_sigma=opp_for_sigma,
            h2h_sigma_min_games=4,
        )

        st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)

        # Header row with logo(s)
        head_left, head_mid, head_right = st.columns([1, 4, 1])
        with head_left:
            if p_logo:
                st.image(p_logo, width=72)
            elif p_team:
                st.write(p_team)

        with head_mid:
            st.markdown("### Projection")
            st.caption("Base (model) + optional matchup layer when opponent is selected.")

        with head_right:
            if opp_logo:
                st.image(opp_logo, width=72)
            elif opp_abbr and opp_abbr != "—":
                st.write(opp_abbr)

        # Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("PTS (Base)", f"{base_pts:.2f}")
        m2.metric("PTS (Adjusted)", f"{float(adj_pts):.2f}")
        m3.metric("REB", f"{base_reb:.2f}")
        m4.metric("AST", f"{base_ast:.2f}")
        m5.metric("PRA (Adj)", f"{adj_pra:.2f}")

        st.markdown("#### How the adjusted PTS was calculated")
        st.write(adj_note)

        # Confidence
        st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
        if conf is not None:
            st.metric("Confidence (Consistency)", f"{conf}%")
            st.caption("Based on how stable the last N games have been (lower volatility = higher confidence).")
        else:
            st.metric("Confidence (Consistency)", "—")

        # Over/Under block (PTS + PRA)
        st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Over / Under Probabilities")

        ou1, ou2 = st.columns(2)

        with ou1:
            st.markdown("**PTS Line**")
            if pts_ou.get("ok"):
                st.metric("Line", f"{pts_ou['line']:.1f}")
                st.metric("Prob OVER", f"{pts_ou['prob_over']*100:.0f}%")
                st.metric("Prob UNDER", f"{pts_ou['prob_under']*100:.0f}%")
                st.caption(f"mu={pts_ou['mu']:.2f} | sigma={pts_ou['sigma']:.2f}")
                st.caption(f"{pts_ou['sigma_note']}")
            else:
                st.info(pts_ou.get("error", "Could not compute PTS probabilities."))

        with ou2:
            st.markdown("**PRA Line**")
            if pra_ou.get("ok"):
                st.metric("Line", f"{pra_ou['line']:.1f}")
                st.metric("Prob OVER", f"{pra_ou['prob_over']*100:.0f}%")
                st.metric("Prob UNDER", f"{pra_ou['prob_under']*100:.0f}%")
                st.caption(f"mu={pra_ou['mu']:.2f} | sigma={pra_ou['sigma']:.2f}")
                st.caption(f"{pra_ou['sigma_note']}")
            else:
                st.info(pra_ou.get("error", "Could not compute PRA probabilities."))

        # Matchup history section
        if opp_abbr and opp_abbr != "—":
            st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Matchup History (H2H)")

            hist = get_matchup_history(df_logs, player, opp_abbr, last_k=10)
            if hist.empty:
                st.info("No matchup history found (or dataset missing MATCHUP/OPP_TEAM_ABBREVIATION).")
            else:
                st.dataframe(hist, width="stretch", hide_index=True)

                if "PTS" in hist.columns:
                    pts_series = pd.to_numeric(hist["PTS"], errors="coerce").dropna()
                    if not pts_series.empty:
                        st.caption(
                            f"H2H vs {opp_abbr}: Games={len(pts_series)}, Avg PTS={pts_series.mean():.1f}, Last PTS={pts_series.iloc[-1]:.0f}"
                        )

        # Last 5 games log (below button, user requested)
        st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Last 5 Games Log")

        last5 = last_games_table(df_logs, player, k=5)
        if last5.empty:
            st.info("No recent game logs found for that player in this dataset.")
        else:
            st.dataframe(last5, width="stretch", hide_index=True)

        # Charts
        st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Trend Charts (Last Games)")

        chart_df = last_games_chart_df(df_logs, player, k=10)
        if chart_df.empty:
            st.info("Not enough data to chart.")
        else:
            chart_df = chart_df.set_index("GAME_DATE")
            if "PTS" in chart_df.columns:
                st.line_chart(chart_df["PTS"], height=180)
            if "REB" in chart_df.columns:
                st.line_chart(chart_df["REB"], height=180)
            if "AST" in chart_df.columns:
                st.line_chart(chart_df["AST"], height=180)
            if "PRA" in chart_df.columns:
                st.line_chart(chart_df["PRA"], height=180)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PICK BUILDER SECTION (kept, but neat)
# ============================================================
st.markdown('<div class="courtiq-card">', unsafe_allow_html=True)
st.markdown("## Pick Builder (Team → Player)")
st.markdown('<div class="courtiq-muted">Build a pool, then generate 2–8 picks. Optional opponent adds adjusted PTS + matchup-aware sigma.</div>', unsafe_allow_html=True)

if df_logs is None or "TEAM_ABBREVIATION" not in df_logs.columns or "PLAYER_NAME" not in df_logs.columns:
    st.error("Pick Builder needs TEAM_ABBREVIATION and PLAYER_NAME in the gamelog CSV.")
else:
    teams = sorted(df_logs["TEAM_ABBREVIATION"].dropna().unique().tolist())
    team = st.selectbox("Select Team", options=teams, key="pb_team")

    players = (
        df_logs.loc[df_logs["TEAM_ABBREVIATION"] == team, "PLAYER_NAME"]
        .dropna()
        .unique()
        .tolist()
    )
    players = sorted(players)
    player_pick = st.selectbox("Select Player", options=players, key="pb_player")

    if "pick_pool" not in st.session_state:
        st.session_state.pick_pool = []

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Add to Pool"):
            if player_pick not in st.session_state.pick_pool:
                st.session_state.pick_pool.append(player_pick)

    with c2:
        if st.button("Clear Pool"):
            st.session_state.pick_pool = []

    with c3:
        st.markdown("**Current Pool**")
        if st.session_state.pick_pool:
            st.dataframe({"Player": st.session_state.pick_pool}, width="stretch", hide_index=True)
        else:
            st.write("—")

    num_picks = st.slider("How many picks? (2–8)", 2, 8, 5, key="pb_num")
    mode = st.radio("Pick Mode", ["Randomize", "Top Points", "Top Edge"], horizontal=True, key="pb_mode")

    opp2 = st.selectbox("Opponent (optional, uses adjusted + matchup sigma)", options=["—"] + teams, key="pb_opp")

    pb1, pb2 = st.columns(2)
    with pb1:
        pb_pts_line = st.number_input("PTS line (Pick Builder)", min_value=0.0, value=20.5, step=0.5, key="pb_pts_line")
    with pb2:
        pb_pra_line = st.number_input("PRA line (Pick Builder)", min_value=0.0, value=30.5, step=0.5, key="pb_pra_line")

    if st.button("Build Picks"):
        pool = st.session_state.pick_pool

        if len(pool) < num_picks:
            st.error(f"Add at least {num_picks} players to the pool.")
        else:
            rows: list[dict[str, Any]] = []
            for name in pool:
                r = predict_from_last_n(player_name=name, n=n)

                base_pts = float(r.get("predicted_points", 0.0))
                base_reb = float(r.get("predicted_rebounds", 0.0))
                base_ast = float(r.get("predicted_assists", 0.0))

                use_pts = base_pts
                opp_for_sigma = None
                if opp2 != "—":
                    use_pts, _note = matchup_adjusted_pts(df_logs, name, opp2, base_pts)
                    opp_for_sigma = opp2

                use_pra = float(use_pts + base_reb + base_ast)

                pts_ou = over_under_probabilities(
                    df=df_logs,
                    player_name=name,
                    n=n,
                    line=float(pb_pts_line),
                    stat="PTS",
                    mu_override=use_pts if opp2 != "—" else None,
                    opp_abbr_for_sigma=opp_for_sigma,
                    h2h_sigma_min_games=4,
                )
                pra_ou = over_under_probabilities(
                    df=df_logs,
                    player_name=name,
                    n=n,
                    line=float(pb_pra_line),
                    stat="PRA",
                    mu_override=use_pra if opp2 != "—" else None,
                    opp_abbr_for_sigma=opp_for_sigma,
                    h2h_sigma_min_games=4,
                )

                pts_edge = None
                pts_over = None
                if pts_ou.get("ok"):
                    pts_over = float(pts_ou["prob_over"])
                    pts_edge = (pts_over - 0.5) * 100.0  # simple edge vs 50/50 baseline

                pra_edge = None
                pra_over = None
                if pra_ou.get("ok"):
                    pra_over = float(pra_ou["prob_over"])
                    pra_edge = (pra_over - 0.5) * 100.0

                best_edge = None
                candidates = [e for e in [pts_edge, pra_edge] if e is not None]
                if candidates:
                    best_edge = max(candidates)

                rows.append(
                    {
                        "Player": r.get("player", name),
                        "PTS": round(float(use_pts), 2),
                        "PRA": round(float(use_pra), 2),
                        "PTS_ProbOver": None if pts_over is None else round(pts_over * 100, 0),
                        "PTS_Edge%": None if pts_edge is None else round(float(pts_edge), 1),
                        "PRA_ProbOver": None if pra_over is None else round(pra_over * 100, 0),
                        "PRA_Edge%": None if pra_edge is None else round(float(pra_edge), 1),
                        "BestEdge%": None if best_edge is None else round(float(best_edge), 1),
                    }
                )

            # pick selection
            if mode == "Top Points":
                picks = sorted(rows, key=lambda x: x["PTS"], reverse=True)[:num_picks]
            elif mode == "Top Edge":
                picks = sorted(
                    rows,
                    key=lambda x: (x["BestEdge%"] is not None, x["BestEdge%"] if x["BestEdge%"] is not None else -999),
                    reverse=True,
                )[:num_picks]
            else:
                import random as _random

                picks = _random.sample(rows, num_picks)

            st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
            st.success(f"{num_picks}-Pick Set")

            # Render picks as cards (2 columns)
            for idx in range(0, len(picks), 2):
                cols = st.columns(2)
                for j in range(2):
                    k = idx + j
                    if k >= len(picks):
                        break

                    p = picks[k]
                    name = p["Player"]
                    pts = float(p["PTS"])
                    pra = float(p["PRA"])

                    with cols[j]:
                        team_abbr = get_player_team_abbr(df_logs, name) if df_logs is not None else None
                        logo = logo_url_from_abbr(team_abbr, team_id_map) if team_abbr else None
                        conf = compute_confidence_from_last_n(df_logs, name, n) if df_logs is not None else None

                        st.markdown('<div class="courtiq-card">', unsafe_allow_html=True)

                        tl, tr = st.columns([1, 3])
                        with tl:
                            if logo:
                                st.image(logo, width=64)
                            elif team_abbr:
                                st.write(team_abbr)

                        with tr:
                            st.markdown(f"### {name}")
                            subtitle = "Projection (Adjusted)" if opp2 != "—" else "Projection"
                            st.markdown(f'<div class="courtiq-muted">{subtitle}</div>', unsafe_allow_html=True)

                        st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)

                        a, b, c = st.columns(3)
                        a.metric("PTS", f"{pts:.2f}")
                        b.metric("PRA", f"{pra:.2f}")
                        c.metric("Confidence", f"{conf}%" if conf is not None else "—")

                        st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
                        st.markdown("**Over Probabilities**")

                        x, y = st.columns(2)
                        with x:
                            st.metric("PTS Over", f'{p.get("PTS_ProbOver")}%' if p.get("PTS_ProbOver") is not None else "—")
                            st.caption(f'Edge: {p.get("PTS_Edge%","—")}%')
                        with y:
                            st.metric("PRA Over", f'{p.get("PRA_ProbOver")}%' if p.get("PRA_ProbOver") is not None else "—")
                            st.caption(f'Edge: {p.get("PRA_Edge%","—")}%')

                        st.caption(f'Best Edge: {p.get("BestEdge%","—")}%')
                        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)