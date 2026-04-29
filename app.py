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

st.markdown(
    """
<style>
.stApp {
    background: #f4f7fb;
    color: #111827;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.courtiq-header {
    text-align: center;
    margin-bottom: 1.5rem;
}

.courtiq-title {
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: 0.5px;
    color: #0f172a;
    margin-bottom: 0.2rem;
}

.courtiq-title span {
    color: #2563eb;
}

.courtiq-subtitle {
    font-size: 1.05rem;
    color: #64748b;
    margin-bottom: 0.9rem;
}

.courtiq-pill-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 1.25rem;
}

.courtiq-pill {
    background: white;
    border: 1px solid #dbeafe;
    border-radius: 999px;
    padding: 8px 14px;
    font-size: 0.82rem;
    font-weight: 700;
    color: #1e3a8a;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.06);
}

.courtiq-hero {
    background: linear-gradient(135deg, #eff6ff 0%, #eef2ff 100%);
    border: 1px solid #dbeafe;
    border-radius: 22px;
    padding: 24px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
}

.courtiq-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}

.courtiq-section-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.35rem;
}

.courtiq-muted {
    color: #64748b;
    font-size: 0.96rem;
}

.result-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 24px;
    border-radius: 20px;
    color: white;
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.18);
}

.result-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #cbd5e1;
    margin-bottom: 0.25rem;
}

.result-value {
    font-size: 2.8rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.45rem;
}

.result-sub {
    color: #93c5fd;
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.result-muted {
    color: #94a3b8;
    font-size: 0.95rem;
}

.courtiq-divider {
    border-top: 1px solid #e2e8f0;
    margin: 16px 0;
}

.stButton > button {
    border-radius: 12px !important;
    font-weight: 800 !important;
    border: none !important;
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
}

@media (max-width: 768px) {
    .courtiq-title {
        font-size: 2.4rem;
    }

    .result-value {
        font-size: 2.2rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="courtiq-header">
    <div class="courtiq-title">COURT <span>IQ</span></div>
    <div class="courtiq-subtitle">Smarter NBA prop insights in seconds</div>
    <div class="courtiq-pill-row">
        <div class="courtiq-pill">PTS Projections</div>
        <div class="courtiq-pill">PRA Insights</div>
        <div class="courtiq-pill">Matchup Trends</div>
        <div class="courtiq-pill">Confidence Score</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("AI-Powered NBA Analytics • Real Game Data • Predictive Insights")


def newest_gamelog_csv() -> Path:
    preferred = Path("data/raw/player_gamelogs_2025_26_Regular_Season_nba_api.csv")

    if preferred.exists():
        return preferred

    files = glob("data/raw/player_gamelogs_*_Regular_Season*_nba_api.csv")

    if not files:
        return Path("data/raw/player_gamelogs_2024_25_Regular_Season_nba_api.csv")

    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(files[0])


DATA_PATH = newest_gamelog_csv()


@st.cache_data
def load_gamelogs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip().upper() for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()

    rename_map = {}

    if "PLAYER" in df.columns:
        rename_map["PLAYER"] = "PLAYER_NAME"
    if "NAME" in df.columns:
        rename_map["NAME"] = "PLAYER_NAME"
    if "PLAYERNAME" in df.columns:
        rename_map["PLAYERNAME"] = "PLAYER_NAME"
    if "PLAYER_FULL_NAME" in df.columns:
        rename_map["PLAYER_FULL_NAME"] = "PLAYER_NAME"
    if "FULL_NAME" in df.columns:
        rename_map["FULL_NAME"] = "PLAYER_NAME"

    if "TEAM" in df.columns:
        rename_map["TEAM"] = "TEAM_ABBREVIATION"
    if "TEAM_ABBR" in df.columns:
        rename_map["TEAM_ABBR"] = "TEAM_ABBREVIATION"
    if "TEAM_ABBREV" in df.columns:
        rename_map["TEAM_ABBREV"] = "TEAM_ABBREVIATION"
    if "TEAM_NAME" in df.columns:
        rename_map["TEAM_NAME"] = "TEAM_ABBREVIATION"
    if "TEAMCODE" in df.columns:
        rename_map["TEAMCODE"] = "TEAM_ABBREVIATION"

    if "OPPONENT" in df.columns:
        rename_map["OPPONENT"] = "OPP_TEAM_ABBREVIATION"
    if "OPP_TEAM" in df.columns:
        rename_map["OPP_TEAM"] = "OPP_TEAM_ABBREVIATION"
    if "OPP" in df.columns:
        rename_map["OPP"] = "OPP_TEAM_ABBREVIATION"

    df = df.rename(columns=rename_map)

    if "MATCHUP" in df.columns:
        matchup_series = df["MATCHUP"].astype(str).str.strip()

        if "TEAM_ABBREVIATION" not in df.columns:
            df["TEAM_ABBREVIATION"] = matchup_series.str.split().str[0]

        if "OPP_TEAM_ABBREVIATION" not in df.columns:
            df["OPP_TEAM_ABBREVIATION"] = matchup_series.str.split().str[-1]

    for col in ["PLAYER_NAME", "TEAM_ABBREVIATION", "MATCHUP", "OPP_TEAM_ABBREVIATION"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "PLAYER_NAME" in df.columns:
        df["PLAYER_NAME"] = df["PLAYER_NAME"].replace({"nan": None, "None": None})

    if "TEAM_ABBREVIATION" in df.columns:
        df["TEAM_ABBREVIATION"] = (
            df["TEAM_ABBREVIATION"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": None, "NONE": None})
        )

    if "OPP_TEAM_ABBREVIATION" in df.columns:
        df["OPP_TEAM_ABBREVIATION"] = (
            df["OPP_TEAM_ABBREVIATION"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": None, "NONE": None})
        )

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    stat_rename_map = {}

    if "POINTS" in df.columns:
        stat_rename_map["POINTS"] = "PTS"
    if "REBOUNDS" in df.columns:
        stat_rename_map["REBOUNDS"] = "REB"
    if "ASSISTS" in df.columns:
        stat_rename_map["ASSISTS"] = "AST"
    if "MINUTES" in df.columns:
        stat_rename_map["MINUTES"] = "MIN"

    df = df.rename(columns=stat_rename_map)

    for col in ["PTS", "REB", "AST", "MIN"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data
def build_team_id_map() -> dict[str, int]:
    if static_teams is None:
        return {}

    all_teams = static_teams.get_teams()

    return {
        t["abbreviation"]: int(t["id"])
        for t in all_teams
        if "abbreviation" in t and "id" in t
    }


@st.cache_data
def build_player_id_map(df: pd.DataFrame) -> dict[str, int]:
    if df is None or df.empty:
        return {}

    temp = df.loc[:, ~df.columns.duplicated()].copy()

    if "PLAYER_NAME" not in temp.columns or "PLAYER_ID" not in temp.columns:
        return {}

    temp = temp[["PLAYER_NAME", "PLAYER_ID"]].dropna().copy()
    temp["PLAYER_NAME"] = temp["PLAYER_NAME"].astype(str).str.strip()

    if isinstance(temp["PLAYER_ID"], pd.DataFrame):
        temp["PLAYER_ID"] = temp["PLAYER_ID"].iloc[:, 0]

    temp["PLAYER_ID"] = pd.to_numeric(temp["PLAYER_ID"], errors="coerce")
    temp = temp.dropna(subset=["PLAYER_ID"]).copy()
    temp["PLAYER_ID"] = temp["PLAYER_ID"].astype(int)
    temp = temp.drop_duplicates(subset=["PLAYER_NAME"], keep="last")

    return dict(zip(temp["PLAYER_NAME"], temp["PLAYER_ID"]))


def logo_url_from_team_id(team_id: int) -> str:
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.png"


def logo_url_from_abbr(team_abbr: str, team_id_map: dict[str, int]) -> str | None:
    if not team_abbr:
        return None

    abbr = team_abbr.upper().strip()

    if abbr in team_id_map:
        return logo_url_from_team_id(team_id_map[abbr])

    return f"https://a.espncdn.com/i/teamlogos/nba/500/{abbr.lower()}.png"


def player_headshot_url(player_id: int) -> str:
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"


def get_player_team_abbr(df: pd.DataFrame, player_name: str) -> str | None:
    if "PLAYER_NAME" not in df.columns or "TEAM_ABBREVIATION" not in df.columns:
        return None

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()

    if sub.empty:
        return None

    if "GAME_DATE" in sub.columns:
        sub = sub.sort_values("GAME_DATE")

    return str(sub.iloc[-1]["TEAM_ABBREVIATION"])


def compute_confidence_from_last_n(df: pd.DataFrame, player_name: str, n: int) -> int | None:
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
    conf = 95 - (pts_std * 4.5)
    conf = max(10, min(95, conf))

    return int(round(conf))


def parse_opponent_from_matchup(matchup: str, team_abbr: str) -> str | None:
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

    if all(c in out.columns for c in ["PTS", "REB", "AST"]):
        out["PRA"] = (
            pd.to_numeric(out["PTS"], errors="coerce")
            + pd.to_numeric(out["REB"], errors="coerce")
            + pd.to_numeric(out["AST"], errors="coerce")
        )

    return out


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_prob_over(mu: float, sigma: float, line: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if mu > line else 0.0

    z = (line - mu) / sigma
    p_under = normal_cdf(z)

    return float(1.0 - p_under)


def last_n_series(df: pd.DataFrame, player_name: str, col: str, n: int) -> pd.Series:
    if "PLAYER_NAME" not in df.columns or col not in df.columns:
        return pd.Series(dtype=float)

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()

    if sub.empty:
        return pd.Series(dtype=float)

    if "GAME_DATE" in sub.columns:
        sub = sub.sort_values("GAME_DATE")

    return pd.to_numeric(sub[col], errors="coerce").dropna().tail(n)


def last_n_pra_series(df: pd.DataFrame, player_name: str, n: int) -> pd.Series:
    pts = last_n_series(df, player_name, "PTS", n)
    reb = last_n_series(df, player_name, "REB", n)
    ast = last_n_series(df, player_name, "AST", n)

    if pts.empty or reb.empty or ast.empty:
        return pd.Series(dtype=float)

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

    if stat == "PRA":
        s = last_n_pra_series(df, player_name, n)
    else:
        s = last_n_series(df, player_name, stat, n)

    if len(s) >= 2:
        sigma = float(s.std(ddof=0))
        sigma = max(1.0, sigma)
        return sigma, f"Sigma used last {n} games (n={len(s)})"

    return safe_fallback_sigma, "Sigma used safe fallback (not enough game data)"


def matchup_adjusted_pts(df: pd.DataFrame, player_name: str, opp_abbr: str, base_pts: float) -> tuple[float, str]:
    opp_abbr = str(opp_abbr).upper().strip()
    hist = get_matchup_history(df, player_name, opp_abbr, last_k=20)

    if hist.empty or "PTS" not in hist.columns:
        return float(base_pts), "No matchup rows found. Using base PTS."

    h2h_pts = pd.to_numeric(hist["PTS"], errors="coerce").dropna()

    if h2h_pts.empty:
        return float(base_pts), "No valid H2H PTS found. Using base PTS."

    last_pts = last_n_series(df, player_name, "PTS", n=10)

    if last_pts.empty:
        return float(base_pts), "No recent PTS series found. Using base PTS."

    h2h_mean = float(h2h_pts.mean())
    recent_mean = float(last_pts.mean())
    delta = h2h_mean - recent_mean
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
    try:
        line = float(line)
    except Exception:
        return {"ok": False, "error": "Line must be a number."}

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
    if "PLAYER_NAME" not in df.columns:
        return pd.DataFrame()

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()

    if sub.empty:
        return pd.DataFrame()

    if "GAME_DATE" in sub.columns:
        sub = sub.sort_values("GAME_DATE")

    sub = sub.tail(k).copy()

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

    for c in ["GAME_DATE", "MATCHUP", "OPP", "PTS", "REB", "AST", "MIN"]:
        if c in sub.columns:
            out_cols.append(c)

    out = sub[out_cols].copy()

    if all(c in out.columns for c in ["PTS", "REB", "AST"]):
        out["PRA"] = (
            pd.to_numeric(out["PTS"], errors="coerce")
            + pd.to_numeric(out["REB"], errors="coerce")
            + pd.to_numeric(out["AST"], errors="coerce")
        )

    if "GAME_DATE" in out.columns:
        out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.date

    preferred = ["GAME_DATE", "MATCHUP", "OPP", "PTS", "REB", "AST", "PRA", "MIN"]
    final_cols = [c for c in preferred if c in out.columns]

    return out[final_cols].sort_values("GAME_DATE", ascending=False)


def verdict_from_prob(prob_over: float) -> tuple[str, str]:
    if prob_over >= 0.60:
        return "Strong OVER", "success"

    if prob_over <= 0.40:
        return "Strong UNDER", "error"

    return "No clear edge", "warning"


df_logs = None
team_id_map: dict[str, int] = {}
player_id_map: dict[str, int] = {}

if DATA_PATH.exists():
    df_logs = load_gamelogs(DATA_PATH)
    team_id_map = build_team_id_map()
    player_id_map = build_player_id_map(df_logs)
else:
    st.error(f"Could not find gamelog file: {DATA_PATH}")
    st.stop()


st.markdown(
"""
<div class="courtiq-card">
  <div class="courtiq-section-title">How does Court IQ work?</div>

  <div class="courtiq-muted" style="margin-top:6px; line-height:1.7;">
    1. Select a player<br>
    2. Set the "Last N Games" slider (1–10)<br>
    3. Enter PTS & PRA lines<br>
    4. Generate prediction
  </div>

</div>
""",
unsafe_allow_html=True
)

st.info("Tip: Look for confidence above 70% for more stable players.")


st.markdown('<div class="courtiq-card">', unsafe_allow_html=True)
st.markdown('<div class="courtiq-section-title">Single Player Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="courtiq-muted">Analyze one player using recent performance, matchup history, and projected over/under probability.</div>',
    unsafe_allow_html=True,
)

players_list: list[str] = []
default_index = 0

if df_logs is not None and "PLAYER_NAME" in df_logs.columns:
    players_list = sorted(df_logs["PLAYER_NAME"].dropna().unique().tolist())

    if "Kevin Durant" in players_list:
        default_index = players_list.index("Kevin Durant")

left_input, right_input = st.columns([2, 1])

with left_input:
    if players_list:
        player = st.selectbox(
            "Select player",
            options=players_list,
            index=default_index,
            help="Choose the player you want to analyze.",
        )
    else:
        player = st.text_input(
            "Player name",
            value="Kevin Durant",
            help="Enter the player you want to analyze.",
        )

with right_input:
    n = st.slider(
        "Last N games",
        min_value=1,
        max_value=10,
        value=5,
        help="How many recent games to use in the projection.",
    )

opp_abbr = None
teams_list: list[str] = []

if df_logs is not None and "TEAM_ABBREVIATION" in df_logs.columns:
    teams_list = sorted(df_logs["TEAM_ABBREVIATION"].dropna().unique().tolist())

    opp_abbr = st.selectbox(
        "Opponent (optional, for matchup history + adjusted PTS)",
        options=["—"] + teams_list,
        help="Optional opponent selection for matchup-adjusted points and head-to-head history.",
    )

c_line1, c_line2 = st.columns(2)

with c_line1:
    pts_line = st.number_input(
        "PTS line (for Over/Under probability)",
        min_value=0.0,
        value=13.5,
        step=0.5,
        help="Betting line for points.",
    )

with c_line2:
    pra_line = st.number_input(
        "PRA line (for Over/Under probability)",
        min_value=0.0,
        value=25.5,
        step=0.5,
        help="Points + Rebounds + Assists line.",
    )

st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)

note_col, btn_col = st.columns([3, 1])

with note_col:
    st.caption("Tip: Higher confidence usually means the player has been more consistent recently.")

with btn_col:
    do_predict = st.button("🚀 Generate Prediction", use_container_width=True)

if do_predict:
    if df_logs is None:
        st.error("Gamelog dataset is missing. Add a CSV into data/raw and re-run the app.")
    else:
        try:
            with st.spinner("Analyzing player data..."):
                result = predict_from_last_n(player_name=player, n=n)

                if not result.get("ok", True):
                    st.error(result.get("error", "Prediction failed."))
                    st.stop()

                base_pts = float(result.get("predicted_points", 0.0))
                base_reb = float(result.get("predicted_rebounds", 0.0))
                base_ast = float(result.get("predicted_assists", 0.0))

                adj_pts = base_pts
                adj_note = "No opponent selected. Using base PTS."
                opp_for_sigma = None

                if opp_abbr and opp_abbr != "—":
                    adj_pts, adj_note = matchup_adjusted_pts(df_logs, player, opp_abbr, base_pts)
                    opp_for_sigma = opp_abbr

                adj_pra = float(adj_pts + base_reb + base_ast)

                p_team = get_player_team_abbr(df_logs, player)
                player_id = player_id_map.get(player)
                headshot = player_headshot_url(player_id) if player_id else None
                p_logo = logo_url_from_abbr(p_team, team_id_map) if p_team else None
                opp_logo = logo_url_from_abbr(opp_abbr, team_id_map) if (opp_abbr and opp_abbr != "—") else None

                conf = compute_confidence_from_last_n(df_logs, player, n)

                pts_ou = over_under_probabilities(
                    df=df_logs,
                    player_name=player,
                    n=n,
                    line=float(pts_line),
                    stat="PTS",
                    mu_override=float(adj_pts),
                    opp_abbr_for_sigma=opp_for_sigma,
                    h2h_sigma_min_games=4,
                )

                pra_ou = over_under_probabilities(
                    df=df_logs,
                    player_name=player,
                    n=n,
                    line=float(pra_line),
                    stat="PRA",
                    mu_override=float(adj_pra),
                    opp_abbr_for_sigma=opp_for_sigma,
                    h2h_sigma_min_games=4,
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        st.markdown("---")
        st.markdown("## Results")

        card_col1, card_col2 = st.columns([1, 4])

        with card_col1:
            if headshot:
                st.image(headshot, width=150)

        with card_col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-title">{player} Projection</div>
                    <div class="result-value">{adj_pts:.1f} PTS</div>
                    <div class="result-sub">PRA: {adj_pra:.1f}</div>
                    <div class="result-muted">Confidence: {conf if conf is not None else "—"}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Projected PTS", f"{adj_pts:.2f}")
        m2.metric("Projected PRA", f"{adj_pra:.2f}")
        m3.metric("PTS Over Chance", f"{pts_ou['prob_over'] * 100:.0f}%" if pts_ou.get("ok") else "—")
        m4.metric("Confidence", f"{conf}%" if conf is not None else "—")

        if pts_ou.get("ok"):
            verdict, status = verdict_from_prob(pts_ou["prob_over"])

            if status == "success":
                st.success(f"Model Verdict: {verdict}")
            elif status == "error":
                st.error(f"Model Verdict: {verdict}")
            else:
                st.warning(f"Model Verdict: {verdict}")

        st.markdown("### Why this projection")
        st.write(
            f"""
- Based on the last {n} games
- Matchup adjustment: {adj_note}
- Player consistency score: {conf if conf is not None else "N/A"}%
"""
        )

        st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)

        left_col, center_col, right_col = st.columns([1, 4, 1])

        with left_col:
            if p_logo:
                st.image(p_logo, width=72)
            elif p_team:
                st.write(p_team)

        with center_col:
            st.markdown("### Projection Details")
            st.caption("Base projection with optional matchup adjustment.")

        with right_col:
            if opp_logo:
                st.image(opp_logo, width=72)
            elif opp_abbr and opp_abbr != "—":
                st.write(opp_abbr)

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("PTS Base", f"{base_pts:.2f}")
        d2.metric("PTS Adj", f"{adj_pts:.2f}")
        d3.metric("REB", f"{base_reb:.2f}")
        d4.metric("AST", f"{base_ast:.2f}")
        d5.metric("PRA Adj", f"{adj_pra:.2f}")

        st.markdown("#### Matchup adjustment note")
        st.write(adj_note)

        st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Over / Under Probabilities")

        ou1, ou2 = st.columns(2)

        with ou1:
            st.markdown("**PTS Line**")
            st.caption("PTS = Points")

            if pts_ou.get("ok"):
                st.metric("Line", f"{pts_ou['line']:.1f}")
                st.metric("Prob OVER", f"{pts_ou['prob_over'] * 100:.0f}%")
                st.metric("Prob UNDER", f"{pts_ou['prob_under'] * 100:.0f}%")
                st.caption(f"mu={pts_ou['mu']:.2f} | sigma={pts_ou['sigma']:.2f}")
                st.caption(pts_ou["sigma_note"])
            else:
                st.info(pts_ou.get("error", "Could not compute PTS probabilities."))

        with ou2:
            st.markdown("**PRA Line**")
            st.caption("PRA = Points + Rebounds + Assists")

            if pra_ou.get("ok"):
                st.metric("Line", f"{pra_ou['line']:.1f}")
                st.metric("Prob OVER", f"{pra_ou['prob_over'] * 100:.0f}%")
                st.metric("Prob UNDER", f"{pra_ou['prob_under'] * 100:.0f}%")
                st.caption(f"mu={pra_ou['mu']:.2f} | sigma={pra_ou['sigma']:.2f}")
                st.caption(pra_ou["sigma_note"])
            else:
                st.info(pra_ou.get("error", "Could not compute PRA probabilities."))

        if opp_abbr and opp_abbr != "—":
            st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
            st.markdown("### Matchup History")

            hist = get_matchup_history(df_logs, player, opp_abbr, last_k=10)

            if hist.empty:
                st.info("No matchup history found for this player and opponent.")
            else:
                st.dataframe(hist, width="stretch", hide_index=True)

        st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Last 5 Games")

        last5 = last_games_table(df_logs, player, k=5)

        if last5.empty:
            st.info("No recent game logs found for that player in this dataset.")
        else:
            st.dataframe(last5, width="stretch", hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
<div class="courtiq-card">
  <div style="color:#dc2626; font-weight:700; font-size:0.95rem;">
    AI-generated projections for informational purposes only and should not be used for financial or wagering decisions.
  </div>
</div>
""",
    unsafe_allow_html=True,
)