from __future__ import annotations
# We use future annotations so Python can handle newer type hints more safely.

import math
# We import math so we can calculate the normal distribution probability.

import os
# We import os so we can check file modified times and environment-related paths.

from glob import glob
# We import glob so we can search for matching CSV files in the data/raw folder.

from pathlib import Path
# We import Path so we can work with file paths in a cleaner way.

from typing import Any
# We import Any so our dictionary type hints can store mixed values.

import pandas as pd
# We import pandas so we can load, clean, filter, and display NBA game log data.

import streamlit as st
# We import Streamlit so we can build the web app interface.

from src.courtiq.models.predict import predict_from_last_n
# We import our team prediction function that uses the trained ML models.


try:
    # We try to import NBA team data so we can pull official team IDs for logos.
    from nba_api.stats.static import teams as static_teams
except Exception:
    # We fall back to None if nba_api is unavailable so the app does not crash.
    static_teams = None


# -------------------------
# PAGE SETUP
# -------------------------

st.set_page_config(page_title="Court IQ", page_icon="🏀", layout="wide")
# We set the browser title, page icon, and wide layout for a presentation-ready app.

st.markdown(
    """
<style>
/* We style the full Streamlit app background and default text color. */
.stApp {
    background: #f4f7fb;
    color: #111827;
}

/* We reduce extra top and bottom spacing so the app feels tighter. */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* We center the main Court IQ header section. */
.courtiq-header {
    text-align: center;
    margin-bottom: 1.5rem;
}

/* We style the main Court IQ title. */
.courtiq-title {
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: 0.5px;
    color: #0f172a;
    margin-bottom: 0.2rem;
}

/* We make the IQ part of the title blue for branding. */
.courtiq-title span {
    color: #2563eb;
}

/* We style the short subtitle under the main title. */
.courtiq-subtitle {
    font-size: 1.05rem;
    color: #64748b;
    margin-bottom: 0.9rem;
}

/* We create a row for small feature badges. */
.courtiq-pill-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 1.25rem;
}

/* We style each feature badge under the title. */
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

/* We style the main hero banner card. */
.courtiq-hero {
    background: linear-gradient(135deg, #eff6ff 0%, #eef2ff 100%);
    border: 1px solid #dbeafe;
    border-radius: 22px;
    padding: 24px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
}

/* We style general white content cards. */
.courtiq-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}

/* We style section headings inside cards. */
.courtiq-section-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.35rem;
}

/* We style muted explanation text. */
.courtiq-muted {
    color: #64748b;
    font-size: 0.96rem;
}

/* We style the dark result card for the main prediction. */
.result-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 24px;
    border-radius: 20px;
    color: white;
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.18);
}

/* We style the small title inside the result card. */
.result-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #cbd5e1;
    margin-bottom: 0.25rem;
}

/* We style the large projected points value. */
.result-value {
    font-size: 2.8rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.45rem;
}

/* We style the PRA value inside the result card. */
.result-sub {
    color: #93c5fd;
    font-weight: 600;
    margin-bottom: 0.25rem;
}

/* We style muted result-card text. */
.result-muted {
    color: #94a3b8;
    font-size: 0.95rem;
}

/* We style divider lines between major app sections. */
.courtiq-divider {
    border-top: 1px solid #e2e8f0;
    margin: 16px 0;
}

/* We style Streamlit buttons to match Court IQ branding. */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 800 !important;
    border: none !important;
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18);
}

/* We style the button hover state for a polished feel. */
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
}

/* We make the title and result values smaller on mobile screens. */
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
# We inject custom CSS so the app looks more like a finished product than a default Streamlit page.

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
# We display the Court IQ branded header and feature badges.

st.caption("Powered by Machine Learning • NBA Game Logs • Player Prop Probability Analysis")
# We add a short credibility caption for the final presentation demo.


# -------------------------
# DATA HELPERS
# -------------------------

def newest_gamelog_csv() -> Path:
    # We define a helper function to find the best available NBA game log CSV file.

    preferred = Path("data/raw/player_gamelogs_2025_26_Regular_Season_nba_api.csv")
    # We first look for the preferred 2025-26 NBA regular season dataset.

    if preferred.exists():
        # We check whether the preferred dataset exists in the project folder.
        return preferred
        # We return the preferred dataset if it is available.

    files = glob("data/raw/player_gamelogs_*_Regular_Season*_nba_api.csv")
    # We search for any regular season NBA game log CSV files in data/raw.

    if not files:
        # We check whether no matching files were found.
        return Path("data/raw/player_gamelogs_2024_25_Regular_Season_nba_api.csv")
        # We return the older fallback dataset path if no newer file exists.

    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    # We sort matching files by most recently modified first.

    return Path(files[0])
    # We return the newest available CSV path.


DATA_PATH = newest_gamelog_csv()
# We store the selected dataset path so the rest of the app can use it.


@st.cache_data
def load_gamelogs(csv_path: Path) -> pd.DataFrame:
    # We cache this function so the CSV does not reload every time the page updates.

    df = pd.read_csv(csv_path)
    # We load the selected CSV into a pandas DataFrame.

    df.columns = [str(col).strip().upper() for col in df.columns]
    # We standardize all column names by trimming spaces and converting to uppercase.

    df = df.loc[:, ~df.columns.duplicated()].copy()
    # We remove duplicate columns to avoid pandas errors later in the app.

    rename_map = {}
    # We create a dictionary to store column names that need to be renamed.

    if "PLAYER" in df.columns:
        # We check for a player column named PLAYER.
        rename_map["PLAYER"] = "PLAYER_NAME"
        # We rename PLAYER to PLAYER_NAME.

    if "NAME" in df.columns:
        # We check for a player column named NAME.
        rename_map["NAME"] = "PLAYER_NAME"
        # We rename NAME to PLAYER_NAME.

    if "PLAYERNAME" in df.columns:
        # We check for a player column named PLAYERNAME.
        rename_map["PLAYERNAME"] = "PLAYER_NAME"
        # We rename PLAYERNAME to PLAYER_NAME.

    if "PLAYER_FULL_NAME" in df.columns:
        # We check for a player column named PLAYER_FULL_NAME.
        rename_map["PLAYER_FULL_NAME"] = "PLAYER_NAME"
        # We rename PLAYER_FULL_NAME to PLAYER_NAME.

    if "FULL_NAME" in df.columns:
        # We check for a player column named FULL_NAME.
        rename_map["FULL_NAME"] = "PLAYER_NAME"
        # We rename FULL_NAME to PLAYER_NAME.

    if "TEAM" in df.columns:
        # We check for a team abbreviation column named TEAM.
        rename_map["TEAM"] = "TEAM_ABBREVIATION"
        # We rename TEAM to TEAM_ABBREVIATION.

    if "TEAM_ABBR" in df.columns:
        # We check for a team abbreviation column named TEAM_ABBR.
        rename_map["TEAM_ABBR"] = "TEAM_ABBREVIATION"
        # We rename TEAM_ABBR to TEAM_ABBREVIATION.

    if "TEAM_ABBREV" in df.columns:
        # We check for a team abbreviation column named TEAM_ABBREV.
        rename_map["TEAM_ABBREV"] = "TEAM_ABBREVIATION"
        # We rename TEAM_ABBREV to TEAM_ABBREVIATION.

    if "TEAM_NAME" in df.columns:
        # We check for a team column named TEAM_NAME.
        rename_map["TEAM_NAME"] = "TEAM_ABBREVIATION"
        # We rename TEAM_NAME to TEAM_ABBREVIATION.

    if "TEAMCODE" in df.columns:
        # We check for a team column named TEAMCODE.
        rename_map["TEAMCODE"] = "TEAM_ABBREVIATION"
        # We rename TEAMCODE to TEAM_ABBREVIATION.

    if "OPPONENT" in df.columns:
        # We check for an opponent column named OPPONENT.
        rename_map["OPPONENT"] = "OPP_TEAM_ABBREVIATION"
        # We rename OPPONENT to OPP_TEAM_ABBREVIATION.

    if "OPP_TEAM" in df.columns:
        # We check for an opponent column named OPP_TEAM.
        rename_map["OPP_TEAM"] = "OPP_TEAM_ABBREVIATION"
        # We rename OPP_TEAM to OPP_TEAM_ABBREVIATION.

    if "OPP" in df.columns:
        # We check for an opponent column named OPP.
        rename_map["OPP"] = "OPP_TEAM_ABBREVIATION"
        # We rename OPP to OPP_TEAM_ABBREVIATION.

    df = df.rename(columns=rename_map)
    # We apply all column name changes to the DataFrame.

    if "MATCHUP" in df.columns:
        # We check whether the matchup column is available.

        matchup_series = df["MATCHUP"].astype(str).str.strip()
        # We convert matchup values to clean strings.

        if "TEAM_ABBREVIATION" not in df.columns:
            # We create the team abbreviation if it is missing.
            df["TEAM_ABBREVIATION"] = matchup_series.str.split().str[0]
            # We pull the team abbreviation from the first word in MATCHUP.

        if "OPP_TEAM_ABBREVIATION" not in df.columns:
            # We create the opponent abbreviation if it is missing.
            df["OPP_TEAM_ABBREVIATION"] = matchup_series.str.split().str[-1]
            # We pull the opponent abbreviation from the last word in MATCHUP.

    for col in ["PLAYER_NAME", "TEAM_ABBREVIATION", "MATCHUP", "OPP_TEAM_ABBREVIATION"]:
        # We loop through text columns that need cleaning.
        if col in df.columns:
            # We only clean the column if it exists.
            df[col] = df[col].astype(str).str.strip()
            # We convert values to strings and remove extra spaces.

    if "PLAYER_NAME" in df.columns:
        # We check whether player names exist.
        df["PLAYER_NAME"] = df["PLAYER_NAME"].replace({"nan": None, "None": None})
        # We replace fake string missing values with real missing values.

    if "TEAM_ABBREVIATION" in df.columns:
        # We check whether team abbreviations exist.
        df["TEAM_ABBREVIATION"] = (
            df["TEAM_ABBREVIATION"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": None, "NONE": None})
        )
        # We clean team abbreviations and convert them to uppercase.

    if "OPP_TEAM_ABBREVIATION" in df.columns:
        # We check whether opponent abbreviations exist.
        df["OPP_TEAM_ABBREVIATION"] = (
            df["OPP_TEAM_ABBREVIATION"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": None, "NONE": None})
        )
        # We clean opponent abbreviations and convert them to uppercase.

    if "GAME_DATE" in df.columns:
        # We check whether game dates exist.
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        # We convert game dates into pandas datetime format.

    stat_rename_map = {}
    # We create a dictionary for stat column name fixes.

    if "POINTS" in df.columns:
        # We check for a full POINTS column name.
        stat_rename_map["POINTS"] = "PTS"
        # We rename POINTS to PTS.

    if "REBOUNDS" in df.columns:
        # We check for a full REBOUNDS column name.
        stat_rename_map["REBOUNDS"] = "REB"
        # We rename REBOUNDS to REB.

    if "ASSISTS" in df.columns:
        # We check for a full ASSISTS column name.
        stat_rename_map["ASSISTS"] = "AST"
        # We rename ASSISTS to AST.

    if "MINUTES" in df.columns:
        # We check for a full MINUTES column name.
        stat_rename_map["MINUTES"] = "MIN"
        # We rename MINUTES to MIN.

    df = df.rename(columns=stat_rename_map)
    # We apply the stat column renaming.

    for col in ["PTS", "REB", "AST", "MIN"]:
        # We loop through key numeric stat columns.
        if col in df.columns:
            # We only convert the stat column if it exists.
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # We convert values to numbers and turn invalid values into NaN.

    return df
    # We return the cleaned game log DataFrame.


@st.cache_data
def build_team_id_map() -> dict[str, int]:
    # We cache the team ID map so it does not reload repeatedly.

    if static_teams is None:
        # We check whether nba_api team data failed to import.
        return {}
        # We return an empty dictionary if team data is unavailable.

    all_teams = static_teams.get_teams()
    # We load all NBA team information from nba_api.

    return {
        t["abbreviation"]: int(t["id"])
        for t in all_teams
        if "abbreviation" in t and "id" in t
    }
    # We return a map like {"LAL": 1610612747} for team logo URLs.


@st.cache_data
def build_player_id_map(df: pd.DataFrame) -> dict[str, int]:
    # We cache player IDs so headshot links load faster.

    if df is None or df.empty:
        # We check if the DataFrame is missing or empty.
        return {}
        # We return an empty map when there is no data.

    temp = df.loc[:, ~df.columns.duplicated()].copy()
    # We copy the DataFrame and remove duplicate columns.

    if "PLAYER_NAME" not in temp.columns or "PLAYER_ID" not in temp.columns:
        # We check whether the needed player name and ID columns exist.
        return {}
        # We return an empty map if player IDs are unavailable.

    temp = temp[["PLAYER_NAME", "PLAYER_ID"]].dropna().copy()
    # We keep only player name and ID rows that are not missing.

    temp["PLAYER_NAME"] = temp["PLAYER_NAME"].astype(str).str.strip()
    # We clean player names.

    if isinstance(temp["PLAYER_ID"], pd.DataFrame):
        # We handle a rare case where PLAYER_ID becomes a duplicate-column DataFrame.
        temp["PLAYER_ID"] = temp["PLAYER_ID"].iloc[:, 0]
        # We keep only the first PLAYER_ID column.

    temp["PLAYER_ID"] = pd.to_numeric(temp["PLAYER_ID"], errors="coerce")
    # We convert player IDs to numeric values.

    temp = temp.dropna(subset=["PLAYER_ID"]).copy()
    # We remove rows where player ID conversion failed.

    temp["PLAYER_ID"] = temp["PLAYER_ID"].astype(int)
    # We convert player IDs to integers for URL formatting.

    temp = temp.drop_duplicates(subset=["PLAYER_NAME"], keep="last")
    # We keep the latest player ID row for each player name.

    return dict(zip(temp["PLAYER_NAME"], temp["PLAYER_ID"]))
    # We return a dictionary mapping player names to NBA player IDs.


def logo_url_from_team_id(team_id: int) -> str:
    # We define a helper to build an NBA official team logo URL.

    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.png"
    # We return the logo URL based on the NBA team ID.


def logo_url_from_abbr(team_abbr: str, team_id_map: dict[str, int]) -> str | None:
    # We define a helper to get a team logo from a team abbreviation.

    if not team_abbr:
        # We check whether the team abbreviation is missing.
        return None
        # We return no logo if there is no abbreviation.

    abbr = team_abbr.upper().strip()
    # We clean and uppercase the team abbreviation.

    if abbr in team_id_map:
        # We check if nba_api has an official team ID for this abbreviation.
        return logo_url_from_team_id(team_id_map[abbr])
        # We return the official NBA logo if available.

    return f"https://a.espncdn.com/i/teamlogos/nba/500/{abbr.lower()}.png"
    # We fall back to ESPN logo URLs if nba_api team IDs are unavailable.


def player_headshot_url(player_id: int) -> str:
    # We define a helper to build an NBA player headshot URL.

    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
    # We return the player headshot URL based on the NBA player ID.


def get_player_team_abbr(df: pd.DataFrame, player_name: str) -> str | None:
    # We define a helper to find a player's latest team abbreviation.

    if "PLAYER_NAME" not in df.columns or "TEAM_ABBREVIATION" not in df.columns:
        # We check that the needed columns exist.
        return None
        # We return None if we cannot find team information.

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()
    # We filter the dataset to only the selected player.

    if sub.empty:
        # We check whether no rows were found for that player.
        return None
        # We return None if the player is missing.

    if "GAME_DATE" in sub.columns:
        # We check whether game dates exist.
        sub = sub.sort_values("GAME_DATE")
        # We sort the player's games from oldest to newest.

    return str(sub.iloc[-1]["TEAM_ABBREVIATION"])
    # We return the team abbreviation from the player's most recent game.


def compute_confidence_from_last_n(df: pd.DataFrame, player_name: str, n: int) -> int | None:
    # We define a simple confidence score based on recent scoring consistency.

    if "PLAYER_NAME" not in df.columns or "PTS" not in df.columns:
        # We check that player names and points exist.
        return None
        # We return None if confidence cannot be calculated.

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()
    # We filter the dataset to the selected player.

    if sub.empty:
        # We check whether the player has no data.
        return None
        # We return None if there is no player data.

    if "GAME_DATE" in sub.columns:
        # We check whether game dates exist.
        sub = sub.sort_values("GAME_DATE")
        # We sort games from oldest to newest.

    last_n = sub.tail(n)
    # We select the player's most recent N games.

    pts = pd.to_numeric(last_n.get("PTS"), errors="coerce").dropna()
    # We convert recent point totals to numbers and remove missing values.

    if pts.empty:
        # We check whether there are no valid point values.
        return None
        # We return None if confidence cannot be calculated.

    pts_std = float(pts.std(ddof=0)) if len(pts) > 1 else 0.0
    # We calculate the standard deviation of recent points as a consistency measure.

    conf = 95 - (pts_std * 4.5)
    # We lower confidence when the player's points vary a lot.

    conf = max(10, min(95, conf))
    # We keep confidence between 10% and 95%.

    return int(round(conf))
    # We return the rounded confidence score.


def parse_opponent_from_matchup(matchup: str, team_abbr: str) -> str | None:
    # We define a helper to pull opponent abbreviation from a MATCHUP string.

    if not matchup or not team_abbr:
        # We check whether the matchup or team abbreviation is missing.
        return None
        # We return None when opponent parsing is not possible.

    parts = str(matchup).strip().split()
    # We split the matchup string into parts.

    if len(parts) < 3:
        # We check whether the matchup string has enough pieces.
        return None
        # We return None if the matchup format is not usable.

    opp = parts[-1].strip().upper()
    # We use the last part of the matchup string as the opponent abbreviation.

    team_abbr = str(team_abbr).strip().upper()
    # We clean the player's team abbreviation.

    if opp == team_abbr:
        # We check whether the parsed opponent is accidentally the same as the team.
        return None
        # We return None if opponent parsing looks wrong.

    return opp
    # We return the opponent abbreviation.


def get_matchup_history(df: pd.DataFrame, player_name: str, opp_abbr: str, last_k: int = 10) -> pd.DataFrame:
    # We define a helper to find a player's recent history against a selected opponent.

    if "PLAYER_NAME" not in df.columns:
        # We check whether player names exist.
        return pd.DataFrame()
        # We return an empty DataFrame if no player column exists.

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()
    # We filter the dataset to the selected player.

    if sub.empty:
        # We check whether no player rows were found.
        return pd.DataFrame()
        # We return an empty DataFrame if the player is missing.

    if "OPP_TEAM_ABBREVIATION" in sub.columns:
        # We use the direct opponent column if it exists.
        sub["OPP"] = sub["OPP_TEAM_ABBREVIATION"].astype(str).str.upper()
        # We create a clean OPP column from opponent abbreviations.

    elif "MATCHUP" in sub.columns and "TEAM_ABBREVIATION" in sub.columns:
        # We use MATCHUP parsing if the direct opponent column does not exist.
        sub["OPP"] = sub.apply(
            lambda r: parse_opponent_from_matchup(r.get("MATCHUP"), r.get("TEAM_ABBREVIATION")),
            axis=1,
        )
        # We create the OPP column by parsing each matchup row.

    else:
        # We handle the case where opponent information is unavailable.
        return pd.DataFrame()
        # We return an empty DataFrame if opponent context cannot be built.

    opp_abbr = str(opp_abbr).upper().strip()
    # We clean the selected opponent abbreviation.

    sub = sub[sub["OPP"] == opp_abbr].copy()
    # We keep only games against the selected opponent.

    if sub.empty:
        # We check whether no head-to-head games were found.
        return pd.DataFrame()
        # We return an empty DataFrame if no matchup history exists.

    if "GAME_DATE" in sub.columns:
        # We check whether game dates exist.
        sub = sub.sort_values("GAME_DATE")
        # We sort head-to-head games from oldest to newest.

    cols = []
    # We create a list of columns we want to show.

    for c in ["GAME_DATE", "MATCHUP", "PTS", "REB", "AST", "MIN"]:
        # We loop through useful matchup history columns.
        if c in sub.columns:
            # We only include columns that exist.
            cols.append(c)
            # We add the column to the output list.

    out = sub[cols].tail(last_k).copy()
    # We keep only the most recent matchup rows.

    if "GAME_DATE" in out.columns:
        # We check whether dates exist in the output.
        out["GAME_DATE"] = out["GAME_DATE"].dt.date
        # We display only the date portion for readability.

    if all(c in out.columns for c in ["PTS", "REB", "AST"]):
        # We check whether we can calculate PRA.
        out["PRA"] = (
            pd.to_numeric(out["PTS"], errors="coerce")
            + pd.to_numeric(out["REB"], errors="coerce")
            + pd.to_numeric(out["AST"], errors="coerce")
        )
        # We calculate PRA as points plus rebounds plus assists.

    return out
    # We return the matchup history table.


def normal_cdf(z: float) -> float:
    # We define a helper for the normal cumulative distribution function.

    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    # We calculate the probability below a z-score.


def normal_prob_over(mu: float, sigma: float, line: float) -> float:
    # We define a helper to estimate the probability of going over a betting line.

    if sigma <= 1e-9:
        # We check whether sigma is too close to zero.
        return 1.0 if mu > line else 0.0
        # We return a simple over/under result when there is no variation.

    z = (line - mu) / sigma
    # We convert the betting line into a z-score.

    p_under = normal_cdf(z)
    # We estimate the probability of going under the line.

    return float(1.0 - p_under)
    # We return the probability of going over the line.


def last_n_series(df: pd.DataFrame, player_name: str, col: str, n: int) -> pd.Series:
    # We define a helper to get a player's last N values for one stat column.

    if "PLAYER_NAME" not in df.columns or col not in df.columns:
        # We check whether the needed columns exist.
        return pd.Series(dtype=float)
        # We return an empty series if the stat cannot be found.

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()
    # We filter rows for the selected player.

    if sub.empty:
        # We check whether no player rows were found.
        return pd.Series(dtype=float)
        # We return an empty series if the player is missing.

    if "GAME_DATE" in sub.columns:
        # We check whether game dates exist.
        sub = sub.sort_values("GAME_DATE")
        # We sort games from oldest to newest.

    s = pd.to_numeric(sub[col], errors="coerce").dropna().tail(n)
    # We convert the selected stat to numbers and keep the last N valid values.

    return s
    # We return the recent stat series.


def last_n_pra_series(df: pd.DataFrame, player_name: str, n: int) -> pd.Series:
    # We define a helper to calculate a player's last N PRA values.

    pts = last_n_series(df, player_name, "PTS", n)
    # We get the player's last N points.

    reb = last_n_series(df, player_name, "REB", n)
    # We get the player's last N rebounds.

    ast = last_n_series(df, player_name, "AST", n)
    # We get the player's last N assists.

    if pts.empty or reb.empty or ast.empty:
        # We check whether any stat series is missing.
        return pd.Series(dtype=float)
        # We return an empty series if PRA cannot be calculated.

    m = min(len(pts), len(reb), len(ast))
    # We use the shortest available length so all stats line up.

    pts = pts.tail(m).reset_index(drop=True)
    # We reset point values to a clean index.

    reb = reb.tail(m).reset_index(drop=True)
    # We reset rebound values to a clean index.

    ast = ast.tail(m).reset_index(drop=True)
    # We reset assist values to a clean index.

    return (pts + reb + ast).dropna()
    # We return PRA values by adding points, rebounds, and assists.


def sigma_with_fallback(
    df: pd.DataFrame,
    player_name: str,
    n: int,
    stat: str,
    opp_abbr_for_sigma: str | None,
    h2h_sigma_min_games: int = 4,
    safe_fallback_sigma: float = 6.0,
) -> tuple[float, str]:
    # We define a helper to choose a reliable standard deviation for probability estimates.

    if opp_abbr_for_sigma:
        # We check whether an opponent was selected for matchup-specific variation.
        hist = get_matchup_history(df, player_name, opp_abbr_for_sigma, last_k=50)
        # We get up to 50 head-to-head games for sigma calculation.

        if not hist.empty:
            # We check whether matchup history exists.
            if stat == "PRA" and "PRA" in hist.columns:
                # We use PRA values when the selected stat is PRA.
                vals = pd.to_numeric(hist["PRA"], errors="coerce").dropna()
                # We convert matchup PRA values to numbers.

            else:
                # We use the selected stat column for non-PRA stats.
                vals = pd.to_numeric(hist.get(stat), errors="coerce").dropna()
                # We convert matchup stat values to numbers.

            if len(vals) >= h2h_sigma_min_games:
                # We only trust matchup sigma if we have enough games.
                sigma = float(vals.std(ddof=0)) if len(vals) > 1 else safe_fallback_sigma
                # We calculate matchup standard deviation or use a fallback.

                sigma = max(1.0, sigma)
                # We prevent sigma from being too small.

                return sigma, f"Sigma used H2H vs {opp_abbr_for_sigma} (n={len(vals)})"
                # We return matchup sigma and an explanation note.

    if stat == "PRA":
        # We check whether the stat is PRA.
        s = last_n_pra_series(df, player_name, n)
        # We get recent PRA values.

    else:
        # We handle non-PRA stats.
        s = last_n_series(df, player_name, stat, n)
        # We get recent values for the selected stat.

    if len(s) >= 2:
        # We check whether we have at least two games for standard deviation.
        sigma = float(s.std(ddof=0))
        # We calculate recent standard deviation.

        sigma = max(1.0, sigma)
        # We prevent sigma from being too small.

        return sigma, f"Sigma used last {n} games (n={len(s)})"
        # We return recent-game sigma and a note.

    return safe_fallback_sigma, "Sigma used safe fallback (not enough game data)"
    # We return a safe fallback sigma if there is not enough data.


def matchup_adjusted_pts(df: pd.DataFrame, player_name: str, opp_abbr: str, base_pts: float) -> tuple[float, str]:
    # We define a helper to adjust projected points using matchup history.

    opp_abbr = str(opp_abbr).upper().strip()
    # We clean the selected opponent abbreviation.

    hist = get_matchup_history(df, player_name, opp_abbr, last_k=20)
    # We get the player's recent games against the selected opponent.

    if hist.empty or "PTS" not in hist.columns:
        # We check whether matchup history points are unavailable.
        return float(base_pts), "No matchup rows found. Using base PTS."
        # We return the base projection if matchup data is missing.

    h2h_pts = pd.to_numeric(hist["PTS"], errors="coerce").dropna()
    # We convert head-to-head points to numbers.

    if h2h_pts.empty:
        # We check whether no valid head-to-head point values exist.
        return float(base_pts), "No valid H2H PTS found. Using base PTS."
        # We return the base projection if H2H points are invalid.

    last_pts = last_n_series(df, player_name, "PTS", n=10)
    # We get the player's last 10 point totals for recent form.

    if last_pts.empty:
        # We check whether recent point values are missing.
        return float(base_pts), "No recent PTS series found. Using base PTS."
        # We return the base projection if recent points are unavailable.

    h2h_mean = float(h2h_pts.mean())
    # We calculate average points against the selected opponent.

    recent_mean = float(last_pts.mean())
    # We calculate average points across recent games.

    delta = h2h_mean - recent_mean
    # We calculate the difference between matchup average and recent average.

    capped_delta = max(-4.0, min(4.0, delta))
    # We cap the adjustment so one matchup does not over-shift the prediction.

    adj = float(base_pts + capped_delta)
    # We add the capped matchup adjustment to the base prediction.

    explanation = (
        f"H2H avg PTS ({h2h_mean:.1f}) vs recent avg PTS ({recent_mean:.1f}) "
        f"-> delta {delta:+.1f}, capped to {capped_delta:+.1f}. "
        f"Shifted mu to adjusted PTS."
    )
    # We build a readable explanation for the matchup adjustment.

    return adj, explanation
    # We return the adjusted points and explanation.


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
    # We define a helper to calculate over/under probabilities for a stat line.

    try:
        # We try to convert the prop line to a number.
        line = float(line)
    except Exception:
        # We handle invalid line values.
        return {"ok": False, "error": "Line must be a number."}
        # We return an error response if the line is not numeric.

    if mu_override is not None:
        # We check whether a predicted mean was provided from the ML model.
        mu = float(mu_override)
        # We use the model projection as the mean.
        mu_source = "mu_override"
        # We label the mean source as a model override.

    else:
        # We calculate the mean from recent data if no model projection is provided.
        if stat == "PRA":
            # We check whether the selected stat is PRA.
            s = last_n_pra_series(df, player_name, n)
            # We get recent PRA values.

        else:
            # We handle non-PRA stats.
            s = last_n_series(df, player_name, stat, n)
            # We get recent values for the selected stat.

        if s.empty:
            # We check whether no recent stat data is available.
            return {"ok": False, "error": f"No {stat} data found for {player_name}."}
            # We return an error response if the stat is missing.

        mu = float(s.mean())
        # We calculate the recent mean for the stat.

        mu_source = f"last_{n}_mean"
        # We label the mean source as a recent-game average.

    sigma, sigma_note = sigma_with_fallback(
        df=df,
        player_name=player_name,
        n=n,
        stat=stat,
        opp_abbr_for_sigma=opp_abbr_for_sigma,
        h2h_sigma_min_games=h2h_sigma_min_games,
        safe_fallback_sigma=6.0 if stat == "PTS" else 8.0,
    )
    # We calculate a standard deviation for the probability model.

    p_over = normal_prob_over(mu=mu, sigma=sigma, line=line)
    # We estimate the probability of going over the prop line.

    p_under = 1.0 - p_over
    # We estimate the probability of going under the prop line.

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
    # We return all probability outputs in one dictionary.


def last_games_table(df: pd.DataFrame, player_name: str, k: int = 5) -> pd.DataFrame:
    # We define a helper to build the player's last K games table.

    if "PLAYER_NAME" not in df.columns:
        # We check whether player names exist.
        return pd.DataFrame()
        # We return an empty DataFrame if the player column is missing.

    sub = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()
    # We filter the game logs to the selected player.

    if sub.empty:
        # We check whether the player has no rows.
        return pd.DataFrame()
        # We return an empty DataFrame if there is no data.

    if "GAME_DATE" in sub.columns:
        # We check whether game dates exist.
        sub = sub.sort_values("GAME_DATE")
        # We sort the player's games from oldest to newest.

    sub = sub.tail(k).copy()
    # We keep only the most recent K games.

    if "OPP_TEAM_ABBREVIATION" in sub.columns:
        # We use the direct opponent column if available.
        sub["OPP"] = sub["OPP_TEAM_ABBREVIATION"].astype(str).str.upper()
        # We create a clean opponent column.

    elif "MATCHUP" in sub.columns and "TEAM_ABBREVIATION" in sub.columns:
        # We parse opponent from MATCHUP if no direct opponent column exists.
        sub["OPP"] = sub.apply(
            lambda r: parse_opponent_from_matchup(r.get("MATCHUP"), r.get("TEAM_ABBREVIATION")),
            axis=1,
        )
        # We create opponent values from matchup strings.

    else:
        # We handle missing opponent data.
        sub["OPP"] = ""
        # We use an empty opponent value if opponent context is unavailable.

    out_cols = []
    # We create a list for table columns.

    if "GAME_DATE" in sub.columns:
        # We check if game date is available.
        out_cols.append("GAME_DATE")
        # We include the game date column.

    if "MATCHUP" in sub.columns:
        # We check if matchup is available.
        out_cols.append("MATCHUP")
        # We include the matchup column.

    if "OPP" in sub.columns:
        # We check if opponent is available.
        out_cols.append("OPP")
        # We include the opponent column.

    for c in ["PTS", "REB", "AST", "MIN"]:
        # We loop through key player stats.
        if c in sub.columns:
            # We check whether each stat exists.
            out_cols.append(c)
            # We include the stat in the output table.

    out = sub[out_cols].copy()
    # We create a clean table with selected columns.

    if all(c in out.columns for c in ["PTS", "REB", "AST"]):
        # We check whether we can calculate PRA.
        out["PRA"] = (
            pd.to_numeric(out["PTS"], errors="coerce")
            + pd.to_numeric(out["REB"], errors="coerce")
            + pd.to_numeric(out["AST"], errors="coerce")
        )
        # We calculate PRA as points plus rebounds plus assists.

    if "GAME_DATE" in out.columns:
        # We check whether game date exists in the table.
        out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce").dt.date
        # We format game date as a date instead of full datetime.

    preferred = ["GAME_DATE", "MATCHUP", "OPP", "PTS", "REB", "AST", "PRA", "MIN"]
    # We define the preferred display order for the table.

    final_cols = [c for c in preferred if c in out.columns]
    # We keep only preferred columns that actually exist.

    return out[final_cols].sort_values("GAME_DATE", ascending=False)
    # We return the recent games table sorted newest first.


def verdict_from_prob(prob_over: float) -> tuple[str, str]:
    # We define a helper to turn probability into a simple model verdict.

    if prob_over >= 0.60:
        # We treat 60% or higher as a strong over lean.
        return "Strong OVER", "success"
        # We return an over verdict and success status.

    if prob_over <= 0.40:
        # We treat 40% or lower over probability as a strong under lean.
        return "Strong UNDER", "error"
        # We return an under verdict and error status for red styling.

    return "No clear edge", "warning"
    # We return a neutral verdict when probability is close to balanced.


# -------------------------
# LOAD DATA
# -------------------------

df_logs = None
# We start with no game log DataFrame loaded.

team_id_map: dict[str, int] = {}
# We create an empty team logo ID map.

player_id_map: dict[str, int] = {}
# We create an empty player ID map for headshots.

if DATA_PATH.exists():
    # We check whether the selected dataset file exists.
    df_logs = load_gamelogs(DATA_PATH)
    # We load and clean the game log dataset.

    team_id_map = build_team_id_map()
    # We build the team ID map for NBA logos.

    player_id_map = build_player_id_map(df_logs)
    # We build the player ID map for player headshots.

else:
    # We handle the case where the dataset is missing.
    st.warning(f"Could not find gamelog file: {DATA_PATH}")
    # We show a warning message in the app.


# -------------------------
# HERO / INTRO
# -------------------------

st.markdown(
    f"""
<div class="courtiq-hero">
  <div class="courtiq-section-title">Smarter NBA prop insights in seconds</div>
  <div class="courtiq-muted" style="margin-bottom:10px;">
    Use recent game trends, matchup context, and probability estimates to make faster player prop decisions.
  </div>
  <div class="courtiq-muted">
    Current dataset: <b>{DATA_PATH.name}</b>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
# We display a hero banner explaining the main value of Court IQ.

st.markdown(
    """
<div class="courtiq-card">
  <div class="courtiq-section-title">How does Court IQ work?</div>
  <div class="courtiq-muted">
    Court IQ uses machine learning models trained on real NBA player game logs.
    It analyzes recent player performance, matchup context, and player consistency
    to generate projected stats, over/under probabilities, and confidence scores.
    <br/><br/>
    <b>Demo Flow:</b> Select a player → choose recent game sample → enter prop lines → generate prediction.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
# We explain the app workflow in presentation-friendly language.

st.info("Tip: Look for confidence above 70% for more stable players.")
# We display a quick user tip about confidence.


# -------------------------
# SINGLE PLAYER PREDICTION
# -------------------------

st.markdown('<div class="courtiq-card">', unsafe_allow_html=True)
# We open a styled card for the prediction controls.

st.markdown('<div class="courtiq-section-title">Single Player Prediction</div>', unsafe_allow_html=True)
# We display the prediction section title.

st.markdown(
    '<div class="courtiq-muted">Analyze one player using recent performance, matchup history, and projected over/under probability.</div>',
    unsafe_allow_html=True,
)
# We describe what the single-player prediction section does.

players_list: list[str] = []
# We start with an empty list of player options.

default_index = 0
# We set the default dropdown index to the first player.

if df_logs is not None and "PLAYER_NAME" in df_logs.columns:
    # We check whether player data is loaded.
    players_list = sorted(df_logs["PLAYER_NAME"].dropna().unique().tolist())
    # We create a sorted list of unique player names.

    if "Kevin Durant" in players_list:
        # We check whether Kevin Durant is in the dataset for an easy demo default.
        default_index = players_list.index("Kevin Durant")
        # We set Kevin Durant as the default selected player.

left_input, right_input = st.columns([2, 1])
# We create two input columns, with the player selector wider than the slider.

with left_input:
    # We place the player input inside the left column.
    if players_list:
        # We use a dropdown if player names are available.
        player = st.selectbox(
            "Select player",
            options=players_list,
            index=default_index,
            help="Choose the player you want to analyze.",
        )
        # We let the user select a player from the dataset.

    else:
        # We use a manual text input if the player list is unavailable.
        player = st.text_input(
            "Player name",
            value="Kevin Durant",
            help="Enter the player you want to analyze.",
        )
        # We let the user type a player name manually.

with right_input:
    # We place the recent game slider inside the right column.
    n = st.slider(
        "Last N games",
        min_value=1,
        max_value=10,
        value=5,
        help="How many recent games to use in the projection.",
    )
    # We let the user choose how many recent games should influence the prediction.

opp_abbr = None
# We start with no opponent selected.

teams_list: list[str] = []
# We start with an empty team list.

if df_logs is not None and "TEAM_ABBREVIATION" in df_logs.columns:
    # We check whether team abbreviations exist in the dataset.
    teams_list = sorted(df_logs["TEAM_ABBREVIATION"].dropna().unique().tolist())
    # We build a sorted list of team abbreviations.

    opp_abbr = st.selectbox(
        "Opponent (optional, for matchup history + adjusted PTS)",
        options=["—"] + teams_list,
        help="Optional opponent selection for matchup-adjusted points and head-to-head history.",
    )
    # We let the user choose an optional opponent for matchup context.

c_line1, c_line2 = st.columns(2)
# We create two columns for PTS and PRA line inputs.

with c_line1:
    # We place the points line input in the first column.
    pts_line = st.number_input(
        "PTS line (for Over/Under probability)",
        min_value=0.0,
        value=13.5,
        step=0.5,
        help="Betting line for points.",
    )
    # We let the user enter the points prop line.

with c_line2:
    # We place the PRA line input in the second column.
    pra_line = st.number_input(
        "PRA line (for Over/Under probability)",
        min_value=0.0,
        value=25.5,
        step=0.5,
        help="Points + Rebounds + Assists line.",
    )
    # We let the user enter the PRA prop line.

st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
# We add a visual divider before the prediction button.

note_col, btn_col = st.columns([3, 1])
# We create a note column and a button column.

with note_col:
    # We place the note in the wider left column.
    st.caption("Tip: Higher confidence usually means the player has been more consistent recently.")
    # We remind users what confidence means.

with btn_col:
    # We place the prediction button in the right column.
    do_predict = st.button("🚀 Generate Prediction", use_container_width=True)
    # We create the main button that starts the prediction process.

if do_predict:
    # We run the prediction logic only after the user clicks the button.

    if df_logs is None:
        # We check whether the dataset failed to load.
        st.error("Gamelog dataset is missing. Add a CSV into data/raw and re-run the app.")
        # We show an error if the dataset is unavailable.

    else:
        # We continue if the dataset is available.
        try:
            # We wrap prediction logic in a try block so the demo does not crash.

            with st.spinner("Analyzing player data..."):
                # We show a spinner while the app calculates the prediction.

                result = predict_from_last_n(player_name=player, n=n)
                # We call our machine learning prediction function.

                if not result.get("ok", True):
                    # We check whether the prediction function returned an error response.
                    st.error(result.get("error", "Prediction failed."))
                    # We show the model error message in the app.
                    st.stop()
                    # We stop the current run safely.

                base_pts = float(result.get("predicted_points", 0.0))
                # We store the base predicted points.

                base_reb = float(result.get("predicted_rebounds", 0.0))
                # We store the base predicted rebounds.

                base_ast = float(result.get("predicted_assists", 0.0))
                # We store the base predicted assists.

                adj_pts = base_pts
                # We start adjusted points equal to base predicted points.

                adj_note = "No opponent selected. Using base PTS."
                # We set the default matchup explanation.

                opp_for_sigma = None
                # We set no opponent sigma context by default.

                if opp_abbr and opp_abbr != "—":
                    # We check whether an opponent was selected.
                    adj_pts, adj_note = matchup_adjusted_pts(df_logs, player, opp_abbr, base_pts)
                    # We adjust projected points using matchup history.

                    opp_for_sigma = opp_abbr
                    # We use the selected opponent when calculating probability variation.

                adj_pra = float(adj_pts + base_reb + base_ast)
                # We calculate adjusted PRA from adjusted points, rebounds, and assists.

                p_team = get_player_team_abbr(df_logs, player)
                # We find the player's latest team abbreviation.

                player_id = player_id_map.get(player)
                # We get the player's NBA ID for headshot display.

                headshot = player_headshot_url(player_id) if player_id else None
                # We build a player headshot URL if an ID exists.

                p_logo = logo_url_from_abbr(p_team, team_id_map) if p_team else None
                # We build the player's team logo URL.

                opp_logo = logo_url_from_abbr(opp_abbr, team_id_map) if (opp_abbr and opp_abbr != "—") else None
                # We build the opponent team logo URL if an opponent is selected.

                conf = compute_confidence_from_last_n(df_logs, player, n)
                # We calculate the player's consistency-based confidence score.

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
                # We calculate over/under probabilities for the PTS line.

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
                # We calculate over/under probabilities for the PRA line.

        except Exception as e:
            # We catch unexpected errors so the app can show a friendly message.
            st.error(f"Prediction error: {e}")
            # We display the error message for debugging.
            st.stop()
            # We stop the app run safely after the error.

        st.markdown("---")
        # We add a Streamlit divider before results.

        st.markdown("## Results")
        # We display the results heading.

        card_col1, card_col2 = st.columns([1, 4])
        # We create columns for player image and result card.

        with card_col1:
            # We place the player headshot in the left column.
            if headshot:
                # We check whether a headshot URL exists.
                st.image(headshot, width=150)
                # We display the player headshot.

        with card_col2:
            # We place the main result card in the wider right column.
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
            # We display the main prediction card with adjusted points, PRA, and confidence.

        m1, m2, m3, m4 = st.columns(4)
        # We create four metric cards for quick result scanning.

        m1.metric("Projected PTS", f"{adj_pts:.2f}")
        # We show adjusted projected points.

        m2.metric("Projected PRA", f"{adj_pra:.2f}")
        # We show adjusted projected PRA.

        m3.metric("PTS Over Chance", f"{pts_ou['prob_over']*100:.0f}%" if pts_ou.get("ok") else "—")
        # We show the probability of going over the PTS line.

        m4.metric("Confidence", f"{conf}%" if conf is not None else "—")
        # We show the consistency confidence score.

        if pts_ou.get("ok"):
            # We check whether the PTS probability calculation succeeded.
            verdict, status = verdict_from_prob(pts_ou["prob_over"])
            # We convert the over probability into a simple verdict.

            if status == "success":
                # We check whether the verdict is a strong over.
                st.success(f"Model Verdict: {verdict}")
                # We display the verdict in a green success box.

            elif status == "error":
                # We check whether the verdict is a strong under.
                st.error(f"Model Verdict: {verdict}")
                # We display the verdict in a red error box.

            else:
                # We handle the neutral verdict case.
                st.warning(f"Model Verdict: {verdict}")
                # We display the verdict in a yellow warning box.

        st.markdown("### Why this projection")
        # We display a heading explaining the model reasoning.

        st.write(
            f"""
- Based on the last {n} games
- Matchup adjustment: {adj_note}
- Player consistency score: {conf if conf is not None else "N/A"}%
"""
        )
        # We show the key reasoning points in plain language.

        st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
        # We add a divider before projection details.

        left_col, center_col, right_col = st.columns([1, 4, 1])
        # We create a row for team logos and projection details.

        with left_col:
            # We place the player's team logo on the left.
            if p_logo:
                # We check whether the player's team logo exists.
                st.image(p_logo, width=72)
                # We display the player's team logo.

            elif p_team:
                # We check whether we at least have a team abbreviation.
                st.write(p_team)
                # We display the team abbreviation if no logo is available.

        with center_col:
            # We place projection detail text in the center.
            st.markdown("### Projection Details")
            # We show the projection details heading.

            st.caption("Base projection with optional matchup adjustment.")
            # We explain that projections may include matchup adjustments.

        with right_col:
            # We place opponent information on the right.
            if opp_logo:
                # We check whether the opponent logo exists.
                st.image(opp_logo, width=72)
                # We display the opponent logo.

            elif opp_abbr and opp_abbr != "—":
                # We check whether an opponent abbreviation exists.
                st.write(opp_abbr)
                # We display the opponent abbreviation if no logo is available.

        d1, d2, d3, d4, d5 = st.columns(5)
        # We create five detailed metric columns.

        d1.metric("PTS Base", f"{base_pts:.2f}")
        # We show the original model points projection.

        d2.metric("PTS Adj", f"{adj_pts:.2f}")
        # We show the matchup-adjusted points projection.

        d3.metric("REB", f"{base_reb:.2f}")
        # We show the model rebounds projection.

        d4.metric("AST", f"{base_ast:.2f}")
        # We show the model assists projection.

        d5.metric("PRA Adj", f"{adj_pra:.2f}")
        # We show the adjusted PRA projection.

        st.markdown("#### Matchup adjustment note")
        # We show a small heading for matchup reasoning.

        st.write(adj_note)
        # We display the matchup adjustment explanation.

        st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
        # We add a divider before over/under probabilities.

        st.markdown("### Over / Under Probabilities")
        # We display the probability section heading.

        ou1, ou2 = st.columns(2)
        # We create two columns for PTS and PRA probability cards.

        with ou1:
            # We place the PTS probability card in the left column.
            st.markdown("**PTS Line**")
            # We label the PTS line section.

            st.caption("PTS = Points")
            # We explain the PTS abbreviation.

            if pts_ou.get("ok"):
                # We check whether PTS probability calculation succeeded.
                st.metric("Line", f"{pts_ou['line']:.1f}")
                # We show the selected PTS line.

                st.metric("Prob OVER", f"{pts_ou['prob_over']*100:.0f}%")
                # We show the estimated probability of going over.

                st.metric("Prob UNDER", f"{pts_ou['prob_under']*100:.0f}%")
                # We show the estimated probability of going under.

                st.caption(f"mu={pts_ou['mu']:.2f} | sigma={pts_ou['sigma']:.2f}")
                # We show the mean and standard deviation used in the probability model.

                st.caption(pts_ou["sigma_note"])
                # We explain where sigma came from.

            else:
                # We handle failed PTS probability calculations.
                st.info(pts_ou.get("error", "Could not compute PTS probabilities."))
                # We show a friendly message if PTS probability cannot be computed.

        with ou2:
            # We place the PRA probability card in the right column.
            st.markdown("**PRA Line**")
            # We label the PRA line section.

            st.caption("PRA = Points + Rebounds + Assists")
            # We explain the PRA abbreviation.

            if pra_ou.get("ok"):
                # We check whether PRA probability calculation succeeded.
                st.metric("Line", f"{pra_ou['line']:.1f}")
                # We show the selected PRA line.

                st.metric("Prob OVER", f"{pra_ou['prob_over']*100:.0f}%")
                # We show the estimated PRA over probability.

                st.metric("Prob UNDER", f"{pra_ou['prob_under']*100:.0f}%")
                # We show the estimated PRA under probability.

                st.caption(f"mu={pra_ou['mu']:.2f} | sigma={pra_ou['sigma']:.2f}")
                # We show the mean and standard deviation used for PRA.

                st.caption(pra_ou["sigma_note"])
                # We explain where PRA sigma came from.

            else:
                # We handle failed PRA probability calculations.
                st.info(pra_ou.get("error", "Could not compute PRA probabilities."))
                # We show a friendly message if PRA probability cannot be computed.

        if opp_abbr and opp_abbr != "—":
            # We only show matchup history when the user selects an opponent.
            st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
            # We add a divider before matchup history.

            st.markdown("### Matchup History")
            # We display the matchup history heading.

            hist = get_matchup_history(df_logs, player, opp_abbr, last_k=10)
            # We get up to 10 recent games against the selected opponent.

            if hist.empty:
                # We check whether matchup history is empty.
                st.info("No matchup history found for this player and opponent.")
                # We inform the user if no matchup history exists.

            else:
                # We handle the case where matchup history exists.
                st.dataframe(hist, width="stretch", hide_index=True)
                # We display the matchup history table.

        st.markdown('<div class="courtiq-divider"></div>', unsafe_allow_html=True)
        # We add a divider before recent games.

        st.markdown("### Last 5 Games")
        # We display the recent games heading.

        last5 = last_games_table(df_logs, player, k=5)
        # We build the player's last five games table.

        if last5.empty:
            # We check whether recent games are available.
            st.info("No recent game logs found for that player in this dataset.")
            # We show a friendly message if no recent games exist.

        else:
            # We handle the case where recent games exist.
            st.dataframe(last5, width="stretch", hide_index=True)
            # We display the recent games table.

st.markdown("</div>", unsafe_allow_html=True)
# We close the styled prediction card.


# -------------------------
# FOOTER
# -------------------------

st.markdown(
    """
<div class="courtiq-card">
  <div style="color:#dc2626; font-weight:800; font-size:1rem; line-height:1.4;">
    ⚠️ Disclaimer: Court IQ is an AI-powered analytics tool designed to provide 
    data-driven projections based on historical NBA performance trends. 
    
    All outputs are probabilistic estimates — not guarantees — and are intended 
    strictly for educational, research, and demonstration purposes. 
    
    Users should not rely on these predictions for financial decisions, wagering, 
    or real-world risk-based activities.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
# We display the app footer and disclaimer.
