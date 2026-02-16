from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.courtiq.utils.nba_http import NBAHttpClient


def _resultset_to_df(payload: Dict[str, Any]) -> pd.DataFrame:
    # stats.nba.com usually returns:
    # {"resultSets":[{"headers":[...], "rowSet":[...]}]}
    rs = payload["resultSets"][0]
    headers = rs["headers"]
    rows = rs["rowSet"]
    return pd.DataFrame(rows, columns=headers)


def fetch_player_gamelog(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Pulls ALL player game logs for a season (one row per player per game).
    season format: "2024-25"
    season_type: "Regular Season" or "Playoffs"
    """
    client = NBAHttpClient()
    params = {
        "DateFrom": "",
        "DateTo": "",
        "GameSegment": "",
        "LastNGames": "0",
        "LeagueID": "00",
        "Location": "",
        "MeasureType": "Base",
        "Month": "0",
        "OpponentTeamID": "0",
        "Outcome": "",
        "PORound": "0",
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": "0",
        "PlayerID": "0",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": season,
        "SeasonSegment": "",
        "SeasonType": season_type,
        "ShotClockRange": "",
        "VsConference": "",
        "VsDivision": "",
    }

    payload = client.get_json("playergamelogs", params=params)
    return _resultset_to_df(payload)


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    season = os.getenv("COURTIQ_SEASON", "2024-25")
    season_type = os.getenv("COURTIQ_SEASON_TYPE", "Regular Season")

    df = fetch_player_gamelog(season=season, season_type=season_type)
    out_file = Path("data/raw") / f"player_gamelogs_{season.replace('-', '_')}_{season_type.replace(' ', '_')}.csv"
    save_csv(df, out_file)

    print(f"Saved {len(df):,} rows -> {out_file}")
    print(df.head(3).to_string(index=False))
