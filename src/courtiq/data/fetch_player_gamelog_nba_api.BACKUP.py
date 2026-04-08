"""
Fetch ALL NBA player game logs for a season using nba_api and save to CSV.

Usage (PowerShell):
  $env:COURTIQ_SEASON="2025-26"
  $env:COURTIQ_SEASON_TYPE="Regular Season"
  python -m src.courtiq.data.fetch_player_gamelog_nba_api

Notes:
- This script writes a NEW file in data/raw/ each time (season-specific).
- It prints where it saved + how many rows.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.library.parameters import SeasonTypeAllStar


def fetch_player_gamelogs_nba_api(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Pulls the full player game log table for the given season.
    """
    season = season.strip()
    season_type = season_type.strip()

    # Map season type to nba_api enum
    season_type_map = {
        "Regular Season": SeasonTypeAllStar.regular,
        "Playoffs": SeasonTypeAllStar.playoffs,
        "Pre Season": SeasonTypeAllStar.preseason,
        "All Star": SeasonTypeAllStar.all_star,
    }

    stype = season_type_map.get(season_type, SeasonTypeAllStar.regular)

    endpoint = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable=stype,
        timeout=120,
    )

    df = endpoint.get_data_frames()[0]
    return df


def main() -> None:
    # Read env vars (defaults keep it safe)
    season = os.getenv("COURTIQ_SEASON", "2024-25").strip()
    season_type = os.getenv("COURTIQ_SEASON_TYPE", "Regular Season").strip()

    # Pull data
    df = fetch_player_gamelogs_nba_api(season=season, season_type=season_type)

    # Build output path
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"player_gamelogs_{season.replace('-', '_')}_{season_type.replace(' ', '_')}_nba_api.csv"

    # Save
    df.to_csv(out_file, index=False)

    # Print so we can verify in terminal
    print(f"Saved: {out_file}")
    print(f"Rows:  {len(df):,}")
    if "GAME_DATE" in df.columns:
        # Show max date if available
        s = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        print(f"Date range: {s.min()} -> {s.max()}")


if __name__ == "__main__":
    main()