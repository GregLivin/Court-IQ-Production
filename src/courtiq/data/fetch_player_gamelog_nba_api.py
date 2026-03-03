from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

# nba_api (recommended)
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as static_players


# -------------------------
# Env Config
# -------------------------
SEASON = os.getenv("COURTIQ_SEASON", "2025-26")
SEASON_TYPE = os.getenv("COURTIQ_SEASON_TYPE", "Regular Season")
SLEEP_SECONDS = float(os.getenv("COURTIQ_SLEEP_SECONDS", "0.8"))

# 1 = only active NBA players (recommended)
ACTIVE_ONLY = os.getenv("COURTIQ_ACTIVE_ONLY", "1").strip() != "0"

# 0 = all (not recommended), otherwise fetch first N players
LIMIT_PLAYERS = int(os.getenv("COURTIQ_LIMIT_PLAYERS", "0"))

# Retries per player call
MAX_RETRIES = int(os.getenv("COURTIQ_MAX_RETRIES", "3"))

OUT_PATH = Path(os.getenv(
    "COURTIQ_OUT_PATH",
    r"data\raw\player_gamelogs_2025_26_Regular_Season_nba_api.csv"
))


# -------------------------
# Helpers
# -------------------------
def get_player_list(active_only: bool) -> list[dict]:
    """
    Return list of player dicts with at least: id, full_name
    """
    if active_only:
        plist = static_players.get_active_players()
    else:
        plist = static_players.get_players()  # huge (includes historical)

    # Standardize keys
    cleaned = []
    for p in plist:
        pid = p.get("id")
        name = p.get("full_name") or p.get("fullName") or p.get("name")
        if pid and name:
            cleaned.append({"id": pid, "full_name": name})
    return cleaned


def fetch_one_player_gamelog(player_id: int) -> pd.DataFrame | None:
    """
    Pull game logs for one player for configured season/type.
    Retries with backoff to survive NBA Stats hiccups.
    """
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=SEASON,
                season_type_all_star=SEASON_TYPE,
                timeout=45,
            )
            df = gl.get_data_frames()[0]
            return df
        except Exception as e:
            last_err = e
            # Exponential-ish backoff
            sleep_s = min(8.0, 1.5 * attempt)
            time.sleep(sleep_s)

    print(f"[WARN] Failed player_id={player_id} after {MAX_RETRIES} retries: {last_err}")
    return None


def main():
    print(f"[CourtIQ] Using season: {SEASON}")
    print(f"[CourtIQ] Using season type: {SEASON_TYPE}")
    print(f"[CourtIQ] Active only: {1 if ACTIVE_ONLY else 0}")
    print(f"[CourtIQ] Limit players: {LIMIT_PLAYERS} (0 = no limit)")
    print(f"[CourtIQ] Sleep seconds: {SLEEP_SECONDS}")
    print(f"[CourtIQ] Output: {OUT_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    players = get_player_list(active_only=ACTIVE_ONLY)
    if LIMIT_PLAYERS > 0:
        players = players[:LIMIT_PLAYERS]

    print(f"[CourtIQ] Players to fetch: {len(players):,}")

    all_rows = []
    fetched = 0

    for i, p in enumerate(players, start=1):
        pid = int(p["id"])
        name = p["full_name"]

        df = fetch_one_player_gamelog(pid)

        # Respect rate limits
        time.sleep(SLEEP_SECONDS)

        if df is None or df.empty:
            continue

        # Add player label
        df["PLAYER_ID"] = pid
        df["PLAYER_NAME"] = name

        all_rows.append(df)
        fetched += 1

        if i % 50 == 0:
            print(f"[CourtIQ] Progress: {i:,}/{len(players):,} players... (successful: {fetched:,})")

    if not all_rows:
        print("[CourtIQ] No data fetched. Exiting.")
        return

    out_df = pd.concat(all_rows, ignore_index=True)

    # Make GAME_DATE consistent
    if "GAME_DATE" in out_df.columns:
        out_df["GAME_DATE"] = pd.to_datetime(out_df["GAME_DATE"], errors="coerce")

    out_df.to_csv(OUT_PATH, index=False)
    print(f"[CourtIQ] Saved: {OUT_PATH} | rows={len(out_df):,} | players_with_games={fetched:,}")


if __name__ == "__main__":
    main()