from __future__ import annotations

from pathlib import Path
import pandas as pd


DATA_DEFAULT = Path("data/raw/player_gamelogs_2024_25_Regular_Season_nba_api.csv")


def _load_gamelogs(csv_path: Path = DATA_DEFAULT) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. Run fetch first:\n"
            f"  python -m src.courtiq.data.fetch_player_gamelog_nba_api"
        )
    df = pd.read_csv(csv_path)
    return df


def predict_from_last_n(
    player_name: str,
    n: int = 5,
    csv_path: Path = DATA_DEFAULT,
) -> dict:
    """
    Baseline predictor:
    - filters player rows
    - sorts by GAME_DATE
    - averages last N games for PTS/REB/AST
    """
    df = _load_gamelogs(csv_path)

    # Normalize match: case-insensitive
    player_df = df[df["PLAYER_NAME"].str.lower() == player_name.strip().lower()].copy()

    if player_df.empty:
        # fallback: contains match (helps when user types partial name)
        player_df = df[df["PLAYER_NAME"].str.lower().str.contains(player_name.strip().lower())].copy()

    if player_df.empty:
        return {
            "ok": False,
            "error": f"Player not found in dataset: '{player_name}'",
        }

    player_df["GAME_DATE"] = pd.to_datetime(player_df["GAME_DATE"], errors="coerce")
    player_df = player_df.sort_values("GAME_DATE")

    last_n = player_df.tail(n)

    pts = float(last_n["PTS"].mean())
    reb = float(last_n["REB"].mean())
    ast = float(last_n["AST"].mean())

    return {
        "ok": True,
        "player": str(last_n["PLAYER_NAME"].iloc[-1]),
        "n_games": int(len(last_n)),
        "predicted_points": round(pts, 1),
        "predicted_rebounds": round(reb, 1),
        "predicted_assists": round(ast, 1),
        "predicted_pra": round(pts + reb + ast, 1),
        "data_source": str(csv_path).replace("\\", "/"),
        "method": f"mean_last_{n}",
    }
