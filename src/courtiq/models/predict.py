from __future__ import annotations

from pathlib import Path
import pandas as pd

# Import the model loader to load trained models and feature columns
from src.courtiq.models.model_loader import load_models


# Automatically select the newest available regular season gamelog file
def _get_latest_gamelog_path() -> Path:
    data_dir = Path("data/raw")

    files = sorted(
        data_dir.glob("player_gamelogs_*_Regular_Season*_nba_api.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Use the most recent file if available
    if files:
        return files[0]

    # Fallback to default season file
    return data_dir / "player_gamelogs_2024_25_Regular_Season_nba_api.csv"


# Default dataset path
DATA_DEFAULT = _get_latest_gamelog_path()


def _load_gamelogs(csv_path: Path = DATA_DEFAULT) -> pd.DataFrame:
    """
    Load the player gamelog CSV file used for predictions.
    """
    csv_path = Path(csv_path)

    # Ensure file exists before loading
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Expected a file inside data/raw."
        )

    return pd.read_csv(csv_path)


def _build_feature_row_from_last_n(
    last_n: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """
    Build a single feature row using averages from the last N games.

    For each feature:
    - use the mean value if available
    - otherwise default to 0.0
    """
    row = {}

    for col in feature_cols:
        if col in last_n.columns:
            val = float(
                pd.to_numeric(last_n[col], errors="coerce").fillna(0).mean()
            )
        else:
            val = 0.0

        row[col] = val

    # Return as a one-row DataFrame for model input
    return pd.DataFrame([row], columns=feature_cols)


# Load trained models once at import time
pts_model, reb_model, ast_model, FEATURE_COLS = load_models()


def predict_from_last_n(
    player_name: str,
    n: int = 5,
    csv_path: Path = DATA_DEFAULT,
) -> dict:
    """
    Predict next-game stats based on the player's last N games.

    Steps:
    - validate inputs
    - load gamelog data
    - find player rows
    - sort by date
    - take last N games
    - build feature input
    - run model predictions
    - return results
    """

    # Validate number of games
    try:
        n = int(n)
    except Exception:
        n = 5

    if n <= 0:
        n = 5

    # Clean player name
    name = str(player_name or "").strip()

    if not name:
        return {
            "ok": False,
            "error": "Player name is required.",
            "method": "ml_models_real",
        }

    # Load dataset
    df = _load_gamelogs(csv_path)

    # Try exact match first
    player_df = df[df["PLAYER_NAME"].astype(str).str.lower() == name.lower()].copy()

    # Fallback to partial match
    if player_df.empty:
        player_df = df[
            df["PLAYER_NAME"].astype(str).str.lower().str.contains(name.lower(), na=False)
        ].copy()

    if player_df.empty:
        return {
            "ok": False,
            "error": f"Player not found in dataset: '{name}'",
            "method": "ml_models_real",
        }

    # Sort games by date
    player_df["GAME_DATE"] = pd.to_datetime(player_df["GAME_DATE"], errors="coerce")
    player_df = player_df.sort_values("GAME_DATE")

    # Select last N games
    last_n = player_df.tail(n).copy()
    n_used = int(len(last_n))

    # Build model input features
    X = _build_feature_row_from_last_n(last_n, FEATURE_COLS)

    # Run predictions
    pred_pts = float(pts_model.predict(X)[0])
    pred_reb = float(reb_model.predict(X)[0])
    pred_ast = float(ast_model.predict(X)[0])

    # Calculate PRA
    pred_pra = pred_pts + pred_reb + pred_ast

    # Return results
    return {
        "ok": True,
        "player": str(last_n["PLAYER_NAME"].iloc[-1]),
        "n_games": n_used,
        "predicted_points": round(pred_pts, 2),
        "predicted_rebounds": round(pred_reb, 2),
        "predicted_assists": round(pred_ast, 2),
        "predicted_pra": round(pred_pra, 2),
        "data_source": str(csv_path).replace("\\", "/"),
        "method": "ml_models_real",
    }