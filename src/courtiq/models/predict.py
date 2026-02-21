from __future__ import annotations

from pathlib import Path
import pandas as pd

# ✅ Use the existing loader you already created
from src.courtiq.models.model_loader import load_models


# ----------------------------
# Data path
# ----------------------------
DATA_DEFAULT = Path("data/raw/player_gamelogs_2024_25_Regular_Season_nba_api.csv")


def _load_gamelogs(csv_path: Path = DATA_DEFAULT) -> pd.DataFrame:
    """
    Load the gamelog CSV.
    Works locally (repo file) and on Streamlit Cloud (repo file).
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Expected: data/raw/player_gamelogs_2024_25_Regular_Season_nba_api.csv"
        )

    return pd.read_csv(csv_path)


def _build_feature_row_from_last_n(last_n: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Build a 1-row dataframe with columns=feature_cols.
    For each feature column:
      - if present in last_n, take mean over last_n
      - else set 0.0
    """
    row = {}
    for col in feature_cols:
        if col in last_n.columns:
            val = float(pd.to_numeric(last_n[col], errors="coerce").fillna(0).mean())
        else:
            val = 0.0
        row[col] = val

    return pd.DataFrame([row], columns=feature_cols)


# ----------------------------
# Load models once (import-time)
# ----------------------------
# load_models() will auto-download + unzip on Streamlit Cloud using COURTIQ_MODELS_ZIP_URL
pts_model, reb_model, ast_model, FEATURE_COLS = load_models()


# ----------------------------
# Main prediction function
# ----------------------------
def predict_from_last_n(
    player_name: str,
    n: int = 5,
    csv_path: Path = DATA_DEFAULT,
) -> dict:
    """
    ML predictor:
    - filters player rows
    - sorts by GAME_DATE
    - takes last N games
    - builds feature vector from last-N averages (feature columns)
    - predicts next-game PTS/REB/AST using trained models
    """
    try:
        n = int(n)
    except Exception:
        n = 5
    if n <= 0:
        n = 5

    name = str(player_name or "").strip()
    if not name:
        return {"ok": False, "error": "Player name is required.", "method": "ml_models_real"}

    df = _load_gamelogs(csv_path)

    # Exact match first (case-insensitive)
    player_df = df[df["PLAYER_NAME"].astype(str).str.lower() == name.lower()].copy()

    # Fallback: contains match
    if player_df.empty:
        player_df = df[
            df["PLAYER_NAME"].astype(str).str.lower().str.contains(name.lower(), na=False)
        ].copy()

    if player_df.empty:
        return {"ok": False, "error": f"Player not found in dataset: '{name}'", "method": "ml_models_real"}

    # Sort by date
    player_df["GAME_DATE"] = pd.to_datetime(player_df["GAME_DATE"], errors="coerce")
    player_df = player_df.sort_values("GAME_DATE")

    last_n = player_df.tail(n).copy()
    n_used = int(len(last_n))

    # Build model features
    X = _build_feature_row_from_last_n(last_n, FEATURE_COLS)

    # Predict
    pred_pts = float(pts_model.predict(X)[0])
    pred_reb = float(reb_model.predict(X)[0])
    pred_ast = float(ast_model.predict(X)[0])
    pred_pra = pred_pts + pred_reb + pred_ast

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