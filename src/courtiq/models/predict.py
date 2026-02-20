from __future__ import annotations

from pathlib import Path
import os
import json

import joblib
import pandas as pd


DATA_DEFAULT = Path("data/raw/player_gamelogs_2024_25_Regular_Season_nba_api.csv")


def _repo_root_from_here() -> Path:
    """
    This file lives at: <repo>/src/courtiq/models/predict.py
    Repo root is 3 levels up from this file's directory.
    """
    here = Path(__file__).resolve()
    return here.parents[3]


REPO_ROOT = _repo_root_from_here()
MODELS_DIR = REPO_ROOT / "models"

FEATURES_PATH = MODELS_DIR / "feature_columns.json"
PTS_MODEL_PATH = MODELS_DIR / "real_best_pts_random_forest.pkl"
REB_MODEL_PATH = MODELS_DIR / "real_best_reb_gradient_boosting.pkl"
AST_MODEL_PATH = MODELS_DIR / "real_best_ast_gradient_boosting.pkl"


def _load_gamelogs(csv_path: Path = DATA_DEFAULT) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. Run fetch first:\n"
            f"  python -m src.courtiq.data.fetch_player_gamelog_nba_api"
        )
    return pd.read_csv(csv_path)


def _load_feature_cols() -> list[str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {FEATURES_PATH}. Create it with:\n"
            f"  python .\\src\\export_model_features.py"
        )
    cols = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    if not isinstance(cols, list) or not cols:
        raise ValueError("feature_columns.json is empty or invalid.")
    return cols


# Load models once (at import time)
FEATURE_COLS = _load_feature_cols()

if not PTS_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing PTS model: {PTS_MODEL_PATH}")
if not REB_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing REB model: {REB_MODEL_PATH}")
if not AST_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing AST model: {AST_MODEL_PATH}")

pts_model = joblib.load(str(PTS_MODEL_PATH))
reb_model = joblib.load(str(REB_MODEL_PATH))
ast_model = joblib.load(str(AST_MODEL_PATH))


def _build_feature_row_from_last_n(last_n: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 1-row dataframe with columns=FEATURE_COLS.
    For each feature column:
      - if present in last_n, take mean over last_n
      - else set 0.0
    """
    row = {}
    for col in FEATURE_COLS:
        if col in last_n.columns:
            val = float(pd.to_numeric(last_n[col], errors="coerce").fillna(0).mean())
        else:
            val = 0.0
        row[col] = val
    return pd.DataFrame([row], columns=FEATURE_COLS)


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
    - builds feature vector from last-N averages (feature_columns.json)
    - predicts next-game PTS/REB/AST using trained models
    """
    if not isinstance(n, int) or n <= 0:
        n = 5

    df = _load_gamelogs(csv_path)

    # Normalize match: case-insensitive exact match first
    player_df = df[df["PLAYER_NAME"].astype(str).str.lower() == player_name.strip().lower()].copy()

    if player_df.empty:
        # fallback: contains match (helps when user types partial name)
        player_df = df[
            df["PLAYER_NAME"].astype(str).str.lower().str.contains(player_name.strip().lower(), na=False)
        ].copy()

    if player_df.empty:
        return {
            "ok": False,
            "error": f"Player not found in dataset: '{player_name}'",
            "method": "ml_models_real",
        }

    # Sort by date
    player_df["GAME_DATE"] = pd.to_datetime(player_df["GAME_DATE"], errors="coerce")
    player_df = player_df.sort_values("GAME_DATE")

    last_n = player_df.tail(n).copy()
    n_used = int(len(last_n))

    # Build model features
    X = _build_feature_row_from_last_n(last_n)

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