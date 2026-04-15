from __future__ import annotations
# Makes type hints behave like strings automatically
# This helps avoid issues with forward references in Python typing


from pathlib import Path
# Path gives us a cleaner way to work with file and folder paths

import pandas as pd
# pandas is used to load and work with CSV data


# Import the model loader to load trained models and feature columns
from src.courtiq.models.model_loader import load_models
# This imports the function that downloads/loads the trained ML models
# and also loads the list of feature columns used for prediction


# -------------------------
# FIND THE LATEST DATASET
# -------------------------
def _get_latest_gamelog_path() -> Path:
    # Define the folder where raw player gamelog CSV files are stored
    data_dir = Path("data/raw")

    # Look through that folder for files matching this pattern:
    # player_gamelogs_*_Regular_Season*_nba_api.csv
    # Then sort them by last modified time, newest first
    files = sorted(
        data_dir.glob("player_gamelogs_*_Regular_Season*_nba_api.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # If matching files exist, use the newest one
    if files:
        return files[0]

    # If no matching files are found, fall back to a default CSV file
    return data_dir / "player_gamelogs_2024_25_Regular_Season_nba_api.csv"


# Save the newest/default dataset path as the default CSV used by the app
DATA_DEFAULT = _get_latest_gamelog_path()


# -------------------------
# LOAD GAMELOG CSV
# -------------------------
def _load_gamelogs(csv_path: Path = DATA_DEFAULT) -> pd.DataFrame:
    """
    Load the player gamelog CSV file used for predictions.
    """

    # Convert the input into a Path object just in case
    csv_path = Path(csv_path)

    # Make sure the file actually exists before trying to load it
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Expected a file inside data/raw."
        )

    # Read the CSV into a pandas DataFrame and return it
    return pd.read_csv(csv_path)


# -------------------------
# BUILD MODEL INPUT FEATURES
# -------------------------
def _build_feature_row_from_last_n(
    last_n: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """
    Build a single feature row using averages from the last N games.

    For each feature:
    - use the mean value if available
    - otherwise default to 0.0
    """

    # Empty dictionary to hold the model input values
    row = {}

    # Loop through every feature the trained model expects
    for col in feature_cols:

        # If the column exists in the player's last N games
        if col in last_n.columns:

            # Convert the values to numbers
            # Replace non-numeric values with NaN
            # Fill missing values with 0
            # Then take the average across the last N games
            val = float(
                pd.to_numeric(last_n[col], errors="coerce").fillna(0).mean()
            )

        else:
            # If the column does not exist, use 0.0 as a safe fallback
            val = 0.0

        # Store the value in the row dictionary
        row[col] = val

    # Convert the dictionary into a one-row DataFrame
    # The model expects input in DataFrame form, with exact feature column order
    return pd.DataFrame([row], columns=feature_cols)


# -------------------------
# LOAD MODELS ONCE
# -------------------------
# Load the trained points, rebounds, assists models,
# plus the feature column names they expect
pts_model, reb_model, ast_model, FEATURE_COLS = load_models()


# -------------------------
# MAIN PREDICTION FUNCTION
# -------------------------
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

    # -------------------------
    # VALIDATE NUMBER OF GAMES
    # -------------------------

    # Try converting n into an integer
    try:
        n = int(n)
    except Exception:
        # If conversion fails, default to 5
        n = 5

    # If n is zero or negative, also default to 5
    if n <= 0:
        n = 5

    # -------------------------
    # CLEAN PLAYER NAME
    # -------------------------

    # Convert player_name to string and remove extra spaces
    name = str(player_name or "").strip()

    # If the name is empty, return an error response
    if not name:
        return {
            "ok": False,
            "error": "Player name is required.",
            "method": "ml_models_real",
        }

    # -------------------------
    # LOAD DATASET
    # -------------------------

    # Load the player gamelog dataset from CSV
    df = _load_gamelogs(csv_path)

    # -------------------------
    # FIND PLAYER IN DATASET
    # -------------------------

    # First try an exact match on PLAYER_NAME
    player_df = df[df["PLAYER_NAME"].astype(str).str.lower() == name.lower()].copy()

    # If exact match fails, try a partial match
    if player_df.empty:
        player_df = df[
            df["PLAYER_NAME"].astype(str).str.lower().str.contains(name.lower(), na=False)
        ].copy()

    # If still no match, return an error
    if player_df.empty:
        return {
            "ok": False,
            "error": f"Player not found in dataset: '{name}'",
            "method": "ml_models_real",
        }

    # -------------------------
    # SORT BY DATE
    # -------------------------

    # Convert GAME_DATE into a proper datetime format
    player_df["GAME_DATE"] = pd.to_datetime(player_df["GAME_DATE"], errors="coerce")

    # Sort the player's games from oldest to newest
    player_df = player_df.sort_values("GAME_DATE")

    # -------------------------
    # TAKE LAST N GAMES
    # -------------------------

    # Grab only the most recent N games
    last_n = player_df.tail(n).copy()

    # Store how many games were actually used
    n_used = int(len(last_n))

    # -------------------------
    # BUILD MODEL INPUT
    # -------------------------

    # Build one row of features from the player's recent games
    X = _build_feature_row_from_last_n(last_n, FEATURE_COLS)

    # -------------------------
    # RUN MODEL PREDICTIONS
    # -------------------------

    # Predict points using the trained points model
    pred_pts = float(pts_model.predict(X)[0])

    # Predict rebounds using the trained rebounds model
    pred_reb = float(reb_model.predict(X)[0])

    # Predict assists using the trained assists model
    pred_ast = float(ast_model.predict(X)[0])

    # -------------------------
    # CALCULATE PRA
    # -------------------------

    # PRA = Points + Rebounds + Assists
    pred_pra = pred_pts + pred_reb + pred_ast

    # -------------------------
    # RETURN RESULTS
    # -------------------------

    # Return a dictionary with prediction results
    return {
        "ok": True,  # shows prediction succeeded
        "player": str(last_n["PLAYER_NAME"].iloc[-1]),  # player name
        "n_games": n_used,  # number of recent games used
        "predicted_points": round(pred_pts, 2),  # rounded points prediction
        "predicted_rebounds": round(pred_reb, 2),  # rounded rebounds prediction
        "predicted_assists": round(pred_ast, 2),  # rounded assists prediction
        "predicted_pra": round(pred_pra, 2),  # rounded PRA prediction
        "data_source": str(csv_path).replace("\\", "/"),  # dataset file path
        "method": "ml_models_real",  # label for the prediction method
    }