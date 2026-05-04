from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data/raw/player_gamelogs_2025_26_Regular_Season_nba_api.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
df = df.sort_values(["PLAYER_NAME", "GAME_DATE"])

df["PTS_PER_MIN"] = df["PTS"] / df["MIN"].replace(0, np.nan)
df["REB_PER_MIN"] = df["REB"] / df["MIN"].replace(0, np.nan)
df["AST_PER_MIN"] = df["AST"] / df["MIN"].replace(0, np.nan)

for col in ["MIN", "PTS", "REB", "AST"]:
    df[f"{col}_LAST5"] = df.groupby("PLAYER_NAME")[col].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

for col in ["PTS_PER_MIN", "REB_PER_MIN", "AST_PER_MIN"]:
    df[f"{col}_LAST5"] = df.groupby("PLAYER_NAME")[col].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

feature_cols = [
    "MIN_LAST5",
    "PTS_LAST5",
    "REB_LAST5",
    "AST_LAST5",
    "PTS_PER_MIN_LAST5",
    "REB_PER_MIN_LAST5",
    "AST_PER_MIN_LAST5",
]

targets = {
    "minutes": "MIN",
    "pts_rate": "PTS_PER_MIN",
    "reb_rate": "REB_PER_MIN",
    "ast_rate": "AST_PER_MIN",
}

X = df[feature_cols]

for name, target_col in targets.items():
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    if name == "minutes":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = GradientBoostingRegressor(random_state=42)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    model_path = MODEL_DIR / f"{name}_model.pkl"
    joblib.dump(model, model_path)

    print(f"Saved {name}: {model_path}")
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")

joblib.dump(feature_cols, MODEL_DIR / "minutes_rate_features.pkl")

print("Saved feature columns.")