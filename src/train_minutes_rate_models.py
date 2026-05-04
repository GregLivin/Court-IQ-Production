from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


DATA_FILE = Path("data/raw/player_gamelogs_multi_season_nba_api.csv")
MODEL_DIR = Path("models")
EXPORT_DIR = Path("exports")

MODEL_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.sort_values(["PLAYER_NAME", "GAME_DATE"])

    for col in ["MIN", "PTS", "REB", "AST"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

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

    results = []

    for name, target_col in targets.items():
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False,
        )

        if name == "minutes":
            model = RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                min_samples_leaf=3,
                n_jobs=-1,
            )
        else:
            model = GradientBoostingRegressor(
                random_state=42,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
            )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        model_path = MODEL_DIR / f"{name}_model.pkl"
        joblib.dump(model, model_path)

        results.append(
            {
                "target": name,
                "target_column": target_col,
                "model_file": str(model_path),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "rows_used": len(df),
                "data_file": str(DATA_FILE),
            }
        )

        print(f"Saved {name}: {model_path}")
        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    joblib.dump(feature_cols, MODEL_DIR / "minutes_rate_features.pkl")

    metrics_df = pd.DataFrame(results)
    metrics_path = EXPORT_DIR / "metrics_minutes_rate_models.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("Saved feature columns.")
    print(f"Saved metrics: {metrics_path}")
    print(f"Rows used: {len(df):,}")
    print(f"Data source: {DATA_FILE}")


if __name__ == "__main__":
    main()