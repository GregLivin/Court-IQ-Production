import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/player_gamelogs_2024_25_Regular_Season_nba_api.csv")


def predict_player(player_name: str):
    df = pd.read_csv(DATA_PATH)

    player_df = df[df["PLAYER_NAME"].str.lower() == player_name.lower()]

    if player_df.empty:
        return {"error": f"No data found for {player_name}"}

    last_5 = player_df.sort_values("GAME_DATE", ascending=False).head(5)

    return {
        "player": player_name,
        "predicted_points": round(last_5["PTS"].mean(), 2),
        "predicted_rebounds": round(last_5["REB"].mean(), 2),
        "predicted_assists": round(last_5["AST"].mean(), 2),
    }
