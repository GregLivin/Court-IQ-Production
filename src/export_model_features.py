import json
import pandas as pd

data_path = r"data\processed\training_data.csv"
targets = ["PTS", "REB", "AST"]

df = pd.read_csv(data_path).select_dtypes(include=["number"]).dropna()

leak_cols = [
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB",
    "PLUS_MINUS", "NBA_FANTASY_PTS", "WNBA_FANTASY_PTS",
]
leak_cols += [c for c in df.columns if c.endswith("_RANK")]
leak_cols = [c for c in leak_cols if c in df.columns]

feature_cols = [c for c in df.columns if (c not in targets and c not in leak_cols)]

with open(r"models\feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("Saved: models\\feature_columns.json")
print("Feature count:", len(feature_cols))
print("Sample:", feature_cols[:12])