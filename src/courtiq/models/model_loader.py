import json
import zipfile
from pathlib import Path
import joblib

CACHE_DIR = Path(".model_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = Path("courtiq_models.zip")

PTS_NAME = "real_best_pts_gradient_boosting.pkl"
REB_NAME = "real_best_reb_gradient_boosting.pkl"
AST_NAME = "real_best_ast_gradient_boosting.pkl"
FEATURES_NAME = "feature_columns.json"


def _extract_if_needed():
    needed = [
        CACHE_DIR / PTS_NAME,
        CACHE_DIR / REB_NAME,
        CACHE_DIR / AST_NAME,
        CACHE_DIR / FEATURES_NAME,
    ]

    if all(p.exists() for p in needed):
        return

    if not ZIP_PATH.exists():
        raise RuntimeError("courtiq_models.zip not found in repo")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(CACHE_DIR)


def load_models():
    _extract_if_needed()

    pts_model = joblib.load(CACHE_DIR / PTS_NAME)
    reb_model = joblib.load(CACHE_DIR / REB_NAME)
    ast_model = joblib.load(CACHE_DIR / AST_NAME)

    feature_cols = json.loads(
        (CACHE_DIR / FEATURES_NAME).read_text(encoding="utf-8")
    )

    return pts_model, reb_model, ast_model, feature_cols