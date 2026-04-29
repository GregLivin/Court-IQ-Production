import os
import json
import zipfile
import urllib.request
from pathlib import Path

import joblib

MODELS_ZIP_URL = os.getenv("COURTIQ_MODELS_ZIP_URL", "").strip()
CACHE_DIR = Path(os.getenv("COURTIQ_MODEL_CACHE", ".model_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = CACHE_DIR / "courtiq_models.zip"
LOCAL_ZIP_PATH = Path("courtiq_models.zip")

PTS_NAME = "real_best_pts_gradient_boosting.pkl"
REB_NAME = "real_best_reb_gradient_boosting.pkl"
AST_NAME = "real_best_ast_gradient_boosting.pkl"
FEATURES_NAME = "feature_columns.json"


def _download_and_extract_if_needed() -> None:
    needed = [
        CACHE_DIR / PTS_NAME,
        CACHE_DIR / REB_NAME,
        CACHE_DIR / AST_NAME,
        CACHE_DIR / FEATURES_NAME,
    ]

    if all(p.exists() for p in needed):
        return

    if MODELS_ZIP_URL:
        urllib.request.urlretrieve(MODELS_ZIP_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(CACHE_DIR)
        return

    if LOCAL_ZIP_PATH.exists():
        with zipfile.ZipFile(LOCAL_ZIP_PATH, "r") as z:
            z.extractall(CACHE_DIR)
        return

    raise RuntimeError(
        "COURTIQ_MODELS_ZIP_URL is not set in Streamlit Secrets, "
        "and courtiq_models.zip was not found in the repo."
    )


def load_models():
    _download_and_extract_if_needed()

    pts_model = joblib.load(CACHE_DIR / PTS_NAME)
    reb_model = joblib.load(CACHE_DIR / REB_NAME)
    ast_model = joblib.load(CACHE_DIR / AST_NAME)

    feature_cols = json.loads(
        (CACHE_DIR / FEATURES_NAME).read_text(encoding="utf-8")
    )

    return pts_model, reb_model, ast_model, feature_cols