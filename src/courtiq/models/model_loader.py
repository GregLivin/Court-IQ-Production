import os
import json
import zipfile
import urllib.request
from pathlib import Path
import joblib

# Set this in Streamlit Cloud Secrets:
# COURTIQ_MODELS_ZIP_URL="https://github.com/<user>/<repo>/releases/download/models-v1/courtiq_models.zip"
MODELS_ZIP_URL = os.getenv("COURTIQ_MODELS_ZIP_URL", "").strip()

# Cache folder inside app directory (works on Streamlit Cloud)
CACHE_DIR = Path(os.getenv("COURTIQ_MODEL_CACHE", ".model_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = CACHE_DIR / "courtiq_models.zip"

# Use deploy-safe small models
PTS_NAME = "real_best_pts_gradient_boosting.pkl"
REB_NAME = "real_best_reb_gradient_boosting.pkl"
AST_NAME = "real_best_ast_gradient_boosting.pkl"
FEATURES_NAME = "feature_columns.json"


def _download_and_extract_if_needed() -> None:
    needed = [CACHE_DIR / PTS_NAME, CACHE_DIR / REB_NAME, CACHE_DIR / AST_NAME, CACHE_DIR / FEATURES_NAME]
    if all(p.exists() for p in needed):
        return

    if not MODELS_ZIP_URL:
        raise RuntimeError("COURTIQ_MODELS_ZIP_URL is not set. Add it in Streamlit Cloud Secrets.")

    urllib.request.urlretrieve(MODELS_ZIP_URL, ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(CACHE_DIR)


def load_models():
    _download_and_extract_if_needed()

    pts_model = joblib.load(CACHE_DIR / PTS_NAME)
    reb_model = joblib.load(CACHE_DIR / REB_NAME)
    ast_model = joblib.load(CACHE_DIR / AST_NAME)

    feature_cols = json.loads((CACHE_DIR / FEATURES_NAME).read_text(encoding="utf-8"))
    return pts_model, reb_model, ast_model, feature_cols