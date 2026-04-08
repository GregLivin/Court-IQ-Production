import os
import json
import zipfile
import urllib.request
from pathlib import Path

import joblib

# Model zip file URL from Streamlit Cloud secrets
MODELS_ZIP_URL = os.getenv("COURTIQ_MODELS_ZIP_URL", "").strip()

# Local cache folder for model files
CACHE_DIR = Path(os.getenv("COURTIQ_MODEL_CACHE", ".model_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Path for the downloaded zip file
ZIP_PATH = CACHE_DIR / "courtiq_models.zip"

# Model file names
PTS_NAME = "real_best_pts_gradient_boosting.pkl"
REB_NAME = "real_best_reb_gradient_boosting.pkl"
AST_NAME = "real_best_ast_gradient_boosting.pkl"
FEATURES_NAME = "feature_columns.json"


def _download_and_extract_if_needed() -> None:
    """
    Download and extract model files if they are not already cached.
    """
    needed = [
        CACHE_DIR / PTS_NAME,
        CACHE_DIR / REB_NAME,
        CACHE_DIR / AST_NAME,
        CACHE_DIR / FEATURES_NAME,
    ]

    # Skip download if all required files already exist
    if all(p.exists() for p in needed):
        return

    # Stop if the model URL is missing
    if not MODELS_ZIP_URL:
        raise RuntimeError(
            "COURTIQ_MODELS_ZIP_URL is not set. Add it in Streamlit Cloud Secrets."
        )

    # Download the zip file
    urllib.request.urlretrieve(MODELS_ZIP_URL, ZIP_PATH)

    # Extract model files into the cache folder
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(CACHE_DIR)


def load_models():
    """
    Load trained models and feature columns from the cache folder.
    """
    _download_and_extract_if_needed()

    pts_model = joblib.load(CACHE_DIR / PTS_NAME)
    reb_model = joblib.load(CACHE_DIR / REB_NAME)
    ast_model = joblib.load(CACHE_DIR / AST_NAME)

    feature_cols = json.loads(
        (CACHE_DIR / FEATURES_NAME).read_text(encoding="utf-8")
    )

    return pts_model, reb_model, ast_model, feature_cols