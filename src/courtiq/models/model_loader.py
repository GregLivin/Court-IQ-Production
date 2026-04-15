import os                  # lets us read environment variables (like secrets)
import json                # used to load feature columns from a JSON file
import zipfile             # used to extract the downloaded zip file
import urllib.request      # used to download files from a URL
from pathlib import Path   # modern way to handle file paths

import joblib              # used to load trained machine learning models (.pkl files)


# -------------------------
# CONFIGURATION
# -------------------------

# Get the model zip file URL from environment variables (Streamlit Secrets)
# If not found, default to empty string
MODELS_ZIP_URL = os.getenv("COURTIQ_MODELS_ZIP_URL", "").strip()


# Define a local folder where models will be stored
# Defaults to ".model_cache" if not set
CACHE_DIR = Path(os.getenv("COURTIQ_MODEL_CACHE", ".model_cache"))

# Create the folder if it doesn't already exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Define where the downloaded zip file will be saved locally
ZIP_PATH = CACHE_DIR / "courtiq_models.zip"


# -------------------------
# MODEL FILE NAMES
# -------------------------

# These are the trained machine learning models (Gradient Boosting)
PTS_NAME = "real_best_pts_gradient_boosting.pkl"  # predicts points
REB_NAME = "real_best_reb_gradient_boosting.pkl"  # predicts rebounds
AST_NAME = "real_best_ast_gradient_boosting.pkl"  # predicts assists

# This file contains the feature column names used during training
FEATURES_NAME = "feature_columns.json"


# -------------------------
# DOWNLOAD + EXTRACT MODELS
# -------------------------

def _download_and_extract_if_needed() -> None:
    """
    Download and extract model files if they are not already cached.
    """

    # List of all required files we expect after extraction
    needed = [
        CACHE_DIR / PTS_NAME,
        CACHE_DIR / REB_NAME,
        CACHE_DIR / AST_NAME,
        CACHE_DIR / FEATURES_NAME,
    ]

    # If ALL required files already exist, skip download
    if all(p.exists() for p in needed):
        return

    # If the URL is missing, we cannot download models
    if not MODELS_ZIP_URL:
        raise RuntimeError(
            "COURTIQ_MODELS_ZIP_URL is not set. Add it in Streamlit Cloud Secrets."
        )

    # Download the zip file from the URL into ZIP_PATH
    urllib.request.urlretrieve(MODELS_ZIP_URL, ZIP_PATH)

    # Open the zip file and extract all contents into CACHE_DIR
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(CACHE_DIR)


# -------------------------
# LOAD MODELS INTO MEMORY
# -------------------------

def load_models():
    """
    Load trained models and feature columns from the cache folder.
    """

    # Make sure models are downloaded and extracted first
    _download_and_extract_if_needed()

    # Load each trained model using joblib
    # These are Gradient Boosting models trained earlier
    pts_model = joblib.load(CACHE_DIR / PTS_NAME)  # points model
    reb_model = joblib.load(CACHE_DIR / REB_NAME)  # rebounds model
    ast_model = joblib.load(CACHE_DIR / AST_NAME)  # assists model

    # Load the feature column names (used to format input data correctly)
    feature_cols = json.loads(
        (CACHE_DIR / FEATURES_NAME).read_text(encoding="utf-8")
    )

    # Return all models and features so they can be used for prediction
    return pts_model, reb_model, ast_model, feature_cols