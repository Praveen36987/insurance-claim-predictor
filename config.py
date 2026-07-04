import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "scored_files"
ASSETS_DIR = BASE_DIR / "assets"

# Specific file paths
MODEL_PATH = MODELS_DIR / "trained_pipeline.joblib"
FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.json"
THRESHOLD_PATH = MODELS_DIR / "threshold_config.json"

# UI Config
APP_TITLE = "Insurance Claim Prediction Analytics"
APP_SUBTITLE = "Upload policyholder data to predict claim probabilities and assess risk."

# Styling configuration (Colors)
COLOR_LOW_RISK = "#2e7d32"      # Green
COLOR_MED_RISK = "#f57c00"      # Amber
COLOR_HIGH_RISK = "#c62828"     # Muted Red
COLOR_PRIMARY = "#006064"       # Deep Teal
