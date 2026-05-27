"""
Production Configuration for RiskLens Model Serving
"""
import os
from pathlib import Path

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
WORKERS = int(os.getenv("WORKERS", 4))

# Model
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/calibrated_model.pkl"))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.pkl"))
FEATURE_NAMES_PATH = Path(os.getenv("FEATURE_NAMES_PATH", "artifacts/feature_names.pkl"))

# Prediction Threshold (from Phase 7)
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", 0.0400))

# Limits
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 10000))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model Metadata
MODEL_VERSION = "1.0.0"
MODEL_NAME = "RiskLens - Vehicle Insurance Claim Prediction"
CREATED_DATE = "2026-05-27"
TRAINED_ON = "381,109 insurance customer records"

# Versioning
API_VERSION = "v1"
