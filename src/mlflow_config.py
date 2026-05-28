"""
MLflow Configuration Module
Purpose: Centralized MLflow setup for both local and DagsHub tracking
Used by all Phase scripts for consistent experiment logging

Author: RiskLens MLOps
Date: 2026-05-27
"""

import os
import logging
import mlflow

logger = logging.getLogger(__name__)

def setup_mlflow_tracking():
    """
    Configure MLflow tracking.
    Checks for DagsHub credentials first, falls back to local tracking.
    """
    
    # Check if DagsHub credentials are set
    dagshub_username = os.getenv("DAGSHUB_USERNAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    
    if dagshub_username and dagshub_token:
        # Use DagsHub remote tracking
        tracking_uri = f"https://{dagshub_username}:{dagshub_token}@dagshub.com/Pandharimaske/RiskLens.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("✓ MLflow configured for DagsHub remote tracking")
        logger.info(f"  Repository: https://dagshub.com/Pandharimaske/RiskLens")
        return "dagshub"
    else:
        # Use local MLflow tracking (default)
        mlflow.set_tracking_uri("file:./mlruns")
        logger.info("✓ MLflow configured for local tracking")
        logger.info("  To enable DagsHub: python setup_dagshub.py")
        return "local"

def get_tracking_mode():
    """Return current tracking mode: 'dagshub' or 'local'"""
    tracking_uri = mlflow.get_tracking_uri()
    if "dagshub.com" in tracking_uri:
        return "dagshub"
    return "local"

# Initialize on import
setup_mlflow_tracking()
