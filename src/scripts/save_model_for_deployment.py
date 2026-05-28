#!/usr/bin/env python
"""
Save calibrated model for deployment.
Required for Phase 8: Model Serving & Deployment
"""

import pickle
import logging
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")

logger.info("Saving calibrated model for deployment...")

# Load data
X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

# Best hyperparameters from Phase 5
best_params = {
    'num_leaves': 34,
    'max_depth': 5,
    'learning_rate': 0.086635,
    'feature_fraction': 0.644715,
    'bagging_fraction': 0.817324,
    'bagging_freq': 6,
    'lambda_l1': 5.802862,
    'lambda_l2': 3.542260,
    'min_data_in_leaf': 75,
    'scale_pos_weight': 7.16,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

# Train base model
logger.info("Training base LightGBM model...")
base_model = lgb.LGBMClassifier(**best_params)
base_model.fit(X_train, y_train)

# Calibrate
logger.info("Calibrating model with sigmoid method...")
calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
calibrated_model.fit(X_val, y_val)

# Save
model_path = ARTIFACTS_DIR / "calibrated_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(calibrated_model, f)

logger.info(f"✓ Model saved: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

# Verify
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)
    logger.info(f"✓ Model verified (type: {type(loaded_model).__name__})")

logger.info("✓ Ready for Phase 8 deployment!")
