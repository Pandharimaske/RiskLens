#!/usr/bin/env python
"""
Quick model regeneration to fix scikit-learn version compatibility.
Uses current environment's scikit-learn 1.8.0 to regenerate artifacts.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from src.config import MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH

print("🔧 Quick Model Regeneration for scikit-learn 1.8.0")
print("=" * 70)

# Load training data
print("Loading training data...")
train_df = pd.read_csv("data/processed_insurance_train.csv")
X = train_df.drop(['id', 'Response'], axis=1)
y = train_df['Response']

print(f"✓ Loaded {len(X)} training samples")
print(f"✓ Features: {list(X.columns)}")

# Feature Engineering
print("\nApplying feature engineering...")
vehicle_age_mapping = {"< 1 Year": 0.5, "1-2 Year": 1.5, "> 2 Years": 3}
X['Vehicle_Age_Numeric'] = X['Vehicle_Age'].map(vehicle_age_mapping)
X['Premium_per_Vehicle_Year'] = X['Annual_Premium'] / (X['Vehicle_Age_Numeric'] + 1)
X['High_Value_Vehicle'] = (X['Annual_Premium'] > 40000).astype(int)

def age_risk_bucket(age):
    if age < 25: return 'very_high_risk'
    elif age < 35: return 'high_risk'
    elif age < 50: return 'medium_risk'
    elif age < 65: return 'low_risk'
    else: return 'very_low_risk'

X['Age_Risk_Bucket'] = X['Age'].apply(age_risk_bucket)

def tenure_segment(vintage):
    if vintage < 1: return 'new_customer'
    elif vintage < 3: return 'growing_customer'
    elif vintage < 5: return 'established_customer'
    else: return 'loyal_customer'

X['Customer_Tenure_Segment'] = X['Vintage'].apply(tenure_segment)

quantiles = X['Annual_Premium'].quantile([0.25, 0.5, 0.75])
def premium_bucket(premium):
    if premium <= quantiles[0.25]: return 'low'
    elif premium <= quantiles[0.5]: return 'medium'
    elif premium <= quantiles[0.75]: return 'high'
    else: return 'very_high'

X['Premium_Bucket'] = X['Annual_Premium'].apply(premium_bucket)
X['Damage_History_Risk'] = X['Vehicle_Damage_encoded'] * (1 - X['Previously_Insured'])

print(f"✓ Created 7 engineered features")

# Preprocessing Pipeline
print("\nBuilding preprocessing pipeline...")
categorical_features = ['Gender', 'Vehicle_Age', 'Vehicle_Damage_encoded', 
                        'Age_Risk_Bucket', 'Customer_Tenure_Segment', 'Premium_Bucket']
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

print(f"✓ Numerical features: {len(numerical_features)}")
print(f"✓ Categorical features: {len(categorical_features)}")

# Fit preprocessor
X_processed = preprocessor.fit_transform(X)
feature_names = list(preprocessor.get_feature_names_out())
print(f"✓ Preprocessor fitted with {len(feature_names)} features")

# Train model with optimized parameters
print("\nTraining LightGBM model...")
lgb = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=7.16,
    random_state=42,
    verbose=-1
)

lgb.fit(X_processed, y)
print("✓ LightGBM model trained")

# Calibrate
print("Calibrating model...")
calibrated_model = CalibratedClassifierCV(lgb, method='sigmoid', cv=5)
calibrated_model.fit(X_processed, y)
print("✓ Model calibrated with sigmoid method")

# Save artifacts with current scikit-learn version
print("\nSaving artifacts with scikit-learn 1.8.0...")
joblib.dump(calibrated_model, MODEL_PATH)
joblib.dump(preprocessor, PREPROCESSOR_PATH)
joblib.dump(feature_names, FEATURE_NAMES_PATH)

print(f"✓ Model saved to: {MODEL_PATH}")
print(f"✓ Preprocessor saved to: {PREPROCESSOR_PATH}")
print(f"✓ Feature names saved to: {FEATURE_NAMES_PATH}")

print("\n" + "=" * 70)
print("✅ Model artifacts regenerated successfully!")
print("All files are now compatible with scikit-learn 1.8.0")
print("=" * 70)
