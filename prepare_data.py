#!/usr/bin/env python
"""
Standalone feature engineering and data preparation script.
Creates domain features, preprocesses data, and saves splits.
"""

import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load raw data
logger.info("Loading raw data...")
df = pd.read_csv('data/raw/data.csv')
logger.info(f"✓ Loaded {df.shape[0]} records with {df.shape[1]} columns")

# Create domain features (VECTORIZED for speed)
logger.info("Creating domain-engineered features...")
df_engineered = df.copy()

# 1. Vehicle Age numeric - vectorized
vehicle_age_map = {"< 1 Year": 0.5, "1-2 Year": 1.5, "> 2 Years": 3}
df_engineered['Vehicle_Age_Numeric'] = df_engineered['Vehicle_Age'].map(vehicle_age_map)

# 2. Premium per vehicle year - vectorized
df_engineered['Premium_per_Vehicle_Year'] = (
    df_engineered['Annual_Premium'] / (df_engineered['Vehicle_Age_Numeric'] + 1)
)

# 3. High-value vehicle flag - vectorized
premium_75th = df_engineered['Annual_Premium'].quantile(0.75)
df_engineered['High_Value_Vehicle'] = (
    df_engineered['Annual_Premium'] > premium_75th
).astype(int)

# 4. Age risk bucket - vectorized with np.select
age = df_engineered['Age']
df_engineered['Age_Risk_Bucket'] = np.select(
    [age < 25, age < 35, age < 50, age < 65],
    ['very_high_risk', 'high_risk', 'medium_risk', 'low_risk'],
    default='very_low_risk'
)

# 5. Customer tenure segment - vectorized with np.select
vintage = df_engineered['Vintage']
df_engineered['Customer_Tenure_Segment'] = np.select(
    [vintage < 30, vintage < 90, vintage < 365],
    ['new_customer', 'growing_customer', 'established_customer'],
    default='loyal_customer'
)

# 6. Premium bucket - vectorized with pd.cut
q25, q50, q75 = df_engineered['Annual_Premium'].quantile([0.25, 0.50, 0.75])
df_engineered['Premium_Bucket'] = pd.cut(
    df_engineered['Annual_Premium'],
    bins=[0, q25, q50, q75, float('inf')],
    labels=['low_premium', 'medium_premium', 'high_premium', 'very_high_premium']
).astype(str)

# 7. Damage history risk - vectorized
damage_numeric = (df_engineered['Vehicle_Damage'] == 'Yes').astype(int)
df_engineered['Damage_History_Risk'] = damage_numeric * (1 - df_engineered['Previously_Insured'])

logger.info("✓ Created 7 domain features")

# Split data with stratification
logger.info("Splitting data with stratification...")
X = df_engineered.drop(['id', 'Response'], axis=1)
y = df_engineered['Response']

# Test split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15/(1-0.15), 
    random_state=42, stratify=y_trainval
)

logger.info(f"Train: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%) | Positive: {y_train.mean():.2%}")
logger.info(f"Val:   {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%) | Positive: {y_val.mean():.2%}")
logger.info(f"Test:  {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%) | Positive: {y_test.mean():.2%}")

# Identify feature types
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

logger.info(f"Numeric features: {len(numeric_features)}")
logger.info(f"Categorical features: {len(categorical_features)}")

# Build preprocessor (fit only on training data)
logger.info("Building and fitting preprocessor on training data...")

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
], n_jobs=-1)  # Parallel processing

# Fit on training data only
logger.info("Fitting preprocessor (this may take 1-2 minutes for 381k records)...")
preprocessor.fit(X_train)
logger.info(f"✓ Preprocessor fitted on {len(X_train)} training samples")

# Transform all splits in parallel
logger.info("Transforming all splits...")
X_train_transformed = preprocessor.transform(X_train)
logger.info(f"  ✓ Train transformed: {X_train_transformed.shape}")

X_val_transformed = preprocessor.transform(X_val)
logger.info(f"  ✓ Val transformed: {X_val_transformed.shape}")

X_test_transformed = preprocessor.transform(X_test)
logger.info(f"  ✓ Test transformed: {X_test_transformed.shape}")

# Save processed data
output_dir = 'data/processed'
Path(output_dir).mkdir(parents=True, exist_ok=True)

np.save(f'{output_dir}/X_train.npy', X_train_transformed)
np.save(f'{output_dir}/X_val.npy', X_val_transformed)
np.save(f'{output_dir}/X_test.npy', X_test_transformed)
np.save(f'{output_dir}/y_train.npy', y_train.values)
np.save(f'{output_dir}/y_val.npy', y_val.values)
np.save(f'{output_dir}/y_test.npy', y_test.values)

pd.DataFrame(X_train_transformed).to_pickle(f'{output_dir}/X_train.pkl')
pd.DataFrame(X_val_transformed).to_pickle(f'{output_dir}/X_val.pkl')
pd.DataFrame(X_test_transformed).to_pickle(f'{output_dir}/X_test.pkl')
y_train.to_pickle(f'{output_dir}/y_train.pkl')
y_val.to_pickle(f'{output_dir}/y_val.pkl')
y_test.to_pickle(f'{output_dir}/y_test.pkl')

logger.info(f"✓ Processed data saved to {output_dir}")

# Save preprocessor
Path('artifacts').mkdir(parents=True, exist_ok=True)
joblib.dump(preprocessor, 'artifacts/preprocessor.pkl')
logger.info("✓ Preprocessor saved to artifacts/preprocessor.pkl")

# Save feature names
joblib.dump(numeric_features + categorical_features, 'artifacts/feature_names.pkl')
logger.info("✓ Feature names saved to artifacts/feature_names.pkl")

print("\n" + "="*70)
print("PHASE 2 - FEATURE ENGINEERING COMPLETE")
print("="*70)
print(f"✓ Domain features created: 7")
print(f"✓ Data splits (stratified):")
print(f"  - Train: {X_train_transformed.shape[0]} samples")
print(f"  - Val:   {X_val_transformed.shape[0]} samples")
print(f"  - Test:  {X_test_transformed.shape[0]} samples")
print(f"✓ Preprocessor fitted on training data only (no leakage)")
print(f"✓ All data saved to {output_dir}/")
print(f"✓ Preprocessor saved to artifacts/preprocessor.pkl")
print(f"✓ Ready for Phase 3: Baseline Modeling")
print("="*70)
