"""Data loading and train/val/test split"""
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# This should load from data/raw/vehicle_insurance_data.csv
# For now, load existing processed data
data_dir = Path("data/processed")
X_train = np.load(data_dir / "X_train.npy")
y_train = np.load(data_dir / "y_train.npy")
X_val = np.load(data_dir / "X_val.npy")
X_test = np.load(data_dir / "X_test.npy")
y_test = np.load(data_dir / "y_test.npy")

print(f"✓ Data loaded: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
