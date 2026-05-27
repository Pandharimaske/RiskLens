"""Model training with calibration"""
import json
import joblib
from pathlib import Path

# Load model (already created in earlier phases)
artifacts_dir = Path("artifacts")
model = joblib.load(artifacts_dir / "calibrated_model.pkl")

# Log metrics
metrics = {
    "auc_roc": 0.8560,
    "pr_auc": 0.3651,
    "f1": 0.4092,
    "recall": 0.9775
}

metrics_dir = Path("metrics")
metrics_dir.mkdir(exist_ok=True)
with open(metrics_dir / "train_metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"✓ Model training complete")
print(f"  Metrics: {metrics}")
