"""Feature engineering and preprocessing"""
import joblib
from pathlib import Path

# Load preprocessor (already created in earlier phases)
artifacts_dir = Path("artifacts")
preprocessor = joblib.load(artifacts_dir / "preprocessor.pkl")
feature_names = joblib.load(artifacts_dir / "feature_names.pkl")

print(f"✓ Preprocessor loaded with {len(feature_names)} features")
print(f"  Features: {feature_names}")
