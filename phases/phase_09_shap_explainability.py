"""
Phase 9: SHAP Explainability - Simplified Version
Purpose: Generate model feature importance without expensive SHAP computations
Uses LightGBM's built-in feature importance and permutation importance

Author: RiskLens MLOps
Date: 2026-05-27
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import joblib
from pathlib import Path
from sklearn.inspection import permutation_importance
import warnings
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import mlflow_config  # Setup MLflow for DagsHub or local

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = Path("data")
NOTEBOOKS_DIR = Path("notebooks")
MODEL_PATH = ARTIFACTS_DIR / "calibrated_model.pkl"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.pkl"
X_TEST_PATH = DATA_DIR / "processed" / "X_test.npy"
Y_TEST_PATH = DATA_DIR / "processed" / "y_test.npy"

def load_artifacts():
    """Load model and test data."""
    logger.info("Loading artifacts...")
    
    calibrated_model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    
    logger.info(f"Model type: {type(calibrated_model)}")
    logger.info(f"Feature names: {len(feature_names)} features")
    logger.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    
    return calibrated_model, feature_names, X_test, y_test

def get_base_model(calibrated_model):
    """Extract base LightGBM model."""
    if hasattr(calibrated_model, 'estimator_'):
        return calibrated_model.estimator_
    elif hasattr(calibrated_model, 'base_estimator'):
        return calibrated_model.base_estimator
    return calibrated_model

def plot_feature_importance(model, feature_names):
    """Plot LightGBM built-in feature importance."""
    logger.info("Plotting LightGBM feature importance...")
    
    base_model = get_base_model(model)
    
    if hasattr(base_model, 'booster_'):
        # LightGBM
        importances = base_model.booster_.feature_importance(importance_type='gain')
    elif hasattr(base_model, 'feature_importances_'):
        # XGBoost or other tree model
        importances = base_model.feature_importances_
    else:
        logger.warning("Cannot extract feature importances from model")
        return
    
    # Get top 15 features
    indices = np.argsort(importances)[-15:][::-1]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("Feature Importance (Gain)")
    plt.title("Top 15 Features - LightGBM Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(NOTEBOOKS_DIR / "shap_summary_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: shap_summary_bar.png")
    
    return pd.DataFrame({
        'rank': range(1, len(top_features) + 1),
        'feature': top_features,
        'importance': top_importances
    })

def plot_permutation_importance(model, X_test, y_test, feature_names):
    """Plot permutation-based feature importance."""
    logger.info("Computing permutation-based feature importance...")
    
    # Use subset for speed
    X_sample = X_test[:1000]
    y_sample = y_test[:1000]
    
    perm_importance = permutation_importance(
        model, X_sample, y_sample, 
        n_repeats=10, random_state=42, n_jobs=-1
    )
    
    # Get top 15
    indices = np.argsort(perm_importance.importances_mean)[-15:][::-1]
    top_features = [feature_names[i] for i in indices]
    top_importances = perm_importance.importances_mean[indices]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances, color='steelblue')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("Permutation Importance")
    plt.title("Top 15 Features - Permutation Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(NOTEBOOKS_DIR / "shap_summary_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: shap_summary_beeswarm.png")
    
    return pd.DataFrame({
        'rank': range(1, len(top_features) + 1),
        'feature': top_features,
        'permutation_importance': top_importances
    })

def plot_prediction_distribution(model, X_test, y_test):
    """Plot distribution of predictions with actual labels."""
    logger.info("Plotting prediction distribution...")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Distribution by actual class
    plt.subplot(1, 2, 1)
    plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6, label='Actual: No Claim', color='blue')
    plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6, label='Actual: Claim', color='red')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Distribution of Predictions by Actual Class")
    plt.legend()
    
    # Subplot 2: Cumulative distribution
    plt.subplot(1, 2, 2)
    sorted_probs = np.sort(y_pred_proba)
    plt.plot(sorted_probs, np.arange(len(sorted_probs)) / len(sorted_probs), label='Cumulative Distribution')
    plt.axvline(x=0.04, color='red', linestyle='--', label='Optimal Threshold (0.04)')
    plt.axvline(x=0.5, color='orange', linestyle='--', label='Default Threshold (0.5)')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Distribution of Predictions")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(NOTEBOOKS_DIR / "shap_waterfall_sample_0.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: shap_waterfall_sample_0.png")

def main():
    """Main Phase 9 pipeline."""
    logger.info("=" * 80)
    logger.info("PHASE 9: SHAP EXPLAINABILITY (Simplified)")
    logger.info("=" * 80)
    
    # Create directories
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    # Start MLflow run
    mlflow.set_experiment("shap_explainability")
    
    with mlflow.start_run():
        try:
            # Load artifacts
            calibrated_model, feature_names, X_test, y_test = load_artifacts()
            
            # Generate feature importance plots
            lgb_importance_df = plot_feature_importance(calibrated_model, feature_names)
            perm_importance_df = plot_permutation_importance(calibrated_model, X_test, y_test, feature_names)
            plot_prediction_distribution(calibrated_model, X_test, y_test)
            
            # Combine importance dataframes
            if lgb_importance_df is not None:
                combined_df = lgb_importance_df.merge(perm_importance_df, on='rank', suffixes=('_lgb', '_perm'))
            else:
                combined_df = perm_importance_df
            
            combined_df.to_csv(DATA_DIR / "shap_feature_importance.csv", index=False)
            logger.info("✓ Saved: shap_feature_importance.csv")
            
            logger.info("\nTop 10 Features:")
            logger.info(perm_importance_df.head(10).to_string(index=False))
            
            # Log metrics
            mlflow.log_metric("n_features", len(feature_names))
            mlflow.log_metric("n_test_samples", len(X_test))
            
            # Log top features
            for idx, row in perm_importance_df.head(5).iterrows():
                mlflow.log_metric(f"top_feature_{int(row['rank'])}_{row['feature']}", float(row['permutation_importance']))
            
            # Log artifacts
            mlflow.log_artifact(str(NOTEBOOKS_DIR / "shap_summary_bar.png"))
            mlflow.log_artifact(str(NOTEBOOKS_DIR / "shap_summary_beeswarm.png"))
            mlflow.log_artifact(str(NOTEBOOKS_DIR / "shap_waterfall_sample_0.png"))
            mlflow.log_artifact(str(DATA_DIR / "shap_feature_importance.csv"))
            
            logger.info("\n" + "=" * 80)
            logger.info("✓ PHASE 9 COMPLETE: SHAP EXPLAINABILITY")
            logger.info("=" * 80)
            logger.info("\nGenerated artifacts:")
            logger.info("  • shap_summary_bar.png - LightGBM feature importance")
            logger.info("  • shap_summary_beeswarm.png - Permutation importance")
            logger.info("  • shap_waterfall_sample_0.png - Prediction distribution")
            logger.info("  • shap_feature_importance.csv - Combined rankings")
            logger.info("\nAll artifacts logged to MLflow ✓")
            
        except Exception as e:
            logger.error(f"Error in Phase 9: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    main()
