#!/usr/bin/env python
"""
================================================================================
PHASE 7 - THRESHOLD OPTIMIZATION
================================================================================
Purpose:
    Find the optimal decision threshold for cost-sensitive classification
    using the calibrated model from Phase 6.

Approach:
    1. Load calibrated model and test data
    2. Generate probability predictions on test set
    3. Define cost matrix (FN cost = 50k, FP cost = 3k)
    4. Calculate expected cost for each threshold
    5. Find threshold that minimizes total cost
    6. Compare default (0.5) vs optimal threshold
    7. Visualize: ROC curve, cost curve, threshold analysis
    8. Log results to MLflow

Cost Matrix:
    - False Negative (FN): 50,000 (missed insurance claims = revenue loss)
    - False Positive (FP): 3,000 (false alarms = outreach cost)
    - True Positive (TP): 0 (correctly identified = revenue)
    - True Negative (TN): 0 (correctly rejected = no cost)

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, f1_score, 
    confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")
NOTEBOOKS_DIR = Path("notebooks")
RESULTS_DIR = Path("data")

# Cost matrix (per observation)
COST_FN = 50000  # False negative: missed claim
COST_FP = 3000   # False positive: false alarm

print("=" * 80)
print("PHASE 7 - THRESHOLD OPTIMIZATION")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
logger.info("Loading processed data...")

X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

logger.info(f"Test: ({X_test.shape[0]}, {X_test.shape[1]}) | Positive: {100*y_test.mean():.2f}%")

# ============================================================================
# 2. LOAD CALIBRATED MODEL FROM PHASE 6
# ============================================================================
logger.info("Loading calibrated model from Phase 6...")

# For Phase 7, we need to retrain the calibrated model as a standalone artifact
# We'll train base + calibrate it here since the MLflow logged model may have serialization issues

from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

# Load training/validation data
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
logger.info("Calibrating model...")
calibrated_model = CalibratedClassifierCV(
    base_model, 
    method='sigmoid', 
    cv=5
)
calibrated_model.fit(X_val, y_val)
logger.info("✓ Model loaded and calibrated")

# ============================================================================
# 3. GENERATE PROBABILITY PREDICTIONS
# ============================================================================
logger.info("Generating probability predictions...")

y_proba = calibrated_model.predict_proba(X_test)[:, 1]
logger.info(f"Prediction range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")

# ============================================================================
# 4. CALCULATE COST FOR DIFFERENT THRESHOLDS
# ============================================================================
logger.info("Calculating cost for different thresholds...")

thresholds = np.linspace(0, 1, 101)
costs = []
f1_scores = []
tn_list = []
fp_list = []
fn_list = []
tp_list = []

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Total cost = (FN_count * FN_cost) + (FP_count * FP_cost)
    total_cost = (fn * COST_FN) + (fp * COST_FP)
    
    # F1 score for comparison
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    costs.append(total_cost)
    f1_scores.append(f1)
    tn_list.append(tn)
    fp_list.append(fp)
    fn_list.append(fn)
    tp_list.append(tp)

costs = np.array(costs)
f1_scores = np.array(f1_scores)

# Find optimal threshold
optimal_idx = np.argmin(costs)
optimal_threshold = thresholds[optimal_idx]
optimal_cost = costs[optimal_idx]

logger.info(f"Optimal threshold: {optimal_threshold:.4f} (cost: ${optimal_cost:,.0f})")

# Default threshold (0.5) performance
default_idx = 50  # threshold = 0.5
default_cost = costs[default_idx]

logger.info(f"Default threshold (0.5) cost: ${default_cost:,.0f}")
logger.info(f"Cost reduction: ${default_cost - optimal_cost:,.0f} ({100*(default_cost - optimal_cost)/default_cost:.2f}%)")

# ============================================================================
# 5. DETAILED METRICS AT OPTIMAL THRESHOLD
# ============================================================================
logger.info(f"\nEvaluating at optimal threshold ({optimal_threshold:.4f})...")

y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_test, y_pred_optimal).ravel()

# Metrics
precision_opt = tp_opt / (tp_opt + fp_opt) if (tp_opt + fp_opt) > 0 else 0
recall_opt = tp_opt / (tp_opt + fn_opt) if (tp_opt + fn_opt) > 0 else 0
f1_opt = 2 * (precision_opt * recall_opt) / (precision_opt + recall_opt) if (precision_opt + recall_opt) > 0 else 0

logger.info(f"  True Positives:  {tp_opt:,}")
logger.info(f"  False Positives: {fp_opt:,}")
logger.info(f"  True Negatives:  {tn_opt:,}")
logger.info(f"  False Negatives: {fn_opt:,}")
logger.info(f"  Precision: {precision_opt:.4f}")
logger.info(f"  Recall:    {recall_opt:.4f}")
logger.info(f"  F1 Score:  {f1_opt:.4f}")

# ============================================================================
# 6. ROC CURVE FOR REFERENCE
# ============================================================================
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# ============================================================================
# 7. GENERATE VISUALIZATIONS
# ============================================================================
logger.info("Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phase 7: Threshold Optimization Analysis', fontsize=16, fontweight='bold')

# Plot 1: Cost vs Threshold
ax = axes[0, 0]
ax.plot(thresholds, costs / 1e6, linewidth=2, color='#d62728', label='Total Cost')
ax.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2, label=f'Optimal: {optimal_threshold:.4f}')
ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Default: 0.5000')
ax.set_xlabel('Threshold', fontsize=11)
ax.set_ylabel('Cost (Millions $)', fontsize=11)
ax.set_title('Total Cost vs Decision Threshold', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)

# Plot 2: Cost Components vs Threshold
ax = axes[0, 1]
fn_costs = np.array(fn_list) * COST_FN / 1e6
fp_costs = np.array(fp_list) * COST_FP / 1e6
ax.plot(thresholds, fn_costs, linewidth=2, label=f'FN Cost (${COST_FN:,}/ea)', color='red')
ax.plot(thresholds, fp_costs, linewidth=2, label=f'FP Cost (${COST_FP:,}/ea)', color='orange')
ax.plot(thresholds, (fn_costs + fp_costs), linewidth=2.5, label='Total', color='#d62728')
ax.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Threshold', fontsize=11)
ax.set_ylabel('Cost (Millions $)', fontsize=11)
ax.set_title('Cost Components vs Threshold', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)

# Plot 3: F1 Score vs Threshold
ax = axes[1, 0]
ax.plot(thresholds, f1_scores, linewidth=2, color='#1f77b4', label='F1 Score')
ax.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2, label=f'Optimal: {optimal_threshold:.4f}')
ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Default: 0.5000')
f1_optimal = f1_scores[optimal_idx]
f1_default = f1_scores[default_idx]
ax.plot([optimal_threshold], [f1_optimal], 'go', markersize=8)
ax.plot([0.5], [f1_default], 'o', color='orange', markersize=8)
ax.set_xlabel('Threshold', fontsize=11)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('F1 Score vs Decision Threshold', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)

# Plot 4: ROC Curve
ax = axes[1, 1]
ax.plot(fpr, tpr, linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='#2ca02c')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(NOTEBOOKS_DIR / 'threshold_optimization.png', dpi=300, bbox_inches='tight')
logger.info("✓ Saved: notebooks/threshold_optimization.png")
plt.close()

# ============================================================================
# 8. COMPARISON TABLE
# ============================================================================
comparison_data = {
    'Metric': [
        'Decision Threshold',
        'Total Cost',
        'False Negatives',
        'False Positives',
        'True Positives',
        'True Negatives',
        'Precision',
        'Recall',
        'F1 Score'
    ],
    'Default (0.5)': [
        0.5,
        f"${default_cost:,.0f}",
        fn_list[default_idx],
        fp_list[default_idx],
        tp_list[default_idx],
        tn_list[default_idx],
        f"{tp_list[default_idx] / (tp_list[default_idx] + fp_list[default_idx]):.4f}" if (tp_list[default_idx] + fp_list[default_idx]) > 0 else 0,
        f"{tp_list[default_idx] / (tp_list[default_idx] + fn_list[default_idx]):.4f}" if (tp_list[default_idx] + fn_list[default_idx]) > 0 else 0,
        f1_scores[default_idx]
    ],
    'Optimal': [
        f"{optimal_threshold:.4f}",
        f"${optimal_cost:,.0f}",
        fn_opt,
        fp_opt,
        tp_opt,
        tn_opt,
        f"{precision_opt:.4f}",
        f"{recall_opt:.4f}",
        f"{f1_opt:.4f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(RESULTS_DIR / 'threshold_optimization_results.csv', index=False)
logger.info("✓ Saved: data/threshold_optimization_results.csv")

print("\n" + "=" * 80)
print("THRESHOLD OPTIMIZATION RESULTS")
print("=" * 80)
print(comparison_df.to_string(index=False))

# ============================================================================
# 9. MLflow LOGGING
# ============================================================================
logger.info("Logging to MLflow...")

mlflow.set_experiment("threshold_optimization")

with mlflow.start_run(run_name="cost_optimized_threshold"):
    # Parameters
    mlflow.log_param("cost_fn", COST_FN)
    mlflow.log_param("cost_fp", COST_FP)
    mlflow.log_param("method", "cost_minimization")
    
    # Optimal threshold metrics
    mlflow.log_metric("optimal_threshold", optimal_threshold)
    mlflow.log_metric("optimal_cost", optimal_cost)
    mlflow.log_metric("cost_reduction", default_cost - optimal_cost)
    mlflow.log_metric("cost_reduction_pct", 100 * (default_cost - optimal_cost) / default_cost)
    
    # Performance metrics at optimal threshold
    mlflow.log_metric("fn_count", fn_opt)
    mlflow.log_metric("fp_count", fp_opt)
    mlflow.log_metric("tp_count", tp_opt)
    mlflow.log_metric("tn_count", tn_opt)
    mlflow.log_metric("precision", precision_opt)
    mlflow.log_metric("recall", recall_opt)
    mlflow.log_metric("f1_score", f1_opt)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Default threshold metrics
    mlflow.log_metric("default_threshold", 0.5)
    mlflow.log_metric("default_cost", default_cost)
    mlflow.log_metric("default_f1", f1_scores[default_idx])
    
    logger.info("✓ Metrics logged to MLflow")

# ============================================================================
# 10. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 7 - THRESHOLD OPTIMIZATION COMPLETE")
print("=" * 80)

print(f"\n📊 Optimal Decision Threshold: {optimal_threshold:.4f}")
print(f"   (Default threshold: 0.5000)")

print(f"\n💰 Cost Analysis:")
print(f"   Default cost:    ${default_cost:,.0f}")
print(f"   Optimal cost:    ${optimal_cost:,.0f}")
print(f"   Cost reduction:  ${default_cost - optimal_cost:,.0f} ({100*(default_cost - optimal_cost)/default_cost:.2f}%)")

print(f"\n📈 Performance at Optimal Threshold ({optimal_threshold:.4f}):")
print(f"   Precision: {precision_opt:.4f}")
print(f"   Recall:    {recall_opt:.4f}")
print(f"   F1 Score:  {f1_opt:.4f}")

print(f"\n✓ Threshold optimization visualized")
print(f"✓ Results saved to data/threshold_optimization_results.csv")
print(f"✓ Metrics logged to MLflow")
print(f"✓ Ready for Phase 8: Model Serving & Deployment")

print("\n" + "=" * 80)
