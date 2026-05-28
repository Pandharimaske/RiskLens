#!/usr/bin/env python3
"""
Phase 6 - Model Calibration
Calibrate LightGBM predictions to make probabilities trustworthy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, brier_score_loss, log_loss
)
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import logging
from time import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("\n" + "="*80)
print("PHASE 6 - MODEL CALIBRATION")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
logging.info("Loading processed data...")

data_dir = 'data/processed'
X_train = np.load(f'{data_dir}/X_train.npy')
X_val = np.load(f'{data_dir}/X_val.npy')
X_test = np.load(f'{data_dir}/X_test.npy')

y_train = np.load(f'{data_dir}/y_train.npy')
y_val = np.load(f'{data_dir}/y_val.npy')
y_test = np.load(f'{data_dir}/y_test.npy')

print(f"Train: {X_train.shape} | Positive: {y_train.mean():.2%}")
print(f"Val:   {X_val.shape} | Positive: {y_val.mean():.2%}")
print(f"Test:  {X_test.shape} | Positive: {y_test.mean():.2%}")

# Calculate class weight
pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
scale_pos_weight = neg_count / pos_count

# ============================================================================
# 2. CONFIGURE MLFLOW
# ============================================================================
logging.info("Configuring MLflow...")
mlflow.set_experiment("model_calibration")

# ============================================================================
# 3. TRAIN BASE MODEL (TUNED LGBM FROM PHASE 5)
# ============================================================================

print("\nTraining base LightGBM model (tuned hyperparameters)...")

# Best hyperparameters from Phase 5
base_model = lgb.LGBMClassifier(
    n_estimators=200,
    num_leaves=34,
    max_depth=5,
    learning_rate=0.086635,
    feature_fraction=0.644715,
    bagging_fraction=0.817324,
    bagging_freq=6,
    lambda_l1=5.802862,
    lambda_l2=3.542260,
    min_data_in_leaf=75,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=-1
)

base_model.fit(X_train, y_train)
print("✓ Base model trained")

# ============================================================================
# 4. EVALUATE BASE MODEL (UNCALIBRATED)
# ============================================================================

print("\nEvaluating base model on validation set...")

y_pred_base_val = base_model.predict(X_val)
y_pred_proba_base_val = base_model.predict_proba(X_val)[:, 1]

y_pred_base_test = base_model.predict(X_test)
y_pred_proba_base_test = base_model.predict_proba(X_test)[:, 1]

# Calibration metrics (val set for calibration curve)
base_brier_val = brier_score_loss(y_val, y_pred_proba_base_val)
base_logloss_val = log_loss(y_val, y_pred_proba_base_val)

# Test set metrics (before calibration)
base_auc_test = roc_auc_score(y_test, y_pred_proba_base_test)
base_pr_auc_test = average_precision_score(y_test, y_pred_proba_base_test)
base_f1_test = f1_score(y_test, y_pred_base_test)
base_brier_test = brier_score_loss(y_test, y_pred_proba_base_test)
base_logloss_test = log_loss(y_test, y_pred_proba_base_test)

print(f"  Brier Score (Val):  {base_brier_val:.4f}")
print(f"  Log Loss (Val):     {base_logloss_val:.4f}")
print(f"  AUC-ROC (Test):     {base_auc_test:.4f}")
print(f"  PR-AUC (Test):      {base_pr_auc_test:.4f}")
print(f"  Brier Score (Test): {base_brier_test:.4f}")
print(f"  Log Loss (Test):    {base_logloss_test:.4f}")

# ============================================================================
# 5. CALIBRATE MODEL
# ============================================================================

print("\nCalibratingmodel using validation set...")
t0 = time()

# CalibratedClassifierCV with sigmoid method
calibrated_model = CalibratedClassifierCV(
    base_model,
    method='sigmoid',
    cv=5  # 5-fold on validation set
)

calibrated_model.fit(X_val, y_val)
calib_time = time() - t0

print(f"✓ Model calibrated in {calib_time:.2f}s")

# ============================================================================
# 6. EVALUATE CALIBRATED MODEL
# ============================================================================

print("\nEvaluating calibrated model on test set...")

y_pred_calib_test = calibrated_model.predict(X_test)
y_pred_proba_calib_test = calibrated_model.predict_proba(X_test)[:, 1]

# Metrics after calibration
calib_auc_test = roc_auc_score(y_test, y_pred_proba_calib_test)
calib_pr_auc_test = average_precision_score(y_test, y_pred_proba_calib_test)
calib_f1_test = f1_score(y_test, y_pred_calib_test)
calib_brier_test = brier_score_loss(y_test, y_pred_proba_calib_test)
calib_logloss_test = log_loss(y_test, y_pred_proba_calib_test)

print(f"  AUC-ROC (Test):     {calib_auc_test:.4f}")
print(f"  PR-AUC (Test):      {calib_pr_auc_test:.4f}")
print(f"  Brier Score (Test): {calib_brier_test:.4f}")
print(f"  Log Loss (Test):    {calib_logloss_test:.4f}")

# ============================================================================
# 7. COMPARE CALIBRATED VS UNCALIBRATED
# ============================================================================

print("\n" + "="*80)
print("CALIBRATION IMPROVEMENT COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Metric': ['AUC-ROC', 'PR-AUC', 'F1', 'Brier Score', 'Log Loss'],
    'Uncalibrated': [base_auc_test, base_pr_auc_test, base_f1_test, base_brier_test, base_logloss_test],
    'Calibrated': [calib_auc_test, calib_pr_auc_test, calib_f1_test, calib_brier_test, calib_logloss_test],
})

comparison_df['Improvement'] = comparison_df['Calibrated'] - comparison_df['Uncalibrated']
comparison_df['% Change'] = (comparison_df['Improvement'] / comparison_df['Uncalibrated'] * 100)

print(comparison_df.to_string(index=False))
print("="*80)

# ============================================================================
# 8. GENERATE CALIBRATION CURVES
# ============================================================================

print("\nGenerating calibration curves...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Calibration curve - uncalibrated
ax = axes[0, 0]
prob_true_base, prob_pred_base = calibration_curve(
    y_test, y_pred_proba_base_test, n_bins=10, strategy='uniform'
)
ax.plot(prob_pred_base, prob_true_base, marker='o', linestyle='-', label='Uncalibrated', linewidth=2)
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curve - Uncalibrated')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Calibration curve - calibrated
ax = axes[0, 1]
prob_true_calib, prob_pred_calib = calibration_curve(
    y_test, y_pred_proba_calib_test, n_bins=10, strategy='uniform'
)
ax.plot(prob_pred_calib, prob_true_calib, marker='o', linestyle='-', label='Calibrated', linewidth=2)
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curve - Calibrated')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Brier score and log loss comparison
ax = axes[1, 0]
metrics = ['Brier Score', 'Log Loss']
uncalib_scores = [base_brier_test, base_logloss_test]
calib_scores = [calib_brier_test, calib_logloss_test]
x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, uncalib_scores, width, label='Uncalibrated', alpha=0.8)
ax.bar(x + width/2, calib_scores, width, label='Calibrated', alpha=0.8)
ax.set_ylabel('Score (Lower is Better)')
ax.set_title('Calibration Quality Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Prediction histogram
ax = axes[1, 1]
ax.hist(y_pred_proba_base_test[y_test == 0], bins=30, alpha=0.6, label='Uncalibrated (Neg)', color='blue')
ax.hist(y_pred_proba_calib_test[y_test == 0], bins=30, alpha=0.6, label='Calibrated (Neg)', color='lightblue')
ax.hist(y_pred_proba_base_test[y_test == 1], bins=30, alpha=0.6, label='Uncalibrated (Pos)', color='red')
ax.hist(y_pred_proba_calib_test[y_test == 1], bins=30, alpha=0.6, label='Calibrated (Pos)', color='lightcoral')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Frequency')
ax.set_title('Probability Distribution Comparison')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/model_calibration.png', dpi=100, bbox_inches='tight')
print("✓ Saved: notebooks/model_calibration.png")
plt.close()

# ============================================================================
# 9. LOG TO MLFLOW
# ============================================================================

with mlflow.start_run(run_name="lgb_calibrated"):
    mlflow.log_param("base_model", "LightGBM_Tuned")
    mlflow.log_param("calibration_method", "sigmoid")
    mlflow.log_param("calibration_cv_folds", 5)
    
    mlflow.log_metric("uncalib_auc_roc", base_auc_test)
    mlflow.log_metric("uncalib_pr_auc", base_pr_auc_test)
    mlflow.log_metric("uncalib_brier", base_brier_test)
    mlflow.log_metric("uncalib_logloss", base_logloss_test)
    
    mlflow.log_metric("calib_auc_roc", calib_auc_test)
    mlflow.log_metric("calib_pr_auc", calib_pr_auc_test)
    mlflow.log_metric("calib_brier", calib_brier_test)
    mlflow.log_metric("calib_logloss", calib_logloss_test)
    
    mlflow.log_metric("brier_improvement", calib_brier_test - base_brier_test)
    mlflow.log_metric("logloss_improvement", calib_logloss_test - base_logloss_test)
    
    mlflow.sklearn.log_model(calibrated_model, "calibrated_model")
    
    print("✓ Metrics logged to MLflow")

# ============================================================================
# 10. PHASE 6 SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PHASE 6 - MODEL CALIBRATION COMPLETE")
print("="*80)

print(f"\n✓ Base model (tuned LightGBM) trained")
print(f"✓ Calibration method: Sigmoid")
print(f"✓ Calibration time: {calib_time:.2f}s")

print(f"\n📊 Before Calibration (Uncalibrated):")
print(f"   AUC-ROC:     {base_auc_test:.4f}")
print(f"   PR-AUC:      {base_pr_auc_test:.4f}")
print(f"   Brier Score: {base_brier_test:.4f}")
print(f"   Log Loss:    {base_logloss_test:.4f}")

print(f"\n📊 After Calibration (Calibrated):")
print(f"   AUC-ROC:     {calib_auc_test:.4f}")
print(f"   PR-AUC:      {calib_pr_auc_test:.4f}")
print(f"   Brier Score: {calib_brier_test:.4f}")
print(f"   Log Loss:    {calib_logloss_test:.4f}")

print(f"\n📈 Improvements:")
print(f"   Brier Score: {(calib_brier_test - base_brier_test):+.4f} ({((calib_brier_test - base_brier_test) / base_brier_test * 100):+.2f}%)")
print(f"   Log Loss:    {(calib_logloss_test - base_logloss_test):+.4f} ({((calib_logloss_test - base_logloss_test) / base_logloss_test * 100):+.2f}%)")

print(f"\n✓ Calibration curves visualized")
print(f"✓ Metrics logged to MLflow")
print(f"✓ Ready for Phase 7: Threshold Optimization")

print("\n" + "="*80)

# Save results
comparison_df.to_csv('data/calibration_results.csv', index=False)
print("✓ Saved: data/calibration_results.csv")
