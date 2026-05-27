#!/usr/bin/env python
"""
Phase 3 - Baseline Modeling Script
Train and compare DummyClassifier and Logistic Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix, roc_curve
)
import mlflow
import mlflow.sklearn
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 5)

# Load processed data
print("Loading processed data from Phase 2...")
data_dir = 'data/processed'

X_train = np.load(f'{data_dir}/X_train.npy')
X_val = np.load(f'{data_dir}/X_val.npy')
X_test = np.load(f'{data_dir}/X_test.npy')

y_train = np.load(f'{data_dir}/y_train.npy')
y_val = np.load(f'{data_dir}/y_val.npy')
y_test = np.load(f'{data_dir}/y_test.npy')

print(f"Train set: {X_train.shape} | Positive rate: {y_train.mean():.2%}")
print(f"Val set:   {X_val.shape} | Positive rate: {y_val.mean():.2%}")
print(f"Test set:  {X_test.shape} | Positive rate: {y_test.mean():.2%}")

# Load preprocessor
preprocessor = joblib.load('artifacts/preprocessor.pkl')
print("✓ Preprocessor loaded")

# Configure MLflow
mlflow.set_experiment("baseline_modeling")
print(f"✓ MLflow experiment: baseline_modeling")

# ============================================================================
# BASELINE 1: DummyClassifier
# ============================================================================
print("\n" + "="*70)
print("TRAINING DUMMY CLASSIFIER...")
print("="*70)

with mlflow.start_run(run_name="dummy_classifier_baseline"):
    # Train DummyClassifier
    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X_train, y_train)
    
    # Predictions
    y_pred_dummy = dummy.predict(X_test)
    y_pred_proba_dummy = dummy.predict_proba(X_test)[:, 1]
    
    # Metrics
    dummy_auc = roc_auc_score(y_test, y_pred_proba_dummy)
    dummy_pr_auc = average_precision_score(y_test, y_pred_proba_dummy)
    dummy_f1 = f1_score(y_test, y_pred_dummy)
    dummy_precision = precision_score(y_test, y_pred_dummy)
    dummy_recall = recall_score(y_test, y_pred_dummy)
    
    # Log to MLflow
    mlflow.log_param("model_type", "DummyClassifier")
    mlflow.log_param("strategy", "most_frequent")
    
    mlflow.log_metric("auc_roc", dummy_auc)
    mlflow.log_metric("pr_auc", dummy_pr_auc)
    mlflow.log_metric("f1", dummy_f1)
    mlflow.log_metric("precision", dummy_precision)
    mlflow.log_metric("recall", dummy_recall)
    
    mlflow.sklearn.log_model(dummy, "model")
    
    print(f"\nDUMMY CLASSIFIER RESULTS:")
    print(f"  AUC-ROC:  {dummy_auc:.4f}")
    print(f"  PR-AUC:   {dummy_pr_auc:.4f}")
    print(f"  F1:       {dummy_f1:.4f}")
    print(f"  Precision: {dummy_precision:.4f}")
    print(f"  Recall:   {dummy_recall:.4f}")
    print(f"✓ Logged to MLflow")

# ============================================================================
# BASELINE 2: Logistic Regression
# ============================================================================
print("\n" + "="*70)
print("TRAINING LOGISTIC REGRESSION...")
print("="*70)

with mlflow.start_run(run_name="logistic_regression_baseline"):
    # Train Logistic Regression
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    lr.fit(X_train, y_train)
    
    # Predictions
    y_pred_lr = lr.predict(X_test)
    y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]
    
    # Metrics
    lr_auc = roc_auc_score(y_test, y_pred_proba_lr)
    lr_pr_auc = average_precision_score(y_test, y_pred_proba_lr)
    lr_f1 = f1_score(y_test, y_pred_lr)
    lr_precision = precision_score(y_test, y_pred_lr)
    lr_recall = recall_score(y_test, y_pred_lr)
    
    # Log to MLflow
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("solver", "lbfgs")
    
    mlflow.log_metric("auc_roc", lr_auc)
    mlflow.log_metric("pr_auc", lr_pr_auc)
    mlflow.log_metric("f1", lr_f1)
    mlflow.log_metric("precision", lr_precision)
    mlflow.log_metric("recall", lr_recall)
    
    mlflow.sklearn.log_model(lr, "model")
    
    print(f"\nLOGISTIC REGRESSION RESULTS:")
    print(f"  AUC-ROC:  {lr_auc:.4f}")
    print(f"  PR-AUC:   {lr_pr_auc:.4f}")
    print(f"  F1:       {lr_f1:.4f}")
    print(f"  Precision: {lr_precision:.4f}")
    print(f"  Recall:   {lr_recall:.4f}")
    print(f"✓ Logged to MLflow")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*70)
print("BASELINE COMPARISON")
print("="*70)

comparison = pd.DataFrame({
    'Model': ['DummyClassifier', 'Logistic Regression'],
    'AUC-ROC': [dummy_auc, lr_auc],
    'PR-AUC': [dummy_pr_auc, lr_pr_auc],
    'F1': [dummy_f1, lr_f1],
    'Precision': [dummy_precision, lr_precision],
    'Recall': [dummy_recall, lr_recall]
})

print(comparison.to_string(index=False))

auc_improvement = (lr_auc - dummy_auc) / dummy_auc * 100
pr_auc_improvement = (lr_pr_auc - dummy_pr_auc) / dummy_pr_auc * 100

print(f"\nAUC-ROC Improvement: {auc_improvement:.2f}%")
print(f"PR-AUC Improvement:  {pr_auc_improvement:.2f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS...")
print("="*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
fpr_dummy, tpr_dummy, _ = roc_curve(y_test, y_pred_proba_dummy)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)

ax1.plot(fpr_dummy, tpr_dummy, label=f'DummyClassifier (AUC={dummy_auc:.4f})', linewidth=2)
ax1.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={lr_auc:.4f})', linewidth=2)
ax1.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve Comparison')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Metrics comparison
metrics = ['AUC-ROC', 'PR-AUC', 'F1', 'Precision', 'Recall']
dummy_scores = [dummy_auc, dummy_pr_auc, dummy_f1, dummy_precision, dummy_recall]
lr_scores = [lr_auc, lr_pr_auc, lr_f1, lr_precision, lr_recall]

x = np.arange(len(metrics))
width = 0.35

ax2.bar(x - width/2, dummy_scores, width, label='DummyClassifier', alpha=0.8)
ax2.bar(x + width/2, lr_scores, width, label='Logistic Regression', alpha=0.8)
ax2.set_ylabel('Score')
ax2.set_title('Metrics Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('notebooks/baseline_comparison.png', dpi=100, bbox_inches='tight')
print("✓ Saved: notebooks/baseline_comparison.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PHASE 3 - BASELINE MODELING COMPLETE")
print("="*70)

print(f"\n✓ DummyClassifier (majority class predictor):")
print(f"    AUC-ROC: {dummy_auc:.4f} (floor baseline)")
print(f"    PR-AUC:  {dummy_pr_auc:.4f}")

print(f"\n✓ Logistic Regression (linear baseline):")
print(f"    AUC-ROC: {lr_auc:.4f} (+{auc_improvement:.1f}%)")
print(f"    PR-AUC:  {lr_pr_auc:.4f} (+{pr_auc_improvement:.1f}%)")

print(f"\n✓ Both models logged to MLflow")
print(f"✓ Test set metrics tracked (never touched during training)")
print(f"✓ Ready for Phase 4: Imbalance Handling")

print(f"\nKey Insight:")
print(f"  Class distribution: 87.74% negative, 12.26% positive")
print(f"  DummyClassifier achieves ~{dummy_auc:.2f} AUC-ROC (useless on imbalanced data)")
print(f"  Logistic Regression beats it but still has room for improvement")
print(f"  → Next: Handle class imbalance with SMOTE + scale_pos_weight")

print("="*70)
