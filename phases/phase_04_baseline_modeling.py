#!/usr/bin/env python3
"""
Phase 4 - Model Comparison Script
Runs without Jupyter to avoid issues and logs everything
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_curve, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import logging
import joblib
from time import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("\n" + "="*80)
print("PHASE 4 - MODEL COMPARISON & SELECTION")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
logger.info("Loading processed data...")

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

print(f"Class weight (scale_pos_weight): {scale_pos_weight:.2f}")

# ============================================================================
# 2. CONFIGURE MLFLOW
# ============================================================================
logger.info("Configuring MLflow...")
mlflow.set_experiment("model_comparison")

# ============================================================================
# 3. TRAIN MODEL 1: XGBOOST
# ============================================================================
logger.info("\n" + "-"*80)
logger.info("Training XGBoost...")
logger.info("-"*80)

with mlflow.start_run(run_name="xgb_model_comparison"):
    t0 = time()
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    train_time_xgb = time() - t0
    
    # Predictions
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)
    xgb_pr_auc = average_precision_score(y_test, y_pred_proba_xgb)
    xgb_f1 = f1_score(y_test, y_pred_xgb)
    xgb_precision = precision_score(y_test, y_pred_xgb)
    xgb_recall = recall_score(y_test, y_pred_xgb)
    
    # Log
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("scale_pos_weight", f"{scale_pos_weight:.2f}")
    
    mlflow.log_metric("auc_roc", xgb_auc)
    mlflow.log_metric("pr_auc", xgb_pr_auc)
    mlflow.log_metric("f1", xgb_f1)
    mlflow.log_metric("precision", xgb_precision)
    mlflow.log_metric("recall", xgb_recall)
    mlflow.log_metric("train_time_sec", train_time_xgb)
    
    mlflow.xgboost.log_model(xgb_model, "model")
    
    print(f"✓ XGBoost trained in {train_time_xgb:.2f}s")
    print(f"  AUC-ROC:  {xgb_auc:.4f}")
    print(f"  PR-AUC:   {xgb_pr_auc:.4f}")
    print(f"  F1:       {xgb_f1:.4f}")
    print(f"  Precision: {xgb_precision:.4f}")
    print(f"  Recall:   {xgb_recall:.4f}")

# ============================================================================
# 4. TRAIN MODEL 2: LIGHTGBM
# ============================================================================
logger.info("\n" + "-"*80)
logger.info("Training LightGBM...")
logger.info("-"*80)

with mlflow.start_run(run_name="lgb_model_comparison"):
    t0 = time()
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    train_time_lgb = time() - t0
    
    # Predictions
    y_pred_lgb = lgb_model.predict(X_test)
    y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    lgb_auc = roc_auc_score(y_test, y_pred_proba_lgb)
    lgb_pr_auc = average_precision_score(y_test, y_pred_proba_lgb)
    lgb_f1 = f1_score(y_test, y_pred_lgb)
    lgb_precision = precision_score(y_test, y_pred_lgb)
    lgb_recall = recall_score(y_test, y_pred_lgb)
    
    # Log
    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("num_leaves", 31)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("scale_pos_weight", f"{scale_pos_weight:.2f}")
    
    mlflow.log_metric("auc_roc", lgb_auc)
    mlflow.log_metric("pr_auc", lgb_pr_auc)
    mlflow.log_metric("f1", lgb_f1)
    mlflow.log_metric("precision", lgb_precision)
    mlflow.log_metric("recall", lgb_recall)
    mlflow.log_metric("train_time_sec", train_time_lgb)
    
    mlflow.lightgbm.log_model(lgb_model, "model")
    
    print(f"✓ LightGBM trained in {train_time_lgb:.2f}s")
    print(f"  AUC-ROC:  {lgb_auc:.4f}")
    print(f"  PR-AUC:   {lgb_pr_auc:.4f}")
    print(f"  F1:       {lgb_f1:.4f}")
    print(f"  Precision: {lgb_precision:.4f}")
    print(f"  Recall:   {lgb_recall:.4f}")

# ============================================================================
# 5. TRAIN MODEL 3: RANDOMFOREST
# ============================================================================
logger.info("\n" + "-"*80)
logger.info("Training RandomForest...")
logger.info("-"*80)

with mlflow.start_run(run_name="rf_model_comparison"):
    t0 = time()
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    train_time_rf = time() - t0
    
    # Predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    rf_auc = roc_auc_score(y_test, y_pred_proba_rf)
    rf_pr_auc = average_precision_score(y_test, y_pred_proba_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)
    rf_precision = precision_score(y_test, y_pred_rf)
    rf_recall = recall_score(y_test, y_pred_rf)
    
    # Log
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("class_weight", "balanced")
    
    mlflow.log_metric("auc_roc", rf_auc)
    mlflow.log_metric("pr_auc", rf_pr_auc)
    mlflow.log_metric("f1", rf_f1)
    mlflow.log_metric("precision", rf_precision)
    mlflow.log_metric("recall", rf_recall)
    mlflow.log_metric("train_time_sec", train_time_rf)
    
    mlflow.sklearn.log_model(rf_model, "model")
    
    print(f"✓ RandomForest trained in {train_time_rf:.2f}s")
    print(f"  AUC-ROC:  {rf_auc:.4f}")
    print(f"  PR-AUC:   {rf_pr_auc:.4f}")
    print(f"  F1:       {rf_f1:.4f}")
    print(f"  Precision: {rf_precision:.4f}")
    print(f"  Recall:   {rf_recall:.4f}")

# ============================================================================
# 6. COMPARE RESULTS
# ============================================================================
logger.info("\n" + "="*80)
logger.info("MODEL COMPARISON RESULTS")
logger.info("="*80)

comparison_df = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'RandomForest'],
    'AUC-ROC': [xgb_auc, lgb_auc, rf_auc],
    'PR-AUC': [xgb_pr_auc, lgb_pr_auc, rf_pr_auc],
    'F1': [xgb_f1, lgb_f1, rf_f1],
    'Precision': [xgb_precision, lgb_precision, rf_precision],
    'Recall': [xgb_recall, lgb_recall, rf_recall],
    'Train Time (s)': [train_time_xgb, train_time_lgb, train_time_rf]
})

print(comparison_df.to_string(index=False))
print("="*80)

# Find best model
best_idx = comparison_df['PR-AUC'].idxmax()
best_model_name = comparison_df.loc[best_idx, 'Model']
best_pr_auc = comparison_df.loc[best_idx, 'PR-AUC']

print(f"\n🏆 BEST MODEL: {best_model_name} (PR-AUC: {best_pr_auc:.4f})")

# ============================================================================
# 7. CREATE VISUALIZATION
# ============================================================================
logger.info("Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# AUC-ROC comparison
ax = axes[0, 0]
ax.bar(comparison_df['Model'], comparison_df['AUC-ROC'], alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC-ROC Comparison')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

# PR-AUC comparison (PRIMARY metric)
ax = axes[0, 1]
ax.bar(comparison_df['Model'], comparison_df['PR-AUC'], alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel('PR-AUC')
ax.set_title('PR-AUC Comparison (Primary Metric)')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

# F1 vs Precision vs Recall
ax = axes[1, 0]
x = np.arange(len(comparison_df))
width = 0.25
ax.bar(x - width, comparison_df['F1'], width, label='F1', alpha=0.8)
ax.bar(x, comparison_df['Precision'], width, label='Precision', alpha=0.8)
ax.bar(x + width, comparison_df['Recall'], width, label='Recall', alpha=0.8)
ax.set_ylabel('Score')
ax.set_title('F1 vs Precision vs Recall')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Training time
ax = axes[1, 1]
ax.bar(comparison_df['Model'], comparison_df['Train Time (s)'], alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel('Training Time (seconds)')
ax.set_title('Training Time Comparison')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/model_comparison.png', dpi=100, bbox_inches='tight')
print("✓ Saved: notebooks/model_comparison.png")
plt.close()

# ============================================================================
# 8. PHASE 4 SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 4 - MODEL COMPARISON & SELECTION COMPLETE")
print("="*80)

print(f"\n✓ Trained 3 models with imbalance handling:")
print(f"  1. XGBoost (scale_pos_weight={scale_pos_weight:.2f})")
print(f"  2. LightGBM (scale_pos_weight={scale_pos_weight:.2f})")
print(f"  3. RandomForest (class_weight='balanced')")

print(f"\n✓ Results (PR-AUC primary metric):")
for idx, row in comparison_df.iterrows():
    print(f"  {row['Model']:15} PR-AUC: {row['PR-AUC']:.4f} | F1: {row['F1']:.4f} | Train: {row['Train Time (s)']:.2f}s")

print(f"\n🏆 SELECTED MODEL: {best_model_name}")
print(f"   Reason: Highest PR-AUC ({best_pr_auc:.4f}) for imbalanced classification")

print(f"\n✓ All 3 models logged to MLflow")
print(f"✓ Comparison visualization saved")
print(f"✓ Ready for Phase 5: Hyperparameter Tuning on {best_model_name}")

print("\n" + "="*80)

# Save comparison results
comparison_df.to_csv('data/model_comparison_results.csv', index=False)
print("✓ Saved: data/model_comparison_results.csv")
