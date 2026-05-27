#!/usr/bin/env python3
"""
Phase 5 - Hyperparameter Tuning with Optuna
Optimize LightGBM (selected model from Phase 4) with 50 trials
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score
)
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import mlflow
import mlflow.lightgbm
import logging
from time import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("\n" + "="*80)
print("PHASE 5 - HYPERPARAMETER TUNING (LightGBM)")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
logging.info("Loading processed data...")

data_dir = 'data/processed'
X_train = np.load(f'{data_dir}/X_train.npy')
X_test = np.load(f'{data_dir}/X_test.npy')

y_train = np.load(f'{data_dir}/y_train.npy')
y_test = np.load(f'{data_dir}/y_test.npy')

print(f"Train: {X_train.shape} | Positive: {y_train.mean():.2%}")
print(f"Test:  {X_test.shape} | Positive: {y_test.mean():.2%}")

# Calculate class weight
pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
scale_pos_weight = neg_count / pos_count

print(f"Class weight (scale_pos_weight): {scale_pos_weight:.2f}")

# ============================================================================
# 2. CONFIGURE MLFLOW
# ============================================================================
logging.info("Configuring MLflow...")
mlflow.set_experiment("hyperparameter_tuning")

# ============================================================================
# 3. DEFINE OPTUNA OBJECTIVE FUNCTION
# ============================================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    """Optuna objective: maximize PR-AUC on 5-fold cross-validation"""
    
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
    }
    
    model = lgb.LGBMClassifier(
        n_estimators=200,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
        **params
    )
    
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_cv_train = X_train[train_idx]
        y_cv_train = y_train[train_idx]
        X_cv_val = X_train[val_idx]
        y_cv_val = y_train[val_idx]
        
        model.fit(X_cv_train, y_cv_train)
        y_pred_proba = model.predict_proba(X_cv_val)[:, 1]
        
        pr_auc = average_precision_score(y_cv_val, y_pred_proba)
        scores.append(pr_auc)
    
    return np.mean(scores)

print("✓ Objective function defined")

# ============================================================================
# 4. RUN OPTUNA STUDY
# ============================================================================

print("\n" + "-"*80)
print("Starting Optuna optimization (50 trials)...")
print("-"*80)

t0 = time()

sampler = TPESampler(seed=42, n_startup_trials=10)
pruner = MedianPruner(n_warmup_steps=5)

study = optuna.create_study(
    direction='maximize',
    sampler=sampler,
    pruner=pruner
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

tune_time = time() - t0

print("-"*80)
print(f"\n✓ Optimization complete in {tune_time:.2f}s")

best_trial = study.best_trial
best_pr_auc_cv = best_trial.value

print(f"\n🏆 Best Trial: {best_trial.number}")
print(f"   CV PR-AUC: {best_pr_auc_cv:.4f}")
print(f"\nBest Hyperparameters:")
for param in sorted(best_trial.params.keys()):
    value = best_trial.params[param]
    if isinstance(value, float):
        print(f"  {param:20} = {value:.6f}")
    else:
        print(f"  {param:20} = {value}")

# ============================================================================
# 5. TRAIN FINAL MODEL
# ============================================================================

with mlflow.start_run(run_name="lgb_tuned_final"):
    logging.info("\nTraining final LightGBM with best hyperparameters...")
    
    t0 = time()
    
    final_model = lgb.LGBMClassifier(
        n_estimators=200,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
        **best_trial.params
    )
    
    final_model.fit(X_train, y_train)
    final_train_time = time() - t0
    
    # Predictions
    y_pred_final = final_model.predict(X_test)
    y_pred_proba_final = final_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    final_auc = roc_auc_score(y_test, y_pred_proba_final)
    final_pr_auc = average_precision_score(y_test, y_pred_proba_final)
    final_f1 = f1_score(y_test, y_pred_final)
    final_precision = precision_score(y_test, y_pred_final)
    final_recall = recall_score(y_test, y_pred_final)
    
    # Log to MLflow
    mlflow.log_param("model_type", "LightGBM_Tuned")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("scale_pos_weight", f"{scale_pos_weight:.2f}")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("optuna_trials", 50)
    
    for param, value in best_trial.params.items():
        mlflow.log_param(f"opt_{param}", value)
    
    mlflow.log_metric("cv_best_pr_auc", best_pr_auc_cv)
    mlflow.log_metric("test_auc_roc", final_auc)
    mlflow.log_metric("test_pr_auc", final_pr_auc)
    mlflow.log_metric("test_f1", final_f1)
    mlflow.log_metric("test_precision", final_precision)
    mlflow.log_metric("test_recall", final_recall)
    mlflow.log_metric("train_time_sec", final_train_time)
    mlflow.log_metric("tune_time_sec", tune_time)
    
    mlflow.lightgbm.log_model(final_model, "model")
    
    print(f"\n✓ Final model trained in {final_train_time:.2f}s")
    print(f"\n  Test Metrics:")
    print(f"    AUC-ROC:   {final_auc:.4f}")
    print(f"    PR-AUC:    {final_pr_auc:.4f}")
    print(f"    F1:        {final_f1:.4f}")
    print(f"    Precision: {final_precision:.4f}")
    print(f"    Recall:    {final_recall:.4f}")

# ============================================================================
# 6. COMPARISON: BASELINE VS TUNED
# ============================================================================

# Baseline from Phase 4
baseline_auc = 0.858583
baseline_pr_auc = 0.364858
baseline_f1 = 0.435652

# Baseline model to get baseline predictions
baseline_model = lgb.LGBMClassifier(
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
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)
baseline_precision = precision_score(y_test, y_pred_baseline)
baseline_recall = recall_score(y_test, y_pred_baseline)

auc_improvement = ((final_auc - baseline_auc) / baseline_auc) * 100
pr_auc_improvement = ((final_pr_auc - baseline_pr_auc) / baseline_pr_auc) * 100
f1_improvement = ((final_f1 - baseline_f1) / baseline_f1) * 100

print("\n" + "="*80)
print("BASELINE vs TUNED MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': ['LightGBM Baseline', 'LightGBM Tuned'],
    'AUC-ROC': [baseline_auc, final_auc],
    'PR-AUC': [baseline_pr_auc, final_pr_auc],
    'F1': [baseline_f1, final_f1],
    'Precision': [baseline_precision, final_precision],
    'Recall': [baseline_recall, final_recall],
})

print(comparison_df.to_string(index=False))
print("="*80)

print(f"\nImprovements from Tuning:")
print(f"  AUC-ROC:  {auc_improvement:+.2f}%")
print(f"  PR-AUC:   {pr_auc_improvement:+.2f}%")
print(f"  F1:       {f1_improvement:+.2f}%")

# ============================================================================
# 7. CREATE VISUALIZATION
# ============================================================================

logging.info("Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Optimization history
ax = axes[0, 0]
trial_numbers = [trial.number for trial in study.trials]
trial_values = [trial.value for trial in study.trials]
ax.plot(trial_numbers, trial_values, marker='o', linestyle='-', alpha=0.6, markersize=4)
ax.axhline(y=best_pr_auc_cv, color='r', linestyle='--', label=f'Best: {best_pr_auc_cv:.4f}')
ax.set_xlabel('Trial')
ax.set_ylabel('PR-AUC (5-fold CV)')
ax.set_title('Optimization History')
ax.legend()
ax.grid(True, alpha=0.3)

# Baseline vs Tuned comparison
ax = axes[0, 1]
metrics = ['AUC-ROC', 'PR-AUC', 'F1']
baseline = [baseline_auc, baseline_pr_auc, baseline_f1]
tuned = [final_auc, final_pr_auc, final_f1]
x = np.arange(len(metrics))
width = 0.35
ax.bar(x - width/2, baseline, width, label='Baseline', alpha=0.8)
ax.bar(x + width/2, tuned, width, label='Tuned', alpha=0.8)
ax.set_ylabel('Score')
ax.set_title('Baseline vs Tuned Model')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

# Top hyperparameters
ax = axes[1, 0]
param_values = {}
for trial in study.trials:
    for param, value in trial.params.items():
        if param not in param_values:
            param_values[param] = []
        param_values[param].append(value)

param_ranges = {param: (min(vals), max(vals)) for param, vals in param_values.items()}
sorted_params = sorted(param_ranges.items(), key=lambda x: x[1][1] - x[1][0], reverse=True)[:5]

param_names = [p[0] for p in sorted_params]
param_range_sizes = [p[1][1] - p[1][0] for p in sorted_params]
ax.barh(param_names, param_range_sizes, alpha=0.8, color='steelblue')
ax.set_xlabel('Parameter Range')
ax.set_title('Top 5 Explored Hyperparameters')
ax.grid(True, alpha=0.3, axis='x')

# Improvement metrics
ax = axes[1, 1]
improvements = [auc_improvement, pr_auc_improvement, f1_improvement]
colors = ['green' if x > 0 else 'red' for x in improvements]
ax.barh(['AUC-ROC', 'PR-AUC', 'F1'], improvements, color=colors, alpha=0.8)
ax.set_xlabel('Improvement (%)')
ax.set_title('Tuning Improvements')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('notebooks/hyperparameter_tuning.png', dpi=100, bbox_inches='tight')
print("✓ Saved: notebooks/hyperparameter_tuning.png")
plt.close()

# ============================================================================
# 8. PHASE 5 SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PHASE 5 - HYPERPARAMETER TUNING COMPLETE")
print("="*80)

print(f"\n✓ Used Optuna with TPESampler (50 trials)")
print(f"✓ 5-fold Stratified Cross-Validation on training data")
print(f"✓ Tuning time: {tune_time:.2f}s")

print(f"\n🏆 Best Hyperparameters (from Trial {best_trial.number}):")
for param in sorted(best_trial.params.keys()):
    value = best_trial.params[param]
    if isinstance(value, float):
        print(f"   {param:20} = {value:.6f}")
    else:
        print(f"   {param:20} = {value}")

print(f"\n📊 Final Model Performance (Test Set):")
print(f"   AUC-ROC:   {final_auc:.4f}")
print(f"   PR-AUC:    {final_pr_auc:.4f}")
print(f"   F1:        {final_f1:.4f}")
print(f"   Precision: {final_precision:.4f}")
print(f"   Recall:    {final_recall:.4f}")

print(f"\n📈 Improvements vs Baseline:")
print(f"   AUC-ROC:  {auc_improvement:+.2f}%")
print(f"   PR-AUC:   {pr_auc_improvement:+.2f}%")
print(f"   F1:       {f1_improvement:+.2f}%")

print(f"\n✓ Tuned model logged to MLflow")
print(f"✓ Optimization history and comparisons visualized")
print(f"✓ Ready for Phase 6: Model Calibration")

print("\n" + "="*80)

# Save results
comparison_df.to_csv('data/hyperparameter_tuning_results.csv', index=False)
print("✓ Saved: data/hyperparameter_tuning_results.csv")
