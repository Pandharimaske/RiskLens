# 🛡️ Vehicle Insurance Claim Prediction — Project Roadmap
> Build it once. Build it right. Make it industry gold.

This roadmap takes you from zero to a fully deployed, monitored, production-grade ML system
with a frontend — covering every gap an industry interviewer would probe.

**Estimated total time:** 12–15 focused days  
**Stack:** Python · XGBoost · LightGBM · Optuna · MLflow · DagsHub · DVC · SHAP · Evidently AI · FastAPI · Streamlit · Docker · GitHub Actions · AWS ECR + EC2

---

## How to Use This Roadmap

Each phase has:
- **Goal** — what you're building
- **Steps** — exact tasks in order
- **Checkpoint** — how you know you're done before moving on
- **Interview talking point** — what this phase lets you say confidently

**📌 PROJECT NAME:** RiskLens  
**📌 ENVIRONMENT:** Using `uv` for Python package management  
**📌 PROCESS:** Complete one phase at a time. Delete completed phases from this file after each checkpoint. Commit meaningfully at phase end.

Work phase by phase. Do not skip ahead.  
Commit at the end of every phase with a meaningful message.

---

## ✅ Phase 2 — Feature Engineering (COMPLETE)

**Status:** ✓ Completed on Day 2

**Deliverables:**
- ✓ 7 domain features created (Vehicle_Age_Numeric, Premium_per_Vehicle_Year, High_Value_Vehicle, Age_Risk_Bucket, Customer_Tenure_Segment, Premium_Bucket, Damage_History_Risk)
- ✓ Preprocessing pipeline built with StandardScaler + OrdinalEncoder
- ✓ Stratified train/val/test splits (70%/15%/15%) with class balance maintained
- ✓ Preprocessor fitted on training data only (prevents data leakage)
- ✓ Processed data saved: 266,775 train, 57,167 val, 57,167 test samples
- ✓ 17 features total (11 numeric + 6 categorical) after engineering
- ✓ Git commit: "feat: Phase 2 - Feature Engineering complete"

**Key Files:**
- `src/features/engineering.py` — Feature engineering & preprocessing module
- `data/processed/` — 12 processed data files (.npy & .pkl formats)
- `artifacts/preprocessor.pkl` — Fitted sklearn ColumnTransformer
- `notebooks/02_feature_engineering.ipynb` — Interactive workflow

**Interview talking point:**  
*"I built a scikit-learn preprocessing pipeline that bundles with the model for deployment — no leakage, features fit only on training data. Seven domain features engineered from first principles: premium-per-vehicle-year captures exposure, age buckets capture underwriting risk, and damage-history interactions model prior claims behavior."*

---

## ✅ Phase 3 — Baseline Modeling (COMPLETE)

**Status:** ✓ Completed on Day 3

**Results:**
- **DummyClassifier (Majority Class):**
  - AUC-ROC: 0.5000 (floor baseline)
  - PR-AUC: 0.1226
  - Precision/Recall: 0.0 (predicts all negative)
  
- **Logistic Regression (Linear Model):**
  - AUC-ROC: 0.8437 (**+68.7% vs dummy**)
  - PR-AUC: 0.3336 (**+172.2% vs dummy**)
  - Precision: 0.3889
  - Recall: 0.0010

**Key Insights:**
- Logistic Regression beats the dummy baseline significantly on both AUC-ROC and PR-AUC
- However, low recall (0.1%) shows imbalance handling is needed in Phase 4
- Class imbalance (87.74% negative, 12.26% positive) requires proper model selection first
- Both models logged to MLflow with full metrics

**Files Created:**
- `baseline_modeling.py` — Standalone baseline script
- `notebooks/03_baseline_modeling.ipynb` — Interactive notebook
- `notebooks/baseline_comparison.png` — ROC curves + metrics visualization
- MLflow experiment: `baseline_modeling` with 2 logged runs

**Interview talking point:**  
*"I established two baselines: a DummyClassifier to prove my model adds value, and Logistic Regression as a simple linear baseline. Both were logged to MLflow. LR beats the dummy by 68.7% on AUC-ROC, but the low recall (0.1%) shows imbalance handling is critical before tuning."*

---

## Phase 4 — Model Comparison + Imbalance Handling (Days 4-5, ~5 hours)

**Goal:** Test multiple models with class imbalance handling, select the best performer before hyperparameter tuning.

### Strategy
Instead of tuning one model extensively, first compare 3+ models with basic imbalance handling:
1. XGBoost + scale_pos_weight
2. LightGBM + scale_pos_weight  
3. RandomForest + class_weight
4. (Optional) CatBoost + scale_pos_weight

Then select winner and move to Phase 5 for hyperparameter tuning.

### Steps

**4.1 — Imbalance strategies**

| Strategy | Implementation | Best for |
|---|---|---|
| `scale_pos_weight` | XGB/LGB native param | Tree models (zero overhead) |
| `class_weight='balanced'` | sklearn native param | Linear/Forest models |
| SMOTE | In imblearn Pipeline | When native strategies insufficient |

**4.2 — Create baseline models with imbalance handling**

```python
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Calculate class weight
pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
scale_pos_weight = neg_count / pos_count

# Model 1: XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# Model 2: LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# Model 3: RandomForest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
```

**4.3 — Compare on test set**

Train each model on X_train, evaluate on X_test:
- AUC-ROC
- PR-AUC (PRIMARY metric for imbalanced data)
- F1
- Precision / Recall

Log all to MLflow in separate runs.

**4.4 — Select winner**

Choose model with highest PR-AUC (best for imbalanced classification).

### Checkpoint
- [ ] 3+ models trained with imbalance handling
- [ ] All models logged to MLflow
- [ ] Test metrics compared (AUC-ROC, PR-AUC, F1)
- [ ] Best model identified and documented
- [ ] Comparison table/visualization created

**Interview talking point:**  
*"Before hyperparameter tuning, I compared 3 models (XGBoost, LightGBM, RandomForest) with native class imbalance handling. Each used appropriate strategies: scale_pos_weight for tree models, class_weight for RF. LightGBM had the highest PR-AUC on test set, so I selected it for Phase 5 hyperparameter optimization."*

---

## Phase 5 — Hyperparameter Tuning with Optuna (Day 5, ~3 hours)

**Goal:** Optimize the selected model's hyperparameters using Bayesian search.

### Steps

**5.1 — Define Optuna objective function (for selected model)**

Example for LightGBM (adjust params based on selected model):

```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        
        model = lgb.LGBMClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,  # Keep from Phase 4
            n_estimators=200,
            random_state=42
        )
        
        # Use cross-validation on full training set
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=5, 
            scoring='average_precision'  # PR-AUC
        )
        pr_auc_mean = scores.mean()
        mlflow.log_metric("cv_pr_auc_mean", pr_auc_mean)
        mlflow.log_metric("cv_pr_auc_std", scores.std())
        
    return pr_auc_mean

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
```

**5.2 — Train final model with best params**

```python
best_model = lgb.LGBMClassifier(
    **study.best_params,
    scale_pos_weight=scale_pos_weight,
    n_estimators=300,  # Increase post-tuning
    random_state=42
)
best_model.fit(X_train, y_train)

# Evaluate on test set
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
best_pr_auc = average_precision_score(y_test, y_pred_proba)

mlflow.log_metric("test_pr_auc_final", best_pr_auc)
mlflow.sklearn.log_model(best_model, "best_model")
```

### Checkpoint
- [ ] 50 trials of Optuna completed
- [ ] Best hyperparameters identified
- [ ] Final model trained on full X_train
- [ ] Test set metrics logged to MLflow
- [ ] Model artifact saved

**Interview talking point:**  
*"I ran 50 Optuna trials with 5-fold cross-validation, optimizing for PR-AUC (not AUC-ROC). Bayesian search is more efficient than grid search on imbalanced data. PR-AUC improved from 0.45 → 0.72 after tuning."*

---

## Phase 6 — Calibration (Day 6, ~2 hours)

**Goal:** Make your model's probabilities trustworthy. This is what separates "classification" from "risk scoring."

### The Core Idea

If your model says P(claim) = 0.8 for 100 applicants, roughly 80 of them should actually claim.  
Out of the box, XGBoost and LightGBM often don't satisfy this — they're optimized for ranking (AUC), not calibration.  
In insurance, uncalibrated probabilities = wrong premiums.

### Steps

**6.1 — Visualize calibration before fixing it (`notebooks/06_calibration.ipynb`)**
```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_pred_proba, n_bins=10, strategy='uniform'
)

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.plot(mean_predicted_value, fraction_of_positives, 
         's-', label='XGBoost (uncalibrated)')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve — Before Calibration')
plt.legend()
plt.savefig('monitoring/reports/calibration_before.png')
```

If the curve bows away from the diagonal, your model is miscalibrated.

**6.2 — Apply calibration using CalibratedClassifierCV**
```python
from sklearn.calibration import CalibratedClassifierCV

# Note: calibrate on validation set, not training set
# Extract the trained classifier from your pipeline first
trained_clf = best_pipeline.named_steps['clf']
X_val_transformed = best_pipeline[:-1].transform(X_val)

calibrated_clf = CalibratedClassifierCV(
    estimator=trained_clf,
    method='isotonic',  # 'sigmoid' (Platt scaling) for small val sets
    cv='prefit'
)
calibrated_clf.fit(X_val_transformed, y_val)
```

**6.3 — Measure improvement with ECE (Expected Calibration Error)**
```python
def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.mean() * abs(bin_acc - bin_conf)
    return ece

ece_before = compute_ece(y_test, uncalibrated_probs)
ece_after = compute_ece(y_test, calibrated_probs)

print(f"ECE before: {ece_before:.4f}")
print(f"ECE after:  {ece_after:.4f}")
```

Log both to MLflow. The improvement here is your talking point.

**6.4 — Plot calibration curve after calibration**

Same code as 6.1 but with calibrated probabilities. Save as `calibration_after.png`.  
Include both plots side by side in your README.

### Checkpoint
- [ ] Calibration curve plotted before and after
- [ ] ECE computed and logged to MLflow
- [ ] Calibrated model saved separately: `artifacts/calibrated_pipeline.pkl`

**Interview talking point:**  
*"XGBoost was overconfident — ECE was 0.08 before calibration, dropped to 0.03 after isotonic regression. In an insurance context this matters because the probabilities feed directly into premium calculation — a miscalibrated model misprices risk."*

---

## Phase 7 — Cost-Sensitive Threshold Optimization (Day 6–7, ~3 hours)

**Goal:** Move beyond default 0.5 threshold using business logic. This is the most impressive phase — almost no student projects do this.

### The Core Idea

Default threshold = 0.5 means "predict claim if P(claim) > 50%."  
But is that right for insurance?

- **False Negative** (predict no-claim, actually claims): Company pays claim without having reserved funds → HIGH cost
- **False Positive** (predict claim, actually doesn't): Company charges higher premium or declines policy → Lower cost but customer loss

These costs are not equal. Optimal threshold is not 0.5.

### Steps

**7.1 — Define your cost matrix (`notebooks/07_threshold_optimization.ipynb`)**
```python
# Hypothetical costs — document your assumptions clearly
COST_FN = 50000  # Expected payout for a missed high-risk claim (INR)
COST_FP = 3000   # Cost of incorrectly declining a good customer (lost premium)
COST_TN = 0      # Correct rejection — no cost
COST_TP = -10000 # Correct identification — company sets aside reserves, avoids surprise
                 # Negative = benefit (avoided surprise claim payout)
```

**7.2 — Sweep thresholds and compute expected cost**
```python
thresholds = np.arange(0.01, 0.99, 0.01)
expected_costs = []

for threshold in thresholds:
    y_pred = (calibrated_probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    total_cost = (fn * COST_FN + fp * COST_FP + 
                  tn * COST_TN + tp * COST_TP)
    
    expected_costs.append({
        'threshold': threshold,
        'total_cost': total_cost,
        'fn': fn, 'fp': fp, 'tn': tn, 'tp': tp,
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    })

results_df = pd.DataFrame(expected_costs)
optimal_threshold = results_df.loc[results_df['total_cost'].idxmin(), 'threshold']
print(f"Optimal business threshold: {optimal_threshold:.2f}")
print(f"Default 0.5 threshold cost: {results_df[results_df.threshold==0.5]['total_cost'].values[0]:,}")
print(f"Optimal threshold cost:     {results_df['total_cost'].min():,}")
```

**7.3 — Plot the cost curve**
```python
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(results_df['threshold'], results_df['total_cost'])
plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
            label=f'Optimal: {optimal_threshold:.2f}')
plt.axvline(x=0.5, color='gray', linestyle='--', label='Default: 0.50')
plt.xlabel('Threshold')
plt.ylabel('Expected Cost (INR)')
plt.title('Business Cost vs Classification Threshold')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(results_df['threshold'], results_df['f1'], label='F1')
plt.plot(results_df['threshold'], results_df['precision'], label='Precision')
plt.plot(results_df['threshold'], results_df['recall'], label='Recall')
plt.axvline(x=optimal_threshold, color='red', linestyle='--')
plt.xlabel('Threshold')
plt.legend()
plt.tight_layout()
plt.savefig('monitoring/reports/threshold_optimization.png')
```

**7.4 — Save the optimal threshold to `configs/params.yaml`**
```yaml
model:
  optimal_threshold: 0.35  # replace with your value
  calibration_method: isotonic
  imbalance_strategy: smote
```

The FastAPI endpoint will load this threshold instead of hardcoding 0.5.

### Checkpoint
- [ ] Cost matrix defined with documented assumptions
- [ ] Threshold sweep run and plotted
- [ ] Optimal threshold computed and saved to params.yaml
- [ ] Quantified how much cost the optimal threshold saves vs default 0.5

**Interview talking point:**  
*"At default 0.5, the model had an expected cost of ₹X. After sweeping thresholds using a cost matrix where FN cost was ₹50,000 and FP cost was ₹3,000 — reflecting the actual asymmetry in insurance — the optimal threshold of 0.35 reduced expected cost by 23%. This is how you translate ML metrics into business value."*

---

## Phase 8 — SHAP Explainability (Day 7, ~2 hours)

**Goal:** Make the model interpretable. Log explainability artifacts.

### Steps

**8.1 — Global explainability (`notebooks/08_evaluation_final.ipynb`)**
```python
import shap

# Get the trained classifier from pipeline
clf = best_calibrated_pipeline.named_steps['clf']
X_test_transformed = best_calibrated_pipeline[:-1].transform(X_test)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_transformed)

# Summary plot — top features
shap.summary_plot(shap_values, X_test_transformed, 
                  feature_names=feature_names,
                  show=False)
plt.savefig('monitoring/reports/shap_summary.png', bbox_inches='tight')
mlflow.log_artifact('monitoring/reports/shap_summary.png')
```

**8.2 — Local explainability (single prediction)**
```python
# Explain one specific prediction
idx = 42  # some test sample
shap.waterfall_plot(shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_test_transformed[idx],
    feature_names=feature_names
), show=False)
plt.savefig('monitoring/reports/shap_local_example.png', bbox_inches='tight')
```

**8.3 — Log top 5 predictive features as MLflow metrics**
```python
mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0), 
    index=feature_names
).sort_values(ascending=False)

for i, (feature, importance) in enumerate(mean_abs_shap.head(5).items()):
    mlflow.log_metric(f"shap_rank_{i+1}_{feature}", importance)
```

### Checkpoint
- [ ] SHAP summary plot saved and logged to MLflow
- [ ] Local waterfall plot for one prediction saved
- [ ] Top 5 features documented in README

---

## Phase 9 — DVC Pipeline (Day 8, ~2 hours)

**Goal:** Replace manual notebook execution with a reproducible, versioned pipeline.

### Steps

**9.1 — Convert your notebooks to scripts in `src/`**

Each script reads from `data/` and writes outputs to `data/processed/` or `artifacts/`.  
Pass parameters from `configs/params.yaml` using the `yaml` library.

**9.2 — Define `pipeline/dvc.yaml`**
```yaml
stages:
  data_ingestion:
    cmd: python src/data/ingestion.py
    deps:
      - src/data/ingestion.py
      - configs/params.yaml
    outs:
      - data/raw/train.csv

  data_validation:
    cmd: python src/data/validation.py
    deps:
      - src/data/validation.py
      - data/raw/train.csv
    outs:
      - data/processed/validation_report.json

  feature_engineering:
    cmd: python src/features/engineering.py
    deps:
      - src/features/engineering.py
      - data/raw/train.csv
      - configs/params.yaml
    outs:
      - data/processed/X_train.pkl
      - data/processed/X_test.pkl
      - data/processed/y_train.pkl
      - data/processed/y_test.pkl
      - artifacts/preprocessor.pkl

  train:
    cmd: python src/models/train.py
    deps:
      - src/models/train.py
      - data/processed/X_train.pkl
      - data/processed/y_train.pkl
      - configs/params.yaml
    outs:
      - artifacts/model_pipeline.pkl
      - artifacts/calibrated_pipeline.pkl

  evaluate:
    cmd: python src/models/evaluate.py
    deps:
      - src/models/evaluate.py
      - data/processed/X_test.pkl
      - data/processed/y_test.pkl
      - artifacts/calibrated_pipeline.pkl
    metrics:
      - metrics/eval_metrics.json:
          cache: false
    plots:
      - monitoring/reports/calibration_after.png
      - monitoring/reports/shap_summary.png
      - monitoring/reports/threshold_optimization.png
```

**9.3 — Run and verify**
```bash
dvc repro          # runs only changed stages
dvc dag            # visualize the pipeline DAG
dvc metrics show   # show evaluation metrics
```

Push everything:
```bash
dvc push
git add .
git commit -m "feat: complete DVC pipeline"
git push
```

### Checkpoint
- [ ] `dvc repro` runs end-to-end without errors
- [ ] `dvc dag` shows the correct pipeline graph
- [ ] All artifacts tracked by DVC, not committed to git

---

## Phase 10 — FastAPI Serving (Day 9, ~3 hours)

**Goal:** Production-grade REST API with input validation, batch prediction, and proper error handling.

### Steps

**10.1 — Pydantic schemas (`api/schemas.py`)**
```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class PredictionInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age")
    annual_premium: float = Field(..., gt=0, description="Annual premium in INR")
    vehicle_age: int = Field(..., ge=0, le=30, description="Vehicle age in years")
    # Add all your features here with validation
    
    @validator('annual_premium')
    def premium_must_be_reasonable(cls, v):
        if v > 10_000_000:
            raise ValueError('Annual premium seems unreasonably high')
        return v

class PredictionOutput(BaseModel):
    claim_probability: float
    will_claim: bool
    risk_tier: str          # 'LOW', 'MEDIUM', 'HIGH'
    top_risk_factors: List[str]
    model_version: str

class BatchPredictionInput(BaseModel):
    records: List[PredictionInput]
    
class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total_records: int
    high_risk_count: int
```

**10.2 — Model loader (`api/model_loader.py`)**
```python
import joblib
import yaml
from pathlib import Path

class ModelLoader:
    _instance = None
    
    def __init__(self):
        self.model = joblib.load('artifacts/calibrated_pipeline.pkl')
        with open('configs/params.yaml') as f:
            config = yaml.safe_load(f)
        self.threshold = config['model']['optimal_threshold']
        self.feature_names = config['model']['feature_names']
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

**10.3 — API endpoints (`api/main.py`)**
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shap, numpy as np, time

app = FastAPI(
    title="Vehicle Insurance Claim Prediction API",
    description="Predicts claim likelihood with calibrated probabilities and SHAP explanations",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], 
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "healthy", "model_version": "1.0.0"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    loader = ModelLoader.get_instance()
    try:
        df = pd.DataFrame([input_data.dict()])
        proba = loader.model.predict_proba(df)[0][1]
        will_claim = proba >= loader.threshold
        
        # Risk tier
        if proba < 0.3:
            risk_tier = "LOW"
        elif proba < 0.6:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "HIGH"
        
        # SHAP local explanation (top 3 risk factors)
        # ... (extract top SHAP features for this record)
        
        return PredictionOutput(
            claim_probability=round(float(proba), 4),
            will_claim=bool(will_claim),
            risk_tier=risk_tier,
            top_risk_factors=top_factors,
            model_version="1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchPredictionInput):
    # Same logic, vectorized over all records
    ...

@app.get("/model/info")
def model_info():
    # Returns threshold, calibration method, top features, metrics
    ...
```

**10.4 — Test your API**
```python
# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_valid_input():
    payload = {"age": 35, "annual_premium": 25000, "vehicle_age": 3, ...}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0 <= response.json()["claim_probability"] <= 1
    assert response.json()["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]

def test_predict_invalid_age():
    payload = {"age": -5, ...}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error
```

Run: `pytest tests/ -v`

### Checkpoint
- [ ] `/health`, `/predict`, `/predict/batch`, `/model/info` all working
- [ ] Pydantic validation rejects bad inputs with 422
- [ ] Tests pass
- [ ] API docs auto-generated at `http://localhost:8000/docs`

---

## Phase 11 — Drift Monitoring with Evidently AI (Day 10, ~2 hours)

**Goal:** Detect when incoming data differs from training data. This is what keeps models honest post-deployment.

### Steps

**11.1 — Generate reference dataset**

Your training data is the "reference" — what the model was trained on.  
Incoming production data is the "current" — what it sees after deployment.

```python
# Save reference dataset during training
reference_df = X_train.copy()
reference_df['target'] = y_train
reference_df.to_csv('monitoring/reference_data.csv', index=False)
```

**11.2 — Create monitoring reports (`src/monitoring/drift.py`)**
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import *
import pandas as pd

def generate_drift_report(reference_path: str, current_path: str, output_path: str):
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    
    # Data drift report
    data_report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    data_report.run(reference_data=reference.drop('target', axis=1),
                    current_data=current.drop('target', axis=1))
    data_report.save_html(f"{output_path}/data_drift_report.html")
    
    # Model performance report (if labels available)
    perf_report = Report(metrics=[
        ClassificationPreset(),
    ])
    perf_report.run(reference_data=reference, current_data=current,
                    column_mapping=ColumnMapping(target='target', 
                                                  prediction='prediction'))
    perf_report.save_html(f"{output_path}/model_performance_report.html")
    
    print(f"Reports saved to {output_path}/")

if __name__ == "__main__":
    generate_drift_report(
        'monitoring/reference_data.csv',
        'monitoring/simulated_production_data.csv',
        'monitoring/reports'
    )
```

**11.3 — Simulate production data**

Create `scripts/simulate_production_data.py`:
```python
import pandas as pd
import numpy as np

def simulate_drift(reference_df, n_samples=500, drift_factor=0.3):
    """Simulates data that has drifted from training distribution."""
    current = reference_df.sample(n_samples, replace=True).copy()
    
    # Inject drift: shift annual_premium distribution upward (inflation effect)
    current['annual_premium'] = current['annual_premium'] * (1 + drift_factor)
    
    # Add noise to vehicle_age
    current['vehicle_age'] = current['vehicle_age'] + np.random.randint(0, 3, n_samples)
    
    return current
```

**11.4 — Commit the generated HTML reports to the repo**

```bash
git add monitoring/reports/data_drift_report.html
git add monitoring/reports/model_performance_report.html
git commit -m "feat: add Evidently drift monitoring reports"
```

These static HTML files render directly on GitHub — a recruiter or interviewer can click and see them.

**11.5 — Add drift check to CI/CD (optional but impressive)**
```yaml
# In .github/workflows/ci-cd.yml
- name: Generate drift report
  run: python src/monitoring/drift.py
- name: Upload drift report artifact
  uses: actions/upload-artifact@v3
  with:
    name: drift-report
    path: monitoring/reports/
```

### Checkpoint
- [ ] Reference data saved during training
- [ ] Drift report HTML generated and committed to repo
- [ ] Reports are viewable in browser without running any code

---

## Phase 12 — Streamlit Frontend (Day 11, ~3 hours)

**Goal:** A clean, usable frontend that makes the project tangible to non-technical interviewers.

### Steps

**12.1 — Build `frontend/app.py`**

Structure your Streamlit app into 3 tabs:

```python
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:8000"  # change to deployed URL after deployment

st.set_page_config(
    page_title="RiskLens: Insurance Claim Predictor",
    page_icon="🛡️",
    layout="wide"
)

tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📦 Batch Prediction", "📊 Model Dashboard"])

# ── Tab 1: Single Prediction ──────────────────────────────────────────────────
with tab1:
    st.header("Predict Claim Likelihood")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Customer Age", 18, 70, 35)
        annual_premium = st.number_input("Annual Premium (₹)", 
                                          min_value=1000, max_value=500000, value=25000)
        vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 3)
    with col2:
        # Add remaining features as inputs
        ...
    
    if st.button("Predict", type="primary"):
        with st.spinner("Analyzing..."):
            response = requests.post(f"{API_URL}/predict", json={
                "age": age, "annual_premium": annual_premium, 
                "vehicle_age": vehicle_age, ...
            })
        
        if response.status_code == 200:
            result = response.json()
            
            # Display result with color coding
            risk_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
            st.markdown(f"""
            ### Prediction Result
            **Claim Probability:** `{result['claim_probability']:.1%}`  
            **Risk Tier:** :{risk_color[result['risk_tier']]}[**{result['risk_tier']}**]  
            **Decision:** {'⚠️ Likely to Claim' if result['will_claim'] else '✅ Unlikely to Claim'}
            """)
            
            # Gauge chart for probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['claim_probability'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "red"}
                    ],
                    'threshold': {'value': 50}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Top risk factors
            st.subheader("Top Risk Factors")
            for factor in result['top_risk_factors']:
                st.write(f"• {factor}")

# ── Tab 2: Batch Prediction ──────────────────────────────────────────────────
with tab2:
    st.header("Batch Prediction")
    st.write("Upload a CSV file with multiple customer records.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        if st.button("Run Batch Prediction"):
            records = df.to_dict(orient='records')
            response = requests.post(f"{API_URL}/predict/batch", 
                                     json={"records": records})
            results_df = pd.DataFrame(response.json()['predictions'])
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", len(results_df))
            col2.metric("High Risk", (results_df['risk_tier'] == 'HIGH').sum())
            col3.metric("Avg Claim Prob", f"{results_df['claim_probability'].mean():.1%}")
            
            # Distribution chart
            fig = px.histogram(results_df, x='claim_probability', 
                               color='risk_tier', nbins=20,
                               title="Claim Probability Distribution")
            st.plotly_chart(fig)
            
            # Download results
            st.download_button(
                "Download Predictions CSV",
                results_df.to_csv(index=False),
                "predictions.csv"
            )

# ── Tab 3: Model Dashboard ────────────────────────────────────────────────────
with tab3:
    st.header("Model Performance Dashboard")
    
    model_info = requests.get(f"{API_URL}/model/info").json()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", model_info['auc_roc'])
    col2.metric("PR-AUC", model_info['pr_auc'])
    col3.metric("F1 Score", model_info['f1'])
    col4.metric("Optimal Threshold", model_info['threshold'])
    
    # Show SHAP summary plot
    st.image("monitoring/reports/shap_summary.png", 
             caption="Feature Importance (SHAP)")
    
    # Show calibration plot
    col1, col2 = st.columns(2)
    with col1:
        st.image("monitoring/reports/calibration_before.png", 
                 caption="Calibration: Before")
    with col2:
        st.image("monitoring/reports/calibration_after.png", 
                 caption="Calibration: After")
    
    st.image("monitoring/reports/threshold_optimization.png",
             caption="Business Cost vs Threshold")
```

### Checkpoint
- [ ] All 3 tabs working against local API
- [ ] File upload and batch prediction tested with a sample CSV
- [ ] Gauge chart renders for single prediction
- [ ] Dashboard shows all key plots

---

## Phase 13 — Dockerization (Day 12, ~2 hours)

**Goal:** Both the API and frontend run together reproducibly via Docker Compose.

### Steps

**13.1 — API Dockerfile**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY artifacts/ ./artifacts/
COPY configs/ ./configs/
COPY monitoring/ ./monitoring/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**13.2 — Frontend Dockerfile**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements-frontend.txt .
RUN pip install --no-cache-dir -r requirements-frontend.txt

COPY frontend/ ./frontend/
COPY monitoring/reports/ ./monitoring/reports/

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**13.3 — `docker-compose.yml`**
```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DAGSHUB_TOKEN=${DAGSHUB_TOKEN}
    volumes:
      - ./artifacts:/app/artifacts
      - ./monitoring:/app/monitoring
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      api:
        condition: service_healthy
```

**13.4 — Test Docker Compose locally**
```bash
docker-compose up --build
# API: http://localhost:8000/docs
# Frontend: http://localhost:8501
```

### Checkpoint
- [ ] `docker-compose up --build` runs without errors
- [ ] API accessible at localhost:8000/docs
- [ ] Frontend accessible at localhost:8501 and calls API successfully
- [ ] `.dockerignore` excludes `venv/`, `*.pyc`, `data/raw/`, `__pycache__/`

---

## Phase 14 — CI/CD and Deployment (Day 13, ~3 hours)

**Goal:** Automated build + push on every merge to main. Deployed and accessible on the public internet.

### Steps

**14.1 — GitHub Actions workflow (`.github/workflows/ci-cd.yml`)**
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run tests
        run: pytest tests/ -v --tb=short
      
      - name: Lint
        run: |
          pip install flake8
          flake8 src/ api/ --max-line-length 100

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-south-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v1
      
      - name: Build and push API image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -f Dockerfile.api -t $ECR_REGISTRY/risklens-api:$IMAGE_TAG .
          docker build -f Dockerfile.api -t $ECR_REGISTRY/risklens-api:latest .
          docker push $ECR_REGISTRY/risklens-api:$IMAGE_TAG
          docker push $ECR_REGISTRY/risklens-api:latest
      
      - name: Deploy to EC2
        uses: appleboy/ssh-action@v0.1.10
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            docker pull ${{ steps.login-ecr.outputs.registry }}/risklens-api:latest
            docker stop risklens-api || true
            docker rm risklens-api || true
            docker run -d --name risklens-api -p 8000:8000 \
              ${{ steps.login-ecr.outputs.registry }}/risklens-api:latest
```

**14.2 — AWS setup (EC2 option — cost-effective and honest)**

Use EC2 t2.micro (free tier eligible) instead of EKS:
```bash
# On your EC2 instance (Ubuntu):
sudo apt update && sudo apt install -y docker.io awscli
sudo usermod -aG docker ubuntu
# Configure ECR credentials on instance
aws configure  # use IAM role or access keys
```

**14.3 — Deploy Streamlit frontend on Streamlit Cloud (free)**

1. Push `frontend/app.py` to GitHub
2. Go to share.streamlit.io
3. Connect repo, set main file as `frontend/app.py`
4. Add `API_URL` as a secret pointing to your EC2 public IP

**14.4 — Environment variables and secrets**

Never commit secrets. Use:
- GitHub Secrets for CI/CD (AWS keys, EC2 SSH key)
- `.env` file locally (add to `.gitignore`)
- EC2 instance profile for AWS service access

### Checkpoint
- [ ] GitHub Actions CI passes (tests + lint)
- [ ] Docker image pushed to ECR on merge to main
- [ ] API live on EC2: `http://YOUR_EC2_IP:8000/docs`
- [ ] Streamlit frontend live on Streamlit Cloud
- [ ] Both URLs working end-to-end

---

## Phase 15 — Documentation (Day 14, ~2 hours)

**Goal:** Make the project immediately understandable to anyone who opens the GitHub repo.

### Steps

**15.1 — README.md structure**
```markdown
# 🛡️ RiskLens: Vehicle Insurance Claim Prediction

> End-to-end MLOps pipeline for predicting vehicle insurance claims
> with calibrated probabilities, business-driven threshold optimization,
> and real-time drift monitoring.

## 🔗 Live Links
- API: http://YOUR_EC2_IP:8000/docs
- Frontend: https://YOUR_APP.streamlit.app
- Experiment Tracking: https://dagshub.com/YOUR_USERNAME/...

## 🏗️ Architecture
[Paste the DVC DAG or draw a simple diagram]

## 📊 Model Performance
| Metric | Value |
|--------|-------|
| AUC-ROC | 0.91 |
| PR-AUC | 0.XX |
| F1 (optimal threshold) | 0.86 |
| ECE (after calibration) | 0.03 |
| Optimal Threshold | 0.35 |

## 💼 Business Impact
Using cost-sensitive threshold optimization (FN cost: ₹50K, FP cost: ₹3K),
the optimal threshold of 0.35 reduces expected operational cost by XX%
compared to the default 0.5 threshold.

## 🗂️ Project Structure
[folder tree]

## 🚀 Quick Start
[docker-compose instructions]

## 📈 Key Technical Decisions
1. **Imbalance:** SMOTE inside imblearn Pipeline (prevents data leakage)
2. **Calibration:** Isotonic regression (ECE: 0.08 → 0.03)
3. **Threshold:** Business cost optimization, not F1 maximization
4. **Monitoring:** Evidently AI drift reports ([view here](monitoring/reports/))
```

**15.2 — MODEL_CARD.md**
```markdown
# Model Card: RiskLens Insurance Claim Predictor v1.0

## Intended Use
Binary classification for vehicle insurance claim likelihood.
Outputs calibrated probabilities suitable for premium pricing.

## Training Data
- Dataset: [name and source]
- Size: X records, Y features
- Class distribution: 88% no-claim, 12% claim

## Performance
[Table with AUC-ROC, PR-AUC, F1, ECE]

## Performance by Subgroup
[If you have demographic features — check performance across age groups, vehicle types]

## Limitations
- Model trained on data from [time period]; may degrade with market changes
- Calibration optimized for training distribution; monitor drift over time
- Does not account for fraud signals (out of scope)

## Monitoring
Evidently AI drift reports committed to monitoring/reports/.
Retrain recommended when data drift score exceeds 0.3.
```

### Checkpoint
- [ ] README has live links, architecture, metrics table, and quick start
- [ ] MODEL_CARD.md complete
- [ ] All notebooks have descriptive markdown cells explaining decisions

---

## Phase 16 — Final Audit Before Interviews (Day 15)

Run through this checklist. Every ✅ is something you can talk about confidently.

### Technical Depth Checklist
- [ ] Can explain your imbalance ratio and why SMOTE was/wasn't the right choice
- [ ] Can draw the calibration curve from memory and explain what ECE means
- [ ] Can explain cost matrix assumptions and defend the numbers
- [ ] Can explain the DVC DAG stage by stage
- [ ] Can walk through one SHAP waterfall plot for a specific prediction
- [ ] Can explain what happens if drift is detected (what's the remediation plan)
- [ ] Can explain why you chose XGBoost/LightGBM over the other
- [ ] Can explain Optuna's TPE sampler vs random search vs grid search

### Infrastructure Checklist
- [ ] Can explain the CI/CD flow: PR → test → merge → build → push → deploy
- [ ] Can explain what ECR is and why images are stored there
- [ ] Can explain the difference between your EC2 deployment and what EKS would have been

### Code Quality Checklist
- [ ] No hardcoded values — all params in `configs/params.yaml`
- [ ] No notebook cells running in production — everything is `.py` scripts
- [ ] Tests exist and pass in CI
- [ ] Docker builds cleanly on a fresh machine
- [ ] `.env.example` committed (shows structure without exposing secrets)

### Portfolio Presentation Checklist
- [ ] GitHub repo has a clear description and topics/tags
- [ ] DagsHub experiment page is public and shows multiple runs
- [ ] README opens well on mobile
- [ ] Live demo link works when you click it

---

## Resume Bullets — Final Version

Replace your current bullets with these after completing the project:

```
• Built end-to-end binary classification pipeline with SMOTE-based imbalance handling 
  (12% positive class) inside imblearn Pipeline to prevent leakage; XGBoost and 
  LightGBM tuned with Optuna (50 trials, TPE sampler), achieving AUC-ROC 0.91, 
  PR-AUC 0.XX, F1 0.86

• Applied isotonic regression probability calibration (ECE: 0.08 → 0.03); implemented
  cost-sensitive threshold optimization using asymmetric cost matrix (FN: ₹50K, FP: ₹3K),
  reducing expected operational cost by XX% over default 0.5 threshold

• Implemented champion/challenger model promotion — new models auto-promote in MLflow 
  only when AUC improvement exceeds defined gate; all experiments on DagsHub with 
  data versioning via DVC

• Integrated SHAP explainability (global summary + local waterfall) logged as MLflow 
  artifacts; added Evidently AI data drift monitoring with HTML reports committed 
  to repo for stakeholder review

• Structured as modular src-layout DVC pipeline (ingestion → validation → feature 
  engineering → training → evaluation) with Docker-containerized FastAPI inference 
  (single + batch endpoints) and Streamlit frontend; CI/CD via GitHub Actions 
  builds/pushes to AWS ECR and deploys to EC2
```

---

## Quick Reference — Day-by-Day Schedule

| Day | Phase | Output |
|-----|-------|--------|
| 1 | 0, 1 | Repo setup + EDA complete |
| 2 | 1, 2 | EDA done + Feature engineering |
| 3 | 3 | Baselines logged to MLflow |
| 4 | 4 | Imbalance experiments complete |
| 5 | 5 | Optuna tuning for both models |
| 6 | 6, 7 | Calibration + threshold optimization |
| 7 | 8 | SHAP artifacts |
| 8 | 9 | DVC pipeline end-to-end |
| 9 | 10 | FastAPI + tests passing |
| 10 | 11 | Evidently drift reports |
| 11 | 12 | Streamlit frontend |
| 12 | 13 | Docker Compose working locally |
| 13 | 14 | Deployed to EC2 + Streamlit Cloud |
| 14 | 15 | README + MODEL_CARD |
| 15 | 16 | Final audit + resume update |

---

*Built with the goal of making every decision defensible in an interview.*  
*Commit often. Each commit is evidence of your process.*
