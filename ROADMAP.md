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

## Phase 0 — Project Setup (Day 1, ~2 hours)

**Goal:** Clean repo structure that looks professional from the first commit.

### Steps

**0.1 — Create the repo structure**
```
risklens/
├── data/
│   ├── raw/                  # original CSV, never touched after download
│   └── processed/            # DVC-tracked outputs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_modeling.ipynb
│   ├── 04_imbalance_handling.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   ├── 06_calibration.ipynb
│   ├── 07_threshold_optimization.ipynb
│   └── 08_evaluation_final.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── ingestion.py
│   │   └── validation.py
│   ├── features/
│   │   └── engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   ├── monitoring/
│   │   └── drift.py
│   └── utils/
│       └── logger.py
├── pipeline/
│   └── dvc.yaml
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── model_loader.py
├── frontend/
│   └── app.py                # Streamlit app
├── monitoring/
│   └── reports/              # Evidently HTML reports committed here
├── tests/
│   └── test_api.py
├── configs/
│   └── params.yaml           # all hyperparameters and thresholds live here
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── MODEL_CARD.md
├── ROADMAP.md
└── README.md
```

**0.2 — Initialize Git, DVC, and connect DagsHub**
```bash
git init
git remote add origin https://github.com/YOUR_USERNAME/risklens
dvc init
# After creating DagsHub repo:
dvc remote add origin https://dagshub.com/YOUR_USERNAME/risklens.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_DAGSHUB_USERNAME
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN
```

**0.3 — Create and activate virtual environment with uv**
```bash
uv venv                        # Creates .venv
source .venv/bin/activate      # Linux/Mac
uv pip install -r requirements.txt  # or use: uv sync
```

**0.4 — requirements.txt (start with these, add as you go)**
```
pandas numpy scikit-learn xgboost lightgbm
optuna mlflow dvc dagshub
imbalanced-learn
shap matplotlib seaborn
evidently
fastapi uvicorn pydantic
streamlit
great-expectations
python-dotenv
pytest httpx
```

**0.5 — Set up MLflow tracking**
```python
# In configs/params.yaml
mlflow:
  tracking_uri: "https://dagshub.com/YOUR_USERNAME/risklens.mlflow"
  experiment_name: "insurance-claim-prediction"
```

### Checkpoint
- [ ] Repo pushed to GitHub with the full folder structure
- [ ] DVC initialized and remote configured
- [ ] DagsHub project created and linked

---

## Phase 1 — Data & EDA (Day 1–2, ~4 hours)

**Goal:** Understand the data deeply enough to make informed modeling decisions.  
Use the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) or  
[Vehicle Insurance dataset on Kaggle](https://www.kaggle.com/datasets/ifteshanajnin/carinsuranceclaimprediction-classification) — pick one and stick with it.

### Steps

**1.1 — Data ingestion script (`src/data/ingestion.py`)**
```python
# Downloads or loads raw CSV
# Saves to data/raw/
# Tracks with DVC: dvc add data/raw/train.csv
```

After adding to DVC:
```bash
dvc add data/raw/train.csv
git add data/raw/train.csv.dvc .gitignore
git commit -m "feat: add raw data via DVC"
dvc push
```

**1.2 — EDA notebook (`notebooks/01_eda.ipynb`) — answer these questions:**

1. **Target distribution** — what is the positive class rate?  
   `df['target'].value_counts(normalize=True)`  
   → This number (e.g., 8%) becomes your imbalance justification in the interview.

2. **Missing values** — which features have nulls and how many?  
   Heatmap + percentage table.

3. **Feature distributions** — histograms for all numeric features.  
   Look for skew, outliers, bimodal distributions.

4. **Correlation analysis** — heatmap.  
   Note any features correlated >0.9 with each other (multicollinearity candidates).

5. **Target vs each feature** — boxplots for numeric, countplots for categorical.  
   → This is where you spot which features are actually predictive.

6. **Class imbalance visualization** — pie chart + annotated bar chart.  
   Save this as `monitoring/reports/class_distribution.png`.

**1.3 — Write down your EDA findings in a markdown cell at the top of the notebook**

Example:
```
Key Findings:
- Target imbalance: 88% no-claim, 12% claim (ratio ~7.3:1)
- 3 features have >20% missing values: [list them]
- vehicle_age and annual_premium are the strongest predictors visually
- policy_sales_channel is high-cardinality categorical (155 unique values)
```

These findings directly drive your Phase 2 decisions. This is what interviewers want to see — that your decisions were *data-driven*, not random.

### Checkpoint
- [ ] EDA notebook complete with all 6 analyses
- [ ] Class imbalance ratio documented
- [ ] Key findings written up in the notebook

**Interview talking point:**  
*"The dataset had an 88/12 split, so I first established that AUC-ROC alone would be misleading — I tracked PR-AUC and F1 at multiple thresholds throughout."*

---

## Phase 2 — Feature Engineering (Day 2–3, ~3 hours)

**Goal:** Create domain-relevant features that improve model performance and demonstrate insurance domain knowledge.

### Steps

**2.1 — Preprocessing pipeline (`src/features/engineering.py`)**

Build a scikit-learn Pipeline (not manual transformations — this matters for deployment):
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])
```

**Why Pipeline matters:** When you save and load your model, the preprocessor is bundled with it. Your FastAPI endpoint receives raw input and the pipeline handles transformation internally. No preprocessing leakage.

**2.2 — Domain-specific feature engineering (`notebooks/02_feature_engineering.ipynb`)**

Add these features *before* fitting the pipeline:
```python
# 1. Vehicle age risk bucket (non-linear relationship with claims)
df['vehicle_age_bucket'] = pd.cut(df['vehicle_age'], 
    bins=[0, 1, 3, 7, 15, 100], 
    labels=['new', 'recent', 'mid', 'old', 'very_old'])

# 2. Premium per year of vehicle age (exposure-adjusted premium)
df['premium_per_vehicle_year'] = df['annual_premium'] / (df['vehicle_age'] + 1)

# 3. High-value vehicle flag
df['high_value_flag'] = (df['annual_premium'] > df['annual_premium'].quantile(0.75)).astype(int)

# 4. Prior damage flag (if feature exists — strong signal)
# df['has_prior_damage'] = (df['vehicle_damage'] == 'Yes').astype(int)
```

Add/remove based on what's in your actual dataset. The principle is: think like an underwriter.

**2.3 — Validation script (`src/data/validation.py`)**

Use Great Expectations or simple Pydantic-based checks:
```python
def validate_dataframe(df: pd.DataFrame) -> dict:
    issues = []
    
    # Schema check
    required_cols = ['age', 'annual_premium', 'vehicle_age', ...]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    # Range checks
    if df['age'].lt(0).any() or df['age'].gt(120).any():
        issues.append("age contains out-of-range values")
    
    # Null rate check
    null_rates = df.isnull().mean()
    high_null = null_rates[null_rates > 0.3].index.tolist()
    if high_null:
        issues.append(f"High null rate features: {high_null}")
    
    return {"valid": len(issues) == 0, "issues": issues}
```

This runs before every training pipeline execution. Log the result to MLflow.

### Checkpoint
- [ ] `preprocessor` pipeline built and saved as `preprocessor.pkl`
- [ ] At least 2 domain-engineered features added
- [ ] Validation function runs without errors on your dataset

---

## Phase 3 — Baseline Modeling (Day 3, ~2 hours)

**Goal:** Establish honest baselines before optimizing anything.

### Steps

**3.1 — Always start with a DummyClassifier**
```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy_auc = roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1])
print(f"Dummy AUC-ROC: {dummy_auc:.4f}")  # Will be ~0.5
```

This is your floor. Log it to MLflow. It forces you to prove your model adds value.

**3.2 — Logistic Regression baseline**
```python
from sklearn.linear_model import LogisticRegression

lr = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
lr.fit(X_train, y_train)
```

**3.3 — Log everything to MLflow from the start**
```python
with mlflow.start_run(run_name="baseline_logistic_regression"):
    mlflow.log_params(lr.get_params())
    
    y_pred_proba = lr.predict_proba(X_test)[:, 1]
    y_pred = lr.predict(X_test)
    
    mlflow.log_metric("auc_roc", roc_auc_score(y_test, y_pred_proba))
    mlflow.log_metric("pr_auc", average_precision_score(y_test, y_pred_proba))
    mlflow.log_metric("f1", f1_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    
    mlflow.sklearn.log_model(lr, "model")
```

**Always log PR-AUC (`average_precision_score`) — not just AUC-ROC.**

**3.4 — Train/validation/test split — do it correctly**
```python
from sklearn.model_selection import train_test_split

# First split off test set (never touched during training)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
# Then split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval
)
```

`stratify=y` is non-negotiable for imbalanced datasets — it preserves the class ratio in each split.

### Checkpoint
- [ ] DummyClassifier baseline logged to MLflow
- [ ] Logistic Regression baseline logged
- [ ] Both AUC-ROC and PR-AUC tracked for every run from this point on
- [ ] Stratified splits confirmed

**Interview talking point:**  
*"I always start with a DummyClassifier to establish the floor. My final model needed to beat it meaningfully on PR-AUC, not just AUC-ROC — because with 88% majority class, even a random model can look decent on AUC-ROC."*

---

## Phase 4 — Imbalance Handling (Day 4, ~3 hours)

**Goal:** Properly handle class imbalance. This is the most common gap in student projects.

### Steps

**4.1 — Understand your options first**

| Strategy | What it does | When to use |
|---|---|---|
| `scale_pos_weight` (XGBoost) | Weights the minority class during loss | Always try first — zero cost |
| `class_weight='balanced'` (sklearn) | Same concept for sklearn models | Always try first |
| SMOTE | Synthetic oversampling of minority class | When `scale_pos_weight` isn't enough |
| ADASYN | Adaptive SMOTE — focuses on hard samples | When SMOTE isn't enough |
| Undersampling | Remove majority class samples | Use only if dataset is huge |

**4.2 — Critical rule: SMOTE only on training data, never on test**
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # NOT sklearn Pipeline

# WRONG — leaks synthetic samples into test evaluation
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
X_train, X_test, ... = train_test_split(X_resampled, y_resampled)

# CORRECT — SMOTE is part of the training pipeline
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, k_neighbors=5)),
    ('clf', XGBClassifier())
])
# SMOTE only sees training data when pipeline.fit(X_train, y_train) is called
```

**4.3 — Experiment systematically in `notebooks/04_imbalance_handling.ipynb`**

Run these 4 experiments and log each to MLflow:
1. XGBoost, no imbalance handling (your baseline)
2. XGBoost + `scale_pos_weight = negative_count / positive_count`
3. XGBoost + SMOTE (in pipeline)
4. XGBoost + ADASYN (in pipeline)

Compare on PR-AUC and F1. Document which wins and why in a markdown cell.

**4.4 — Log the class distribution to MLflow**
```python
pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
mlflow.log_metric("train_positive_rate", pos_count / len(y_train))
mlflow.log_metric("imbalance_ratio", neg_count / pos_count)
```

### Checkpoint
- [ ] All 4 imbalance experiments logged to MLflow
- [ ] Best strategy identified and documented
- [ ] SMOTE is inside the pipeline (not applied before splitting)
- [ ] Winning approach noted in notebook markdown

**Interview talking point:**  
*"I ran four experiments with different imbalance strategies. SMOTE inside an imblearn Pipeline gave the best PR-AUC improvement. Critically, I never applied SMOTE before splitting — that would leak synthetic samples into the test set and inflate metrics."*

---

## Phase 5 — Hyperparameter Tuning with Optuna (Day 5, ~3 hours)

**Goal:** Find the best model. Track everything. Use real search, not grid search.

### Steps

**5.1 — Define Optuna objective function (`src/models/train.py`)**
```python
import optuna
import mlflow

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': neg_count / pos_count,  # from Phase 4
    }
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        
        model = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('clf', XGBClassifier(**params, random_state=42, eval_metric='aucpr'))
        ])
        
        # Use cross-validation, not a single train/val split
        scores = cross_val_score(model, X_train, y_train, 
                                  cv=5, scoring='average_precision')
        pr_auc_mean = scores.mean()
        mlflow.log_metric("cv_pr_auc_mean", pr_auc_mean)
        mlflow.log_metric("cv_pr_auc_std", scores.std())
        
    return pr_auc_mean

with mlflow.start_run(run_name="xgb_optuna_study"):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_pr_auc", study.best_value)
```

**5.2 — Do the same for LightGBM**

Same structure, different param names. Key LightGBM params to tune:
`num_leaves`, `min_child_samples`, `learning_rate`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`

**5.3 — Compare XGBoost vs LightGBM final models**

After tuning both, train each on full `X_train` with best params.  
Compare on `X_test` (first time test set is touched):
- AUC-ROC, PR-AUC, F1, Precision, Recall
- Training time
- Inference latency (100 predictions, measure time)

Document the winner with reasoning.

**5.4 — Save the winning model properly**
```python
import joblib

best_pipeline.fit(X_train, y_train)  # retrain on full training data

joblib.dump(best_pipeline, 'artifacts/model_pipeline.pkl')
mlflow.sklearn.log_model(best_pipeline, "final_model",
                          registered_model_name="risklens-claim-predictor")
```

### Checkpoint
- [ ] 50-trial Optuna study for XGBoost logged to MLflow/DagsHub
- [ ] Same for LightGBM
- [ ] Winner retrained on full X_train and saved as `artifacts/model_pipeline.pkl`
- [ ] DagsHub experiment comparison screenshot taken for README

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
