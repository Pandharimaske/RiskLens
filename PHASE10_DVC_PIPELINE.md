# DVC Pipeline Documentation

## Overview
This DVC pipeline orchestrates the complete RiskLens MLOps workflow from data loading to model evaluation.

## Pipeline Stages

### 1. Data Loading (`data_loading`)
- **Input**: `data/raw/vehicle_insurance_data.csv`
- **Output**: Train/val/test splits as numpy arrays
- **Parameters**: Random seed, split ratios, stratification
- **Purpose**: Load raw insurance data and create train/val/test splits

### 2. Feature Engineering (`feature_engineering`)
- **Input**: Raw train/val/test sets
- **Output**: `artifacts/preprocessor.pkl`, `artifacts/feature_names.pkl`
- **Parameters**: Scaling method, encoding strategy
- **Purpose**: Create feature transformer pipeline (StandardScaler + OrdinalEncoder)

### 3. Model Training (`model_training`)
- **Input**: Preprocessor, feature names, train/val data
- **Output**: Base model, calibrated model
- **Parameters**: Model type, hyperparameters, calibration method
- **Metrics**: AUC-ROC, PR-AUC, F1, Recall
- **Purpose**: Train LightGBM with sigmoid calibration

### 4. Model Evaluation (`model_evaluation`)
- **Input**: Trained model, test data
- **Output**: Test metrics, confusion matrix, ROC curve plots
- **Purpose**: Comprehensive evaluation on held-out test set

## Running the Pipeline

### Full reproducible run:
```bash
dvc repro
```

### Run specific stage:
```bash
dvc repro scripts/02_train_model.py
```

### View pipeline DAG:
```bash
dvc dag
```

### Track pipeline changes:
```bash
dvc plots diff
```

## Parameters Management

All model parameters are centralized in `params.yaml`. Update any parameter and re-run:

```bash
# Edit params.yaml
vim params.yaml

# Re-run pipeline (only affected stages)
dvc repro
```

## Metrics Tracking

Metrics are tracked in `metrics/` directory:
- `metrics/train_metrics.json` - Training performance
- `metrics/test_metrics.json` - Test performance

View metrics:
```bash
dvc metrics show
```

## Integration with Git

DVC automatically creates `.gitignore` entries for outputs. Track pipeline definition:

```bash
git add dvc.yaml dvc.lock params.yaml .gitignore
git commit -m "Add DVC pipeline configuration"
```

## Next Steps

1. **Drift Monitoring** - Detect data/performance drift over time
2. **Production Serving** - Integrate with FastAPI for inference
3. **CI/CD Integration** - Automate pipeline on Git pushes
