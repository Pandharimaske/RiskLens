# 🔍 RiskLens: AI-Powered Vehicle Insurance Risk Prediction MLOps Platform

## Overview

**RiskLens** is a production-ready machine learning operations (MLOps) platform for predicting vehicle insurance claim risk. It combines modern ML best practices with comprehensive monitoring, explainability, and deployment infrastructure.

### Key Features
- 🤖 **LightGBM Model** with 85.60% AUC-ROC and optimized decision threshold
- 📊 **SHAP Explainability** for model interpretability and feature importance analysis
- 🔄 **DVC Pipeline** for reproducible end-to-end workflows
- 📈 **Drift Monitoring** with real-time data and prediction drift detection
- 🚀 **FastAPI Service** with 7 production-grade endpoints
- 🐳 **Docker Containerization** for easy deployment
- 🎨 **Streamlit Dashboard** for interactive predictions and monitoring
- 🧪 **Comprehensive Testing** and CI/CD integration

---

## Quick Start

### Prerequisites
- **Python**: 3.12+
- **Package Manager**: `uv` (recommended) or `pip`
- **Docker**: Optional, for containerization
- **Git**: For version control

### Installation

```bash
# Clone the repository
git clone https://github.com/Pandharimaske/RiskLens.git
cd RiskLens

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies using uv (faster)
uv pip install -e .
# OR using pip
pip install -r requirements.txt

# Activate MLflow server (optional)
mlflow ui
```

### Running the Application

**Option 1: Interactive Streamlit Dashboard**
```bash
streamlit run streamlit_app.py
```

**Option 2: Production FastAPI Service**
```bash
python app.py
# OR with multiple workers
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

**Option 3: Docker Container**
```bash
docker build -t risklens:latest .
docker run -p 8000:8000 risklens:latest
# With docker-compose
docker-compose up -d
```

---

## Architecture & Technology Stack

### ML/Data Stack
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Model** | LightGBM | 4.1.1 | Gradient boosting classifier |
| **Calibration** | scikit-learn | 1.3.2 | Sigmoid probability calibration |
| **Tuning** | Optuna | 4.8.0 | Bayesian hyperparameter optimization |
| **Explainability** | SHAP | 0.51.0 | Feature importance & local explanations |
| **Experiment Tracking** | MLflow | 3.12.0 | Reproducible experiment management |
| **Data Versioning** | DVC | 3.26.0 | Version control for datasets |
| **Drift Detection** | Evidently AI | - | Statistical drift monitoring |

### Backend & Deployment
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI | REST endpoints with async support |
| **Web Server** | Uvicorn | ASGI application server |
| **Frontend** | Streamlit | Interactive dashboard |
| **Containerization** | Docker | Reproducible deployment |
| **Orchestration** | docker-compose | Local multi-container setup |

### Infrastructure
| Component | Configuration | Purpose |
|-----------|---|---------|
| **Base Image** | Python 3.12-slim | Minimal production image |
| **User** | Non-root | Security best practice |
| **Health Checks** | Built-in | Container orchestration support |
| **Volume Mounts** | Model artifacts | Persistent storage |

---

## Dataset & Features

### Dataset Characteristics
- **Records**: 381,109 vehicle insurance customers
- **Class Distribution**: 87.74% negative (no claim), 12.26% positive (claim)
- **Train/Val/Test Split**: 70%/12%/18% with stratification to preserve class balance

### Features (17 Total)
**Engineered Features:**
- Premium_Bucket: Premium category bucketing
- Customer_Tenure_Segment: Customer relationship duration segments
- Premium_per_Vehicle_Year: Normalized premium metric
- Age_Bracket: Age categorization
- Income_Level: Income segment classification
- Risk_Category: Risk-based segmentation
- Loyalty_Score: Customer loyalty metric

**Original Features:**
- Age, Driving_License, Region_Code, Previously_Insured, Vehicle_Age, Annual_Premium, Policy_Sales_Channel, Vintage

---

## Model Performance

### Phase 5: Hyperparameter Tuning
- **Best Trial**: Trial 13 (Optuna TPE sampler, 50 trials)
- **CV PR-AUC**: 0.3711
- **Hyperparameters**: num_leaves=31, learning_rate=0.05, n_estimators=100

### Phase 6: Calibration
- **Method**: Sigmoid (Platt scaling)
- **Brier Score Improvement**: -48.81%
- **Log Loss Improvement**: -41.35%

### Phase 7: Threshold Optimization
- **Optimal Threshold**: 0.0400 (vs default 0.5000)
- **Cost Matrix**: False Negative=₹50K, False Positive=₹3K
- **Cost Savings**: ₹283.6M (80.95% reduction)
- **Recall**: 97.75% (catches 98% of claims)

### Final Test Set Performance
| Metric | Value |
|--------|-------|
| AUC-ROC | 0.8560 |
| PR-AUC | 0.3651 |
| F1 Score | 0.4092 |
| Recall | 0.9775 |
| Accuracy | 0.9176 |
| Decision Threshold | 0.0400 |

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```bash
GET /health
```
Returns service status for orchestrators.

#### 2. Model Info
```bash
GET /model-info
```
Returns model metadata including version, calibration method, and threshold.

#### 3. Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "Age": 35,
  "Annual_Premium": 40000,
  "Vehicle_Age_Numeric": 5,
  "Previously_Insured": 1,
  "Driving_License": 1,
  "Region_Code": 5,
  "Policy_Sales_Channel": 1,
  "Vintage": 365,
  "Premium_Bucket": 0,
  "Customer_Tenure_Segment": 0,
  "Premium_per_Vehicle_Year": 0,
  "Age_Bracket": 0,
  "Income_Level": 0,
  "Risk_Category": 0,
  "Loyalty_Score": 0,
  "Fuel_Type": 0,
  "Vehicle_Damage": 0
}
```
Returns: `{"prediction": 0, "probability": 0.0234, "threshold": 0.04}`

#### 4. Batch Prediction
```bash
POST /predict-batch
Content-Type: application/json

{
  "records": [
    { /* record 1 */ },
    { /* record 2 */ }
  ]
}
```
Max batch size: 10,000 records

#### 5. Interactive Docs
```
GET /docs          # Swagger OpenAPI
GET /redoc         # ReDoc alternative
```

---

## DVC Pipeline Stages

### Reproducible Workflow
Execute the complete pipeline:
```bash
dvc repro
```

View pipeline DAG:
```bash
dvc dag
```

#### Stages
1. **data_loading**: Load and split raw data
2. **feature_engineering**: Create features and fit preprocessor
3. **model_training**: Train LightGBM with calibration
4. **model_evaluation**: Evaluate on test set

#### Parameters (`params.yaml`)
Centralized configuration for all stages. Update and re-run:
```bash
# Edit parameters
vim params.yaml

# Re-run only affected stages
dvc repro
```

---

## Monitoring & Drift Detection

### Phase 11: Drift Monitoring Dashboard
```bash
# Generate drift metrics
python run_phase11.py

# View interactive dashboard
open monitoring/drift_dashboard.html
```

### Monitored Metrics
- **Data Drift**: Statistical drift in input features
- **Prediction Drift**: Changes in model output distribution
- **Feature Drift Score**: Standardized mean difference per feature

Threshold: ±0.1 standard deviations

---

## Streamlit Dashboard Usage

### Three-Tab Interface

**Tab 1: Single Prediction**
- Interactive form for customer and vehicle details
- Real-time risk assessment with gauge chart
- Decision explanation and recommendation

**Tab 2: Batch Prediction**
- CSV file upload for bulk predictions
- Risk distribution visualization
- Download results as CSV

**Tab 3: Monitoring Dashboard**
- Real-time drift metrics and KPIs
- Feature-level drift analysis
- Prediction probability distribution

---

## Deployment Options

### Local Deployment
```bash
# Development server
streamlit run streamlit_app.py
# Production API
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```bash
# Build image
docker build -t risklens:latest .

# Run container
docker run -p 8000:8000 risklens:latest

# With docker-compose
docker-compose up -d
```

### Cloud Deployment
Refer to [DEPLOYMENT.md](DEPLOYMENT.md) for:
- **Kubernetes** manifests
- **AWS ECS** configuration
- **Azure Container Instances** setup
- Environment variables and security best practices

---

## Project Structure

```
RiskLens/
├── artifacts/               # Model, preprocessor, feature names
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Train/val/test splits (numpy)
├── monitoring/             # Drift dashboards and metrics
├── notebooks/              # SHAP visualizations
├── scripts/                # DVC pipeline stage scripts
├── app.py                  # FastAPI production service
├── streamlit_app.py        # Interactive dashboard
├── config.py               # Configuration management
├── Dockerfile              # Multi-stage container
├── docker-compose.yml      # Local orchestration
├── dvc.yaml                # Pipeline definition
├── params.yaml             # Configuration parameters
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Development & Testing

### Run Tests
```bash
# API tests
python test_api_local.py

# Run with pytest (if configured)
pytest tests/
```

### MLflow Experiment Tracking
```bash
# View experiments
mlflow ui

# Access at http://localhost:5000
```

### Code Quality
```bash
# Check for syntax errors
python -m py_compile *.py

# Linting (if configured)
pylint app.py
```

---

## MLOps Phases Completed

✅ **Phase 0**: Project Setup with Python 3.12 & uv  
✅ **Phase 1**: EDA on 381K insurance records  
✅ **Phase 2**: Feature Engineering (17 features)  
✅ **Phase 3**: Baseline Modeling (LogisticRegression)  
✅ **Phase 4**: Model Comparison (XGBoost, LightGBM, RF)  
✅ **Phase 5**: Hyperparameter Tuning (Optuna 50 trials)  
✅ **Phase 6**: Probability Calibration (Sigmoid method)  
✅ **Phase 7**: Threshold Optimization (cost matrix)  
✅ **Phase 8**: Production Deployment (FastAPI + Docker)  
✅ **Phase 9**: SHAP Explainability  
✅ **Phase 10**: DVC Pipeline  
✅ **Phase 11**: Drift Monitoring  

⏳ **Planned**:
- Phase 12: Production Grafana Dashboard
- Phase 13: Advanced Monitoring (Evidently reports)
- Phase 14: A/B Testing Framework
- Phase 15: Model Retraining Pipeline
- Phase 16: Governance & Audit Logging

---

## Key Metrics & Business Impact

| Metric | Value | Impact |
|--------|-------|--------|
| AUC-ROC | 0.8560 | Strong discriminative ability |
| Recall | 97.75% | Catches 98% of claims |
| Cost Savings | ₹283.6M | 80.95% reduction vs baseline |
| Decision Threshold | 0.0400 | Optimized for business cost matrix |
| Prediction Latency | <10ms | Production-ready response time |
| Model Size | 1.5 MB | Efficient deployment footprint |

---

## Troubleshooting

### Model Loading Issues
```
Error: "Cannot extract feature importances"
→ Model is wrapped in CalibratedClassifierCV
→ Access base model via .estimator_ attribute
```

### API Connection
```bash
# Check if service is running
curl http://localhost:8000/health

# View detailed logs
docker logs risklens
```

### DVC Pipeline Errors
```bash
# Recalculate dependencies
dvc dag --md

# Force re-run specific stage
dvc repro --force-downstream scripts/02_train_model.py
```

---

## Contributing

1. Create feature branch: `git checkout -b feature/xyz`
2. Implement changes and test thoroughly
3. Commit with clear messages: `git commit -m "Phase X: Description"`
4. Push and create pull request

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## Contact & Support

- **Repository**: [github.com/Pandharimaske/RiskLens](https://github.com/Pandharimaske/RiskLens)
- **Issues**: Use GitHub Issues for bug reports
- **Documentation**: See [PHASE*_SUMMARY.md](.) files for detailed phase documentation

---

## Acknowledgments

- **LightGBM** team for the efficient gradient boosting library
- **SHAP** project for model explainability tools
- **MLflow** community for experiment tracking
- **Streamlit** for rapid dashboard development
- **Evidently AI** for drift detection capabilities

---

**Last Updated**: May 27, 2026  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
