# RiskLens - Vehicle Insurance Claim Prediction MLOps

**AI-powered insurance risk assessment with production-ready ML infrastructure**

🚀 **Status:** 11/16 Phases Complete (69%) | All Core Systems Deployed

---

## 📚 Documentation

**Start here:** → [**Full Project Guide**](docs/README.md)

### Key Documents
- [**Project Roadmap**](docs/ROADMAP.md) - Complete phase breakdown and status
- [**Deployment Guide**](docs/DEPLOYMENT.md) - Docker, Kubernetes, production setup
- [**DVC Pipeline**](docs/PHASE10_DVC_PIPELINE.md) - Reproducible ML workflow
- [**DagsHub Setup**](docs/DAGSHUB_SETUP.md) - MLflow remote tracking
- [**Phase 8 Summary**](docs/PHASE8_SUMMARY.md) - Model serving and deployment

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.12 | uv package manager | Docker
```

### Local Setup
```bash
# 1. Clone and setup environment
git clone https://github.com/Pandharimaske/RiskLens.git
cd RiskLens
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Configure DagsHub (optional, for remote experiment tracking)
python src/scripts/setup_dagshub.py
source .env.dagshub

# 3. Run API server
python -m uvicorn src.app:app --reload

# 4. Open Streamlit dashboard (in another terminal)
streamlit run src/dashboard.py
```

### Docker Deployment
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f api

# Access services
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- Grafana: http://localhost:3000
```

---

## 🏗️ Project Structure

```
RiskLens/
├── src/                          # Application code
│   ├── app.py                    # FastAPI server
│   ├── dashboard.py              # Streamlit UI
│   ├── config.py                 # Configuration
│   ├── mlflow_config.py          # MLflow setup
│   └── scripts/                  # Utilities
│       ├── setup_dagshub.py
│       ├── prepare_data.py
│       └── save_model_for_deployment.py
│
├── phases/                       # ML pipeline phases
│   ├── phase_04_baseline_modeling.py
│   ├── phase_05_hyperparameter_tuning.py
│   ├── phase_06_calibration.py
│   ├── phase_07_threshold_optimization.py
│   ├── phase_09_shap_explainability.py
│   ├── phase_10_dvc_pipeline.py
│   └── phase_11_drift_monitoring.py
│
├── docs/                         # Documentation
│   ├── README.md
│   ├── ROADMAP.md
│   ├── DEPLOYMENT.md
│   ├── DAGSHUB_SETUP.md
│   ├── PHASE8_SUMMARY.md
│   └── PHASE10_DVC_PIPELINE.md
│
├── src/features/                 # Feature engineering
├── data/                         # Data storage
├── artifacts/                    # Model artifacts
├── monitoring/                   # Drift dashboards
├── dvc.yaml                      # DVC pipeline config
├── params.yaml                   # Pipeline parameters
├── Dockerfile                    # Container image
├── docker-compose.yml            # Orchestration
├── requirements.txt              # Dependencies
└── pyproject.toml                # Project metadata
```

---

## 📊 API Endpoints

All endpoints documented in [Full API Reference](docs/README.md#api-documentation)

### Core Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | API health and status |
| `/model-info` | GET | Model metadata |
| `/predict` | POST | Single prediction |
| `/predict-batch` | POST | Batch predictions (CSV) |

**Example:** Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "id": "CUST_001",
    "gender": "Male",
    "age": 35,
    "driving_license": 1,
    "region_code": 28,
    "previously_insured": 0,
    "vehicle_age": "1-2 Year",
    "vehicle_damage": "No",
    "annual_premium": 40000,
    "policy_sales_channel": 26,
    "vintage": 200
  }'
```

---

## 📈 Model Performance

**LightGBM + Calibration + Threshold Optimization**

| Metric | Value | Benefit |
|--------|-------|---------|
| AUC-ROC | 0.8560 | 85.6% discrimination |
| PR-AUC | 0.3651 | Better minority class focus |
| F1-Score | 0.4092 | Balanced precision/recall |
| Recall | 0.9775 | 97.75% claim detection |
| **Threshold** | 0.0400 | ₹283.6M business value |

---

## ✅ Completed Phases

| Phase | Status | Description |
|-------|--------|-------------|
| **0** | ✅ Complete | Project setup & environment |
| **1** | ✅ Complete | Data exploration & analysis |
| **2** | ✅ Complete | Feature engineering (17 features) |
| **3** | ✅ Complete | Baseline modeling |
| **4** | ✅ Complete | Model comparison & selection |
| **5** | ✅ Complete | Hyperparameter tuning (Optuna 50 trials) |
| **6** | ✅ Complete | Model calibration (Sigmoid) |
| **7** | ✅ Complete | Threshold optimization (₹283.6M savings) |
| **8** | ✅ Complete | Model serving & deployment (FastAPI) |
| **9** | ✅ Complete | SHAP explainability |
| **10** | ✅ Complete | DVC pipeline setup |
| **11** | ✅ Complete | Drift monitoring |
| **Medium** | ✅ Complete | Streamlit dashboard, README, CI/CD |
| **DagsHub** | ✅ Complete | MLflow remote tracking |

---

## ⏳ Remaining Phases

- **Phase 12:** Grafana production dashboard
- **Phase 13:** Advanced Evidently reports
- **Phase 14:** A/B testing framework
- **Phase 15:** Model retraining pipeline
- **Phase 16:** Governance & audit logging

---

## 🔧 Development

### Running Tests
```bash
# API tests
python -m pytest tests/test_api.py -v

# All tests
python -m pytest -v
```

### Code Quality
```bash
# Format code
black src/ phases/ tests/

# Lint code
flake8 src/ phases/ tests/ --max-line-length=120

# Type checking
mypy src/ --ignore-missing-imports
```

### Running Phases
```bash
# Run any phase (from project root)
python -m phases.phase_09_shap_explainability
python -m phases.phase_10_dvc_pipeline
python -m phases.phase_11_drift_monitoring
```

---

## 📦 Dependencies

Key packages (see `requirements.txt` for complete list):
- **ML:** lightgbm, scikit-learn, optuna, shap
- **API:** fastapi, uvicorn, pydantic
- **UI:** streamlit, plotly
- **MLOps:** mlflow, dvc, dvc-s3, evidently
- **Monitoring:** prometheus-client

---

## 🔐 Security

- Non-root Docker user (appuser, UID 1000)
- Secrets in `.env` (git-ignored)
- API validation & error handling
- Model artifact integrity checks
- Health checks for all services

---

## 📞 Support

**Issues?** Check [Troubleshooting](docs/DEPLOYMENT.md#troubleshooting)

**Questions?** See [FAQ](docs/README.md#faq)

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

**Last Updated:** May 28, 2026 | **Commit:** c6c8b30
