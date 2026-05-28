# DagsHub Integration Setup

## Quick Start

### 1. Set Environment Variables
```bash
# Either manually
export DAGSHUB_USERNAME=your_username
export DAGSHUB_TOKEN=your_token

# OR source the .env file
source .env.dagshub
```

### 2. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 3. Run MLflow Experiments
All Phase scripts will automatically log to DagsHub:
```bash
python run_phase9.py   # SHAP explainability
python run_phase10.py  # DVC pipeline setup
python run_phase11.py  # Drift monitoring
```

### 4. View on DagsHub
Visit: https://dagshub.com/Pandharimaske/RiskLens

## Features Now Enabled

### MLflow Integration
- ✅ Experiment tracking on DagsHub
- ✅ Metrics and parameters logging
- ✅ Model artifacts storage
- ✅ Experiment comparison UI
- ✅ Run history and metadata

### DagsHub Features
- 👥 Team collaboration on experiments
- 📊 Visual experiment comparisons
- 🔗 Git integration with experiment commits
- 📝 Experiment notes and documentation
- 🎯 Versioning and reproducibility

## Troubleshooting

### Cannot connect to DagsHub
```bash
# Check credentials
echo $DAGSHUB_USERNAME
echo $DAGSHUB_TOKEN

# Test connection
python -c "
import mlflow
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
print('Connected!')
"
```

### Experiments not showing up
1. Check MLflow tracking URI is set correctly
2. Verify credentials are valid
3. Check internet connection
4. Try: `mlflow.end_run()` to ensure run is closed

### Need a new token?
Visit: https://dagshub.com/settings/tokens

## Next Steps

1. **Reconfigure existing experiments**: Run Phase scripts again to log fresh experiments
2. **Monitor on DagsHub**: Watch experiments in real-time on the DagsHub UI
3. **Team collaboration**: Invite team members to review experiments
4. **Model registry**: Use DagsHub Model Registry for version control

---

For more info: https://dagshub.com/docs/collaboration/mlflow/
