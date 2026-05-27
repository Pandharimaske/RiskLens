"""
DagsHub MLflow Integration Setup
Purpose: Configure MLflow to log experiments to DagsHub remote server
Enables collaborative experiment tracking and model registry

Author: RiskLens MLOps
Date: 2026-05-27
"""

import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_dagshub_mlflow():
    """Configure MLflow to use DagsHub remote server."""
    
    logger.info("=" * 80)
    logger.info("DAGSHUB MLFLOW INTEGRATION SETUP")
    logger.info("=" * 80)
    
    # DagsHub credentials
    DAGSHUB_REPO = "https://dagshub.com/Pandharimaske/RiskLens"
    MLFLOW_REMOTE_URL = "https://dagshub.com/Pandharimaske/RiskLens.mlflow"
    
    # Get credentials from environment or prompt user
    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")
    
    if not username:
        logger.warning("DAGSHUB_USERNAME environment variable not set")
        username = input("Enter your DagsHub username: ")
    
    if not token:
        logger.warning("DAGSHUB_TOKEN environment variable not set")
        token = input("Enter your DagsHub token: ")
    
    # Set MLflow environment variables
    os.environ["MLFLOW_TRACKING_URI"] = f"https://{username}:{token}@dagshub.com/Pandharimaske/RiskLens.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    
    # Create .env file for easy setup
    env_file = Path(".env.dagshub")
    env_content = f"""# DagsHub MLflow Configuration
export DAGSHUB_USERNAME={username}
export DAGSHUB_TOKEN={token}
export MLFLOW_TRACKING_URI=https://{username}:{token}@dagshub.com/Pandharimaske/RiskLens.mlflow
export MLFLOW_TRACKING_USERNAME={username}
export MLFLOW_TRACKING_PASSWORD={token}
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    logger.info(f"✓ Created: {env_file}")
    logger.info("  To use in future sessions, run: source .env.dagshub")
    
    # Test connection
    logger.info("\nTesting DagsHub connection...")
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        
        # Try to get experiments
        client = mlflow.MlflowClient()
        experiments = client.search_experiments()
        
        logger.info(f"✓ Successfully connected to DagsHub")
        logger.info(f"✓ Found {len(experiments)} existing experiments")
        logger.info(f"✓ Tracking URI: {os.environ['MLFLOW_TRACKING_URI'][:50]}...")
        
    except Exception as e:
        logger.error(f"✗ Connection failed: {e}")
        logger.info("  Troubleshooting:")
        logger.info("  1. Verify username and token are correct")
        logger.info("  2. Check internet connection")
        logger.info("  3. Visit https://dagshub.com/settings/tokens to generate new token")
        return False
    
    # Create setup instructions file
    setup_file = Path("DAGSHUB_SETUP.md")
    setup_content = """# DagsHub Integration Setup

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
"""
    
    with open(setup_file, 'w') as f:
        f.write(setup_content)
    
    logger.info(f"✓ Created: {setup_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ DAGSHUB INTEGRATION CONFIGURED")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Source environment: source .env.dagshub")
    logger.info("2. Re-run Phase scripts to log experiments to DagsHub")
    logger.info("3. View on DagsHub: https://dagshub.com/Pandharimaske/RiskLens")
    logger.info("\nImportant: Keep .env.dagshub file secure (contains credentials)")
    
    return True

if __name__ == "__main__":
    success = setup_dagshub_mlflow()
    exit(0 if success else 1)
