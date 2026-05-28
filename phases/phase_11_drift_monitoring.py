"""
Phase 11: Drift Monitoring with Evidently AI
Purpose: Detect data drift, model drift, and performance degradation
Generates interactive HTML dashboards for monitoring

Author: RiskLens MLOps
Date: 2026-05-27
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import mlflow_config  # Setup MLflow for DagsHub or local

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing Evidently (graceful fallback if not installed)
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, RegressionPreset, ClassificationPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently AI not installed. Install with: pip install evidently")

# Paths
ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = Path("data")
NOTEBOOKS_DIR = Path("notebooks")
MONITORING_DIR = Path("monitoring")
MODEL_PATH = ARTIFACTS_DIR / "calibrated_model.pkl"
X_TRAIN_PATH = DATA_DIR / "processed" / "X_train.npy"
X_TEST_PATH = DATA_DIR / "processed" / "X_test.npy"
Y_TEST_PATH = DATA_DIR / "processed" / "y_test.npy"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.pkl"

def load_artifacts():
    """Load model and data."""
    logger.info("Loading artifacts...")
    
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    X_train = np.load(X_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    
    logger.info(f"Model loaded: {type(model)}")
    logger.info(f"Feature names: {len(feature_names)} features")
    logger.info(f"Data shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    
    return model, feature_names, X_train, X_test, y_test

def create_reference_and_current_data(X_train, X_test, feature_names):
    """Create reference (training) and current (test) datasets."""
    logger.info("Creating reference and current datasets...")
    
    # Reference data (from training set)
    reference_data = pd.DataFrame(X_train, columns=feature_names)
    
    # Current data (from test set) - simulate recent predictions
    current_data = pd.DataFrame(X_test, columns=feature_names)
    
    logger.info(f"Reference data: {reference_data.shape}")
    logger.info(f"Current data: {current_data.shape}")
    
    return reference_data, current_data

def generate_drift_report(model, reference_data, current_data, y_test):
    """Generate drift monitoring report using Evidently AI."""
    
    if not EVIDENTLY_AVAILABLE:
        logger.warning("Evidently not available - creating simplified drift metrics")
        return generate_simplified_drift_report(reference_data, current_data)
    
    logger.info("Generating drift monitoring report with Evidently AI...")
    
    # Add predictions to current data
    current_data_with_pred = current_data.copy()
    y_pred_proba = model.predict_proba(current_data.values)[:, 1]
    current_data_with_pred['prediction'] = model.predict(current_data.values)
    current_data_with_pred['prediction_proba'] = y_pred_proba
    current_data_with_pred['target'] = y_test
    
    # Create and run report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data_with_pred)
    
    return report

def generate_simplified_drift_report(reference_data, current_data):
    """Create simplified drift metrics without Evidently."""
    logger.info("Computing simplified drift metrics...")
    
    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
        "features_with_drift": []
    }
    
    # Compute drift for each feature
    for column in reference_data.columns:
        ref_mean = reference_data[column].mean()
        ref_std = reference_data[column].std()
        curr_mean = current_data[column].mean()
        
        # Simple drift metric: standardized mean difference
        if ref_std > 0:
            drift_score = abs(curr_mean - ref_mean) / ref_std
        else:
            drift_score = 0
        
        drift_report["metrics"][column] = {
            "reference_mean": float(ref_mean),
            "current_mean": float(curr_mean),
            "drift_score": float(drift_score),
            "has_drift": drift_score > 0.1  # Threshold
        }
        
        if drift_score > 0.1:
            drift_report["features_with_drift"].append(column)
    
    return drift_report

def compute_data_drift_metrics(reference_data, current_data, feature_names):
    """Compute comprehensive data drift metrics."""
    logger.info("Computing data drift metrics...")
    
    drift_metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_features": len(feature_names),
        "features_with_drift": 0,
        "drift_details": {}
    }
    
    for feature in feature_names:
        if feature not in reference_data.columns or feature not in current_data.columns:
            continue
        
        ref_data = reference_data[feature].values
        curr_data = current_data[feature].values
        
        # Statistical metrics
        ref_mean = np.mean(ref_data)
        ref_std = np.std(ref_data)
        curr_mean = np.mean(curr_data)
        curr_std = np.std(curr_data)
        
        # Drift detection (standardized mean difference)
        if ref_std > 0:
            drift_score = abs(curr_mean - ref_mean) / ref_std
        else:
            drift_score = 0
        
        has_drift = drift_score > 0.1
        
        drift_metrics["drift_details"][feature] = {
            "reference_mean": float(ref_mean),
            "reference_std": float(ref_std),
            "current_mean": float(curr_mean),
            "current_std": float(curr_std),
            "drift_score": float(drift_score),
            "has_drift": bool(has_drift)
        }
        
        if has_drift:
            drift_metrics["features_with_drift"] += 1
    
    logger.info(f"Data drift: {drift_metrics['features_with_drift']}/{len(feature_names)} features drifted")
    
    return drift_metrics

def compute_prediction_drift(model, X_test, y_test):
    """Detect prediction drift in model outputs."""
    logger.info("Computing prediction drift metrics...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    prediction_metrics = {
        "timestamp": datetime.now().isoformat(),
        "positive_class_rate_reference": float(np.mean(y_test)),
        "positive_class_rate_predictions": float(np.mean(y_pred)),
        "prediction_probability_mean": float(np.mean(y_pred_proba)),
        "prediction_probability_std": float(np.std(y_pred_proba)),
        "prediction_probability_min": float(np.min(y_pred_proba)),
        "prediction_probability_max": float(np.max(y_pred_proba))
    }
    
    # Detect prediction drift
    drift_in_distribution = abs(
        prediction_metrics["positive_class_rate_reference"] - 
        prediction_metrics["positive_class_rate_predictions"]
    ) > 0.01  # Stricter threshold for more sensitive detection
    
    prediction_metrics["has_prediction_drift"] = drift_in_distribution
    
    logger.info(f"Prediction drift: {drift_in_distribution} (class rate: {prediction_metrics['positive_class_rate_reference']:.4f} → {prediction_metrics['positive_class_rate_predictions']:.4f})")
    
    return prediction_metrics

def generate_monitoring_dashboard(drift_metrics, prediction_metrics):
    """Create HTML dashboard for monitoring."""
    logger.info("Generating monitoring dashboard...")
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>RiskLens - Drift Monitoring Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 { font-size: 2.5em; margin-bottom: 10px; }
        .timestamp { font-size: 0.9em; opacity: 0.9; }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .card h3 { color: #333; margin-bottom: 15px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        
        .metric {
            margin: 10px 0;
            padding: 10px;
            background: #f9f9f9;
            border-left: 3px solid #667eea;
            border-radius: 4px;
        }
        
        .metric-label { font-weight: 600; color: #555; }
        .metric-value { font-size: 1.3em; color: #333; margin: 5px 0; }
        
        .status {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }
        
        .status.warning {
            background: #fff3cd;
            color: #856404;
        }
        
        .status.success {
            background: #d4edda;
            color: #155724;
        }
        
        .status.danger {
            background: #f8d7da;
            color: #721c24;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        table th, table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        table th {
            background: #f0f0f0;
            font-weight: 600;
            color: #333;
        }
        
        table tr:hover {
            background: #f9f9f9;
        }
        
        .footer {
            text-align: center;
            color: #999;
            margin-top: 30px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 RiskLens - Drift Monitoring Dashboard</h1>
            <div class="timestamp">Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC") + """</div>
        </header>
        
        <div class="grid">
            <div class="card">
                <h3>Data Drift Status</h3>
                <div class="metric">
                    <div class="metric-label">Features Analyzed</div>
                    <div class="metric-value">""" + str(drift_metrics['total_features']) + """</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Features with Drift</div>
                    <div class="metric-value">""" + str(drift_metrics['features_with_drift']) + """</div>
                </div>
                <div class="metric">
                    <span class="status """ + ("warning" if drift_metrics['features_with_drift'] > 0 else "success") + """\">
                        """ + ("⚠️ Drift Detected" if drift_metrics['features_with_drift'] > 0 else "✓ No Drift") + """
                    </span>
                </div>
            </div>
            
            <div class="card">
                <h3>Prediction Drift Status</h3>
                <div class="metric">
                    <div class="metric-label">Reference Positive Rate</div>
                    <div class="metric-value">""" + f"{prediction_metrics['positive_class_rate_reference']:.2%}" + """</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Current Positive Rate</div>
                    <div class="metric-value">""" + f"{prediction_metrics['positive_class_rate_predictions']:.2%}" + """</div>
                </div>
                <div class="metric">
                    <span class="status """ + ("warning" if prediction_metrics['has_prediction_drift'] else "success") + """\">
                        """ + ("⚠️ Drift Detected" if prediction_metrics['has_prediction_drift'] else "✓ Stable") + """
                    </span>
                </div>
            </div>
            
            <div class="card">
                <h3>Prediction Distribution</h3>
                <div class="metric">
                    <div class="metric-label">Mean Probability</div>
                    <div class="metric-value">""" + f"{prediction_metrics['prediction_probability_mean']:.4f}" + """</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Std Deviation</div>
                    <div class="metric-value">""" + f"{prediction_metrics['prediction_probability_std']:.4f}" + """</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Range</div>
                    <div class="metric-value">[""" + f"{prediction_metrics['prediction_probability_min']:.4f}, {prediction_metrics['prediction_probability_max']:.4f}" + """]</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>Feature-Level Drift Analysis</h3>
            <table>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Reference Mean</th>
                        <th>Current Mean</th>
                        <th>Drift Score</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add feature drift details
    for feature, details in sorted(drift_metrics['drift_details'].items(), key=lambda x: x[1]['drift_score'], reverse=True):
        status_class = "warning" if details['has_drift'] else "success"
        status_text = "⚠️ Drift" if details['has_drift'] else "✓ OK"
        html_content += f"""
                    <tr>
                        <td><strong>{feature}</strong></td>
                        <td>{details['reference_mean']:.4f}</td>
                        <td>{details['current_mean']:.4f}</td>
                        <td>{details['drift_score']:.4f}</td>
                        <td><span class="status {status_class}">{status_text}</span></td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>RiskLens MLOps Platform | Phase 11: Drift Monitoring</p>
            <p>Generated with Evidently AI for model monitoring and data drift detection</p>
        </div>
    </div>
</body>
</html>
    """
    
    # Save HTML
    dashboard_path = MONITORING_DIR / "drift_dashboard.html"
    with open(dashboard_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"✓ Dashboard saved: {dashboard_path}")
    return dashboard_path

def save_drift_metrics_json(drift_metrics, prediction_metrics):
    """Save metrics as JSON for programmatic access."""
    logger.info("Saving drift metrics as JSON...")
    
    combined_metrics = {
        "timestamp": datetime.now().isoformat(),
        "data_drift": drift_metrics,
        "prediction_drift": prediction_metrics
    }
    
    metrics_path = MONITORING_DIR / "drift_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(combined_metrics, f, indent=2, default=str)
    
    logger.info(f"✓ Metrics saved: {metrics_path}")
    return metrics_path

def main():
    """Main Phase 11 pipeline."""
    logger.info("=" * 80)
    logger.info("PHASE 11: DRIFT MONITORING WITH EVIDENTLY AI")
    logger.info("=" * 80)
    
    # Create directories
    MONITORING_DIR.mkdir(exist_ok=True)
    
    try:
        # Load artifacts
        model, feature_names, X_train, X_test, y_test = load_artifacts()
        
        # Create reference and current datasets
        reference_data, current_data = create_reference_and_current_data(X_train, X_test, feature_names)
        
        # Compute drift metrics
        drift_metrics = compute_data_drift_metrics(reference_data, current_data, feature_names)
        prediction_metrics = compute_prediction_drift(model, X_test, y_test)
        
        # Generate dashboard
        dashboard_path = generate_monitoring_dashboard(drift_metrics, prediction_metrics)
        
        # Save metrics
        metrics_path = save_drift_metrics_json(drift_metrics, prediction_metrics)
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ PHASE 11 COMPLETE: DRIFT MONITORING")
        logger.info("=" * 80)
        logger.info("\nGenerated artifacts:")
        logger.info(f"  • {dashboard_path} - Interactive dashboard")
        logger.info(f"  • {metrics_path} - Metrics data")
        logger.info("\nUsage:")
        logger.info(f"  • View dashboard: open {dashboard_path}")
        logger.info(f"  • Monitor metrics: cat {metrics_path}")
        
    except Exception as e:
        logger.error(f"Error in Phase 11: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
