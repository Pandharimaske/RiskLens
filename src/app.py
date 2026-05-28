"""
FastAPI Application for RiskLens Model Serving

Provides REST API endpoints for:
- Single predictions
- Batch predictions
- Model health checks
- Model metadata
"""

import logging
import pickle
import io
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, File, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, ConfigDict
import uvicorn

from src.config import (
    HOST, PORT, WORKERS, DEBUG, MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH,
    DECISION_THRESHOLD, MAX_BATCH_SIZE, REQUEST_TIMEOUT, LOG_LEVEL,
    MODEL_VERSION, MODEL_NAME, CREATED_DATE, TRAINED_ON, API_VERSION
)

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL MODEL STATE
# ============================================================================
model = None
preprocessor = None
feature_names = None

def load_model():
    """Load calibrated model, preprocessor, and feature names using joblib."""
    global model, preprocessor, feature_names
    
    try:
        # Try joblib first (more robust), fallback to pickle
        try:
            model = joblib.load(MODEL_PATH)
            logger.info(f"✓ Model loaded (joblib): {MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Joblib load failed, trying pickle: {e}")
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✓ Model loaded (pickle): {MODEL_PATH}")
        
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logger.info(f"✓ Preprocessor loaded (joblib): {PREPROCESSOR_PATH}")
        except Exception as e:
            logger.warning(f"Joblib load failed, trying pickle: {e}")
            with open(PREPROCESSOR_PATH, 'rb') as f:
                preprocessor = pickle.load(f)
            logger.info(f"✓ Preprocessor loaded (pickle): {PREPROCESSOR_PATH}")
        
        try:
            feature_names = joblib.load(FEATURE_NAMES_PATH)
            logger.info(f"✓ Feature names loaded (joblib): {FEATURE_NAMES_PATH}")
        except Exception as e:
            logger.warning(f"Joblib load failed, trying pickle: {e}")
            with open(FEATURE_NAMES_PATH, 'rb') as f:
                feature_names = pickle.load(f)
            logger.info(f"✓ Feature names loaded (pickle): {FEATURE_NAMES_PATH}")
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Unexpected error loading model: {e}")

# ============================================================================
# MODEL LOADING LIFESPAN
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model on startup, cleanup on shutdown."""
    # Startup
    logger.info("Starting RiskLens API...")
    load_model()
    logger.info("✓ API startup complete")
    yield
    # Shutdown
    logger.info("Shutting down RiskLens API...")

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================
app = FastAPI(
    title=MODEL_NAME,
    description="REST API for vehicle insurance claim prediction",
    version=API_VERSION,
    lifespan=lifespan
)

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================
class PredictionRequest(BaseModel):
    """Single prediction request schema."""
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
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
            }
        }
    )
    
    id: str = Field(..., description="Customer ID")
    gender: str = Field(..., description="Male/Female")
    age: int = Field(..., ge=18, le=150, description="Customer age")
    driving_license: int = Field(..., ge=0, le=1, description="Has driving license")
    region_code: int = Field(..., ge=0, description="Region code")
    previously_insured: int = Field(..., ge=0, le=1, description="Previously insured")
    vehicle_age: str = Field(..., description="Vehicle age category")
    vehicle_damage: str = Field(..., description="Vehicle damage history")
    annual_premium: float = Field(..., gt=0, description="Annual premium amount")
    policy_sales_channel: int = Field(..., ge=0, description="Sales channel")
    vintage: int = Field(..., ge=0, description="Days as customer")

class PredictionResponse(BaseModel):
    """Single prediction response schema."""
    id: str
    probability: float = Field(..., ge=0, le=1, description="Prediction probability")
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction (0/1)")
    threshold: float = Field(..., description="Decision threshold used")
    model_version: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float

class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_name: str
    model_version: str
    created_date: str
    trained_on: str
    decision_threshold: float
    api_version: str
    
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def prepare_features(data: Dict[str, Any]) -> np.ndarray:
    """
    Convert input data to feature array using preprocessor.
    
    Args:
        data: Dictionary with raw input features
        
    Returns:
        Preprocessed feature array (1D numpy array)
    """
    try:
        # Create DataFrame with single row
        df = pd.DataFrame([data])
        
        # Apply feature engineering (same as training pipeline)
        # 1. Vehicle Age as numeric
        vehicle_age_mapping = {
            "< 1 Year": 0.5,
            "1-2 Year": 1.5,
            "> 2 Years": 3
        }
        df['Vehicle_Age_Numeric'] = df['Vehicle_Age'].map(vehicle_age_mapping)
        
        # 2. Premium per vehicle year
        df['Premium_per_Vehicle_Year'] = (
            df['Annual_Premium'] / (df['Vehicle_Age_Numeric'] + 1)
        )
        
        # 3. High-value vehicle flag (using 75th percentile - from training)
        # Approximate 75th percentile from training data
        premium_75th = 40000  # Estimated from training data
        df['High_Value_Vehicle'] = (
            df['Annual_Premium'] > premium_75th
        ).astype(int)
        
        # 4. Age risk bucket
        def age_risk_bucket(age):
            if age < 25:
                return 'very_high_risk'
            elif age < 35:
                return 'high_risk'
            elif age < 50:
                return 'medium_risk'
            elif age < 65:
                return 'low_risk'
            else:
                return 'very_low_risk'
        
        df['Age_Risk_Bucket'] = df['Age'].apply(age_risk_bucket)
        
        # 5. Customer tenure segments
        def tenure_segment(vintage):
            if vintage < 30:
                return 'new_customer'
            elif vintage < 90:
                return 'growing_customer'
            elif vintage < 365:
                return 'established_customer'
            else:
                return 'loyal_customer'
        
        df['Customer_Tenure_Segment'] = df['Vintage'].apply(tenure_segment)
        
        # 6. Premium bucket (using training data quantiles)
        def premium_bucket(premium):
            # Approximate quantiles from training: 25%=18000, 50%=31000, 75%=40000
            if premium < 18000:
                return 'low_premium'
            elif premium < 31000:
                return 'medium_premium'
            elif premium < 40000:
                return 'high_premium'
            else:
                return 'very_high_premium'
        
        df['Premium_Bucket'] = df['Annual_Premium'].apply(premium_bucket)
        
        # 7. Damage history risk
        df['Damage_History_Risk'] = (
            df['Vehicle_Damage'].map({'Yes': 1, 'No': 0}) * 
            (1 - df['Previously_Insured'])
        )
        
        # Now prepare the final feature set for the preprocessor
        # Order matters! Must match training order
        all_features = [
            'Gender', 'Age', 'Driving_License', 'Region_Code',
            'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
            'Annual_Premium', 'Policy_Sales_Channel', 'Vintage',
            'Vehicle_Age_Numeric', 'Premium_per_Vehicle_Year',
            'High_Value_Vehicle', 'Age_Risk_Bucket', 'Customer_Tenure_Segment',
            'Premium_Bucket', 'Damage_History_Risk'
        ]
        
        df_features = df[all_features].copy()
        
        # Apply preprocessor
        X = preprocessor.transform(df_features)
        
        return X
    except Exception as e:
        logger.error(f"Feature preprocessing error: {e}")
        raise ValueError(f"Failed to preprocess features: {e}")

def make_prediction(X: np.ndarray) -> tuple:
    """
    Generate prediction using calibrated model.
    
    Args:
        X: Preprocessed feature array
        
    Returns:
        Tuple of (probability, binary_prediction)
    """
    try:
        prob = model.predict_proba(X)[0, 1]
        pred = int(prob >= DECISION_THRESHOLD)
        return prob, pred
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise RuntimeError(f"Failed to generate prediction: {e}")

# ============================================================================
# API ENDPOINTS
# ============================================================================



@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model is not None,
        model_version=MODEL_VERSION,
        uptime_seconds=0  # Could track actual uptime
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model metadata and performance info."""
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        created_date=CREATED_DATE,
        trained_on=TRAINED_ON,
        decision_threshold=DECISION_THRESHOLD,
        api_version=API_VERSION
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate single prediction.
    
    Takes customer features and returns claim prediction probability
    and binary prediction using optimized threshold.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Prepare input
        input_data = {
            'Gender': request.gender,
            'Age': request.age,
            'Driving_License': request.driving_license,
            'Region_Code': request.region_code,
            'Previously_Insured': request.previously_insured,
            'Vehicle_Age': request.vehicle_age,
            'Vehicle_Damage': request.vehicle_damage,
            'Annual_Premium': request.annual_premium,
            'Policy_Sales_Channel': request.policy_sales_channel,
            'Vintage': request.vintage
        }
        
        # Preprocess and predict
        X = prepare_features(input_data)
        prob, pred = make_prediction(X)
        
        logger.info(f"Prediction for {request.id}: prob={prob:.4f}, pred={pred}")
        
        return PredictionResponse(
            id=request.id,
            probability=float(prob),
            prediction=int(pred),
            threshold=DECISION_THRESHOLD,
            model_version=MODEL_VERSION,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch prediction from CSV file.
    
    Accepts CSV with columns matching PredictionRequest schema.
    Returns JSON with predictions for each row.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Read uploaded CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Normalize column names to title case for consistency
        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
        
        # Map common variations to standard names
        column_mapping = {
            'id': 'id', 'ID': 'id',
            'gender': 'Gender', 'Gender': 'Gender', 'GENDER': 'Gender',
            'age': 'Age', 'Age': 'Age', 'AGE': 'Age',
            'driving_license': 'Driving_License', 'Driving_License': 'Driving_License',
            'region_code': 'Region_Code', 'Region_Code': 'Region_Code',
            'previously_insured': 'Previously_Insured', 'Previously_Insured': 'Previously_Insured',
            'vehicle_age': 'Vehicle_Age', 'Vehicle_Age': 'Vehicle_Age',
            'vehicle_damage': 'Vehicle_Damage', 'Vehicle_Damage': 'Vehicle_Damage',
            'annual_premium': 'Annual_Premium', 'Annual_Premium': 'Annual_Premium',
            'policy_sales_channel': 'Policy_Sales_Channel', 'Policy_Sales_Channel': 'Policy_Sales_Channel',
            'vintage': 'Vintage', 'Vintage': 'Vintage'
        }
        
        # Rename columns
        df.rename(columns=column_mapping, inplace=True)
        
        # Validate size
        if len(df) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Batch size {len(df)} exceeds maximum {MAX_BATCH_SIZE}"
            )
        
        logger.info(f"Processing batch with {len(df)} records")
        
        # Apply feature engineering to batch
        df_engineered = df.copy()
        
        # 1. Vehicle Age as numeric
        vehicle_age_mapping = {
            "< 1 Year": 0.5,
            "1-2 Year": 1.5,
            "> 2 Years": 3
        }
        df_engineered['Vehicle_Age_Numeric'] = df_engineered['Vehicle_Age'].map(vehicle_age_mapping)
        
        # 2. Premium per vehicle year
        df_engineered['Premium_per_Vehicle_Year'] = (
            df_engineered['Annual_Premium'] / (df_engineered['Vehicle_Age_Numeric'] + 1)
        )
        
        # 3. High-value vehicle flag
        premium_75th = 40000  # From training data
        df_engineered['High_Value_Vehicle'] = (
            df_engineered['Annual_Premium'] > premium_75th
        ).astype(int)
        
        # 4. Age risk bucket
        def age_risk_bucket(age):
            if age < 25:
                return 'very_high_risk'
            elif age < 35:
                return 'high_risk'
            elif age < 50:
                return 'medium_risk'
            elif age < 65:
                return 'low_risk'
            else:
                return 'very_low_risk'
        
        df_engineered['Age_Risk_Bucket'] = df_engineered['Age'].apply(age_risk_bucket)
        
        # 5. Customer tenure segments
        def tenure_segment(vintage):
            if vintage < 30:
                return 'new_customer'
            elif vintage < 90:
                return 'growing_customer'
            elif vintage < 365:
                return 'established_customer'
            else:
                return 'loyal_customer'
        
        df_engineered['Customer_Tenure_Segment'] = df_engineered['Vintage'].apply(tenure_segment)
        
        # 6. Premium bucket
        def premium_bucket(premium):
            if premium < 18000:
                return 'low_premium'
            elif premium < 31000:
                return 'medium_premium'
            elif premium < 40000:
                return 'high_premium'
            else:
                return 'very_high_premium'
        
        df_engineered['Premium_Bucket'] = df_engineered['Annual_Premium'].apply(premium_bucket)
        
        # 7. Damage history risk
        df_engineered['Damage_History_Risk'] = (
            df_engineered['Vehicle_Damage'].map({'Yes': 1, 'No': 0}) * 
            (1 - df_engineered['Previously_Insured'])
        )
        
        # Prepare all features in correct order
        all_features = [
            'Gender', 'Age', 'Driving_License', 'Region_Code',
            'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
            'Annual_Premium', 'Policy_Sales_Channel', 'Vintage',
            'Vehicle_Age_Numeric', 'Premium_per_Vehicle_Year',
            'High_Value_Vehicle', 'Age_Risk_Bucket', 'Customer_Tenure_Segment',
            'Premium_Bucket', 'Damage_History_Risk'
        ]
        
        df_features = df_engineered[all_features].copy()
        
        X = preprocessor.transform(df_features)
        
        # Generate predictions
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= DECISION_THRESHOLD).astype(int)
        
        # Prepare response
        results = []
        for i, row_id in enumerate(df.get('id', range(len(df)))):
            results.append({
                "id": str(row_id),
                "probability": float(probs[i]),
                "prediction": int(preds[i]),
                "threshold": DECISION_THRESHOLD
            })
        
        logger.info(f"✓ Batch processing complete: {len(results)} predictions")
        
        return {
            "batch_size": len(results),
            "predictions": results,
            "model_version": MODEL_VERSION,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        raise HTTPException(status_code=400, detail=f"Missing required column: {str(e)}")
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "message": "RiskLens Model Serving API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "predict": "/predict",
        "predict_batch": "/predict-batch"
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    logger.error(f"HTTP Error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.utcnow().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.utcnow().isoformat()}
    )

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        workers=WORKERS if not DEBUG else 1,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )
