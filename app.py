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

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from config import (
    HOST, PORT, WORKERS, MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH,
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
    """Load calibrated model, preprocessor, and feature names."""
    global model, preprocessor, feature_names
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"✓ Model loaded: {MODEL_PATH}")
        
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"✓ Preprocessor loaded: {PREPROCESSOR_PATH}")
        
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        logger.info(f"✓ Feature names loaded: {FEATURE_NAMES_PATH}")
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Unexpected error loading model: {e}")

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================
app = FastAPI(
    title=MODEL_NAME,
    description="REST API for vehicle insurance claim prediction",
    version=API_VERSION
)

# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================
class PredictionRequest(BaseModel):
    """Single prediction request schema."""
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
    
    class Config:
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
        
        # Reorder columns to match training order
        df = df[['Gender', 'Age', 'Driving_License', 'Region_Code', 
                 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 
                 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']]
        
        # Apply preprocessor
        X = preprocessor.transform(df)
        
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

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting RiskLens API...")
    load_model()
    logger.info("✓ API startup complete")

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
        
        # Validate size
        if len(df) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Batch size {len(df)} exceeds maximum {MAX_BATCH_SIZE}"
            )
        
        logger.info(f"Processing batch with {len(df)} records")
        
        # Prepare all features
        df_features = df[['Gender', 'Age', 'Driving_License', 'Region_Code',
                          'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage',
                          'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']].copy()
        
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
    
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")

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
